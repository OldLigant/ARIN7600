"""Durable, one-check-per-tick orchestration for Vertex Batch stages.

Only the owner of a successful generation-precondition write may submit a job.
An interrupted submission is reconciled by its deterministic display name and
is never automatically retried. Request preparation and result collection must
be idempotent: workers can repeat those operations after a crash or CAS race.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import time
from typing import Any

from .batch_cloud import StateConflict


STAGES = ("audio", "annotation", "review")
TERMINAL_JOBS = {"SUCCEEDED", "PARTIALLY_SUCCEEDED", "FAILED", "EXPIRED", "CANCELLED"}
ACTIVE_JOBS = {"PENDING", "QUEUED", "RUNNING", "CANCELLING", "UPDATING"}
STOPPED = {"complete", "complete_with_errors", "paused", "needs_attention"}
SUBMISSION_GRACE_SECONDS = 3600


class BatchEngine:
    """Advance the persisted authorized scope without a waiting daemon."""

    def __init__(self, cloud: Any, state_uri: str, planner: Any):
        self.cloud = cloud
        self.state_uri = state_uri
        self.planner = planner

    def initialize(self, config: dict) -> dict:
        """Create state once. Existing state and its scope are never replaced."""
        existing, _ = self.cloud.read_state(self.state_uri)
        if existing is not None:
            if existing.get("config") != config:
                raise ValueError("Existing batch configuration differs from requested config")
            return existing
        state = {
            "schema_version": 1,
            "config": deepcopy(config),
            "status": "ready",
            "current_stage": STAGES[0],
            "eligible_rows": deepcopy(config.get("rows", [])),
            "history": [],
            "batches": {},
            "failures": [],
            "completed": [],
        }
        saved, _, _ = self._save(state, 0)
        if saved.get("config") != config:
            raise ValueError("Concurrent batch configuration differs from requested config")
        return saved

    def start(self) -> dict:
        """Submit an initialized stage, or safely reconcile an interrupted start."""
        state, generation = self._read()
        if state["status"] == "ready":
            return self._submit_ready(state, generation)
        if state["status"] == "submitting":
            return self._reconcile(state, generation)
        return state

    def reconcile(self) -> dict:
        """Explicitly look up an unresolved submission; never submit a job.

        This also permits recovery after an earlier lookup found no job (for
        example, before the provider's list operation reflected acceptance).
        Only a recorded submission intent for the current stage is eligible.
        """
        state, generation = self._read()
        if state["status"] not in {"submitting", "needs_attention"}:
            return state
        stage = state["current_stage"]
        batch = state["batches"].get(stage)
        if not batch or batch.get("job") or not batch.get("display_name"):
            return state
        has_intent = any(
            item.get("event") == "submission_intent"
            and item.get("stage") == stage
            and item.get("display_name") == batch["display_name"]
            for item in state["history"]
        )
        if not has_intent:
            return state
        return self._reconcile(state, generation)

    def tick(self) -> dict:
        """Check one existing job, collect terminal results, and advance scope."""
        state, generation = self._read()
        if state["status"] in STOPPED:
            return state
        if state["status"] == "ready":
            return self._submit_ready(state, generation)
        if state["status"] == "submitting":
            return self._reconcile(state, generation)
        if state["status"] != "running":
            return self._attention(state, generation, "Unrecognized engine status")

        stage = state["current_stage"]
        batch = state["batches"][stage]
        job = self.cloud.get_batch(batch["job"]["name"])
        batch["job"] = job
        job_state = self._job_state(job)
        if job_state in ACTIVE_JOBS:
            return self._save(state, generation)[0]
        if job_state == "PAUSED":
            state["status"] = "paused"
            return self._save(state, generation)[0]
        if job_state not in TERMINAL_JOBS:
            return self._attention(state, generation, f"Unrecognized batch status: {job_state}")

        # Collection is keyed by stable request identifiers by the planner.
        # Persist collection and the next-stage transition together, so a crash
        # can repeat collection without duplicating durable failures/results.
        result = self.planner.collect(stage, job, state)
        state["eligible_rows"] = deepcopy(result["eligible_rows"])
        state["failures"].extend(deepcopy(result["failures"]))
        state["completed"].extend(deepcopy(result["completed"]))
        state["history"].append({"stage": stage, "event": "collected", "job_state": job_state})
        self._advance(state)
        state, generation, won = self._save(state, generation)
        if won and state["status"] == "ready":
            return self._submit_ready(state, generation)
        return state

    def _read(self) -> tuple[dict, Any]:
        state, generation = self.cloud.read_state(self.state_uri)
        if state is None:
            raise ValueError("Batch state is missing; initialize before starting")
        return state, generation

    def _save(self, state: dict, generation: Any) -> tuple[dict, Any, bool]:
        try:
            next_generation = self.cloud.write_state(self.state_uri, state, generation)
        except StateConflict:
            current, current_generation = self._read()
            return current, current_generation, False
        return state, next_generation, True

    def _submit_ready(self, state: dict, generation: Any) -> dict:
        while state["status"] == "ready":
            if not state["eligible_rows"]:
                self._finish(state)
                return self._save(state, generation)[0]
            stage = state["current_stage"]
            prepared = deepcopy(self.planner.prepare(stage, state["eligible_rows"], state))
            if not prepared["request_ids"]:
                state["history"].append({"stage": stage, "event": "skipped"})
                self._advance(state)
                state, generation, won = self._save(state, generation)
                if not won:
                    return state
                continue

            identity = json.dumps(
                [self.state_uri, stage, sorted(prepared["request_ids"])],
                separators=(",", ":"), ensure_ascii=True,
            )
            prepared["display_name"] = f"castle-{stage}-{sha256(identity.encode()).hexdigest()[:32]}"
            prepared["submission_started_at"] = time.time()
            state["batches"][stage] = prepared
            state["status"] = "submitting"
            state["history"].append({"stage": stage, "event": "submission_intent",
                                      "display_name": prepared["display_name"]})
            state, generation, won = self._save(state, generation)
            if not won:
                return state
            try:
                job = self.cloud.create_batch(
                    state["config"]["model"], prepared["input_uri"],
                    prepared["output_uri"], prepared["display_name"],
                )
            except Exception as exc:
                # Even a timeout can mean the provider accepted the paid job.
                # Leave the intent durable; only a later lookup can resolve it.
                state["last_error"] = type(exc).__name__
                code = getattr(exc, "code", None)
                state["last_error_code"] = code if isinstance(code, int) else None
                return self._save(state, generation)[0]
            state["batches"][stage]["job"] = job
            state["status"] = "running"
            return self._save(state, generation)[0]
        return state

    def _reconcile(self, state: dict, generation: Any) -> dict:
        batch = state["batches"][state["current_stage"]]
        matches = self.cloud.find_batches(batch["display_name"])
        started_at = batch.get("submission_started_at")
        if (not matches and state["status"] == "submitting"
                and isinstance(started_at, (int, float)) and not isinstance(started_at, bool)
                and 0 <= time.time() - started_at < SUBMISSION_GRACE_SECONDS):
            # Another worker may still be inside create_batch, or the provider
            # may not list the accepted job yet. Do not write state here: that
            # would invalidate the submitter's CAS when it records acceptance.
            return state
        if len(matches) != 1:
            return self._attention(
                state, generation,
                f"Uncertain submission: found {len(matches)} matching jobs; manual reconciliation required",
            )
        batch["job"] = matches[0]
        state["status"] = "running"
        state.pop("last_error", None)
        state.pop("last_error_code", None)
        state.pop("attention_reason", None)
        state["history"].append({"stage": state["current_stage"], "event": "submission_reconciled"})
        return self._save(state, generation)[0]

    def _attention(self, state: dict, generation: Any, reason: str) -> dict:
        state["status"] = "needs_attention"
        state["attention_reason"] = reason
        return self._save(state, generation)[0]

    @staticmethod
    def _job_state(job: dict) -> str:
        return str(job["state"]).rsplit(".", 1)[-1].removeprefix("JOB_STATE_")

    def _advance(self, state: dict) -> None:
        index = STAGES.index(state["current_stage"]) + 1
        if index == len(STAGES) or not state["eligible_rows"]:
            self._finish(state)
        else:
            state["current_stage"] = STAGES[index]
            state["status"] = "ready"

    @staticmethod
    def _finish(state: dict) -> None:
        job_errors = any(item.get("job_state") in TERMINAL_JOBS - {"SUCCEEDED"}
                         for item in state["history"])
        state["current_stage"] = None
        state["status"] = "complete_with_errors" if state["failures"] or job_errors else "complete"


# Historical design-plan spelling; the public CLI uses BatchEngine.
Engine = BatchEngine
