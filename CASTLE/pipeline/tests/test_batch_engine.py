"""Offline durability tests; cloud operations are the only external boundary."""

from copy import deepcopy

import pytest

from castle_pipeline.batch_engine import BatchEngine
from castle_pipeline.batch_cloud import StateConflict


class MemoryCloud:
    def __init__(self):
        self.state = None
        self.generation = 0
        self.jobs = {}
        self.creates = []
        self.gets = []
        self.create_error = False
        self.fail_job_save = False
        self.before_write = None
        self.during_create = None

    def read_state(self, uri):
        return deepcopy(self.state), self.generation

    def write_state(self, uri, state, expected_generation):
        if self.before_write:
            hook, self.before_write = self.before_write, None
            hook()
        if expected_generation != self.generation:
            raise StateConflict("state changed")
        if self.fail_job_save and state["status"] == "running":
            self.fail_job_save = False
            raise OSError("storage unavailable after job create")
        self.state = deepcopy(state)
        self.generation += 1
        return self.generation

    def create_batch(self, model, input_uri, output_uri, display_name):
        assert self.state["status"] == "submitting"
        assert self.state["batches"][self.state["current_stage"]]["display_name"] == display_name
        self.creates.append((model, input_uri, output_uri, display_name))
        if self.during_create:
            hook, self.during_create = self.during_create, None
            hook()
        job = {"name": f"jobs/{len(self.creates)}", "state": "JOB_STATE_PENDING",
               "output_uri": output_uri, "display_name": display_name}
        self.jobs[job["name"]] = job
        if self.create_error:
            raise TimeoutError("response lost after acceptance")
        return deepcopy(job)

    def get_batch(self, name):
        self.gets.append(name)
        return deepcopy(self.jobs[name])

    def find_batches(self, display_name):
        return [deepcopy(job) for job in self.jobs.values()
                if job["display_name"] == display_name]


class Planner:
    def __init__(self):
        self.prepared = []
        self.results = {}
        self.empty = set()
        self.collected = []

    def prepare(self, stage, eligible_rows, state):
        self.prepared.append((stage, deepcopy(eligible_rows)))
        return {"input_uri": f"gs://test/{stage}.jsonl", "output_uri": f"gs://test/{stage}/",
                "request_ids": [] if stage in self.empty else [r["clip_id"] for r in eligible_rows]}

    def collect(self, stage, job, state):
        self.collected.append((stage, job["state"]))
        return deepcopy(self.results[stage])


@pytest.fixture
def run():
    cloud, planner = MemoryCloud(), Planner()
    config = {"model": "gemini-test", "project": "test-project", "location": "global",
              "gcs_prefix": "gs://test/run", "rows": [{"clip_id": "a"}, {"clip_id": "b"}]}
    engine = BatchEngine(cloud, "gs://test/run/state.json", planner)
    engine.initialize(config)
    return cloud, planner, engine, config


def test_initialization_preserves_config_and_start_is_idempotent(run):
    cloud, planner, engine, config = run
    config["rows"].append({"clip_id": "not-authorized"})
    state = engine.start()
    assert state["config"]["rows"] == [{"clip_id": "a"}, {"clip_id": "b"}]
    assert state["current_stage"] == "audio"
    assert state["status"] == "running"
    engine.start()
    assert engine.initialize(deepcopy(state["config"])) == state
    with pytest.raises(ValueError, match="config"):
        engine.initialize({"model": "other"})
    assert len(cloud.creates) == 1
    assert cloud.creates[0][:3] == ("gemini-test", "gs://test/audio.jsonl", "gs://test/audio/")


@pytest.mark.parametrize("job_state", ["JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_QUEUED"])
def test_pending_tick_checks_once_and_does_not_collect_or_submit(run, job_state):
    cloud, planner, engine, _ = run
    engine.start()
    cloud.jobs["jobs/1"]["state"] = job_state
    state = engine.tick()
    assert state["current_stage"] == "audio"
    assert cloud.gets == ["jobs/1"]
    assert planner.collected == []
    assert len(cloud.creates) == 1


@pytest.mark.parametrize("job_state", ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_EXPIRED", "JOB_STATE_CANCELLED"])
def test_terminal_job_advances_only_valid_partial_rows(run, job_state):
    cloud, planner, engine, _ = run
    engine.start()
    cloud.jobs["jobs/1"]["state"] = job_state
    planner.results["audio"] = {"eligible_rows": [{"clip_id": "b", "audio_uri": "gs://test/b.json"}],
                                "failures": [{"clip_id": "a", "reason": "missing"}], "completed": []}
    state = engine.tick()
    assert state["current_stage"] == "annotation"
    assert planner.prepared[-1] == ("annotation", [{"clip_id": "b", "audio_uri": "gs://test/b.json"}])
    assert state["failures"][0]["clip_id"] == "a"
    assert len(cloud.creates) == 2
    assert state["completed"] == []


def test_results_complete_once_and_failed_rows_are_never_retried(run):
    cloud, planner, engine, _ = run
    engine.start()
    planner.results["audio"] = {"eligible_rows": [{"clip_id": "b"}], "failures": [{"clip_id": "a"}], "completed": []}
    cloud.jobs["jobs/1"]["state"] = "JOB_STATE_SUCCEEDED"
    engine.tick()
    planner.results["annotation"] = {"eligible_rows": [], "failures": [], "completed": [{"clip_id": "b", "result_uri": "gs://test/result"}]}
    cloud.jobs["jobs/2"]["state"] = "JOB_STATE_SUCCEEDED"
    state = engine.tick()
    assert state["status"] == "complete_with_errors"
    assert state["completed"] == [{"clip_id": "b", "result_uri": "gs://test/result"}]
    assert engine.tick() == state
    assert engine.start() == state
    assert len(cloud.creates) == 2


def test_empty_stage_skips_to_next_stage_without_paid_job(run):
    cloud, planner, engine, _ = run
    planner.empty.add("audio")
    state = engine.start()
    assert state["current_stage"] == "annotation"
    assert len(cloud.creates) == 1
    assert cloud.creates[0][1] == "gs://test/annotation.jsonl"


def test_restart_recovers_job_accepted_before_create_timeout(run):
    cloud, planner, engine, _ = run
    cloud.create_error = True
    assert engine.start()["status"] == "submitting"
    restarted = BatchEngine(cloud, "gs://test/run/state.json", planner)
    state = restarted.tick()
    assert state["status"] == "running"
    assert state["batches"]["audio"]["job"]["name"] == "jobs/1"
    assert len(cloud.creates) == 1


@pytest.mark.parametrize("matches", [0, 2])
def test_uncertain_submission_requires_attention_without_retry(run, matches):
    cloud, planner, engine, _ = run
    cloud.create_error = True
    engine.start()
    first = cloud.jobs["jobs/1"]
    cloud.jobs = {} if not matches else {"jobs/1": first, "jobs/2": dict(first, name="jobs/2")}
    cloud.state["batches"]["audio"]["submission_started_at"] = 0
    state = engine.tick()
    assert state["status"] == "needs_attention"
    assert engine.tick() == state
    assert engine.start() == state
    assert len(cloud.creates) == 1


def test_restart_recovers_after_post_create_state_write_fails(run):
    cloud, planner, engine, _ = run
    cloud.fail_job_save = True
    with pytest.raises(OSError):
        engine.start()
    assert cloud.state["status"] == "submitting"
    state = engine.tick()
    assert state["status"] == "running"
    assert len(cloud.creates) == 1


def test_competing_start_loses_cas_and_never_creates_another_job(run):
    cloud, planner, engine, _ = run
    competing = BatchEngine(cloud, "gs://test/run/state.json", planner)
    cloud.before_write = competing.start
    state = engine.start()
    assert state["status"] == "running"
    assert len(cloud.creates) == 1


@pytest.mark.parametrize("status", ["paused", "needs_attention"])
def test_paused_and_attention_states_have_no_cloud_job_side_effects(run, status):
    cloud, planner, engine, _ = run
    cloud.state["status"] = status
    assert engine.tick()["status"] == status
    assert engine.start()["status"] == status
    assert not cloud.creates
    assert not cloud.gets


def test_review_completion_is_terminal(run):
    cloud, planner, engine, _ = run
    engine.start()
    for index, stage in enumerate(("audio", "annotation", "review"), 1):
        planner.results[stage] = {"eligible_rows": [{"clip_id": "a"}] if stage != "review" else [],
                                  "failures": [], "completed": [{"clip_id": "a"}] if stage == "review" else []}
        cloud.jobs[f"jobs/{index}"]["state"] = "JOB_STATE_SUCCEEDED"
        state = engine.tick()
    assert state["status"] == "complete"
    assert state["current_stage"] is None
    assert len(cloud.creates) == 3


def test_failed_job_without_usable_rows_is_not_reported_complete(run):
    cloud, planner, engine, _ = run
    engine.start()
    cloud.jobs["jobs/1"]["state"] = "JOB_STATE_FAILED"
    planner.results["audio"] = {"eligible_rows": [], "failures": [], "completed": []}
    state = engine.tick()
    assert state["status"] == "complete_with_errors"
    assert state["completed"] == []
    assert len(cloud.creates) == 1


def test_failed_collection_write_can_restart_without_duplicate_results(run):
    cloud, planner, engine, _ = run
    engine.start()
    cloud.jobs["jobs/1"]["state"] = "JOB_STATE_SUCCEEDED"
    planner.results["audio"] = {"eligible_rows": [], "failures": [{"clip_id": "a"}],
                                "completed": [{"clip_id": "b"}]}

    def storage_outage():
        raise OSError("collection state save interrupted")

    cloud.before_write = storage_outage
    with pytest.raises(OSError):
        engine.tick()
    assert cloud.state["status"] == "running"
    state = engine.tick()
    assert state["completed"] == [{"clip_id": "b"}]
    assert state["failures"] == [{"clip_id": "a"}]
    assert len(cloud.creates) == 1


def test_competing_terminal_ticks_create_next_stage_once(run):
    cloud, planner, engine, _ = run
    engine.start()
    cloud.jobs["jobs/1"]["state"] = "JOB_STATE_SUCCEEDED"
    planner.results["audio"] = {"eligible_rows": [{"clip_id": "b"}],
                                "failures": [{"clip_id": "a"}], "completed": []}
    competing = BatchEngine(cloud, "gs://test/run/state.json", planner)
    cloud.before_write = competing.tick
    state = engine.tick()
    assert state["current_stage"] == "annotation"
    assert state["failures"] == [{"clip_id": "a"}]
    assert len(cloud.creates) == 2


def test_failed_submission_claim_has_no_paid_side_effect(run):
    cloud, planner, engine, _ = run

    def storage_outage():
        raise OSError("submission intent save interrupted")

    cloud.before_write = storage_outage
    with pytest.raises(OSError):
        engine.start()
    assert not cloud.creates
    assert cloud.state["status"] == "ready"


def test_job_pause_blocks_subsequent_ticks(run):
    cloud, planner, engine, _ = run
    engine.start()
    cloud.jobs["jobs/1"]["state"] = "JOB_STATE_PAUSED"
    assert engine.tick()["status"] == "paused"
    engine.tick()
    assert len(cloud.gets) == 1
    assert len(cloud.creates) == 1


def test_unrecognized_job_state_requires_attention(run):
    cloud, planner, engine, _ = run
    engine.start()
    cloud.jobs["jobs/1"]["state"] = "JOB_STATE_UNKNOWN_NEW_VALUE"
    assert engine.tick()["status"] == "needs_attention"
    assert not planner.collected
    assert len(cloud.creates) == 1


def test_empty_scope_completes_without_submission():
    cloud, planner = MemoryCloud(), Planner()
    engine = BatchEngine(cloud, "gs://test/empty/state.json", planner)
    engine.initialize({"model": "test", "rows": []})
    assert engine.start()["status"] == "complete"
    assert not cloud.creates


def test_missing_state_does_not_submit():
    cloud = MemoryCloud()
    engine = BatchEngine(cloud, "gs://test/missing/state.json", Planner())
    with pytest.raises(ValueError, match="initialize"):
        engine.tick()
    assert not cloud.creates


def test_explicit_reconcile_recovers_attention_without_changing_scope(run):
    cloud, planner, engine, _ = run
    cloud.create_error = True
    engine.start()
    accepted_job = cloud.jobs.pop("jobs/1")
    cloud.state["batches"]["audio"]["submission_started_at"] = 0
    assert engine.tick()["status"] == "needs_attention"
    original_config = deepcopy(cloud.state["config"])
    original_rows = deepcopy(cloud.state["eligible_rows"])
    cloud.jobs["jobs/1"] = accepted_job
    state = engine.reconcile()
    assert state["status"] == "running"
    assert state["batches"]["audio"]["job"]["name"] == "jobs/1"
    assert state["config"] == original_config
    assert state["eligible_rows"] == original_rows
    assert "attention_reason" not in state
    assert "last_error" not in state
    assert len(cloud.creates) == 1


@pytest.mark.parametrize("matches", [0, 2])
def test_explicit_reconcile_leaves_ambiguous_jobs_for_attention(run, matches):
    cloud, planner, engine, _ = run
    cloud.create_error = True
    engine.start()
    accepted_job = cloud.jobs["jobs/1"]
    cloud.jobs = {} if matches == 0 else {
        "jobs/1": accepted_job, "jobs/2": dict(accepted_job, name="jobs/2")}
    cloud.state["batches"]["audio"]["submission_started_at"] = 0
    engine.tick()
    state = engine.reconcile()
    assert state["status"] == "needs_attention"
    assert "job" not in state["batches"]["audio"]
    assert len(cloud.creates) == 1


@pytest.mark.parametrize("case", ["ready", "paused", "running", "no_intent", "saved_job"])
def test_explicit_reconcile_is_noop_without_unresolved_submission(run, case):
    cloud, planner, engine, _ = run
    if case != "ready":
        engine.start()
        if case == "no_intent":
            cloud.state["history"] = []
            cloud.state["status"] = "needs_attention"
            cloud.state["batches"]["audio"].pop("job")
        elif case == "saved_job":
            cloud.state["status"] = "needs_attention"
        elif case == "paused":
            cloud.state["status"] = "paused"
    original = deepcopy(cloud.state)
    create_count = len(cloud.creates)

    def unexpected_lookup(display_name):
        pytest.fail("reconcile looked up a resolved or unauthorized submission")

    cloud.find_batches = unexpected_lookup
    assert engine.reconcile() == original
    assert cloud.state == original
    assert len(cloud.creates) == create_count


def test_submission_error_stores_class_without_raw_exception_text(run):
    cloud, planner, engine, _ = run
    cloud.create_error = True
    state = engine.start()
    assert state["last_error"] == "TimeoutError"


@pytest.mark.parametrize("lose_create_response", [False, True])
def test_tick_overlapping_create_keeps_intent_recoverable(run, lose_create_response):
    cloud, planner, engine, _ = run
    other = BatchEngine(cloud, "gs://test/run/state.json", planner)
    observed = []
    cloud.during_create = lambda: observed.append(other.tick()["status"])
    cloud.create_error = lose_create_response
    engine.start()
    assert observed == ["submitting"]
    state = other.tick()
    assert state["status"] == "running"
    assert state["batches"]["audio"]["job"]["name"] == "jobs/1"
    assert len(cloud.creates) == 1


def test_zero_matches_have_bounded_grace_without_submission_retry(run, monkeypatch):
    cloud, planner, engine, _ = run
    monkeypatch.setattr("castle_pipeline.batch_engine.time.time", lambda: 10000)
    cloud.create_error = True
    engine.start()
    cloud.jobs = {}
    assert cloud.state["batches"]["audio"]["submission_started_at"] == 10000
    monkeypatch.setattr("castle_pipeline.batch_engine.time.time", lambda: 13599)
    assert engine.tick()["status"] == "submitting"
    monkeypatch.setattr("castle_pipeline.batch_engine.time.time", lambda: 13600)
    assert engine.tick()["status"] == "needs_attention"
    assert len(cloud.creates) == 1


def test_old_submission_without_timestamp_requires_attention(run):
    cloud, planner, engine, _ = run
    cloud.create_error = True
    engine.start()
    cloud.jobs = {}
    cloud.state["batches"]["audio"].pop("submission_started_at", None)
    assert engine.tick()["status"] == "needs_attention"
    assert len(cloud.creates) == 1


@pytest.mark.parametrize("same_config", [False, True])
def test_initialization_race_checks_winner_configuration(same_config):
    cloud, planner = MemoryCloud(), Planner()
    engine = BatchEngine(cloud, "gs://test/run/state.json", planner)
    other = BatchEngine(cloud, "gs://test/run/state.json", planner)
    requested = {"model": "requested", "rows": [{"clip_id": "a"}]}
    winner = deepcopy(requested) if same_config else {"model": "other", "rows": [{"clip_id": "b"}]}
    cloud.before_write = lambda: other.initialize(winner)
    if same_config:
        assert engine.initialize(requested)["config"] == requested
    else:
        with pytest.raises(ValueError, match="config"):
            engine.initialize(requested)
    assert cloud.state["config"] == winner
    assert not cloud.creates
