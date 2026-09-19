"""Vertex-only multimodal requests with shared request-level admission control.

One provider is shared by all clip workers and all stages. Its clients are local
to each worker thread; its AIMD limit, pacing and cooldown are shared. No client
or credentials are loaded until the first request on that thread.
"""
from __future__ import annotations

import json
import math
import mimetypes
import os
from pathlib import Path
import threading
import time
import uuid
from typing import Any, Callable


MAX_INLINE_BYTES = 18 * 1024 * 1024


class ProviderError(RuntimeError):
    """Safe-to-persist error; never embeds raw SDK responses or credentials."""

    def __init__(self, message: str, category: str = "output", fatal: bool = False,
                 *, attempts: int = 0, usage: dict | None = None, code: int | str | None = None):
        super().__init__(message)
        self.category = category
        self.fatal = fatal
        self.attempts = attempts
        self.usage = usage or {}
        self.code = code if code is not None else category.upper()


class RequestLimiter:
    """Paced RPM admission with additive increase / multiplicative decrease.

    Pacing avoids bursts: request starts are at least 60/rpm seconds apart.
    Four successful requests (or a full larger window) increase the cap by one.
    Congestion halves it immediately. Waiting requests hold no in-flight slot.
    """

    def __init__(self, initial_concurrency: int, max_concurrency: int, rpm: float,
                 *, clock: Callable = time.monotonic, sleep: Callable = time.sleep):
        self.limit = initial_concurrency
        self.maximum = max_concurrency
        self.interval = 60.0 / rpm
        self.clock, self.sleep = clock, sleep
        self.condition = threading.Condition()
        self.in_flight = 0
        self.next_start = 0.0
        self.cooldown_until = 0.0
        self.successes = 0

    def acquire(self) -> None:
        while True:
            with self.condition:
                if self.in_flight >= self.limit:
                    self.condition.wait(timeout=0.25)
                    continue
                now = self.clock()
                delay = max(self.next_start, self.cooldown_until) - now
                if delay <= 0:
                    self.in_flight += 1
                    self.next_start = now + self.interval
                    return
            # Sleep outside the lock and before acquiring an in-flight slot.
            self.sleep(min(delay, 1.0))

    def release(self, *, success: bool = False, backoff: float | None = None) -> dict:
        with self.condition:
            old_limit = self.limit
            self.in_flight -= 1
            if backoff is not None:
                self.limit = max(1, self.limit // 2)
                self.successes = 0
                self.cooldown_until = max(self.cooldown_until, self.clock() + backoff)
            elif success:
                self.successes += 1
                if self.successes >= max(4, self.limit):
                    self.limit = min(self.maximum, self.limit + 1)
                    self.successes = 0
            self.condition.notify_all()
            return {"old_limit": old_limit, "new_limit": self.limit,
                    "in_flight": self.in_flight,
                    "cooldown_sec": max(0.0, self.cooldown_until - self.clock())}

    def snapshot(self) -> dict:
        with self.condition:
            return {"limit": self.limit, "in_flight": self.in_flight,
                    "cooldown_sec": max(0.0, self.cooldown_until - self.clock())}


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _usage(response: Any) -> dict:
    metadata = _field(response, "usage_metadata")
    return {name: _field(metadata, source) for name, source in (
        ("input_tokens", "prompt_token_count"),
        ("output_tokens", "candidates_token_count"),
        ("thought_tokens", "thoughts_token_count"),
        ("total_tokens", "total_token_count"),
        ("cached_tokens", "cached_content_token_count"))}


def _classify(exc: Exception) -> tuple[str, bool]:
    # The message is inspected only to classify; it is never logged or returned.
    message = str(exc).lower()
    name = " ".join(cls.__name__.lower() for cls in type(exc).__mro__)
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    try:
        code = int(code)
    except (ValueError, TypeError):
        code = None
    if any(marker in message for marker in ("billing", "payment required", "payment_required")):
        return "billing", True
    if code in (401, 403) or any(marker in name for marker in ("auth", "credential", "refresherror")):
        return "auth", True
    if code == 404:
        return "model", True
    if code == 429:
        return "quota", False
    if code is not None and 500 <= code <= 599:
        return "server", False
    if code == 408 or isinstance(exc, (TimeoutError, ConnectionError)) or any(
            marker in name for marker in ("timeout", "connecterror", "network", "transport", "protocolerror")):
        return "network", False
    return "config", True


def _failure_code(exc: Exception, category: str) -> int | str:
    """Keep an HTTP status; never persist arbitrary SDK status/message strings."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    try:
        code = int(code)
    except (TypeError, ValueError):
        code = None
    return code if code is not None and 100 <= code <= 599 else category.upper()


def _retry_delay(exc: Exception, attempt: int) -> float:
    headers = _field(getattr(exc, "response", None), "headers", {}) or {}
    value = headers.get("Retry-After", headers.get("retry-after", 0))
    try:
        retry_after = float(value)
        if not math.isfinite(retry_after):
            retry_after = 0
    except (TypeError, ValueError):
        retry_after = 0
    return min(1800.0, max(2.0 ** min(attempt, 8), retry_after))


class VertexProvider:
    def __init__(self, model: str, project: str, location: str = "global",
                 service_tier: str = "standard", initial_concurrency: int = 2,
                 max_concurrency: int = 4, rpm: float = 30, attempts: int = 3,
                 timeout_sec: float = 600, *, client_factory: Callable | None = None,
                 clock: Callable = time.monotonic, sleep: Callable = time.sleep, events=None):
        if (any(type(count) is not int for count in (initial_concurrency, max_concurrency, attempts))
                or not model or not project or not location or service_tier not in ("standard", "flex")
                or (service_tier == "flex" and location != "global")
                or not 1 <= initial_concurrency <= max_concurrency
                or attempts < 1 or not math.isfinite(rpm) or rpm <= 0
                or not math.isfinite(timeout_sec) or not 0 < timeout_sec <= 1800):
            raise ProviderError("Invalid Vertex provider configuration.", "config", True)
        self.model, self.project, self.location = model, project, location
        self.service_tier, self.attempts, self.timeout_sec = service_tier, attempts, timeout_sec
        self.client_factory, self.clock = client_factory, clock
        if events is None:
            from .events import EventLog
            events = EventLog()
        self.events = events
        self.local = threading.local()
        self.limiter = RequestLimiter(initial_concurrency, max_concurrency, rpm, clock=clock, sleep=sleep)

    def _emit(self, event: str, **fields) -> None:
        """Telemetry cannot discard a paid response or strand an admission slot."""
        try:
            self.events.emit(event, **fields)
        except Exception:
            pass

    def _client(self):
        if not hasattr(self.local, "client"):
            factory = self.client_factory
            if factory is None:
                try:
                    from google import genai
                except ImportError:
                    raise ProviderError("Install the pinned google-genai dependency.", "config", True) from None
                factory = genai.Client
            headers = {}
            if self.service_tier == "flex":
                headers = {"X-Vertex-AI-LLM-Request-Type": "shared",
                           "X-Vertex-AI-LLM-Shared-Request-Type": "flex"}
            options = {"vertexai": True, "project": self.project, "location": self.location,
                       "http_options": {"api_version": "v1", "timeout": int(self.timeout_sec * 1000),
                                        "retry_options": {"attempts": 1}, "headers": headers}}
            # Explicit service-bound key only. Project/location force Vertex ADC
            # when absent, so an unrelated GEMINI_API_KEY is never selected.
            api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
            if api_key:
                options["api_key"] = api_key
            self.local.client = factory(**options)
        return self.local.client

    def _contents(self, prompt: str, context: str, images: list[tuple[str, Path]], audio: Path | None):
        files = [(label, Path(path), None) for label, path in images]
        if audio is not None:
            files.append((None, Path(audio), "audio/wav" if Path(audio).suffix.lower() == ".wav" else None))
        estimate = len(json.dumps([prompt, context], ensure_ascii=True).encode("utf-8")) + 4096
        sizes = []
        try:
            # Stat every file before reading any. Allow for base64 and JSON overhead.
            for label, path, mime in files:
                size = path.stat().st_size
                estimate += 4 * ((size + 2) // 3) + len(json.dumps(label)) + 256
                sizes.append(size)
            if estimate > MAX_INLINE_BYTES:
                raise ProviderError("Inline request exceeds 18 MiB; reduce image size or clip length.", "payload")
            parts = [{"text": context}]
            for (label, path, mime), size in zip(files, sizes):
                with path.open("rb") as stream:
                    data = stream.read(size + 1)
                if len(data) != size:
                    raise ProviderError("Media file changed while building the request.", "payload")
                if label is not None:
                    parts.append({"text": label})
                parts.append({"inline_data": {"mime_type": mime or mimetypes.guess_type(path.name)[0]
                                              or "application/octet-stream", "data": data}})
            return [{"role": "user", "parts": parts}]
        except OSError:
            raise ProviderError("Unable to read request media.", "payload") from None

    def generate(self, prompt: str, context: str, images: list[tuple[str, Path]],
                 audio: Path | None = None, max_output_tokens: int = 16384) -> dict:
        started = self.clock()
        try:
            metadata = json.loads(context)
        except (ValueError, TypeError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        fields = {"request_id": uuid.uuid4().hex,
                  "clip_id": metadata.get("clip_id") if isinstance(metadata.get("clip_id"), str) else None,
                  "phase": metadata.get("task_phase") if isinstance(metadata.get("task_phase"), str) else None,
                  "model": self.model, "service_tier": self.service_tier}
        # Retain only identifiers; context may contain private annotation data.
        del metadata
        self._emit("request_queued", **fields)

        def failed(exc, attempt, attempt_start, *, will_retry=False, delay=0.0, error_type=None):
            self._emit("request_failed", **fields, attempt=attempt, code=exc.code,
                             category=exc.category, fatal=exc.fatal, will_retry=will_retry,
                             error_type=error_type or type(exc).__name__, retry_delay_sec=delay,
                             elapsed_sec=self.clock() - attempt_start,
                             request_elapsed_sec=self.clock() - started, usage=exc.usage)

        def release(attempt, *, success=False, backoff=None, reason=None):
            change = self.limiter.release(success=success, backoff=backoff)
            if change["old_limit"] != change["new_limit"]:
                self._emit("aimd_change", **fields, attempt=attempt,
                                 reason=reason or "success", **change)
            return change

        try:
            if max_output_tokens < 1:
                raise ProviderError("max_output_tokens must be positive.", "config", True)
            contents = self._contents(prompt, context, images, audio)
        except ProviderError as exc:
            failed(exc, 0, started)
            raise
        except Exception as exc:
            error = ProviderError("Unable to build request payload.", "payload")
            failed(error, 0, started, error_type=type(exc).__name__)
            raise error from None
        for attempt in range(1, self.attempts + 1):
            queued = self.clock()
            self.limiter.acquire()
            attempt_started = self.clock()
            try:
                self._emit("request_start", **fields, attempt=attempt,
                           queue_wait_sec=attempt_started - queued)
                response = self._client().models.generate_content(
                    model=self.model, contents=contents,
                    config={"system_instruction": prompt, "response_mime_type": "application/json",
                            "max_output_tokens": max_output_tokens})
            except ProviderError as exc:
                release(attempt)
                exc.attempts = attempt
                failed(exc, attempt, attempt_started)
                raise
            except Exception as exc:
                category, fatal = _classify(exc)
                delay = 0.0 if fatal else _retry_delay(exc, attempt)
                will_retry = not fatal and attempt < self.attempts
                error = ProviderError(f"Vertex request failed ({category}).", category, fatal,
                                      attempts=attempt, code=_failure_code(exc, category),
                                      usage=_usage(exc) if _field(exc, "usage_metadata") is not None else {})
                try:
                    failed(error, attempt, attempt_started, will_retry=will_retry,
                           delay=delay, error_type=type(exc).__name__)
                finally:
                    change = release(attempt, backoff=None if fatal else delay, reason=category)
                if not will_retry:
                    raise error from None
                self._emit("retry_scheduled", **fields, attempt=attempt, next_attempt=attempt + 1,
                                 code=error.code, category=category, retry_delay_sec=delay,
                                 cooldown_sec=change["cooldown_sec"])
                continue
            except BaseException:
                release(attempt)
                raise
            release(attempt, success=True)
            usage = _usage(response)
            candidates = _field(response, "candidates", []) or []
            finish = _field(candidates[0], "finish_reason") if candidates else None
            finish = getattr(finish, "value", finish)
            try:
                if finish not in (None, "STOP"):
                    raise ValueError("Incomplete response")
                data = json.loads(_field(response, "text", ""))
                if not isinstance(data, dict):
                    raise ValueError("JSON object required")
            except (ValueError, TypeError, AttributeError):
                error = ProviderError("Vertex returned incomplete or invalid JSON output.", "output",
                                      attempts=attempt, usage=usage, code="INVALID_OUTPUT")
                failed(error, attempt, attempt_started)
                raise error from None
            traffic = _field(_field(response, "usage_metadata"), "traffic_type")
            traffic = getattr(traffic, "value", traffic)
            self._emit("request_success", **fields, attempt=attempt,
                             elapsed_sec=self.clock() - attempt_started,
                             request_elapsed_sec=self.clock() - started,
                             usage=usage, traffic_type=traffic)
            return {"data": data, "usage": usage, "elapsed_sec": self.clock() - started,
                    "attempts": attempt, "traffic_type": traffic}
        raise AssertionError("Unreachable retry state")
