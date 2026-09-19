"""Offline behavioral checks. The only double is the paid SDK boundary."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace as NS
import threading
import time

import pytest

from castle_pipeline.vertex import VertexProvider, ProviderError, RequestLimiter


class Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []
        self.on_sleep = None

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        if self.on_sleep:
            self.on_sleep()
        self.waits.append(seconds)
        self.now += seconds


class APIError(Exception):
    def __init__(self, code, message="request failed", retry_after=None):
        super().__init__(message)
        self.code = code
        self.response = NS(headers={} if retry_after is None else {"Retry-After": str(retry_after)})


def response(text='{"segments": []}', finish="STOP"):
    return NS(text=text, candidates=[NS(finish_reason=finish)], usage_metadata=NS(
        prompt_token_count=11, candidates_token_count=22, thoughts_token_count=33,
        total_token_count=66, cached_content_token_count=0, traffic_type="ON_DEMAND_FLEX"))


class Boundary:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.options = []
        self.requests = []
        self.models = self

    def __call__(self, **options):
        self.options.append(options)
        return self

    def generate_content(self, **request):
        self.requests.append(request)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


def provider(boundary, **kwargs):
    return VertexProvider("gemini-test", "project-test", client_factory=boundary, **kwargs)


def test_request_preserves_parts_and_parses_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "service-bound-test-key")
    photo, audio = tmp_path / "frame.jpg", tmp_path / "audio.wav"
    photo.write_bytes(b"jpeg")
    audio.write_bytes(b"wave")
    boundary = Boundary([response()])
    p = provider(boundary, service_tier="flex", timeout_sec=45)
    assert boundary.options == []  # Initialization cannot perform paid work.
    result = p.generate("system prompt", "clip context", [("t=0", photo)], audio, 1234)
    assert result["data"] == {"segments": []}
    assert result["usage"]["input_tokens"] == 11
    assert result["usage"]["output_tokens"] == 22
    assert result["usage"]["thought_tokens"] == 33
    assert result["usage"]["total_tokens"] == 66
    assert result["traffic_type"] == "ON_DEMAND_FLEX"
    assert result["attempts"] == 1
    assert result["elapsed_sec"] >= 0
    opts = boundary.options[0]
    assert opts["vertexai"] is True
    assert opts["project"] == "project-test"
    assert opts["location"] == "global"
    assert opts["api_key"] == "service-bound-test-key"
    assert opts["http_options"]["timeout"] == 45000
    assert opts["http_options"]["retry_options"] == {"attempts": 1}
    assert opts["http_options"]["headers"] == {
        "X-Vertex-AI-LLM-Request-Type": "shared",
        "X-Vertex-AI-LLM-Shared-Request-Type": "flex"}
    req = boundary.requests[0]
    assert req["model"] == "gemini-test"
    assert req["config"]["system_instruction"] == "system prompt"
    assert req["config"]["max_output_tokens"] == 1234
    assert req["config"]["response_mime_type"] == "application/json"
    assert req["contents"] == [{"role": "user", "parts": [
        {"text": "clip context"}, {"text": "t=0"},
        {"inline_data": {"mime_type": "image/jpeg", "data": b"jpeg"}},
        {"inline_data": {"mime_type": "audio/wav", "data": b"wave"}}]}]


def test_adc_does_not_take_developer_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "developer-only")
    boundary = Boundary([response()])
    provider(boundary).generate("p", "c", [])
    assert "api_key" not in boundary.options[0]
    assert boundary.options[0]["vertexai"] is True


@pytest.mark.parametrize("service_key", [True, False])
def test_installed_sdk_constructs_vertex_client_offline_and_ignores_developer_key(monkeypatch, service_key):
    """Characterize the installed SDK at the actual auth/endpoint boundary.

    Removing vertexai/project or accidentally selecting the Developer API key
    must fail this check. Construction only: never call generate_content.
    """
    pytest.importorskip("google.genai")
    import socket

    def forbid_network(*args, **kwargs):
        pytest.fail("SDK construction must remain offline")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket, "create_connection", forbid_network)
    monkeypatch.setenv("GEMINI_API_KEY", "offline-developer-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_VERTEX_BASE_URL", raising=False)
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)
    if service_key:
        monkeypatch.setenv("GOOGLE_API_KEY", "offline-service-bound-key")

    p = VertexProvider("gemini-test", "project-test", service_tier="flex", timeout_sec=123)
    client = p._client()
    try:
        actual = client._api_client
        assert actual.vertexai is True
        assert actual.project == "project-test"
        assert actual.location == "global"
        assert actual._http_options.base_url == "https://aiplatform.googleapis.com/"
        assert actual._http_options.api_version == "v1"
        assert actual._http_options.timeout == 123000
        assert actual._http_options.retry_options.attempts == 1
        assert actual._http_options.headers["X-Vertex-AI-LLM-Request-Type"] == "shared"
        assert actual._http_options.headers["X-Vertex-AI-LLM-Shared-Request-Type"] == "flex"
        if service_key:
            assert actual.api_key == "offline-service-bound-key"
        else:
            assert actual.api_key is None
            assert "x-goog-api-key" not in actual._http_options.headers
    finally:
        client.close()


@pytest.mark.parametrize("kwargs", [{"service_tier": "flex", "location": "us-central1"},
    {"rpm": 0}, {"attempts": 0}, {"initial_concurrency": 5, "max_concurrency": 4},
    {"timeout_sec": -1}, {"service_tier": "bogus"}])
def test_invalid_configuration_fails_before_client_creation(kwargs):
    boundary = Boundary([])
    with pytest.raises(ProviderError) as caught:
        provider(boundary, **kwargs)
    assert caught.value.fatal and caught.value.category == "config"
    assert not boundary.options


@pytest.mark.parametrize("error,category", [(APIError(400), "config"),
    (APIError(401), "auth"), (APIError(403), "auth"), (APIError(404), "model"),
    (APIError(429, "BILLING_DISABLED"), "billing")])
def test_terminal_errors_never_retry_and_redact_raw_messages(error, category):
    error.args = (str(error) + " Bearer SECRET_TOKEN https://host/path?key=PRIVATE",)
    boundary = Boundary([error, response()])
    with pytest.raises(ProviderError) as caught:
        provider(boundary).generate("p", "c", [])
    assert caught.value.fatal and caught.value.category == category
    assert "SECRET_TOKEN" not in str(caught.value)
    assert "PRIVATE" not in str(caught.value)
    assert len(boundary.requests) == 1


@pytest.mark.parametrize("failure", [APIError(429, retry_after=7), APIError(503), TimeoutError("secret")])
def test_transient_retry_uses_shared_cooldown_and_releases_slot(failure):
    clock = Clock()
    boundary = Boundary([failure, response()])
    p = provider(boundary, clock=clock, sleep=clock.sleep, initial_concurrency=4)
    clock.on_sleep = lambda: assert_no_inflight(p)
    result = p.generate("p", "c", [])
    assert result["attempts"] == 2
    assert p.limiter.snapshot()["limit"] == 2
    assert clock.now >= (7 if getattr(failure, "code", None) == 429 else 2)


def assert_no_inflight(p):
    assert p.limiter.snapshot()["in_flight"] == 0


def test_retry_exhaustion_is_nonfatal_and_bounded():
    clock = Clock()
    boundary = Boundary([APIError(429)] * 5)
    p = provider(boundary, attempts=3, clock=clock, sleep=clock.sleep)
    with pytest.raises(ProviderError) as caught:
        p.generate("p", "c", [])
    assert caught.value.category == "quota" and not caught.value.fatal
    assert caught.value.attempts == 3
    assert len(boundary.requests) == 3
    assert_no_inflight(p)


def test_http_transport_read_error_is_retryable():
    import httpx
    clock = Clock()
    boundary = Boundary([httpx.ReadError("connection reset with secret"), response()])
    result = provider(boundary, clock=clock, sleep=clock.sleep).generate("p", "c", [])
    assert result["data"] == {"segments": []}
    assert result["attempts"] == 2


def test_http_request_timeout_is_retryable():
    clock = Clock()
    boundary = Boundary([APIError(408), response()])
    result = provider(boundary, clock=clock, sleep=clock.sleep).generate("p", "c", [])
    assert result["attempts"] == 2


@pytest.mark.parametrize("kwargs", [{"attempts": 1.5}, {"initial_concurrency": 1.5},
                                    {"max_concurrency": 4.5}])
def test_noninteger_counts_are_configuration_errors(kwargs):
    with pytest.raises(ProviderError) as caught:
        provider(Boundary([]), **kwargs)
    assert caught.value.fatal and caught.value.category == "config"


@pytest.mark.parametrize("reply", [response("{"), response("[]"),
    response('{"ok":true}', "MAX_TOKENS"), response("", "SAFETY")])
def test_invalid_output_is_resumable_without_paid_automatic_retry(reply):
    boundary = Boundary([reply, response()])
    with pytest.raises(ProviderError) as caught:
        provider(boundary).generate("p", "c", [])
    assert caught.value.category == "output" and not caught.value.fatal
    assert caught.value.usage["thought_tokens"] == 33
    assert len(boundary.requests) == 1


def test_rpm_pacing_shared_across_audio_vision_review(tmp_path):
    clock = Clock()
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio")
    boundary = Boundary([response()] * 3)
    p = provider(boundary, rpm=30, clock=clock, sleep=clock.sleep)
    p.generate("audio", "c", [], audio)
    p.generate("vision", "c", [])
    p.generate("review", "c", [])
    assert clock.now == pytest.approx(4.0)


def test_oversized_inline_payload_is_rejected_before_open(tmp_path, monkeypatch):
    large = tmp_path / "large.jpg"
    with large.open("wb") as stream:
        stream.truncate(14 * 1024 * 1024)
    def forbidden_open(*args, **kwargs):
        pytest.fail("oversized payload must be rejected before reading")
    monkeypatch.setattr(Path, "open", forbidden_open)
    boundary = Boundary([])
    with pytest.raises(ProviderError) as caught:
        provider(boundary).generate("p", "c", [("0", large)])
    assert caught.value.category == "payload" and not caught.value.fatal
    assert not boundary.options


def test_limiter_additive_increase_is_gradual_and_capped():
    clock = Clock()
    limiter = RequestLimiter(1, 3, 6000, clock=clock, sleep=clock.sleep)
    for _ in range(4):
        limiter.acquire()
        limiter.release(success=True)
    assert limiter.snapshot()["limit"] == 2
    for _ in range(30):
        limiter.acquire()
        limiter.release(success=True)
    assert limiter.snapshot()["limit"] == 3
    limiter.acquire()
    limiter.release(backoff=5)
    assert limiter.snapshot()["limit"] == 1
    limiter.acquire()
    assert clock.now >= 5
    limiter.release()


def test_real_threads_share_concurrency_limit_and_use_thread_local_clients():
    lock = threading.Lock()
    active = 0
    peak = 0
    client_threads = []
    def factory(**kwargs):
        client_threads.append(threading.get_ident())
        class Client:
            @property
            def models(self):
                return self
            def generate_content(self, **request):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(active, peak)
                time.sleep(0.02)
                with lock:
                    active -= 1
                return response()
        return Client()
    p = provider(factory, initial_concurrency=2, max_concurrency=2, rpm=60000)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: p.generate("p", "c", []), range(8)))
    assert len(results) == 8 and peak == 2
    assert len(client_threads) == len(set(client_threads)) == 4
    assert_no_inflight(p)


class Events:
    def __init__(self):
        self.records = []

    def emit(self, event, **fields):
        self.records.append({"event": event, **fields})


def test_request_events_connect_429_retry_success_usage_and_aimd_without_secrets():
    events, clock = Events(), Clock()
    boundary = Boundary([APIError(429, "Bearer SECRET https://host?key=SECRET", retry_after=7), response()])
    p = provider(boundary, events=events, clock=clock, sleep=clock.sleep, initial_concurrency=4)
    result = p.generate("PRIVATE PROMPT", json.dumps({"clip_id": "day1/Allie:3", "task_phase": "annotation",
                                                   "private": "PRIVATE CONTEXT"}), [])
    rows = events.records
    assert [r["event"] for r in rows] == ["request_queued", "request_start", "request_failed",
        "aimd_change", "retry_scheduled", "request_start", "request_success"]
    assert len({r["request_id"] for r in rows}) == 1
    assert all(r["clip_id"] == "day1/Allie:3" and r["phase"] == "annotation" for r in rows)
    failed = rows[2]
    assert failed["code"] == 429 and failed["category"] == "quota"
    assert failed["will_retry"] is True and failed["fatal"] is False
    assert failed["retry_delay_sec"] == 7 and failed["attempt"] == 1
    assert rows[3]["old_limit"] == 4 and rows[3]["new_limit"] == 2
    assert rows[3]["reason"] == "quota"
    assert rows[4]["cooldown_sec"] == 7
    assert rows[5]["attempt"] == 2 and rows[5]["queue_wait_sec"] == 7
    assert rows[5]["model"] == "gemini-test" and rows[5]["service_tier"] == "standard"
    assert rows[-1]["usage"] == result["usage"]
    assert rows[-1]["usage"]["thought_tokens"] == 33
    assert rows[-1]["traffic_type"] == "ON_DEMAND_FLEX"
    serialized = json.dumps(rows)
    assert all(secret not in serialized for secret in ("SECRET", "PRIVATE PROMPT", "PRIVATE CONTEXT", "https://host"))


def test_fatal_auth_event_propagates_http_code_without_retry():
    events = Events()
    p = provider(Boundary([APIError(401, "Bearer SECRET")]), events=events)
    with pytest.raises(ProviderError) as caught:
        p.generate("p", '{"clip_id":"c:0","task_phase":"audio"}', [])
    assert caught.value.code == 401
    assert [r["event"] for r in events.records] == ["request_queued", "request_start", "request_failed"]
    failed = events.records[-1]
    assert failed["fatal"] is True and failed["will_retry"] is False
    assert failed["code"] == 401 and failed["category"] == "auth"
    assert "SECRET" not in json.dumps(events.records)


def test_invalid_output_event_records_consumed_tokens_without_success():
    events = Events()
    p = provider(Boundary([response("{")]), events=events)
    with pytest.raises(ProviderError) as caught:
        p.generate("p", '{"clip_id":"c:1","task_phase":"review"}', [])
    assert caught.value.code == "INVALID_OUTPUT"
    failed = events.records[-1]
    assert failed["event"] == "request_failed" and failed["code"] == "INVALID_OUTPUT"
    assert failed["usage"]["output_tokens"] == 22 and failed["usage"]["thought_tokens"] == 33
    assert failed["will_retry"] is False and failed["fatal"] is False
    assert not any(r["event"] == "request_success" for r in events.records)


def test_preflight_error_has_phase_identity_but_no_request_start(tmp_path):
    events = Events()
    p = provider(Boundary([]), events=events)
    with pytest.raises(ProviderError) as caught:
        p.generate("p", '{"clip_id":"c:2","task_phase":"annotation"}', [("0", tmp_path / "absent.jpg")])
    assert caught.value.code == "PAYLOAD"
    assert [r["event"] for r in events.records] == ["request_queued", "request_failed"]
    assert events.records[-1]["phase"] == "annotation" and events.records[-1]["attempt"] == 0


def test_successful_requests_emit_only_actual_aimd_cap_changes():
    events, clock = Events(), Clock()
    p = provider(Boundary([response()] * 8), events=events, initial_concurrency=1, max_concurrency=2,
                 clock=clock, sleep=clock.sleep)
    for _ in range(8):
        p.generate("p", "unstructured context", [])
    changes = [r for r in events.records if r["event"] == "aimd_change"]
    assert len(changes) == 1
    assert changes[0]["old_limit"] == 1 and changes[0]["new_limit"] == 2
    assert changes[0]["reason"] == "success"
    assert len({r["request_id"] for r in events.records if r["event"] == "request_queued"}) == 8


def test_real_event_log_clears_waiting_and_counts_retry_and_invalid_output_tokens(tmp_path):
    from castle_pipeline.events import EventLog
    lines, clock = [], Clock()
    events = EventLog(sink=lines.append)
    p = provider(Boundary([APIError(429), response(), response("{")]), events=events,
                 clock=clock, sleep=clock.sleep)
    p.generate("p", '{"clip_id":"c:0","task_phase":"annotation"}', [])
    with pytest.raises(ProviderError):
        p.generate("p", '{"clip_id":"c:1","task_phase":"review"}', [])
    with pytest.raises(ProviderError):
        p.generate("p", '{"clip_id":"c:2","task_phase":"audio"}', [], tmp_path / "missing.wav")
    state = events.snapshot()
    assert state["requests_started"] == 3
    assert state["requests_succeeded"] == 1 and state["requests_failed"] == 3
    assert state["input_tokens"] == 22 and state["output_tokens"] == 44
    assert state["thought_tokens"] == 66 and state["cached_tokens"] == 0
    assert state["active_requests"] == [] and state["waiting_requests"] == []
    assert all(isinstance(json.loads(line), dict) for line in lines)


@pytest.mark.parametrize("broken_event", ["request_queued", "request_start", "request_success"])
def test_logging_failure_does_not_discard_response_or_leak_request_slot(broken_event):
    class BrokenEvents:
        def emit(self, event, **fields):
            if event == broken_event:
                raise BrokenPipeError("log sink closed")
    p = provider(Boundary([response()]), events=BrokenEvents())
    result = p.generate("p", "c", [])
    assert result["data"] == {"segments": []}
    assert_no_inflight(p)


def test_failure_logging_exception_does_not_break_retry_or_leak_request_slot():
    class BrokenEvents:
        def emit(self, event, **fields):
            if event in ("request_failed", "aimd_change", "retry_scheduled"):
                raise RuntimeError("log sink failed")
    clock = Clock()
    p = provider(Boundary([APIError(429), response()]), events=BrokenEvents(),
                 clock=clock, sleep=clock.sleep)
    assert p.generate("p", "c", [])["attempts"] == 2
    assert_no_inflight(p)


def test_keyboard_interrupt_during_start_logging_releases_slot_and_propagates():
    class InterruptedEvents:
        def emit(self, event, **fields):
            if event == "request_start":
                raise KeyboardInterrupt()
    p = provider(Boundary([]), events=InterruptedEvents())
    with pytest.raises(KeyboardInterrupt):
        p.generate("p", "c", [])
    assert_no_inflight(p)
