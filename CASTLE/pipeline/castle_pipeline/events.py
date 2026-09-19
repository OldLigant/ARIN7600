"""Structured, flushed observability without retaining media or response text."""
import json
import threading
import time
import math
from pathlib import Path


class EventLog:
    def __init__(self, sink=None):
        self.sink = sink or (lambda line: print(line, flush=True))
        self.lock = threading.RLock()
        self.path = None
        self.started = time.monotonic()
        self.counts = dict(requests_started=0, requests_succeeded=0, requests_failed=0,
                           input_tokens=0, output_tokens=0, thought_tokens=0, cached_tokens=0, total_tokens=0,
                           responses_without_usage=0, logging_errors=0)
        self.requests, self.waiting, self.stages = {}, {}, {}

    def attach(self, path):
        with self.lock:
            self.path = Path(path) if path is not None else None
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event, **fields):
        now = time.monotonic()
        record = {'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  'event': event, **fields}
        with self.lock:
            rid = fields.get('request_id')
            cid = fields.get('clip_id')
            if event == 'request_queued' or event == 'retry_scheduled':
                self.waiting[rid] = {'request_id': rid, 'clip_id': cid, 'phase': fields.get('phase'), 'since': now}
            if event == 'request_start':
                self.counts['requests_started'] += 1
                self.waiting.pop(rid, None)
                self.requests[rid] = {'request_id': rid, 'clip_id': cid, 'phase': fields.get('phase'),
                                      'attempt': fields.get('attempt'), 'since': now}
            if event in ('request_success', 'request_failed'):
                self.requests.pop(rid, None)
                self.waiting.pop(rid, None)
                self.counts['requests_succeeded' if event == 'request_success' else 'requests_failed'] += 1
                usage = fields.get('usage') or {}
                if not usage or usage.get('input_tokens') is None:
                    self.counts['responses_without_usage'] += 1
                for key in ('input_tokens', 'output_tokens', 'thought_tokens', 'cached_tokens', 'total_tokens'):
                    value = usage.get(key)
                    if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
                        self.counts[key] += value
            if event == 'stage_start':
                self.stages[cid] = {'clip_id': cid, 'phase': fields.get('phase'), 'since': now}
            if event in ('stage_success', 'stage_failed', 'stage_reused', 'clip_failed', 'clip_completed', 'clip_reused'):
                self.stages.pop(cid, None)
            try:
                line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            except (ValueError, TypeError):
                self.counts['logging_errors'] += 1
                line = json.dumps({'timestamp_utc': record['timestamp_utc'], 'event': 'log_serialization_failed'})
            self._safe_sink(line)
            if self.path:
                try:
                    with self.path.open('a', encoding='utf-8') as handle:
                        handle.write(line + '\n')
                        handle.flush()
                except Exception:
                    # A log mount error must not erase a paid response/checkpoint.
                    self.path = None
                    self.counts['logging_errors'] += 1
                    self._safe_sink(json.dumps({'timestamp_utc': record['timestamp_utc'], 'event': 'log_file_unavailable'}))

    def _safe_sink(self, line):
        try:
            self.sink(line)
        except Exception:
            # Logging must not discard a paid model response or hold an API slot.
            self.counts['logging_errors'] += 1

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            def elapsed(items):
                return [{**{k: v for k, v in item.items() if k != 'since'},
                         'elapsed_sec': round(now - item['since'], 1)} for item in items.values()]
            return {**self.counts, 'elapsed_sec': round(now - self.started, 1),
                    'active_requests': elapsed(self.requests), 'waiting_requests': elapsed(self.waiting),
                    'active_stages': elapsed(self.stages), 'usage_scope': 'current_process_received_responses_only'}


class ProgressReporter:
    """Independent heartbeat: works even while every API worker is blocked."""
    def __init__(self, events, interval_sec, status):
        self.events, self.interval, self.status = events, interval_sec, status
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name='progress', daemon=True)

    def _run(self):
        while not self.stop.wait(self.interval):
            try:
                self.events.emit('progress', **self.status(), telemetry=self.events.snapshot())
            except Exception as error:
                self.events.emit('progress_unavailable', error_type=type(error).__name__)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
