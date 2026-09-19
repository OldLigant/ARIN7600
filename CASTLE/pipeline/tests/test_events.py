import json
import threading
from castle_pipeline.events import EventLog, ProgressReporter


def test_usage_counts_received_outputs_including_invalid_json(tmp_path):
    lines = []
    log = EventLog(sink=lines.append)
    log.attach(tmp_path/'events.jsonl')
    log.emit('request_start', request_id='r1', attempt=1)
    log.emit('request_failed', request_id='r1', attempt=1, code='INVALID_OUTPUT', usage={'input_tokens': 10, 'output_tokens': 2})
    log.emit('request_start', request_id='r2', attempt=1)
    log.emit('request_success', request_id='r2', attempt=1, usage={'input_tokens': 20, 'output_tokens': 5})
    s = log.snapshot()
    assert s['requests_started'] == 2 and s['requests_failed'] == 1
    assert s['input_tokens'] == 30 and s['output_tokens'] == 7
    assert s['active_requests'] == []
    assert len((tmp_path/'events.jsonl').read_text().splitlines()) == 4
    assert all('timestamp_utc' in json.loads(line) for line in lines)


def test_heartbeat_runs_while_worker_has_not_completed():
    fired = threading.Event()
    lines = []
    def sink(line):
        lines.append(json.loads(line))
        if lines[-1]['event'] == 'progress':
            fired.set()
    log = EventLog(sink=sink)
    log.emit('stage_start', clip_id='c1', phase='audio')
    with ProgressReporter(log, .02, lambda: {'completed': 0, 'selected': 3}):
        assert fired.wait(2)
    progress = next(line for line in lines if line['event'] == 'progress')
    assert progress['completed'] == 0
    assert progress['telemetry']['active_stages'][0]['phase'] == 'audio'


def test_logging_failures_cannot_break_completed_work(tmp_path):
    def broken_sink(line):
        raise BrokenPipeError('unavailable stdout')
    log = EventLog(sink=broken_sink)
    log.attach(tmp_path/'events.jsonl')
    log.emit('request_success', request_id='r', usage={'input_tokens': 12})
    log.emit('bad_metadata', value=object())
    assert log.snapshot()['input_tokens'] == 12
    assert log.snapshot()['logging_errors'] >= 2
    assert len((tmp_path/'events.jsonl').read_text().splitlines()) == 2


def test_heartbeat_recovers_from_transient_status_failure():
    delivered = threading.Event()
    calls = []
    def status():
        calls.append(1)
        if len(calls) == 1:
            raise OSError('transient unavailable telemetry')
        return {'completed': 0}
    def sink(line):
        if json.loads(line)['event'] == 'progress':
            delivered.set()
    with ProgressReporter(EventLog(sink=sink), .01, status):
        assert delivered.wait(2)
    assert len(calls) >= 2
