"""Offline tests for the auxiliary tick worker; never use production outputs."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import batch_tick


class Blob:
    def __init__(self, name, data, digest=True):
        self.name, self.data, self.generation = name, data, 7
        self.metadata = {'sha256': hashlib.sha256(data).hexdigest()} if digest else {}
        self.downloads = 0

    def download_as_bytes(self, *, if_generation_match):
        assert if_generation_match == 7
        self.downloads += 1
        return self.data


class Storage:
    def __init__(self, blobs):
        self.blobs = blobs

    def list_blobs(self, bucket, *, prefix):
        assert bucket == 'test'
        assert prefix == 'run/results/'
        return iter(self.blobs)


def test_mirror_copies_all_stages_and_skips_identical_files_on_replay(tmp_path):
    blobs = [Blob(f'run/results/{stage}/c1.json', json.dumps({'data': {'phase': stage}, 'usage': {}}).encode())
             for stage in ['audio', 'annotation', 'review']]
    cloud = SimpleNamespace(storage=Storage(blobs))
    output = tmp_path / 'output'
    assert batch_tick.mirror_results(cloud, 'gs://test/run', output) == {'copied': 3, 'unchanged': 0}
    for blob in blobs:
        assert (output / blob.name.removeprefix('run/')).read_bytes() == blob.data
    assert batch_tick.mirror_results(cloud, 'gs://test/run', output) == {'copied': 0, 'unchanged': 3}
    assert [blob.downloads for blob in blobs] == [1, 1, 1]
    assert not (output / 'final').exists()


def test_mirror_rejects_existing_different_content_without_overwriting(tmp_path):
    target = tmp_path / 'results/audio/c1.json'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'original-user-data')
    cloud = SimpleNamespace(storage=Storage([Blob('run/results/audio/c1.json', b'{"data":{}}')]))
    with pytest.raises(ValueError, match='different content'):
        batch_tick.mirror_results(cloud, 'gs://test/run', tmp_path)
    assert target.read_bytes() == b'original-user-data'


def test_mirror_rechecks_destination_after_download(tmp_path, monkeypatch):
    target = tmp_path / 'results/audio/c1.json'
    blob = Blob('run/results/audio/c1.json', b'{"data":{}}')

    def download(**kwargs):
        target.parent.mkdir(parents=True)
        target.write_bytes(b'intervening-data')
        return blob.data

    monkeypatch.setattr(blob, 'download_as_bytes', download)
    with pytest.raises(ValueError, match='different content'):
        batch_tick.mirror_results(SimpleNamespace(storage=Storage([blob])), 'gs://test/run', tmp_path)
    assert target.read_bytes() == b'intervening-data'


def test_mirror_ignores_unrelated_and_unsafe_paths(tmp_path):
    blobs = [Blob(name, b'{}') for name in [
        'run/results/audio/../../escape.json', 'run/results/audio/subdir/c1.json',
        'run/results/unknown/c1.json', 'run/results/audio/c1.txt', 'other/results/audio/c1.json',
        'run/media/c1/native-000.jpg', 'run/media/c1/frame-000.jpg', 'run/raw/audio/c1.json',
        'run/errors/audio/c1.json', 'run/final/c1.json', 'run/requests/audio-abc.jsonl']]
    assert batch_tick.mirror_results(SimpleNamespace(storage=Storage(blobs)), 'gs://test/run', tmp_path) == {
        'copied': 0, 'unchanged': 0}
    assert list(tmp_path.iterdir()) == []
    assert all(blob.downloads == 0 for blob in blobs)


@pytest.mark.parametrize('workers', [1, 4, 16])
def test_mirror_threading_publishes_every_object_with_exact_counts(tmp_path, workers):
    """Concurrency must not change what is copied, only how fast."""
    blobs = [Blob('run/results/%s/c%03d.json' % (stage, i),
                  json.dumps({'data': {'phase': stage, 'i': i}}).encode())
             for stage in ['audio', 'annotation', 'review'] for i in range(20)]
    cloud = SimpleNamespace(storage=Storage(blobs))
    output = tmp_path / ('out-%d' % workers)
    assert batch_tick.mirror_results(cloud, 'gs://test/run', output, workers=workers) == {
        'copied': len(blobs), 'unchanged': 0}
    for blob in blobs:
        assert (output / blob.name.removeprefix('run/')).read_bytes() == blob.data
    assert [blob.downloads for blob in blobs] == [1] * len(blobs)
    assert not list(output.rglob('*.tmp'))
    assert batch_tick.mirror_results(cloud, 'gs://test/run', output, workers=workers) == {
        'copied': 0, 'unchanged': len(blobs)}


@pytest.mark.parametrize('workers', [1, 8])
def test_mirror_reports_the_same_conflict_whatever_the_thread_order(tmp_path, workers):
    """A failing object set must not report whichever thread lost the race."""
    output = tmp_path / ('out-%d' % workers)
    blobs = []
    for name in ['audio/b.json', 'audio/a.json', 'annotation/c.json']:
        blob = Blob('run/results/' + name, b'{"data":{}}')
        blobs.append(blob)
        target = output / 'results' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'private-old-content')
    with pytest.raises(batch_tick.MirrorConflict) as caught:
        batch_tick.mirror_results(SimpleNamespace(storage=Storage(blobs)), 'gs://test/run', output, workers=workers)
    # Lexicographically first object name wins, so 1 and 8 threads agree.
    assert caught.value.relative == 'annotation/c.json'
    assert all(blob.downloads == 0 for blob in blobs)


def test_mirror_worker_count_is_env_driven_and_bounded(tmp_path, monkeypatch):
    cloud = SimpleNamespace(storage=Storage([Blob('run/results/audio/c1.json', b'{"data":{}}')]))
    monkeypatch.delenv('CASTLE_MIRROR_WORKERS', raising=False)
    assert batch_tick.env_mirror_workers() == batch_tick.DEFAULT_MIRROR_WORKERS
    assert batch_tick.mirror_results(cloud, 'gs://test/run', tmp_path) == {'copied': 1, 'unchanged': 0}
    monkeypatch.setenv('CASTLE_MIRROR_WORKERS', '3')
    assert batch_tick.env_mirror_workers() == 3
    for bad in ['0', '65', 'many', '-1', '']:
        monkeypatch.setenv('CASTLE_MIRROR_WORKERS', bad)
        if bad == '':
            assert batch_tick.env_mirror_workers() == batch_tick.DEFAULT_MIRROR_WORKERS
            continue
        with pytest.raises(ValueError):
            batch_tick.env_mirror_workers()
    for bad in [0, 65, True, '4']:
        with pytest.raises(ValueError):
            batch_tick.mirror_results(cloud, 'gs://test/run', tmp_path, workers=bad)


def test_mirror_still_copies_when_the_worker_count_env_is_invalid(worker, monkeypatch, tmp_path, capsys):
    """A bad knob must not stop the tick; the mirror reports failure and exits non-zero."""
    cloud, _, command = worker
    cloud.storage.blobs.append(Blob('run/results/audio/c1.json', b'{"data":{}}'))
    monkeypatch.setattr(batch_tick.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    monkeypatch.setenv('CASTLE_MIRROR_WORKERS', 'lots')
    assert batch_tick.main(command) == 1
    assert json.loads(capsys.readouterr().err)['event'] == 'batch_stage_mirror_failed'
    assert not (tmp_path / 'output/results/audio/c1.json').exists()


@pytest.mark.parametrize('bad', ['digest', 'json'])
def test_mirror_validates_before_publishing_a_file(tmp_path, bad):
    blob = Blob('run/results/audio/c1.json', b'not-json' if bad == 'json' else b'{}')
    if bad == 'digest':
        blob.metadata['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        batch_tick.mirror_results(SimpleNamespace(storage=Storage([blob])), 'gs://test/run', tmp_path)
    assert not (tmp_path / 'results/audio/c1.json').exists()


def test_mirror_can_read_legacy_objects_without_digest_metadata(tmp_path):
    blob = Blob('run/results/audio/c1.json', b'{"data": {}, "skipped": true}', digest=False)
    cloud = SimpleNamespace(storage=Storage([blob]))
    assert batch_tick.mirror_results(cloud, 'gs://test/run', tmp_path)['copied'] == 1
    assert batch_tick.mirror_results(cloud, 'gs://test/run', tmp_path)['unchanged'] == 1


@pytest.fixture
def worker(tmp_path, monkeypatch):
    import batch_pipeline
    state = {'config': {'project': 'test-project', 'location': 'global', 'gcs_prefix': 'gs://test/run',
                        'code_hash': batch_pipeline.code_hash()}, 'status': 'running'}
    cloud = SimpleNamespace(read_state=lambda uri: (state, 1), storage=Storage([]))
    monkeypatch.setattr(batch_tick, 'BatchCloud', lambda *a, **kw: cloud)
    command = ['tick', '--project', 'test-project', '--location', 'global',
               '--state-uri', 'gs://test/run/state.json', '--output-dir', str(tmp_path / 'output'),
               '--scratch-dir', str(tmp_path / 'scratch')]
    return cloud, state, command


def test_hourly_worker_waits_without_tick_or_output_if_state_missing(worker, monkeypatch, capsys, tmp_path):
    cloud, _, command = worker
    cloud.read_state = lambda uri: (None, 0)
    monkeypatch.setattr(batch_tick.subprocess, 'run', lambda *a, **kw: pytest.fail('Must not tick'))
    assert batch_tick.main(command + ['--allow-missing-state']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'waiting_for_state'
    assert not (tmp_path / 'output').exists()
    with pytest.raises(ValueError, match='No batch state'):
        batch_tick.main(command)


@pytest.mark.parametrize('field,value', [('code_hash', 'wrong'), ('project', 'wrong'),
                                       ('location', 'wrong'), ('gcs_prefix', 'gs://test/other')])
def test_wrapper_rejects_incompatible_state_before_tick_or_mirror(worker, monkeypatch, field, value):
    _, state, command = worker
    state['config'][field] = value
    monkeypatch.setattr(batch_tick.subprocess, 'run', lambda *a, **kw: pytest.fail('Must not tick'))
    monkeypatch.setattr(batch_tick, 'mirror_results', lambda *a: pytest.fail('Must not mirror'))
    with pytest.raises(ValueError):
        batch_tick.main(command)


@pytest.mark.parametrize('exit_code', [0, 1, 2])
def test_wrapper_mirrors_collected_results_even_when_tick_reports_failure(worker, monkeypatch, exit_code, tmp_path):
    cloud, _, command = worker
    blob = Blob('run/results/audio/c1.json', b'{"data": {"summary": "quiet"}}')

    def tick(argv, **kwargs):
        assert Path(argv[1]).name == 'batch_pipeline.py'
        assert argv[2:] == command
        cloud.storage.blobs.append(blob)
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(batch_tick.subprocess, 'run', tick)
    assert batch_tick.main(command) == exit_code
    assert (tmp_path / 'output/results/audio/c1.json').read_bytes() == blob.data


def test_complete_run_backfills_stage_files_without_model_calls(worker, monkeypatch, tmp_path):
    cloud, state, command = worker
    state['status'] = 'complete'
    cloud.storage.blobs.append(Blob('run/results/annotation/c1.json', b'{"data":{}}'))
    monkeypatch.setattr(batch_tick.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    assert batch_tick.main(command) == 0
    assert (tmp_path / 'output/results/annotation/c1.json').exists()


def test_mirror_failure_does_not_report_success(worker, monkeypatch):
    _, _, command = worker
    monkeypatch.setattr(batch_tick.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(batch_tick, 'mirror_results', lambda *a: (_ for _ in ()).throw(OSError('disk full')))
    assert batch_tick.main(command) != 0


def test_conflict_log_identifies_safe_relative_path_without_output_contents(worker, monkeypatch, capsys, tmp_path):
    cloud, _, command = worker
    cloud.storage.blobs.append(Blob('run/results/audio/c1.json', b'{"data":{}}'))
    target = tmp_path / 'output/results/audio/c1.json'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'private-old-content')
    monkeypatch.setattr(batch_tick.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    assert batch_tick.main(command) == 1
    output = capsys.readouterr().err
    assert json.loads(output)['file'] == 'audio/c1.json'
    assert json.loads(output)['reason'] == 'different_content'
    assert 'private-old-content' not in output
