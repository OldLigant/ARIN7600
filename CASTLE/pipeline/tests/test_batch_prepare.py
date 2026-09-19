import json
import subprocess
from pathlib import Path
import pytest
from batch_pipeline import prepare_media, resolve_media_tuning, build_parser
from test_batch_tasks import CloudFiles
from castle_pipeline.batch_tasks import TaskPlanner


def _source(tmp_path, color='green', seconds='1.2'):
    source = tmp_path / f'source-{color}-{seconds}.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'color=c={color}:s=320x180:r=50',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000', '-t', seconds,
                    '-c:v', 'libx264', '-threads', '1', '-c:a', 'aac', str(source)], check=True)
    return source


def _cfg(**overrides):
    cfg = {'gcs_prefix': 'gs://b/run', 'fps': 1., 'clip_seconds': 30., 'max_dim': 1440,
           'start_clip': 0, 'max_clips': 2, 'review': True, 'prepare_workers': 1}
    cfg.update(overrides)
    return cfg


def test_preparation_persists_model_native_frames_audio_and_relative_times(tmp_path):
    source = _source(tmp_path)
    cloud = CloudFiles(); planner = TaskPlanner(cloud, tmp_path/'scratch', tmp_path/'output')
    rows = prepare_media(source, {'source_id': 'fixture', 'viewpoint': 'egocentric'}, _cfg(), planner)
    assert len(rows) == 1
    meta = json.loads(cloud.files[rows[0]['metadata_uri']])
    assert meta['duration_sec'] == 1.2 and meta['frame_times_sec'] == [0., 1.]
    assert len(meta['native_frames']) == len(meta['frames']) == 2
    assert all(uri in cloud.files for uri in meta['frames']+meta['native_frames']+[meta['audio_uri']])
    assert not list((tmp_path/'scratch').rglob('*.jpg'))


def test_concurrent_preparation_is_deterministic_and_keeps_row_order(tmp_path):
    """Parallel preparation must not change what is persisted, or in what order.

    Byte identity is compared between two identically-configured runs rather than
    against the serial run: FFmpeg's JPEG encoder output is only stable for a
    fixed thread count, and thread count is deliberately a tuning knob.
    """
    source = _source(tmp_path, seconds='4.2')
    rows = {}
    files = {}
    for label, workers in (('serial', 1), ('parallel-a', 3), ('parallel-b', 3)):
        cloud = CloudFiles()
        settings = _cfg(max_clips=4, clip_seconds=1., prepare_workers=workers)
        if workers == 3:
            settings.update(media_threads=2, decode_slots=2, footer_workers=3)
        rows[label] = prepare_media(source, {'source_id': 'fixture', 'viewpoint': 'egocentric'},
                                    settings, TaskPlanner(cloud, tmp_path/f'scratch-{label}', tmp_path/f'out-{label}'))
        files[label] = cloud.files
    for label in ('parallel-a', 'parallel-b'):
        assert [row['clip_id'] for row in rows[label]] == [row['clip_id'] for row in rows['serial']]
        assert len(rows[label]) == 4
    assert files['parallel-a'] == files['parallel-b']
    serial_meta = [json.loads(files['serial'][row['metadata_uri']]) for row in rows['serial']]
    parallel_meta = [json.loads(files['parallel-a'][row['metadata_uri']]) for row in rows['parallel-a']]
    assert [m['frame_times_sec'] for m in parallel_meta] == [m['frame_times_sec'] for m in serial_meta]
    assert [m['start_offset_sec'] for m in parallel_meta] == [m['start_offset_sec'] for m in serial_meta]
    assert [m['native_frames'] for m in parallel_meta] == [m['native_frames'] for m in serial_meta]
    assert [m['frames'] for m in parallel_meta] == [m['frames'] for m in serial_meta]


def test_preparation_rejects_non_positive_pool_sizes(tmp_path):
    source = _source(tmp_path)
    planner = TaskPlanner(CloudFiles(), tmp_path/'scratch', tmp_path/'output')
    for bad in (0, -1):
        with pytest.raises(ValueError):
            prepare_media(source, {'source_id': 'fixture', 'viewpoint': 'egocentric'},
                          _cfg(prepare_workers=bad), planner)
        with pytest.raises(ValueError):
            prepare_media(source, {'source_id': 'fixture', 'viewpoint': 'egocentric'},
                          _cfg(media_threads=bad), planner)


def _prepare_args(*extra):
    return build_parser().parse_args(['prepare', '--project', 'p', '--state-uri', 'gs://b/x/state.json',
                                      '--output-dir', 'out', '--scratch-dir', 'scratch',
                                      '--gcs-prefix', 'gs://b/x', '--source', 's.mp4', *extra])


def test_decoder_budget_guard_rejects_more_threads_than_the_machine_can_cover(monkeypatch):
    """Parallelism must be admitted against a stated RAM budget, not assumed free."""
    for name in ('CASTLE_MEDIA_THREADS', 'CASTLE_FOOTER_WORKERS', 'CASTLE_PREPARE_WORKERS'):
        monkeypatch.delenv(name, raising=False)
    # 4 x 2 x 40 MiB = 320 MiB, comfortably inside the 16 GiB default hint.
    assert resolve_media_tuning(_prepare_args('--media-threads', '4', '--decode-slots', '2')) == (3, 4, 4)
    # 32 x 16 x 40 MiB = 20 GiB of decoder buffers alone, above the stated 16 GiB.
    with pytest.raises(ValueError, match='decoder buffers'):
        resolve_media_tuning(_prepare_args('--media-threads', '32', '--decode-slots', '16'))
    # The same request is admitted once the operator states the larger machine.
    assert resolve_media_tuning(_prepare_args('--media-threads', '32', '--decode-slots', '16',
                                              '--memory-hint-gib', '32')) == (3, 32, 4)
    with pytest.raises(ValueError, match='memory-hint-gib must be positive'):
        resolve_media_tuning(_prepare_args('--memory-hint-gib', '0'))
    for bad in ('0', '33'):
        with pytest.raises(ValueError, match='media-threads'):
            resolve_media_tuning(_prepare_args('--media-threads', bad))
