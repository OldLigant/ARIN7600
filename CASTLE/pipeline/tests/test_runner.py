import itertools
import threading
import time
import pytest
from castle_pipeline.runner import (bounded_map, Checkpoints, clip_windows, memory_estimate,
                                    source_lease, RunConfig, Pipeline)


def test_lazy_scheduler_does_not_exhaust_video_iterator():
    produced = []
    release = threading.Event()
    def inputs():
        for i in range(1000):
            produced.append(i)
            yield i
    def worker(i):
        assert release.wait(3)
        return i
    result = []
    thread = threading.Thread(target=lambda: result.append(next(bounded_map(inputs(), worker, 3))))
    thread.start()
    time.sleep(.1)
    assert len(produced) == 3
    release.set()
    thread.join(4)
    assert not thread.is_alive()
    assert len(produced) == 3


def test_tail_clip_is_not_dropped_and_no_zero_length_window():
    assert list(clip_windows(65, 30)) == [(0, 0., 30.), (1, 30., 30.), (2, 60., 5.)]
    assert list(clip_windows(60, 30)) == [(0, 0., 30.), (1, 30., 30.)]
    with pytest.raises(ValueError):
        list(clip_windows(60, 0))


def test_checkpoint_fingerprint_prevents_stale_resume(tmp_path):
    checkpoint = Checkpoints(tmp_path, 'fingerprint-a')
    checkpoint.save('audio', {'data': {'utterances': []}})
    assert checkpoint.load('audio')['data'] == {'utterances': []}
    assert Checkpoints(tmp_path, 'fingerprint-b').load('audio') is None
    (tmp_path / 'audio.json').write_text('{partial', encoding='utf-8')
    assert checkpoint.load('audio') is None


def test_memory_bound_depends_on_workers_not_source_length():
    one = memory_estimate(source_seconds=3600, workers=3)
    ten = memory_estimate(source_seconds=36000, workers=3)
    assert one['bounded_working_set_mib_estimate'] == ten['bounded_working_set_mib_estimate']
    assert ten['all_raw_1fps_gib'] == pytest.approx(one['all_raw_1fps_gib'] * 10)


def test_two_jobs_cannot_write_the_same_source_run(tmp_path):
    with source_lease(tmp_path):
        with pytest.raises(RuntimeError):
            with source_lease(tmp_path):
                pytest.fail('Overlapping run acquired the same output lease')
    assert not (tmp_path/'run.lock').exists()


def test_run_config_rejects_unusable_media_tuning(tmp_path):
    for kwargs in ({'media_threads': 0}, {'media_threads': 33}, {'media_threads': True},
                   {'decode_slots': 0}, {'decode_slots': 17}, {'footer_workers': 0}, {'footer_workers': 17}):
        with pytest.raises(ValueError):
            RunConfig(tmp_path, tmp_path, **kwargs)
    config = RunConfig(tmp_path, tmp_path, media_threads=4, decode_slots=2, footer_workers=3)
    assert (config.media_threads, config.decode_slots, config.footer_workers) == (4, 2, 3)


def test_run_config_bounds_the_generation_budget(tmp_path):
    """The online default matches Vertex Batch; the bound mirrors its CLI range."""
    from castle_pipeline.request_spec import DEFAULT_MAX_OUTPUT_TOKENS
    assert RunConfig(tmp_path, tmp_path).max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS == 32768
    for kwargs in ({'max_output_tokens': 1023}, {'max_output_tokens': 65537},
                   {'max_output_tokens': True}, {'max_output_tokens': 'x'}):
        with pytest.raises(ValueError):
            RunConfig(tmp_path, tmp_path, **kwargs)
    assert RunConfig(tmp_path, tmp_path, max_output_tokens=65536).max_output_tokens == 65536


def test_pipeline_builds_its_extractor_from_the_config(tmp_path):
    """The online path must honour the tuning knobs, not the old hardcoded 1/1."""
    config = RunConfig(tmp_path, tmp_path, media_threads=5, decode_slots=3, footer_workers=7)

    class Provider:
        model = 'offline-model'
        service_tier = 'standard'

    pipeline = Pipeline(config, Provider())
    assert pipeline.extractor.threads == 5
    assert pipeline.extractor.decode_slots == 3
    assert pipeline.extractor.footer_workers == 7
