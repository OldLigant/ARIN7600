"""Real media/checkpoints; only the paid remote model boundary is replaced."""
import copy
import json
import shutil
import subprocess
import threading
from pathlib import Path
import pytest
from castle_pipeline.runner import Pipeline, RunConfig
from castle_pipeline.vertex import ProviderError


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=320x180:r=50',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000', '-t', '20',
                    '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(path)], check=True)
    return path


class ModelBoundary:
    model = 'offline-fixture'
    def __init__(self, fail_review=False):
        self.calls = []
        self.fail_review = fail_review

    def generate(self, prompt, context, images, audio=None, max_output_tokens=16384):
        ctx = json.loads(context)
        phase = ctx['task_phase']
        self.calls.append(phase)
        if phase == 'audio':
            assert audio.is_file() and not images
            data = {'summary': 'Tone.', 'utterances': [], 'sound_events': [], 'uncertainties': []}
        elif phase == 'annotation':
            assert len(images) == 20
            assert ctx['frame_times_sec'] == list(range(20))
            assert ctx['audio_annotation']['summary'] == 'Tone.'
            path = Path(__file__).resolve().parents[2] / 'annotation_design_v1' / '输出示例_假设场景.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            data['review_regions'] = [{'frame_index': 0, 'box_2d': [0, 0, 500, 500],
                                       'label': 'fixture crop', 'reason': 'exercise native crop path'}]
        else:
            assert phase == 'review' and len(images) == 1
            if self.fail_review:
                raise ProviderError('fixture quota', category='quota', fatal=False)
            data = {'findings': [], 'segment_replacements': [], 'initial_environment_replacement': None,
                    'scene_summary_replacement': None, 'resegmentation_requests': []}
        return {'data': data, 'usage': {'input_tokens': 10, 'output_tokens': 20}, 'attempts': 1}


def test_successful_phases_survive_failed_review_and_resume(source, tmp_path):
    config = RunConfig(output_dir=tmp_path/'out', scratch_dir=tmp_path/'scratch', workers=1, review=True)
    provider = ModelBoundary(fail_review=True)
    pipeline = Pipeline(config, provider)
    first = pipeline.run_source(source, {'source_id': 'fixture-hour', 'day': 'day1', 'stream': 'Allie', 'hour': 8})
    assert first['failed'] == 1 and first['completed'] == 0
    assert provider.calls == ['audio', 'annotation', 'review']
    provider.fail_review = False
    second = pipeline.run_source(source, {'source_id': 'fixture-hour', 'day': 'day1', 'stream': 'Allie', 'hour': 8})
    assert second['failed'] == 0 and second['completed'] == 1
    assert provider.calls == ['audio', 'annotation', 'review', 'review']
    third = pipeline.run_source(source, {'source_id': 'fixture-hour', 'day': 'day1', 'stream': 'Allie', 'hour': 8})
    assert third['reused'] == 1
    assert len(provider.calls) == 4
    final = json.loads(next((tmp_path/'out').rglob('final.json')).read_text(encoding='utf-8'))
    assert final['annotation']['schema_version'] == 'castle-caption-v1'
    assert 'review_regions' not in final['annotation']
    assert final['source']['stream'] == 'Allie'
    assert final['clip']['start_offset_sec'] == 0
    assert not list((tmp_path/'scratch').rglob('*.jpg'))


def test_model_change_cannot_reuse_previous_annotations(source, tmp_path):
    config = RunConfig(output_dir=tmp_path/'out', scratch_dir=tmp_path/'scratch', workers=1, review=False)
    provider = ModelBoundary()
    pipeline = Pipeline(config, provider)
    metadata = {'source_id': 'fixture-hour', 'day': 'day1', 'stream': 'Allie', 'hour': 8}
    pipeline.run_source(source, metadata)
    provider.model = 'different-model'
    Pipeline(config, provider).run_source(source, metadata)
    assert provider.calls == ['audio', 'annotation', 'audio', 'annotation']


def test_progress_reports_blocked_audio_and_stage_logs_survive_resume(source, tmp_path):
    audio_progress = threading.Event()
    records = []
    def sink(line):
        row = json.loads(line)
        records.append(row)
        if row['event'] == 'progress' and any(s['phase'] == 'audio' for s in row['telemetry']['active_stages']):
            audio_progress.set()
    class DelayedModel(ModelBoundary):
        def generate(self, prompt, context, images, audio=None, max_output_tokens=16384):
            if json.loads(context)['task_phase'] == 'audio':
                assert audio_progress.wait(3), 'Heartbeat must run before the blocked audio call returns'
            return super().generate(prompt, context, images, audio, max_output_tokens)
    cfg = RunConfig(tmp_path/'out', tmp_path/'scratch', workers=1, review=False, progress_interval_sec=.02)
    pipeline = Pipeline(cfg, DelayedModel(), log=sink)
    meta = {'source_id': 'fixture', 'day': 'day1', 'stream': 'Allie', 'hour': 8}
    result = pipeline.run_source(source, meta)
    assert result['completed'] == 1 and audio_progress.is_set()
    assert [(r['phase']) for r in records if r['event'] == 'stage_start'] == ['prepare', 'audio', 'annotation']
    assert records[-1]['event'] == 'source_completed'
    log_file = next((tmp_path/'out').rglob('events.jsonl'))
    assert len(log_file.read_text(encoding='utf-8').splitlines()) == len(records)
    pipeline.run_source(source, meta)
    assert any(r['event'] == 'clip_reused' for r in records)
    assert pipeline.events.snapshot()['active_stages'] == []


def test_normalized_response_keeps_raw_evidence_and_flags_boundary_review(source, tmp_path):
    class UnorderedModel(ModelBoundary):
        def generate(self, prompt, context, images, audio=None, max_output_tokens=16384):
            result = super().generate(prompt, context, images, audio, max_output_tokens)
            if json.loads(context)['task_phase'] == 'annotation':
                result['data']['segments'].reverse()
                result['data']['segments'][0]['boundary']['start'] = 'ongoing'
            return result
    cfg = RunConfig(tmp_path/'out', tmp_path/'scratch', workers=1, review=False)
    summary = Pipeline(cfg, UnorderedModel(), log=lambda _: None).run_source(source, {'source_id': 'fixture'})
    assert summary['completed'] == 1 and summary['review_required'] == 1
    checkpoint = json.loads(next((tmp_path/'out').rglob('annotation.json')).read_text(encoding='utf-8'))['result']
    assert checkpoint['raw_data']['segments'][0]['segment_id'] == 's006'
    assert checkpoint['raw_data']['segments'][0]['boundary']['start'] == 'ongoing'
    assert checkpoint['data']['segments'][0]['segment_id'] == 's001'
    normalized = next(s for s in checkpoint['data']['segments'] if s['segment_id'] == 's006')
    assert normalized['start_sec'] == 12 and normalized['boundary']['start'] == 'uncertain'
    assert {c['code'] for c in checkpoint['normalization']} == {'SORT_SEGMENTS', 'NONEDGE_ONGOING_TO_UNCERTAIN'}
