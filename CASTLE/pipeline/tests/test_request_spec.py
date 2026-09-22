"""The canonical request shape is pinned here.

Released batch runs (v1-v6) produced exactly these bytes; the pin fails if the
shared builders drift, and it is the reference the online runner must mirror
(see test_pipeline_integration.test_online_requests_mirror_the_canonical_request_spec).
Synthetic prompts keep the pin readable; the real prompt files are hashed into
run identity separately.
"""
import json
from pathlib import Path
import pytest

from castle_pipeline import request_spec
from castle_pipeline.batch_tasks import TaskPlanner, DEFAULT_MAX_OUTPUT_TOKENS, request_row


PROMPTS = {'audio': 'AUDIO_PROMPT_TEXT', 'annotation': 'ANNOTATION_PROMPT_TEXT',
           'review': 'REVIEW_PROMPT_TEXT', 'review_regions': 'REGIONS_PROMPT MAX_REVIEW_REGIONS'}


class CloudFiles:
    def __init__(self):
        self.files = {}

    def upload(self, path, uri):
        self.files[uri] = Path(path).read_bytes()

    def download(self, uri, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(self.files[uri])

    def write_jsonl(self, uri, rows):
        self.files[uri] = ''.join(json.dumps(r) + '\n' for r in rows).encode()


def _meta():
    return {'clip_id': 'c1', 'duration_sec': 30, 'source': {'viewpoint': 'egocentric'},
            'start_offset_sec': 90, 'frame_times_sec': [0.0, 1.5],
            'frames': ['gs://b/f0.jpg', 'gs://b/f1.jpg'], 'native_frames': [],
            'audio_uri': 'gs://b/a.wav'}


def _prepare(tmp_path, stage, item, cfg):
    cloud = CloudFiles()
    planner = TaskPlanner(cloud, tmp_path / 's', tmp_path / 'o')
    cloud.files['gs://b/meta.json'] = json.dumps(_meta()).encode()
    cloud.files['gs://b/audio.json'] = json.dumps({'data': {'summary': 'tone'}, 'usage': {}}).encode()
    cloud.files['gs://b/annotation.json'] = json.dumps({'data': {'scene_summary': 's'}, 'usage': {}}).encode()
    prepared = planner.prepare(stage, [item], {'config': cfg})
    lines = [json.loads(line) for line in cloud.files[prepared['input_uri']].decode().splitlines()]
    return prepared, lines


CFG = {'gcs_prefix': 'gs://b/run', 'prompts': PROMPTS, 'review': True,
       'max_review_regions': 4, 'max_output_tokens': 32768}


def test_annotation_request_is_pinned_byte_for_byte(tmp_path):
    item = {'clip_id': 'c1', 'metadata_uri': 'gs://b/meta.json', 'audio_uri': 'gs://b/audio.json'}
    prepared, lines = _prepare(tmp_path, 'annotation', item, CFG)
    assert prepared['request_ids'] == ['annotation:c1'] and len(lines) == 1
    row = lines[0]
    assert row['request']['systemInstruction']['parts'][0]['text'] == (
        'ANNOTATION_PROMPT_TEXT'
        '\nThe audio_annotation is inherited audio_0 evidence from a separate raw-audio pass. '
        'No raw audio is supplied in this visual request. Do not guess visual speaker identity from voice proximity.'
        '\nREGIONS_PROMPT 4')
    parts = row['request']['contents'][0]['parts']
    assert parts[0]['text'] == 'CASTLE_BATCH_ID:annotation:c1\n' + json.dumps({
        'task_phase': 'annotation', 'clip_id': 'c1', 'duration_sec': 30,
        'source': {'viewpoint': 'egocentric'}, 'source_start_offset_sec': 90,
        'visual_source_id': 'video_0', 'visual_input': 'timestamped_frames',
        'frame_times_sec': [0.0, 1.5], 'audio_input': 'absent_in_this_request',
        'audio_annotation': {'summary': 'tone'}, 'audio_annotation_verification': 'audio_checked',
        'ocr_input': 'absent', 'eye_tracking': 'absent', 'trusted_identity_mapping': 'absent',
        'timestamp_footer': 'Synthetic clip-local seconds; exclude footer from scene text.'}, ensure_ascii=False)
    assert [p.get('text') or p['fileData']['fileUri'] for p in parts[1:]] == [
        'gs://b/f0.jpg', 'frame_index=0; clip_sec=0.0; source_id=video_0',
        'gs://b/f1.jpg', 'frame_index=1; clip_sec=1.5; source_id=video_0']
    assert row['request']['generationConfig'] == {'responseMimeType': 'application/json', 'maxOutputTokens': 32768}


def test_audio_and_review_requests_are_pinned(tmp_path):
    audio_item = {'clip_id': 'c1', 'metadata_uri': 'gs://b/meta.json'}
    _, audio_rows = _prepare(tmp_path, 'audio', audio_item, CFG)
    audio = audio_rows[0]['request']
    assert audio['systemInstruction']['parts'][0]['text'] == 'AUDIO_PROMPT_TEXT'
    assert audio['contents'][0]['parts'] == [
        {'text': 'CASTLE_BATCH_ID:audio:c1\n' + json.dumps(
            {'task_phase': 'audio', 'clip_id': 'c1', 'duration_sec': 30,
             'source': {'viewpoint': 'egocentric'}, 'source_start_offset_sec': 90,
             'audio_source_id': 'audio_0'}, ensure_ascii=False)},
        {'fileData': {'fileUri': 'gs://b/a.wav', 'mimeType': 'audio/wav'}},
        {'text': 'source_id=audio_0'}]

    review_item = {'clip_id': 'c1', 'metadata_uri': 'gs://b/meta.json', 'audio_uri': 'gs://b/audio.json',
                   'annotation_uri': 'gs://b/annotation.json',
                   'crops': [{'source_id': 'crop_0', 'time_sec': 1.5, 'uri': 'gs://b/crop-0.jpg'}]}
    _, review_rows = _prepare(tmp_path, 'review', review_item, CFG)
    review = review_rows[0]['request']
    assert review['systemInstruction']['parts'][0]['text'] == (
        'ANNOTATION_PROMPT_TEXT\nFor this review use this replacement output contract instead:\n'
        'REVIEW_PROMPT_TEXT')
    assert review['contents'][0]['parts'] == [
        {'text': 'CASTLE_BATCH_ID:review:c1\n' + json.dumps(
            {'task_phase': 'review', 'clip_id': 'c1', 'duration_sec': 30,
             'source': {'viewpoint': 'egocentric'}, 'source_start_offset_sec': 90,
             'annotation': {'scene_summary': 's'}, 'audio_input': 'absent',
             'crop_sources': [{'source_id': 'crop_0', 'time_sec': 1.5}]}, ensure_ascii=False)},
        {'fileData': {'fileUri': 'gs://b/crop-0.jpg', 'mimeType': 'image/jpeg'}},
        {'text': 'source_id=crop_0; clip_sec=1.5'}]


def test_spec_constants_and_key_order_are_identity():
    assert DEFAULT_MAX_OUTPUT_TOKENS == request_spec.DEFAULT_MAX_OUTPUT_TOKENS == 32768
    assert list(request_spec.base_context('audio', 'c', 30, {}, 0)) == [
        'task_phase', 'clip_id', 'duration_sec', 'source', 'source_start_offset_sec']
    annotation = request_spec.annotation_stage_context(
        request_spec.base_context('annotation', 'c', 30, {}, 0), [0.0], {}, True)
    assert list(annotation) == ['task_phase', 'clip_id', 'duration_sec', 'source',
                                'source_start_offset_sec', 'visual_source_id', 'visual_input',
                                'frame_times_sec', 'audio_input', 'audio_annotation',
                                'audio_annotation_verification', 'ocr_input', 'eye_tracking',
                                'trusted_identity_mapping', 'timestamp_footer']
    review = request_spec.review_stage_context(request_spec.base_context('review', 'c', 30, {}, 0),
                                               {}, [])
    assert list(review) == ['task_phase', 'clip_id', 'duration_sec', 'source',
                            'source_start_offset_sec', 'annotation', 'audio_input', 'crop_sources']


def test_annotation_prompt_appends_regions_before_exocentric():
    prompt = request_spec.annotation_prompt(PROMPTS, review_enabled=True,
                                            max_review_regions=2, exocentric=True)
    assert prompt == ('ANNOTATION_PROMPT_TEXT' + request_spec.AUDIO_INHERITANCE_NOTE
                      + '\nREGIONS_PROMPT 2' + request_spec.EXOCENTRIC_NOTE)
    assert request_spec.annotation_prompt(PROMPTS, review_enabled=False,
                                          max_review_regions=4, exocentric=False) == \
        'ANNOTATION_PROMPT_TEXT' + request_spec.AUDIO_INHERITANCE_NOTE
