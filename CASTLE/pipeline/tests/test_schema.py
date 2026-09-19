import copy
import json
from pathlib import Path
import pytest

from castle_pipeline.schema import validate_annotation, validate_audio, apply_review
from castle_pipeline.schema import normalize_annotation


@pytest.fixture
def annotation():
    path = Path(__file__).resolve().parents[2] / 'annotation_design_v1' / '输出示例_假设场景.json'
    return json.loads(path.read_text(encoding='utf-8'))


def test_rejects_out_of_clip_evidence_and_unknown_actor(annotation):
    validate_annotation(annotation, 20, {'video_0', 'audio_0'})
    annotation['segments'][0]['evidence'][0]['end_sec'] = 21
    with pytest.raises(ValueError):
        validate_annotation(annotation, 20, {'video_0', 'audio_0'})
    annotation['segments'][0]['evidence'][0]['end_sec'] = 4
    annotation['segments'][0]['actor_ids'] = ['person_99']
    with pytest.raises(ValueError):
        validate_annotation(annotation, 20, {'video_0', 'audio_0'})


def test_visual_only_call_cannot_claim_audio_checked_speech(annotation):
    with pytest.raises(ValueError):
        validate_annotation(annotation, 20, {'video_0'})


@pytest.mark.parametrize('source,modality', [('audio_0', 'visual'), ('video_0', 'transcript'), ('video_0', 'ocr')])
def test_source_modality_cannot_be_fabricated(annotation, source, modality):
    annotation['segments'][0]['evidence'][0].update(source_id=source, modality=modality)
    with pytest.raises(ValueError):
        validate_annotation(annotation, 20, {'video_0', 'audio_0'})


def test_transcript_only_speech_requires_supplied_transcript(annotation):
    annotation['segments'][2]['speech']['verification'] = 'transcript_only'
    with pytest.raises(ValueError):
        validate_annotation(annotation, 20, {'video_0', 'audio_0'})


def test_review_matches_ids_and_applies_empty_corrections(annotation):
    replacement = copy.deepcopy(annotation['segments'][0])
    replacement['activity'] = None
    replacement['details'] = None
    replacement['objects'] = []
    review = {'findings': [], 'segment_replacements': [
        {'segment_id': 's001', 'reason': 'The evidence does not establish the activity.', 'replacement': replacement}],
        'initial_environment_replacement': None, 'scene_summary_replacement': None,
        'resegmentation_requests': []}
    revised = apply_review(annotation, review, 20, {'video_0', 'audio_0'})
    assert revised['segments'][0]['activity'] is None
    assert revised['segments'][0]['objects'] == []
    assert annotation['segments'][0]['activity'] == 'Preparing a drink'
    review['segment_replacements'][0]['replacement']['start_sec'] = 1
    with pytest.raises(ValueError):
        apply_review(annotation, review, 20, {'video_0', 'audio_0'})


def test_audio_distinguishes_unintelligible_speech_from_non_speech():
    payload = {'summary': 'A voice and a clatter.', 'utterances': [
        {'start_sec': 1, 'end_sec': 2, 'speaker_id': 'speaker_1', 'source': 'unknown',
         'text': None, 'intelligibility': 'unintelligible'}],
        'sound_events': [{'start_sec': 3, 'end_sec': 4, 'description': 'A clatter.', 'source': 'unknown'}],
        'uncertainties': []}
    validate_audio(payload, 5)
    payload['utterances'][0]['end_sec'] = None
    with pytest.raises(ValueError):
        validate_audio(payload, 5)


def test_normalization_sorts_overlaps_without_mutating_raw(annotation):
    raw = copy.deepcopy(annotation)
    raw['segments'][0], raw['segments'][1] = raw['segments'][1], raw['segments'][0]
    raw['activity_chain'] = ['s002', 's001', 's005']
    fixed, changes = normalize_annotation(raw, 20, {'video_0', 'audio_0'})
    assert fixed['segments'] == annotation['segments']
    assert fixed['activity_chain'] == annotation['activity_chain']
    assert raw['segments'][0]['segment_id'] == 's002'
    assert {c['code'] for c in changes} == {'SORT_SEGMENTS', 'SORT_ACTIVITY_CHAIN'}


def test_nonedge_ongoing_becomes_uncertain_not_extended(annotation):
    annotation['segments'][0]['boundary']['start'] = 'ongoing'
    annotation['segments'][0]['boundary']['end'] = 'ongoing'
    fixed, changes = normalize_annotation(annotation, 20, {'video_0', 'audio_0'})
    segment = fixed['segments'][0]
    assert (segment['start_sec'], segment['end_sec']) == (2, 4)
    assert segment['boundary']['start'] == segment['boundary']['end'] == 'uncertain'
    assert segment['uncertainties']
    assert all(c['code'] == 'NONEDGE_ONGOING_TO_UNCERTAIN' for c in changes)
    assert annotation['segments'][0]['boundary']['start'] == 'ongoing'


def test_normalization_does_not_fabricate_times_evidence_or_references(annotation):
    for mutation in ('time', 'evidence', 'actor', 'chain'):
        data = copy.deepcopy(annotation)
        if mutation == 'time': data['segments'][0]['end_sec'] = 21
        if mutation == 'evidence': data['segments'][0]['evidence'] = []
        if mutation == 'actor': data['segments'][0]['actor_ids'] = ['invented_person']
        if mutation == 'chain': data['activity_chain'].append('invented_segment')
        with pytest.raises(ValueError):
            normalize_annotation(data, 20, {'video_0', 'audio_0'})
