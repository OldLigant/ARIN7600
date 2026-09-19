"""Local contracts: API JSON syntax is insufficient for safe checkpoint reuse."""
import copy
import math
from jsonschema import Draft202012Validator


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def array(item):
    return {'type': 'array', 'items': item}


def enum(*values):
    return {'enum': list(values)}


TEXT = {'type': 'string'}
NUMBER = {'type': 'number'}
STRINGS = array(TEXT)
NULL_TEXT = {'type': ['string', 'null']}
ENV = obj({'world_changes': STRINGS, 'newly_observed': STRINGS})
INITIAL = obj({'time_sec': NUMBER, 'description': TEXT})
SPEECH = obj({'speaker_id': TEXT, 'source': enum('in_person', 'media', 'unknown'),
              'text': NULL_TEXT, 'verification': enum('audio_checked', 'transcript_only')})
EVIDENCE = obj({'source_id': TEXT, 'modality': enum('visual', 'audio', 'transcript', 'ocr'),
                'start_sec': NUMBER, 'end_sec': NUMBER, 'support': enum('direct', 'indirect'), 'note': TEXT})
BOUNDARY = obj({'start': enum('observed', 'ongoing', 'uncertain'),
                'end': enum('observed', 'ongoing', 'uncertain'),
                'precision': enum('video_estimate', 'sampled_frames', 'audio_estimate', 'transcript_only')})
SEGMENT = obj({'segment_id': TEXT, 'start_sec': NUMBER, 'end_sec': NUMBER,
               'event_type': enum('action', 'speech', 'observation', 'environment'),
               'actor_ids': STRINGS, 'activity': NULL_TEXT, 'action': TEXT, 'action_brief': TEXT,
               'objects': STRINGS, 'environment': ENV, 'text_visible': STRINGS,
               'speech': {'anyOf': [SPEECH, {'type': 'null'}]}, 'details': NULL_TEXT,
               'evidence': {**array(EVIDENCE), 'minItems': 1}, 'boundary': BOUNDARY, 'uncertainties': STRINGS})
ANNOTATION = obj({'schema_version': enum('castle-caption-v1'), 'scene_summary': TEXT,
                  'actors': array(obj({'actor_id': TEXT, 'description': TEXT})),
                  'initial_environment': {'anyOf': [INITIAL, {'type': 'null'}]},
                  'visual_unavailable_intervals': array(obj({'start_sec': NUMBER, 'end_sec': NUMBER, 'reason': TEXT})),
                  'segments': array(SEGMENT), 'activity_chain': STRINGS})
AUDIO = obj({'summary': TEXT,
             'utterances': array(obj({'start_sec': NUMBER, 'end_sec': NUMBER, 'speaker_id': TEXT,
                                     'source': enum('in_person', 'media', 'unknown'), 'text': NULL_TEXT,
                                     'intelligibility': enum('clear', 'partial', 'unintelligible')})),
             'sound_events': array(obj({'start_sec': NUMBER, 'end_sec': NUMBER, 'description': TEXT, 'source': TEXT})),
             'uncertainties': STRINGS})
REVIEW = obj({'findings': array(obj({'source_id': TEXT, 'time_sec': NUMBER, 'observation': TEXT,
                                   'readable_text': STRINGS, 'limitations': STRINGS})),
              'segment_replacements': array(obj({'segment_id': TEXT, 'reason': TEXT, 'replacement': SEGMENT})),
              'initial_environment_replacement': {'anyOf': [INITIAL, {'type': 'null'}]},
              'scene_summary_replacement': NULL_TEXT,
              'resegmentation_requests': array(obj({'segment_ids': STRINGS, 'source_id': TEXT, 'reason': TEXT}))})


def check_schema(data, schema):
    errors = list(Draft202012Validator(schema).iter_errors(data))
    if errors:
        error = errors[0]
        # Do not log model text or transcript contents in validation errors.
        raise ValueError(f'JSON contract violation at {list(error.absolute_path)} ({error.validator})')


def timestamp(value, duration):
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= duration:
        raise ValueError('Timestamp must be finite and inside the clip')


def interval(item, duration, point=False):
    start, end = item['start_sec'], item['end_sec']
    timestamp(start, duration)
    timestamp(end, duration)
    if end < start or (not point and end == start):
        raise ValueError('Invalid time interval')


def validate_audio(data, duration):
    check_schema(data, AUDIO)
    for item in data['utterances'] + data['sound_events']:
        interval(item, duration)
    for item in data['utterances']:
        if item['intelligibility'] == 'clear' and not item['text']:
            raise ValueError('Clear speech requires words')
    return data


def validate_annotation(data, duration, sources):
    check_schema(data, ANNOTATION)
    actor_list = [item['actor_id'] for item in data['actors']]
    actors = set(actor_list) | {'unknown'}
    if len(set(actor_list)) != len(actor_list):
        raise ValueError('Duplicate actor ID')
    segments = data['segments']
    ids = [s['segment_id'] for s in segments]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate segment ID')
    if segments != sorted(segments, key=lambda s: (s['start_sec'], s['segment_id'])):
        raise ValueError('Segments must be sorted by start_sec and ID')
    if data['initial_environment'] is not None:
        timestamp(data['initial_environment']['time_sec'], duration)
    for item in data['visual_unavailable_intervals']:
        interval(item, duration)
    for seg in segments:
        interval(seg, duration)
        if not set(seg['actor_ids']) <= actors:
            raise ValueError('Unknown actor reference')
        if seg['boundary']['start'] == 'ongoing' and seg['start_sec'] != 0:
            raise ValueError('Ongoing start must touch clip start')
        if seg['boundary']['end'] == 'ongoing' and abs(seg['end_sec'] - duration) > .001:
            raise ValueError('Ongoing end must touch clip end')
        for evidence in seg['evidence']:
            interval(evidence, duration, point=True)
            if evidence['source_id'] not in sources:
                raise ValueError('Unavailable evidence source')
            sid = evidence['source_id']
            expected_modality = ('visual' if sid == 'video_0' or sid.startswith('crop_') else
                                 'audio' if sid == 'audio_0' else
                                 'transcript' if sid.startswith('transcript_') else
                                 'ocr' if sid.startswith('ocr_') else None)
            if expected_modality != evidence['modality']:
                raise ValueError('Evidence source/modality mismatch')
            if evidence['modality'] == 'audio' and 'audio_0' not in sources:
                raise ValueError('No audio evidence supplied')
        speech = seg['speech']
        if seg['event_type'] == 'speech':
            if speech is None or speech['speaker_id'] not in actors:
                raise ValueError('Speech event needs a registered or unknown speaker')
            if speech['verification'] == 'audio_checked' and 'audio_0' not in sources:
                raise ValueError('Audio-checked speech requires an audio source')
            if speech['verification'] == 'audio_checked' and not any(e['source_id'] == 'audio_0' and e['modality'] == 'audio' for e in seg['evidence']):
                raise ValueError('Audio-checked speech must cite the audio evidence')
            if speech['verification'] == 'transcript_only' and not any(s.startswith('transcript_') for s in sources):
                raise ValueError('No unverified transcript input was supplied')
        elif speech is not None:
            raise ValueError('Speech must be its own event')
    expected = [s['segment_id'] for s in segments if s['event_type'] == 'action' and 'wearer' in s['actor_ids']]
    if data['activity_chain'] != expected:
        raise ValueError('Activity chain must reference wearer action segments in order')
    return data


def normalize_annotation(data, duration, sources):
    """Canonicalize presentation, never invent evidence or move event times.

    Keep validate_annotation strict for saved canonical outputs. A model's
    non-edge 'ongoing' is an uncertain boundary, not evidence of clip coverage.
    """
    check_schema(data, ANNOTATION)
    revised = copy.deepcopy(data)
    changes = []
    segments = revised['segments']
    ordered = sorted(segments, key=lambda s: (s['start_sec'], s['segment_id']))
    if ordered != segments:
        changes.append({'code': 'SORT_SEGMENTS', 'before': [s['segment_id'] for s in segments],
                        'after': [s['segment_id'] for s in ordered]})
        revised['segments'] = ordered
    expected = [s['segment_id'] for s in ordered if s['event_type'] == 'action' and 'wearer' in s['actor_ids']]
    chain = revised['activity_chain']
    # Reorder existing references only; missing/extra/duplicate references remain errors.
    if chain != expected and len(chain) == len(expected) and set(chain) == set(expected):
        changes.append({'code': 'SORT_ACTIVITY_CHAIN', 'before': chain[:], 'after': expected[:]})
        revised['activity_chain'] = expected
    for seg in ordered:
        for edge, target in [('start', 0), ('end', duration)]:
            value = seg[edge + '_sec']
            nonedge = value != target if edge == 'start' else abs(value - target) > .001
            if seg['boundary'][edge] == 'ongoing' and nonedge:
                seg['boundary'][edge] = 'uncertain'
                note = f'Model marked {edge} as ongoing at {value}s, away from the clip edge; the boundary is uncertain and its timestamp was not changed.'
                if note not in seg['uncertainties']:
                    seg['uncertainties'].append(note)
                changes.append({'code': 'NONEDGE_ONGOING_TO_UNCERTAIN', 'segment_id': seg['segment_id'],
                                'edge': edge, 'time_sec': value, 'before': 'ongoing', 'after': 'uncertain'})
    validate_annotation(revised, duration, sources)
    return revised, changes


def apply_review(annotation, review, duration, sources):
    check_schema(review, REVIEW)
    revised = copy.deepcopy(annotation)
    by_id = {s['segment_id']: s for s in revised['segments']}
    seen = set()
    for item in review['segment_replacements']:
        sid, replacement = item['segment_id'], item['replacement']
        if sid not in by_id or sid in seen:
            raise ValueError('Unknown or duplicated replacement ID')
        seen.add(sid)
        for key in ('segment_id', 'start_sec', 'end_sec', 'actor_ids', 'event_type'):
            if replacement[key] != by_id[sid][key]:
                raise ValueError('Detail review cannot change identity, actors, event type or timing')
        by_id[sid] = copy.deepcopy(replacement)
    revised['segments'] = [by_id[s['segment_id']] for s in revised['segments']]
    for item in review['findings']:
        if item['source_id'] not in sources:
            raise ValueError('Unknown finding source')
        timestamp(item['time_sec'], duration)
    for item in review['resegmentation_requests']:
        if not set(item['segment_ids']) <= set(by_id) or item['source_id'] not in sources:
            raise ValueError('Unknown contextual review reference')
    for field in ('initial_environment', 'scene_summary'):
        value = review[field + '_replacement']
        if value is not None:
            revised[field] = value
    return validate_annotation(revised, duration, sources)
