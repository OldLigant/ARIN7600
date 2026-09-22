"""Canonical request construction shared by the online runner and Vertex Batch.

Both execution modes must issue semantically identical requests for the same
clip: identical prompts (base prompt files plus the suffixes below), identical
context JSON (same keys in the same insertion order, because the context is
serialized verbatim into the first text part), identical media labels in the
same part order (media first, then its text label), and the same stage response
schema. Prompt/context/media order preserves the shape of released batch runs
(v1-v6); structured output is a new request parameter for new runs. An online
run is a preview of a batch run, never a differently-worded variant.

Only the transport differs and stays in the callers: inline base64 versus GCS
``fileData``, plus the batch-only ``CASTLE_BATCH_ID`` correlation marker.
"""

# Thinking tokens share the generation budget with the visible answer, so a tight
# cap truncates the JSON mid-document on content-heavy clips. Measured on
# gemini-3.8-flash annotation: every MAX_TOKENS row landed at exactly 16384
# combined (5-11k of it thinking), losing the whole clip. Both 32768 and 65536
# are accepted by the Batch endpoint; batch stores this as a run parameter, and
# the online default matches it so smoke results predict batch behaviour.
DEFAULT_MAX_OUTPUT_TOKENS = 32768

AUDIO_SOURCE_ID = 'audio_0'
AUDIO_SOURCE_LABEL = 'source_id=audio_0'

# Stage prompt suffixes. This text is request identity: both modes append it
# verbatim, and released runs already sent exactly this wording.
AUDIO_INHERITANCE_NOTE = ('\nThe audio_annotation is inherited audio_0 evidence from a separate '
                          'raw-audio pass. No raw audio is supplied in this visual request. '
                          'Do not guess visual speaker identity from voice proximity.')
REVIEW_CONTRACT_HEADER = '\nFor this review use this replacement output contract instead:\n'
EXOCENTRIC_NOTE = '\nFIXED EXOCENTRIC camera: there is no wearer. Use person IDs, no I, and activity_chain=[].'

TIMESTAMP_FOOTER_NOTE = 'Synthetic clip-local seconds; exclude footer from scene text.'


def base_context(stage, clip_id, duration_sec, source, start_offset_sec):
    """The first five context keys, in the released insertion order."""
    return {'task_phase': stage, 'clip_id': clip_id, 'duration_sec': duration_sec,
            'source': source, 'source_start_offset_sec': start_offset_sec}


def audio_stage_context(context):
    context['audio_source_id'] = AUDIO_SOURCE_ID
    return context


def annotation_stage_context(context, frame_times_sec, audio_annotation, audio_checked):
    """Extend a fresh base context; key order is serialized request identity."""
    context.update(visual_source_id='video_0', visual_input='timestamped_frames',
                   frame_times_sec=frame_times_sec, audio_input='absent_in_this_request',
                   audio_annotation=audio_annotation,
                   audio_annotation_verification='audio_checked' if audio_checked else 'absent',
                   ocr_input='absent', eye_tracking='absent', trusted_identity_mapping='absent',
                   timestamp_footer=TIMESTAMP_FOOTER_NOTE)
    return context


def review_stage_context(context, annotation, crop_sources):
    """``crop_sources`` is the released list of {source_id, time_sec} objects."""
    context.update(annotation=annotation, audio_input='absent', crop_sources=crop_sources)
    return context


def annotation_prompt(prompts, *, review_enabled, max_review_regions, exocentric):
    """Base annotation prompt plus suffixes in the released order."""
    prompt = prompts['annotation'] + AUDIO_INHERITANCE_NOTE
    if review_enabled:
        prompt += '\n' + prompts['review_regions'].replace('MAX_REVIEW_REGIONS', str(max_review_regions))
    if exocentric:
        prompt += EXOCENTRIC_NOTE
    return prompt


def review_prompt(prompts):
    return prompts['annotation'] + REVIEW_CONTRACT_HEADER + prompts['review']


def frame_label(index, clip_sec):
    return f'frame_index={index}; clip_sec={clip_sec}; source_id=video_0'


def crop_label(source_id, clip_sec):
    return f'source_id={source_id}; clip_sec={clip_sec}'


# Vertex structured output accepts a subset of OpenAPI Schema, not the full
# JSON Schema used by our local validator. Keep this small enough for all three
# model stages; cross-field and evidence checks remain in schema.py.
def _object(properties, required=None):
    return {'type': 'OBJECT', 'properties': properties,
            'required': list(properties) if required is None else required}


def _array(item):
    return {'type': 'ARRAY', 'items': item}


def _enum(*values):
    return {'type': 'STRING', 'enum': list(values)}


_TEXT = {'type': 'STRING'}
_NUMBER = {'type': 'NUMBER'}
_STRINGS = _array(_TEXT)
_NULL_TEXT = {'type': 'STRING', 'nullable': True}
_ENVIRONMENT = _object({'world_changes': _STRINGS, 'newly_observed': _STRINGS})
_INITIAL_ENVIRONMENT = _object({'time_sec': _NUMBER, 'description': _TEXT})
_INITIAL_ENVIRONMENT['nullable'] = True
_SPEECH = _object({'speaker_id': _TEXT, 'source': _enum('in_person', 'media', 'unknown'),
                   'text': _NULL_TEXT, 'verification': _enum('audio_checked', 'transcript_only')})
_SPEECH['nullable'] = True
_EVIDENCE = _object({'source_id': _TEXT, 'modality': _enum('visual', 'audio', 'transcript', 'ocr'),
                     'start_sec': _NUMBER, 'end_sec': _NUMBER,
                     'support': _enum('direct', 'indirect'), 'note': _TEXT})
_BOUNDARY = _object({'start': _enum('observed', 'ongoing', 'uncertain'),
                     'end': _enum('observed', 'ongoing', 'uncertain'),
                     'precision': _enum('video_estimate', 'sampled_frames', 'audio_estimate', 'transcript_only')})
_SEGMENT = _object({
    'segment_id': _TEXT, 'start_sec': _NUMBER, 'end_sec': _NUMBER,
    'event_type': _enum('action', 'speech', 'observation', 'environment'),
    'actor_ids': _STRINGS, 'activity': _NULL_TEXT, 'action': _TEXT,
    'action_brief': _TEXT, 'objects': _STRINGS, 'environment': _ENVIRONMENT,
    'text_visible': _STRINGS, 'speech': _SPEECH, 'details': _NULL_TEXT,
    'evidence': _array(_EVIDENCE), 'boundary': _BOUNDARY, 'uncertainties': _STRINGS,
})
_AUDIO_SCHEMA = _object({
    'summary': _TEXT,
    'utterances': _array(_object({
        'start_sec': _NUMBER, 'end_sec': _NUMBER, 'speaker_id': _TEXT,
        'source': _enum('in_person', 'media', 'unknown'), 'text': _NULL_TEXT,
        'intelligibility': _enum('clear', 'partial', 'unintelligible')})),
    'sound_events': _array(_object({'start_sec': _NUMBER, 'end_sec': _NUMBER,
                                   'description': _TEXT, 'source': _TEXT})),
    'uncertainties': _STRINGS,
})
_ANNOTATION_SCHEMA = _object({
    'schema_version': _enum('castle-caption-v1'), 'scene_summary': _TEXT,
    'actors': _array(_object({'actor_id': _TEXT, 'description': _TEXT})),
    'initial_environment': _INITIAL_ENVIRONMENT,
    'visual_unavailable_intervals': _array(_object({'start_sec': _NUMBER, 'end_sec': _NUMBER,
                                                    'reason': _TEXT})),
    'segments': _array(_SEGMENT), 'activity_chain': _STRINGS,
    'review_regions': _array(_object({'frame_index': {'type': 'INTEGER'},
                                     'box_2d': _array(_NUMBER), 'label': _TEXT,
                                     'reason': _TEXT})),
}, required=['schema_version', 'scene_summary', 'actors', 'initial_environment',
             'visual_unavailable_intervals', 'segments', 'activity_chain'])
_REVIEW_SCHEMA = _object({
    'findings': _array(_object({'source_id': _TEXT, 'time_sec': _NUMBER,
                                'observation': _TEXT, 'readable_text': _STRINGS,
                                'limitations': _STRINGS})),
    'segment_replacements': _array(_object({'segment_id': _TEXT, 'reason': _TEXT,
                                            'replacement': _SEGMENT})),
    'initial_environment_replacement': _INITIAL_ENVIRONMENT,
    'scene_summary_replacement': _NULL_TEXT,
    'resegmentation_requests': _array(_object({'segment_ids': _STRINGS,
                                                'source_id': _TEXT, 'reason': _TEXT})),
})
_RESPONSE_SCHEMAS = {'audio': _AUDIO_SCHEMA, 'annotation': _ANNOTATION_SCHEMA,
                     'review': _REVIEW_SCHEMA}


def response_schema_for_stage(stage):
    """Return the Vertex request schema for a production stage."""
    try:
        return _RESPONSE_SCHEMAS[stage]
    except KeyError:
        raise ValueError(f'Unknown response schema stage: {stage}') from None
