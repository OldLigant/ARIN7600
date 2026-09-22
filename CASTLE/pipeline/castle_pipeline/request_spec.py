"""Canonical request construction shared by the online runner and Vertex Batch.

Both execution modes must issue semantically identical requests for the same
clip: identical prompts (base prompt files plus the suffixes below), identical
context JSON (same keys in the same insertion order, because the context is
serialized verbatim into the first text part), and identical media labels in
the same part order (media first, then its text label). The canonical shape is
the one the released batch runs (v1-v6) already produced, so those requests
stay byte-identical and the existing corpus remains the reference. An online
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
