You are reviewing an existing CASTLE dense annotation against additional evidence. You receive the original input context, its annotation JSON, and explicitly identified crops, short video excerpts, or audio excerpts with source IDs and clip-local timestamps.

Use the same grounding, actor attribution, language, time, and output-field rules as the primary annotation prompt. The additional evidence may show text, a hand-object interaction, a tool, an appliance control, a state change, or an audio detail. Your task is to correct or enrich supported facts, not to make every description longer.

1. Inspect each provided item. Report readable text or concrete visible/audible facts. If it is unreadable, obstructed, or ambiguous, state that limitation. Do not complete text from context.
2. Determine which existing segment IDs the evidence supports. A single image proves a state at its timestamp, not the direction of a movement or an unseen transition. A visual crop cannot establish spoken words. If you receive only images, preserve speech unless marking an unsupported visual speaker match as uncertain.
3. Where a correction is justified, return a COMPLETE replacement segment with the same segment_id, start_sec, and end_sec. Reconcile action, action_brief, activity, objects, environment, details, and evidence together. Preserve unaffected fields, including their null or empty values. Empty values are valid corrections and must not be discarded.
4. Do not add/remove segments or change their time ranges in this detail review. If new temporal evidence requires splitting, merging, or shifting boundaries, report a resegmentation_request referencing the affected IDs, evidence source, and reason. This request requires review with full temporal context; it is not an automatic timing correction.
5. Correct initial_environment only with evidence belonging to its establishing observation; later evidence must not be moved backward in time. Preserve the distinction between world_changes and newly_observed. Correct the scene_summary if a corrected fact invalidates it.
6. Return changed segments only. Use explicit replacement objects and IDs, not positional matching. Never use one segment's correction for another merely because their timestamps happen to match.
7. Do not change actor IDs, event_type, or actor_ids in this detail-only pass. If attribution or event classification needs correction, issue a resegmentation_request for contextual review rather than silently changing activity_chain or the actor registry. The term resegmentation_request includes such attribution review.

Return only a JSON object with these fields:

- findings: an array of {source_id: string, time_sec: number, observation: string, readable_text: string[], limitations: string[]}.
- segment_replacements: an array of {segment_id: string, reason: string, replacement: object}. Each replacement is a complete segment object following the primary output contract, never a string or a partial patch.
- initial_environment_replacement: null, or the complete {time_sec: number, description: string} object.
- scene_summary_replacement: null, or a string.
- resegmentation_requests: an array of {segment_ids: string[], source_id: string, reason: string}.

For no findings or changes, the exact valid output is:
{
  "findings": [],
  "segment_replacements": [],
  "initial_environment_replacement": null,
  "scene_summary_replacement": null,
  "resegmentation_requests": []
}

A null top-level replacement means leave the original field unchanged. Explicit null or [] values INSIDE a complete segment replacement must be applied as written.
