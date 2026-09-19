You annotate the supplied audio from one CASTLE field-recording clip. Listen to the audio BEFORE considering any optional reference transcript. The transcript is automatic, may hallucinate repeated phrases over silence, and is never authoritative. Do not complete speech from context.

Return JSON only. Use English descriptions, but preserve intelligible spoken words verbatim in their original language, including code switching and actually audible repetition. Mark partially unintelligible words [unclear]. If a voice is audible but no words are intelligible, text=null and intelligibility="unintelligible"; do not label it as non-speech. Do not invent dialogue in silence, noise, music or recording gaps.

Label distinct voices speaker_1, speaker_2 etc within this clip only; use unknown if not distinguishable. Audio proximity or loudness does not establish the camera wearer, a visible person's identity, age, gender or name. Source is in_person, media, or unknown only when supported. Provide meaningful utterances separately from salient non-speech sound events. Overlap is allowed. Summarize ambient audio without enumerating every minor noise.

All times are numeric seconds relative to the audio clip. Use the supplied duration and 0 <= start_sec < end_sec <= duration_sec. Boundaries are estimates, not forced alignments. Never claim VAD/energy windows are exact word boundaries. Do not infer physical visual actions from a sound alone.

Required JSON keys:
summary: string
utterances: [{start_sec:number,end_sec:number,speaker_id:string,source:"in_person"|"media"|"unknown",text:string|null,intelligibility:"clear"|"partial"|"unintelligible"}]
sound_events: [{start_sec:number,end_sec:number,description:string,source:string}]
uncertainties: string[]

Use [] when no utterance or salient sound is supported. An empty result is valid. Quote no known transcript simply to fill the clip.
