# CASTLE annotation pipeline implementation plan

Historical implementation plan (2026-09-16), not the current run status or authorization. Later operation instructions supersede this turn's original scope. See README.md, STATUS.md and docs/version-evolution-decisions.md for current interfaces, timestamped state and deferred work.

User-approved scope: implement the preceding caption design with 1 fps / 30-second clips, audio first, Vertex AI, AIMD, bounded CPU memory, HF Jobs execution, documentation and an agent skill. No paid job or model call is requested in this development turn.

Architecture: one source-hour file on disk at a time, bounded clip workers, one FFmpeg decoder by default; per-clip audio -> image annotation -> optional detail review. Atomic phase checkpoints on a persistent output mount prevent repeating successful stages. All tests write explicitly under pipeline/_test.

The workspace is a new non-Git directory; no existing branch or original Aria code is modified. Prompt design: ../annotation_design_v1 (sibling of pipeline). Runtime prompts will be packaged with the pipeline for portable deployment.

Files and owned interfaces:

- castle_pipeline/media.py and tests/test_media.py: MediaExtractor.prepare(source,start,duration,outdir,fps=1,max_dim=1440,stamp=True) -> PreparedClip(frame_paths,frame_times,audio_path); probe(source) -> duration,width,height,has_audio; extract_crop(source,absolute_sec,box,outpath,max_dim=1440) -> Path. One decoder semaphore, temporary files only, independent of whole-video duration.
- castle_pipeline/vertex.py and tests/test_vertex.py: VertexProvider(model,project,location='global',service_tier='standard',initial_concurrency=2,max_concurrency=4,rpm=30,attempts=3,timeout_sec=600); generate(prompt,context,images,audio=None,max_output_tokens=16384) -> {data,usage}; source images are (label,Path). Shared request-level AIMD/rate limiter; typed errors; no live credentials in tests.
- castle_pipeline/schema.py and tests/test_schema.py: validate annotation/audio/detail outputs against real schema and clip-local times, reject invalid references and unsupported modalities; apply detail replacements by ID, preserving null corrections and exposing resegmentation requests.
- castle_pipeline/runner.py and tests/test_runner.py: bounded scheduling, input fingerprint, phase checkpoint/reuse, local media preparation, disk-only JSONL merge, per-clip errors, graceful stop on fatal API errors. Metadata never inferred by model.
- run_pipeline.py: list, estimate, run, status; explicit output/scratch directories; local file or pinned HF source path, optional manifest/shard selection.
- jobs.py, Dockerfile, requirements.txt: validate and submit explicit HF Jobs commands using supplied code/output volume URLs and secret names; dry-run default, explicit --submit, no implicit paid calls.
- README.md, skills/castle-caption-jobs/SKILL.md: usage, measured and theoretical resource limits, independent Jobs shard rate budgets, persistent resume, failure rules, no automatic full-day spending.

Task checklist:

- [x] Write and run failing behavioral tests for each owned module before implementing it.
- [x] Implement bounded media decode and footer time overlays; test real FFmpeg synthetic video and tail clip.
- [x] Implement Vertex construction, retries, request-level AIMD and RPM pacing; test a controlled external-call boundary.
- [x] Validate audio/annotation/review payloads and phase checkpoints; test resume, failed phase, fingerprint drift, bounded lazy scheduling and safe corrections.
- [x] Integrate CLI and HF input selection with immutable revisions; run offline end-to-end synthetic media through a test provider.
- [x] Measure memory at fixed concurrency for short versus longer synthetic videos; include process-tree RSS, do not claim a 16 GB cloud run was tested locally.
- [x] Document Jobs Docker/code/output mounts and secrets; exercise submit command generation without submitting.
- [x] Write and validate agent skill, independently assess realistic resume/failed-job scenarios.
- [x] Run full tests, CLI checks and code review; record remaining live-service validation limits.

Verification commands: python -m pytest tests -q --basetemp _test/pytest; python run_pipeline.py --help; python jobs.py --help; local media smoke scripts explicitly write _test/media-smoke. Never use runtime production output paths for tests.
