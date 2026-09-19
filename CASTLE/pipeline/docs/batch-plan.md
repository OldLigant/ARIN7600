# Vertex Batch implementation plan

Historical plan and test record from 2026-09-16. References below to unconfigured credentials or unverified cloud activation describe that date, not the current deployment. See docs/batch.md, STATUS.md and docs/version-evolution-decisions.md before operating the pipeline.

Scope: add resumable offline Batch execution without replacing online standard/flex. User requested two HF worker roles, hourly status checks and skill updates. Actual cloud activation awaits a supplied GCS prefix and usable Google ADC identity; no inferred bucket creation/IAM changes or paid submissions.

Architecture: persist media, per-clip metadata, JSONL requests and cloud operation state in GCS; HF submitter prepares one source sequentially then creates audio batch and exits. A tick worker checks each stage once, exits while pending, collects terminal output keyed by embedded stable request identifier, validates per row, and submits only eligible next-stage requests. Audio → annotation → optional native-frame review are separate batches. Missing/invalid rows never become success; partial successes advance. Final annotations and audits also copy to mounted HF output. No waiting daemon.

Interfaces / ownership:
- batch_cloud.py: ADC-backed Google storage/genai adapter. upload(path,uri), download(uri,path), write_jsonl(uri,rows), iter_jsonl(prefix), read_state(uri)->(state|None,generation), write_state(uri,state,expected_generation)->generation; create_batch(model,input_uri,output_uri,display_name)->{name,state,output_uri}, get_batch(name)->dict, find_batches(display_name)->list. GCS state uses generation preconditions. Raw provider response data is persisted, not printed. Storage objects are immutable.
- batch_engine.py: Engine(cloud,state_uri,planner). initialize(config); start(); tick(). Planner.prepare(stage,eligible_rows,state)-> {input_uri,output_uri,request_ids}; planner.collect(stage,job,state)-> {eligible_rows,failures,completed}. Persist submission intent before create; no automatic repeat of uncertain submissions. Planner role implemented by root.
- batch_pipeline.py and batch_tasks.py: CLI prepare/tick/status plus media/request/result planner. Exact allowed run scope persisted. Vertex JSONL format {request:{contents,systemInstruction,generationConfig}}; marker in echoed request text identifies row, not output order. GCS fileData references; no large inline base64 file. Originals remain available for selective review crops.
- batch_jobs.py: dry-run by default; initial submit worker and hourly non-overlapping scheduled tick argv. Actual commands require explicit cloud configuration. Scheduled task creation is not performed until those settings are supplied.
- tests/test_batch_*.py: failing tests then implement; output in fresh _test dirs. Offline cloud doubles only at external boundary; test out-of-order output, partial/missing rows, expired/failedjob handling, ambiguous create, compare-and-swap races and repeated ticks.

Steps:
- [x] Official Batch capability, limits, JSONL/auth/output research.
- [x] Implement transport and durable state machine in parallel with planner/CLI.
- [x] Synthetic media end-to-end with fake cloud/model results, no paid requests.
- [x] HF worker command rendering, docs, skill update and independent review.
- [x] Full tests and precise reporting of unactivated cloud configuration.

Verification (2026-09-16): `python -m pytest tests -q -p no:cacheprovider --basetemp _test/batch-full-20260916b` using `_test/runtime/Scripts/python.exe`: **184 passed in 10.85s**. The skill validator passed under Python UTF-8 mode; installed and repository SKILL.md hashes match. Submit and hourly launcher dry-run parsing verified. Independent review's malformed-crop, duplicate-scope, initialization-race and submission-visibility findings addressed and regression-tested. No real cloud jobs, schedules, IAM or buckets created. Cloud authentication, live Batch latency/quality, and HF memory remain unverified until configuration is supplied.
