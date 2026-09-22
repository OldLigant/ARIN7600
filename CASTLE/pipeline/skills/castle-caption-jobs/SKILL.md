---
name: castle-caption-jobs
description: Use when preparing, submitting, monitoring, resuming, or diagnosing CASTLE video-captioning runs on Hugging Face Jobs using the CASTLE Vertex AI pipeline. Also applies to checking its stage outputs and resource budgets.
---

# CASTLE caption jobs

Use the maintained pipeline rather than regenerating a one-off worker. Locate it from the user's supplied path or `CASTLE_PIPELINE_DIR`; on the author's machine it is `D:/QLD/CASTLE/pipeline`. If unavailable, request its location. Read that directory's `README.md`, `run_pipeline.py --help` and `jobs.py --help` for actual interfaces; do not assume the separate Aria pipeline has the same options.

## Choose execution mode

The user is moving bulk annotation to **Vertex Batch** because online standard/flex is slow or throttled. Prefer Batch for that requested workflow when GCS/ADC are available; do not silently fall back to paid online work. Batch is not a service tier. Read `<pipeline>/docs/batch.md`, `batch_pipeline.py --help` and `batch_jobs.py --help` before preparing Batch runs. The sections after this Batch block describe the existing online runner unless explicitly shared.

### Batch workflow

- For multiple GCP accounts, read `<pipeline>/credentials/README.md` first. Persist the run → local SA JSON filename → execution project → exact HF owner/bucket/output prefix association only in ignored `<pipeline>/credentials/run-bindings.jsonl`, through `private_runs.py append` and `validate --check-keys`; never hand-edit it. Use `lookup_binding(credentials_dir, run_id)` in memory before preparing, probing or resuming. If a binding is missing, recover it from verified local receipts or ask; never infer the account from the currently configured default, HF username or run name. Key-owner project and execution project can legitimately differ. Keep SA paths/emails/account labels and private associations out of the public ledger (including note/detail), STATUS, tracked docs, release notes and cloud artifacts. `show --reveal` is explicit private inspection only; do not paste its output into those destinations.
- Selecting launcher `--credentials-file` only selects credentials for local GCS pin lookup. It does **not** populate the HF worker secret. For an authorized launch, load the same bound SA JSON into the named secret variable (normally GOOGLE_ADC_JSON), use the binding's execution project and HF output URI, then clear the variable. Existing cloud probes require an explicit credential override and still hardcode the historical project/RUNS list: changing `--credentials` alone does not target a new project. Query new runs with their explicit bound project/state/key instead of treating the old report as coverage. The registry is an operator workflow, not automatic credential routing or permission to switch an existing run's project/key.
- Do not infer activation state from this skill's age. Read the pipeline's timestamped STATUS.md and ledger/ledger.jsonl; use a read-only cloud probe when current state matters. Existing configuration or a past submission does not itself authorize a new run, retry or schedule. Reuse the scope explicitly authorized in the current conversation.
- Require project/location/model, dedicated unused GCS run prefix, exact distinct source paths and clip range, pinned HF code prefix, separate HF output prefix, and ADC usable inside the HF container. A Google API key or local gcloud login alone does not satisfy this. Never print ADC or put it in code/output buckets; the launcher passes a secret variable name and materializes private credentials on ephemeral disk. Existing HF login can submit jobs without injecting HF_TOKEN into this worker.
- There are two HF worker roles, not two total executions: `batch_jobs.py submit` prepares/uploads and starts audio Batch, then detaches; `hourly-tick` creates a non-overlapping hourly schedule whose short executions check once, collect terminal results, and submit annotation then optional crop review. Model default is gemini-3.8-flash, location global; verify access when activating. Batch may queue despite lacking online RPM control; do not promise instant concurrency or turnaround.
- Both launcher roles render argv by default; `--execute` actually submits/creates schedule. Put runtime prepare flags after `--`, launcher flags before it. `batch_pipeline.py prepare` itself uploads data and is NOT a dry-run. Start with one source and 2–3 clips when asked for a smoke test; max-clips is per source, total default bound 2,000. Publish bootstrap_batch.py and requirements-batch.txt as well as the Batch entrypoints/package/prompts.
- Media preparation supports bounded CPU parallelism; use docs/batch.md and the chosen flavor's actual resources to select media_threads/decode_slots/prepare_workers and the memory hint. These values are set at prepare time and stored in Batch config; an initialized Batch run's tick does not expose a retuning interface. Online resume excludes execution-tuning values from its fingerprint, but this does not imply Batch config is mutable or that changing FFmpeg thread counts preserves JPEG bytes. The measured byte-invariance test covers footer_workers with fixed decoder settings.
- Record initial HF Job ID, schedule ID, GCS state URI, actual Vertex Job IDs, code version, selected sources/range and output prefix in the local ledger. Each run has its own GCS prefix; do not share a state between shards. Use `batch_pipeline.py status` and `batch-summary.json`; online status/checkpoint paths do not scan Batch finals.
- GCS state uses generation preconditions and immutable artifacts. Never edit state hashes or clear submission intent. If create response was lost, no repeated create is allowed: automatic reconciliation has a bounded visibility grace; explicit `reconcile` only searches/adopts exactly one matching Job. Ambiguous/no matches require inspection. Keep the pinned code version throughout recovery.
- Failed/invalid/missing rows remain recorded; valid rows advance. Final JSON appears in GCS and HF `final/<clip_id>.json`. Batch collect logs per-row outcome and tokens, but there is no live per-request AIMD/10-minute heartbeat while HF is off. Replayed collection logs can repeat; deduplicate by request_id for accounting. Native 1 fps images persist in GCS for later crop review; budget their storage too.
- On `stop_schedule: true` (complete, complete_with_errors, needs_attention or paused), use logged-in `hf jobs scheduled suspend SCHEDULE_ID`; this is part of managing an authorized run. The worker does NOT self-suspend, so leaving the schedule running incurs repeated HF startup costs. Suspending it does not cancel an already accepted Vertex Batch. Report partial failures and review_required counts; do not automatically resubmit a whole run. Any targeted retry uses a new prefix and explicit unresolved scope; cross-run checkpoint import is not implemented.
- Before reporting activation success, verify one real scoped run's credentials/media/outputs. Offline tests alone do not prove model access or annotation quality. Keep standard as an explicitly selected fallback; do not restart cancelled Flex jobs.

## Establish a concrete run

Identify the dataset commit and selected source paths or manifest/shard, Vertex model/project/location/tier, code version, persistent output bucket, clip range, RPM allocation, and job timeout. Reuse already-authorized values. Do not infer a model name from examples or echo credential values. CASTLE defaults are 1 fps, 30 seconds per clip, 1440 max dimension, 3 in-flight clips, one decoder, audio first, and optional crop review.

For explicitly selected online runs, the user's preference is gemini-3.8-flash with standard service tier. The slow Flex smoke was cancelled; do not restart it automatically. An explicit later model/tier instruction takes precedence. Switching tier changes the fingerprint and is not transparent checkpoint reuse.

`--max-clips` limits each source, not the entire manifest. Calculate the total selected clips before submission. One hour is 120 clips, normally 240 audio+visual calls and up to 120 review calls, excluding retries. If asked for a smoke test, select one explicit source and 2–3 clips, not a whole day's manifest with max-clips=3.

Use `python jobs.py` to render the argv first. It does not submit without `--submit`. Building/submitting jobs is authorized only within the user's requested run scope; a request to explain or prepare commands is not a request to start paid work. If already asked to submit a concrete run, proceed without asking again. Record the returned job ID, not just its non-unique name.

## Persistence and resume

- Code is a dedicated versioned bucket prefix mounted read-only at `/workspace`. Results use a separate persistent bucket prefix at `/output`; scratch/downloads remain local. Never sync the full workspace, test virtualenv, media, or secrets into the code bucket.
- Recover the entire output tree if one exists. A successful audio log is not a persisted audio checkpoint. A killed job with ephemeral-only output may have no recoverable stages; explain that limitation rather than promise free resume.
- Use `run_pipeline.py status --output-dir ...` and inspect source `run.json`, `summary.json`, phase `audio.json`/`annotation.json`/`review.json`, and `final.json`. An old error.json may remain next to a subsequently successful final; final validation controls reuse.
- Keep code, prompts, dataset commit, model, tier, fps, clip duration, image size, stamp and review settings unchanged to reuse fingerprints. Do not edit hashes or relabel earlier outputs. Workers, RPM and retry limits can be reduced without invalidating the annotation identity.
- Recovered phase checkpoints can save model calls even if media must be downloaded/extracted again. Changing code/tier/model yields new output branches and may repeat prior work; disclose before doing this to a paid run.
- A stale `run.lock` may follow a kill. Read owner job_id/host/pid/nonce and confirm the old job has terminated. Remove only that exact stale lock when authorized to resume; never clear an active lock or delete the output tree. Remote mount locks are not a substitute for disjoint shards.

## Scale out: naming and the local ledger

Two long-lived buckets carry every run. Never create a bucket per job, per hour or per day.

- Code: `hf://buckets/<owner>/castle-code/<code-version>` (e.g. `castle-v3`). Published versions are immutable: bump on any pipeline or prompt change, never overwrite one that already produced outputs.
- Output: `hf://buckets/<owner>/castle-output/<batch>/<scope>`.

`<batch>` is the campaign - one dataset commit plus one code version - named `b<YYYYMMDD>-<purpose>` (e.g. `b20260916-main`). Keep model, tier, fps and image size out of the batch name: the pipeline writes a `<fingerprint>/<source-hash>/` branch underneath and that fingerprint already encodes them, so a tier or model change is a new branch, not a new bucket.

`<scope>` is the readable data scope of one submitted job: `<day>/<stream>/<HH>` for a single source, `shard-<i>of<n>` for a manifest shard, `<day>` for a whole day.

Jobs may share one output prefix only when their sources are disjoint, which sharding guarantees: the pipeline keys its lease, stage checkpoints and `captions.jsonl` on `<fingerprint>/<source-hash>`. Never let two active jobs cover the same source under one prefix, or run two jobs with different `shard_count` over one manifest. Name jobs `castle-<batch>-<scope>-<shard>` and always record the returned job id, since names are not unique.

Use `<pipeline>/ledger/ledger.jsonl` as the submission ledger. Read `python ledger.py append --help` and write through that entry point; run `ledger.py validate` after changes. Preserve Job ID, role, scope, pin, state URI and the output-prefix information in the schema's supported fields. Add later status events rather than editing historical rows. Old `D:/QLD/out/castle-runs/<batch>/ledger.jsonl` and PROGRESS.md files are historical records, not the current write destination.

`tools/report.py` combines that ledger with a cloud probe to generate timestamped STATUS.md. Job notes are historical prose; use the run table and probe time for stage status, and distinguish HF worker completion from whole-run completion. Read the ledger before submission and append immediately after HF returns a Job ID. For online recovery, reuse the compatible source/config output scope; Batch failed-row retries need a separate authorized prefix, while existing successful stages advance through the pinned run's tick.

After appending to the ledger or regenerating STATUS.md, publish the operator view with `python tools/push_ops.py --execute`: it validates the ledger and uploads `ledger.jsonl` plus `STATUS.md` to `hf://buckets/Ligant/castle-output/ops/` (render first without `--execute`). Those are overwritable viewing copies for operators without a checkout — never add credentials, run bindings or release directories there, and never treat the bucket copy as more current than local Git.

Read `<pipeline>/docs/version-evolution-decisions.md` before proposing upgrades. Cross-version audio inheritance, identity decoupling and additional hash-report tooling are deferred designs, not implemented capabilities or standing permission to migrate running work.

When syncing code on PowerShell, use single-star exclude patterns only (`_test/*`, `.pytest_cache/*`, `__pycache__/*`, `.env*`, `*.secrets`, `credentials/*`) and run the sync from a directory where they match nothing, e.g. the workspace root. Patterns containing `**` are expanded by the shell before `hf` sees them, and the sync dies with "Got unexpected extra arguments". `credentials/*` is required: that directory holds the service-account JSON and the job API key, and neither `*.secrets` nor `.env*` covers a directory.

At scale, keep one manifest per batch with its pinned commit and divide the project RPM budget across concurrent jobs. Measured on a full 120-clip hour at `--rpm 30 --workers 3`: 416 requests, 314 succeeded, 102 were HTTP 429, and AIMD sat at limit 1-2 for most of the run. Lower `--rpm` (e.g. 15) to cut 429 churn; RPM, workers and retry limits do not change the annotation fingerprint, so a resume can retune them without losing completed clips.

## Monitor and respond

Use `hf jobs inspect JOB_ID` and `hf jobs logs JOB_ID`; inspect output summaries where available. Do not automatically resubmit on every failed status. Retry only the unresolved scope after understanding its category:

Current logs include stage_start/success/reused/skipped, request_queued/start/success/failed, retry_scheduled and aimd_change. Correlate request_id plus attempt; success usage contains input/output/thought/cached/total tokens, failures contain safe HTTP or normalized codes. The independent progress heartbeat defaults to600seconds and reports active stages/requests, waits, AIMD, counts, RSS and received-usage totals even when no clip finishes. Source processing appends events.jsonl to persistent output; downloads log to Job stdout. Do not treat a quiet request between heartbeats as proof of a hang, or missing failure usage as zero cost.

- quota/server/network: bounded retries already happen; after exhaustion preserve successes, lower the job's RPM/concurrency if appropriate, then resume within the authorized budget.
- auth/model/config/billing: stop retrying until the configuration or account problem is corrected. Do not route to a different service, model or tier silently.
- validation/output: inspect the failing phase and its contract; do not convert malformed/empty responses into successful annotations. Repeated deterministic failure needs a prompt/contract correction and may change the fingerprint.
- OOM/disk: read peak process-tree RSS, cgroup figures and download size; distinguish RAM from local disk. Reduce in-flight workers first; keep one decoder. Do not restore whole-hour frame accumulation. `--keep-source`/`--keep-media` can fill disk.

Multiple Jobs do not share an AIMD limiter. Divide the project-wide RPM budget among jobs, and ensure identical manifest/filter/shard_count with distinct shard_index. Do not run the same source/config simultaneously against the same output prefix. Expanding from smoke to hour/day/full dataset needs corresponding user authorization, not just a green smoke test.

## Quality and completion

Check a few final annotations against media: test-card intervals, timestamp footer excluded from scene text, voice attribution, wearer versus other people's actions, meaningful action granularity, and world_changes versus newly_observed. FACE metadata is not used for identity; there is no global person tracker. Do not infer those capabilities from local person IDs.

Report selected/completed/failed/unprocessed/reused counts, pending review_required clips, observed memory, token usage and remaining verification limits. A clean process exit proves automatic processing completed, not annotation accuracy. Preserve exact dataset/code/model settings so another agent can continue.
