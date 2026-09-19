#!/usr/bin/env python3
"""Pre-flight a stage's Vertex Batch output before collecting it (R-09 companion).

A tick turns whatever the batch returned into durable results: valid rows become
final annotations, invalid rows become permanent failures. This checks the output
*before* that happens, so a systematically broken stage is visible while it is
still just a file in a bucket.

Reported per run: expected rows (from the run's eligible set), rows actually
returned, how many are valid JSON, and how many are schema-valid. Read-only; it
collects nothing and changes no state.

Usage:
  python tools/preflight_stage.py                       # every known run
  python tools/preflight_stage.py --run bjorn-10-14     # one run
  python tools/preflight_stage.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.cloud_probe import DEFAULT_CREDENTIALS, RUNS, load_credentials  # noqa: E402


def _text_of(row) -> str:
    candidates = (row.get('response') or {}).get('candidates') or []
    parts = ((candidates[0].get('content') or {}).get('parts') or []) if candidates else []
    return ''.join(p.get('text', '') for p in parts
                   if isinstance(p, dict) and not p.get('thought'))


def classify(stage: str, data: dict, duration: float) -> list[str]:
    """Apply the same validation the collector will apply, without saving anything."""
    from castle_pipeline.schema import validate_annotation, validate_audio, normalize_annotation
    problems = []
    try:
        if stage == 'audio':
            validate_audio(data, duration)
        elif stage == 'annotation':
            normalized, _ = normalize_annotation(
                {k: v for k, v in data.items() if k != 'review_regions'}, duration, {'video_0'})
            validate_annotation(normalized, duration, {'video_0'})
        elif stage == 'review':
            return []          # review is a replacement contract, checked against the first pass
        else:
            return [f'unknown stage {stage}']
    except (ValueError, KeyError, TypeError) as error:
        problems.append(str(error)[:110])
    return problems


def inspect_run(label, state, client, cloud, parse_gs_uri) -> dict:
    stage = state['current_stage']
    if stage is None:
        # A finished run has no current stage; report its terminal counts instead.
        return {'run': label, 'terminal': True, 'status': state.get('status'),
                'completed': len(state.get('completed') or []),
                'failures': len(state.get('failures') or [])}
    record = (state.get('batches') or {}).get(stage) or {}
    job = record.get('job') or {}
    entry = {'run': label, 'stage': stage, 'vertex_recorded': job.get('state'),
             'expected_rows': len(state.get('eligible_rows') or [])}
    output_uri = job.get('output_uri')
    if not output_uri:
        entry['note'] = 'no output uri recorded yet'
        return entry

    bucket_name, name = parse_gs_uri(output_uri)
    blobs = [b for b in client.list_blobs(bucket_name, prefix=name.rstrip('/') + '/')
             if b.name.endswith('.jsonl')]
    entry['jsonl_files'] = len(blobs)
    if not blobs:
        entry['note'] = 'no output published yet (Vertex writes predictions.jsonl at job end)'
        return entry

    rows = parseable = schema_ok = 0
    unparseable_examples, schema_examples = [], []
    for blob in blobs:
        with blob.open('rt', encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if 'response' not in row:          # echoed request row
                    continue
                rows += 1
                text = _text_of(row)
                try:
                    data = json.loads(text)
                except ValueError:
                    if len(unparseable_examples) < 3:
                        unparseable_examples.append(text[-40:].replace('\n', '\\n'))
                    continue
                parseable += 1
                # Duration is only needed to bound times; a generous bound keeps this
                # a smoke check rather than a second implementation of the contract.
                problems = classify(entry['stage'], data, 3600.0)
                if problems:
                    if len(schema_examples) < 3:
                        schema_examples.append(problems[0])
                else:
                    schema_ok += 1
    entry.update({'response_rows': rows, 'json_parseable': parseable,
                  'schema_valid': schema_ok,
                  'json_unparseable': rows - parseable,
                  'schema_invalid': parseable - schema_ok,
                  'unparseable_tails': unparseable_examples,
                  'schema_examples': schema_examples})
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--credentials', type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument('--run', default=None, help='limit to one run label from cloud_probe.RUNS')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    _, credentials = load_credentials(args.credentials)
    from google.cloud import storage
    from castle_pipeline.batch_cloud import BatchCloud, parse_gs_uri
    client = storage.Client(project='my-project-omni-507802', credentials=credentials)
    cloud = BatchCloud('my-project-omni-507802', 'global', credentials=credentials)

    results = []
    for label, prefix in RUNS.items():
        if args.run and args.run not in label:
            continue
        try:
            state, _ = cloud.read_state(f'gs://castle-caption-batch/{prefix}/state.json')
        except Exception as error:
            results.append({'run': label, 'error': f'{type(error).__name__}'})
            continue
        if state is None:
            results.append({'run': label, 'note': 'state not initialised yet'})
            continue
        results.append(inspect_run(label, state, client, cloud, parse_gs_uri))

    if args.json:
        print(json.dumps({'runs': results}, indent=2, ensure_ascii=False))
        return 0
    header = f"{'run':24s} {'stage':11s} {'expected':>8s} {'rows':>6s} {'json_ok':>8s} " \
             f"{'schema_ok':>9s} {'bad':>5s}"
    print(header)
    for entry in results:
        if entry.get('terminal'):
            print(f"{entry['run']:24s} terminal: {entry['status']}  "
                  f"completed={entry['completed']} failures={entry['failures']}")
            continue
        if 'stage' not in entry:
            print(f"{entry['run']:24s} {entry.get('note') or entry.get('error')}")
            continue
        bad = entry.get('json_unparseable', 0) + entry.get('schema_invalid', 0)
        print(f"{entry['run']:24s} {entry['stage']:11s} {entry['expected_rows']:8d} "
              f"{entry.get('response_rows', 0):6d} {entry.get('json_parseable', 0):8d} "
              f"{entry.get('schema_valid', 0):9d} {bad:5d}")
        for tail in entry.get('unparseable_tails', []):
            print(f"    unparseable tail: {tail!r}")
        for example in entry.get('schema_examples', []):
            print(f"    schema problem  : {example}")
        if entry.get('note'):
            print(f"    note: {entry['note']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
