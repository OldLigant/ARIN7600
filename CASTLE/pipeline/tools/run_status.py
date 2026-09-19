#!/usr/bin/env python3
"""Read-only run status: persisted GCS state + live Vertex stages + per-stage object counts.

The run list comes from the private registry (``private_runs.py``), so every run is
read with its own bound account, project, state URI and output prefix instead of a
hardcoded list. Nothing here writes: no state, no job, no ledger. Credential paths
and key contents are never printed.

``media/`` is deliberately not counted: it holds tens of thousands of objects per
run, and the mirror never touches it.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import private_runs
from castle_pipeline.batch_cloud import BatchCloud
from tools.cloud_probe import load_credentials

STAGES = ('audio', 'annotation', 'review')
COUNTED = ('final', 'errors', 'requests') + tuple('results/' + stage for stage in STAGES)


def _prefix_of(state_uri: str) -> tuple[str, str]:
    bucket, name = state_uri[len('gs://'):].split('/', 1)
    return bucket, name.rsplit('/', 1)[0] + '/'


def _counts(cloud: BatchCloud, state_uri: str) -> dict:
    seen = Counter()
    bucket, prefix = _prefix_of(state_uri)
    for blob in cloud.storage.list_blobs(bucket, prefix=prefix):
        rest = blob.name[len(prefix):]
        parts = rest.split('/')
        if rest.startswith('results/') and len(parts) > 2:
            seen['results/' + parts[1]] += 1
        else:
            seen[parts[0]] += 1
    return {key: seen.get(key, 0) for key in COUNTED}


def describe(directory: Path, record: dict) -> dict:
    _, credentials = load_credentials(directory / record['sa_key_file'])
    cloud = BatchCloud(record['gcp_project'], 'global', credentials=credentials)
    entry = {'run_id': record['run_id'], 'state_uri': record['state_uri']}
    state, generation = cloud.read_state(record['state_uri'])
    entry['state_generation'] = generation
    if state is None:
        entry['state'] = 'not initialised'
        return entry
    config = state.get('config') or {}
    entry.update({
        'status': state.get('status'), 'stage': state.get('current_stage'),
        'code_hash': config.get('code_hash'), 'model': config.get('model'),
        'max_output_tokens': config.get('max_output_tokens'),
        'selected': len(config.get('rows') or []), 'completed': len(state.get('completed') or []),
        'failures': len(state.get('failures') or []), 'eligible_next': len(state.get('eligible_rows') or []),
        'attention_reason': state.get('attention_reason'), 'last_error': state.get('last_error'),
        'stages': {},
    })
    for stage in STAGES:
        recorded = (state.get('batches') or {}).get(stage) or {}
        job = recorded.get('job') or {}
        if not job.get('name'):
            continue
        entry['stages'][stage] = {
            'recorded': job.get('state'),
            'live': cloud.get_batch(job['name'])['state'],
            'request_rows': len(recorded.get('request_ids') or []),
        }
    entry['gcs_objects'] = _counts(cloud, record['state_uri'])
    return entry


def build(directory: Path, run_ids: list[str] | None) -> list[dict]:
    records = private_runs.read_bindings(directory)
    if run_ids:
        wanted = set(run_ids)
        missing = wanted - {record['run_id'] for record in records}
        if missing:
            raise SystemExit('Unknown run id(s) in the private registry: ' + ', '.join(sorted(missing)))
        records = [record for record in records if record['run_id'] in wanted]
    return [describe(directory, record) for record in records]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--credentials-dir', type=Path, default=REPO / 'credentials')
    parser.add_argument('--run', action='append', default=None,
                        help='Run id from the private registry; repeatable. Default: every registered run.')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args(argv)
    text = json.dumps(build(args.credentials_dir, args.run), indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text, encoding='utf-8', newline='\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
