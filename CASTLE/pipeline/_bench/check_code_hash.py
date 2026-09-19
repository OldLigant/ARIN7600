#!/usr/bin/env python3
"""Compare the persisted batch code_hash against what each published version computes.

Read-only. Prints hashes only, never credential or response content.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS = {
    '08': 'gs://castle-caption-batch/castle/day1-bjorn-08-v1/state.json',
    '09': 'gs://castle-caption-batch/castle/day1-bjorn-09-v2/state.json',
    '10-14': 'gs://castle-caption-batch/castle/day1-bjorn-10-14-v2/state.json',
}
# Files code_hash() covers, in the shape it hashes them.
COVERED = ['batch_pipeline.py'] + [f'castle_pipeline/{name}' for name in
    ('__init__.py', 'batch_cloud.py', 'batch_engine.py', 'batch_tasks.py', 'events.py',
     'inputs.py', 'media.py', 'runner.py', 'schema.py', 'vertex.py')]


def code_hash_over(paths):
    from castle_pipeline.runner import fingerprint
    return fingerprint({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in paths if p.is_file()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--credentials', type=Path,
                        default=Path(r'D:\QLD\out\castle-runs\b20260917-bjorn08\secrets.env'))
    args = parser.parse_args()

    # Local tree with the v3 change applied.
    local = [ROOT / name for name in COVERED]
    local_hash = code_hash_over(local)
    print(json.dumps({'local_tree_code_hash': local_hash}, indent=2))

    raw = args.credentials.read_text(encoding='utf-8').strip()
    info = json.loads(raw.split('=', 1)[1] if raw.startswith('GOOGLE_ADC_JSON=') else raw)
    from google.oauth2 import service_account
    from google.cloud import storage
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=['https://www.googleapis.com/auth/cloud-platform'])
    client = storage.Client(project=info['project_id'], credentials=credentials)

    from castle_pipeline.batch_cloud import parse_gs_uri
    for label, uri in RUNS.items():
        bucket_name, name = parse_gs_uri(uri)
        payload = json.loads(client.bucket(bucket_name).blob(name).download_as_bytes())
        stored = (payload.get('config') or {}).get('code_hash')
        print(json.dumps({'run': label, 'stored_code_hash': stored,
                          'matches_local_v3': stored == local_hash,
                          'status': payload.get('status'),
                          'current_stage': payload.get('current_stage'),
                          'config_keys': sorted((payload.get('config') or {}).keys())}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
