#!/usr/bin/env python3
"""Compute code_hash() for each published code version and match it to run state.

Downloads each published version into scratch, hashes exactly the files
code_hash() covers, and reports which version matches each run's stored hash.
Read-only with respect to GCS state.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VERSIONS = ['castle-batch-v1', 'castle-batch-v2', 'castle-batch-v3']
RUNS = {
    '08': 'gs://castle-caption-batch/castle/day1-bjorn-08-v1/state.json',
    '09': 'gs://castle-caption-batch/castle/day1-bjorn-09-v2/state.json',
    '10-14': 'gs://castle-caption-batch/castle/day1-bjorn-10-14-v2/state.json',
}


def covered(root: Path):
    return [root / 'batch_pipeline.py'] + sorted((root / 'castle_pipeline').glob('*.py'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--credentials', type=Path,
                        default=Path(r'D:\QLD\out\castle-runs\b20260917-bjorn08\secrets.env'))
    parser.add_argument('--work', type=Path, default=Path(r'D:\QLD\out\castle-runs\_version-check'))
    args = parser.parse_args()

    raw = args.credentials.read_text(encoding='utf-8').strip()
    info = json.loads(raw.split('=', 1)[1] if raw.startswith('GOOGLE_ADC_JSON=') else raw)
    from google.oauth2 import service_account
    from google.cloud import storage
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=['https://www.googleapis.com/auth/cloud-platform'])
    client = storage.Client(project=info['project_id'], credentials=credentials)

    from castle_pipeline.batch_cloud import parse_gs_uri
    from castle_pipeline.runner import fingerprint

    # Code versions live in HF buckets; state lives in GCS.
    import subprocess
    version_hashes = {}
    for version in VERSIONS:
        root = args.work / version
        if root.exists():
            import shutil
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(['hf', 'buckets', 'sync',
                               f'hf://buckets/Ligant/castle-code/{version}', str(root)],
                              capture_output=True, text=True, shell=True)
        files = [p for p in covered(root) if p.is_file()]
        if not files:
            version_hashes[version] = {'hash': None, 'files': [], 'error': proc.stderr[-200:]}
            continue
        digest = fingerprint({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        version_hashes[version] = {'hash': digest, 'files': [str(p.relative_to(root)) for p in files]}

    print(json.dumps({'published_versions': {k: v['hash'] for k, v in version_hashes.items()}}, indent=2))

    for label, uri in RUNS.items():
        bucket_name, name = parse_gs_uri(uri)
        payload = json.loads(client.bucket(bucket_name).blob(name).download_as_bytes())
        stored = (payload.get('config') or {}).get('code_hash')
        match = [v for v, info_ in version_hashes.items() if info_['hash'] == stored]
        print(json.dumps({'run': label, 'stored_hash': stored, 'matching_version': match or 'NONE',
                          'current_stage': payload.get('current_stage')}, indent=2))

    # Which files differ between v2 and v3 explains the hash change.
    print(json.dumps({'v2_file_count': len(version_hashes['castle-batch-v2']['files']),
                      'v3_file_count': len(version_hashes['castle-batch-v3']['files'])}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
