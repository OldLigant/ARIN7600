#!/usr/bin/env python3
"""Verify the release registry against what each cloud run actually pinned.

Three-way check for every batch run in GCS state:
  1. the release manifest in releases/<release>.json recomputes to code_hash
  2. the manifest's identity_files match the bytes published in the code bucket
  3. the run's stored code_hash equals the manifest's code_hash

Read-only. Prints hashes and release names, never credentials or model output.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS = {
    'bjorn-08-v1': 'gs://castle-caption-batch/castle/day1-bjorn-08-v1/state.json',
    'bjorn-09-v2': 'gs://castle-caption-batch/castle/day1-bjorn-09-v2/state.json',
    'bjorn-10-14-v2': 'gs://castle-caption-batch/castle/day1-bjorn-10-14-v2/state.json',
    'allie-13-14-18-20-v3': 'gs://castle-caption-batch/castle/day1-allie-13-14-18-20-v3/state.json',
    'bjorn-15-20-v3': 'gs://castle-caption-batch/castle/day1-bjorn-15-20-v3/state.json',
}


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def published_files(release, work):
    target = work / release
    if target.exists():
        import shutil
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(['hf', 'buckets', 'sync', f'hf://buckets/Ligant/castle-code/{release}', str(target)],
                   capture_output=True, text=True, shell=True)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--credentials', type=Path,
                        default=Path(r'D:\QLD\out\castle-runs\b20260917-bjorn08\secrets.env'))
    parser.add_argument('--work', type=Path, default=Path(tempfile.gettempdir()) / 'castle-release-check')
    args = parser.parse_args()

    raw = args.credentials.read_text(encoding='utf-8').strip()
    info = json.loads(raw.split('=', 1)[1] if raw.startswith('GOOGLE_ADC_JSON=') else raw)
    from google.oauth2 import service_account
    from google.cloud import storage
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=['https://www.googleapis.com/auth/cloud-platform'])
    client = storage.Client(project=info['project_id'], credentials=credentials)

    manifests = {}
    for path in sorted((ROOT / 'releases').glob('*.json')):
        manifest = json.loads(path.read_text(encoding='utf-8'))
        release = manifest['release']
        # Rule R-03: code_hash must be recomputable from identity_files as basenames.
        recomputed = fingerprint({Path(k).name: v for k, v in manifest['identity_files'].items()})
        bucket = published_files(release, args.work)
        published = {}
        mismatched = []
        for rel, digest in manifest['identity_files'].items():
            local = bucket / rel
            if not local.is_file():
                published[rel] = None
                mismatched.append(rel)
                continue
            actual = hashlib.sha256(local.read_bytes()).hexdigest()
            published[rel] = actual
            if actual != digest:
                mismatched.append(rel)
        manifests[release] = {'manifest': manifest, 'recomputed': recomputed,
                              'published_mismatches': mismatched}
        print(json.dumps({
            'release': release,
            'manifest_code_hash': manifest['code_hash'],
            'recomputed_from_identity_files': recomputed,
            'hash_rule_ok': recomputed == manifest['code_hash'],
            'git_commit_in_manifest': (manifest.get('provenance') or {}).get('git_commit'),
            'files_mismatching_published_bucket': mismatched,
        }, indent=2))

    print(json.dumps({'--- run pins ---': True}))
    from castle_pipeline.batch_cloud import parse_gs_uri
    for label, uri in RUNS.items():
        bucket_name, name = parse_gs_uri(uri)
        blob = client.bucket(bucket_name).blob(name)
        try:
            payload = json.loads(blob.download_as_bytes())
        except Exception as error:
            print(json.dumps({'run': label, 'error': f'{type(error).__name__}'}))
            continue
        config = payload.get('config') or {}
        stored = config.get('code_hash')
        record = config.get('release')
        match = [r for r, m in manifests.items() if m['manifest']['code_hash'] == stored]
        print(json.dumps({'run': label, 'state_release_field': record, 'stored_code_hash': stored,
                          'matching_release': match or 'NONE',
                          'status': payload.get('status'), 'stage': payload.get('current_stage')}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
