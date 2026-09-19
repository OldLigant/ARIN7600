#!/usr/bin/env python3
"""Read every declared file back out of a release prefix and compare hashes.

This is the strongest form of the publish check: not "the objects exist" but
"the bytes in the immutable prefix are the bytes the manifest describes".
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', required=True)
    parser.add_argument('--bucket', default='Ligant/castle-code')
    args = parser.parse_args()

    manifest = json.loads((REPO / 'releases' / f'{args.release}.json').read_text(encoding='utf-8'))
    declared = {}
    for key in ('identity_files', 'auxiliary_files', 'prompt_files'):
        declared.update(manifest[key])

    with tempfile.TemporaryDirectory(prefix='release-readback-') as work:
        target = Path(work)
        proc = subprocess.run(['hf', 'buckets', 'sync',
                               f'hf://buckets/{args.bucket}/{args.release}', str(target)],
                              capture_output=True, text=True, check=False, shell=True)
        if proc.returncode != 0:
            print(json.dumps({'ok': False, 'error': 'SYNC_FAILED',
                              'detail': proc.stderr.strip()[-200:]}, indent=2))
            return 1
        mismatched, missing = [], []
        for name, expected in sorted(declared.items()):
            local = target / name
            if not local.is_file():
                missing.append(name)
                continue
            actual = hashlib.sha256(local.read_bytes()).hexdigest()
            if actual != expected:
                mismatched.append({'file': name, 'manifest': expected[:12], 'bucket': actual[:12]})
        # recompute the published code_hash from the bucket bytes themselves
        from release import fingerprint, identity_paths  # noqa: F401  (scheme kept explicit)
        identity_basenames = {Path(n).name: hashlib.sha256((target / n).read_bytes()).hexdigest()
                              for n in manifest['identity_files'] if (target / n).is_file()}
        recomputed = fingerprint(identity_basenames)
        result = {'ok': not missing and not mismatched and recomputed == manifest['code_hash'],
                  'release': args.release,
                  'files_checked': len(declared),
                  'missing': missing,
                  'mismatched': mismatched,
                  'manifest_code_hash': manifest['code_hash'],
                  'code_hash_from_bucket_bytes': recomputed,
                  'hash_matches': recomputed == manifest['code_hash']}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
