#!/usr/bin/env python3
"""Trim a published release prefix back to exactly what its manifest declares.

An earlier publish shipped the whole repository tree into an immutable prefix.
This removes only objects that the manifest does not declare; it never touches a
declared file, and it never touches another release.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', required=True)
    parser.add_argument('--bucket', default='Ligant/castle-code')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()

    manifest = json.loads((REPO / 'releases' / f'{args.release}.json').read_text(encoding='utf-8'))
    declared = sorted(set(manifest['identity_files']) | set(manifest['auxiliary_files'])
                      | set(manifest['prompt_files']))
    prefix = f'{args.bucket}/{args.release}'
    listed = subprocess.run(['hf', 'buckets', 'list', prefix, '-R', '-q'],
                            capture_output=True, text=True, check=True, shell=True)
    present = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    # -q yields paths relative to the prefix; rebuild the full object id.
    inside = []
    for entry in present:
        rel = entry.split(f'{args.release}/', 1)[-1].lstrip('/')
        full = entry if entry.startswith(args.bucket) else f'{prefix}/{rel}'
        inside.append((rel, full))
    extra = [(rel, full) for rel, full in inside if rel not in declared]
    missing = [name for name in declared if name not in {rel for rel, _ in inside}]

    print(json.dumps({'release': args.release, 'declared': len(declared),
                      'present': len(inside), 'extra': len(extra), 'missing': missing,
                      'extra_preview': [rel for rel, _ in extra[:10]]}, indent=2, ensure_ascii=False))
    if not extra and not missing:
        print('already exactly the manifest set')
        return 0
    if not args.execute:
        print('dry run; pass --execute to remove the extra objects')
        return 0
    failed = []
    for rel, full in extra:
        proc = subprocess.run(['hf', 'buckets', 'remove', full, '--yes'],
                              capture_output=True, text=True, check=False, shell=True)
        if proc.returncode != 0:
            failed.append({'object': rel, 'stderr': proc.stderr.strip()[-120:]})
    print(json.dumps({'removed': len(extra) - len(failed), 'failed': failed}, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
