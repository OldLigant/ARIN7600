#!/usr/bin/env python3
"""Read-only mirror audit: does the HF output prefix hold exactly the GCS artifacts?

Compares, per run, the object-id sets of GCS ``final/`` and ``results/<stage>/``
against the same relative paths under the run's HF output prefix, and reports what
is missing, extra, or unexpectedly present (images, audio, ``media/``). The run
list and output prefix come from the private registry, so the mapping is never
hardcoded and no credential path is printed.

This is the check to run after a tick that mirrored: GCS stays the source of
truth, the HF copy is a derived view, and a mismatch means the mirror did not
finish rather than that the run changed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import private_runs
from castle_pipeline.batch_cloud import BatchCloud
from tools.cloud_probe import load_credentials

STAGES = ('audio', 'annotation', 'review')
MEDIA_SUFFIXES = ('.jpg', '.jpeg', '.png', '.wav')


def _prefix_of(state_uri: str) -> tuple[str, str]:
    bucket, name = state_uri[len('gs://'):].split('/', 1)
    return bucket, name.rsplit('/', 1)[0] + '/'


def gcs_ids(cloud: BatchCloud, state_uri: str) -> dict:
    seen = {key: set() for key in [('final',)] + [('results', stage) for stage in STAGES]}
    bucket, prefix = _prefix_of(state_uri)
    for blob in cloud.storage.list_blobs(bucket, prefix=prefix):
        rest = blob.name[len(prefix):]
        parts = rest.split('/')
        if not rest.endswith('.json'):
            continue
        if len(parts) == 2 and parts[0] == 'final':
            seen[('final',)].add(parts[1])
        elif len(parts) == 3 and parts[0] == 'results' and parts[1] in STAGES:
            seen[('results', parts[1])].add(parts[2])
    return seen


def hf_listing(output_uri: str) -> list[str]:
    proc = subprocess.run(['hf', 'buckets', 'list', '-R', output_uri, '--format', 'json'],
                          capture_output=True, text=True, check=False, shell=True)
    if proc.returncode != 0:
        raise SystemExit('Could not list %s: %s' % (output_uri, proc.stderr.strip()[:200]))
    payload = json.loads(proc.stdout.lstrip('\ufeff') or '[]')
    return [item['path'].split('/', 1)[1] if '/' in item['path'] else item['path'] for item in payload]


def audit(directory: Path, record: dict) -> dict:
    _, credentials = load_credentials(directory / record['sa_key_file'])
    cloud = BatchCloud(record['gcp_project'], 'global', credentials=credentials)
    source = gcs_ids(cloud, record['state_uri'])
    listing = hf_listing(record['hf_output_uri'])
    mirror = {key: set() for key in source}
    others, media = [], []
    for path in listing:
        parts = path.split('/')
        if len(parts) == 2 and parts[0] == 'final' and path.endswith('.json'):
            mirror[('final',)].add(parts[1])
        elif len(parts) == 3 and parts[0] == 'results' and parts[1] in STAGES and path.endswith('.json'):
            mirror[('results', parts[1])].add(parts[2])
        else:
            (media if path.endswith(MEDIA_SUFFIXES) or '/media/' in path else others).append(path)
    report = {'run_id': record['run_id'], 'output': record['hf_output_uri'],
              'categories': {}, 'missing_in_hf': [], 'extra_in_hf': [],
              'media_like_paths': media, 'other_paths': sorted(others)}
    for key in sorted(source):
        label = key[0] if key[0] == 'final' else 'results/%s' % key[1]
        missing = sorted(source[key] - mirror[key])
        extra = sorted(mirror[key] - source[key])
        report['categories'][label] = {'gcs': len(source[key]), 'hf': len(mirror[key]),
                                       'missing_in_hf': len(missing), 'extra_in_hf': len(extra)}
        report['missing_in_hf'] += ['%s/%s' % (label, name) for name in missing]
        report['extra_in_hf'] += ['%s/%s' % (label, name) for name in extra]
    report['ok'] = not report['missing_in_hf'] and not report['extra_in_hf'] and not media
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--credentials-dir', type=Path, default=REPO / 'credentials')
    parser.add_argument('--run', action='append', default=None,
                        help='Run id from the private registry; repeatable. Default: every registered run.')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args(argv)
    records = private_runs.read_bindings(args.credentials_dir)
    if args.run:
        wanted = set(args.run)
        missing = wanted - {record['run_id'] for record in records}
        if missing:
            raise SystemExit('Unknown run id(s) in the private registry: ' + ', '.join(sorted(missing)))
        records = [record for record in records if record['run_id'] in wanted]
    reports = [audit(args.credentials_dir, record) for record in records]
    text = json.dumps(reports, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text, encoding='utf-8', newline='\n')
    return 0 if all(item['ok'] for item in reports) else 1


if __name__ == '__main__':
    raise SystemExit(main())
