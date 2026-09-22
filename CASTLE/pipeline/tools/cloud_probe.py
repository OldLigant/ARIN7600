#!/usr/bin/env python3
"""Read-only probe of CASTLE cloud state (docs/release-process.md R-09).

Facts about the cloud belong in generated output, not in hand-written prose that
rots within the hour. This module reads the persisted GCS state and the live
Vertex Batch jobs and returns a plain dict; ``tools/report.py`` renders it, and
nothing here writes.

Read-only by construction: it only calls state reads and batch lookups. It never
creates, cancels or collects anything, and it prints no model output or
credential material.

Usage:
  python tools/cloud_probe.py --credentials credentials/<sa-key>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Every Batch run prefix under gs://castle-caption-batch/castle/.
RUNS = {
    'day1-bjorn-08-v1': 'castle/day1-bjorn-08-v1',
    'day1-bjorn-09-v2': 'castle/day1-bjorn-09-v2',
    'day1-bjorn-10-14-v2': 'castle/day1-bjorn-10-14-v2',
    'day1-allie-13-14-18-20-v3': 'castle/day1-allie-13-14-18-20-v3',
    'day1-bjorn-15-20-v3': 'castle/day1-bjorn-15-20-v3',
    'day1-florian-08-09-v6': 'castle/day1-florian-08-09-v6',
}
PROJECT = 'my-project-omni-507802'
LOCATION = 'global'
BUCKET = 'castle-caption-batch'
DEFAULT_CREDENTIALS = REPO / 'credentials' / 'my-project-omni-507802-9a6505abf0f7.json'


def load_credentials(path: Path):
    """Accept a service-account JSON, or a KEY=<json> secrets file."""
    raw = path.read_text(encoding='utf-8').strip()
    if not raw.startswith('{'):
        _, _, raw = raw.partition('=')
        raw = raw.strip()
    info = json.loads(raw)
    from google.oauth2 import service_account
    return info, service_account.Credentials.from_service_account_info(
        info, scopes=['https://www.googleapis.com/auth/cloud-platform'])


def state_summary(state: dict) -> dict:
    config = state.get('config') or {}
    batches = {}
    for stage, record in (state.get('batches') or {}).items():
        job = record.get('job') or {}
        batches[stage] = {'recorded_state': job.get('state'), 'job': job.get('name'),
                          'output_uri': job.get('output_uri')}
    return {
        'status': state.get('status'),
        'current_stage': state.get('current_stage'),
        'selected': len(config.get('rows') or []),
        'completed': len(state.get('completed') or []),
        'failures': len(state.get('failures') or []),
        'eligible_next': len(state.get('eligible_rows') or []),
        'release': config.get('release'),
        'code_hash': config.get('code_hash'),
        'model': config.get('model'),
        'review': config.get('review'),
        'max_output_tokens': config.get('max_output_tokens'),
        'media_tuning': {k: config.get(k) for k in
                         ('media_threads', 'decode_slots', 'footer_workers', 'prepare_workers')},
        'attention_reason': state.get('attention_reason'),
        'last_error': state.get('last_error'),
        'batches': batches,
    }


def _access_token(credentials) -> str | None:
    try:
        import google.auth.transport.requests
        credentials.refresh(google.auth.transport.requests.Request())
        return credentials.token
    except Exception:
        return None


def batch_timing(job_name: str, token: str | None) -> dict:
    """Measured queue and run time for one batch job, direct from the REST API.

    Vertex does return createTime/startTime/endTime/completionStats, but the SDK's
    object mapping drops them, so they are read over REST. Timing is diagnostic:
    an unavailable lookup degrades to {} rather than failing the whole probe.
    """
    if not token or not job_name:
        return {}
    import urllib.request
    from datetime import datetime
    job_id = job_name.rsplit('/', 1)[-1]
    url = (f'https://aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{LOCATION}/'
           f'batchPredictionJobs/{job_id}')

    def seconds(start, end):
        if not start or not end:
            return None
        parse = lambda v: datetime.fromisoformat(v.replace('Z', '+00:00'))
        return round((parse(end) - parse(start)).total_seconds(), 1)

    try:
        request = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except Exception:
        return {}
    stats = payload.get('completionStats') or {}
    return {
        'create_time': payload.get('createTime'),
        'start_time': payload.get('startTime'),
        'end_time': payload.get('endTime'),
        'queued_sec': seconds(payload.get('createTime'), payload.get('startTime')),
        'running_sec': seconds(payload.get('startTime'), payload.get('endTime')),
        'successful_count': int(stats.get('successfulCount') or 0),
        'failed_count': int(stats.get('failedCount') or 0),
    }


def probe(credentials_path: Path = DEFAULT_CREDENTIALS) -> dict:
    info, credentials = load_credentials(credentials_path)
    from google.cloud import storage
    client = storage.Client(project=info.get('project_id', PROJECT), credentials=credentials)
    from castle_pipeline.batch_cloud import BatchCloud
    cloud = BatchCloud(PROJECT, LOCATION, credentials=credentials)
    token = _access_token(credentials)

    report = {'probed_utc': None, 'project': PROJECT, 'bucket': BUCKET, 'runs': {}}
    from datetime import datetime, timezone
    report['probed_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    for label, prefix in RUNS.items():
        uri = f'gs://{BUCKET}/{prefix}/state.json'
        entry = {'state_uri': uri}
        try:
            state, generation = cloud.read_state(uri)
        except Exception as error:
            entry['error'] = f'{type(error).__name__}'
            report['runs'][label] = entry
            continue
        if state is None:
            entry['state'] = 'not initialised (media still being prepared)'
            report['runs'][label] = entry
            continue
        entry['generation'] = generation
        entry['summary'] = state_summary(state)
        live = {}
        for stage, record in (state.get('batches') or {}).items():
            name = (record.get('job') or {}).get('name')
            if not name:
                continue
            try:
                live[stage] = cloud.get_batch(name)
            except Exception as error:
                live[stage] = {'error': f'{type(error).__name__}'}
            timing = batch_timing(name, token)
            if timing:
                live.setdefault(stage, {}).update(timing)
        entry['live_vertex'] = live
        report['runs'][label] = entry
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--credentials', type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    report = probe(args.credentials)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text, encoding='utf-8', newline='\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
