#!/usr/bin/env python3
"""Probe which maxOutputTokens values the Batch endpoint accepts for a model.

Submits one tiny batch job per candidate limit, each with a single trivial
request. A rejected value fails at create time; an accepted value returns a job
name. Costs one trivial request per accepted value.

Usage: python _bench/probe_max_tokens.py --model gemini-3.8-flash --limits 16384 32768 65536
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--credentials', type=Path,
                        default=Path(r'D:\QLD\out\castle-runs\b20260917-bjorn08\secrets.env'))
    parser.add_argument('--project', default='my-project-omni-507802')
    parser.add_argument('--location', default='global')
    parser.add_argument('--model', default='gemini-3.8-flash')
    parser.add_argument('--prefix', default='gs://castle-caption-batch/castle/_probe-max-tokens')
    parser.add_argument('--limits', type=int, nargs='+', default=[32768])
    args = parser.parse_args()

    raw = args.credentials.read_text(encoding='utf-8').strip()
    info = json.loads(raw.split('=', 1)[1] if raw.startswith('GOOGLE_ADC_JSON=') else raw)
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=['https://www.googleapis.com/auth/cloud-platform'])

    from castle_pipeline.batch_cloud import BatchCloud
    cloud = BatchCloud(args.project, args.location, credentials=credentials)

    results = []
    for limit in args.limits:
        row = {'request': {
            'contents': [{'role': 'user', 'parts': [{'text': 'Reply with exactly {"ok":true} and nothing else.'}]}],
            'generationConfig': {'responseMimeType': 'application/json', 'maxOutputTokens': limit}}}
        input_uri = f'{args.prefix}/input-{limit}.jsonl'
        output_uri = f'{args.prefix}/output-{limit}/'
        with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False, encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            local = Path(handle.name)
        try:
            cloud.upload(local, input_uri)
            job = cloud.create_batch(args.model, input_uri, output_uri, f'castle-maxtok-probe-{limit}')
            results.append({'limit': limit, 'accepted': True, 'job': job.get('name'), 'state': job.get('state')})
        except Exception as error:
            results.append({'limit': limit, 'accepted': False,
                            'error_type': type(error).__name__, 'error': str(error)[:220],
                            'code': getattr(error, 'code', None)})
        finally:
            local.unlink(missing_ok=True)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
