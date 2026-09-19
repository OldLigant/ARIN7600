#!/usr/bin/env python3
"""Submit one-off tick jobs. Each run must use the code version that initialised it.

Run 08 -> v1, run 09 and 10-14 -> v2 (their stored code_hash admits only that
version), Bjorn 15-20 -> v3 (carries the raised maxOutputTokens).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRETS = Path(r'D:\QLD\out\castle-runs\b20260917-bjorn08\secrets.env')

RUNS = [
    {'label': '10-14 audio->annotation', 'code': 'castle-batch-v2', 'role': 'tick',
     'output': 'hf://buckets/Ligant/castle-output/day1-bjorn-10-14-batch-v2',
     'state': 'gs://castle-caption-batch/castle/day1-bjorn-10-14-v2/state.json',
     'name': 'castle-tick-bjorn10-14-annotate-v2'},
]


def main():
    execute = '--execute' in sys.argv
    raw = SECRETS.read_text(encoding='utf-8').strip()
    key, _, value = raw.partition('=')
    os.environ[key] = value
    results = []
    for run in RUNS:
        command = [sys.executable, str(ROOT / 'batch_jobs.py'), run.get('role', 'tick'),
                   '--code-volume', f'hf://buckets/Ligant/castle-code/{run["code"]}',
                   '--output-volume', run['output'],
                   '--name', run['name'],
                   '--project', 'my-project-omni-507802',
                   '--state-uri', run['state'],
                   '--credential-secret', 'GOOGLE_ADC_JSON']
        if run.get('flavor'):
            command += ['--flavor', run['flavor'], '--timeout', run['timeout']]
        if execute:
            command.append('--execute')
        if run.get('pipeline_args'):
            command += ['--'] + run['pipeline_args']
        proc = subprocess.run(command, capture_output=True, text=True, cwd=str(ROOT))
        results.append({'label': run['label'], 'code': run['code'], 'exit': proc.returncode,
                        'stdout_tail': proc.stdout.strip()[-800:],
                        'stderr_tail': proc.stderr.strip()[-300:]})
    print(json.dumps({'executed': execute, 'results': results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
