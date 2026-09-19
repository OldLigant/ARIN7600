#!/usr/bin/env python3
"""Submit CASTLE Batch prepare jobs for the remaining Bjorn and Allie hours.

Both use castle-batch-v3, which carries the raised maxOutputTokens, and export
the measured media-tuning values as CASTLE_* environment defaults.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRETS = Path(r'D:\QLD\out\castle-runs\b20260917-bjorn08\secrets.env')
PROJECT = 'my-project-omni-507802'
CODE = 'castle-batch-v3'

# Allie 15-17 are excluded on purpose: their failure rates (21%, 66%, 85%) put 15
# below the keep threshold and 16/17 were never processed, so the retained set is
# the hours that failed at or above roughly a third of their clips.
JOBS = [
    {'label': 'Bjorn 15-20',
     'name': 'castle-batch-bjorn15-20-v3',
     'run_prefix': 'day1-bjorn-15-20-v3',
     'output': 'day1-bjorn-15-20-batch-v3',
     'sources': [f'main/day1/Bjorn/video/{h}.mp4' for h in ('15', '16', '17', '18', '19', '20')],
     'timeout': '6h'},
    {'label': 'Allie 13,14,18,19,20',
     'name': 'castle-batch-allie-13-14-18-20-v3',
     'run_prefix': 'day1-allie-13-14-18-20-v3',
     'output': 'day1-allie-13-14-18-20-batch-v3',
     'sources': [f'main/day1/Allie/video/{h}.mp4' for h in ('13', '14', '18', '19', '20')],
     'timeout': '6h'},
]


def main():
    execute = '--execute' in sys.argv
    raw = SECRETS.read_text(encoding='utf-8').strip()
    key, _, value = raw.partition('=')
    os.environ[key] = value
    results = []
    for job in JOBS:
        command = [sys.executable, str(ROOT / 'batch_jobs.py'), 'submit',
                   '--code-volume', f'hf://buckets/Ligant/castle-code/{CODE}',
                   '--output-volume', f'hf://buckets/Ligant/castle-output/{job["output"]}',
                   '--name', job['name'],
                   '--project', PROJECT,
                   '--state-uri', f'gs://castle-caption-batch/castle/{job["run_prefix"]}/state.json',
                   '--credential-secret', 'GOOGLE_ADC_JSON',
                   '--flavor', 'cpu-upgrade', '--timeout', job['timeout']]
        if execute:
            command.append('--execute')
        command += ['--', '--gcs-prefix', f'gs://castle-caption-batch/castle/{job["run_prefix"]}']
        for source in job['sources']:
            command += ['--source', source]
        command += ['--model', 'gemini-3.8-flash', '--max-clips', '120',
                    '--media-threads', '4', '--decode-slots', '3',
                    '--prepare-workers', '3', '--footer-workers', '4']
        proc = subprocess.run(command, capture_output=True, text=True, cwd=str(ROOT))
        results.append({'label': job['label'], 'sources': len(job['sources']),
                        'expected_clips': 120 * len(job['sources']), 'timeout': job['timeout'],
                        'exit': proc.returncode, 'stdout_tail': proc.stdout.strip()[-500:],
                        'stderr_tail': proc.stderr.strip()[-200:]})
    print(json.dumps({'executed': execute, 'results': results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
