#!/usr/bin/env python3
"""Classify why annotation rows failed to parse, without printing model text.

Distinguishes truncation (output hit the token cap) from other malformed output,
using only length, tail characters and JSON error position.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TARGETS = {
    '08 annotation': 'castle/day1-bjorn-08-v1/batch-output/annotation-f83fb98cfd1366df/'
                     'prediction-model-2026-09-17T03:32:28.265067Z/predictions.jsonl',
    '09 annotation': 'castle/day1-bjorn-09-v2/batch-output/annotation-badb724113b635b2/'
                     'prediction-model-2026-09-17T03:32:25.401891Z/predictions.jsonl',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--credentials', type=Path,
                        default=Path(r'D:\QLD\out\castle-runs\b20260917-bjorn08\secrets.env'))
    args = parser.parse_args()

    raw = args.credentials.read_text(encoding='utf-8').strip()
    info = json.loads(raw.split('=', 1)[1] if raw.startswith('GOOGLE_ADC_JSON=') else raw)
    from google.oauth2 import service_account
    from google.cloud import storage
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=['https://www.googleapis.com/auth/cloud-platform'])
    client = storage.Client(project=info['project_id'], credentials=credentials)

    for label, name in TARGETS.items():
        blob = client.bucket('castle-caption-batch').blob(name)
        lengths = []
        bad = []
        with blob.open('rt', encoding='utf-8') as stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    continue
                row = json.loads(line)
                response = row.get('response') or {}
                candidates = response.get('candidates') or []
                parts = ((candidates[0].get('content') or {}).get('parts') or []) if candidates else []
                text = ''.join(p.get('text', '') for p in parts
                               if isinstance(p, dict) and not p.get('thought'))
                finish = candidates[0].get('finishReason') if candidates else None
                output_tokens = (response.get('usageMetadata') or {}).get('candidatesTokenCount')
                if not text.strip():
                    bad.append({'row': index, 'why': 'no_text', 'finish': finish,
                                'out_tokens': output_tokens, 'len': 0, 'tail': ''})
                    continue
                lengths.append(len(text))
                try:
                    json.loads(text)
                except ValueError as error:
                    bad.append({'row': index, 'why': 'bad_json', 'finish': finish,
                                'out_tokens': output_tokens, 'len': len(text),
                                'error': str(error)[:70],
                                'tail': text[-25:].replace('\n', '\\n'),
                                'head_ok': text.lstrip()[:1]})
        # A truncation signature: JSON error at the very end of a long document.
        signatures = Counter()
        for item in bad:
            if item['why'] == 'bad_json':
                at_end = 'end of document' in item.get('error', '') or 'Unterminated' in item.get('error', '')
                signatures['truncated_at_end' if at_end else 'malformed_elsewhere'] += 1
            else:
                signatures[item['why']] += 1
        print(json.dumps({
            'stage': label,
            'rows': len(lengths) + len(bad),
            'ok': len(lengths),
            'bad': len(bad),
            'bad_signatures': dict(signatures),
            'ok_text_len_min': min(lengths) if lengths else None,
            'ok_text_len_max': max(lengths) if lengths else None,
            'bad_detail': bad[:8],
        }, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
