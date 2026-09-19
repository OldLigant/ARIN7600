#!/usr/bin/env python3
"""Count how many clips in a finished annotation batch request review crops.

Parses each annotation response's review_regions (a cheap, schema-agnostic scan)
to size the review stage. Prints counts only.
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
        usable = with_crops = no_crops = bad = 0
        crop_counts = Counter()
        with blob.open('rt', encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if 'response' not in row:          # echoed request row
                    continue
                response = row.get('response') or {}
                candidates = response.get('candidates') or []
                if not candidates or candidates[0].get('finishReason') != 'STOP':
                    bad += 1
                    continue
                parts = ((candidates[0].get('content') or {}).get('parts') or [])
                text = ''.join(p.get('text', '') for p in parts
                               if isinstance(p, dict) and not p.get('thought'))
                try:
                    data = json.loads(text)
                except ValueError:
                    bad += 1
                    continue
                usable += 1
                regions = data.get('review_regions')
                count = len(regions) if isinstance(regions, list) else 0
                if count:
                    with_crops += 1
                    crop_counts[min(count, 8)] += 1
                else:
                    no_crops += 1
        print(json.dumps({
            'stage': label,
            'annotation_usable': usable,
            'bad': bad,
            'clips_needing_review': with_crops,
            'clips_final_without_review': no_crops,
            'total_crops': sum(k * v for k, v in crop_counts.items()),
            'crops_per_clip_distribution': dict(sorted(crop_counts.items())),
        }, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
