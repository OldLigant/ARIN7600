#!/usr/bin/env python3
"""Call batch_pipeline.main directly so the real exception surfaces.

batch_pipeline's __main__ guard intentionally prints a normalized error, which
hides the cause during debugging. This wrapper keeps that behaviour intact in
production while exposing the traceback for local diagnosis.
"""
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

secrets = Path(r'D:\QLD\out\castle-runs\b20260917-bjorn08\secrets.env')
raw = secrets.read_text(encoding='utf-8').strip()
key, _, value = raw.partition('=')
os.environ[key] = value
adc = Path(r'D:\QLD\out\castle-runs\b20260917-bjorn08\adc-tmp.json')
adc.write_text(value, encoding='utf-8')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(adc)

import batch_pipeline

argv = sys.argv[1:]
print('argv:', ' '.join(argv), flush=True)
try:
    code = batch_pipeline.main(argv)
    print('main returned', code)
except BaseException:
    traceback.print_exc()
    sys.exit(9)
