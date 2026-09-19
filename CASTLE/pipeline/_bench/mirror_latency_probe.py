#!/usr/bin/env python3
"""Read-only latency probe: is the GCS->bucket mirror I/O-concurrency bound?

Sequentially downloads the same object set that the mirror copies, then repeats
it with thread pools, so the per-object latency and the scaling factor are
measured rather than assumed. Reads only; writes nothing to any cloud store.
"""
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(r'D:\QLD\CASTLE\pipeline')
sys.path.insert(0, str(REPO / '_test' / 'runtime' / 'Lib' / 'site-packages'))

from google.cloud import storage
from google.oauth2 import service_account

CRED = REPO / 'credentials' / 'my-project-omni-507802-9a6505abf0f7.json'
PROJECT = 'my-project-omni-507802'
BUCKET = 'castle-caption-batch'
PREFIX = 'castle/day1-allie-13-14-18-20-v3/results/'
SMALL = 150
LARGE = 15


def timed(fn, names, workers):
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(fn, names))
    return time.perf_counter() - started


def main():
    info = json.loads(CRED.read_text(encoding='utf-8'))
    cred = service_account.Credentials.from_service_account_info(
        info, scopes=['https://www.googleapis.com/auth/cloud-platform'])
    client = storage.Client(project=PROJECT, credentials=cred)

    listing_start = time.perf_counter()
    blobs = list(client.list_blobs(BUCKET, prefix=PREFIX))
    listing_sec = time.perf_counter() - listing_start
    sizes = sorted((b.size or 0) for b in blobs)
    print('list_blobs: %d objects in %.2f s (bucket-side listing)' % (len(blobs), listing_sec))
    print('  size p50=%.1f KiB p95=%.1f KiB max=%.1f KiB' % (
        sizes[len(sizes) // 2] / 1024, sizes[int(len(sizes) * 0.95)] / 1024, sizes[-1] / 1024))

    small = [b for b in blobs if (b.size or 0) < 8000][:SMALL]
    large = sorted(blobs, key=lambda b: -(b.size or 0))[:LARGE]

    def fetch(blob, sink):
        sink.append(len(blob.download_as_bytes()))

    for label, group in (('small (<8 KiB)', small), ('large (top sizes)', large)):
        print('\n-- %s  n=%d' % (label, len(group)))
        base = None
        for workers in (1, 4, 8, 16, 32):
            if workers > len(group):
                continue
            sink = []
            secs = timed(lambda b: fetch(b, sink), group, workers)
            per = 1000 * secs / len(group)
            if base is None:
                base = secs
            print('   workers=%2d  %6.2f s  %6.1f ms/object  %6.1f obj/s  speedup=%.1fx'
                  % (workers, secs, per, len(group) / secs, base / secs))


if __name__ == '__main__':
    main()
