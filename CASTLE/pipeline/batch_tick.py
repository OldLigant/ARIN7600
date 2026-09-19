#!/usr/bin/env python3
"""Run the pinned tick, then mirror collected stage JSON into the HF output mount.

This auxiliary entrypoint deliberately lives outside castle_pipeline/: copying
artifacts must not change the code identity of existing runs. GCS remains the
source of truth; the mirror is never used as a checkpoint or submission input.

The mirror threads its per-object round trips (CASTLE_MIRROR_WORKERS, default 16)
because the work is latency-bound, not bandwidth-bound. Only the walk stays
sequential; the copied bytes, the validation and the failure reported do not.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from castle_pipeline.batch_cloud import BatchCloud, parse_gs_uri


class MirrorConflict(ValueError):
    """Safe diagnostic fields: no result data or provider error payloads."""

    def __init__(self, relative):
        super().__init__('Stage mirror already has different content: ' + relative)
        self.relative = relative


MIRROR_RELATIVE = re.compile(r'(audio|annotation|review)/[A-Za-z0-9_-]+\.json')
# Latency-bound work with a cgroup-blind default, like the other run knobs: the
# measured scaling was 4.0x at 4 threads and 14.3x at 16 on a 2 vCPU machine, so
# the default stays well inside what a cpu-basic container can drive and every
# run may override it. It is a run parameter, never an identity input.
DEFAULT_MIRROR_WORKERS = 16
MAX_MIRROR_WORKERS = 64


def env_mirror_workers():
    """Mirror concurrency from CASTLE_MIRROR_WORKERS, else the modest default."""
    raw = os.environ.get('CASTLE_MIRROR_WORKERS')
    if raw is None or not raw.strip():
        return DEFAULT_MIRROR_WORKERS
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError('Environment variable CASTLE_MIRROR_WORKERS must be a positive integer') from None
    if not 1 <= value <= MAX_MIRROR_WORKERS:
        raise ValueError('Environment variable CASTLE_MIRROR_WORKERS must be in 1..%d' % MAX_MIRROR_WORKERS)
    return value


def _mirror_object(blob, prefix, output):
    """Mirror one GCS object. Returns 'copied', 'unchanged', or None if not ours.

    Runs on a mirror thread, so it only touches its own destination path. The
    validation, hash and conflict rules are unchanged from the sequential walk.
    """
    relative = blob.name[len(prefix):]
    if not MIRROR_RELATIVE.fullmatch(relative):
        return None
    target = output / 'results' / relative
    expected = (blob.metadata or {}).get('sha256')
    existing = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
    if expected and existing == expected:
        return 'unchanged'
    if expected and existing is not None:
        raise MirrorConflict(relative)
    data = blob.download_as_bytes(if_generation_match=int(blob.generation))
    actual = hashlib.sha256(data).hexdigest()
    if expected and actual != expected:
        raise ValueError('GCS stage content hash mismatch: ' + relative)
    if not isinstance(json.loads(data), dict):
        raise ValueError('Stage result must be a JSON object: ' + relative)
    if existing is not None:
        if existing != actual:
            raise MirrorConflict(relative)
        return 'unchanged'
    target.parent.mkdir(parents=True, exist_ok=True)
    # Publish only complete bytes. This temp file belongs to this invocation.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.stage-', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        # Catch a destination populated during the download. This is not a
        # cross-mount lock: each output prefix must still have one writer.
        if target.is_file():
            if hashlib.sha256(target.read_bytes()).hexdigest() != actual:
                raise MirrorConflict(relative)
            return 'unchanged'
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return 'copied'


def mirror_results(cloud, gcs_prefix, output_dir, workers=None):
    """Copy collected stage results into the HF output mount, with bounded concurrency.

    Mirroring is latency-bound, not bandwidth-bound: the median object is 4.9 KiB
    but still costs one or two cross-store round trips, and the walk was strictly
    sequential (measured in-container: 10-13 objects/s, 1.76 Mbps, ~2 min for
    ~1200 objects). A read-only probe of the same object set scaled 14x at 16
    threads, so only the per-object work is threaded here. The walk, the counts
    and the failure reported stay independent of scheduling.
    """
    bucket, run = parse_gs_uri(gcs_prefix)
    prefix = run.rstrip('/') + '/results/'
    output = Path(output_dir)
    total = env_mirror_workers() if workers is None else workers
    if isinstance(total, bool) or not isinstance(total, int) or not 1 <= total <= MAX_MIRROR_WORKERS:
        raise ValueError('Mirror workers must be an integer in 1..%d' % MAX_MIRROR_WORKERS)
    targets = [blob for blob in cloud.storage.list_blobs(bucket, prefix=prefix)
               if blob.name.startswith(prefix)]
    counts = {'copied': 0, 'unchanged': 0}
    failures = []
    if targets:
        with ThreadPoolExecutor(max_workers=min(total, len(targets)), thread_name_prefix='mirror') as pool:
            pending = {pool.submit(_mirror_object, blob, prefix, output): blob.name for blob in targets}
            for future in as_completed(pending):
                try:
                    outcome = future.result()
                except Exception as error:  # re-raised deterministically below
                    failures.append((pending[future], error))
                else:
                    if outcome is not None:
                        counts[outcome] += 1
    if failures:
        # The same object set must always report the same failure, whatever order
        # the threads happened to finish in.
        failures.sort(key=lambda item: item[0])
        raise failures[0][1]
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['tick'])
    parser.add_argument('--project', required=True)
    parser.add_argument('--location', default='global')
    parser.add_argument('--state-uri', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--allow-missing-state', action='store_true')
    args = parser.parse_args(argv)
    parse_gs_uri(args.state_uri)
    cloud = BatchCloud(args.project, args.location)
    state, _ = cloud.read_state(args.state_uri)
    if state is None:
        if not args.allow_missing_state:
            raise ValueError('No batch state exists')
        print(json.dumps({'event': 'batch_tick_waiting', 'status': 'waiting_for_state',
                          'state_uri': args.state_uri, 'stop_schedule': False}), flush=True)
        return 0
    config = state['config']
    if config['project'] != args.project or config['location'] != args.location:
        raise ValueError('Project/location must match the saved run')
    if args.state_uri != config['gcs_prefix'].rstrip('/') + '/state.json':
        raise ValueError('State URI must belong to the saved GCS run prefix')
    from batch_pipeline import code_hash
    if config.get('code_hash') != code_hash():
        raise ValueError('Use the pinned code version that initialized this run')

    command = [sys.executable, str(Path(__file__).with_name('batch_pipeline.py')), 'tick',
               '--project', args.project, '--location', args.location, '--state-uri', args.state_uri,
               '--output-dir', str(args.output_dir), '--scratch-dir', str(args.scratch_dir)]
    result = subprocess.run(command, check=False)
    # Even a failed tick can have persisted valid stage results before failing.
    # Replaying this mirror never creates model requests or alters GCS state.
    try:
        workers = env_mirror_workers()
        counts = mirror_results(cloud, config['gcs_prefix'], args.output_dir, workers=workers)
    except Exception as error:
        details = {'file': error.relative, 'reason': 'different_content'} if isinstance(error, MirrorConflict) else {}
        print(json.dumps({'event': 'batch_stage_mirror_failed', 'error_type': type(error).__name__,
                          'message': 'Stage mirror failed; GCS results are preserved. Inspect and retry.', **details}),
              file=sys.stderr, flush=True)
        return result.returncode or 1
    print(json.dumps({'event': 'batch_stage_mirror', 'workers': workers, **counts}), flush=True)
    return result.returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(json.dumps({'ok': False, 'error_type': type(error).__name__,
                          'message': 'Batch tick wrapper failed; inspect configuration and persisted state.'}),
              file=sys.stderr)
        raise SystemExit(1)
