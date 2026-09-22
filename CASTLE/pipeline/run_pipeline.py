#!/usr/bin/env python3
"""CASTLE -> audio first -> dense annotation -> optional native-detail review."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

# Bound Xet cache and CPU threading before importing download/SDK libraries.
os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES', '0')
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', '0')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from castle_pipeline.inputs import REPO, REVISION, source_metadata, select_sources, list_remote, download_source, remove_downloaded_source
from castle_pipeline.media import env_decoder_default, env_footer_default
from castle_pipeline.runner import Pipeline, RunConfig, atomic_json, memory_estimate
from castle_pipeline.vertex import VertexProvider
from castle_pipeline.events import ProgressReporter


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list', help='Read HF inventory into an explicit JSONL manifest')
    listing.add_argument('--output', type=Path, required=True)
    listing.add_argument('--revision', default=REVISION)
    listing.add_argument('--day', choices=['day1', 'day2', 'day3', 'day4'])
    listing.add_argument('--stream')
    estimate = commands.add_parser('estimate', help='Theoretical memory/call budget, no network')
    estimate.add_argument('--source-seconds', type=float, default=3600)
    estimate.add_argument('--workers', type=int, default=3)
    estimate.add_argument('--fps', type=float, default=1)
    estimate.add_argument('--clip-seconds', type=float, default=30)
    estimate.add_argument('--max-dim', type=int, default=1440)
    status = commands.add_parser('status', help='Summarize local or mounted checkpoints')
    status.add_argument('--output-dir', type=Path, required=True)
    run = commands.add_parser('run', help='Run selected media; this makes paid Vertex API calls')
    sources = run.add_mutually_exclusive_group(required=True)
    sources.add_argument('--local-video', type=Path)
    sources.add_argument('--source', action='append', help='Exact HF path, repeatable')
    sources.add_argument('--manifest', type=Path)
    run.add_argument('--revision', help='Default: manifest revision, or investigated pinned commit for explicit source paths')
    run.add_argument('--day', choices=['day1', 'day2', 'day3', 'day4'])
    run.add_argument('--stream')
    run.add_argument('--hour', type=int, choices=range(24))
    run.add_argument('--viewpoint', choices=['ego', 'exo', 'all'], default='ego')
    run.add_argument('--shard-index', type=int, default=0)
    run.add_argument('--shard-count', type=int, default=1)
    run.add_argument('--output-dir', type=Path, required=True)
    run.add_argument('--scratch-dir', type=Path, required=True)
    run.add_argument('--model', required=True, help='Explicit deployed Vertex model ID; no guessed default')
    run.add_argument('--project', default=os.environ.get('GOOGLE_CLOUD_PROJECT', ''))
    run.add_argument('--location', default=os.environ.get('GOOGLE_CLOUD_LOCATION', 'global'))
    run.add_argument('--service-tier', choices=['standard', 'flex'], default='standard')
    run.add_argument('--workers', type=int, default=3)
    run.add_argument('--initial-concurrency', type=int, default=2)
    run.add_argument('--rpm', type=float, default=30)
    run.add_argument('--attempts', type=int, default=3)
    run.add_argument('--timeout-sec', type=float, default=600)
    run.add_argument('--fps', type=float, default=1)
    run.add_argument('--clip-seconds', type=float, default=30)
    run.add_argument('--max-dim', type=int, default=1440)
    run.add_argument('--start-clip', type=int, default=0)
    run.add_argument('--max-clips', type=int, help='Cap clips per source, not a global cap')
    run.add_argument('--no-review', action='store_true')
    run.add_argument('--max-review-regions', type=int, default=4)
    run.add_argument('--max-output-tokens', type=int, default=32768,
                     help='Generation budget shared by thinking and visible output; default 32768, '
                          'matching Vertex Batch so smoke results predict batch behaviour')
    run.add_argument('--no-stamp', action='store_true')
    run.add_argument('--keep-media', action='store_true', help='Retain sampled frames/crops for inspection; consumes scratch disk')
    run.add_argument('--keep-source', action='store_true', help='Retain managed downloaded source; check disk before long runs')
    run.add_argument('--memory-soft-limit-gib', type=float, default=12)
    run.add_argument('--media-threads', type=int, default=None,
                     help='FFmpeg decoder/filter threads per invocation; default: CASTLE_MEDIA_THREADS or CPU-derived')
    run.add_argument('--decode-slots', type=int, default=1,
                     help='Clips allowed to decode concurrently; raise only with RAM to cover threads*slots')
    run.add_argument('--footer-workers', type=int, default=None,
                     help='Threads stamping the timestamp footer; default: CASTLE_FOOTER_WORKERS or CPU-derived')
    run.add_argument('--progress-interval-sec', type=float, default=600, help='Periodic progress heartbeat, including blocked requests')
    return root


def status(directory):
    counts = {'completed_clips': 0, 'failed_clips_without_final': 0, 'review_required': 0, 'corrupt_final_files': 0}
    for file in directory.rglob('final.json'):
        try:
            record = json.loads(file.read_text(encoding='utf-8'))
            if record.get('ok') and 'annotation' in record:
                counts['completed_clips'] += 1
                counts['review_required'] += bool(record.get('review_required'))
            else:
                counts['corrupt_final_files'] += 1
        except (OSError, ValueError):
            counts['corrupt_final_files'] += 1
    counts['failed_clips_without_final'] = sum(not (p.parent/'final.json').exists() for p in directory.rglob('error.json'))
    return counts


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'estimate':
        result = memory_estimate(args.source_seconds, args.workers, fps=args.fps,
                                 clip_seconds=args.clip_seconds, max_dim=args.max_dim)
        import math
        count = math.ceil(args.source_seconds / args.clip_seconds)
        result.update(clips=count, audio_plus_visual_calls=count*2, maximum_calls_with_review=count*3,
                      retry_calls_excluded=True)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == 'status':
        print(json.dumps(status(args.output_dir), indent=2))
        return 0
    if args.command == 'list':
        rows, revision = list_remote(args.revision, args.day, args.stream)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects an existing manifest from accidental replacement.
        with args.output.open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row) + '\n')
        print(json.dumps({'revision': revision, 'videos': len(rows), 'manifest': str(args.output)}))
        return 0
    config = RunConfig(args.output_dir, args.scratch_dir, workers=args.workers, fps=args.fps,
                       clip_seconds=args.clip_seconds, max_dim=args.max_dim, review=not args.no_review,
                       max_review_regions=args.max_review_regions, stamp=not args.no_stamp,
                       max_output_tokens=args.max_output_tokens,
                       max_clips=args.max_clips, start_clip=args.start_clip,
                       memory_soft_limit_gib=args.memory_soft_limit_gib, keep_media=args.keep_media,
                       progress_interval_sec=args.progress_interval_sec,
                       media_threads=args.media_threads, decode_slots=args.decode_slots,
                       footer_workers=args.footer_workers)
    provider = VertexProvider(args.model, args.project, location=args.location, service_tier=args.service_tier,
                              initial_concurrency=min(args.initial_concurrency, args.workers), max_concurrency=args.workers,
                              rpm=args.rpm, attempts=args.attempts, timeout_sec=args.timeout_sec)
    pipeline = Pipeline(config, provider)
    if args.local_video:
        source = args.local_video.resolve(strict=True)
        info = source.stat()
        metadata = {'source_id': str(source), 'file_size': info.st_size, 'mtime_ns': info.st_mtime_ns,
                    'day': args.day, 'stream': args.stream, 'hour': args.hour,
                    'viewpoint': 'exocentric' if args.viewpoint == 'exo' else 'egocentric'}
        result = pipeline.run_source(source, metadata)
        print(json.dumps(result, ensure_ascii=False))
        return 1 if result['failed'] or result['unprocessed'] else 0
    from huggingface_hub import HfApi
    rows = select_sources(args.manifest, args.day, args.stream, args.viewpoint,
                          args.shard_index, args.shard_count, hour=args.hour) if args.manifest else [{'path': p} for p in args.source]
    if not rows:
        raise ValueError('Source selection is empty')
    if not args.manifest and (args.shard_index != 0 or args.shard_count != 1):
        raise ValueError('Sharding requires a manifest; exact source paths already define the run scope')
    if not args.manifest:
        for row in rows:
            meta = source_metadata(row['path'], REVISION)
            if any(value is not None and value != meta[key] for key, value in [('day', args.day), ('stream', args.stream), ('hour', args.hour)]):
                raise ValueError('Exact source path conflicts with the selected day/stream/hour')
    recorded = {row['revision'] for row in rows if row.get('revision')}
    if len(recorded) > 1:
        raise ValueError('A run must use one immutable dataset revision')
    # Resolve aliases once; never silently override a recorded manifest revision.
    wanted_revision = args.revision or next(iter(recorded), REVISION)
    revision = HfApi().dataset_info(REPO, revision=wanted_revision).sha
    if recorded and recorded != {revision}:
        raise ValueError('CLI dataset revision disagrees with manifest revision')
    failed = False
    for row in rows:
        metadata = source_metadata(row['path'], revision)
        started = time.monotonic()
        pipeline.events.emit('download_start', source=metadata['source_id'])
        with ProgressReporter(pipeline.events, args.progress_interval_sec,
                              lambda: {'source': metadata['source_id'], 'phase': 'download',
                                       'elapsed_sec': round(time.monotonic() - started, 1)}):
            try:
                local = download_source(row['path'], revision, args.scratch_dir)
            except Exception as error:
                pipeline.events.emit('download_failed', source=metadata['source_id'], error_type=type(error).__name__)
                raise
        pipeline.events.emit('download_success', source=metadata['source_id'],
                             elapsed_sec=round(time.monotonic() - started, 3), bytes=local.stat().st_size)
        try:
            result = pipeline.run_source(local, metadata)
            failed |= bool(result['failed'] or result['unprocessed'])
            print(json.dumps(result, ensure_ascii=False))
            if result.get('fatal'):
                break
        finally:
            if not args.keep_source:
                remove_downloaded_source(local, args.scratch_dir)
    return int(failed)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        # Never echo credentials, signed download URLs, or raw SDK responses.
        print(json.dumps({'ok': False, 'error_type': type(error).__name__,
                          'category': getattr(error, 'category', 'configuration_or_io')}), file=sys.stderr)
        raise SystemExit(1)
