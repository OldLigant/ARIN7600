#!/usr/bin/env python3
"""Render a reproducible HF Jobs command; submit only with explicit --submit."""
import argparse
import json
import re
import shlex
import subprocess


def build_command(*, code_volume, output_volume, name, pipeline_args, secrets,
                  project='', image='python:3.12-slim', flavor='cpu-basic', timeout='3h',
                  media_threads=None, footer_workers=None):
    if not code_volume.startswith('hf://buckets/') or not output_volume.startswith('hf://buckets/'):
        raise ValueError('Use explicit HF bucket URLs for code and persistent output')
    if any(':' in value[5:] or '\n' in value or '..' in value.split('/') for value in (code_volume, output_volume)):
        raise ValueError('Invalid volume URL')
    if code_volume.rstrip('/') == output_volume.rstrip('/'):
        raise ValueError('Code and output volumes must be distinct')
    if any(v.split('=')[0] in {'--output-dir', '--scratch-dir'} for v in pipeline_args):
        raise ValueError('Jobs output and scratch are managed by this launcher')
    if any(not re.fullmatch(r'[A-Z][A-Z0-9_]*', key) for key in secrets):
        raise ValueError('Secrets must be environment variable names, never literal values')
    for value, label in ((media_threads, 'media_threads'), (footer_workers, 'footer_workers')):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f'{label} must be a positive integer when given')
    command = ['hf', 'jobs', 'run', '--detach', '--flavor', flavor, '--timeout', timeout, '--name', name,
               '--volume', code_volume + ':/workspace:ro', '--volume', output_volume + ':/output',
               '--env', 'PYTHONUNBUFFERED=1', '--env', 'PYTHONDONTWRITEBYTECODE=1',
               '--env', 'HF_HOME=/scratch/hf', '--env', 'HF_XET_CHUNK_CACHE_SIZE_BYTES=0',
               '--env', 'HF_XET_HIGH_PERFORMANCE=0', '--env', 'OMP_NUM_THREADS=1']
    # The media knobs are exported as the environment defaults the pipeline reads,
    # so a run stays reproducible from the rendered argv even if the container
    # cannot see its own cgroup CPU quota.
    if media_threads is not None:
        command += ['--env', f'CASTLE_MEDIA_THREADS={media_threads}']
    if footer_workers is not None:
        command += ['--env', f'CASTLE_FOOTER_WORKERS={footer_workers}']
    if project:
        command += ['--env', 'GOOGLE_CLOUD_PROJECT=' + project]
    for key in secrets:
        command += ['--secrets', key]
    run = ['python', '/workspace/run_pipeline.py', 'run', *pipeline_args,
           '--output-dir', '/output', '--scratch-dir', '/scratch/castle']
    bootstrap = ('set -eu; mkdir -p /scratch/castle; '
                 'if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then '
                 'apt-get update && apt-get install -y --no-install-recommends ffmpeg; fi; '
                 'python -m pip install --no-cache-dir -r /workspace/requirements.txt; '
                 'exec ' + shlex.join(run))
    # Terminate option parsing before the image. Without this separator the HF
    # Jobs CLI consumes the dash-leading '-lc' as its own option, so the
    # container records command ["bash", "<bootstrap>"] and bash exits 127
    # trying to run the bootstrap text as a filename.
    return command + ['--', image, 'bash', '-lc', bootstrap]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-volume', required=True)
    p.add_argument('--output-volume', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--project', default='')
    p.add_argument('--secret', action='append', default=[])
    p.add_argument('--image', default='python:3.12-slim')
    p.add_argument('--flavor', default='cpu-basic')
    p.add_argument('--timeout', default='3h')
    p.add_argument('--media-threads', type=int, default=None,
                   help='FFmpeg decoder threads per invocation; omit to let the container decide')
    p.add_argument('--footer-workers', type=int, default=None,
                   help='Footer-stamping threads; omit to let the container decide')
    p.add_argument('--submit', action='store_true')
    p.add_argument('pipeline_args', nargs=argparse.REMAINDER, help='After --, supply run_pipeline.py run flags')
    args = p.parse_args(argv)
    forwarded = args.pipeline_args[1:] if args.pipeline_args[:1] == ['--'] else args.pipeline_args
    # Validate the actual runtime arguments locally before spending on a job.
    from run_pipeline import parser
    runtime = parser().parse_args(['run', *forwarded, '--output-dir', '/output', '--scratch-dir', '/scratch/castle'])
    from castle_pipeline.runner import RunConfig
    from castle_pipeline.vertex import VertexProvider
    resolved_project = args.project or runtime.project
    if args.project and any(v == '--project' or v.startswith('--project=') for v in forwarded) and runtime.project != args.project:
        raise ValueError('Launcher and runtime project arguments disagree')
    # Only override the container-side defaults when the launcher was told to.
    tuning = {}
    if args.media_threads is not None:
        tuning['media_threads'] = args.media_threads
    if args.footer_workers is not None:
        tuning['footer_workers'] = args.footer_workers
    RunConfig(runtime.output_dir, runtime.scratch_dir, workers=runtime.workers, fps=runtime.fps,
              clip_seconds=runtime.clip_seconds, max_dim=runtime.max_dim, start_clip=runtime.start_clip,
              max_clips=runtime.max_clips, max_review_regions=runtime.max_review_regions,
              memory_soft_limit_gib=runtime.memory_soft_limit_gib, progress_interval_sec=runtime.progress_interval_sec,
              decode_slots=runtime.decode_slots, **tuning)
    # Constructor validates settings but does not construct a client or make requests.
    VertexProvider(runtime.model, resolved_project, location=runtime.location,
                   service_tier=runtime.service_tier, initial_concurrency=min(runtime.initial_concurrency, runtime.workers),
                   max_concurrency=runtime.workers, rpm=runtime.rpm, attempts=runtime.attempts,
                   timeout_sec=runtime.timeout_sec)
    command = build_command(code_volume=args.code_volume, output_volume=args.output_volume, name=args.name,
                            pipeline_args=forwarded, secrets=args.secret, project=resolved_project,
                            image=args.image, flavor=args.flavor, timeout=args.timeout,
                            media_threads=args.media_threads, footer_workers=args.footer_workers)
    if not args.submit:
        print(json.dumps({'submitted': False, 'argv': command}, indent=2))
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
