#!/usr/bin/env python3
"""CLI smoke checks for the media-tuning knobs, run without any paid API call."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def show(label, value):
    print(f"{label}: {value}")


def main():
    from jobs import build_command, main as jobs_main
    from run_pipeline import parser as run_parser

    print("=== 1) jobs.py renders the tuning knobs as environment defaults ===")
    command = build_command(code_volume='hf://buckets/u/code', output_volume='hf://buckets/u/out',
                            name='plan', pipeline_args=['--source', 'main/day1/Bjorn/video/08.mp4',
                                                        '--model', 'm', '--max-clips', '2'],
                            secrets=[], media_threads=4, footer_workers=4)
    show('CASTLE_* env', [v for v in command if v.startswith('CASTLE_')])
    show('hf jobs present', command[:3] == ['hf', 'jobs', 'run'])

    print("\n=== 2) runtime flags reach RunConfig ===")
    runtime = run_parser().parse_args(['run', '--source', 'p', '--model', 'm',
                                       '--output-dir', '/output', '--scratch-dir', '/s',
                                       '--media-threads', '2', '--decode-slots', '3', '--footer-workers', '5',
                                       '--workers', '3'])
    show('parsed', (runtime.media_threads, runtime.decode_slots, runtime.footer_workers))
    from castle_pipeline.runner import RunConfig
    config = RunConfig(runtime.output_dir, runtime.scratch_dir, workers=runtime.workers,
                       media_threads=runtime.media_threads, decode_slots=runtime.decode_slots,
                       footer_workers=runtime.footer_workers)
    show('RunConfig', (config.media_threads, config.decode_slots, config.footer_workers))

    print("\n=== 3) defaults when no flag is given (CPU-derived, cgroup-blind) ===")
    for name in ('CASTLE_MEDIA_THREADS', 'CASTLE_FOOTER_WORKERS', 'CASTLE_PREPARE_WORKERS'):
        os.environ.pop(name, None)
    from castle_pipeline.media import env_decoder_default, env_footer_default, decoder_memory_warning
    from batch_pipeline import default_prepare_workers
    show('cpu_count', os.cpu_count())
    show('env_decoder_default', env_decoder_default())
    show('env_footer_default', env_footer_default())
    show('default_prepare_workers', default_prepare_workers())
    show('decoder budget MiB @4x3', decoder_memory_warning(4, 3))

    print("\n=== 4) environment overrides ===")
    os.environ['CASTLE_MEDIA_THREADS'] = '7'
    os.environ['CASTLE_FOOTER_WORKERS'] = '5'
    os.environ['CASTLE_PREPARE_WORKERS'] = '4'
    show('env decoder', env_decoder_default())
    show('env footer', env_footer_default())
    show('env prepare', default_prepare_workers())

    print("\n=== 5) invalid values fail closed ===")
    for name, module_call in (('CASTLE_MEDIA_THREADS', 'env_decoder_default'),):
        os.environ[name] = 'abc'
        try:
            env_decoder_default()
            show(f'{name}=abc', 'NOT REJECTED (bug)')
        except ValueError as error:
            show(f'{name}=abc', f'rejected: {error}')
        os.environ[name] = '0'
        try:
            env_decoder_default()
            show(f'{name}=0', 'NOT REJECTED (bug)')
        except ValueError as error:
            show(f'{name}=0', f'rejected: {error}')
        os.environ.pop(name, None)

    os.environ['CASTLE_PREPARE_WORKERS'] = '0'
    try:
        default_prepare_workers()
        show('CASTLE_PREPARE_WORKERS=0', 'NOT REJECTED (bug)')
    except ValueError as error:
        show('CASTLE_PREPARE_WORKERS=0', f'rejected: {error}')
    os.environ.pop('CASTLE_PREPARE_WORKERS', None)

    print("\n=== 6) batch CLI rejects an impossible decoder budget (before any client) ===")
    proc = subprocess.run([sys.executable, str(ROOT / 'batch_pipeline.py'), 'prepare',
                           '--project', 'p', '--state-uri', 'gs://b/x/state.json',
                           '--output-dir', str(ROOT / '_bench/batch-out'),
                           '--scratch-dir', str(ROOT / '_bench/batch-scratch'),
                           '--gcs-prefix', 'gs://b/x', '--source', 'main/day1/Bjorn/video/08.mp4',
                           # 32 x 16 x 40 MiB = 20 GiB of decoder buffers, over the 16 GiB hint.
                           '--media-threads', '32', '--decode-slots', '16'],
                          capture_output=True, text=True)
    show('exit code', proc.returncode)
    show('stderr', proc.stderr.strip()[:400])
    return 0


if __name__ == '__main__':
    sys.exit(main())
