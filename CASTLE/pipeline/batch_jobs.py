#!/usr/bin/env python3
"""Render short HF Batch workers. No submission unless --execute is supplied.

``--code-volume auto`` reads the run's persisted state and mounts the release it
is actually pinned to, instead of trusting an operator to remember. Getting this
wrong fails every tick within seconds, and did so three times before this existed
(docs/release-process.md R-05, appendix B-1).

With an explicit release, ``hourly-tick`` can be scheduled before preprocessing
finishes. Each scheduled worker exits successfully while GCS state is absent.
"""
import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
CODE_BUCKET = 'hf://buckets/Ligant/castle-code'
DEFAULT_CREDENTIALS = REPO / 'credentials' / 'my-project-omni-507802-9a6505abf0f7.json'


class BatchStateMissing(SystemExit):
    """No pin exists yet; only an explicitly versioned schedule may wait for it."""


# Dispatch to the auxiliary mirror worker when the mounted release includes it.
# Older immutable releases retain their tick and the pre-state schedule guard.
TICK_DISPATCH = """import json
import os
import sys
from pathlib import Path

role, command = sys.argv[1], sys.argv[2:]
wrapper = Path(command[1]).with_name('batch_tick.py')
if wrapper.is_file():
    command[1] = str(wrapper)
    if role == 'hourly-tick':
        command.append('--allow-missing-state')
elif role == 'hourly-tick':
    sys.path.insert(0, str(Path(command[1]).parent))
    from castle_pipeline.batch_cloud import BatchCloud
    project = command[command.index('--project') + 1]
    location = command[command.index('--location') + 1]
    state_uri = command[command.index('--state-uri') + 1]
    state, _ = BatchCloud(project, location).read_state(state_uri)
    if state is None:
        print(json.dumps({'event': 'batch_tick_waiting', 'status': 'waiting_for_state',
                          'state_uri': state_uri, 'stop_schedule': False}), flush=True)
        raise SystemExit(0)
os.execvp(command[0], command)
"""


def load_credentials(path: Path):
    """Accept a service-account JSON or a KEY=<json> secrets file."""
    raw = Path(path).read_text(encoding='utf-8').strip()
    if not raw.startswith('{'):
        _, _, raw = raw.partition('=')
        raw = raw.strip()
    info = json.loads(raw)
    from google.oauth2 import service_account
    return service_account.Credentials.from_service_account_info(
        info, scopes=['https://www.googleapis.com/auth/cloud-platform'])


def known_releases() -> list[str]:
    return sorted(p.stem for p in (REPO / 'releases').glob('*.json'))


def resolve_release(state_uri: str, credentials_path: Path) -> dict:
    """Read the run's pin from GCS state and map its code_hash to a release name.

    The state records ``code_hash``; the release name comes from the manifests in
    ``releases/``. This is the pre-R-04 path: once state also carries ``release``
    this lookup becomes a verification rather than a resolution.
    """
    from google.cloud import storage
    from castle_pipeline.batch_cloud import parse_gs_uri, BatchCloud
    credentials = load_credentials(credentials_path)
    cloud = BatchCloud(_project_of(state_uri, credentials_path), 'global', credentials=credentials)
    state, _ = cloud.read_state(state_uri)
    if state is None:
        raise BatchStateMissing(f'No batch state at {state_uri}; nothing is pinned yet. '
                                'To schedule before preparation finishes, use hourly-tick '
                                'with the same explicit --code-volume as submit.')
    config = state.get('config') or {}
    recorded = config.get('release')
    stored = config.get('code_hash')
    releases = {}
    for path in sorted((REPO / 'releases').glob('*.json')):
        manifest = json.loads(path.read_text(encoding='utf-8'))
        releases[manifest['code_hash']] = manifest['release']
    resolved = recorded or releases.get(stored)
    if not resolved:
        raise SystemExit(
            'E_CODE_VERSION_UNKNOWN\n'
            f'  run code_hash : {stored}\n'
            '  known         : ' + ', '.join(f'{v} ({k[:8]})' for k, v in releases.items()) + '\n'
            '  fix           : add the manifest via release.py, or pass an explicit --code-volume')
    return {'release': resolved, 'code_hash': stored,
            'code_volume': f'{CODE_BUCKET}/{resolved}',
            'stage': state.get('current_stage'), 'status': state.get('status')}


def _project_of(state_uri: str, credentials_path: Path) -> str:
    raw = Path(credentials_path).read_text(encoding='utf-8').strip()
    if not raw.startswith('{'):
        _, _, raw = raw.partition('=')
    return json.loads(raw).get('project_id') or ''


def build_command(*, role, code_volume, output_volume, name, project, state_uri,
                  pipeline_args, credential_secret, image='python:3.12-slim',
                  flavor='cpu-basic', timeout=None, location='global'):
    if role not in {'submit','hourly-tick','tick'}:
        raise ValueError('Unknown worker role')
    for value in (code_volume,output_volume):
        if not value.startswith('hf://buckets/') or any(c in value[5:] for c in ':\r\n') or '..' in value.split('/'):
            raise ValueError('Use explicit HF bucket prefixes')
    if code_volume.rstrip('/')==output_volume.rstrip('/'):
        raise ValueError('Code and output must be distinct')
    if not re.fullmatch(r'[A-Z][A-Z0-9_]*',credential_secret):
        raise ValueError('Supply the environment variable name containing ADC JSON, never its value')
    if not project or not state_uri.startswith('gs://'):
        raise ValueError('Explicit Google project and GCS state URI are required')
    managed={'--project','--location','--state-uri','--output-dir','--scratch-dir','--submit'}
    if any(arg.split('=')[0] in managed for arg in pipeline_args):
        raise ValueError('Launcher owns project/location/state/output/scratch/submit flags')
    if role!='submit' and pipeline_args:
        raise ValueError('Tick scope comes entirely from persistent state')
    cmd=['hf','jobs']
    if role=='hourly-tick':
        # The API takes a CRON expression. The CLI's --help also lists bare names
        # such as 'hourly', but it forwards the value verbatim, so the server
        # rejects it with "Invalid CRON expression"; '@hourly' is the accepted
        # spelling (see huggingface_hub.hf_api.create_scheduled_job).
        cmd+=['scheduled','run','@hourly','--no-concurrency']
    else:
        cmd+=['run','--detach']
    cmd+=['--flavor',flavor,'--timeout',timeout or ('3h' if role=='submit' else '30m'),
          '--name',name,'--volume',code_volume.rstrip('/')+':/workspace:ro',
          '--volume',output_volume.rstrip('/')+':/output',
          '--secrets',credential_secret,'--env','CASTLE_ADC_SECRET_NAME='+credential_secret,
          '--env','GOOGLE_APPLICATION_CREDENTIALS=/scratch/google-adc.json',
          '--env','PYTHONUNBUFFERED=1','--env','PYTHONDONTWRITEBYTECODE=1',
          '--env','HF_HOME=/scratch/hf','--env','HF_XET_CHUNK_CACHE_SIZE_BYTES=0',
          '--env','HF_XET_HIGH_PERFORMANCE=0','--env','OMP_NUM_THREADS=1']
    runtime=['python','/workspace/batch_pipeline.py','prepare' if role=='submit' else 'tick',
             '--project',project,'--location',location,'--state-uri',state_uri,
             '--output-dir','/output','--scratch-dir','/scratch/castle',*pipeline_args]
    if role=='submit':runtime+=['--submit']
    if role!='submit':runtime=['python','-c',TICK_DISPATCH,role,*runtime]
    install_media=('if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then '
                   'apt-get update && apt-get install -y --no-install-recommends ffmpeg; fi; ') if role=='submit' else ''
    bootstrap=('set -eu; mkdir -p /scratch/castle; '+install_media+
               'python -m pip install --no-cache-dir -r /workspace/requirements-batch.txt; '
               'exec python /workspace/bootstrap_batch.py '+shlex.join(runtime))
    return cmd+['--',image,'bash','-lc',bootstrap]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('role',choices=['submit','hourly-tick','tick'])
    for field in ['code-volume','output-volume','name','project','state-uri']:
        p.add_argument('--'+field,required=True)
    p.add_argument('--credential-secret',default='GOOGLE_ADC_JSON')
    p.add_argument('--credentials-file',type=Path,default=DEFAULT_CREDENTIALS,
                   help='Local ADC used only to resolve --code-volume auto; never passed to the Job')
    p.add_argument('--location',default='global')
    p.add_argument('--image',default='python:3.12-slim')
    p.add_argument('--flavor',default='cpu-basic')
    p.add_argument('--timeout')
    p.add_argument('--execute',action='store_true')
    argv=list(sys.argv[1:] if argv is None else argv)
    split=argv.index('--') if '--' in argv else len(argv)
    args=p.parse_args(argv[:split]);forwarded=argv[split+1:]
    values=vars(args).copy();execute=values.pop('execute');values.pop('credentials_file')
    values['pipeline_args']=forwarded
    resolved={'release':None,'code_hash':None,'stage':None,'status':None}
    if args.code_volume=='auto':
        if args.role=='submit':
            raise SystemExit('--code-volume auto needs an existing state, so it cannot be used for the '
                             'first submit; name the release explicitly.')
        resolved=resolve_release(args.state_uri,args.credentials_file)
        values['code_volume']=resolved['code_volume']
    elif args.code_volume.startswith('hf://buckets/'):
        # A first submit has no pin yet. The worker checks that the prefix is unused.
        # Existing runs require the same identity, not necessarily the same release name:
        # auxiliary-only releases (v3/v4) legitimately share a code_hash.
        release=args.code_volume.rstrip('/').rsplit('/',1)[-1]
        if release in known_releases():
            manifest=json.loads((REPO/'releases'/f'{release}.json').read_text(encoding='utf-8'))
            if args.role!='submit':
                try:
                    pinned=resolve_release(args.state_uri,args.credentials_file)
                except BatchStateMissing:
                    if args.role!='hourly-tick':raise
                    pinned=None
                    resolved={**resolved,'status':'waiting_for_state'}
                if pinned is not None and pinned['code_hash']!=manifest['code_hash']:
                    raise SystemExit(
                        'E_CODE_VERSION_MISMATCH\n'
                        f"  run pinned to : {pinned['release']} (code_hash {str(pinned['code_hash'])[:8]})\n"
                        f"  requested     : {release} (code_hash {manifest['code_hash'][:8]})\n"
                        f"  fix           : --code-volume auto\n"
                        f"                  or --code-volume {pinned['code_volume']}")
                if pinned is not None:resolved=pinned
            resolved={**resolved,'release':release,'code_hash':manifest['code_hash'],
                      'code_volume':args.code_volume.rstrip('/')}
    command=build_command(**values)
    if args.role=='submit':
        from batch_pipeline import build_parser
        build_parser().parse_args(['prepare',*forwarded,'--project',args.project,'--location',args.location,
                                  '--state-uri',args.state_uri,'--output-dir','/output','--scratch-dir','/scratch/castle'])
    if not execute:
        print(json.dumps({'submitted':False,'resolved':resolved,'argv':command},indent=2));return 0
    return subprocess.run(command,check=False).returncode


if __name__=='__main__':
    raise SystemExit(main())
