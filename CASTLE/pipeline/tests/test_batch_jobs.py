import json
import os
import shlex
import sys
from types import SimpleNamespace

import pytest
import batch_jobs
from batch_jobs import build_command


def options():
    return dict(role='submit', code_volume='hf://buckets/test/code/v1',
                output_volume='hf://buckets/test/output/run1', name='test-batch',
                project='test-project', state_uri='gs://test/run1/state.json',
                pipeline_args=['--gcs-prefix','gs://test/run1','--source','day/file.mp4'],
                credential_secret='GOOGLE_ADC_JSON')


def test_submit_detaches_and_adc_is_materialized_without_literal_secret():
    cmd=build_command(**options())
    assert cmd[:4]==['hf','jobs','run','--detach']
    assert '--secrets' in cmd and 'GOOGLE_ADC_JSON' in cmd
    assert cmd[cmd.index('--')+1]=='python:3.12-slim'
    assert 'batch_pipeline.py prepare' in cmd[-1] and '--submit' in cmd[-1]
    assert 'bootstrap_batch.py' in cmd[-1]


def test_hourly_tick_is_nonoverlapping_and_cannot_reprepare_sources():
    cfg=options();cfg.update(role='hourly-tick',pipeline_args=[])
    cmd=build_command(**cfg)
    # '@hourly' is the spelling the API accepts; bare 'hourly' is rejected as an
    # invalid CRON expression even though the CLI --help lists it.
    assert cmd[:5]==['hf','jobs','scheduled','run','@hourly']
    assert '--no-concurrency' in cmd and '--detach' not in cmd
    assert 'batch_pipeline.py tick' in cmd[-1]
    cfg['pipeline_args']=['--source','anything']
    with pytest.raises(ValueError):build_command(**cfg)


@pytest.mark.parametrize('mutation',[
    {'pipeline_args':['--output-dir=/bad']},
    {'credential_secret':'literal=secret'},
    {'credential_secret':''},
    {'output_volume':'hf://buckets/test/code/v1'},
])
def test_unsafe_or_conflicting_configuration_rejected(mutation):
    cfg=options();cfg.update(mutation)
    with pytest.raises(ValueError):build_command(**cfg)


def test_cli_role_before_flags_and_default_is_dry_run(capsys,monkeypatch):
    from batch_jobs import main
    monkeypatch.setattr('subprocess.run',lambda *a,**kw:pytest.fail('Must not launch on a dry run'))
    assert main(['hourly-tick','--code-volume','hf://buckets/test/code/v1',
                 '--output-volume','hf://buckets/test/output/run1','--name','test',
                 '--project','test','--state-uri','gs://test/run/state.json'])==0
    assert '"submitted": false' in capsys.readouterr().out


def release_manifest(name):
    return json.loads((batch_jobs.REPO/'releases'/f'{name}.json').read_text(encoding='utf-8'))


def launcher_args(role, release):
    volume = release if release == 'auto' else f'hf://buckets/Ligant/castle-code/{release}'
    return [role, '--code-volume', volume,
            '--output-volume', 'hf://buckets/test/output/run1', '--name', 'test',
            '--project', 'test', '--state-uri', 'gs://test/run/state.json']


@pytest.mark.parametrize('release', ['castle-batch-v3', 'castle-batch-v4'])
def test_first_submit_of_registered_release_does_not_need_existing_state(release, capsys, monkeypatch):
    def no_state_lookup(*args, **kwargs):
        pytest.fail('A first submit has no run pin to resolve')
    monkeypatch.setattr(batch_jobs, 'resolve_release', no_state_lookup)
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: pytest.fail('Dry run must not submit'))
    args = launcher_args('submit', release) + [
        '--', '--gcs-prefix', 'gs://test/run', '--source', 'main/day1/Bjorn/video/08.mp4']
    assert batch_jobs.main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['submitted'] is False
    assert payload['resolved']['release'] == release
    assert payload['resolved']['code_hash'] == release_manifest(release)['code_hash']
    assert f'hf://buckets/Ligant/castle-code/{release}:/workspace:ro' in payload['argv']


@pytest.mark.parametrize('role', ['tick', 'hourly-tick'])
@pytest.mark.parametrize('requested, resolved', [
    ('castle-batch-v3', 'castle-batch-v4'),
    ('castle-batch-v4', 'castle-batch-v3'),
    ('castle-batch-v5', 'castle-batch-v3'),
    ('castle-batch-v5', 'castle-batch-v4'),
])
def test_explicit_compatible_release_preserves_requested_mount(role, requested, resolved, capsys, monkeypatch):
    digest = release_manifest(resolved)['code_hash']
    monkeypatch.setattr(batch_jobs, 'resolve_release', lambda *a: {
        'release': resolved, 'code_hash': digest,
        'code_volume': f'hf://buckets/Ligant/castle-code/{resolved}',
        'stage': 'audio', 'status': 'running'})
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: pytest.fail('Dry run must not submit'))
    assert batch_jobs.main(launcher_args(role, requested)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['resolved']['release'] == requested
    assert payload['resolved']['code_hash'] == digest
    assert payload['resolved']['code_volume'] == f'hf://buckets/Ligant/castle-code/{requested}'
    assert f'hf://buckets/Ligant/castle-code/{requested}:/workspace:ro' in payload['argv']


@pytest.mark.parametrize('role', ['tick', 'hourly-tick'])
def test_different_identity_is_rejected_before_submission(role, monkeypatch):
    digest = release_manifest('castle-batch-v2')['code_hash']
    monkeypatch.setattr(batch_jobs, 'resolve_release', lambda *a: {
        'release': 'castle-batch-v2', 'code_hash': digest,
        'code_volume': 'hf://buckets/Ligant/castle-code/castle-batch-v2',
        'stage': 'audio', 'status': 'running'})
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: pytest.fail('Mismatch must not submit'))
    with pytest.raises(SystemExit, match='E_CODE_VERSION_MISMATCH') as error:
        batch_jobs.main(launcher_args(role, 'castle-batch-v4') + ['--execute'])
    assert digest[:8] in str(error.value)
    assert '--code-volume auto' in str(error.value)


def test_auto_preserves_resolved_mount_and_identity(capsys, monkeypatch):
    pin = {'release': 'castle-batch-v4',
           'code_hash': release_manifest('castle-batch-v4')['code_hash'],
           'code_volume': 'hf://buckets/Ligant/castle-code/castle-batch-v4',
           'stage': 'audio', 'status': 'running'}
    monkeypatch.setattr(batch_jobs, 'resolve_release', lambda *a: pin)
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: pytest.fail('Dry run must not submit'))
    assert batch_jobs.main(launcher_args('tick', 'auto')) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['resolved'] == pin
    assert pin['code_volume'] + ':/workspace:ro' in payload['argv']


@pytest.fixture
def state_lookup(monkeypatch):
    """Replace only remote state/credentials; exercise the real launcher resolver."""
    from castle_pipeline import batch_cloud

    def install(state=None, error=None):
        reads = []

        def read_state(uri):
            reads.append(uri)
            if error is not None:
                raise error
            return state, 1 if state is not None else 0

        monkeypatch.setattr(batch_cloud, 'BatchCloud',
                            lambda *a, **kw: SimpleNamespace(read_state=read_state))
        monkeypatch.setattr(batch_jobs, 'load_credentials', lambda *a: None)
        monkeypatch.setattr(batch_jobs, '_project_of', lambda *a: 'test')
        return reads

    return install


def test_hourly_schedule_can_be_created_before_state_exists(state_lookup, monkeypatch):
    reads = state_lookup()
    commands = []
    monkeypatch.setattr('subprocess.run', lambda command, **kw:
                        commands.append(command) or SimpleNamespace(returncode=0))
    assert batch_jobs.main(launcher_args('hourly-tick', 'castle-batch-v4') + ['--execute']) == 0
    assert reads == ['gs://test/run/state.json']
    assert len(commands) == 1
    assert commands[0][:5] == ['hf', 'jobs', 'scheduled', 'run', '@hourly']
    assert 'hf://buckets/Ligant/castle-code/castle-batch-v4:/workspace:ro' in commands[0]


def test_early_schedule_dry_run_reports_waiting_without_submission(state_lookup, capsys, monkeypatch):
    state_lookup()
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: pytest.fail('Dry run must not submit'))
    assert batch_jobs.main(launcher_args('hourly-tick', 'castle-batch-v4')) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['submitted'] is False
    assert payload['resolved']['status'] == 'waiting_for_state'
    assert payload['resolved']['code_hash'] == release_manifest('castle-batch-v4')['code_hash']


@pytest.mark.parametrize('role,release', [
    ('tick', 'castle-batch-v4'), ('tick', 'auto'), ('hourly-tick', 'auto'),
])
def test_missing_state_still_rejects_manual_tick_and_auto(role, release, state_lookup, monkeypatch):
    state_lookup()
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: pytest.fail('Must not submit'))
    with pytest.raises(SystemExit, match='No batch state'):
        batch_jobs.main(launcher_args(role, release) + ['--execute'])


@pytest.mark.parametrize('code', [403, 503])
def test_early_schedule_does_not_treat_cloud_errors_as_missing_state(code, state_lookup, monkeypatch):
    from castle_pipeline.batch_cloud import CloudError
    state_lookup(error=CloudError('State lookup failed', code=code))
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: pytest.fail('Must not submit'))
    with pytest.raises(CloudError):
        batch_jobs.main(launcher_args('hourly-tick', 'castle-batch-v4') + ['--execute'])


def scheduled_worker(monkeypatch):
    """Execute the actual shell-quoted Python guard, with process/cloud boundaries replaced."""
    cfg = options()
    cfg.update(role='hourly-tick', pipeline_args=[])
    shell = shlex.split(build_command(**cfg)[-1])
    runtime = shell[shell.index('/workspace/bootstrap_batch.py') + 1:]
    assert runtime[:2] == ['python', '-c'], 'Scheduled worker needs a pre-state guard'
    monkeypatch.setattr(sys, 'argv', ['-c', *runtime[3:]])
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    return runtime[2], runtime[4:]


def test_scheduled_worker_exits_successfully_without_state_or_tick(state_lookup, capsys, monkeypatch):
    reads = state_lookup()
    monkeypatch.setattr(os, 'execvp', lambda *a: pytest.fail('Missing state must not run tick'))
    script, _ = scheduled_worker(monkeypatch)
    with pytest.raises(SystemExit) as result:
        exec(compile(script, '<scheduled-worker>', 'exec'), {'__name__': '__main__'})
    assert result.value.code == 0
    assert reads == ['gs://test/run1/state.json']
    payload = json.loads(capsys.readouterr().out)
    assert payload['status'] == 'waiting_for_state'
    assert payload['stop_schedule'] is False


@pytest.mark.parametrize('status', ['ready', 'submitting', 'running', 'complete', 'needs_attention'])
def test_scheduled_worker_delegates_existing_state_to_pinned_tick(status, state_lookup, monkeypatch):
    # In particular, ready/submitting can have no job ID yet and must retain recovery behavior.
    reads = state_lookup(state={'status': status, 'batches': {}})
    executed = []
    monkeypatch.setattr(os, 'execvp', lambda executable, argv: executed.append((executable, argv)))
    script, command = scheduled_worker(monkeypatch)
    exec(compile(script, '<scheduled-worker>', 'exec'), {'__name__': '__main__'})
    assert reads == ['gs://test/run1/state.json']
    assert executed == [('python', command)]
    assert command[:3] == ['python', '/workspace/batch_pipeline.py', 'tick']


@pytest.mark.parametrize('code', [403, 503])
def test_scheduled_worker_does_not_hide_state_read_failures(code, state_lookup, monkeypatch):
    from castle_pipeline.batch_cloud import CloudError
    state_lookup(error=CloudError('State lookup failed', code=code))
    monkeypatch.setattr(os, 'execvp', lambda *a: pytest.fail('Failed lookup must not run tick'))
    script, _ = scheduled_worker(monkeypatch)
    with pytest.raises(CloudError):
        exec(compile(script, '<scheduled-worker>', 'exec'), {'__name__': '__main__'})


@pytest.mark.parametrize('role', ['tick', 'hourly-tick'])
def test_tick_uses_auxiliary_wrapper_when_present_in_mounted_release(role, monkeypatch, state_lookup):
    state_lookup(state={'status': 'running'})
    cfg = options()
    cfg.update(role=role, pipeline_args=[])
    shell = shlex.split(build_command(**cfg)[-1])
    runtime = shell[shell.index('/workspace/bootstrap_batch.py') + 1:]
    assert runtime[:2] == ['python', '-c']
    monkeypatch.setattr(sys, 'argv', ['-c', *runtime[3:]])
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    monkeypatch.setattr(batch_jobs.Path, 'is_file', lambda path: path.name == 'batch_tick.py')
    executed = []
    monkeypatch.setattr(os, 'execvp', lambda executable, argv: executed.append((executable, argv)))
    exec(compile(runtime[2], '<tick-dispatch>', 'exec'), {'__name__': '__main__'})
    assert len(executed) == 1
    command = executed[0][1]
    assert command[0] == 'python' and command[2] == 'tick'
    assert batch_jobs.Path(command[1]).as_posix() == '/workspace/batch_tick.py'
    assert ('--allow-missing-state' in command) == (role == 'hourly-tick')
