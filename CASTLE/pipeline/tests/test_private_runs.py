"""Private binding tests use only fake keys under explicitly isolated tmp_path."""
import json
from pathlib import Path

import pytest

import private_runs


@pytest.fixture
def private_dir(tmp_path):
    directory = tmp_path / 'credentials'
    directory.mkdir()
    key = {'type': 'service_account', 'project_id': 'key-owner-project',
           'client_email': 'fake@key-owner-project.iam.gserviceaccount.com',
           'private_key': 'FAKE-KEY-FOR-OFFLINE-TESTS'}
    (directory / 'team-b.json').write_text(json.dumps(key), encoding='utf-8')
    return directory


def arguments(private_dir, **changes):
    fields = {'run-id': 'gamma', 'state-uri': 'gs://test/run-gamma/state.json',
              'project': 'execution-project', 'sa-key': 'team-b.json',
              'hf-output': 'hf://buckets/ExampleOwner/castle-output/campaign/gamma'}
    fields.update(changes)
    return ['append', '--credentials-dir', str(private_dir),
            *(part for key, value in fields.items() for part in ['--' + key, value])]


def test_append_persists_private_binding_but_prints_no_association_or_key(private_dir, capsys):
    assert private_runs.main(arguments(private_dir)) == 0
    output = capsys.readouterr().out
    assert json.loads(output)['appended'] is True
    for hidden in ['team-b.json', 'execution-project', 'ExampleOwner', 'FAKE-KEY', 'fake@']:
        assert hidden not in output
    record = private_runs.lookup_binding(private_dir, 'gamma')
    assert record['sa_key_file'] == 'team-b.json'
    assert record['gcp_project'] == 'execution-project'  # May differ from key's owning project.
    assert record['hf_owner'] == 'ExampleOwner'
    assert record['hf_bucket'] == 'castle-output'
    assert record['hf_prefix'] == 'campaign/gamma'
    assert record['hf_output_uri'] == 'hf://buckets/ExampleOwner/castle-output/campaign/gamma'
    raw = (private_dir / 'run-bindings.jsonl').read_text(encoding='utf-8')
    assert 'private_key' not in raw and 'client_email' not in raw and 'FAKE-KEY' not in raw


def test_repeated_append_is_idempotent(private_dir, capsys):
    assert private_runs.main(arguments(private_dir)) == 0
    original = (private_dir / 'run-bindings.jsonl').read_bytes()
    assert private_runs.main(arguments(private_dir)) == 0
    assert (private_dir / 'run-bindings.jsonl').read_bytes() == original
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['appended'] is False


@pytest.mark.parametrize('changes', [
    {'project': 'different-project'}, {'hf-output': 'hf://buckets/OtherOwner/out/gamma'},
    {'run-id': 'renamed'}, {'state-uri': 'gs://test/other/state.json'},
    {'run-id': 'other', 'state-uri': 'gs://test/other/state.json'},
])
def test_conflicting_binding_cannot_silently_reassign_a_run(private_dir, changes):
    assert private_runs.main(arguments(private_dir)) == 0
    original = (private_dir / 'run-bindings.jsonl').read_bytes()
    assert private_runs.main(arguments(private_dir, **changes)) == 1
    assert (private_dir / 'run-bindings.jsonl').read_bytes() == original


@pytest.mark.parametrize('changes', [
    {'sa-key': '../outside.json'}, {'sa-key': 'missing.json'},
    {'hf-output': 'hf://buckets/ExampleOwner/castle-output'},
    {'hf-output': 'hf://buckets/ExampleOwner/castle-output/../gamma'},
    {'state-uri': 'gs://test/run-gamma/other.json'},
])
def test_invalid_binding_is_rejected_without_writing(private_dir, changes):
    assert private_runs.main(arguments(private_dir, **changes)) == 1
    assert not (private_dir / 'run-bindings.jsonl').exists()


def test_malformed_key_does_not_leak_its_contents(private_dir, capsys):
    (private_dir / 'team-b.json').write_text('PRIVATE_INVALID_KEY_CONTENT', encoding='utf-8')
    assert private_runs.main(arguments(private_dir)) == 1
    assert 'PRIVATE_INVALID_KEY_CONTENT' not in capsys.readouterr().out


def test_validation_rejects_corrupt_existing_history_without_rewriting(private_dir):
    path = private_dir / 'run-bindings.jsonl'
    path.write_text('{not-json}\n', encoding='utf-8')
    assert private_runs.main(['validate', '--credentials-dir', str(private_dir)]) == 1
    assert private_runs.main(arguments(private_dir)) == 1
    assert path.read_text(encoding='utf-8') == '{not-json}\n'


def test_existing_writer_lock_is_not_removed(private_dir):
    lock = private_dir / '.run-bindings.lock'
    lock.write_text('another writer', encoding='utf-8')
    assert private_runs.main(arguments(private_dir)) == 1
    assert lock.read_text(encoding='utf-8') == 'another writer'
    assert not (private_dir / 'run-bindings.jsonl').exists()


def test_show_requires_explicit_reveal_to_print_association(private_dir, capsys):
    assert private_runs.main(arguments(private_dir)) == 0
    capsys.readouterr()
    common = ['show', '--credentials-dir', str(private_dir), '--run-id', 'gamma']
    assert private_runs.main(common) == 0
    assert 'team-b.json' not in capsys.readouterr().out
    assert private_runs.main(common + ['--reveal']) == 0
    assert json.loads(capsys.readouterr().out)['binding']['sa_key_file'] == 'team-b.json'


def test_nonignored_repository_destination_is_rejected(private_dir, monkeypatch):
    monkeypatch.setattr(private_runs, 'REPO', private_dir.parent)
    monkeypatch.setattr(private_runs.subprocess, 'run', lambda *a, **kw: type('Result', (), {'returncode': 1})())
    assert private_runs.main(arguments(private_dir)) == 1
    assert not (private_dir / 'run-bindings.jsonl').exists()


def test_registry_does_not_write_the_public_ledger(private_dir, monkeypatch):
    import ledger
    monkeypatch.setattr(ledger, 'write', lambda *a: pytest.fail('Public ledger must not be touched'))
    assert private_runs.main(arguments(private_dir)) == 0
    assert private_runs.main(['validate', '--credentials-dir', str(private_dir), '--check-keys']) == 0


@pytest.mark.parametrize('mutation', [{'private_key': 'PRIVATE-CONTENT'}, {'client_email': 'private@example.test'},
                                     {'hf_owner': 'contradicting-owner'}])
def test_imported_writer_enforces_same_schema_as_cli(private_dir, mutation):
    fields = private_runs.binding_fields('gamma', 'gs://test/run/state.json', 'execution-project',
                                         'team-b.json', 'hf://buckets/ExampleOwner/out/gamma')
    fields.update(mutation)
    with pytest.raises(private_runs.BindingError):
        private_runs.append_binding(private_dir, fields)
    assert not (private_dir / 'run-bindings.jsonl').exists()


def test_registry_symlink_is_not_followed(private_dir, tmp_path):
    public = tmp_path / 'public.jsonl'
    public.write_text('', encoding='utf-8')
    try:
        (private_dir / 'run-bindings.jsonl').symlink_to(public)
    except OSError as error:
        if getattr(error, 'winerror', None) == 1314:
            pytest.skip('Windows user lacks symlink creation privilege')
        raise
    assert private_runs.main(arguments(private_dir)) == 1
    assert public.read_bytes() == b''
    assert private_runs.main(['validate', '--credentials-dir', str(private_dir)]) == 1
