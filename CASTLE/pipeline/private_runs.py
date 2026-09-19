#!/usr/bin/env python3
"""Local-only run/credential bindings. Never publish the registry or reveal it in public reports.

append is idempotent and refuses conflicting assignments. This tool does not
submit jobs, load credentials into the environment, or write the public ledger.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parent
CREDENTIALS = REPO / 'credentials'
REGISTRY = 'run-bindings.jsonl'
FIELDS = {'schema_version', 'recorded_utc', 'run_id', 'state_uri', 'gcp_project',
          'sa_key_file', 'hf_output_uri', 'hf_owner', 'hf_bucket', 'hf_prefix'}


class BindingError(ValueError):
    """Messages contain codes only, never key contents or binding values."""


def binding_fields(run_id, state_uri, project, sa_key, hf_output):
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise BindingError('INVALID_RUN_ID')
    if not isinstance(project, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]*', project):
        raise BindingError('INVALID_PROJECT')
    if (not isinstance(sa_key, str) or '\\' in sa_key or Path(sa_key).is_absolute()
            or ':' in sa_key or any(p in {'', '.', '..'} for p in sa_key.split('/'))
            or not sa_key.endswith('.json')):
        raise BindingError('SA_KEY_MUST_BE_RELATIVE_JSON_PATH')
    state = urlsplit(state_uri)
    if (state.scheme != 'gs' or not state.netloc or state.query or state.fragment
            or '@' in state.netloc or ':' in state.netloc or '\\' in state_uri
            or not state.path.endswith('/state.json')
            or any(p in {'', '.', '..'} for p in state.path.lstrip('/').split('/'))):
        raise BindingError('INVALID_STATE_URI')
    output = urlsplit(hf_output.rstrip('/'))
    parts = output.path.lstrip('/').split('/')
    if (output.scheme != 'hf' or output.netloc != 'buckets' or output.query or output.fragment
            or len(parts) < 3 or any(p in {'', '.', '..'} for p in parts) or '\\' in hf_output
            or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', p) for p in parts[:2])):
        raise BindingError('REQUIRE_HF_OWNER_BUCKET_AND_RUN_PREFIX')
    return {'run_id': run_id, 'state_uri': state_uri, 'gcp_project': project, 'sa_key_file': sa_key,
            'hf_output_uri': 'hf://buckets/' + '/'.join(parts),
            'hf_owner': parts[0], 'hf_bucket': parts[1], 'hf_prefix': '/'.join(parts[2:])}


def validate_key(directory, relative):
    directory = Path(directory).resolve()
    key = (directory / relative).resolve()
    if not key.is_relative_to(directory) or not key.is_file():
        raise BindingError('SA_KEY_MISSING_OR_OUTSIDE_PRIVATE_DIRECTORY')
    try:
        data = json.loads(key.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise BindingError('SA_KEY_UNREADABLE_OR_INVALID_JSON') from None
    if (not isinstance(data, dict) or data.get('type') != 'service_account'
            or not all(isinstance(data.get(k), str) and data[k] for k in ['project_id', 'client_email', 'private_key'])):
        raise BindingError('EXPECTED_SERVICE_ACCOUNT_KEY')
    # The execution project can differ from the key's owning project. IAM access
    # must be checked separately; this registry does not assert cloud permissions.


def registry_path(directory):
    path = Path(directory) / REGISTRY
    if path.is_symlink():
        raise BindingError('REGISTRY_SYMLINK_REJECTED')
    return path


def read_bindings(directory=CREDENTIALS):
    path = registry_path(directory)
    if not path.exists():
        return []
    records = []
    seen = {field: set() for field in ['run_id', 'state_uri', 'hf_output_uri']}
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict) or set(record) != FIELDS or record['schema_version'] != 1:
                raise ValueError()
            datetime.strptime(record['recorded_utc'], '%Y-%m-%dT%H:%M:%SZ')
            normal = binding_fields(record['run_id'], record['state_uri'], record['gcp_project'],
                                    record['sa_key_file'], record['hf_output_uri'])
            if any(record[k] != value for k, value in normal.items()):
                raise ValueError()
            for field, values in seen.items():
                if record[field] in values:
                    raise ValueError()
                values.add(record[field])
        except (ValueError, TypeError, KeyError, AttributeError):
            raise BindingError(f'INVALID_REGISTRY_ROW_{number}') from None
        records.append(record)
    return records


def lookup_binding(directory, run_id):
    """Return a binding to the caller in memory, without logging the association."""
    for record in read_bindings(directory):
        if record['run_id'] == run_id:
            return record
    raise BindingError('UNKNOWN_RUN')


def ensure_private_store(directory):
    directory = Path(directory).resolve()
    registry = registry_path(directory)
    if directory.is_relative_to(REPO.resolve()):
        result = subprocess.run(['git', '-C', str(REPO), 'check-ignore', '--quiet', str(registry)],
                                capture_output=True, check=False)
        if result.returncode != 0:
            raise BindingError('REGISTRY_MUST_BE_GIT_IGNORED')
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@contextmanager
def writer_lock(directory):
    lock = directory / '.run-bindings.lock'
    try:
        handle = lock.open('x', encoding='utf-8')
    except FileExistsError:
        raise BindingError('REGISTRY_WRITER_LOCKED') from None
    try:
        with handle:
            handle.write('Private run registry writer\n')
        yield
    finally:
        lock.unlink()  # Only remove the exact lock created by this invocation.


def append_binding(directory, fields):
    if not isinstance(fields, dict) or set(fields) != FIELDS - {'schema_version', 'recorded_utc'}:
        raise BindingError('INVALID_BINDING_FIELDS')
    canonical = binding_fields(fields['run_id'], fields['state_uri'], fields['gcp_project'],
                               fields['sa_key_file'], fields['hf_output_uri'])
    if fields != canonical:
        raise BindingError('NONCANONICAL_BINDING_FIELDS')
    directory = ensure_private_store(directory)
    validate_key(directory, fields['sa_key_file'])
    with writer_lock(directory):
        records = read_bindings(directory)
        for record in records:
            if any(record[field] == fields[field] for field in ['run_id', 'state_uri', 'hf_output_uri']):
                if all(record[key] == value for key, value in fields.items()):
                    return False
                raise BindingError('BINDING_CONFLICT_NO_REASSIGNMENT')
        record = {'schema_version': 1, 'recorded_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                  **fields}
        path = registry_path(directory)
        # Preserve a valid last line even when a manually recovered file lacks LF.
        separator = '\n' if path.is_file() and path.stat().st_size and not path.read_bytes().endswith(b'\n') else ''
        with path.open('a', encoding='utf-8', newline='\n') as handle:
            handle.write(separator + json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
        return True


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['append', 'validate', 'show']:
        child = sub.add_parser(name)
        child.add_argument('--credentials-dir', type=Path, default=CREDENTIALS,
                           help='Private directory; override explicitly for isolated tests')
        if name == 'append':
            for field in ['run-id', 'state-uri', 'project', 'sa-key', 'hf-output']:
                child.add_argument('--' + field, required=True)
        elif name == 'show':
            child.add_argument('--run-id', required=True)
            child.add_argument('--reveal', action='store_true', help='Print private mapping; never paste into public records')
        else:
            child.add_argument('--check-keys', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'append':
            fields = binding_fields(args.run_id, args.state_uri, args.project, args.sa_key, args.hf_output)
            added = append_binding(args.credentials_dir, fields)
            result = {'ok': True, 'appended': added}
        elif args.command == 'show':
            record = lookup_binding(args.credentials_dir, args.run_id)
            result = {'ok': True, 'found': True}
            if args.reveal:
                result['binding'] = record
        else:
            records = read_bindings(args.credentials_dir)
            if args.check_keys:
                for record in records:
                    validate_key(args.credentials_dir, record['sa_key_file'])
            result = {'ok': True, 'bindings': len(records), 'keys_checked': args.check_keys}
    except BindingError as error:
        result = {'ok': False, 'error': str(error)}
    except (OSError, ValueError, TypeError):
        result = {'ok': False, 'error': 'PRIVATE_REGISTRY_OPERATION_FAILED'}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
