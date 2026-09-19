"""Guards for the version-controlled ledger (docs/release-process.md R-08).

The ledger is evidence: it records what was submitted, on which release, and what
was decided. These tests keep it append-only, schema-valid and honest about
superseding an earlier row rather than rewriting it.
"""
import json

import ledger


def _row(**overrides):
    row = {'ts_utc': '2026-09-17T20:00:00Z', 'kind': 'job', 'status': 'submitted',
           'job_id': 'Ligant/abc', 'role': 'prepare'}
    row.update(overrides)
    return row


def test_required_fields_and_enumerations_are_enforced():
    assert ledger.validate_record(_row()) == []
    assert any('missing required field' in p for p in ledger.validate_record({'kind': 'job'}))
    assert any('unknown kind' in p for p in ledger.validate_record(_row(kind='nonsense')))
    assert any('unknown status' in p for p in ledger.validate_record(_row(status='weird')))
    assert any('unknown role' in p for p in ledger.validate_record(_row(role='nonsense')))
    # A job row without an id cannot be reconciled against anything.
    problems = ledger.validate_record(_row(job_id=None))
    assert any("requires 'job_id'" in p for p in problems)


def test_timestamps_must_be_explicit_utc():
    assert any('ts_utc must be UTC' in p
               for p in ledger.validate_record(_row(ts_utc='2026-09-17T20:00:00+00:00')))
    assert any('ts_utc must be UTC' in p for p in ledger.validate_record(_row(ts_utc='yesterday')))


def test_unknown_fields_are_rejected_so_the_schema_stays_closed(tmp_path):
    problems = ledger.validate_record(_row(media_thing=1))
    assert any('unknown top-level field' in p for p in problems)
    problems = ledger.validate_record(_row(media_tuning={'media_threads': 4, 'typo': 1}))
    assert any('media_tuning has unknown keys' in p for p in problems)
    assert ledger.validate_record(_row(media_tuning={'media_threads': 4, 'decode_slots': 3})) == []


def test_append_is_the_only_writer_and_refuses_duplicates(tmp_path):
    path = tmp_path / 'ledger.jsonl'
    base = ['append', '--path', str(path), '--kind', 'job', '--status', 'submitted',
            '--job-id', 'Ligant/abc', '--role', 'prepare', '--ts-utc', '2026-09-17T20:00:00Z']
    assert ledger.main(base) == 0
    assert ledger.main(base) == 1, 'an identical row must not be appended twice'
    assert len(ledger.read(path)) == 1


def test_superseding_a_job_appends_rather_than_editing(tmp_path):
    path = tmp_path / 'ledger.jsonl'
    ledger.main(['append', '--path', str(path), '--kind', 'job', '--status', 'submitted',
                 '--job-id', 'Ligant/abc', '--role', 'prepare', '--ts-utc', '2026-09-17T20:00:00Z'])
    ledger.main(['append', '--path', str(path), '--kind', 'job', '--status', 'succeeded',
                 '--job-id', 'Ligant/abc', '--role', 'prepare', '--ts-utc', '2026-09-17T21:00:00Z'])
    rows = ledger.read(path)
    assert [r['status'] for r in rows] == ['submitted', 'succeeded']
    # Readers take the newest row; history is preserved.
    assert ledger.latest_by_job(rows)['Ligant/abc']['status'] == 'succeeded'


def test_invalid_record_never_reaches_the_file(tmp_path):
    path = tmp_path / 'ledger.jsonl'
    assert ledger.main(['append', '--path', str(path), '--kind', 'job', '--status', 'submitted',
                        '--job-id', 'Ligant/abc']) == 1  # missing --role
    assert ledger.read(path) == []


def test_validate_reports_unparsable_rows(tmp_path):
    path = tmp_path / 'ledger.jsonl'
    path.write_text('{"ts_utc": "2026-09-17T20:00:00Z", "kind": "job", "status": "submitted"}\n'
                    '{not json\n', encoding='utf-8')
    assert ledger.main(['validate', '--path', str(path)]) == 1
