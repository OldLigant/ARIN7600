#!/usr/bin/env python3
"""Version-controlled run ledger: one schema, one writer (docs/release-process.md R-08).

The ledger answers "what did we submit, pin, and decide" 鈥?a code bucket records
what finished, but not who submitted it, on what release, or what is still
claimed. It therefore lives in the repository and is never hand-edited:
``append`` is the only writer, ``validate`` gates it, and ``report`` renders the
documents that used to be maintained by hand.

Records are append-only. A later record may supersede an earlier one for the same
job id; readers take the newest row per job for the current view.

Usage:
  python ledger.py validate                      # every row against the schema
  python ledger.py append --kind job --status submitted --job-id Ligant/xxx ...
  python ledger.py show --kind job --status submitted
  python ledger.py migrate-legacy --from <path>   # one-off import of old rows
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
LEDGER = REPO / "ledger" / "ledger.jsonl"

# Append-only event kind, then the lifecycle status of the thing it describes.
KINDS = ("job", "audit", "decision")
STATUSES = ("submitted", "running", "succeeded", "failed", "canceled", "info", "superseded")
ROLES = ("prepare", "submit", "tick", "hourly-tick", "reconcile", "status", "online-run", "audit")

REQUIRED = ("ts_utc", "kind", "status")
# Fields that only make sense for some kinds.
KIND_REQUIRED = {
    "job": ("job_id", "role"),
}
JOB_FIELDS = ("job_id", "role", "release", "code_hash", "state_uri", "scope", "sources",
              "model", "flavor", "timeout", "expected_clips", "note", "detail")
NESTED = {"pin": ("release", "code_hash"), "media_tuning": ("media_threads", "decode_slots",
                                                            "footer_workers", "prepare_workers")}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_utc(value) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return True
    except ValueError:
        return False


def validate_record(record, index=None) -> list[str]:
    """Return a list of human-readable problems; empty means valid."""
    where = f"row {index}" if index is not None else "record"
    problems = []
    if not isinstance(record, dict):
        return [f"{where}: must be a JSON object"]
    for field in REQUIRED:
        if field not in record:
            problems.append(f"{where}: missing required field {field!r}")
    if "ts_utc" in record and not _is_utc(record["ts_utc"]):
        problems.append(f"{where}: ts_utc must be UTC 'YYYY-MM-DDTHH:MM:SSZ', got {record['ts_utc']!r}")
    kind = record.get("kind")
    if kind is not None and kind not in KINDS:
        problems.append(f"{where}: unknown kind {kind!r} (allowed: {', '.join(KINDS)})")
    status = record.get("status")
    if status is not None and status not in STATUSES:
        problems.append(f"{where}: unknown status {status!r} (allowed: {', '.join(STATUSES)})")
    role = record.get("role")
    if role is not None and role not in ROLES:
        problems.append(f"{where}: unknown role {role!r} (allowed: {', '.join(ROLES)})")
    for field in KIND_REQUIRED.get(kind, ()):
        if not record.get(field):
            problems.append(f"{where}: kind {kind!r} requires {field!r}")
    if "sources" in record and not (isinstance(record["sources"], list)
                                    and all(isinstance(s, str) for s in record["sources"])):
        problems.append(f"{where}: sources must be a list of strings")
    for name, fields in NESTED.items():
        if name in record:
            if not isinstance(record[name], dict):
                problems.append(f"{where}: {name} must be an object")
            else:
                unknown = sorted(set(record[name]) - set(fields))
                if unknown:
                    problems.append(f"{where}: {name} has unknown keys {unknown}")
    known = set(REQUIRED) | set(JOB_FIELDS) | set(NESTED) | {"failure_detail", "expected_clips"}
    unknown = sorted(set(record) - known)
    if unknown:
        problems.append(f"{where}: unknown top-level field(s) {unknown}")
    if "expected_clips" in record and not isinstance(record["expected_clips"], int):
        problems.append(f"{where}: expected_clips must be an integer")
    return problems


def read(path: Path = LEDGER) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                rows.append({"_unparsable": True, "_line": number, "_raw": line[:120]})
    return rows


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8", newline="\n")


def latest_by_job(rows: list[dict]) -> dict:
    """Newest record per job id; the ledger is append-only, so later rows win."""
    view = {}
    for row in rows:
        job = row.get("job_id")
        if job:
            view[job] = row
    return view


def cmd_validate(args) -> int:
    path = args.path
    rows = read(path)
    problems = []
    for index, row in enumerate(rows, 1):
        if row.get("_unparsable"):
            problems.append(f"row {index}: not valid JSON")
            continue
        problems.extend(validate_record(row, index))
    # Append-only means no two rows may share (ts_utc, kind, job_id).
    seen = {}
    for index, row in enumerate(rows, 1):
        if row.get("_unparsable"):
            continue
        key = (row.get("ts_utc"), row.get("kind"), row.get("job_id"), row.get("status"))
        if key in seen:
            problems.append(f"row {index}: duplicate of row {seen[key]} for {key}")
        seen[key] = index
    print(json.dumps({"ok": not problems, "path": str(path), "rows": len(rows),
                      "jobs": len(latest_by_job(rows)), "problems": problems},
                     indent=2, ensure_ascii=False))
    return 1 if problems else 0


def cmd_append(args) -> int:
    path = args.path
    record = {"ts_utc": args.ts_utc or utcnow(), "kind": args.kind, "status": args.status}
    for field in JOB_FIELDS:
        value = getattr(args, field, None)
        if value not in (None, [], ""):
            record[field] = value
    if args.pin_release or args.pin_code_hash:
        record["pin"] = {"release": args.pin_release, "code_hash": args.pin_code_hash}
    tuning = {name: getattr(args, f"mt_{name}") for name in NESTED["media_tuning"]}
    if any(value is not None for value in tuning.values()):
        record["media_tuning"] = {k: v for k, v in tuning.items() if v is not None}
    if args.sources:
        record["sources"] = list(args.sources)
    problems = validate_record(record)
    if problems:
        print(json.dumps({"ok": False, "error": "SCHEMA_REJECTED", "problems": problems},
                         indent=2, ensure_ascii=False))
        return 1
    rows = read(path)
    if any(row.get("ts_utc") == record["ts_utc"] and row.get("kind") == record["kind"]
           and row.get("job_id") == record.get("job_id") and row.get("status") == record.get("status")
           for row in rows):
        print(json.dumps({"ok": False, "error": "DUPLICATE_APPEND",
                          "detail": "an identical row already exists; the ledger is append-only"},
                         indent=2, ensure_ascii=False))
        return 1
    rows.append(record)
    write(path, rows)
    print(json.dumps({"ok": True, "appended": record, "rows": len(rows), "path": str(path)},
                     indent=2, ensure_ascii=False))
    return 0


def cmd_show(args) -> int:
    rows = read(args.path)
    if args.kind:
        rows = [r for r in rows if r.get("kind") == args.kind]
    if args.status:
        rows = [r for r in rows if r.get("status") == args.status]
    if args.job_id:
        rows = [r for r in rows if r.get("job_id") == args.job_id]
    if args.current:
        rows = list(latest_by_job(rows).values())
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


LEGACY_ROLE = {"submit": "prepare", "tick": "tick", "collect": "tick", "audit": "audit"}
LEGACY_STATUS = {"ERROR": "failed", "submitted": "submitted", "succeeded": "succeeded",
                 "CANCELED": "canceled", "info": "info", "running": "running"}


def _normalise_utc(value) -> str:
    """Accept the legacy '+00:00' spelling and emit the schema's 'Z' form."""
    if not isinstance(value, str) or not value.strip():
        return utcnow()
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return utcnow()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cmd_migrate_legacy(args) -> int:
    """Import rows written before the schema existed, normalising field names.

    Legacy rows are quoted into the new schema: unknown prose is preserved in
    ``detail`` rather than dropped, because it records what was actually done.
    """
    source_rows = read(args.source)
    imported = []
    for row in source_rows:
        if row.get("_unparsable"):
            continue
        kind = "job" if row.get("job_id") else "audit"
        status = LEGACY_STATUS.get(str(row.get("status")), None)
        if status is None:
            status = "info" if kind == "audit" else "submitted"
        record = {"ts_utc": _normalise_utc(row.get("submitted_at") or row.get("updated_at")),
                  "kind": kind, "status": status}
        role = LEGACY_ROLE.get(str(row.get("role")), None)
        if kind == "job":
            record["job_id"] = row["job_id"]
            record["role"] = role or "prepare"
        for field in ("state_uri", "scope", "model", "flavor", "timeout", "expected_clips"):
            if row.get(field) is not None:
                record[field] = row[field]
        if row.get("code_version"):
            record["pin"] = {"release": row["code_version"], "code_hash": None}
        if row.get("sources"):
            record["sources"] = row["sources"]
        if row.get("note"):
            record["note"] = row["note"]
        if row.get("detail"):
            record["detail"] = row["detail"]
        problems = validate_record(record)
        if problems:
            print(json.dumps({"ok": False, "source_row": row.get("job_id"), "problems": problems},
                             indent=2, ensure_ascii=False))
            return 1
        imported.append(record)
    write(args.path, imported)
    print(json.dumps({"ok": True, "imported": len(imported), "source": str(args.source),
                      "path": str(args.path)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--path", type=Path, default=LEDGER)

    p = sub.add_parser("validate"); common(p)

    p = sub.add_parser("append"); common(p)
    p.add_argument("--ts-utc", default=None, help="Override the timestamp (tests only)")
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--status", choices=STATUSES, required=True)
    p.add_argument("--job-id", default=None)
    p.add_argument("--role", choices=ROLES, default=None)
    p.add_argument("--release", default=None)
    p.add_argument("--code-hash", default=None)
    p.add_argument("--pin-release", default=None)
    p.add_argument("--pin-code-hash", default=None)
    p.add_argument("--state-uri", default=None)
    p.add_argument("--scope", default=None)
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--flavor", default=None)
    p.add_argument("--timeout", default=None)
    p.add_argument("--expected-clips", type=int, default=None)
    p.add_argument("--note", default=None)
    p.add_argument("--detail", default=None)
    for name in NESTED["media_tuning"]:
        p.add_argument(f"--mt-{name.replace('_', '-')}", dest=f"mt_{name}", type=int, default=None)

    p = sub.add_parser("show"); common(p)
    p.add_argument("--kind", choices=KINDS, default=None)
    p.add_argument("--status", choices=STATUSES, default=None)
    p.add_argument("--job-id", default=None)
    p.add_argument("--current", action="store_true", help="newest row per job id")

    p = sub.add_parser("migrate-legacy")
    p.add_argument("--from", dest="source", type=Path, required=True)
    common(p)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"validate": cmd_validate, "append": cmd_append, "show": cmd_show,
            "migrate-legacy": cmd_migrate_legacy}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
