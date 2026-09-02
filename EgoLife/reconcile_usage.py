"""Reconcile a caption_pipeline.py output directory's token accounting.

Background: caption_pipeline.py (before the 2026-09-01 fix) silently dropped the
G1 (generate) UsageRecord of every clip whose FIRST annotation attempt succeeded
— `_process_clip_legacy` appended P2 inference records before checking
`elif not usage_records`, so the clean-success G1 record never made it into
*_usage.jsonl. Effects on finished runs:

  * *_usage.jsonl is missing ~1 generate record per clean-success clip
  * summary.json / final log `tokens: in= out= cached=` under-count by the
    sum of those calls (video prompts are ~8k tokens each, so this is ~40-50%
    of the true input total in best_of_n=1 runs)
  * generate-stage latency stats only sample rejected/retried calls (biased)

The caption records themselves DO carry the G1 tokens (`tokens: {in, out,
cached}`), so the missing records can be reconstructed exactly for token
accounting (latency is unrecoverable).

This tool is REPORT-ONLY: it reads <stem>.jsonl, <stem>_usage.jsonl and
<stem>_summary.json, reconstructs the missing generate records virtually
(recovery label `backfilled_from_caption_tokens`), and prints:

  * records on disk vs reconstructed, per run
  * corrected in/out/cached totals vs what summary.json reported
  * the post-reconstruction invariant check: every caption clip has >=1
    generate usage record

Nothing is ever written.

Usage (from EgoLife/):
    python reconcile_usage.py captions/A1_JAKE/DAY1/full captions/A1_JAKE/DAY2/full
"""

import json
import sys
from pathlib import Path


def iter_jsonl(path: Path):
    """Tolerant JSONL reader (skips blank/corrupted lines), same spirit as
    caption_pipeline._iter_caption_records."""
    if not path.exists():
        return
    decoder = json.JSONDecoder()
    text = path.read_text(encoding="utf-8")
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\n\r":
            i += 1
        if i >= n:
            return
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            i += 1
            continue
        yield obj
        i += end


def reconcile(stem: Path) -> bool:
    caps_path = stem.with_suffix(".jsonl")
    use_path = stem.parent / f"{stem.name}_usage.jsonl"
    sum_path = stem.parent / f"{stem.name}_summary.json"

    caps = [r for r in iter_jsonl(caps_path) if r.get("clip_id")]
    usage = list(iter_jsonl(use_path))
    if not caps:
        print(f"[skip] {caps_path}: no caption records")
        return True

    gen_by_clip = {}
    for u in usage:
        if u.get("stage") == "generate":
            gen_by_clip.setdefault(u.get("clip_id"), 0)
            gen_by_clip[u.get("clip_id")] += 1

    # Reconstruct one virtual generate record per caption record whose clip has
    # none on disk (guard against duplicate caption records for the same clip).
    missing, backfilled_clip_ids = [], set()
    for cap in caps:
        cid = cap["clip_id"]
        if cid in backfilled_clip_ids:
            continue
        if gen_by_clip.get(cid, 0) > 0:
            continue
        tok = cap.get("tokens") or {}
        cin, cout = int(tok.get("in", 0) or 0), int(tok.get("out", 0) or 0)
        ccached = int(tok.get("cached", 0) or 0)
        missing.append({
            "clip_id": cid, "global_idx": cap.get("global_idx"), "attempt": 1,
            "call_index": 0, "latency_s": None, "stage": "generate",
            "recovery": "backfilled_from_caption_tokens",
            "usage": {"prompt_tokens": cin, "completion_tokens": cout,
                      "total_tokens": cin + cout, "cached_tokens": ccached, "raw": {}},
            "raw_content": "",
        })
        backfilled_clip_ids.add(cid)

    def totals(records):
        tin = sum(int((u.get("usage") or {}).get("prompt_tokens", 0) or 0) for u in records)
        tout = sum(int((u.get("usage") or {}).get("completion_tokens", 0) or 0) for u in records)
        tcached = sum(int((u.get("usage") or {}).get("cached_tokens", 0) or 0) for u in records)
        return tin, tout, tcached

    disk_in, disk_out, disk_cached = totals(usage)
    fix_in, fix_out, fix_cached = totals(missing)

    print(f"\n=== {caps_path.parent / caps_path.stem} ===")
    print(f"caption records           : {len(caps)}  (clips: {len({c['clip_id'] for c in caps})})")
    print(f"usage records on disk     : {len(usage)}  (generate: {sum(gen_by_clip.values())})")
    print(f"missing generate records  : {len(missing)}  (reconstructed from caption tokens)")
    print(f"tokens ON DISK            : in={disk_in:,} out={disk_out:,} cached={disk_cached:,}")
    print(f"tokens CORRECTED          : in={disk_in + fix_in:,} out={disk_out + fix_out:,} "
          f"cached={disk_cached + fix_cached:,}")
    print(f"  delta from disk         : in=+{fix_in:,} out=+{fix_out:,} cached=+{fix_cached:,}")
    if sum_path.exists():
        try:
            stok = (json.loads(sum_path.read_text(encoding="utf-8")).get("tokens") or {})
            s_in, s_out, s_cached = (int(stok.get("total_in", 0)), int(stok.get("total_out", 0)),
                                     int(stok.get("total_cached", 0)))
            print(f"summary.json reported     : in={s_in:,} out={s_out:,} cached={s_cached:,} "
                  f"(under-counts in by {disk_in + fix_in - s_in:,} after correction)")
        except Exception as e:
            print(f"summary.json unreadable: {e}")

    # Invariant: after reconstruction every caption clip must have >=1 generate record.
    fixed_gen = dict(gen_by_clip)
    for m in missing:
        fixed_gen[m["clip_id"]] = fixed_gen.get(m["clip_id"], 0) + 1
    uncovered = [c["clip_id"] for c in caps if fixed_gen.get(c["clip_id"], 0) == 0]
    if uncovered:
        print(f"INVARIANT VIOLATION: {len(uncovered)} caption clips still have no generate "
              f"record: {uncovered[:10]}")
        return False
    print("invariant check           : OK — every caption clip has >=1 generate record")
    return True


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    ok = True
    for a in argv:
        stem = Path(a)
        if stem.suffix:  # tolerate full filenames
            stem = stem.with_suffix("")
        ok = reconcile(stem) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
