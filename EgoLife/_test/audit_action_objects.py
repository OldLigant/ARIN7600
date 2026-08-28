"""audit_action_objects.py — cross-validate legacy single-turn actions against the
multiturn T1 lexicon.

Question this answers: do the objects legacy's single-pass actions interact with
(a) land on LOW-CONFIDENCE lexicon entries (votes=1/k — the near-hallucination
zone the voting machinery is designed to flag), or (b) reference things that
never made it into the lexicon at all (legacy-hallucination candidates, or
lexicon misses — the report distinguishes nothing, a human reads the list)?

Method (bias-aware, two phases per clip):
  1. BLIND extraction call: send only the action texts (no lexicon!) to Mimo,
     ask it to quote the interacted object phrase verbatim. Extraction must not
     see the reference vocabulary, or it would snap everything onto it.
  2. Code substring match of each object phrase vs the lexicon (+ T2
     new_objects) of the SAME clip (matched by clip_id).
  3. One reconciliation call per clip for the leftovers only (cross-language
     synonyms like "phone" == 手机, hypernyms like 设备): it may pick a lexicon
     id or null — never invent.

Buckets for each extracted object:
  hit_high  matched lexicon entry with votes=k/k
  hit_mid   matched lexicon entry with 1 < votes < k (e.g. 2/3)
  hit_low   matched lexicon entry with votes=1/k        <- stepped on low confidence
  hit_new   matched a T2 new_objects escape-hatch entry
  unmatched no lexicon entry plausibly refers to it     <- "thing that never appeared"

Usage:
  python audit_action_objects.py \
      --legacy captions/_test/legacy_day4/full.jsonl \
      --t1 captions/_test/multiturn_day4/v3_votes3_ds_t1.jsonl \
      --multiturn captions/_test/multiturn_day4/v3_votes3_ds.jsonl \
      --out-dir captions/_test/action_object_audit
  python audit_action_objects.py --report-only ...     # stats only, no API calls
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent  # .../EgoLife (this script lives in _test/)

# ===========================================================================
# Prompts
# ===========================================================================

SYSTEM_MSG = """You are a meticulous audit assistant for first-person video annotations.
Follow exactly the one task you are given. Return ONLY a single JSON object: double
quotes, no trailing commas, no markdown fences, no text outside JSON."""

EXTRACT_TASK = """From the action lines below (all from one ~10s first-person clip), list every
action that manipulates / touches / holds / operates a PHYSICAL object, and quote
the object phrase EXACTLY as written in the action text — same language, no
translation, no normalization, no invention. Actions with no physical object
(walking, sitting, looking around, listening) produce nothing. One entry per
(action, object) pair; an action with two objects yields two entries.

Action lines:
{actions_json}

Return ONLY JSON:
{{"objects": [{{"action": "<verbatim action text>", "object": "<verbatim object phrase>"}}]}}"""

RECON_TASK = """For each extracted object phrase below, decide whether it refers to the SAME
physical object as one of the lexicon entries of the same clip (same referent;
synonym, translation, or generic reference are all OK — e.g. "phone" == "白色手机",
"设备" == the single specific device entry when only one is plausible). If no
lexicon entry plausibly refers to it, answer null. Do NOT force a match; when in
doubt, null.

Lexicon (id | label):
{lexicon}

Extracted object phrases:
{objects_json}

Return ONLY JSON:
{{"matches": [{{"object": "<verbatim>", "id": "obj01" | null}}]}}"""

# ===========================================================================
# Small helpers (self-contained on purpose)
# ===========================================================================

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def parse_json_obj(content: str) -> dict:
    text = content.strip()
    m = _JSON_FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("top-level not object")
    return data


class RateLimiter:
    def __init__(self, max_rpm: int):
        self.max_rpm = max(1, max_rpm)
        self._ts = collections.deque()
        self._cond = threading.Condition(threading.Lock())

    def acquire(self):
        with self._cond:
            while True:
                now = time.monotonic()
                while self._ts and self._ts[0] <= now - 60.0:
                    self._ts.popleft()
                if len(self._ts) < self.max_rpm:
                    self._ts.append(now)
                    return
                self._cond.wait(timeout=max(self._ts[0] + 60.0 - now + 0.05, 0.01))


def call_json(client, model, task_text, limiter, tag, log):
    """One small text call -> parsed dict. Single retry on failure."""
    messages = [{"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": task_text}]
    for attempt in (1, 2):
        limiter.acquire()
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=2048, temperature=1.0,
                extra_body={"thinking": {"type": "disabled"}},
                response_format={"type": "json_object"})
            content = resp.choices[0].message.content or ""
            return parse_json_obj(content)
        except Exception as e:
            log(f"[{tag}] attempt {attempt} failed: {type(e).__name__}: {e}")
    return None


# ===========================================================================
# Matching
# ===========================================================================

def substring_match(obj: str, entries: list):
    """Longest bidirectional-substring match over [{id,label,votes,confidence,src}]."""
    o = obj.strip()
    best = None
    for e in entries:
        lbl = e["label"]
        if not lbl:
            continue
        if o == lbl or o in lbl or lbl in o:
            if best is None or len(lbl) > len(best["label"]):
                best = e
    return best


def bucket_of(entry: dict) -> str:
    if entry["src"] == "new":
        return "hit_new"
    votes = entry.get("votes", "")
    m = re.match(r"^(\d+)/(\d+)$", votes)
    if m:
        got, k = int(m.group(1)), int(m.group(2))
        if got <= 1:
            return "hit_low"
        if got >= k:
            return "hit_high"
        return "hit_mid"
    return "hit_mid"


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--legacy", type=Path, required=True,
                    help="legacy single-turn captions jsonl (self_actions source)")
    ap.add_argument("--t1", type=Path, required=True,
                    help="multiturn T1 store jsonl (lexicon with votes)")
    ap.add_argument("--multiturn", type=Path, required=True,
                    help="multiturn final jsonl (source of new_objects)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "captions/_test/action_object_audit")
    ap.add_argument("--include-others", action="store_true",
                    help="also audit others' actions (default: self_actions only)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-rpm", type=int, default=60)
    ap.add_argument("--model", default="mimo-v2.5")
    ap.add_argument("--report-only", action="store_true",
                    help="skip API calls; report from the existing audit jsonl")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "audit.jsonl"
    report_path = args.out_dir / "audit_report.json"

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    # --- load inputs ---
    legacy = {}
    for line in open(args.legacy, encoding="utf-8"):
        r = json.loads(line)
        legacy[r["clip_id"]] = r
    lexica = {}
    for line in open(args.t1, encoding="utf-8"):
        r = json.loads(line)
        if r.get("ok"):
            lexica[r["clip_id"]] = r["parsed"]["lexicon"]
    new_objects = {}
    for line in open(args.multiturn, encoding="utf-8"):
        r = json.loads(line)
        new_objects[r["clip_id"]] = r.get("new_objects") or []

    common = [cid for cid in legacy if cid in lexica]
    log(f"clips: legacy={len(legacy)} t1={len(lexica)} overlap={len(common)}")

    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["clip_id"])
            except Exception:
                pass
    log(f"resume: {len(done)} clips already audited")

    # --- worker ---
    out_lock = threading.Lock()
    out_f = out_path.open("a", encoding="utf-8")
    limiter = RateLimiter(args.max_rpm)
    client = None

    def process(cid):
        rec_legacy = legacy[cid]
        texts = [a["text"] for a in (rec_legacy.get("self_actions") or [])]
        if args.include_others:
            texts += [a["text"] for a in (rec_legacy.get("others") or [])]
        texts = [t for t in texts if t.strip()]
        result = {"clip_id": cid, "items": [], "error": ""}

        if texts:
            actions_json = json.dumps(texts, ensure_ascii=False, indent=0)
            extracted = call_json(client, args.model,
                                  EXTRACT_TASK.format(actions_json=actions_json),
                                  limiter, f"{cid}/extract", log)
            if extracted is None:
                result["error"] = "extract_failed"
            else:
                # build the clip's match universe
                entries = [{"id": e["id"], "label": e["label"], "votes": e.get("votes", ""),
                            "confidence": e.get("confidence", ""), "src": "lex"}
                           for e in lexica[cid]]
                entries += [{"id": f"new{i:02d}", "label": n.get("label", ""),
                             "votes": "new", "confidence": "", "src": "new"}
                            for i, n in enumerate(new_objects.get(cid, []))]
                items = []
                leftovers = []
                for it in extracted.get("objects", []) or []:
                    if not isinstance(it, dict):
                        continue
                    obj = str(it.get("object", "")).strip()
                    act = str(it.get("action", "")).strip()
                    if not obj:
                        continue
                    entry = substring_match(obj, entries)
                    if entry:
                        items.append({"action": act, "object": obj, "match": "code",
                                      "entry": entry["id"], "label": entry["label"],
                                      "votes": entry["votes"], "bucket": bucket_of(entry)})
                    else:
                        leftovers.append(obj)
                if leftovers:
                    uniq = list(dict.fromkeys(leftovers))
                    lex_txt = "\n".join(f"{e['id']} | {e['label']}" for e in entries)
                    matches = call_json(client, args.model,
                                        RECON_TASK.format(lexicon=lex_txt,
                                                          objects_json=json.dumps(uniq, ensure_ascii=False)),
                                        limiter, f"{cid}/recon", log)
                    m_map = {}
                    if matches:
                        for m in matches.get("matches", []) or []:
                            if isinstance(m, dict):
                                m_map[str(m.get("object", "")).strip()] = m.get("id")
                    by_id = {e["id"]: e for e in entries}
                    for obj in leftovers:
                        mid = m_map.get(obj)
                        entry = by_id.get(mid) if mid else None
                        if entry:
                            items.append({"action": "", "object": obj, "match": "llm",
                                          "entry": entry["id"], "label": entry["label"],
                                          "votes": entry["votes"], "bucket": bucket_of(entry)})
                        else:
                            items.append({"action": "", "object": obj, "match": "none",
                                          "entry": "", "label": "", "votes": "",
                                          "bucket": "unmatched"})
                result["items"] = items

        with out_lock:
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

    if not args.report_only:
        for ep in [Path.cwd() / ".env", ROOT / ".env"]:
            if ep.exists():
                load_dotenv(ep)
                break
        api_key = os.environ.get("MIMO_API_KEY")
        if not api_key:
            print("ERROR: MIMO_API_KEY not set", file=sys.stderr)
            sys.exit(2)
        client = OpenAI(api_key=api_key, base_url="https://api.xiaomimimo.com/v1")

        todo = [cid for cid in common if cid not in done]
        if args.limit:
            todo = todo[:args.limit]
        log(f"auditing {len(todo)} clips ...")
        q: queue.Queue = queue.Queue()
        for cid in todo:
            q.put(cid)
        for _ in range(args.workers):
            q.put(None)

        def worker():
            while True:
                cid = q.get()
                if cid is None:
                    q.task_done()
                    return
                try:
                    process(cid)
                except Exception as e:
                    log(f"[{cid}] crash: {type(e).__name__}: {e}")
                    with out_lock:
                        out_f.write(json.dumps({"clip_id": cid, "items": [],
                                                "error": f"crash:{e}"}, ensure_ascii=False) + "\n")
                        out_f.flush()
                q.task_done()

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        out_f.close()
        log("done.")

    # --- report ---
    if not out_path.exists():
        log("no audit file; nothing to report")
        return
    buckets = collections.Counter()
    methods = collections.Counter()
    low_hits, unmatched = [], []
    n_items = 0
    for line in open(out_path, encoding="utf-8"):
        r = json.loads(line)
        for it in r.get("items", []):
            n_items += 1
            buckets[it["bucket"]] += 1
            methods[it["match"]] += 1
            if it["bucket"] == "hit_low":
                low_hits.append({"clip": r["clip_id"], **it})
            elif it["bucket"] == "unmatched":
                unmatched.append({"clip": r["clip_id"], **it})
    report = {
        "n_items": n_items,
        "buckets": dict(buckets),
        "match_methods": dict(methods),
        "hit_low_pct": round(buckets["hit_low"] / max(1, n_items), 4),
        "unmatched_pct": round(buckets["unmatched"] / max(1, n_items), 4),
        "low_hits": low_hits,
        "unmatched": unmatched,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_items", "buckets", "match_methods",
                                             "hit_low_pct", "unmatched_pct")},
                     ensure_ascii=False, indent=1))
    print(f"\nlow-hit sample (first 5):")
    for x in low_hits[:5]:
        print(f"  {x['clip']}: {x['object']!r} -> {x['label']} ({x['votes']})")
    print(f"unmatched sample (first 10 of {len(unmatched)}):")
    for x in unmatched[:10]:
        print(f"  {x['clip']}: {x['object']!r}")
    print(f"\nfull report -> {report_path}")


if __name__ == "__main__":
    main()
