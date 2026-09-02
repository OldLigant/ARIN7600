"""Regression tests for the 2026-09-01 caption_pipeline.py fixes.

Covers three bugs found during the DAY2 run investigation:

  1. usage-record loss: `_process_clip_legacy` dropped the G1 (generate)
     UsageRecord on every clean-success clip, because P2 inference populated
     `usage_records` before the `elif not usage_records` fallback could fire.
     Unit tests below assert the exact usage-record sets per flow.
  2. progress-bar starvation: the main thread used to feed api_in_q (bounded
     queue) itself and stalled there for whole runs once the queue saturated.
     The feeder now runs on its own thread; the end-to-end smoke of main()
     would deadlock (and time out) if that regression came back.
  3. --skip-preprocess startup deadlock: jobs used to be pushed into the
     bounded queue BEFORE the api workers existed. The smoke runs 30 jobs with
     --api-workers 2 (queue maxsize 4) — pre-fix this deadlocks instantly.

Run:  python _test/test_usage_and_pipeline.py      (from EgoLife/, offline)

The main() smokes re-invoke this file with --child in a subprocess so a
deadlock/hang surfaces as a timeout instead of freezing the test itself.
All artifacts go into tempfile dirs; no real API is ever called.
"""
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

TEST_DIR = Path(__file__).resolve().parent
ROOT = TEST_DIR.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import caption_pipeline as cp  # noqa: E402

FAILURES = []
SAFETY_TEXT = "I cannot help with that."

DAY2_CAPTIONS = ROOT / "captions" / "A1_JAKE" / "DAY2"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeUsage:
    def __init__(self, prompt=100, completion=50, cached=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        self.prompt_tokens_details = {"cached_tokens": cached}

    def model_dump(self):
        return {"prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "prompt_tokens_details": self.prompt_tokens_details}


class FakeResp:
    def __init__(self, content, usage=None):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.usage = usage or FakeUsage()


class ScriptedModel:
    """Pops scripted responses in order; items are str, (str, FakeUsage) or Exception."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.scripts:
            raise AssertionError(f"script exhausted after {len(self.calls)} calls")
        item = self.scripts.pop(0)
        if isinstance(item, Exception):
            raise item
        content, usage = item if isinstance(item, tuple) else (item, FakeUsage())
        return FakeResp(content, usage)


class UniversalModel:
    """Thread-safe responder for main() smokes: answers G1 with a valid basic
    annotation and P2 with a valid inference payload, distinguished by the
    task marker in the user text."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0

    def create(self, **kwargs):
        with self._lock:
            self.calls += 1
        text = ""
        for part in kwargs["messages"][-1]["content"]:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
        if "TASK: INFER" in text:
            content = json.dumps(infer_payload(), ensure_ascii=False)
        else:
            content = json.dumps(annotation("smoke"), ensure_ascii=False)
        return FakeResp(content)


class FakeOpenAI:
    def __init__(self, api_key=None, base_url=None, **kw):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=UniversalModel().create))


# ---------------------------------------------------------------------------
# Payloads / helpers
# ---------------------------------------------------------------------------

ANNOTATION_BASE = {
    "environment": {"setting": "明亮的室内", "near_field": ["白色马克杯（桌面右手边）"], "background": ""},
    "people": [],
    "self_actions": [{"time": "11:09:42", "time_end": "11:09:44", "text": "X"}],
    "other_actions": [],
    "env_changes": [{"time": "11:09:42", "time_end": "11:09:44", "text": "杯子离开桌面", "cause": "self"}],
    "speech": [],
    "sound": [],
    "interface": [],
}


def annotation(marker):
    a = json.loads(json.dumps(ANNOTATION_BASE))
    a["self_actions"][0]["text"] = f"我拿起杯子 ({marker})"
    return a


def infer_payload():
    return {"psychology": {"emotion": "neutral", "mental_activity": "无特别线索"},
            "causal_links": [{"type": "action->env", "cause": "11:09:42 我拿起杯子",
                              "effect": "11:09:42 杯子离开桌面", "strength": "strong"}]}


def make_row(clip_id="DAY1_A1_JAKE_11100000_p1", gid=1, clip_kind="15s"):
    return pd.Series({
        "clip_id": clip_id, "day": 1, "user": "A1_JAKE",
        "start_ts": "11:10:00", "end_ts": "11:10:15+15.0s",
        "src_file": "DAY1_A1_JAKE_11100000.mp4", "clip_idx": 1, "global_idx": gid,
        "duration": 15.0, "is_day_open": False, "is_day_close": False, "status": "pending",
        "slice_path": "x.mp4", "clip_kind": clip_kind, "target_s": 15, "start_hms": "11:10:00",
    })


def make_client(scripts):
    m = ScriptedModel(scripts)
    return SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=m.create))), m


def unit_setup():
    log = logging.getLogger("usage_fix_test")
    log.setLevel(logging.WARNING)
    log.handlers.clear()
    log.addHandler(logging.NullHandler())
    limiter = cp.SlidingWindowRateLimiter(10000)
    tmp = Path(tempfile.mkdtemp(prefix="usage_fix_unit_"))
    vid = tmp / "clip.mp4"
    vid.write_bytes(b"fake-video-bytes")
    return log, limiter, tmp, vid


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILURES.append(name)
    print(f"[{status}] {name} {detail}")


def std_script(content):
    return (content, FakeUsage(prompt=100, completion=50))


# ---------------------------------------------------------------------------
# Unit tests: _process_clip_legacy usage records
# ---------------------------------------------------------------------------

def test_clean_success_records_g1():
    log, limiter, tmp, vid = unit_setup()
    cfg = cp.ApiCallConfig("disabled", "disabled", "disabled", run_infer=True)
    client, m = make_client([std_script(json.dumps(annotation("ok"))),
                             std_script(json.dumps(infer_payload()))])
    caps, uses, fails = cp._process_clip_legacy(client, "m", vid, make_row(), log,
                                                limiter, cfg, tmp, "disabled")
    check("clean: 1 caption, no failures", len(caps) == 1 and not fails)
    check("clean: 2 usage records (G1 + infer)", len(uses) == 2,
          f"got {len(uses)}")
    check("clean: stages generate->infer",
          [u.stage for u in uses] == ["generate", "infer"])
    check("clean: G1 usage tokens kept", uses[0].usage["prompt_tokens"] == 100
          and uses[0].recovery == "ok")
    check("clean: caption tokens mirror G1",
          caps[0].tokens == {"in": 100, "out": 50, "cached": 0})


def test_rejection_then_success_no_double_append():
    log, limiter, tmp, vid = unit_setup()
    cfg = cp.ApiCallConfig("disabled", "disabled", "disabled", run_infer=True)
    client, m = make_client([
        (SAFETY_TEXT, FakeUsage(prompt=100, completion=10)),          # G1 rejected
        std_script(json.dumps(annotation("retry"))),                  # G1 retry ok
        std_script(json.dumps(infer_payload())),                      # P2
    ])
    caps, uses, fails = cp._process_clip_legacy(client, "m", vid, make_row(), log,
                                                limiter, cfg, tmp, "disabled")
    check("rej: 1 caption", len(caps) == 1 and not fails)
    check("rej: 3 usage records (rej + retry G1 + infer)", len(uses) == 3,
          f"got {len(uses)}")
    check("rej: exactly one first_attempt_rejection",
          sum(1 for u in uses if u.recovery == "first_attempt_rejection") == 1)
    check("rej: stages generate,generate,infer",
          [u.stage for u in uses] == ["generate", "generate", "infer"])


def test_both_fail_records_attempts():
    log, limiter, tmp, vid = unit_setup()
    cfg = cp.ApiCallConfig("disabled", "disabled", "disabled", run_infer=True)
    client, m = make_client([
        (SAFETY_TEXT, FakeUsage(prompt=100, completion=10)),
        (SAFETY_TEXT, FakeUsage(prompt=100, completion=10)),
    ])
    caps, uses, fails = cp._process_clip_legacy(client, "m", vid,
                                                make_row(clip_kind="15s"), log,
                                                limiter, cfg, tmp, "disabled")
    check("fail: no caption, failure recorded", len(caps) == 0
          and len(fails) == 1 and fails[0]["error"] == "all_attempts_failed")
    check("fail: both G1 attempts recorded", len(uses) == 2
          and [u.recovery for u in uses] == ["first_attempt_rejection", "safety_rejection"])


def test_no_infer_single_record():
    log, limiter, tmp, vid = unit_setup()
    cfg = cp.ApiCallConfig("disabled", "disabled", "disabled", run_infer=False)
    client, m = make_client([std_script(json.dumps(annotation("ok")))])
    caps, uses, fails = cp._process_clip_legacy(client, "m", vid, make_row(), log,
                                                limiter, cfg, tmp, "disabled")
    check("no-infer: 1 caption, inference skipped",
          len(caps) == 1 and caps[0].inference == {"status": "skipped"})
    check("no-infer: exactly 1 generate usage record", len(uses) == 1
          and uses[0].stage == "generate")


# ---------------------------------------------------------------------------
# main() smoke, child mode (patched OpenAI, offline)
# ---------------------------------------------------------------------------

def prep_skip_cache(tmp_root, n_rows=30):
    """Build a temp captions dir: subset of the DAY2 clips.parquet + slice files."""
    out_dir = Path(tmp_root) / "skip"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "_cache"
    slices = cache / "slices"
    slices.mkdir(parents=True, exist_ok=True)
    src_cache = DAY2_CAPTIONS / "_cache"
    df = pd.read_parquet(src_cache / "clips.parquet").sort_values("global_idx").head(n_rows)
    for sp in df["slice_path"]:
        sp = Path(sp)
        shutil.copyfile(src_cache / sp, slices / sp.name)
    df.to_parquet(cache / "clips.parquet", index=False)
    return out_dir / "day2_smoke.jsonl"


def child_skip_preprocess(tmp_root):
    out_file = prep_skip_cache(tmp_root, 30)
    os.environ.setdefault("MIMO_API_KEY", "dummy-key")
    cp.OpenAI = FakeOpenAI
    sys.argv = ["caption_pipeline.py", "--participant", "A1_JAKE", "--day", "2",
                "--skip-preprocess", "--limit", "30", "--api-workers", "2",
                "--max-rpm", "100000", "--best-of-n", "1",
                "--thinking-p1", "disabled", "--thinking-critic", "disabled",
                "--thinking-infer", "disabled", "--out", str(out_file)]
    cp.main()


def child_full_pipeline(tmp_root):
    out_dir = Path(tmp_root) / "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "day2_window.jsonl"
    os.environ.setdefault("MIMO_API_KEY", "dummy-key")
    cp.OpenAI = FakeOpenAI
    sys.argv = ["caption_pipeline.py", "--participant", "A1_JAKE", "--day", "2",
                "--start-time", "1044", "--end-time", "1046", "--limit", "4",
                "--api-workers", "8", "--max-rpm", "100000", "--best-of-n", "1",
                "--thinking-p1", "disabled", "--thinking-critic", "disabled",
                "--thinking-infer", "disabled", "--out", str(out_file)]
    cp.main()


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def check_smoke_outputs(out_file, exp_captions, tag):
    caps = load_jsonl(out_file)
    usage = load_jsonl(out_file.parent / f"{out_file.stem}_usage.jsonl")
    summary = json.loads((out_file.parent / f"{out_file.stem}_summary.json")
                         .read_text(encoding="utf-8"))
    check(f"{tag}: caption count == {exp_captions}", len(caps) == exp_captions,
          f"got {len(caps)}")
    gen = sum(1 for u in usage if u.get("stage") == "generate")
    inf = sum(1 for u in usage if u.get("stage") == "infer")
    check(f"{tag}: usage = {exp_captions} generate + {exp_captions} infer",
          gen == exp_captions and inf == exp_captions,
          f"got gen={gen} infer={inf}")
    tok = summary.get("tokens", {})
    want_in, want_out = exp_captions * 2 * 100, exp_captions * 2 * 50
    check(f"{tag}: summary tokens include G1 (in={want_in})",
          tok.get("total_in") == want_in and tok.get("total_out") == want_out,
          f"got in={tok.get('total_in')} out={tok.get('total_out')}")


# ---------------------------------------------------------------------------
# Parent driver
# ---------------------------------------------------------------------------

def run_child(mode, tmp_root, timeout):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", mode, str(tmp_root)]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           timeout=timeout)
        return r
    except subprocess.TimeoutExpired as e:
        check(f"child {mode}: finished within {timeout}s", False,
              "TIMEOUT — pipeline deadlocked/hung")
        return None


def main():
    print("== unit tests: _process_clip_legacy usage records ==")
    test_clean_success_records_g1()
    test_rejection_then_success_no_double_append()
    test_both_fail_records_attempts()
    test_no_infer_single_record()

    print("\n== smoke: main() --skip-preprocess (30 jobs, 2 workers, bounded queue) ==")
    with tempfile.TemporaryDirectory(prefix="captest_skip_") as td:
        r = run_child("skip", td, timeout=300)
        if r is not None:
            check("child skip: exit code 0", r.returncode == 0,
                  "" if r.returncode == 0 else f"rc={r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-3000:]}")
            out_file = Path(td) / "skip" / "day2_smoke.jsonl"
            if out_file.exists():
                check_smoke_outputs(out_file, 30, "skip")

    print("\n== smoke: main() full preprocess path (feeder thread, real ffmpeg) ==")
    with tempfile.TemporaryDirectory(prefix="captest_full_") as td:
        r = run_child("full", td, timeout=900)
        if r is not None:
            check("child full: exit code 0", r.returncode == 0,
                  "" if r.returncode == 0 else f"rc={r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-3000:]}")
            out_file = Path(td) / "full" / "day2_window.jsonl"
            if out_file.exists():
                caps = load_jsonl(out_file)
                if caps:
                    check_smoke_outputs(out_file, len(caps), "full")

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--child":
        mode = sys.argv[2]
        if mode == "skip":
            child_skip_preprocess(sys.argv[3])
        elif mode == "full":
            child_full_pipeline(sys.argv[3])
        else:
            print(f"unknown child mode {mode}", file=sys.stderr)
            sys.exit(2)
        sys.exit(0)
    main()
