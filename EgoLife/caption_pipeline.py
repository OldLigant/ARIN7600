"""caption_pipeline.py — self-contained Mimo captioning pipeline for EgoLife.

Turns 30-second first-person (Meta Aria) video segments into 5-layer structured
JSON annotations: self_actions / others / environment / speech / psychology
(with an `awareness` field scoring how strongly emotion drove behavior).

Single file, no sibling-module imports. Depends only on the OpenAI SDK,
pandas, python-dotenv, jsonschema, and ffmpeg/ffprobe on PATH.

Architecture
------------
  [PreprocessWorker pool]   [RateLimiter]   [API Worker pool]   [Writer]
  ffmpeg re-encode   ->   produced_q   ->  acquire()  ->  Mimo call  ->  result_q  ->  ordered jsonl
   (CPU/subprocess)         (bounded)     (global, RPM cap)       (IO)                (single thread)

Usage
-----
  # standard run: 30s clips, JSON output, no thinking
  python caption_pipeline.py --participant A1_JAKE --day 1 --max-rpm 90
  # time-windowed
  python caption_pipeline.py --start-time 1110 --end-time 1130
  # resume after an interruption (skips clips already in the output file)
  python caption_pipeline.py --skip-existing
  # caption only (slices already encoded in _cache/)
  python caption_pipeline.py --skip-preprocess
  # higher quality (more granular actions, ~4x latency/token cost)
  python caption_pipeline.py --thinking enabled --max-completion-tokens 8192
"""
from __future__ import annotations

import argparse
import base64
import collections
import heapq
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# jsonschema is optional — validation degrades to manual checks if absent.
try:
    import jsonschema  # type: ignore
except ImportError:
    jsonschema = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ROOT = Path(__file__).resolve().parent  # .../ARIN7600/EgoLife/

# ===========================================================================
# Constants & prompt
# ===========================================================================

MAX_COMPLETION_TOKENS = 1024       # raise to ~8192 when thinking=enabled
THINKING_DEFAULT = "disabled"      # "enabled" | "disabled"

# ffmpeg re-encode defaults (1024x1024 @ 2fps is the validated sweet spot:
# watermark readable, whiteboard legible, ~2MB / 30s clip).
DEFAULT_RESOLUTION = 1024
DEFAULT_FPS = 2
DEFAULT_CRF = 28
DEFAULT_AUDIO_BITRATE_K = 64

# A gap >this between consecutive source timestamps is treated as a recording
# break (e.g. the multi-hour gaps in EgoLife). 60s pairing never bridges these.
_GAP_THRESHOLD_S = 35.0

SYSTEM_MSG = """You are a dense first-person life-log captioner.
The footage is from participant A1_JAKE wearing Meta Aria glasses. People may speak Chinese or English.

# Watermark anchor (CRITICAL)
Every video frame carries a watermark in the TOP-RIGHT corner showing the current time and day,
formatted "HH:MM:SS:FF DAYn" (e.g. "11:10:02:00 DAY1"). FF is a frame counter 00-19.
The user message tells you the day label and the approximate start time of this clip.
You MUST read the watermark to timestamp the actions you describe, so they can be located on a timeline.

# Output format
Return ONLY a single JSON object. No explanations, no markdown code fences, no text outside JSON.
Use exactly this structure (fill every field; use empty arrays/strings when a section is truly empty):

{
  "self_actions": [
    {"time": "HH:MM:SS", "time_end": "HH:MM:SS", "text": "..."}
  ],
  "others": [
    {"time": "HH:MM:SS", "time_end": "HH:MM:SS", "text": "..."}
  ],
  "environment": "...",
  "speech": [
    {"lang": "zh", "speaker": "...", "text": "..."}
  ],
  "psychology": {"awareness": "low", "emotion": "...", "note": "..."},
  "tags": ["..."]
}

## Field rules

self_actions: 3-6 objects, one per self-action, in first person ("I"), present tense, verb-led.
Cover the whole clip span. Each "time"/"time_end" is a watermark-based HH:MM:SS (~5s resolution).
Describe hand-object contact, posture, gaze, locomotion, device use. Mirror these real examples:
{"time":"11:10:02","time_end":"11:10:08","text":"我拿起手机划开屏幕"} /
{"time":"11:10:22","time_end":"11:10:30","text":"我把手机传给一位穿着黄色上衣的女士"}.
Use Chinese when the scene is Chinese-speaking. Never invent names for yourself; use neutral
descriptors for others unless their name is shown or spoken.

others: zero or more objects, same shape, timestamped the same way. Lead text with the person
descriptor ("a woman in blue", "Shure"). Empty array if you are alone.

environment: one or two sentences on the setting and any CHANGES during the clip: room, lighting,
screen content, object layout, weather/outdoor cues, location transitions. Empty string if static.

speech: zero or more objects. lang in {"zh","en"}; speaker optional; text is the quoted utterance.
Transcribe only what is actually heard; do not invent. Speech has no reliable timestamp from the
watermark, so do NOT timestamp it. Empty array if silent.

psychology: an object with exactly:
  awareness: "low" | "medium" | "high"  (required)
  emotion: one or two lowercase keywords (required; "neutral" if none readable)
  note: optional short sentence
"awareness" measures how strongly emotion CAUSALLY drove visible behavior in this clip:
  low    = emotion was background color; behavior was driven by habit or task, not by feeling.
  medium = emotion colored HOW an action was performed (tone, pace, expression, vigor) but did not
           change its direction.
  high   = emotion directly triggered a behavior or a turn/pivot (e.g. embarrassment -> covering face).
"emotion" picks from: neutral calm relaxed bored focused amused happy excited surprised confused
curious anxious nervous stressed frustrated angry embarrassed sad tired sleepy hungry, or similar.

tags: 5-10 short lowercase keywords (objects, actions, location, people descriptors). MANDATORY,
never empty.

# Rules
- Stay strictly within the watermark time range of THIS clip; never describe other videos.
- Be concrete and observational; do not speculate beyond what is visible or audible.
- Pick ONE language for all fields based on the dominant spoken language in the clip (Chinese if
  participants speak Chinese, English otherwise). Do NOT translate or duplicate content.
- Output each action, utterance, or fact exactly ONCE; never repeat in another language.
- Output must be valid JSON: double quotes, no trailing commas, no comments."""

USER_TASK_TMPL = (
    "Annotate this {duration_desc} first-person video ({clip_id}) and return the JSON object.\n"
    "Source file: {src_file}, recorded on {day_label}; the segment starts at approximately {start_hms}.\n"
    "Read the top-right watermark (HH:MM:SS:FF {day_label}) to timestamp the self_actions and others. "
    "Cover the full segment from {start_hms} onward."
)


# ===========================================================================
# ffmpeg helpers
# ===========================================================================

def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True,
    )
    return float(out.strip())


def ffmpeg_reencode(src: Path, out: Path, *, resolution: int, fps: int, crf: int, audio_k: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-vf", f"fps={fps},scale={resolution}:{resolution}:flags=lanczos",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-c:a", "aac", "-b:a", f"{audio_k}k", "-ac", "1",
        "-movflags", "+faststart", str(out),
    ]
    rc = subprocess.run(cmd, capture_output=True, text=True)
    if rc.returncode != 0:
        raise RuntimeError(f"ffmpeg re-encode failed: {rc.stderr.strip()[:300]}")


def parse_ts(filename: str) -> str:
    """`DAY1_A1_JAKE_11094208.mp4` -> `11:09:42.08`."""
    stem = Path(filename).stem
    ts_raw = stem.split("_")[-1]
    if len(ts_raw) != 8 or not ts_raw.isdigit():
        raise ValueError(f"unexpected filename {filename}")
    hh, mm, ss, cc = ts_raw[0:2], ts_raw[2:4], ts_raw[4:6], ts_raw[6:8]
    return f"{hh}:{mm}:{ss}.{cc}"


def concat_two_clips(src_a: Path, src_b: Path, out: Path, clip_id: str) -> None:
    """Concat two already-re-encoded clips into one 60s clip (ffmpeg concat demuxer,
    stream-copy). Only used when --clip-duration 60."""
    out.parent.mkdir(parents=True, exist_ok=True)
    list_path = (out.parent / f"{clip_id}_concat.txt").resolve()
    try:
        list_path.write_text(
            f"file '{src_a.resolve().as_posix()}'\nfile '{src_b.resolve().as_posix()}'\n",
            encoding="utf-8",
        )
        rc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(list_path), "-c", "copy", "-movflags", "+faststart", str(out.resolve())],
            capture_output=True, text=True,
        )
        if rc.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {rc.stderr.strip()[:300]}")
    finally:
        try:
            list_path.unlink()
        except OSError:
            pass


def slice_to_10s(src_video: Path, clip_id: str, out_dir: Path) -> list[Path]:
    """Recovery: slice a rejected clip into up to 3 x ~10s pieces (re-encoded for
    accurate cuts). Each piece is then captioned on its own."""
    dur = ffprobe_duration(src_video)
    n_slices = max(1, int(dur // 10) + (1 if dur % 10 > 1 else 0))
    n_slices = min(n_slices, 3)
    slice_dur = dur / n_slices
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_slices):
        start = i * slice_dur
        p = out_dir / f"{clip_id}_10s_{i + 1}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(src_video), "-t", f"{slice_dur:.3f}",
            "-vf", "fps=2,scale=1024:1024:flags=lanczos",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-movflags", "+faststart", str(p),
        ], check=True, capture_output=True, text=True)
        paths.append(p)
    return paths


# ===========================================================================
# JSON parsing & output classification
# ===========================================================================

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)
_TAGS_RE = re.compile(r"^Tags:\s*(.+)$", re.MULTILINE)
_TS_RANGE_RE = re.compile(r"^(\d{2}:\d{2}:\d{2})(?:\s*-\s*(\d{2}:\d{2}:\d{2}))?\s*(.*)$")
_SPEECH_RE = re.compile(r"^\[(zh|en|other)\]\s*(.*)$", re.IGNORECASE)
_SECTION_HEADERS = ("Self", "Others", "Environment", "Speech", "Psychology", "Tags")
_QUOTE_PAIRS = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})

_ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "self_actions": {"type": "array", "items": {"type": "object"}},
        "others": {"type": "array", "items": {"type": "object"}},
        "environment": {"type": "string"},
        "speech": {"type": "array", "items": {"type": "object"}},
        "psychology": {
            "type": "object",
            "properties": {
                "awareness": {"type": "string", "enum": ["low", "medium", "high"]},
                "emotion": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["awareness", "emotion"],
        },
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["self_actions", "psychology", "tags"],
}


class ParseFailed(Exception):
    def __init__(self, reason: str, raw: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.raw = raw


def _strip_bullet(ln: str) -> str:
    return re.sub(r"^[-*\u2022\u2013]+\s*", "", ln).strip()


def _split_sections(text: str) -> dict:
    sections: dict = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        header = None
        m = re.match(r"^(?:#+\s*)?\[?\s*([A-Za-z]+)\s*\]?\s*:\s*$", line)
        if m and m.group(1).capitalize() in _SECTION_HEADERS:
            header = m.group(1).capitalize()
        if header is None:
            m2 = re.match(r"^(?:#+\s*)?\[\s*([A-Za-z]+)\s*\]$", line)
            if m2 and m2.group(1).capitalize() in _SECTION_HEADERS:
                header = m2.group(1).capitalize()
        if header is not None:
            current = header
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections.setdefault(current, []).append(line)
    return sections


def _parse_actions(lines: list) -> list:
    out = []
    for ln in lines:
        ln = _strip_bullet(ln)
        if not ln:
            continue
        m = _TS_RANGE_RE.match(ln)
        if m and re.fullmatch(r"\d{2}:\d{2}:\d{2}", m.group(1)):
            out.append({"time": m.group(1), "time_end": m.group(2) or "", "text": m.group(3).strip()})
        else:
            out.append({"time": "", "time_end": "", "text": ln})
    return out


def _parse_speech(lines: list) -> list:
    out = []
    for ln in lines:
        ln = _strip_bullet(ln).translate(_QUOTE_PAIRS)
        if not ln:
            continue
        m = _SPEECH_RE.match(ln)
        if m:
            lang = m.group(1).lower()
            rest = m.group(2).strip()
            speaker = ""
            sm = re.match(r'^([^:"]+?):\s*(.*)$', rest)
            if sm and sm.group(2).startswith('"'):
                speaker = sm.group(1).strip()
                rest = sm.group(2).strip()
            qm = re.match(r'^"([^"]*)"', rest)
            text = qm.group(1) if qm else rest.strip('"').strip()
            out.append({"lang": lang, "speaker": speaker, "text": text})
        else:
            out.append({"lang": "", "speaker": "", "text": ln})
    return out


def _parse_psychology(lines: list) -> dict:
    psych = {"awareness": "", "emotion": "", "note": ""}
    for ln in lines:
        ln = _strip_bullet(ln)
        if not ln:
            continue
        low = ln.lower()
        if low.startswith("awareness"):
            val = ln.split(":", 1)[-1].strip().lower().strip("`*")
            psych["awareness"] = val.split()[0] if val else ""
        elif low.startswith("emotion"):
            psych["emotion"] = ln.split(":", 1)[-1].strip().lower().strip("`*")
        elif low.startswith("note"):
            psych["note"] = ln.split(":", 1)[-1].strip()
    return psych


def parse_layered_caption(text: str) -> dict:
    """Fallback parser for sectioned text (used if the model ignores json_object
    mode and emits [Self]/[Psychology] style output). Returns the same dict shape
    as parse_json_caption."""
    tags = []
    body = text
    tm = _TAGS_RE.search(text)
    if tm:
        tags = [t.strip() for t in tm.group(1).split(",") if t.strip()]
        body = text[: tm.start()].rstrip()
    sections = _split_sections(body)
    if not tags and "Tags" in sections:
        tag_blob = ", ".join(sections["Tags"])
        tags = [t.strip() for t in re.split(r"[,\uff0c]", tag_blob) if t.strip()]
        sections.pop("Tags", None)
    return {
        "self_actions": _parse_actions(sections.get("Self", [])),
        "others": _parse_actions(sections.get("Others", [])),
        "environment": " ".join(sections.get("Environment", [])).strip(),
        "speech": _parse_speech(sections.get("Speech", [])),
        "psychology": _parse_psychology(sections.get("Psychology", [])),
        "tags": tags,
    }


def parse_json_caption(content: str) -> dict:
    """Primary parser. Strips code fences, json.loads, validates with jsonschema,
    normalizes to the canonical shape. Raises ParseFailed on any error."""
    text = content.strip()
    if not text:
        raise ParseFailed("empty content", content)
    m = _JSON_FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ParseFailed(f"json decode: {e.msg}", content)
    if not isinstance(data, dict):
        raise ParseFailed(f"top-level not object (got {type(data).__name__})", content)
    if jsonschema is not None:
        try:
            jsonschema.validate(instance=data, schema=_ANNOTATION_SCHEMA)
        except Exception as e:
            raise ParseFailed(f"schema: {e}", content)
    else:
        psych = data.get("psychology")
        if not (isinstance(psych, dict) and psych.get("awareness") in ("low", "medium", "high")):
            raise ParseFailed("psychology.awareness missing or invalid", content)
        if not isinstance(data.get("self_actions"), list):
            raise ParseFailed("self_actions not array", content)

    def _norm_action(a):
        if not isinstance(a, dict):
            return {"time": "", "time_end": "", "text": str(a)}
        return {"time": str(a.get("time", "") or ""), "time_end": str(a.get("time_end", "") or ""),
                "text": str(a.get("text", "") or "")}

    def _norm_speech(s):
        if not isinstance(s, dict):
            return {"lang": "", "speaker": "", "text": str(s)}
        return {"lang": str(s.get("lang", "") or ""), "speaker": str(s.get("speaker", "") or ""),
                "text": str(s.get("text", "") or "")}

    psych = data.get("psychology") or {}
    return {
        "self_actions": [_norm_action(a) for a in data.get("self_actions", [])],
        "others": [_norm_action(a) for a in data.get("others", [])],
        "environment": str(data.get("environment", "") or ""),
        "speech": [_norm_speech(s) for s in data.get("speech", [])],
        "psychology": {"awareness": str(psych.get("awareness", "") or ""),
                       "emotion": str(psych.get("emotion", "") or ""),
                       "note": str(psych.get("note", "") or "")},
        "tags": [str(t) for t in data.get("tags", [])],
    }


# Outcome categories for a model response.
OUTCOME_OK = "ok"
OUTCOME_SAFETY = "safety_rejection"
OUTCOME_PARSE_FAILED = "parse_failed"
OUTCOME_EMPTY = "empty"


def classify_output(content: str, out_tokens: int) -> str:
    """Cheap pre-classification (does not parse). The actual JSON parse is
    attempted in build_records_from_response so it can capture the fallback."""
    if not content or not content.strip():
        return OUTCOME_EMPTY
    if out_tokens <= 25:
        stripped = content.strip()
        has_structure = (stripped.startswith("{") or "[Self]" in stripped
                         or "awareness" in stripped.lower() or '"psychology"' in stripped)
        if not has_structure:
            return OUTCOME_SAFETY
    return OUTCOME_OK


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class CaptionRecord:
    clip_id: str
    global_idx: int
    day: int
    user: str
    duration_s: float
    narrative: str               # raw model output (full JSON text), for traceability
    tags: list
    model: str
    ts_captioned: str
    slice_path: str
    tokens: dict
    self_actions: list = None    # [{time, time_end, text}]
    others: list = None          # [{time, time_end, text}]
    environment: str = ""
    speech: list = None          # [{lang, speaker, text}]
    psychology: dict = None      # {awareness, emotion, note}
    clip_kind: str = "30s"       # "30s" | "60s" | "segment_open" | "10s"
    output_format: str = "json"  # "json" | "fallback_sectioned"
    recovery: str = ""           # "ok" | "fallback_sectioned" | "10s_slices"


@dataclass
class UsageRecord:
    clip_id: str
    global_idx: int
    attempt: int
    latency_s: float
    recovery: str                # "ok" | "safety_rejection" | "parse_failed" | "empty" | ...
    usage: dict
    raw_content: str = ""        # captured for non-ok outcomes, for diagnostics


# ===========================================================================
# API helpers
# ===========================================================================

def b64_video(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def make_user_message(video_path: Path, clip_row: pd.Series, is_10s: bool = False) -> dict:
    src = clip_row["src_file"]
    ts_raw = Path(src).stem.split("_")[-1]
    day_label = f"DAY{int(clip_row['day'])}"
    start_hms = f"{ts_raw[0:2]}:{ts_raw[2:4]}:{ts_raw[4:6]}"
    clip_kind = clip_row.get("clip_kind", "30s")
    if is_10s:
        duration_desc = "10-second"
    elif clip_kind == "60s":
        duration_desc = "~60-second"
    else:
        duration_desc = "~30-second"
    task = USER_TASK_TMPL.format(
        clip_id=clip_row["clip_id"], src_file=src, day_label=day_label,
        start_hms=start_hms, duration_desc=duration_desc,
    )
    return {"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64_video(video_path)}"}, "fps": 2},
        {"type": "text", "text": task},
    ]}


def extract_usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if u is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0, "raw": {}}
    raw = u.model_dump() if hasattr(u, "model_dump") else dict(u)
    cached = 0
    pd_ = raw.get("prompt_tokens_details") or {}
    if isinstance(pd_, dict):
        cached = pd_.get("cached_tokens", 0) or 0
    return {"prompt_tokens": raw.get("prompt_tokens", 0), "completion_tokens": raw.get("completion_tokens", 0),
            "total_tokens": raw.get("total_tokens", 0), "cached_tokens": int(cached), "raw": raw}


def build_records_from_response(content, usage, latency, attempt, clip_row, model):
    """Turn a raw model response into (CaptionRecord | None, UsageRecord).

    On any non-ok outcome the raw content is captured in UsageRecord.raw_content.
    recovery field carries the specific outcome label."""
    clip_id = clip_row["clip_id"]
    gid = int(clip_row["global_idx"])
    out_tokens = usage["completion_tokens"]

    outcome = classify_output(content, out_tokens)
    if outcome in (OUTCOME_EMPTY, OUTCOME_SAFETY):
        return None, UsageRecord(clip_id, gid, attempt, round(latency, 3), outcome, usage, content)

    try:
        layered = parse_json_caption(content)
        output_format = "json"
    except ParseFailed:
        # Fallback: maybe sectioned text despite json_mode.
        if "[Self]" in content or "[Psychology]" in content:
            try:
                layered = parse_layered_caption(content)
                output_format = "fallback_sectioned"
            except Exception:
                return None, UsageRecord(clip_id, gid, attempt, round(latency, 3),
                                         OUTCOME_PARSE_FAILED, usage, content)
        else:
            return None, UsageRecord(clip_id, gid, attempt, round(latency, 3),
                                     OUTCOME_PARSE_FAILED, usage, content)

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    recovery = "fallback_sectioned" if output_format == "fallback_sectioned" else "ok"
    cap_rec = CaptionRecord(
        clip_id=clip_id, global_idx=gid, day=int(clip_row["day"]), user=clip_row["user"],
        duration_s=float(clip_row["duration"]), narrative=content.strip(), tags=layered["tags"],
        model=model, ts_captioned=ts, slice_path=clip_row["slice_path"],
        tokens={"in": usage["prompt_tokens"], "out": usage["completion_tokens"], "cached": usage["cached_tokens"]},
        self_actions=layered["self_actions"], others=layered["others"], environment=layered["environment"],
        speech=layered["speech"], psychology=layered["psychology"],
        clip_kind=clip_row.get("clip_kind", "30s"), output_format=output_format, recovery=recovery,
    )
    use_rec = UsageRecord(clip_id, gid, attempt, round(latency, 3), recovery, usage)
    return cap_rec, use_rec


# ===========================================================================
# Rate limiter
# ===========================================================================

class SlidingWindowRateLimiter:
    """Thread-safe sliding-window RPM limiter. acquire() blocks until fewer than
    max_rpm requests have been started in the trailing 60 seconds."""

    def __init__(self, max_rpm: int):
        self.max_rpm = max(1, max_rpm)
        self._timestamps: collections.deque = collections.deque()
        self._cond = threading.Condition(threading.Lock())

    def acquire(self) -> None:
        with self._cond:
            while True:
                now = time.monotonic()
                while self._timestamps and self._timestamps[0] <= now - 60.0:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_rpm:
                    self._timestamps.append(now)
                    return
                wait = self._timestamps[0] + 60.0 - now + 0.05
                self._cond.wait(timeout=max(wait, 0.01))


# ===========================================================================
# Source selection & 60s pairing
# ===========================================================================

def _fname_to_seconds(name: str) -> float:
    ts_raw = Path(name).stem.split("_")[-1]
    hh, mm, ss, cs = int(ts_raw[0:2]), int(ts_raw[2:4]), int(ts_raw[4:6]), int(ts_raw[6:8])
    return hh * 3600 + mm * 60 + ss + cs / 100.0


def _hhmm_to_seconds(hhmm: str) -> int:
    if len(hhmm) != 4 or not hhmm.isdigit():
        raise ValueError(f"--start-time/--end-time must be HHMM (4 digits), got {hhmm!r}")
    return int(hhmm[0:2]) * 3600 + int(hhmm[2:4]) * 60


def _ts_label(name: str) -> str:
    ts_raw = Path(name).stem.split("_")[-1]
    return f"{ts_raw[0:2]}:{ts_raw[2:4]}:{ts_raw[4:6]}"


def select_source_files(src_root, participant, day, start_time, end_time):
    """Glob + time-filter source MP4s. Returns (files, break_indices).

    src_root is the ROOT of the video tree (e.g. ./videos or D:\\...\\EgoLife).
    The actual clips live under {src_root}/{participant}/DAY{day}/.
    """
    day_dir = src_root / participant / f"DAY{day}"
    glob_pat = f"DAY{day}_{participant}_*.mp4"
    files = sorted(day_dir.glob(glob_pat))
    if not files:
        raise FileNotFoundError(
            f"no source files matching {day_dir / glob_pat} "
            f"(check --src-dir points at the root containing {participant}/DAY{day}/)")
    t_start = _hhmm_to_seconds(start_time) if start_time else None
    t_end = _hhmm_to_seconds(end_time) if end_time else None
    if t_start is not None or t_end is not None:
        kept = []
        for f in files:
            cs = _fname_to_seconds(f.name)
            ce = cs + 30.0
            if t_start is not None and ce <= t_start:
                continue
            if t_end is not None and cs >= t_end:
                continue
            kept.append(f)
        files = kept
    if not files:
        raise FileNotFoundError(
            f"no source files in {day_dir} fall within {start_time or '00:00'}-{end_time or '23:59'}")
    breaks = []
    for i in range(1, len(files)):
        if _fname_to_seconds(files[i].name) - _fname_to_seconds(files[i - 1].name) > _GAP_THRESHOLD_S:
            breaks.append(i)
    return files, breaks


class ClipUnit:
    """A preprocess unit: single source (30s/segment_open) or a 60s pair."""
    __slots__ = ("kind", "srcs", "clip_id", "start_ts")

    def __init__(self, kind, srcs, clip_id, start_ts):
        self.kind = kind        # "30s" | "60s" | "segment_open"
        self.srcs = srcs
        self.clip_id = clip_id
        self.start_ts = start_ts


def pair_into_minutes(files, breaks, participant, day, clip_duration):
    """Group source files into ClipUnits. clip_duration=60 pairs consecutive
    :00+:30 files of the same minute (never across a recording break)."""
    break_set = set(breaks)
    units = []
    i, n = 0, len(files)
    while i < n:
        f = files[i]
        ts_raw = Path(f.stem).stem.split("_")[-1]
        ss_field = ts_raw[4:6]
        is_segment_open = ss_field not in ("00", "30")
        if (clip_duration == 60 and ss_field == "00" and (i + 1) < n
                and (i + 1) not in break_set):
            nxt = files[i + 1]
            nxt_ts = Path(nxt.stem).stem.split("_")[-1]
            if nxt_ts[0:4] == ts_raw[0:4] and nxt_ts[4:6] == "30":
                clip_id = f"DAY{day}_{participant}_{ts_raw[0:6]}_60s"
                units.append(ClipUnit("60s", [f, nxt], clip_id, _ts_label(f.name)))
                i += 2
                continue
        kind = "segment_open" if is_segment_open else "30s"
        units.append(ClipUnit(kind, [f], f"DAY{day}_{participant}_{ts_raw}", _ts_label(f.name)))
        i += 1
    return units


# ===========================================================================
# Workers (producer-consumer)
# ===========================================================================

class ApiCallConfig:
    __slots__ = ("thinking", "max_completion_tokens", "json_mode")

    def __init__(self, thinking, max_completion_tokens, json_mode):
        self.thinking = thinking
        self.max_completion_tokens = max_completion_tokens
        self.json_mode = json_mode


class PreprocessJob:
    __slots__ = ("unit", "idx", "day", "user", "skip_if_exists", "resolution",
                 "fps", "crf", "audio_k", "slices_dir", "out_dir")

    def __init__(self, unit, idx, *, day, user, skip_if_exists, resolution, fps, crf, audio_k,
                 slices_dir, out_dir):
        self.unit = unit
        self.idx = idx
        self.day = day
        self.user = user
        self.skip_if_exists = skip_if_exists
        self.resolution = resolution
        self.fps = fps
        self.crf = crf
        self.audio_k = audio_k
        self.slices_dir = slices_dir
        self.out_dir = out_dir


class PreprocessResult:
    __slots__ = ("clip_id", "global_idx", "slice_path", "row", "error")

    def __init__(self, *, clip_id, global_idx, slice_path, row, error=""):
        self.clip_id = clip_id
        self.global_idx = global_idx
        self.slice_path = slice_path
        self.row = row
        self.error = error


def _preprocess_worker(job_q, out_q, log):
    while True:
        job = job_q.get()
        if job is None:
            job_q.task_done()
            return
        unit = job.unit
        clip_id = unit.clip_id
        try:
            slice_path = job.slices_dir / f"{clip_id}.mp4"
            if unit.kind == "60s":
                if job.skip_if_exists and slice_path.exists() and slice_path.stat().st_size > 0:
                    log.debug(f"[preprocess] {clip_id} merged slice exists, skipping")
                else:
                    halves = []
                    for k, src in enumerate(unit.srcs):
                        h = job.slices_dir / f"{clip_id}_half{k + 1}.mp4"
                        if not (job.skip_if_exists and h.exists() and h.stat().st_size > 0):
                            ffmpeg_reencode(src, h, resolution=job.resolution, fps=job.fps,
                                            crf=job.crf, audio_k=job.audio_k)
                        halves.append(h)
                    concat_two_clips(halves[0], halves[1], slice_path, clip_id)
                duration = ffprobe_duration(slice_path)
                ts = unit.start_ts
                row = pd.Series({
                    "clip_id": clip_id, "day": job.day, "user": job.user, "start_ts": ts,
                    "end_ts": f"{ts}+{duration:.3f}s",
                    "src_file": "+".join(Path(s).name for s in unit.srcs), "clip_idx": 1,
                    "global_idx": job.idx + 1, "duration": round(duration, 3),
                    "is_day_open": False, "is_day_close": False, "status": "pending",
                    "slice_path": str(slice_path.relative_to(job.out_dir)), "clip_kind": "60s",
                })
            else:
                src = unit.srcs[0]
                if job.skip_if_exists and slice_path.exists() and slice_path.stat().st_size > 0:
                    log.debug(f"[preprocess] {clip_id} slice exists, skipping encode")
                else:
                    ffmpeg_reencode(src, slice_path, resolution=job.resolution, fps=job.fps,
                                    crf=job.crf, audio_k=job.audio_k)
                duration = ffprobe_duration(slice_path)
                ts_base = parse_ts(src.name)
                is_open = (job.idx == 0) and (unit.kind == "segment_open")
                row = pd.Series({
                    "clip_id": clip_id, "day": job.day, "user": job.user, "start_ts": ts_base,
                    "end_ts": f"{ts_base}+{duration:.3f}s", "src_file": src.name, "clip_idx": 1,
                    "global_idx": job.idx + 1, "duration": round(duration, 3),
                    "is_day_open": is_open, "is_day_close": False, "status": "pending",
                    "slice_path": str(slice_path.relative_to(job.out_dir)), "clip_kind": unit.kind,
                })
            out_q.put(PreprocessResult(clip_id=clip_id, global_idx=job.idx + 1,
                                       slice_path=slice_path, row=row))
        except Exception as e:
            log.error(f"[preprocess] {clip_id} failed: {type(e).__name__}: {e}")
            out_q.put(PreprocessResult(clip_id=clip_id, global_idx=job.idx + 1,
                                       slice_path=job.slices_dir / f"{clip_id}.mp4", row=None,
                                       error=f"{type(e).__name__}: {e}"))
        finally:
            job_q.task_done()


class ApiJob:
    __slots__ = ("row", "slice_path")

    def __init__(self, row, slice_path):
        self.row = row
        self.slice_path = slice_path


class ApiResult:
    __slots__ = ("global_idx", "clip_id", "caption_records", "usage_records", "failures")

    def __init__(self, *, global_idx=0, clip_id="", caption_records=None, usage_records=None, failures=None):
        self.global_idx = global_idx
        self.clip_id = clip_id
        self.caption_records = caption_records or []
        self.usage_records = usage_records or []
        self.failures = failures or []


def _call_api_limited(client, model, messages, log, clip_id, limiter, cfg):
    """One HTTP call gated by the rate limiter. thinking=enabled omits
    temperature (Mimo forces its own defaults under deep thinking)."""
    last_exc = None
    kwargs = dict(model=model, messages=messages,
                  max_completion_tokens=cfg.max_completion_tokens,
                  extra_body={"thinking": {"type": cfg.thinking}})
    if cfg.thinking == "disabled":
        kwargs["temperature"] = 1.0
    if cfg.json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(1, 3):
        limiter.acquire()
        try:
            t0 = time.time()
            resp = client.chat.completions.create(**kwargs)
            return resp, time.time() - t0, attempt
        except Exception as e:
            last_exc = e
            log.debug(f"[{clip_id}] API attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt < 2:
                time.sleep(min(2 ** attempt, 30))
    raise last_exc


def _process_one_clip_limited(client, model, video_path, clip_row, is_10s, log, limiter, cfg):
    messages = [{"role": "system", "content": SYSTEM_MSG},
                make_user_message(video_path, clip_row, is_10s=is_10s)]
    resp, latency, attempt = _call_api_limited(client, model, messages, log, clip_row["clip_id"], limiter, cfg)
    usage = extract_usage(resp)
    content = resp.choices[0].message.content or ""
    cap_rec, use_rec = build_records_from_response(content, usage, latency, attempt, clip_row, model)
    return cap_rec, use_rec, content


def _api_worker(in_q, out_q, client, model, limiter, cfg, tmp_10s_dir, log, worker_id):
    while True:
        job = in_q.get()
        if job is None:
            in_q.task_done()
            return
        row, slice_path = job.row, job.slice_path
        clip_id, gid = row["clip_id"], int(row["global_idx"])
        caption_records, usage_records, failures = [], [], []

        if not slice_path.exists():
            log.error(f"[{clip_id}] slice not found: {slice_path}")
            failures.append({"clip_id": clip_id, "global_idx": gid, "error": "slice_not_found"})
            out_q.put(ApiResult(global_idx=gid, clip_id=clip_id, caption_records=caption_records,
                                usage_records=usage_records, failures=failures))
            in_q.task_done()
            continue

        try:
            cap_rec, use_rec, content = _process_one_clip_limited(
                client, model, slice_path, row, False, log, limiter, cfg)
            use_rec2 = None
            if cap_rec is None:
                reason = use_rec.recovery
                # Mimo often rejects the very first video request; the second
                # request hits the cheap prefix cache and almost always succeeds.
                # Treat this as normal, not a crisis.
                use_rec.recovery = "first_attempt_rejection"
                usage_records.append(use_rec)
                log.info(f"[{clip_id}] first attempt returned '{reason}'; retrying once "
                         f"(normal for Mimo, retry is cheap via prefix cache)")
                try:
                    cap_rec, use_rec2, content2 = _process_one_clip_limited(
                        client, model, slice_path, row, False, log, limiter, cfg)
                except Exception as e:
                    log.warning(f"[{clip_id}] retry crashed: {type(e).__name__}: {e}")
                    cap_rec = None
                    use_rec2 = None

            if cap_rec is None:
                reason2 = use_rec2.recovery if use_rec2 is not None else "retry_failed"
                if use_rec2 is not None:
                    usage_records.append(use_rec2)
                log.warning(f"[{clip_id}] second attempt also failed ({reason2}); "
                            f"falling back to ~10s slices")
                try:
                    slice_paths = slice_to_10s(slice_path, clip_id, tmp_10s_dir)
                except Exception as e:
                    log.error(f"[{clip_id}] 10s slicing failed: {e}")
                    failures.append({"clip_id": clip_id, "global_idx": gid, "error": f"slice_failed: {e}",
                                     "reason": reason2, "raw_content_preview": (content or "")[:200]})
                    out_q.put(ApiResult(global_idx=gid, clip_id=clip_id, caption_records=caption_records,
                                        usage_records=usage_records, failures=failures))
                    in_q.task_done()
                    continue
                log.info(f"[{clip_id}] sliced into {len(slice_paths)} pieces; captioning each ~10s clip")
                ten_s_ok = 0
                for sp in slice_paths:
                    sub_row = row.copy()
                    sub_row["clip_id"] = f"{clip_id}_10s_{ten_s_ok + 1}"
                    sub_row["duration"] = 10.0
                    try:
                        sub_cap, sub_use, _ = _process_one_clip_limited(
                            client, model, sp, sub_row, True, log, limiter, cfg)
                    except Exception as e:
                        log.warning(f"[{clip_id}] 10s slice {sp.name} failed: {type(e).__name__}: {e}")
                        continue
                    if sub_cap is not None:
                        sub_cap.recovery = "10s_slices"
                        caption_records.append(sub_cap)
                        usage_records.append(sub_use)
                        ten_s_ok += 1
                        log.info(f"[{clip_id}] 10s slice {sp.name} succeeded")
                    else:
                        log.warning(f"[{clip_id}] 10s slice {sp.name} rejected/parse-failed")
                log.info(f"[{clip_id}] 10s fallback result: {ten_s_ok}/{len(slice_paths)} slices succeeded")
                if ten_s_ok == 0:
                    failures.append({"clip_id": clip_id, "global_idx": gid, "error": "all_attempts_failed",
                                     "reason": reason2, "raw_content_preview": (content or "")[:200]})
            else:
                caption_records.append(cap_rec)
                if use_rec2 is not None:
                    usage_records.append(use_rec2)
                elif not usage_records:
                    usage_records.append(use_rec)
        except Exception as e:
            log.error(f"[{clip_id}] worker crash: {type(e).__name__}: {e}")
            failures.append({"clip_id": clip_id, "global_idx": gid,
                             "error": f"worker_crash: {type(e).__name__}: {e}"})
        out_q.put(ApiResult(global_idx=gid, clip_id=clip_id, caption_records=caption_records,
                            usage_records=usage_records, failures=failures))
        in_q.task_done()


# ===========================================================================
# Ordered writer (single thread, strict JSONL)
# ===========================================================================

class OrderedWriter:
    """Flush results to jsonl in strict global_idx order. API workers finish out
    of order, so we buffer future results in a heap and emit when next_idx arrives."""

    def __init__(self, captions_path, usage_path, log, next_idx=1):
        self.captions_path = captions_path
        self.usage_path = usage_path
        self.log = log
        self.next_idx = next_idx
        self._heap = []
        self._seq = 0
        self._lock = threading.Lock()
        self.n_written = 0
        self.cap_f = captions_path.open("a", encoding="utf-8")
        self.use_f = usage_path.open("a", encoding="utf-8")

    def submit(self, result):
        with self._lock:
            self._seq += 1
            heapq.heappush(self._heap, (result.global_idx, self._seq, result))
            self._flush_locked()

    def _flush_locked(self):
        while self._heap and self._heap[0][0] <= self.next_idx:
            _, _, res = heapq.heappop(self._heap)
            for cap in res.caption_records:
                self.cap_f.write(json.dumps(asdict(cap), ensure_ascii=False) + "\n")
            for use in res.usage_records:
                self.use_f.write(json.dumps(asdict(use), ensure_ascii=False) + "\n")
            self.n_written += 1
            if res.global_idx >= self.next_idx:
                self.next_idx = res.global_idx + 1
            ids = [c.clip_id for c in res.caption_records] or ["<none>"]
            # Demoted to DEBUG: this fires per-flush and produces a bursty "wrote gid=..." dump
            # whenever the ordered heap catches up. Per-clip failure signal is already covered by
            # the API-worker warnings (retry / fallback / 10s-slice-failed); the heartbeat
            # progress log in main() gives the positive-feedback signal.
            if res.failures:
                self.log.debug(f"[writer] wrote gid={res.global_idx} caps={ids} failures={res.failures}")
            else:
                self.log.debug(f"[writer] wrote gid={res.global_idx} caps={ids}")

    def close(self):
        with self._lock:
            if self._heap:
                self.log.warning(f"[writer] {len(self._heap)} results never flushed (missing global_idx)")
            self.cap_f.close()
            self.use_f.close()


# ===========================================================================
# Logging helper that plays nicely with tqdm
# ===========================================================================

class TqdmLoggingHandler(logging.Handler):
    """Emit log records through tqdm.write() so messages stay above an
    active progress bar instead of destroying it."""

    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            if tqdm is not None:
                tqdm.write(msg, file=sys.stdout)
            else:
                sys.stdout.write(msg + "\n")
                sys.stdout.flush()
        except Exception:
            self.handleError(record)


# ===========================================================================
# Resume helpers
# ===========================================================================

def _iter_caption_records(captions_path):
    """Yield dict records from a (possibly multi-line tolerant) captions.jsonl."""
    if not captions_path.exists():
        return
    decoder = json.JSONDecoder()
    text = captions_path.read_text(encoding="utf-8")
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


def load_existing_clip_ids(captions_path):
    return {rec["clip_id"] for rec in _iter_caption_records(captions_path)}


def max_written_global_idx(captions_path):
    top = 0
    for rec in _iter_caption_records(captions_path):
        top = max(top, int(rec.get("global_idx", 0)))
    return top


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Self-contained Mimo captioning pipeline for EgoLife (JSON structured output).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Selection ---
    ap.add_argument("--participant", default="A1_JAKE")
    ap.add_argument("--day", type=int, default=1)
    ap.add_argument("--start-time", default=None, help="start HHMM inclusive, e.g. 1110")
    ap.add_argument("--end-time", default=None, help="end HHMM exclusive, e.g. 1130")
    ap.add_argument("--clip-duration", type=int, default=30, choices=[30, 60],
                    help="30 = per source file (recommended). 60 = merge pairs "
                         "(WARNING: model under-covers 60s, often only the first ~6s).")
    # --- Paths ---
    ap.add_argument("--src-dir", type=Path, default=None,
                    help="ROOT of the video tree (default: ./videos). Clips are read from "
                         "{src-dir}/{participant}/DAY{day}/")
    ap.add_argument("--out", type=Path, default=None,
                    help="output jsonl (default: ./captions/{participant}/DAY{day}/{start}-{end}.jsonl)")
    # --- Encoding ---
    ap.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--crf", type=int, default=DEFAULT_CRF)
    ap.add_argument("--audio-k", type=int, default=DEFAULT_AUDIO_BITRATE_K)
    # --- Model / API ---
    ap.add_argument("--model", default="mimo-v2.5")
    ap.add_argument("--base-url", default="https://api.xiaomimimo.com/v1")
    ap.add_argument("--api-key-env", default="MIMO_API_KEY")
    ap.add_argument("--env-file", type=Path, default=None,
                    help=".env file to load MIMO_API_KEY from (default: search CWD + script dir)")
    ap.add_argument("--thinking", default=THINKING_DEFAULT, choices=["enabled", "disabled"],
                    help="reasoning mode. When enabled, raise --max-completion-tokens (e.g. 8192). "
                         "Under thinking Mimo ignores temperature/top_p.")
    ap.add_argument("--max-completion-tokens", type=int, default=MAX_COMPLETION_TOKENS)
    ap.add_argument("--no-json-mode", action="store_true",
                    help="disable response_format=json_object (debug; falls back to sectioned parser)")
    # --- Concurrency ---
    ap.add_argument("--max-rpm", type=int, default=90, help="global RPM cap (Mimo limit is 100)")
    ap.add_argument("--api-workers", type=int, default=6)
    ap.add_argument("--preprocess-workers", type=int, default=2)
    # --- Flow ---
    ap.add_argument("--skip-preprocess", action="store_true",
                    help="caption only, reading slices from the _cache/ dir")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                    help="skip clips already in the output file (default: on)")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--reset", action="store_true", help="delete output files before running")
    ap.add_argument("--reset-yes-i-know", action="store_true",
                    help="confirm --reset when the output already has >10 records (prevents accidental data loss)")
    ap.add_argument("--limit", type=int, default=None, help="only process first N clips (debug)")
    args = ap.parse_args()

    # --- Load API key ---
    env_paths = []
    if args.env_file:
        env_paths.append(args.env_file)
    env_paths += [Path.cwd() / ".env", ROOT / ".env"]
    for ep in env_paths:
        if ep.exists():
            load_dotenv(ep)
            break
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"ERROR: {args.api_key_env} not set. Put it in a .env file or export it.\n"
              f"Searched: {[str(p) for p in env_paths]}", file=sys.stderr)
        sys.exit(2)

    # --- Resolve paths ---
    # src_dir is the ROOT of the video tree; clips live under {src_dir}/{participant}/DAY{day}/.
    src_root = args.src_dir or (ROOT / "videos")
    time_tag = f"{args.start_time}-{args.end_time}" if (args.start_time and args.end_time) else "full"
    if args.out is None:
        # Mirror the input layout: captions/{participant}/DAY{day}/{time_tag}.jsonl
        out_dir_default = ROOT / "captions" / args.participant / f"DAY{args.day}"
        out_file = out_dir_default / f"{time_tag}.jsonl"
    else:
        out_file = args.out
    out_file.parent.mkdir(parents=True, exist_ok=True)
    captions_dir = out_file.parent
    stem = out_file.stem  # e.g. DAY1_1110-1130
    usage_path = captions_dir / f"{stem}_usage.jsonl"
    summary_path = captions_dir / f"{stem}_summary.json"
    log_path = captions_dir / f"{stem}_run.log"
    cache_dir = captions_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    slices_dir = cache_dir / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    tmp_10s_dir = cache_dir / "tmp_10s_slices"
    tmp_10s_dir.mkdir(parents=True, exist_ok=True)
    clips_parquet = cache_dir / "clips.parquet"

    if args.reset:
        # Safety: refuse to --reset if the main output already has many records,
        # unless the user passes an explicit confirmation. Prevents accidental
        # loss of a long captioning run.
        existing = 0
        if out_file.exists():
            try:
                existing = sum(1 for line in out_file.open(encoding="utf-8") if line.strip().startswith("{"))
            except Exception:
                pass
        if existing > 10 and not getattr(args, "reset_confirm", False):
            print(f"ERROR: --reset would delete {out_file} which already has {existing} caption records.\n"
                  f"  If you really mean it, add --reset-yes-i-know.", file=sys.stderr)
            sys.exit(2)
        for p in [out_file, usage_path, summary_path, log_path]:
            if p.exists():
                p.unlink()

    # --- Logging ---
    log = logging.getLogger("caption_pipeline")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt); log.addHandler(fh)
    if tqdm is not None:
        th = TqdmLoggingHandler()
        th.setFormatter(fmt); log.addHandler(th)
    else:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt); log.addHandler(sh)

    json_mode = not args.no_json_mode
    cfg = ApiCallConfig(args.thinking, args.max_completion_tokens, json_mode)
    log.info(f"participant={args.participant} day={args.day} "
             f"time={args.start_time or '00:00'}-{args.end_time or '23:59'} "
             f"clip_duration={args.clip_duration}s src_root={src_root} "
             f"-> {src_root / args.participant / f'DAY{args.day}'}")
    log.info(f"model={args.model} thinking={args.thinking} "
             f"max_completion_tokens={args.max_completion_tokens} json_mode={json_mode}")
    log.info(f"output: {out_file}")

    # --- Decide clip rows / units ---
    if args.skip_preprocess:
        if not clips_parquet.exists():
            log.error(f"--skip-preprocess but {clips_parquet} not found")
            sys.exit(2)
        clips_df = pd.read_parquet(clips_parquet).sort_values("global_idx").reset_index(drop=True)
        log.info(f"--skip-preprocess: loaded {len(clips_df)} rows from {clips_parquet}")
        units = None
    else:
        try:
            files, breaks = select_source_files(src_root, args.participant, args.day,
                                                args.start_time, args.end_time)
        except FileNotFoundError as e:
            log.error(str(e))
            sys.exit(2)
        log.info(f"source: {len(files)} files, {len(breaks)} recording break(s)")
        units = pair_into_minutes(files, breaks, args.participant, args.day, args.clip_duration)
        n60 = sum(1 for u in units if u.kind == "60s")
        log.info(f"units: {len(units)} ({n60} x 60s merged, {len(units) - n60} standalone) "
                 f"@ {args.resolution}x{args.resolution}/{args.fps}fps")
        clips_df = None

    # --- Resume ---
    resume_idx = max_written_global_idx(out_file) if args.skip_existing else 0
    done_ids = load_existing_clip_ids(out_file) if args.skip_existing else set()
    log.info(f"resume: skip_existing={args.skip_existing} already_done={len(done_ids)} max_gid={resume_idx}")

    limiter = SlidingWindowRateLimiter(args.max_rpm)
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    produced_q: queue.Queue = queue.Queue()
    api_in_q: queue.Queue = queue.Queue(maxsize=args.api_workers * 2)
    result_q: queue.Queue = queue.Queue()
    writer = OrderedWriter(out_file, usage_path, log, next_idx=resume_idx + 1)

    # --- Stats ---
    all_usage, all_records, all_failures = [], [], []
    outcome_counts = {OUTCOME_OK: 0, OUTCOME_SAFETY: 0, OUTCOME_PARSE_FAILED: 0,
                      OUTCOME_EMPTY: 0, "fallback_sectioned": 0, "10s_slices": 0,
                      "first_attempt_rejection": 0}
    skipped_existing = 0
    t_start = time.time()

    # --- Preprocess pool (or skip) ---
    pp_job_q: queue.Queue = queue.Queue()
    pp_threads = []
    if args.skip_preprocess:
        rows = clips_df
        if args.limit:
            rows = rows.head(args.limit)
        fed = 0
        for _, row in rows.iterrows():
            clip_id = row["clip_id"]
            if args.skip_existing and (clip_id in done_ids or int(row["global_idx"]) <= resume_idx):
                skipped_existing += 1
                continue
            sp = row["slice_path"]
            api_in_q.put(ApiJob(row=row, slice_path=(cache_dir / sp) if not Path(sp).is_absolute() else Path(sp)))
            fed += 1
        expected_total = fed
        api_fed = fed
        for _ in range(args.api_workers):
            api_in_q.put(None)
        pp_done, pp_expected = True, 0
    else:
        for w in range(args.preprocess_workers):
            t = threading.Thread(target=_preprocess_worker, args=(pp_job_q, produced_q, log),
                                 name=f"pp-{w}", daemon=True)
            t.start()
            pp_threads.append(t)
        units_to_enqueue = units[:args.limit] if args.limit else units
        for i, unit in enumerate(units_to_enqueue):
            pp_job_q.put(PreprocessJob(unit=unit, idx=i, day=args.day, user=args.participant,
                                       skip_if_exists=(not args.reset), resolution=args.resolution,
                                       fps=args.fps, crf=args.crf, audio_k=args.audio_k,
                                       slices_dir=slices_dir, out_dir=cache_dir))
        expected_total = len(units_to_enqueue)
        for _ in range(args.preprocess_workers):
            pp_job_q.put(None)
        pp_done, pp_expected, api_fed = False, expected_total, 0

    # --- Progress bar ---
    pbar = None
    if tqdm is not None and expected_total > 0:
        pbar = tqdm(total=expected_total, unit="clip", desc="captioning",
                    mininterval=0.5, smoothing=0.3)
        pbar.refresh()

    # --- API worker pool ---
    api_threads = []
    for w in range(args.api_workers):
        t = threading.Thread(target=_api_worker,
                             args=(api_in_q, result_q, client, args.model, limiter, cfg,
                                   tmp_10s_dir, log, w), name=f"api-{w}", daemon=True)
        t.start()
        api_threads.append(t)

    # --- Bridge & main loop ---
    pp_results_for_parquet = []
    pp_received = 0
    results_received = 0

    def bridge_produced():
        nonlocal pp_received, pp_done, api_fed, skipped_existing
        drained = 0
        while True:
            try:
                pres = produced_q.get_nowait()
            except queue.Empty:
                break
            drained += 1
            if pres.row is None:
                all_failures.append({"clip_id": pres.clip_id, "global_idx": pres.global_idx, "error": pres.error})
                continue
            pp_results_for_parquet.append(pres.row)
            clip_id = pres.row["clip_id"]
            if args.skip_existing and (clip_id in done_ids or int(pres.row["global_idx"]) <= resume_idx):
                skipped_existing += 1
                continue
            api_in_q.put(ApiJob(row=pres.row, slice_path=pres.slice_path))
            api_fed += 1
        pp_received += drained
        if not args.skip_preprocess and pp_received >= pp_expected:
            pp_done = True

    # Progress feedback: tqdm bar updates per finished clip. If tqdm is not
    # installed we fall back to the old 60-second heartbeat log line.
    PROGRESS_INTERVAL_S = 60.0
    last_progress_t = time.time()
    pbar_total_adjusted = False

    def _update_pbar():
        if pbar is None:
            return
        elapsed_min = max((time.time() - t_start) / 60.0, 1e-9)
        rpm = results_received / elapsed_min
        remaining = max((pbar.total or expected_total) - pbar.n, 0)
        eta_min = remaining / rpm if rpm > 0 else 0.0
        pbar.set_postfix(
            ok=len(all_records),
            failed=len(all_failures),
            rpm=f"{rpm:.1f}",
            eta=f"{eta_min:.1f}min",
            refresh=False,
        )
        pbar.update(1)

    while True:
        if not args.skip_preprocess:
            bridge_produced()
            if pbar is not None and pp_done and not pbar_total_adjusted:
                # Preprocessing revealed how many clips were skipped-existing;
                # shrink the bar so it actually reaches 100%.
                if api_fed != pbar.total:
                    pbar.total = max(api_fed, pbar.n)
                    pbar.refresh()
                pbar_total_adjusted = True
        now = time.time()
        if pbar is None and now - last_progress_t >= PROGRESS_INTERVAL_S and expected_total > 0:
            elapsed = now - t_start
            rate = results_received / max(elapsed / 60, 1e-9)
            eta_min = (expected_total - results_received) / rate if rate > 0 else float("inf")
            if args.skip_preprocess:
                log.info(f"[progress] api={results_received}/{expected_total} "
                         f"ok={len(all_records)} failed={len(all_failures)} "
                         f"elapsed={elapsed/60:.1f}min rpm={rate:.1f} eta={eta_min:.1f}min")
            else:
                log.info(f"[progress] preprocess={pp_received}/{pp_expected} "
                         f"api={results_received}/{api_fed} "
                         f"ok={len(all_records)} failed={len(all_failures)} "
                         f"elapsed={elapsed/60:.1f}min rpm={rate:.1f} eta={eta_min:.1f}min")
            last_progress_t = now
        try:
            res = result_q.get(timeout=0.2)
        except queue.Empty:
            if pp_done and api_fed <= results_received:
                break
            continue
        results_received += 1
        for cap in res.caption_records:
            all_records.append(cap)
        for use in res.usage_records:
            all_usage.append(use)
            outcome_counts[use.recovery if use.recovery in outcome_counts else OUTCOME_OK] += 1
            if use.recovery == "":
                outcome_counts[OUTCOME_OK] += 1
        for cap in res.caption_records:
            if cap.recovery == "10s_slices":
                outcome_counts["10s_slices"] += 1
                break
        all_failures.extend(res.failures)
        writer.submit(res)
        _update_pbar()
        if pp_done and results_received >= api_fed:
            break

    for _ in range(args.api_workers):
        api_in_q.put(None)
    for t in api_threads:
        t.join(timeout=5)
    for t in pp_threads:
        t.join(timeout=5)
    if pbar is not None:
        pbar.close()
    writer.close()
    elapsed = time.time() - t_start

    if pp_results_for_parquet:
        try:
            pd.DataFrame([r.to_dict() for r in pp_results_for_parquet]).to_parquet(clips_parquet, index=False)
        except Exception as e:
            log.warning(f"could not write {clips_parquet}: {e}")

    total_in = sum(u.usage["prompt_tokens"] for u in all_usage)
    total_out = sum(u.usage["completion_tokens"] for u in all_usage)
    total_cached = sum(u.usage["cached_tokens"] for u in all_usage)
    n_ok = len(all_records)
    summary = {
        "model": args.model, "participant": args.participant, "day": args.day,
        "time_range": {"start": args.start_time, "end": args.end_time},
        "clip_duration": args.clip_duration, "thinking": args.thinking, "json_mode": json_mode,
        "max_completion_tokens": args.max_completion_tokens, "temperature": 1.0,
        "max_rpm": args.max_rpm, "api_workers": args.api_workers,
        "preprocess_workers": args.preprocess_workers,
        "skip_preprocess": args.skip_preprocess, "skip_existing": args.skip_existing,
        "n_clips_input": expected_total, "n_clips_ok": n_ok, "n_clips_failed": len(all_failures),
        "n_skipped_existing": skipped_existing, "outcomes": dict(outcome_counts),
        "n_safety_rejection": outcome_counts[OUTCOME_SAFETY],
        "n_parse_failed": outcome_counts[OUTCOME_PARSE_FAILED],
        "n_empty": outcome_counts[OUTCOME_EMPTY],
        "n_fallback_sectioned": outcome_counts["fallback_sectioned"],
        "n_recovery_10s": outcome_counts["10s_slices"],
        "n_first_attempt_rejection": outcome_counts["first_attempt_rejection"],
        "elapsed_s": round(elapsed, 2),
        "effective_rpm": round(results_received / max(elapsed / 60, 1e-9), 2),
        "tokens": {"total_in": total_in, "total_out": total_out, "total_cached": total_cached},
        "round_breakdown": [
            {"clip_id": u.clip_id, "global_idx": u.global_idx, "in": u.usage["prompt_tokens"],
             "out": u.usage["completion_tokens"], "cached": u.usage["cached_tokens"],
             "latency_s": u.latency_s, "recovery": u.recovery,
             "raw_content_preview": (u.raw_content or "")[:200]}
            for u in all_usage
        ],
        "failures": all_failures,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    unique = len(set(c.narrative for c in all_records)) if all_records else 0
    log.info(f"done. ok={n_ok} failed={len(all_failures)} skipped={skipped_existing} "
             f"first_rejection={outcome_counts['first_attempt_rejection']} "
             f"safety={outcome_counts[OUTCOME_SAFETY]} parse_failed={outcome_counts[OUTCOME_PARSE_FAILED]} "
             f"empty={outcome_counts[OUTCOME_EMPTY]} recovered_10s={outcome_counts['10s_slices']}")
    log.info(f"  elapsed={elapsed:.1f}s ({elapsed/60:.1f}min)  "
             f"effective_rpm={results_received / max(elapsed / 60, 1e-9):.1f}")
    log.info(f"  tokens: in={total_in} out={total_out} cached={total_cached}")
    if n_ok:
        log.info(f"  unique narratives: {unique}/{n_ok}")
    log.info(f"  captions -> {out_file}")
    log.info(f"  summary  -> {summary_path}")


if __name__ == "__main__":
    main()
