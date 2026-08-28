"""caption_pipeline_multiturn.py — multi-turn Mimo captioning pipeline for EgoLife.

Implements the spec in caption_pipeline_multiturn.md. Each clip (~10s piece) is
annotated through a chain of narrow, single-question turns, run back-to-back by
one worker so the video prefix stays cache-hot (cache has a TTL — never sweep
turn-by-turn across the whole library):

    T1  perception x k votes  -> merge (object/person lexicon + transcript + sounds)
    [T1c screen/text-surface reading, conditional: native-res keyframes as images]
    T2  behavior  (atomic actions / env_changes, lexicon-anchored, intent-banned)
    T3  psychology (context-isolated latent state)
    T4  causal integration (text-only; 8 edge types, counterfactual strength)
    -> code-assembled final record (no LLM merge)

All turns share ONE system prompt; each user message is laid out
[media -> verbatim prior-turn context blocks -> turn task text last] to maximize
prefix-cache hits (cached input is 1/50 the price). Whether the server really
preserves media-first ordering is measured via usage.cached_tokens (spec §6.0
Plan A vs Plan B).

Per-turn raw outputs are persisted to {stem}_t{N}.jsonl as soon as they finish,
so resume granularity is per turn and a crash loses nothing.

Usage:
    python caption_pipeline_multiturn.py --participant A1_JAKE --day 1
    python caption_pipeline_multiturn.py --limit 1 --clip-duration 30 \
        --start-time 1121 --end-time 1122 \
        --out captions/_test/multiturn/1121-1122.jsonl      # smoke test
"""
from __future__ import annotations

import argparse
import base64
import collections
import heapq
import json
import logging
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ROOT = Path(__file__).resolve().parent

# ===========================================================================
# Constants
# ===========================================================================

DEFAULT_VOTES = 2
NATIVE_FRAME_FPS = 1.0     # T1c keyframe sampling (~1 frame/sec, native resolution)

DEFAULT_RESOLUTION = 1024
DEFAULT_FPS = 2
DEFAULT_CRF = 28
DEFAULT_AUDIO_BITRATE_K = 64

_GAP_THRESHOLD_S = 35.0

_TURN_NAMES = ("t1", "t1c", "t2", "t3", "t4")
_TURN_PREREQ = {"t1c": {"t1"}, "t2": {"t1"}, "t3": {"t1", "t2"}, "t4": {"t1", "t2", "t3"}}

_CAUSAL_TYPES = ("env->action", "env->emotion", "emotion->action", "action->env",
                 "other->action", "other->env", "other->emotion", "env->env")
_STRENGTH_LEVELS = ("strong", "moderate", "weak")

# ===========================================================================
# Prompts — one SHARED system message (identical bytes for every call, so the
# prefix cache never diverges at token 0) + per-turn task texts that go at the
# END of the user message, after the media and the verbatim context blocks.
# ===========================================================================

SHARED_SYSTEM_MSG = """You are a dense first-person life-log captioner producing SIMULATION-GRADE structured annotations
for a digital twin of the wearer's daily life. The footage is from participant A1_JAKE wearing
Meta Aria glasses; people may speak Chinese or English. You will receive ONE narrow task per
request (perception / behavior / psychology / causal analysis); follow exactly that task.

# Watermark anchor
Every video frame carries a watermark TOP-RIGHT showing time and day, "HH:MM:SS:FF DAYn"
(e.g. "11:10:02:00 DAY1"). When your task asks for watermark timestamps, read it; when it asks
for in-clip seconds, count from the clip start.

# Honesty (absolute)
Report only what is visible or audible in this clip. If something cannot be confidently
identified, describe its observable attributes and mark it uncertain — never guess a confident
specific category. Never invent names, titles, words, or events.

# Output format
Return ONLY a single JSON object following the task's schema: double quotes, no trailing commas,
no markdown fences, no text outside JSON. Pick ONE content language from the dominant spoken
language (Chinese if participants speak Chinese, English otherwise); never translate or duplicate
content in both languages. This applies to EVERY string field you produce — object labels,
attributes, sound labels, notes — not only free-text descriptions."""

T1_TASK = """You are a first-person PERCEPTION annotator. Your ONLY job is to inventory what is
PRESENT in this clip: objects, people, spoken words, and non-speech sounds. Do NOT describe
actions, state changes, intentions, emotions, or causes — later annotation rounds handle those.

# Priorities (CRITICAL)
Focus on the NEAR FIELD and on objects that are being touched / held / manipulated by the wearer
or by other people, or are plausibly about to be. Identify these as precisely as the footage
allows. Distant or background clutter does NOT need exhaustive enumeration: for a crowded table
a summary like "桌上散落多个包装盒与线缆" is enough — counting 4 cups vs 5 is NOT required.

# Honesty rule (CRITICAL — never hallucinate a category)
If an object cannot be confidently identified (too close / out of focus / partially occluded /
reflective or transparent / covered by stickers), DO NOT guess a specific category. Describe its
observable attributes honestly — shape, size, color, material, transparency, attached items —
and set uncertain=true. "一个贴有标签的半透明玻璃容器，距离过近细节模糊" is a CORRECT answer;
confidently calling it "一块布" is a hallucination — the exact failure this round exists to kill.

# People
List visible people with neutral descriptors (clothing, hair, position). The wearer is not
visible except hands/body edges. Never invent names; use one only if shown on screen or spoken.

# Speech (transcription)
Transcribe all audible speech verbatim in its original language. Attribute a speaker to every
utterance using the people you see ("佩戴者(我的声音)" / "穿橙色T恤的男士") and voice cues.
Add approximate in-clip time offsets in seconds when you can — precision is NOT required; omit
rather than guess. Mark unclear hearing with uncertain=true. Never invent words.

# Non-speech sounds
Also report NON-SPEECH audio: music playing (from a speaker or earbuds; identify song/artist if
you can, otherwise describe style and language), continuous ambient sounds (keyboard typing,
fan, range hood, street traffic, distant crowd murmur), and discrete sound events (door, phone
notification/buzz, object clinking). Give the source when distinguishable (played by a device /
made nearby / from far away). Mark uncertain identifications honestly.

# Screen / text surfaces
If a screen is in use or a large text surface is visible (whiteboard / paper / sign), only flag
it with device + kind here; a dedicated round reads its content. Do not transcribe it now.

# Language
Object/person labels, attrs, sound labels and all notes MUST be in the clip's dominant spoken
language (Chinese when the scene is Chinese-speaking) — do NOT switch to English because this
task text is English. Speech is quoted verbatim in whatever language it was spoken.

# Output format
Return ONLY a single JSON object:
{
  "objects":  [{"label": "...", "attrs": ["..."], "where": "near|mid|background",
                "interacted_with": true, "first_seen_s": 0, "last_seen_s": 9,
                "uncertain": false, "uncertainty_note": ""}],
  "persons":  [{"label": "...", "attrs": ["..."], "first_seen_s": 0, "last_seen_s": 9,
                "uncertain": false}],
  "transcript": [{"t_start_s": 0.0, "t_end_s": 2.5, "lang": "zh", "speaker": "...",
                  "text": "...", "uncertain": false}],
  "sounds": [{"kind": "music|ambient|event", "label": "...", "source": "...",
              "t_start_s": 0, "t_end_s": 9, "uncertain": false}],
  "text_surfaces": [{"kind": "screen|whiteboard|paper|sign", "device": "...", "in_use": true}]
}
Times are seconds from clip start (~1s resolution is fine). Empty arrays are valid."""

T1_MERGE_TASK = """You reconcile TWO (or THREE) independent perception inventories of the SAME
first-person clip into one canonical object/person lexicon and one merged transcript.
The clip's video is attached as REFERENCE: use it to adjudicate identity and appearance
conflicts — is run A's "black rectangular device" the same thing as run B's "白色外接设备盒"?
Look, then decide. Do NOT add objects that no run reported: this is reconciliation of the given
inventories, not a fresh perception pass.

Rules:
- Two entries are the SAME object/person if one label is a synonym or hyponym of the other
  (杯子 vs 马克杯) and their time ranges overlap. Merge under the most specific label BOTH runs
  support; union the attrs.
- votes = how many runs reported it (e.g. "2/2" or "1/2"). A near-field interacted object at 1/2
  is either a hallucination or a miss — keep it with confidence "low" and note the disagreement;
  NEVER silently drop a near-field interacted object.
- confidence: high = all runs, none uncertain; medium = all runs with one uncertain, or labels
  merged after disagreement; low = seen by one run only, or all runs uncertain.
- transcript: union, dedupe near-identical utterances preferring the more complete wording;
  reported by all runs = confidence high, by one = low.
- sounds: union, dedupe (two descriptions of the same sound keep the more specific one).
- text_surfaces: union.
- Output labels in the clip's dominant scene language; do not translate entries into English.
- NEVER emit two lexicon entries with identical or synonym labels — they are the same object;
  merge them into one entry.

Assign stable ids obj01..., psn01... (type "object"|"person"). Return ONLY JSON:
{"lexicon": [{"id": "obj01", "label": "...", "type": "object", "attrs": [], "votes": "2/2",
              "confidence": "high", "interacted_with": true, "first_seen_s": 0,
              "last_seen_s": 9, "uncertainty_note": ""}],
 "transcript": [{"t_start_s": 0.0, "t_end_s": 2.5, "lang": "zh", "speaker": "...",
                 "text": "...", "confidence": "high"}],
 "sounds": [{"kind": "music", "label": "...", "source": "...", "t_start_s": 0,
             "t_end_s": 9, "confidence": "high"}],
 "text_surfaces": [{"kind": "screen", "device": "laptop", "in_use": true}],
 "agreement_notes": "..."}"""

T1C_TASK = """You read INFORMATION SURFACES in the N timestamp-ordered frames (native
resolution, ~1 per second) provided with this task: screens in use (laptop / monitor / phone)
and large text surfaces (whiteboard, paper documents, signs). Beyond raw OCR your job is
GUI-level understanding: WHAT application / website / content is being used, and what the
wearer is doing with it.

For each surface report:
- app / site identity when identifiable from chrome, logo, layout (e.g. "Bilibili 网页版",
  "文件资源管理器", "知乎");
- the content being consumed or operated, with verbatim titles when legible (video title, post
  title, file names, document headings);
- the visible operation (watching / scrolling / typing / clicking through ...).

Honesty: transcribe only what is legible; if a title is partially readable, transcribe the
readable part and note it. Never fabricate a title or app name. Do NOT read the top-right
timestamp watermark. Return ONLY JSON:
{"surfaces": [{"kind": "screen", "device": "laptop", "app": "...", "content": "...",
               "operation": "watching", "details": "...", "legibility": "good|partial|poor",
               "note": ""}]}"""

T2_TASK = """You are a first-person BEHAVIOR annotator producing SIMULATION-GRADE atomic
records of WHAT THE WEARER AND OTHER PEOPLE DID and WHAT VISIBLY CHANGED in this clip's
environment. Speech transcription, psychology and causal analysis belong to other rounds —
do not output them (you may quote speech already transcribed in the context).

# Watermark anchor (CRITICAL)
Read the TOP-RIGHT watermark "HH:MM:SS:FF" to timestamp every entry as "HH:MM:SS" (~1-3s
resolution). Cover the full clip from its start time onward.

# Object anchoring (CRITICAL)
The context provides this clip's object/person LEXICON from an independent perception round
(with per-entry confidence). When an action or change involves an object, USE THE LEXICON LABEL
as its name — especially for near-field interacted objects. A low-confidence entry must keep its
honest descriptive label; do NOT upgrade an uncertain description into a confident category. If
you genuinely need an object NOT in the lexicon, describe it AND append it to "new_objects"
(never silently invent). Objectless actions (walking, sitting down, turning, gaze shifts,
posture changes) need NO anchor — report them as freely as object actions. NEVER drop or vaguely
blur an action because its object is missing from the lexicon: describe it concretely and use
new_objects. If the screen layer in the context describes what was on screen, you may reference
that content when phrasing screen-related actions.

# Intent ban (CRITICAL)
Never describe intent or future actions (准备 / 打算 / 想要 / 试图 to do X) unless X is actually
observed within this clip. If the clip ends mid-activity, the last action simply ends there.
Describe only what is visible: postures, contacts, movements — not goals. An unmade bed does
NOT license "准备整理床铺" unless tidying is actually observed.

# Field rules
self_actions: one object per ATOMIC self-action, first person ("我"), present tense, verb-led.
An atomic action is a single verb-level step with one object and one immediate goal: reach,
grasp, pick up, put down, open, close, turn on, look at, walk to, sit down. DECOMPOSE compound
behavior ("我拿起手机划开屏幕看消息" -> "我拿起手机" + "我划开屏幕" + "我低头浏览消息"). Do NOT
split one continuous gesture into artificial micro-frames; do NOT merge separate manipulations.
NO count quota — atomicity is the standard: dense object manipulation may need 6-10 entries per
10s; a still 10s may honestly be 1-2. Describe hand-object contact, posture, gaze, locomotion,
device use. Never invent names; neutral descriptors for others.
others: same shape and atomicity standard, text led by the person descriptor. Empty if alone.
environment: one or two sentences on the OVERALL setting — place type, room layout, lighting,
weather/indoor-outdoor cues — the context a simulator would place the agent into; a scene
description, NOT a changelog. In-clip CHANGES belong in env_changes; screen content belongs to
the perception round's screen layer (do not duplicate it here).
env_changes: one object per ATOMIC observable state change DURING this clip: object appears /
disappears / moves; device state changes (screen lights/sleeps, door opens, cap comes off, cup
empties); lighting/soundscape shifts; person enters/leaves. Every entry carries
"cause": "self"|"other"|"external" (no visible agent). Only report changes visible in this clip.
new_objects: [{"label": "...", "why": "manipulated at 11:21:35 but absent from lexicon"}].
tags: 5-10 short lowercase keywords.

# Output format
Return ONLY JSON:
{"self_actions": [{"time": "HH:MM:SS", "time_end": "HH:MM:SS", "text": "..."}],
 "others": [{"time": "HH:MM:SS", "time_end": "HH:MM:SS", "text": "..."}],
 "environment": "...",
 "env_changes": [{"time": "...", "time_end": "...", "text": "...", "cause": "self"}],
 "new_objects": [{"label": "...", "why": "..."}],
 "tags": ["..."]}"""

T3_TASK = """You estimate the wearer's psychological state as a LATENT VARIABLE from this one
short first-person clip. The context provides the behavior round's structured output (actions,
changes, transcript, sounds).

# Isolation rule (definitional)
Estimate the mental state AS IF THIS CLIP WERE YOUR ONLY EVIDENCE — ignore anything that might
have happened before it. The question is: "considering only this clip's situation, what would a
plausible mental activity be?" — not "what is the wearer's state given their whole day?"

# Output format
Return ONLY JSON:
{"emotion": "1-2 lowercase keywords (e.g. focused, relaxed; 'neutral' if none readable)",
 "mental_activity": "2-3 sentences describing the plausible inner state — state, not plans",
 "evidence": ["11:10:15 <quote from the behavior output / audible cue>", "..."],
 "confidence": "low|medium|high"}
Evidence must cite observable cues (timestamped action quotes, tone of voice, speech content,
interaction pace). No intent or future-plan speculation here either. confidence reflects how
strongly the evidence constrains the inference."""

T4_TASK = """You are a causal analyst. The context provides one clip's object lexicon, behavior
JSON (self_actions / others / env_changes / environment / transcript / sounds) and psychology
JSON (emotion / mental_activity). There is no video. Build the clip's directed causal graph
from this evidence alone.

Edge types (use EXACTLY these strings):
  env->action      an environment event/state changed MY behavior (phone vibrates -> I pick it up)
  env->emotion     an environment event changed my emotion (loud noise -> startled)
  emotion->action  my emotion directly drove my behavior (bored -> I start scrolling)
  action->env      my action changed the environment (I flip the switch -> lights on); mirrors
                   env_changes entries with cause "self"
  other->action    another person's action triggered my action (colleague waves -> I walk over)
  other->env       another person's action changed the environment (she opens the curtain)
  other->emotion   another person's action changed my emotion (guest laughs -> I relax)
  env->env         an environmental event with NO visible agent causes another environmental
                   change (云遮住阳光 -> 室内变暗; 风把门吹开)

strength (counterfactual, REQUIRED on every edge):
  strong   = trigger: without this cause the effect likely would NOT have happened, or would
             have gone a different direction;
  moderate = shaper: changed HOW/WHEN/how vigorously the effect happened, but it would still
             have occurred;
  weak     = background: one contributory factor among several; the effect was mostly driven
             by habit or task.

Rules: the causal direction must be supported by visible temporal order (use the timestamps in
the input) + plausible mechanism — never fabricate; quote the atomic action/change texts with
their times as cause/effect; object names must match the lexicon; every cause="self"
env_change should be mirrored by an action->env edge; 0-3 links is typical, empty array is
fine; physical edges are almost always strong — expected, not lazy. Return ONLY JSON:
{"causal_links": [{"type": "...", "cause": "...", "effect": "...", "strength": "..."}]}"""

_CLIP_HEADER_TMPL = "Clip: {clip_id} | {day_label} | watermark start {start_hms} | ~{dur:.0f}s.\n\n"

# Context block prefixes. The SAME block bytes must be reused verbatim by every
# later turn that includes them, so the token prefix keeps matching (spec §6.0).
_CTX_HDR = "# Context from earlier rounds (verbatim outputs; use as anchors, do not re-annotate)\n"


def _ctx_t1(t1_raw: str) -> str:
    return _CTX_HDR + "## Perception round (merged lexicon/transcript/sounds)\n" + t1_raw


def _ctx_t2(t2_raw: str) -> str:
    return "\n\n## Behavior round output\n" + t2_raw


def _ctx_t3(t3_raw: str) -> str:
    return "\n\n## Psychology round output\n" + t3_raw


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


def parse_ts(filename: str) -> str:
    """`DAY1_A1_JAKE_11094208.mp4` -> `11:09:42.08`."""
    stem = Path(filename).stem
    ts_raw = stem.split("_")[-1]
    if len(ts_raw) != 8 or not ts_raw.isdigit():
        raise ValueError(f"unexpected filename {filename}")
    hh, mm, ss, cc = ts_raw[0:2], ts_raw[2:4], ts_raw[4:6], ts_raw[6:8]
    return f"{hh}:{mm}:{ss}.{cc}"


def _ts_label(name: str) -> str:
    ts_raw = Path(name).stem.split("_")[-1]
    return f"{ts_raw[0:2]}:{ts_raw[2:4]}:{ts_raw[4:6]}"


def _fname_to_seconds(name: str) -> float:
    ts_raw = Path(name).stem.split("_")[-1]
    hh, mm, ss, cs = int(ts_raw[0:2]), int(ts_raw[2:4]), int(ts_raw[4:6]), int(ts_raw[6:8])
    return hh * 3600 + mm * 60 + ss + cs / 100.0


def _hhmm_to_seconds(hhmm: str) -> int:
    if len(hhmm) != 4 or not hhmm.isdigit():
        raise ValueError(f"--start-time/--end-time must be HHMM (4 digits), got {hhmm!r}")
    return int(hhmm[0:2]) * 3600 + int(hhmm[2:4]) * 60


def _add_seconds_to_hms(hms: str, seconds: float) -> str:
    """Add `seconds` to a "HH:MM:SS" watermark timestamp, rolling over minutes/hours."""
    hh, mm, ss = (int(x) for x in hms.split(":"))
    total = hh * 3600 + mm * 60 + ss + max(0.0, float(seconds))
    total = total % 86400
    return f"{int(total // 3600):02d}:{int((total % 3600) // 60):02d}:{int(total % 60):02d}"


def _sec_field_to_hms(start_hms: str, val) -> str:
    """Best-effort: an in-clip seconds value (str/float) -> absolute watermark HH:MM:SS."""
    try:
        return _add_seconds_to_hms(start_hms, float(val))
    except (TypeError, ValueError):
        return ""


def _n_pieces_for_duration(dur: float, target_s: int) -> int:
    snapped = round(dur / target_s) * target_s
    if abs(snapped - dur) <= 0.5:
        dur = float(snapped)
    if dur <= target_s:
        return 1
    return max(2, math.ceil(dur / target_s - 1e-9))


@dataclass
class SliceInfo:
    path: Path
    duration: float
    piece_idx: int
    start_hms: str
    is_whole: bool


def split_source_into_slices(src: Path, clip_id: str, base_hms: str, target_s: int,
                             out_dir: Path, *, resolution: int, fps: int, crf: int,
                             audio_k: int, skip_if_exists: bool = True) -> list[SliceInfo]:
    """Re-encode `src` into ceil(dur/target_s) evenly-sized pieces (media for T1/T2/T3)."""
    dur = ffprobe_duration(src)
    n = _n_pieces_for_duration(dur, target_s)
    slice_dur = dur / n
    out_dir.mkdir(parents=True, exist_ok=True)
    pieces: list[SliceInfo] = []
    for i in range(n):
        start_off = i * slice_dur
        is_whole = (n == 1)
        suffix = "" if is_whole else f"_p{i + 1}"
        out_path = out_dir / f"{clip_id}{suffix}.mp4"
        if not (skip_if_exists and out_path.exists() and out_path.stat().st_size > 0):
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(src), "-ss", f"{start_off:.3f}", "-t", f"{slice_dur:.3f}",
                "-vf", f"fps={fps},scale={resolution}:{resolution}:flags=lanczos",
                "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-c:a", "aac", "-b:a", f"{audio_k}k", "-ac", "1",
                "-movflags", "+faststart", str(out_path),
            ]
            rc = subprocess.run(cmd, capture_output=True, text=True)
            if rc.returncode != 0:
                raise RuntimeError(f"ffmpeg slice failed: {rc.stderr.strip()[:300]}")
        pieces.append(SliceInfo(
            path=out_path, duration=ffprobe_duration(out_path), piece_idx=i + 1,
            start_hms=_add_seconds_to_hms(base_hms, start_off), is_whole=is_whole,
        ))
    return pieces


def extract_native_frames(src: Path, offset_s: float, dur_s: float, clip_id: str,
                          out_dir: Path, fps: float = NATIVE_FRAME_FPS, long_edge: int = 0,
                          skip_if_exists: bool = True) -> list[Path]:
    """T1c media: ~1fps JPEG keyframes from the source. long_edge=0 keeps native
    resolution; a positive value caps the long edge (the image API has no
    server-side resolution knob and bills (w/32)*(h/32) tokens per frame, so
    downscaling large frames is on us)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    first = out_dir / f"{clip_id}_f01.jpg"
    if skip_if_exists and first.exists():
        return sorted(out_dir.glob(f"{clip_id}_f*.jpg"))
    for p in out_dir.glob(f"{clip_id}_f*.jpg"):
        p.unlink()
    vf = f"fps={fps}"
    if long_edge and long_edge > 0:
        vf += f",scale={long_edge}:{long_edge}:force_original_aspect_ratio=decrease"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src), "-ss", f"{offset_s:.3f}", "-t", f"{dur_s:.3f}",
        "-vf", vf, "-q:v", "2", str(out_dir / f"{clip_id}_f%02d.jpg"),
    ]
    rc = subprocess.run(cmd, capture_output=True, text=True)
    if rc.returncode != 0:
        raise RuntimeError(f"native frame extraction failed: {rc.stderr.strip()[:300]}")
    frames = sorted(out_dir.glob(f"{clip_id}_f*.jpg"))
    if not frames:
        raise RuntimeError("native frame extraction produced no frames")
    return frames


# ===========================================================================
# JSON parsing & per-turn normalization
# ===========================================================================

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


class ParseFailed(Exception):
    def __init__(self, reason: str, raw: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.raw = raw


def parse_json_obj(content: str) -> dict:
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
    return data


def _s(x) -> str:
    return "" if x is None else str(x)


def _b(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    return str(x).strip().lower() in ("true", "yes", "1")


def _norm_strlist(v) -> list:
    if not isinstance(v, list):
        return []
    return [str(t) for t in v]


def norm_t1(d: dict) -> dict:
    objs = []
    for o in d.get("objects", []) or []:
        if not isinstance(o, dict):
            continue
        objs.append({"label": _s(o.get("label")), "attrs": _norm_strlist(o.get("attrs")),
                     "where": _s(o.get("where")), "interacted_with": _b(o.get("interacted_with")),
                     "first_seen_s": _s(o.get("first_seen_s")), "last_seen_s": _s(o.get("last_seen_s")),
                     "uncertain": _b(o.get("uncertain")), "uncertainty_note": _s(o.get("uncertainty_note"))})
    persons = []
    for p in d.get("persons", []) or []:
        if not isinstance(p, dict):
            continue
        persons.append({"label": _s(p.get("label")), "attrs": _norm_strlist(p.get("attrs")),
                        "first_seen_s": _s(p.get("first_seen_s")), "last_seen_s": _s(p.get("last_seen_s")),
                        "uncertain": _b(p.get("uncertain"))})
    tr = []
    for t in d.get("transcript", []) or []:
        if not isinstance(t, dict):
            continue
        tr.append({"t_start_s": _s(t.get("t_start_s")), "t_end_s": _s(t.get("t_end_s")),
                   "lang": _s(t.get("lang")), "speaker": _s(t.get("speaker")),
                   "text": _s(t.get("text")), "uncertain": _b(t.get("uncertain"))})
    snd = []
    for s_ in d.get("sounds", []) or []:
        if not isinstance(s_, dict):
            continue
        snd.append({"kind": _s(s_.get("kind")), "label": _s(s_.get("label")),
                    "source": _s(s_.get("source")), "t_start_s": _s(s_.get("t_start_s")),
                    "t_end_s": _s(s_.get("t_end_s")), "uncertain": _b(s_.get("uncertain"))})
    tsf = []
    for t in d.get("text_surfaces", []) or []:
        if not isinstance(t, dict):
            continue
        tsf.append({"kind": _s(t.get("kind")), "device": _s(t.get("device")), "in_use": _b(t.get("in_use"))})
    if not (objs or persons or tr or snd or tsf):
        raise ParseFailed("t1: all sections empty (objects/persons/transcript/sounds/text_surfaces)")
    return {"objects": objs, "persons": persons, "transcript": tr, "sounds": snd,
            "text_surfaces": tsf}


def norm_merge(d: dict) -> dict:
    lex = []
    for e in d.get("lexicon", []) or []:
        if not isinstance(e, dict):
            continue
        lex.append({"id": _s(e.get("id")), "label": _s(e.get("label")),
                    "type": _s(e.get("type")) or "object", "attrs": _norm_strlist(e.get("attrs")),
                    "votes": _s(e.get("votes")), "confidence": _s(e.get("confidence")),
                    "interacted_with": _b(e.get("interacted_with")),
                    "first_seen_s": _s(e.get("first_seen_s")), "last_seen_s": _s(e.get("last_seen_s")),
                    "uncertainty_note": _s(e.get("uncertainty_note"))})
    tr = []
    for t in d.get("transcript", []) or []:
        if not isinstance(t, dict):
            continue
        tr.append({"t_start_s": _s(t.get("t_start_s")), "t_end_s": _s(t.get("t_end_s")),
                   "lang": _s(t.get("lang")), "speaker": _s(t.get("speaker")),
                   "text": _s(t.get("text")), "confidence": _s(t.get("confidence"))})
    snd = []
    for s_ in d.get("sounds", []) or []:
        if not isinstance(s_, dict):
            continue
        snd.append({"kind": _s(s_.get("kind")), "label": _s(s_.get("label")),
                    "source": _s(s_.get("source")), "t_start_s": _s(s_.get("t_start_s")),
                    "t_end_s": _s(s_.get("t_end_s")), "confidence": _s(s_.get("confidence"))})
    tsf = []
    for t in d.get("text_surfaces", []) or []:
        if not isinstance(t, dict):
            continue
        tsf.append({"kind": _s(t.get("kind")), "device": _s(t.get("device")), "in_use": _b(t.get("in_use"))})
    if not lex:
        raise ParseFailed("t1 merge: empty lexicon")
    return {"lexicon": lex, "transcript": tr, "sounds": snd, "text_surfaces": tsf,
            "agreement_notes": _s(d.get("agreement_notes"))}


def norm_t1c(d: dict) -> dict:
    surfaces = []
    for t in d.get("surfaces", []) or []:
        if not isinstance(t, dict):
            continue
        surfaces.append({"kind": _s(t.get("kind")), "device": _s(t.get("device")),
                         "app": _s(t.get("app")), "content": _s(t.get("content")),
                         "operation": _s(t.get("operation")), "details": _s(t.get("details")),
                         "legibility": _s(t.get("legibility")), "note": _s(t.get("note"))})
    # Empty surfaces is a VALID outcome (T1 flagged a surface, but nothing on it
    # is legible/worth reporting) — not a parse failure.
    return {"surfaces": surfaces}


def _norm_actions(v) -> list:
    out = []
    if not isinstance(v, list):
        return out
    for a in v:
        if isinstance(a, dict):
            out.append({"time": _s(a.get("time")), "time_end": _s(a.get("time_end")),
                        "text": _s(a.get("text"))})
        else:
            out.append({"time": "", "time_end": "", "text": str(a)})
    return out


def norm_t2(d: dict) -> dict:
    env_changes = []
    for e in d.get("env_changes", []) or []:
        if not isinstance(e, dict):
            continue
        env_changes.append({"time": _s(e.get("time")), "time_end": _s(e.get("time_end")),
                            "text": _s(e.get("text")), "cause": _s(e.get("cause"))})
    new_objects = []
    for o in d.get("new_objects", []) or []:
        if not isinstance(o, dict):
            continue
        new_objects.append({"label": _s(o.get("label")), "why": _s(o.get("why"))})
    self_actions = _norm_actions(d.get("self_actions"))
    if not self_actions:
        raise ParseFailed("t2: empty self_actions")
    return {"self_actions": self_actions,
            "others": _norm_actions(d.get("others")),
            "environment": _s(d.get("environment")),
            "env_changes": env_changes,
            "new_objects": new_objects,
            "tags": _norm_strlist(d.get("tags"))}


def norm_t3(d: dict) -> dict:
    if not _s(d.get("emotion")):
        raise ParseFailed("t3: missing emotion")
    return {"emotion": _s(d.get("emotion")),
            "mental_activity": _s(d.get("mental_activity")),
            "evidence": _norm_strlist(d.get("evidence")),
            "confidence": _s(d.get("confidence"))}


def norm_t4(d: dict) -> dict:
    links = []
    for c in d.get("causal_links", []) or []:
        if not isinstance(c, dict):
            continue
        strength = _s(c.get("strength")).strip().lower()
        if strength not in _STRENGTH_LEVELS:
            strength = ""
        links.append({"type": _s(c.get("type")), "cause": _s(c.get("cause")),
                      "effect": _s(c.get("effect")), "strength": strength})
    # Empty link list is legal; invalid types are normalized leniently (counted in QC).
    return {"causal_links": links}


# ===========================================================================
# Message builders (cache-layout: system -> media -> verbatim context -> task)
# ===========================================================================

def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def video_part(slice_path: Path, fps: int = DEFAULT_FPS) -> dict:
    return {"type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{_b64(slice_path)}"},
            "fps": fps}


def image_part(jpg_path: Path) -> dict:
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_b64(jpg_path)}"}}


def build_messages(media_parts: list, context_text: str, task_text: str) -> list:
    text = (context_text + "\n\n" + task_text) if context_text else task_text
    return [{"role": "system", "content": SHARED_SYSTEM_MSG},
            {"role": "user", "content": [*media_parts, {"type": "text", "text": text}]}]


def clip_header(row) -> str:
    return _CLIP_HEADER_TMPL.format(
        clip_id=row["clip_id"], day_label=f"DAY{int(row['day'])}",
        start_hms=row["start_hms"], dur=float(row["duration"]))


# ===========================================================================
# API calling
# ===========================================================================

class TurnApiConfig:
    __slots__ = ("thinking", "json_mode")

    def __init__(self, thinking, json_mode=True):
        self.thinking = thinking
        self.json_mode = json_mode


def _raw_call(client, model, messages, cfg, limiter, tag, log, api_style="mimo") -> tuple:
    """One HTTP call (network-level retry x2). Returns (content, usage, latency, attempt).
    api_style "mimo": thinking extra_body, no completion cap (server default = max).
    api_style "openai": generic OpenAI-compatible VLM (T1c plug-in) — no vendor body."""
    last_exc = None
    if api_style == "mimo":
        kwargs = dict(model=model, messages=messages,
                      extra_body={"thinking": {"type": cfg.thinking}})
        if cfg.thinking == "disabled":
            kwargs["temperature"] = 1.0
    else:
        kwargs = dict(model=model, messages=messages, temperature=1.0)
    if cfg.json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(1, 3):
        limiter.acquire()
        try:
            t0 = time.time()
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            u = getattr(resp, "usage", None)
            usage = u.model_dump() if (u is not None and hasattr(u, "model_dump")) else dict(u or {})
            return content, usage, time.time() - t0, attempt
        except Exception as e:
            last_exc = e
            log.debug(f"[{tag}] API attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt < 2:
                time.sleep(min(2 ** attempt, 30))
    raise last_exc


def _usage_small(usage: dict) -> dict:
    pd_ = usage.get("prompt_tokens_details") or {}
    cached = pd_.get("cached_tokens", 0) if isinstance(pd_, dict) else 0
    return {"prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cached_tokens": int(cached or 0)}


def call_turn(client, model, messages, cfg, limiter, log, tag, parser,
              usage_sink, api_style="mimo") -> Optional[dict]:
    """One turn call + parse, with a single retry on rejection/parse failure
    (Mimo often rejects the very first video request; the retry hits the
    prefix cache). Returns {"raw", "parsed", "usage", "latency_s", "recovery"}
    or None on failure."""
    content = usage = None
    last_reason = ""
    for attempt_idx in (1, 2):
        try:
            content, usage_full, latency, attempt = _raw_call(
                client, model, messages, cfg, limiter, tag, log, api_style=api_style)
        except Exception as e:
            log.warning(f"[{tag}] call crashed: {type(e).__name__}: {e}")
            return None
        usage = _usage_small(usage_full)
        usage_sink({"clip_id": tag.split("/")[0], "turn": tag.split("/")[-1],
                    "attempt": attempt_idx, "latency_s": round(latency, 3),
                    "recovery": "ok" if attempt_idx == 1 else "first_attempt_rejection",
                    "usage": usage})
        try:
            parsed = parser(parse_json_obj(content))
            return {"raw": content.strip(), "parsed": parsed, "usage": usage,
                    "latency_s": latency,
                    "recovery": "ok" if attempt_idx == 1 else "first_attempt_rejection"}
        except ParseFailed as e:
            last_reason = e.reason
            log.info(f"[{tag}] attempt {attempt_idx} parse failed ({e.reason}); "
                     f"{'retrying once' if attempt_idx == 1 else 'giving up'}")
    if not (content or "").strip():
        last_reason = last_reason or "empty content"
    log.warning(f"[{tag}] turn failed after retry ({last_reason})")
    return None


# ===========================================================================
# Turn store: thread-safe per-turn jsonl (append-as-finished + resume source)
# ===========================================================================

class TurnStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._f = None

    def _file(self):
        if self._f is None:
            self._f = self.path.open("a", encoding="utf-8")
        return self._f

    def append(self, rec: dict):
        with self._lock:
            self._file().write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._file().flush()

    def close(self):
        with self._lock:
            if self._f is not None:
                self._f.close()
                self._f = None

    def load(self) -> dict:
        """clip_id -> last record for that clip."""
        out: dict = {}
        if not self.path.exists():
            return out
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
                out[rec.get("clip_id", "")] = rec
            except json.JSONDecodeError:
                continue
        return out


# ===========================================================================
# QC checks (spec §5.4 / §8) — warnings and counters, never hard rejections
# ===========================================================================

_INTENT_RE = re.compile(r"准备|打算|想要|试图|about to|going to|trying to")
_SCREEN_HINT_RE = re.compile(r"屏幕|显示器|电脑|笔记本|手机|monitor|laptop|screen|phone")


def run_qc(final: dict, log) -> dict:
    clip_id = final["clip_id"]
    t2_texts = [a["text"] for a in final["self_actions"]] + [o["text"] for o in final["others"]]
    intent_hits = sum(1 for t in t2_texts if _INTENT_RE.search(t))
    has_action_env = any(l["type"] == "action->env" for l in final["causal_links"])
    mirror_missing = sum(1 for e in final["env_changes"]
                         if e.get("cause") == "self") if not has_action_env else 0
    screen_hint = any(_SCREEN_HINT_RE.search(t) for t in t2_texts)
    screen_missing = bool(screen_hint and not final["screen"])
    labels = [e["label"] for e in final["inventory"] if len(e["label"]) >= 2]
    unanchored = sum(1 for t in t2_texts if not any(lbl in t for lbl in labels))
    invalid_edges = sum(1 for l in final["causal_links"] if l["type"] not in _CAUSAL_TYPES)
    qc = {"intent_markers": intent_hits, "mirror_missing": mirror_missing,
          "screen_hint_without_layer": screen_missing, "unanchored_texts": unanchored,
          "invalid_edge_types": invalid_edges}
    if intent_hits:
        log.warning(f"[{clip_id}] QC: {intent_hits} intent-marker text(s) in T2")
    if screen_missing:
        log.warning(f"[{clip_id}] QC: T2 mentions screen use but screen layer is empty")
    if invalid_edges:
        log.warning(f"[{clip_id}] QC: {invalid_edges} causal edge(s) with invalid type")
    return qc


# ===========================================================================
# Final record assembly (pure code, no LLM)
# ===========================================================================

def assemble_final(row, t1: dict, t1c: Optional[dict], t2: dict, t3: dict, t4: dict,
                   model: str) -> dict:
    start_hms = row["start_hms"]
    lex = t1["parsed"]["lexicon"]
    for e in lex:
        e["first_seen"] = _sec_field_to_hms(start_hms, e.get("first_seen_s"))
        e["last_seen"] = _sec_field_to_hms(start_hms, e.get("last_seen_s"))
    speech = [{"lang": t.get("lang", ""), "speaker": t.get("speaker", ""), "text": t.get("text", ""),
               "t_start": _sec_field_to_hms(start_hms, t.get("t_start_s")),
               "t_end": _sec_field_to_hms(start_hms, t.get("t_end_s")),
               "confidence": t.get("confidence", "")} for t in t1["parsed"]["transcript"]]
    sounds = [{"kind": s.get("kind", ""), "label": s.get("label", ""), "source": s.get("source", ""),
               "t_start": _sec_field_to_hms(start_hms, s.get("t_start_s")),
               "t_end": _sec_field_to_hms(start_hms, s.get("t_end_s")),
               "confidence": s.get("confidence", "")} for s in t1["parsed"]["sounds"]]
    turns_meta = {"t1": {"runs": len(t1.get("runs", [])), "tokens": t1["usage"],
                         "latency_s": round(t1.get("latency_s", 0.0), 3)},
                  "t1c": ({"tokens": t1c["usage"], "latency_s": round(t1c.get("latency_s", 0.0), 3)}
                          if t1c else None),
                  "t2": {"tokens": t2["usage"], "latency_s": round(t2.get("latency_s", 0.0), 3)},
                  "t3": {"tokens": t3["usage"], "latency_s": round(t3.get("latency_s", 0.0), 3)},
                  "t4": {"tokens": t4["usage"], "latency_s": round(t4.get("latency_s", 0.0), 3)}}
    return {
        "clip_id": row["clip_id"], "global_idx": int(row["global_idx"]),
        "day": int(row["day"]), "user": row["user"], "duration_s": float(row["duration"]),
        "model": model, "ts_captioned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slice_path": row["slice_path"],
        "turns": turns_meta,
        "inventory": lex,
        "speech": speech,
        "sounds": sounds,
        "screen": (t1c["parsed"]["surfaces"] if t1c else []),
        "self_actions": t2["parsed"]["self_actions"],
        "others": t2["parsed"]["others"],
        "environment": t2["parsed"]["environment"],
        "env_changes": t2["parsed"]["env_changes"],
        "new_objects": t2["parsed"]["new_objects"],
        "psychology": t3["parsed"],
        "causal_links": t4["parsed"]["causal_links"],
        "tags": t2["parsed"]["tags"],
        "recovery": {"t1": t1.get("recovery", "ok"),
                     "t1c": (t1c.get("recovery", "ok") if t1c else "skipped"),
                     "t2": t2.get("recovery", "ok"), "t3": t3.get("recovery", "ok"),
                     "t4": t4.get("recovery", "ok")},
    }


# ===========================================================================
# Rate limiter / writer / logging (adapted from caption_pipeline.py)
# ===========================================================================

class SlidingWindowRateLimiter:
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


class OrderedWriter:
    """Flush final records to jsonl in strict global_idx order."""

    def __init__(self, captions_path, log, next_idx=1):
        self.captions_path = captions_path
        self.log = log
        self.next_idx = next_idx
        self._heap = []
        self._seq = 0
        self._lock = threading.Lock()
        self.n_written = 0
        self.cap_f = captions_path.open("a", encoding="utf-8")

    def submit(self, result):
        with self._lock:
            self._seq += 1
            heapq.heappush(self._heap, (result.global_idx, self._seq, result))
            self._flush_locked()

    def _flush_locked(self):
        while self._heap and self._heap[0][0] <= self.next_idx:
            _, _, res = heapq.heappop(self._heap)
            for cap in res.caption_records:
                self.cap_f.write(json.dumps(cap, ensure_ascii=False) + "\n")
                self.n_written += 1
            if res.global_idx >= self.next_idx:
                self.next_idx = res.global_idx + 1
            self.log.debug(f"[writer] wrote gid={res.global_idx} "
                           f"caps={[c['clip_id'] for c in res.caption_records] or ['<none>']} "
                           f"failures={res.failures}")

    def close(self):
        with self._lock:
            if self._heap:
                self.log.warning(f"[writer] {len(self._heap)} results never flushed (missing global_idx)")
            self.cap_f.close()


class TqdmLoggingHandler(logging.Handler):
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
# Source selection & clip units
# ===========================================================================

def select_source_files(src_root, participant, day, start_time, end_time):
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
    __slots__ = ("kind", "src", "clip_id", "start_ts", "target_s")

    def __init__(self, kind, src, clip_id, start_ts, target_s):
        self.kind = kind
        self.src = src
        self.clip_id = clip_id
        self.start_ts = start_ts
        self.target_s = target_s


def group_into_clips(files, participant, day, target_s):
    units = []
    for f in files:
        ts_raw = Path(f.stem).stem.split("_")[-1]
        ss_field = ts_raw[4:6]
        kind = "segment_open" if ss_field not in ("00", "30") else "30s"
        units.append(ClipUnit(kind, f, f"DAY{day}_{participant}_{ts_raw}",
                              _ts_label(f.name), target_s))
    return units


def count_pieces_for_units(units, log=None) -> list:
    out = []
    for u in units:
        try:
            dur = ffprobe_duration(u.src)
            n = _n_pieces_for_duration(dur, u.target_s)
        except Exception as e:
            if log:
                log.warning(f"[count] ffprobe failed for {u.src.name}: {e}; assuming 1 piece")
            n = 1
        out.append(n)
    return out


# ===========================================================================
# Preprocess workers
# ===========================================================================

class PreprocessJob:
    __slots__ = ("unit", "idx_base", "is_first_unit", "day", "user", "skip_if_exists",
                 "resolution", "fps", "crf", "audio_k", "slices_dir", "out_dir")

    def __init__(self, unit, idx_base, *, is_first_unit, day, user, skip_if_exists,
                 resolution, fps, crf, audio_k, slices_dir, out_dir):
        self.unit = unit
        self.idx_base = idx_base
        self.is_first_unit = is_first_unit
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
        base_clip_id = unit.clip_id
        src = unit.src
        try:
            pieces = split_source_into_slices(
                src, base_clip_id, unit.start_ts, unit.target_s, job.slices_dir,
                resolution=job.resolution, fps=job.fps, crf=job.crf, audio_k=job.audio_k,
                skip_if_exists=job.skip_if_exists,
            )
            dur_src = None
            for piece in pieces:
                clip_kind = unit.kind if piece.is_whole else f"{unit.target_s}s"
                piece_clip_id = base_clip_id if piece.is_whole else f"{base_clip_id}_p{piece.piece_idx}"
                gid = job.idx_base + piece.piece_idx - 1
                # src_offset: this piece's offset inside the source (for native-frame
                # extraction in T1c). For is_whole pieces the offset is 0.
                if piece.is_whole:
                    src_offset = 0.0
                else:
                    if dur_src is None:
                        dur_src = ffprobe_duration(src)
                    n = _n_pieces_for_duration(dur_src, unit.target_s)
                    src_offset = (piece.piece_idx - 1) * (dur_src / n)
                row = {
                    "clip_id": piece_clip_id, "day": job.day, "user": job.user,
                    "start_ts": piece.start_hms,
                    "end_ts": f"{piece.start_hms}+{piece.duration:.3f}s",
                    "src_file": src.name, "src_path": str(src),
                    "src_offset": round(src_offset, 3),
                    "clip_idx": piece.piece_idx,
                    "global_idx": gid, "duration": round(piece.duration, 3),
                    "is_day_open": False, "is_day_close": False, "status": "pending",
                    "slice_path": str(piece.path.relative_to(job.out_dir)),
                    "clip_kind": clip_kind,
                    "target_s": (30 if piece.is_whole else unit.target_s),
                    "start_hms": piece.start_hms,
                }
                out_q.put(PreprocessResult(clip_id=piece_clip_id, global_idx=gid,
                                           slice_path=piece.path, row=row))
        except Exception as e:
            log.error(f"[preprocess] {base_clip_id} failed: {type(e).__name__}: {e}")
            out_q.put(PreprocessResult(clip_id=base_clip_id, global_idx=job.idx_base,
                                       slice_path=job.slices_dir / f"{base_clip_id}.mp4", row=None,
                                       error=f"{type(e).__name__}: {e}"))
        finally:
            job_q.task_done()


# ===========================================================================
# API workers — per-clip turn state machine
# ===========================================================================

class ApiJob:
    __slots__ = ("row", "slice_path")

    def __init__(self, row, slice_path):
        self.row = row
        self.slice_path = slice_path


class ApiResult:
    __slots__ = ("global_idx", "clip_id", "caption_records", "failures", "turn_stats")

    def __init__(self, *, global_idx=0, clip_id="", caption_records=None, failures=None,
                 turn_stats=None):
        self.global_idx = global_idx
        self.clip_id = clip_id
        self.caption_records = caption_records or []
        self.failures = failures or []
        self.turn_stats = turn_stats or {}


def _needs_arbitration(merged: dict, k: int) -> bool:
    full = f"{k}/{k}"
    return any(e["type"] == "object" and e["interacted_with"] and e["votes"] != full
               for e in merged["lexicon"])


class PipelineCtx:
    """Shared, read-mostly context for API workers. `priors` holds each turn
    store preloaded once at startup (clip_id -> last record) for resume; new
    results are appended via `stores` (thread-safe). T1c may run on a separate,
    pluggable OpenAI-compatible image VLM (client/model/api_style)."""

    def __init__(self, args, stores, cfgs, client, limiter, log, paths, priors,
                 t1c_client=None, t1c_model=None, t1c_api_style="mimo"):
        self.args = args
        self.stores = stores
        self.cfgs = cfgs
        self.client = client
        self.limiter = limiter
        self.log = log
        self.paths = paths  # dict: out_dir, cache_dir, frames_dir
        self.priors = priors
        self.usage_store = stores["_usage"]
        self.t1c_client = t1c_client or client
        self.t1c_model = t1c_model
        self.t1c_api_style = t1c_api_style


def _run_t1(ctx: PipelineCtx, row, slice_path: Path) -> Optional[dict]:
    """k perception runs -> merge -> optional arbitration + re-merge.
    Returns the T1 turn dict (parsed=merged, raw=merge output, runs=[...]) or None."""
    clip_id = row["clip_id"]
    tag = f"{clip_id}/t1"
    k = ctx.args.votes
    header = clip_header(row)
    media = [video_part(slice_path)]
    runs = []
    for i in range(1, k + 1):
        task = header + f"(Independent perception run {i} of {k}.)\n\n" + T1_TASK
        r = call_turn(ctx.client, ctx.args.model, build_messages(media, "", task),
                      ctx.cfgs["t1"], ctx.limiter, ctx.log, f"{tag}/run{i}", norm_t1,
                      ctx.usage_store.append)
        if r is None:
            return None
        runs.append(r)
    merged = _merge_with(ctx, clip_id, header, runs, media)
    if merged is None:
        return None
    n_runs = len(runs)
    if n_runs == k and _needs_arbitration(merged["parsed"], k):
        ctx.log.info(f"[{tag}] near-field interacted object disagreement -> arbitration run {k + 1}")
        task = (header + f"(ARBITRATION run {k + 1}: two runs disagreed on a near-field "
                 f"interacted object; look carefully and honestly. Still apply the near-field "
                 f"priority — do not switch to exhaustive background enumeration.)\n\n" + T1_TASK)
        arb = call_turn(ctx.client, ctx.args.model, build_messages(media, "", task),
                        ctx.cfgs["t1"], ctx.limiter, ctx.log, f"{tag}/arb", norm_t1,
                        ctx.usage_store.append)
        if arb is not None:
            runs.append(arb)
            merged = _merge_with(ctx, clip_id, header, runs, media)
            if merged is None:
                return None
            merged["recovery"] = "arbitrated"
    return {"parsed": merged["parsed"], "raw": merged["raw"], "usage": merged["usage"],
            "latency_s": merged["latency_s"], "runs": [{"raw": r["raw"], "usage": r["usage"]}
                                                       for r in runs],
            "recovery": merged.get("recovery", "ok")}


def _merge_with(ctx: PipelineCtx, clip_id: str, header: str, runs: list,
                media: list) -> Optional[dict]:
    """Merge k inventories into the canonical lexicon. The clip's video is
    attached (media-first, byte-identical to the perception runs, so its
    prefix hits the cache): judging whether two labels are the SAME object
    often needs eyes, not just synonym matching."""
    runs_json = json.dumps([{"run": i + 1, "output": json.loads(r["raw"])} for i, r in enumerate(runs)],
                           ensure_ascii=False, indent=1)
    task = header + f"The inventories of {len(runs)} independent runs follow; the original clip\n" \
        f"is attached as REFERENCE for adjudicating identity/appearance conflicts.\n\n" + runs_json + \
        "\n\n" + T1_MERGE_TASK
    return call_turn(ctx.client, ctx.args.model, build_messages(media, "", task),
                     ctx.cfgs["t1m"], ctx.limiter, ctx.log, f"{clip_id}/t1m", norm_merge,
                     ctx.usage_store.append)


def _run_t1c(ctx: PipelineCtx, row) -> Optional[dict]:
    clip_id = row["clip_id"]
    tag = f"{clip_id}/t1c"
    try:
        frames = extract_native_frames(Path(row["src_path"]), float(row["src_offset"]),
                                       float(row["duration"]), clip_id, ctx.paths["frames_dir"],
                                       fps=ctx.args.t1c_frame_fps,
                                       long_edge=ctx.args.t1c_frame_size)
    except Exception as e:
        ctx.log.warning(f"[{tag}] native frame extraction failed: {e}")
        return None
    media = [image_part(f) for f in frames]
    task = clip_header(row) + T1C_TASK
    return call_turn(ctx.t1c_client, ctx.t1c_model or ctx.args.model,
                     build_messages(media, "", task),
                     ctx.cfgs["t1c"], ctx.limiter, ctx.log, tag, norm_t1c,
                     ctx.usage_store.append, api_style=ctx.t1c_api_style)


def _process_clip(job: ApiJob, ctx: PipelineCtx) -> ApiResult:
    """The per-clip turn chain: t1 -> [t1c] -> t2 -> t3 -> t4 -> assemble.
    Turns run back-to-back here on purpose: prefix cache has a TTL, so a clip's
    turns must be adjacent in time (never sweep turn-by-turn across clips)."""
    row, slice_path = job.row, job.slice_path
    clip_id, gid = row["clip_id"], int(row["global_idx"])
    log = ctx.log
    turn_stats: dict = {}
    failures = []

    def _stat(turn, ok, rec=None):
        st = turn_stats.setdefault(turn, {"ok": 0, "failed": 0, "prompt_tokens": 0,
                                          "cached_tokens": 0, "completion_tokens": 0})
        if ok:
            st["ok"] += 1
            if rec:
                st["prompt_tokens"] += rec["usage"]["prompt_tokens"]
                st["cached_tokens"] += rec["usage"]["cached_tokens"]
                st["completion_tokens"] += rec["usage"]["completion_tokens"]
        else:
            st["failed"] += 1

    # ---- T1 (with resume) ----
    t1 = None
    prior = ctx.priors["t1"].get(clip_id)
    if prior and prior.get("ok"):
        t1 = prior
        _stat("t1", True)
        log.info(f"[{clip_id}] t1 resumed from store")
    else:
        t1 = _run_t1(ctx, row, slice_path)
        if t1 is not None:
            ctx.stores["t1"].append({"clip_id": clip_id, "global_idx": gid, "turn": "t1",
                                     "ok": True, **{k: t1[k] for k in
                                                    ("parsed", "raw", "usage", "runs", "recovery")},
                                     "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            _stat("t1", True, t1)
        else:
            ctx.stores["t1"].append({"clip_id": clip_id, "global_idx": gid, "turn": "t1",
                                     "ok": False, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                      time.gmtime())})
            _stat("t1", False)
            failures.append({"clip_id": clip_id, "global_idx": gid, "turn": "t1",
                             "error": "t1_failed"})
            return ApiResult(global_idx=gid, clip_id=clip_id, failures=failures,
                             turn_stats=turn_stats)

    # ---- T1c (conditional, with resume) ----
    t1c = None
    screen_selected = "t1c" in ctx.args.turns_expanded
    # Spec §2.3: only screens IN USE, or non-screen text surfaces (whiteboard /
    # paper / sign — those were flagged precisely because visible text was seen).
    # A closed/sleeping background screen is not worth ~20k image tokens (DAY4:
    # this filter would have skipped 12/117 triggers, 5 of which returned empty).
    surfaces = t1["parsed"]["text_surfaces"]
    trigger = (screen_selected and not ctx.args.no_screen_round
               and any(s["in_use"] or s["kind"] != "screen" for s in surfaces))
    if trigger:
        prior = ctx.priors["t1c"].get(clip_id)
        if prior and prior.get("ok"):
            t1c = prior
            _stat("t1c", True)
            log.info(f"[{clip_id}] t1c resumed from store")
        else:
            t1c = _run_t1c(ctx, row)
            ok = t1c is not None
            ctx.stores["t1c"].append({"clip_id": clip_id, "global_idx": gid, "turn": "t1c",
                                      "ok": ok,
                                      **({"parsed": t1c["parsed"], "raw": t1c["raw"],
                                          "usage": t1c["usage"]}
                                         if ok else {}),
                                      "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            _stat("t1c", ok, t1c if ok else None)
            if not ok:
                log.warning(f"[{clip_id}] t1c failed; continuing without screen layer")

    # ---- T2 (with resume) ----
    t2 = None
    prior = ctx.priors["t2"].get(clip_id)
    if prior and prior.get("ok"):
        t2 = prior
        _stat("t2", True)
        log.info(f"[{clip_id}] t2 resumed from store")
    else:
        media = [video_part(slice_path)]
        task = clip_header(row) + T2_TASK
        t2 = call_turn(ctx.client, ctx.args.model,
                       build_messages(media, _ctx_t1(t1["raw"]), task),
                       ctx.cfgs["t2"], ctx.limiter, ctx.log, f"{clip_id}/t2", norm_t2,
                       ctx.usage_store.append)
        ok = t2 is not None
        ctx.stores["t2"].append({"clip_id": clip_id, "global_idx": gid, "turn": "t2", "ok": ok,
                                 **({"parsed": t2["parsed"], "raw": t2["raw"],
                                     "usage": t2["usage"], "recovery": t2["recovery"]}
                                    if ok else {}),
                                 "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        _stat("t2", ok, t2 if ok else None)
        if not ok:
            failures.append({"clip_id": clip_id, "global_idx": gid, "turn": "t2",
                             "error": "t2_failed"})
            return ApiResult(global_idx=gid, clip_id=clip_id, failures=failures,
                             turn_stats=turn_stats)

    # ---- T3 (with resume) ----
    t3 = None
    prior = ctx.priors["t3"].get(clip_id)
    if prior and prior.get("ok"):
        t3 = prior
        _stat("t3", True)
        log.info(f"[{clip_id}] t3 resumed from store")
    else:
        media = [video_part(slice_path)]
        task = clip_header(row) + T3_TASK
        ctx_text = _ctx_t1(t1["raw"]) + _ctx_t2(t2["raw"])
        t3 = call_turn(ctx.client, ctx.args.model, build_messages(media, ctx_text, task),
                       ctx.cfgs["t3"], ctx.limiter, ctx.log, f"{clip_id}/t3", norm_t3,
                       ctx.usage_store.append)
        ok = t3 is not None
        ctx.stores["t3"].append({"clip_id": clip_id, "global_idx": gid, "turn": "t3", "ok": ok,
                                 **({"parsed": t3["parsed"], "raw": t3["raw"],
                                     "usage": t3["usage"], "recovery": t3["recovery"]}
                                    if ok else {}),
                                 "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        _stat("t3", ok, t3 if ok else None)
        if not ok:
            failures.append({"clip_id": clip_id, "global_idx": gid, "turn": "t3",
                             "error": "t3_failed"})
            return ApiResult(global_idx=gid, clip_id=clip_id, failures=failures,
                             turn_stats=turn_stats)

    # ---- T4 (with resume; text-only unless --t4-video) ----
    t4 = None
    prior = ctx.priors["t4"].get(clip_id)
    if prior and prior.get("ok"):
        t4 = prior
        _stat("t4", True)
        log.info(f"[{clip_id}] t4 resumed from store")
    else:
        media = [video_part(slice_path)] if ctx.args.t4_video else []
        task = clip_header(row) + T4_TASK
        ctx_text = _ctx_t1(t1["raw"]) + _ctx_t2(t2["raw"]) + _ctx_t3(t3["raw"])
        t4 = call_turn(ctx.client, ctx.args.model, build_messages(media, ctx_text, task),
                       ctx.cfgs["t4"], ctx.limiter, ctx.log, f"{clip_id}/t4", norm_t4,
                       ctx.usage_store.append)
        ok = t4 is not None
        ctx.stores["t4"].append({"clip_id": clip_id, "global_idx": gid, "turn": "t4", "ok": ok,
                                 **({"parsed": t4["parsed"], "raw": t4["raw"],
                                     "usage": t4["usage"], "recovery": t4["recovery"]}
                                    if ok else {}),
                                 "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        _stat("t4", ok, t4 if ok else None)
        if not ok:
            failures.append({"clip_id": clip_id, "global_idx": gid, "turn": "t4",
                             "error": "t4_failed"})
            return ApiResult(global_idx=gid, clip_id=clip_id, failures=failures,
                             turn_stats=turn_stats)

    # ---- assemble + QC ----
    final = assemble_final(row, t1, t1c, t2, t3, t4, ctx.args.model)
    final["qc"] = run_qc(final, log)
    return ApiResult(global_idx=gid, clip_id=clip_id, caption_records=[final],
                     failures=failures, turn_stats=turn_stats)


def _api_worker(in_q, out_q, ctx: PipelineCtx, worker_id):
    log = ctx.log
    while True:
        job = in_q.get()
        if job is None:
            in_q.task_done()
            return
        row, slice_path = job.row, job.slice_path
        clip_id, gid = row["clip_id"], int(row["global_idx"])
        if not slice_path.exists():
            log.error(f"[{clip_id}] slice not found: {slice_path}")
            out_q.put(ApiResult(global_idx=gid, clip_id=clip_id,
                                failures=[{"clip_id": clip_id, "global_idx": gid,
                                           "error": "slice_not_found"}]))
            in_q.task_done()
            continue
        try:
            res = _process_clip(job, ctx)
        except Exception as e:
            log.error(f"[{clip_id}] worker crash: {type(e).__name__}: {e}")
            res = ApiResult(global_idx=gid, clip_id=clip_id,
                            failures=[{"clip_id": clip_id, "global_idx": gid,
                                       "error": f"worker_crash: {type(e).__name__}: {e}"}])
        out_q.put(res)
        in_q.task_done()


# ===========================================================================
# Resume helpers for the final record file
# ===========================================================================

def _iter_caption_records(captions_path: Path):
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
        i = end


def load_existing_clip_ids(captions_path: Path):
    return {rec["clip_id"] for rec in _iter_caption_records(captions_path)}


def max_written_global_idx(captions_path: Path):
    top = 0
    for rec in _iter_caption_records(captions_path):
        top = max(top, int(rec.get("global_idx", 0)))
    return top


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Multi-turn Mimo captioning pipeline for EgoLife (spec: caption_pipeline_multiturn.md).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Selection ---
    ap.add_argument("--participant", default="A1_JAKE")
    ap.add_argument("--day", type=int, default=1)
    ap.add_argument("--start-time", default=None, help="start HHMM inclusive, e.g. 1121")
    ap.add_argument("--end-time", default=None, help="end HHMM exclusive, e.g. 1122")
    ap.add_argument("--clip-duration", type=int, default=10, choices=[5, 6, 10, 15, 30],
                    help="split target in seconds; 30 = no splitting")
    # --- Paths ---
    ap.add_argument("--src-dir", type=Path, default=None,
                    help="ROOT of the video tree (default: ./videos)")
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
    ap.add_argument("--env-file", type=Path, default=None)
    ap.add_argument("--thinking", default="enabled", choices=["enabled", "disabled"],
                    help="thinking for T2/T3/T4 (reasoning-heavy rounds)")
    ap.add_argument("--t4-thinking", default=None, choices=["enabled", "disabled"],
                    help="override thinking for T4 only (default: follow --thinking). T4 is "
                         "the slowest thinking round; no-think T4 writes more invalid edge "
                         "types (QC flags them) but runs ~3x faster")
    ap.add_argument("--t1-thinking", default="disabled", choices=["enabled", "disabled"],
                    help="thinking for T1/T1c perception rounds (k votes each anyway)")
    # --- Multiturn-specific ---
    ap.add_argument("--votes", type=int, default=DEFAULT_VOTES,
                    help="T1 independent perception runs per clip (spec: k=2 + arbitration)")
    ap.add_argument("--turns", default="t1,t1c,t2,t3,t4",
                    help="comma list of turns to run; prerequisites auto-included; "
                         "final record assembled when t1..t4 are all ok")
    ap.add_argument("--no-screen-round", action="store_true",
                    help="never trigger T1c even when text surfaces are detected")
    ap.add_argument("--t4-video", action="store_true",
                    help="include the video in T4 (default off: text-only causal reasoning)")
    ap.add_argument("--t1c-frame-fps", type=float, default=NATIVE_FRAME_FPS,
                    help="T1c keyframe sampling rate (frames/sec from the source)")
    ap.add_argument("--t1c-frame-size", type=int, default=0,
                    help="cap T1c frame long edge in px (0 = native). The image API has no "
                         "server-side resolution knob and bills (w/32)*(h/32) tokens per "
                         "frame (~1900 for a 1408x1408 frame vs ~300 for a default video "
                         "frame), so downscaling is on us.")
    ap.add_argument("--t1c-provider", default="mimo", choices=["mimo", "custom"],
                    help="T1c backend: mimo (default) or custom = any OpenAI-compatible "
                         "image VLM via --t1c-base-url/--t1c-model/--t1c-api-key-env "
                         "(screen reading needs vision only, no audio)")
    ap.add_argument("--t1c-base-url", default=None, help="custom T1c provider base URL")
    ap.add_argument("--t1c-model", default=None, help="custom T1c provider model name")
    ap.add_argument("--t1c-api-key-env", default=None,
                    help="env var holding the custom T1c provider API key")
    ap.add_argument("--t1c-no-json-mode", action="store_true",
                    help="omit response_format=json_object for the custom T1c provider "
                         "(use if it rejects json_object, e.g. some vision models; the "
                         "prompt already demands JSON and parsing is tolerant)")
    # --- Concurrency ---
    ap.add_argument("--max-rpm", type=int, default=90, help="global RPM cap")
    ap.add_argument("--api-workers", type=int, default=None,
                    help="API worker threads (default: min(--max-rpm/2, 20) — thinking-mode "
                         "requests take ~30s+ each, so ~2 requests/min per worker; scales with "
                         "--max-rpm but caps at 20 to keep one machine's connections modest)")
    ap.add_argument("--preprocess-workers", type=int, default=2)
    # --- Flow ---
    ap.add_argument("--skip-preprocess", action="store_true",
                    help="caption only, reading slices from _cache/")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--reset", action="store_true", help="delete output files before running")
    ap.add_argument("--reset-yes-i-know", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only process first N units (debug)")
    args = ap.parse_args()

    # Thinking-mode requests run ~30s+; scale workers with the RPM cap but cap at
    # 20 so a single machine doesn't open excessive connections.
    if args.api_workers is None:
        args.api_workers = min(args.max_rpm // 2, 20)

    # turn selection with prerequisites
    selected = {t.strip() for t in args.turns.split(",") if t.strip()}
    unknown = selected - set(_TURN_NAMES)
    if unknown:
        print(f"ERROR: unknown turns {sorted(unknown)}; valid: {list(_TURN_NAMES)}", file=sys.stderr)
        sys.exit(2)
    expanded = set(selected)
    for t in selected:
        expanded |= _TURN_PREREQ.get(t, set())
    args.turns_expanded = expanded

    # --- API key ---
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

    # --- Paths ---
    src_root = args.src_dir or (ROOT / "videos")
    time_tag = f"{args.start_time}-{args.end_time}" if (args.start_time and args.end_time) else "full"
    if args.out is None:
        out_file = ROOT / "captions" / args.participant / f"DAY{args.day}" / f"{time_tag}.jsonl"
    else:
        out_file = args.out
    out_file.parent.mkdir(parents=True, exist_ok=True)
    captions_dir = out_file.parent
    stem = out_file.stem
    usage_path = captions_dir / f"{stem}_usage.jsonl"
    summary_path = captions_dir / f"{stem}_summary.json"
    log_path = captions_dir / f"{stem}_run.log"
    cache_dir = captions_dir / "_cache"
    slices_dir = cache_dir / "slices"
    frames_dir = cache_dir / "frames_native"
    for d in (cache_dir, slices_dir, frames_dir):
        d.mkdir(parents=True, exist_ok=True)
    clips_parquet = cache_dir / "clips.parquet"

    if args.reset:
        existing = 0
        if out_file.exists():
            try:
                existing = sum(1 for line in out_file.open(encoding="utf-8")
                               if line.strip().startswith("{"))
            except Exception:
                pass
        if existing > 10 and not args.reset_yes_i_know:
            print(f"ERROR: --reset would delete {out_file} which already has {existing} records.\n"
                  f"  If you really mean it, add --reset-yes-i-know.", file=sys.stderr)
            sys.exit(2)
        for p in [out_file, usage_path, summary_path, log_path] + \
                 [captions_dir / f"{stem}_{t}.jsonl" for t in _TURN_NAMES]:
            if p.exists():
                p.unlink()

    # --- Logging ---
    log = logging.getLogger("caption_pipeline_multiturn")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    if tqdm is not None:
        th = TqdmLoggingHandler()
        th.setFormatter(fmt)
        log.addHandler(th)
    else:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)

    log.info(f"multiturn pipeline: participant={args.participant} day={args.day} "
             f"time={args.start_time or '00:00'}-{args.end_time or '23:59'} "
             f"clip_duration={args.clip_duration}s votes={args.votes} "
             f"turns={sorted(args.turns_expanded)} t4_video={args.t4_video}")
    log.info(f"model={args.model} thinking(t2-4)={args.thinking} thinking(t1)={args.t1_thinking}")
    log.info(f"output: {out_file}")

    # --- Turn configs / stores / client ---
    cfgs = {
        "t1": TurnApiConfig(args.t1_thinking),
        "t1m": TurnApiConfig(args.t1_thinking),
        "t1c": TurnApiConfig(args.t1_thinking, json_mode=not args.t1c_no_json_mode),
        "t2": TurnApiConfig(args.thinking),
        "t3": TurnApiConfig(args.thinking),
        "t4": TurnApiConfig(args.t4_thinking or args.thinking),
    }
    stores = {t: TurnStore(captions_dir / f"{stem}_{t}.jsonl") for t in _TURN_NAMES}
    stores["_usage"] = TurnStore(usage_path)
    priors = {t: stores[t].load() for t in _TURN_NAMES}
    log.info("priors loaded: " + ", ".join(f"{t}={len(v)}" for t, v in priors.items()))
    limiter = SlidingWindowRateLimiter(args.max_rpm)
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    if args.t1c_provider == "custom":
        t1c_key_env = args.t1c_api_key_env or args.api_key_env
        t1c_key = os.environ.get(t1c_key_env)
        if not t1c_key:
            print(f"ERROR: {t1c_key_env} not set for --t1c-provider custom.", file=sys.stderr)
            sys.exit(2)
        t1c_client = OpenAI(api_key=t1c_key, base_url=args.t1c_base_url or args.base_url)
        t1c_model = args.t1c_model or args.model
        t1c_api_style = "openai"
        log.info(f"t1c provider: custom model={t1c_model} base_url={t1c_client.base_url}")
    else:
        t1c_client, t1c_model, t1c_api_style = client, args.model, "mimo"
    paths = {"out_dir": captions_dir, "cache_dir": cache_dir, "frames_dir": frames_dir}
    ctx = PipelineCtx(args, stores, cfgs, client, limiter, log, paths, priors,
                      t1c_client=t1c_client, t1c_model=t1c_model, t1c_api_style=t1c_api_style)

    # --- Decide clip rows / units ---
    import pandas as pd
    if args.skip_preprocess:
        if not clips_parquet.exists():
            log.error(f"--skip-preprocess but {clips_parquet} not found")
            sys.exit(2)
        clips_df = pd.read_parquet(clips_parquet).sort_values("global_idx").reset_index(drop=True)
        log.info(f"--skip-preprocess: loaded {len(clips_df)} rows from {clips_parquet}")
        units = None
        piece_counts = None
    else:
        try:
            files, breaks = select_source_files(src_root, args.participant, args.day,
                                                args.start_time, args.end_time)
        except FileNotFoundError as e:
            log.error(str(e))
            sys.exit(2)
        log.info(f"source: {len(files)} files, {len(breaks)} recording break(s)")
        units = group_into_clips(files, args.participant, args.day, args.clip_duration)
        piece_counts = count_pieces_for_units(units, log=log)
        log.info(f"units: {len(units)} source files -> {sum(piece_counts)} pieces "
                 f"(target={args.clip_duration}s)")
        clips_df = None

    # --- Resume ---
    resume_idx = max_written_global_idx(out_file) if args.skip_existing else 0
    done_ids = load_existing_clip_ids(out_file) if args.skip_existing else set()
    log.info(f"resume: skip_existing={args.skip_existing} already_done={len(done_ids)} "
             f"max_gid={resume_idx}")

    produced_q: queue.Queue = queue.Queue()
    api_in_q: queue.Queue = queue.Queue(maxsize=args.api_workers * 2)
    result_q: queue.Queue = queue.Queue()
    writer = OrderedWriter(out_file, log, next_idx=resume_idx + 1)

    all_failures: list = []
    turn_totals: dict = {}
    qc_totals: dict = {}
    t_start = time.time()

    # --- Preprocess pool (or skip) ---
    pp_job_q: queue.Queue = queue.Queue()
    pp_threads = []
    if args.skip_preprocess:
        rows = clips_df
        if args.limit:
            rows = rows.head(args.limit)
        fed = 0
        skipped_existing = 0
        for _, row in rows.iterrows():
            clip_id = row["clip_id"]
            if args.skip_existing and (clip_id in done_ids or int(row["global_idx"]) <= resume_idx):
                skipped_existing += 1
                continue
            sp = row["slice_path"]
            api_in_q.put(ApiJob(row=row.to_dict(),
                                slice_path=(cache_dir / sp) if not Path(sp).is_absolute() else Path(sp)))
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
        limit_n = args.limit if args.limit else len(units)
        piece_counts_enq = piece_counts[:limit_n]
        expected_total = sum(piece_counts_enq)
        idx_bases_enq = []
        _running = 1
        for n in piece_counts_enq:
            idx_bases_enq.append(_running)
            _running += n
        for i, unit in enumerate(units_to_enqueue):
            pp_job_q.put(PreprocessJob(unit=unit, idx_base=idx_bases_enq[i],
                                       is_first_unit=(i == 0), day=args.day, user=args.participant,
                                       skip_if_exists=(not args.reset), resolution=args.resolution,
                                       fps=args.fps, crf=args.crf, audio_k=args.audio_k,
                                       slices_dir=slices_dir, out_dir=cache_dir))
        for _ in range(args.preprocess_workers):
            pp_job_q.put(None)
        pp_done, pp_expected, api_fed = False, expected_total, 0
        skipped_existing = 0

    # --- Progress bar ---
    pbar = None
    if tqdm is not None and expected_total > 0:
        pbar = tqdm(total=expected_total, unit="clip", desc="multiturn",
                    mininterval=0.5, smoothing=0.3)
        pbar.refresh()

    # --- API worker pool ---
    api_threads = []
    for w in range(args.api_workers):
        t = threading.Thread(target=_api_worker,
                             args=(api_in_q, result_q, ctx, w), name=f"api-{w}", daemon=True)
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
                all_failures.append({"clip_id": pres.clip_id, "global_idx": pres.global_idx,
                                     "error": pres.error})
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
        pbar.set_postfix(rpm=f"{rpm:.1f}", eta=f"{eta_min:.1f}min", refresh=False)
        pbar.update(1)

    while True:
        if not args.skip_preprocess:
            bridge_produced()
            if pbar is not None and pp_done and not pbar_total_adjusted:
                if api_fed != pbar.total:
                    pbar.total = max(api_fed, pbar.n)
                    pbar.refresh()
                pbar_total_adjusted = True
        now = time.time()
        if pbar is None and now - last_progress_t >= PROGRESS_INTERVAL_S and expected_total > 0:
            elapsed = now - t_start
            rate = results_received / max(elapsed / 60, 1e-9)
            eta_min = (expected_total - results_received) / rate if rate > 0 else float("inf")
            log.info(f"[progress] api={results_received}/{api_fed} failed={len(all_failures)} "
                     f"elapsed={elapsed/60:.1f}min rpm={rate:.1f} eta={eta_min:.1f}min")
            last_progress_t = now
        try:
            res = result_q.get(timeout=0.2)
        except queue.Empty:
            if pp_done and api_fed <= results_received:
                break
            continue
        results_received += 1
        all_failures.extend(res.failures)
        for turn, st in res.turn_stats.items():
            tgt = turn_totals.setdefault(turn, {"ok": 0, "failed": 0, "prompt_tokens": 0,
                                                "cached_tokens": 0, "completion_tokens": 0})
            for key in tgt:
                tgt[key] += st.get(key, 0)
        for cap in res.caption_records:
            for kq, vq in cap.get("qc", {}).items():
                qc_totals[kq] = qc_totals.get(kq, 0) + (int(vq) if isinstance(vq, bool) else vq)
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
    for st in stores.values():
        st.close()
    elapsed = time.time() - t_start

    if pp_results_for_parquet:
        try:
            pd.DataFrame(pp_results_for_parquet).to_parquet(clips_parquet, index=False)
        except Exception as e:
            log.warning(f"could not write {clips_parquet}: {e}")

    # cache stats per turn — the Plan A vs Plan B evidence (spec §6.0)
    cache_stats = {}
    for turn, st in turn_totals.items():
        pt, ct = st["prompt_tokens"], st["cached_tokens"]
        cache_stats[turn] = {**st,
                             "cache_hit_rate": round(ct / pt, 4) if pt else None}
    n_ok = writer.n_written
    summary = {
        "pipeline": "multiturn", "model": args.model,
        "participant": args.participant, "day": args.day,
        "time_range": {"start": args.start_time, "end": args.end_time},
        "clip_duration": args.clip_duration, "votes": args.votes,
        "turns_selected": sorted(args.turns_expanded),
        "thinking": {"t1": args.t1_thinking, "t234": args.thinking},
        "t4_video": args.t4_video, "no_screen_round": args.no_screen_round,
        "t1c": {"provider": args.t1c_provider, "model": (args.t1c_model or args.model),
                "frame_fps": args.t1c_frame_fps, "frame_size": args.t1c_frame_size},
        "max_rpm": args.max_rpm, "api_workers": args.api_workers,
        "n_clips_input": expected_total, "n_clips_ok": n_ok,
        "n_clips_failed": len(all_failures), "n_skipped_existing": skipped_existing,
        "elapsed_s": round(elapsed, 2),
        "effective_rpm": round(results_received / max(elapsed / 60, 1e-9), 2),
        "turns_stats": turn_totals,
        "cache_stats": cache_stats,
        "qc_totals": qc_totals,
        "failures": all_failures,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info(f"done. ok={n_ok} failed={len(all_failures)} skipped={skipped_existing}")
    log.info(f"  elapsed={elapsed:.1f}s ({elapsed/60:.1f}min)  "
             f"effective_rpm={results_received / max(elapsed / 60, 1e-9):.1f}")
    for turn in _TURN_NAMES:
        if turn in turn_totals:
            st = turn_totals[turn]
            rate = (f"{st['cached_tokens']}/{st['prompt_tokens']}"
                    f"={st['cached_tokens']/st['prompt_tokens']:.1%}") if st["prompt_tokens"] else "-"
            log.info(f"  {turn}: ok={st['ok']} failed={st['failed']} cache={rate}")
    if qc_totals:
        log.info(f"  qc: {qc_totals}")
    log.info(f"  captions -> {out_file}")
    log.info(f"  summary  -> {summary_path}")


if __name__ == "__main__":
    main()
