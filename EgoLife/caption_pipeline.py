"""caption_pipeline.py — self-contained Mimo captioning pipeline for EgoLife.

Turns first-person (Meta Aria) video segments into simulation-grade ATOMIC annotations
for a life digital twin: self_actions / others / environment (overall scene) /
env_changes (atomic state transitions, cause-attributed) / speech / psychology
(emotion) / causal_links (directed env<->action<->emotion edges, incl. env->env),
plus optional OCR. The schema encodes the causal loop a simulator needs:
emotion drives actions, actions mutate the environment, the environment feeds
back into both actions and emotion. Every `causal_links` edge carries a counterfactual
`strength` (strong/moderate/weak), so the causal graph is weighted, not just binary.

By default each ~30s source file is split into 3 x ~10s pieces and each piece is
captioned independently (finer granularity, lower per-call rejection rate). Use
--clip-duration to choose a different split target (5/6/10/15/30); 30 disables
splitting (one caption per source file, the legacy behaviour).

Single file, no sibling-module imports. Depends only on the OpenAI SDK,
pandas, python-dotenv, jsonschema, and ffmpeg/ffprobe on PATH.

Architecture
------------
  [PreprocessWorker pool]   [RateLimiter]   [API Worker pool]   [Writer]
  ffmpeg split/re-encode ->  produced_q  ->  acquire()  ->  Mimo call  ->  result_q  ->  ordered jsonl
   (CPU/subprocess)          (bounded)      (global, RPM cap)       (IO)                (single thread)

Usage
-----
  # standard run: 10s pieces (default), JSON output, thinking ON (causal
  # annotation is reasoning-heavy; ~4x latency/tokens vs non-thinking)
  python caption_pipeline.py --participant A1_JAKE --day 1 --max-rpm 90
  # cheaper/faster: no reasoning chain (action/env layers still fine, causal
  # links get noticeably weaker)
  python caption_pipeline.py --thinking disabled
  # legacy: one caption per 30s source file (no splitting)
  python caption_pipeline.py --participant A1_JAKE --day 1 --clip-duration 30
  # time-windowed
  python caption_pipeline.py --start-time 1110 --end-time 1130
  # resume after an interruption (skips clips already in the output file)
  python caption_pipeline.py --skip-existing
  # caption only (slices already encoded in _cache/)
  python caption_pipeline.py --skip-preprocess
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

THINKING_DEFAULT = "enabled"       # causal-graph annotation is reasoning-heavy; disable for ~4x speedup

# ffmpeg re-encode defaults (1024x1024 @ 2fps is the validated sweet spot:
# watermark readable, whiteboard legible, ~2MB / 30s clip).
DEFAULT_RESOLUTION = 1024
DEFAULT_FPS = 2
DEFAULT_CRF = 28
DEFAULT_AUDIO_BITRATE_K = 64

# A gap >this between consecutive source timestamps is treated as a recording
# break (e.g. the multi-hour gaps in EgoLife). 60s pairing never bridges these.
_GAP_THRESHOLD_S = 35.0

SYSTEM_MSG = """You are a dense first-person life-log captioner producing SIMULATION-GRADE atomic
annotations. Your output feeds a digital twin of the wearer's daily life, so it must capture not
only WHAT happened but the latent and causal structure of the clip: the people present, the
stateful environment, the wearer's hidden psychology (emotion, mental activity, intent), and the
directed edges through which emotion drives actions, actions change the environment, and the
environment feeds back into both.

The footage is from participant A1_JAKE wearing Meta Aria glasses. People may speak Chinese or English.

# Clip sampling (CRITICAL)
The clip is downsampled to 2 fps: you receive roughly 2 frames per second. Motion between
consecutive frames can JUMP — hands and objects may teleport between samples. Always read
timestamps from the watermark, never by counting frames. Fast gestures may be only partially
captured: describe the visible endpoints honestly and do NOT interpolate unobserved intermediate
micro-steps. If an object appears or disappears between two samples, timestamp the change with
the first frame where the new state is visible.

# Watermark anchor (CRITICAL)
Every video frame carries a watermark in the TOP-LEFT corner showing the current time and day,
formatted "HH:MM:SS:FF DAYn" (e.g. "11:10:02:00 DAY1"). FF is a frame counter 00-19.
The user message tells you the day label and the approximate start time of this clip.
You MUST read the watermark to timestamp the actions and changes you describe, so they can be
located on a timeline. Never transcribe the watermark itself as screen text.

# Output format
Return ONLY a single JSON object. No explanations, no markdown code fences, no text outside JSON.
Use exactly this structure (fill every field; use empty arrays/strings when a section is truly empty):

{
  "environment": {
    "setting": "...",
    "near_field": ["..."],
    "background": "..."
  },
  "people": [
    {"id": "P1", "descriptor": "...", "name": "", "note": ""}
  ],
  "self_actions": [
    {"time": "HH:MM:SS", "time_end": "HH:MM:SS", "text": "..."}
  ],
  "other_actions": [
    {"time": "HH:MM:SS", "time_end": "HH:MM:SS", "person": "P1", "text": "..."}
  ],
  "env_changes": [
    {"time": "HH:MM:SS", "time_end": "HH:MM:SS", "text": "...", "cause": "self"}
  ],
  "speech": [
    {"lang": "zh", "speaker": "P1", "text": "..."}
  ],
  "sound": [
    {"time": "HH:MM:SS", "text": "...", "source": ""}
  ],
  "interface": [
    {"time": "HH:MM:SS", "where": "laptop screen", "app_or_site": "...", "content": "...", "note": ""}
  ],
  "psychology": {"emotion": "...", "mental_activity": "..."},
  "causal_links": [
    {"type": "env->action", "cause": "...", "effect": "...", "strength": "strong"}
  ]
}

## Field rules

environment: an object with three sub-fields describing the scene a simulator would spawn the
agent into — it should read like a scene description, NOT a changelog. In-clip CHANGES belong in
env_changes, not here (a location transition appears both as a one-clause note in "setting" and
as env_changes entries).
  setting: one or two sentences on the OVERALL stable backdrop the clip happens in: place type,
    room layout, lighting, weather / indoor-outdoor cues, ambient soundscape.
  near_field: an array of short phrases, one per notable NEAR-FIELD object — things I or others
    touch, hold, or are likely to interact with (on the desk, in hand, within arm's reach) —
    each with a rough location ("白色马克杯（桌面右手边）", "laptop, open, directly ahead").
    # Object naming (anchoring, CRITICAL): the label you pick here is the object's canonical
    name for the whole clip — reuse it verbatim in self_actions, other_actions, env_changes and
    causal_links; never rename the same object between entries. If an object cannot be
    confidently identified (too close / out of focus / partially occluded / reflective /
    covered by stickers), DO NOT guess a specific category: describe its observable attributes
    honestly (shape, size, color, material, transparency) — "一个贴有标签的半透明玻璃容器，距离
    过近细节模糊" is a CORRECT entry; confidently calling it "一块布" is a hallucination.
  background: one short clause summarizing the rest of the scene ("桌上散落多个包装盒与线缆");
    empty string if nothing worth noting.

people: the cast list of OTHER people who appear in, or are clearly present during, this clip —
never yourself. Give each a stable short id ("P1", "P2", ...) and ONE stable visual descriptor
("穿蓝衬衫的女性", "the man in the grey hoodie"); reuse the same id and descriptor everywhere
(other_actions, speech, causal_links). Fill "name" ONLY if a name is actually spoken or shown in
this clip; otherwise leave it empty — never invent names. "note" is optional: position relative
to me, role cues, partial visibility ("只在画面右侧边缘出现"). Empty array if I am alone.

self_actions: one object per ATOMIC self-action, in first person ("我"), present tense, verb-led.
An atomic action is a single verb-level step with one object and one immediate goal: reach, grasp,
pick up, put down, open, close, turn on, flip, slide, look at, walk to, sit down. DECOMPOSE compound
behavior: "我拿起手机划开屏幕看消息" must become at least "我拿起手机" + "我划开屏幕" + "我低头浏览消息".
Conversely, do NOT split one continuous gesture into artificial micro-frames, and do NOT merge
separate manipulations into one entry. There is NO count quota — atomicity is the standard, not
quantity: a dense 10s of object manipulation may need 6-10 entries; a still 10s of sitting may
honestly be 1-2 entries. Cover the whole clip span. Each "time"/"time_end" is a watermark-based
HH:MM:SS (~1-3s resolution). Describe only the OBSERVABLE: hand-object contact, posture, gaze,
locomotion, device use.
  # Intent ban (CRITICAL): actions and intent live in DIFFERENT fields. This field describes only
  what is observed: postures, contacts, movements. Never write 准备 / 打算 / 想要 / 试图 to do X
  here unless X is actually observed within this clip. If the clip ends mid-activity, the last
  action simply ends there. An unmade bed does NOT license "准备整理床铺" unless tidying is
  actually observed. Intent, plans and goals are LATENT states — they belong in
  psychology.mental_activity, and must never leak into actions, env_changes, or causal_links.

other_actions: zero or more objects, same shape and same atomicity standard as self_actions,
timestamped the same way, plus a "person" field carrying the id from people. Lead the text with
the person descriptor. The same intent ban applies: observed movements only, no goals. Empty
array if I am alone.

env_changes: one object per ATOMIC observable state change of the environment DURING this clip:
an object appears / disappears / moves; a device or aperture changes state (screen lights up or
sleeps, door opens, cap comes off, cup empties); lighting or soundscape shifts; layout is
rearranged; a person enters or leaves the scene. Even in a short clip the environment is a
stateful actor, not a static backdrop — report every discrete transition you can see. Timestamp
each entry like an action. Every entry carries "cause": "self" (my action did it), "other"
(another person did it — name their id in the text), or "external" (no visible agent — automatic
door, weather, a timer). Changes are the object-level mirror of actions: if I pick up a cup, the
pick-up is a self_action AND "杯子离开桌面进入我手中" is an env_change with cause "self". Only
report changes actually visible in this clip. Empty array if the environment truly does not change.

speech: zero or more objects. lang in {"zh","en"}; speaker is "self" for me, otherwise the person
id from people (or a short descriptor when unclear); text is the quoted utterance. Transcribe only
what is actually heard; do not invent. Speech has no reliable timestamp from the watermark, so do
NOT timestamp it. Empty array if silent.

sound: zero or more objects for discrete NON-SPEECH sounds: notification chimes, ringtones,
vibration buzz, keyboard clatter, door slams, knocks, footsteps, music from a speaker, TV audio,
traffic, AC hum, microwave beeps, a cup set down hard — and non-linguistic human sounds such as
laughter, coughing or sighing (name the person id in the text if identifiable). "text" describes
the sound; "source" names the visible or confidently inferred source, empty string if unknown.
Sounds have no watermark of their own, so include "time" ONLY when the moment is visually anchored
(the phone visibly lights up as it chimes; a door visibly slams) — otherwise omit "time".
Continuous background ambience belongs in environment, not here. Empty array if nothing notable.

interface: CONDITIONAL — include entries ONLY when a screen / electronic device (phone, laptop,
monitor, TV, tablet) or a substantial legible text surface (whiteboard, paper, sign, packaging)
appears. For SCREENS, go beyond raw OCR: identify and understand the page —
  "where": the device ("手机屏幕", "laptop screen", ...);
  "app_or_site": which app or website, if identifiable (bilibili, 知乎, 微信, 文件资源管理器, ...);
  "content": the page type plus its key legible content, verbatim when readable — on a bilibili
    video page, the video title and UP主 name; on 知乎, the question title and the visible answer;
    in a file explorer, the visible folder path and file names; in a chat window, the latest
    message snippets; on a shopping or news site, the headline / item title.
For physical text surfaces, transcribe what is legible, verbatim. When the on-screen content
changes mid-clip (app switch, new page, a scroll that reveals a new section), add a NEW entry
anchored with its "time". "note" records legibility caveats ("标题只露出前半句", "screen partially
off-frame"). Do NOT transcribe the top-left time watermark. Transcribe only what is actually
legible; never invent. Empty array if no screen or text surface appears.

psychology: an object with exactly two keys, estimated AS IF THIS CLIP WERE YOUR ONLY EVIDENCE —
ignore anything that might have happened before it. The question is: "considering only this
clip's situation, what would a plausible inner state be?"
  emotion: one or two lowercase keywords (required; "neutral" if none readable), picked from:
    neutral calm relaxed bored focused amused happy excited surprised confused curious anxious
    nervous stressed frustrated angry embarrassed sad tired sleepy hungry, or similar.
  mental_activity: one or two short sentences of plausible ongoing thought, in first person: what
    has my attention right now, what I am mulling over, and — where the clip's situation alone
    supports it — my INTENT: what I seem to be getting done next. This is the ONLY field where
    intent may appear. Ground every word in this clip's visible situation; when the clip offers
    no inner-state cues, an honest "无特别线索，注意力在手头的事情上" beats an invented agenda.
When an environment event or another person moved my emotion or thought, say so here AND record
it as an "env->emotion" / "other->emotion" causal_link.

causal_links: the directed causal edges you can observe or confidently infer WITHIN this clip.
One object per edge: {"type": "...", "cause": "<short phrase, prefix with HH:MM:SS when clear>",
"effect": "<short phrase, prefix with HH:MM:SS when clear>", "strength": "strong|moderate|weak"}.
Use EXACTLY these "type" strings:
  "env->action"     an environment event/state changed MY behavior (phone vibrates -> I pick it up).
  "env->emotion"    an environment event changed my emotion (loud noise -> startled/annoyed).
  "emotion->action" my emotion directly drove my behavior (bored -> I start scrolling my phone).
  "action->env"     my action changed the environment (I flip the switch -> lights turn on);
                    mirrors env_changes entries with cause "self".
  "other->action"   another person's action triggered my action (colleague waves me over -> I walk over).
  "other->env"      another person's action changed the environment (she opens the curtain -> room brightens).
  "other->emotion"  another person's action changed my emotion (guest laughs -> I relax).
  "env->env"        an environmental event with NO visible agent causes another environmental
                    change (云遮住阳光 -> 室内变暗; 风把门吹开).
Every edge carries a REQUIRED "strength" grading how strongly the cause drove the effect,
defined counterfactually:
  strong  = trigger: without this cause the effect likely would NOT have happened, or would have
            gone a different direction (phone vibrates -> I pick it up; I flip the switch ->
            lights on).
  moderate= shaper: the cause changed HOW/WHEN/how vigorously the effect happened, but the effect
            would still have occurred (boredom speeds up my scrolling; his tone makes me answer
            more carefully).
  weak    = background: one contributory factor among several; the effect was mostly driven by
            habit or task (mild tiredness -> I rub my eyes once).
Physical edges (action->env, other->env) will almost always be "strong" — that is expected, not
lazy. Grade honestly; do not pad the graph with weak background edges.
Rules: the causal direction must be visible or strongly implied by temporal order and mechanism —
never fabricate; omit an edge only when you doubt it EXISTS (strength "weak" is the honest way to
record a real but minor influence). 0-3 links is typical; empty array is fine. Quote the atomic
action / change texts (with their times) rather than vague summaries.

# Rules
- Stay strictly within the watermark time range of THIS clip; never describe other videos.
- Be concrete and observational; do not speculate beyond what is visible or audible. The single
  exception is psychology, which is explicitly an inference field — but it too must be grounded
  in this clip alone. causal_links must stay evidence-based (visible temporal order + plausible
  mechanism).
- Pick ONE language for all fields based on the dominant spoken language in the clip (Chinese if
  participants speak Chinese, English otherwise). Do NOT translate or duplicate content.
- Output each action, utterance, or fact exactly ONCE; never repeat in another language.
- Output must be valid JSON: double quotes, no trailing commas, no comments."""

USER_TASK_TMPL = (
    "Annotate this {duration_desc} first-person video ({clip_id}) and return the JSON object.\n"
    "Source file: {src_file}, recorded on {day_label}; the segment starts at approximately {start_hms}.\n"
    "The clip is downsampled to 2 fps: read the top-left watermark (HH:MM:SS:FF {day_label}) to "
    "timestamp self_actions, other_actions, env_changes, and any visually anchored sounds or screen "
    "page changes — never time anything by counting frames. Cover the full segment from {start_hms} onward.\n"
    "Describe the scene as setting + near-field objects + background, and list the people present. "
    "Decompose what happens into ATOMIC actions and ATOMIC environment state changes — there is NO "
    "count quota; atomicity is the standard. Transcribe speech, log notable non-speech sounds, and "
    "for any screen on camera identify the app/site, the page type and its key visible text "
    "(bilibili 视频标题 + UP主, 知乎 问题 + 回答, 文件资源管理器里的文件名, ...). Keep actions "
    "strictly observable — intent belongs only in psychology.mental_activity. Then record every "
    "causal edge you can defend from the clip itself: env->action, env->emotion, emotion->action, "
    "action->env, other->action, other->env, other->emotion, env->env — each graded strength "
    "strong/moderate/weak by how strongly the cause drove the effect."
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


def slice_to_10s(src_video: Path, clip_id: str, out_dir: Path) -> list[Path]:
    """Recovery-only: slice a rejected ~30s clip into up to 3 x ~10s pieces
    (re-encoded for accurate cuts). Each piece is then captioned on its own.
    Only invoked when --clip-duration 30; first-class 10s mode never reaches here."""
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


@dataclass
class SliceInfo:
    """One output piece from splitting a source file. For target_s that divides the
    source evenly (e.g. 30s→3×10s), every piece has the same duration. The last piece
    of an irregular clip (e.g. 22s→3×7.33s) carries any remainder implicitly via the
    even division. start_hms is the watermark-derived HH:MM:SS at the piece start."""
    path: Path
    duration: float
    piece_idx: int        # 1-based
    start_hms: str        # "HH:MM:SS"
    is_whole: bool        # True when n==1 (no splitting happened)


def _add_seconds_to_hms(hms: str, seconds: float) -> str:
    """Add `seconds` to a "HH:MM:SS" watermark timestamp, rolling over minutes/hours.
    Used to compute each slice's watermark start from the source clip's start."""
    hh, mm, ss = (int(x) for x in hms.split(":"))
    total = hh * 3600 + mm * 60 + ss + seconds
    total = total % 86400  # wrap at midnight (DAY boundary unlikely within a 30s clip)
    hh2 = int(total // 3600)
    mm2 = int((total % 3600) // 60)
    ss2 = total % 60
    return f"{hh2:02d}:{mm2:02d}:{int(ss2):02d}"


def _n_pieces_for_duration(dur: float, target_s: int) -> int:
    """How many pieces to split a clip of `dur` seconds into, targeting `target_s` each.

    Rule (matches the user's spec): anything strictly longer than the target gets split.
    A 30.04s source at target=10 yields 3 pieces (not 4) — the extra 0.04s is just encoder
    padding. So: 8s→1, 10s→1, 11s→2, 15s→2, 22s→3, 30s→3 (at target=10).

    Implementation: clamp the effective duration to an exact multiple when within 0.5s of
    one (kills the 30.04→4 problem), then ceil so anything over a boundary rounds UP.
    """
    # Snap dur to the nearest exact multiple of target_s when within 0.5s (encoder padding).
    snapped = round(dur / target_s) * target_s
    if abs(snapped - dur) <= 0.5:
        dur = float(snapped)
    if dur <= target_s:
        return 1
    return max(2, math.ceil(dur / target_s - 1e-9))


def split_source_into_slices(src: Path, clip_id: str, base_hms: str, target_s: int,
                             out_dir: Path, *, resolution: int, fps: int, crf: int,
                             audio_k: int, skip_if_exists: bool = True) -> list[SliceInfo]:
    """Re-encode `src` into ceil(dur/target_s) evenly-sized pieces.

    - target_s=30 (or any target >= dur): one whole-clip re-encode, kind stays 30s/segment_open.
    - target_s=10, dur=30: 3 pieces of ~10s each, kind "10s".
    - dur=22, target_s=10: 3 pieces of ~7.33s (ceil(22/10)=3, evenly divided).
    - dur=8,  target_s=10: 1 piece (whole clip, not split — 8s ≤ target).

    Uses accurate seek (-ss after -i, -t) since we re-encode anyway; cuts are frame-accurate.
    Returns one SliceInfo per piece, in playback order.
    """
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
        if skip_if_exists and out_path.exists() and out_path.stat().st_size > 0:
            pass  # keep existing encode
        else:
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
        actual_dur = ffprobe_duration(out_path)
        pieces.append(SliceInfo(
            path=out_path, duration=actual_dur, piece_idx=i + 1,
            start_hms=_add_seconds_to_hms(base_hms, start_off), is_whole=is_whole,
        ))
    return pieces


# ===========================================================================
# JSON parsing & output classification
# ===========================================================================

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)

_ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "environment": {"type": "object", "properties": {
            "setting": {"type": "string"},
            "near_field": {"type": "array"},
            "background": {"type": "string"}},
        },
        "people": {"type": "array", "items": {"type": "object"}},
        "self_actions": {"type": "array", "items": {"type": "object"}},
        "other_actions": {"type": "array", "items": {"type": "object"}},
        "env_changes": {"type": "array", "items": {"type": "object"}},
        "speech": {"type": "array", "items": {"type": "object"}},
        "sound": {"type": "array", "items": {"type": "object"}},
        "interface": {"type": "array", "items": {"type": "object"}},
        "psychology": {
            "type": "object",
            "properties": {
                "emotion": {"type": "string"},
                "mental_activity": {"type": "string"},
            },
            "required": ["emotion"],
        },
        # causal_links items are normalized leniently in _norm_causal (invalid/missing
        # strength -> ""), so a stray value on one edge never rejects the whole response.
        "causal_links": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["self_actions", "psychology"],
}


class ParseFailed(Exception):
    def __init__(self, reason: str, raw: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.raw = raw


_CAUSAL_TYPES = ("env->action", "env->emotion", "emotion->action", "action->env",
                 "other->action", "other->env", "other->emotion", "env->env")
# Edge strength grades each causal link counterfactually (weak/moderate/strong).
_STRENGTH_LEVELS = ("strong", "moderate", "weak")


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
        if not (isinstance(psych, dict) and psych.get("emotion")):
            raise ParseFailed("psychology.emotion missing or invalid", content)
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

    def _norm_person(p):
        if not isinstance(p, dict):
            return {"id": "", "descriptor": str(p), "name": "", "note": ""}
        return {"id": str(p.get("id", "") or ""), "descriptor": str(p.get("descriptor", "") or ""),
                "name": str(p.get("name", "") or ""), "note": str(p.get("note", "") or "")}

    def _norm_other(a):
        if not isinstance(a, dict):
            return {"time": "", "time_end": "", "person": "", "text": str(a)}
        return {"time": str(a.get("time", "") or ""), "time_end": str(a.get("time_end", "") or ""),
                "person": str(a.get("person", "") or ""), "text": str(a.get("text", "") or "")}

    def _norm_sound(s):
        if not isinstance(s, dict):
            return {"time": "", "text": str(s), "source": ""}
        return {"time": str(s.get("time", "") or ""), "text": str(s.get("text", "") or ""),
                "source": str(s.get("source", "") or "")}

    def _norm_interface(o):
        if not isinstance(o, dict):
            return {"time": "", "where": "", "app_or_site": "", "content": str(o), "note": ""}
        return {"time": str(o.get("time", "") or ""), "where": str(o.get("where", "") or ""),
                "app_or_site": str(o.get("app_or_site", "") or ""),
                "content": str(o.get("content", "") or o.get("text", "") or ""),
                "note": str(o.get("note", "") or "")}

    def _norm_env_change(e):
        if not isinstance(e, dict):
            return {"time": "", "time_end": "", "text": str(e), "cause": ""}
        return {"time": str(e.get("time", "") or ""), "time_end": str(e.get("time_end", "") or ""),
                "text": str(e.get("text", "") or ""), "cause": str(e.get("cause", "") or "")}

    def _norm_causal(c):
        if not isinstance(c, dict):
            return {"type": "", "cause": str(c), "effect": "", "strength": ""}
        strength = str(c.get("strength", "") or "").strip().lower()
        if strength not in _STRENGTH_LEVELS:
            strength = ""
        return {"type": str(c.get("type", "") or ""), "cause": str(c.get("cause", "") or ""),
                "effect": str(c.get("effect", "") or ""), "strength": strength}

    psych = data.get("psychology") or {}
    env = data.get("environment") or {}
    if isinstance(env, dict):
        environment = {"setting": str(env.get("setting", "") or ""),
                       "near_field": [str(x) for x in (env.get("near_field") or []) if str(x).strip()],
                       "background": str(env.get("background", "") or "")}
    else:  # unreachable under jsonschema; keeps the no-jsonschema path safe
        environment = {"setting": "", "near_field": [], "background": ""}
    return {
        "environment": environment,
        "people": [_norm_person(p) for p in data.get("people", []) or []],
        "self_actions": [_norm_action(a) for a in data.get("self_actions", []) or []],
        "other_actions": [_norm_other(a) for a in data.get("other_actions", []) or []],
        "env_changes": [_norm_env_change(e) for e in data.get("env_changes", []) or []],
        "speech": [_norm_speech(s) for s in data.get("speech", []) or []],
        "sound": [_norm_sound(s) for s in data.get("sound", []) or []],
        "interface": [_norm_interface(o) for o in data.get("interface", []) or []],
        "psychology": {"emotion": str(psych.get("emotion", "") or ""),
                       "mental_activity": str(psych.get("mental_activity", "") or "")},
        "causal_links": [_norm_causal(c) for c in data.get("causal_links", []) or []],
    }


# Outcome categories for a model response.
OUTCOME_OK = "ok"
OUTCOME_SAFETY = "safety_rejection"
OUTCOME_PARSE_FAILED = "parse_failed"
OUTCOME_EMPTY = "empty"


def classify_output(content: str, out_tokens: int) -> str:
    """Cheap pre-classification (does not parse). The actual JSON parse is
    attempted in build_records_from_response."""
    if not content or not content.strip():
        return OUTCOME_EMPTY
    if out_tokens <= 25:
        # JSON-mode output must start with the object (a fence is tolerated —
        # the parser strips it). Anything else that short is a refusal.
        if not content.strip().startswith(("{", "```")):
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
    model: str
    ts_captioned: str
    slice_path: str
    tokens: dict
    self_actions: list = None    # [{time, time_end, text}] — atomic, verb-level steps
    other_actions: list = None   # [{time, time_end, person, text}] — person = people.id
    people: list = None          # [{id, descriptor, name, note}] — cast list, never self
    environment: dict = None     # {setting, near_field, background} — scene to spawn the
                                 # agent into, not a changelog
    env_changes: list = None     # [{time, time_end, text, cause}] — atomic state transitions
    speech: list = None          # [{lang, speaker, text}]
    sound: list = None           # [{time, text, source}] — discrete non-speech sounds
    interface: list = None       # [{time, where, app_or_site, content, note}] — screens /
                                 # legible text surfaces (page understanding, not raw OCR)
    psychology: dict = None      # {emotion, mental_activity} — the ONLY field intent may
                                 # appear in; estimated from this clip alone
    causal_links: list = None    # [{type, cause, effect, strength}] — strength-graded directed
                                 # env<->action<->emotion/env edges
    clip_kind: str = "30s"       # "30s" | "segment_open" | "10s" | "15s" | "6s" | "5s"
    recovery: str = ""           # "ok" | "10s_slices"


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
    # start_hms comes from the row when available (slice-aware), else from the source filename.
    start_hms = clip_row.get("start_hms") or f"{ts_raw[0:2]}:{ts_raw[2:4]}:{ts_raw[4:6]}"
    clip_kind = clip_row.get("clip_kind", "30s")
    target_s = int(clip_row.get("target_s", 30))
    # Duration descriptor: "~Ns-second" using the actual target (10/15/30/5/6).
    # is_10s (recovery slice) overrides to the legacy "10-second" wording for back-compat.
    if is_10s:
        duration_desc = "10-second"
    else:
        duration_desc = f"~{target_s}-second"
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
    except ParseFailed:
        return None, UsageRecord(clip_id, gid, attempt, round(latency, 3),
                                 OUTCOME_PARSE_FAILED, usage, content)

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cap_rec = CaptionRecord(
        clip_id=clip_id, global_idx=gid, day=int(clip_row["day"]), user=clip_row["user"],
        duration_s=float(clip_row["duration"]), narrative=content.strip(),
        model=model, ts_captioned=ts, slice_path=clip_row["slice_path"],
        tokens={"in": usage["prompt_tokens"], "out": usage["completion_tokens"], "cached": usage["cached_tokens"]},
        self_actions=layered["self_actions"], other_actions=layered["other_actions"],
        people=layered["people"], environment=layered["environment"],
        env_changes=layered["env_changes"], speech=layered["speech"],
        sound=layered["sound"], interface=layered["interface"],
        psychology=layered["psychology"], causal_links=layered["causal_links"],
        clip_kind=clip_row.get("clip_kind", "30s"), recovery="ok",
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
# Source selection & clip grouping
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
    """A preprocess unit: one source file, to be split into ceil(dur/target_s) pieces.
    The 60s-merge mode has been removed; every unit is single-source."""
    __slots__ = ("kind", "srcs", "clip_id", "start_ts", "target_s")

    def __init__(self, kind, srcs, clip_id, start_ts, target_s):
        self.kind = kind        # "30s" | "segment_open"  (pre-split; pieces get "10s"/"15s"/etc.)
        self.srcs = srcs        # always len==1 now
        self.clip_id = clip_id
        self.start_ts = start_ts
        self.target_s = target_s


def group_into_clips(files, participant, day, target_s):
    """Group source files into single-source ClipUnits. No merging (60s mode removed).
    Each unit carries target_s so _preprocess_worker knows how many pieces to slice into."""
    units = []
    for i, f in enumerate(files):
        ts_raw = Path(f.stem).stem.split("_")[-1]
        ss_field = ts_raw[4:6]
        kind = "segment_open" if ss_field not in ("00", "30") else "30s"
        units.append(ClipUnit(kind, [f], f"DAY{day}_{participant}_{ts_raw}",
                              _ts_label(f.name), target_s))
    return units


def count_pieces_for_units(units, log=None) -> list[int]:
    """Pre-scan durations via ffprobe to compute how many output pieces each unit yields.
    Returns a parallel list; sum of it = expected_total.
    One ffprobe per source (~tens of ms); acceptable even for ~1000 files."""
    out = []
    for u in units:
        try:
            dur = ffprobe_duration(u.srcs[0])
            n = _n_pieces_for_duration(dur, u.target_s)
        except Exception as e:
            if log:
                log.warning(f"[count] ffprobe failed for {u.srcs[0].name}: {e}; assuming 1 piece")
            n = 1
        out.append(n)
    return out


# ===========================================================================
# Workers (producer-consumer)
# ===========================================================================

class ApiCallConfig:
    __slots__ = ("thinking", "json_mode")

    def __init__(self, thinking, json_mode):
        self.thinking = thinking
        self.json_mode = json_mode


class PreprocessJob:
    __slots__ = ("unit", "idx_base", "is_first_unit", "day", "user", "skip_if_exists",
                 "resolution", "fps", "crf", "audio_k", "slices_dir", "out_dir")

    def __init__(self, unit, idx_base, *, is_first_unit, day, user, skip_if_exists,
                 resolution, fps, crf, audio_k, slices_dir, out_dir):
        self.unit = unit
        self.idx_base = idx_base        # global_idx of this unit's FIRST piece (1-based)
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
        src = unit.srcs[0]
        try:
            # Split (or whole-encode) the source into ceil(dur/target_s) pieces.
            # split_source_into_slices re-encodes each piece at the configured quality.
            pieces = split_source_into_slices(
                src, base_clip_id, unit.start_ts, unit.target_s, job.slices_dir,
                resolution=job.resolution, fps=job.fps, crf=job.crf, audio_k=job.audio_k,
                skip_if_exists=job.skip_if_exists,
            )
            # If only one piece came back (short clip or target_s>=dur), keep the
            # unit's original kind (30s / segment_open); otherwise mark pieces as
            # "{target_s}s" so the prompt and recovery path know the granularity.
            for piece in pieces:
                clip_kind = unit.kind if piece.is_whole else f"{unit.target_s}s"
                piece_clip_id = base_clip_id if piece.is_whole else f"{base_clip_id}_p{piece.piece_idx}"
                gid = job.idx_base + piece.piece_idx - 1
                is_day_open = job.is_first_unit and piece.piece_idx == 1 and unit.kind == "segment_open"
                row = pd.Series({
                    "clip_id": piece_clip_id, "day": job.day, "user": job.user,
                    "start_ts": piece.start_hms,
                    "end_ts": f"{piece.start_hms}+{piece.duration:.3f}s",
                    "src_file": src.name, "clip_idx": piece.piece_idx,
                    "global_idx": gid, "duration": round(piece.duration, 3),
                    "is_day_open": is_day_open, "is_day_close": False, "status": "pending",
                    "slice_path": str(piece.path.relative_to(job.out_dir)),
                    "clip_kind": clip_kind,
                    "target_s": (30 if piece.is_whole else unit.target_s),
                    "start_hms": piece.start_hms,
                })
                out_q.put(PreprocessResult(clip_id=piece_clip_id, global_idx=gid,
                                           slice_path=piece.path, row=row))
        except Exception as e:
            log.error(f"[preprocess] {base_clip_id} failed: {type(e).__name__}: {e}")
            out_q.put(PreprocessResult(clip_id=base_clip_id, global_idx=job.idx_base,
                                       slice_path=job.slices_dir / f"{base_clip_id}.mp4", row=None,
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
                # Sub-30 clips (first-class 10s/15s/5s/6s pieces) cannot be sliced
                # further, so give up after the single retry. Only legacy 30s /
                # segment_open clips (from --clip-duration 30) fall back to 10s slicing.
                clip_kind = row.get("clip_kind", "30s")
                if clip_kind not in ("30s", "segment_open"):
                    log.warning(f"[{clip_id}] second attempt also failed ({reason2}); "
                                f"giving up ({clip_kind} clip, no further slicing)")
                    failures.append({"clip_id": clip_id, "global_idx": gid,
                                     "error": "all_attempts_failed", "reason": reason2,
                                     "raw_content_preview": (content or "")[:200]})
                else:
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
    ap.add_argument("--clip-duration", type=int, default=10, choices=[5, 6, 10, 15, 30],
                    help="split target in seconds. Each source file is sliced into "
                         "ceil(dur/target) pieces (e.g. 30s->3x10s). Must divide 30 evenly. "
                         "30 = no splitting (one caption per source file, legacy behaviour).")
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
                    help="reasoning mode (default: enabled — causal-graph annotation is "
                         "reasoning-heavy). Use --thinking disabled "
                         "for ~4x cheaper/faster runs (causal_links quality drops). "
                         "Under thinking Mimo ignores temperature/top_p.")
    ap.add_argument("--no-json-mode", action="store_true",
                    help="disable response_format=json_object (debug)")
    # --- Concurrency ---
    ap.add_argument("--max-rpm", type=int, default=90, help="global RPM cap (Mimo limit is 100)")
    ap.add_argument("--api-workers", type=int, default=None,
                    help="API worker threads (default: min(--max-rpm/2, 20) — thinking-mode "
                         "requests take ~30s+ each, so ~2 requests/min per worker; scales with "
                         "--max-rpm but caps at 20 to keep one machine's connections modest)")
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

    # Thinking-mode requests run ~30s+; scale workers with the RPM cap but cap at
    # 20 so a single machine doesn't open excessive connections.
    if args.api_workers is None:
        args.api_workers = min(args.max_rpm // 2, 20)

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
    cfg = ApiCallConfig(args.thinking, json_mode)
    log.info(f"participant={args.participant} day={args.day} "
             f"time={args.start_time or '00:00'}-{args.end_time or '23:59'} "
             f"clip_duration={args.clip_duration}s src_root={src_root} "
             f"-> {src_root / args.participant / f'DAY{args.day}'}")
    log.info(f"model={args.model} thinking={args.thinking} json_mode={json_mode}")
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
        units = group_into_clips(files, args.participant, args.day, args.clip_duration)
        # Pre-scan durations to compute how many pieces each unit yields; assign each
        # unit a contiguous global_idx block so pieces come out in deterministic order.
        piece_counts = count_pieces_for_units(units, log=log)
        idx_bases = []
        running = 1  # global_idx is 1-based
        for n in piece_counts:
            idx_bases.append(running)
            running += n
        expected_total_pre = sum(piece_counts)
        n_split = sum(1 for u in units if u.target_s < 30)
        log.info(f"units: {len(units)} source files -> {expected_total_pre} pieces "
                 f"(target={args.clip_duration}s, {n_split} will be split) "
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
                      OUTCOME_EMPTY: 0, "10s_slices": 0,
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
        limit_n = args.limit if args.limit else len(units)
        # Recompute piece counts / idx_bases over the (possibly limited) slice, so
        # global_idx assignment stays contiguous and matches expected_total.
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
        "temperature": (1.0 if args.thinking == "disabled" else None),
        "max_rpm": args.max_rpm, "api_workers": args.api_workers,
        "preprocess_workers": args.preprocess_workers,
        "skip_preprocess": args.skip_preprocess, "skip_existing": args.skip_existing,
        "n_clips_input": expected_total, "n_clips_ok": n_ok, "n_clips_failed": len(all_failures),
        "n_skipped_existing": skipped_existing, "outcomes": dict(outcome_counts),
        "n_safety_rejection": outcome_counts[OUTCOME_SAFETY],
        "n_parse_failed": outcome_counts[OUTCOME_PARSE_FAILED],
        "n_empty": outcome_counts[OUTCOME_EMPTY],
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
