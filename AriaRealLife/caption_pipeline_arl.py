"""caption_pipeline_arl.py — Mimo captioning pipeline for AriaRealLife/test.

Adapts the EgoLife caption_pipeline to the AriaRealLife sample format:
  - 300 masked RGB frames @ 1fps (frame_00000.jpg .. frame_00299.jpg)
  - one continuous 300s voice-anonymized WAV (audio/anonymized.wav)

The sample is 60 INDEPENDENT 5-second scenes (5 frames + 5s audio each).
Scenes have no temporal continuity with each other.

Each scene is fed to MiMo as one user message: 5 image_url blocks (1024x1024
JPEGs, downsampled from 2880x2880) + 1 input_audio block (5s WAV slice) + a
text task. Output is the same 5-layer structured JSON as EgoLife
(self_actions / others / environment / speech / psychology / tags).

Reuses EgoLife.caption_pipeline verbatim for: JSON parsing, schema, rate
limiter, ordered writer, record dataclasses, API helpers, logging. Only the
system prompt, message construction, preprocessing, and scene enumeration are
ARL-specific.

Usage
-----
  # probe: run scene 0 only, to verify the 10-image + audio combo is accepted
  python AriaRealLife/caption_pipeline_arl.py --probe
  # full run (30 scenes)
  python AriaRealLife/caption_pipeline_arl.py --max-rpm 90
  # resume after an interruption
  python AriaRealLife/caption_pipeline_arl.py --skip-existing
  # caption only (slices already encoded in _cache/)
  python AriaRealLife/caption_pipeline_arl.py --skip-preprocess
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Make the sibling EgoLife package importable without touching sys.path hacks
# that could shadow stdlib. caption_pipeline is a single module.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO / "EgoLife") not in sys.path:
    sys.path.insert(0, str(_REPO / "EgoLife"))

# ---- reuse EgoLife components verbatim -------------------------------------
from caption_pipeline import (  # noqa: E402
    ApiCallConfig,
    ApiJob,
    ApiResult,
    CaptionRecord,
    UsageRecord,
    OUTCOME_OK,
    OUTCOME_SAFETY,
    OUTCOME_PARSE_FAILED,
    OUTCOME_EMPTY,
    SlidingWindowRateLimiter,
    OrderedWriter,
    TqdmLoggingHandler,
    extract_usage,
    _call_api_limited,
    build_records_from_response,
    load_existing_clip_ids,
    max_written_global_idx,
)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ROOT = _HERE  # .../ARIN7600/AriaRealLife/

# ===========================================================================
# Constants
# ===========================================================================

DEFAULT_RESOLUTION = 1024      # downsample 2880x2880 -> 1024x1024 (matches EgoLife)
DEFAULT_JPEG_Q = 85
# Each scene is N consecutive frames @ 1fps = N seconds. Verified by MiMo
# (see captions/_test/scene_verification.md) that the true scene unit is 5s:
# the OCR JSON's VRS source is named "combined_5min_60x5s", and 27/30 of the
# old 10-frame blocks contained a cut at position 4 or 5.
SCENE_FRAME_COUNT = 5          # 5 frames @ 1fps = 5s
SCENE_DURATION_S = 5.0
AUDIO_SAMPLE_RATE = 48000
# Cache subdir names are tagged with the scene frame count so that re-runs with
# a different SCENE_FRAME_COUNT never read a stale, differently-grouped cache.
IMG_CACHE_NAME = f"img1024_f{SCENE_FRAME_COUNT}"
AUDIO_CACHE_NAME = f"audio{int(SCENE_DURATION_S)}"

MAX_COMPLETION_TOKENS = 1024
THINKING_DEFAULT = "disabled"

# ===========================================================================
# System prompt (ARL-specific)
# ===========================================================================

SYSTEM_MSG_ARL = """You are a dense first-person life-log captioner.
The footage is from a person wearing Project Aria glasses. People may speak Chinese or English.

# Input layout
Each request is ONE independent 5-second scene. The scene contains 5 RGB frames sampled at 1fps
(frame 0 = scene second 00:00, frame 4 = scene second 00:04) plus a 5-second audio clip recorded
during the same interval. Scenes are INDEPENDENT: do not assume continuity with any other scene,
and never reference events outside this 5-second window.

# Audio caveat (CRITICAL)
The audio has been VOICE-ANONYMIZED (pitch/timbre shifted). Therefore:
- The spoken CONTENT (what was said) and the LANGUAGE (Chinese/English) are reliable.
- The timbre, pitch, gender, age, and speaker identity are NOT reliable. Do NOT infer a speaker's
  gender, age, identity, or emotional intensity from voice quality. Do NOT use voice pitch as a cue
  for who is speaking. Identify speakers only by what is visible on camera (e.g. "a woman in blue"),
  never by voice.

# Timestamps (no watermark)
There is no watermark. Timestamp all self_actions and others using the scene-relative clock: the
first frame is 00:00, the last is 00:04. Each "time"/"time_end" MUST be an "MM:SS" string in
{00:00, 00:01, 00:02, 00:03, 00:04} — never 00:05 or higher (the scene ends at 00:04).
time <= time_end, and time_end <= 00:04.

# Output format
Return ONLY a single JSON object. No explanations, no markdown code fences, no text outside JSON.
Use exactly this structure (fill every field; use empty arrays/strings when a section is truly empty):

{
  "self_actions": [
    {"time": "00:02", "time_end": "00:05", "text": "..."}
  ],
  "others": [
    {"time": "00:02", "time_end": "00:05", "text": "..."}
  ],
  "environment": "...",
  "speech": [
    {"lang": "zh", "speaker": "...", "text": "..."}
  ],
  "psychology": {"awareness": "low", "emotion": "...", "note": "..."},
  "tags": ["..."]
}

## Field rules

self_actions: 1-4 objects, one per self-action, in first person ("I"), present tense, verb-led.
Because the scene is short (5s) use a FINE granularity: break continuous behavior into small action
units of ~1-2 seconds each (e.g. reach -> grasp -> lift). Cover the whole 5s span when there is
enough motion; a single action object is fine if the scene is essentially static. Describe hand-object
contact, posture, gaze direction, locomotion, device/phone use.
Each "time"/"time_end" is a scene-relative MM:SS (~1s resolution). Use Chinese when the scene is
Chinese-speaking, English otherwise. Never invent a name for yourself; use neutral descriptors for
others unless their name is shown or spoken.

others: zero or more objects, same shape, timestamped the same way. Lead text with the person
descriptor ("a woman in blue", "a child"). Empty array if you are alone.

environment: one or two sentences on the setting and any CHANGES during the 5s: room, lighting,
screen content, object layout, weather/outdoor cues, location. Empty string if perfectly static.

speech: zero or more objects. lang in {"zh","en"}; speaker is a NEUTRAL visual descriptor only (never
inferred from voice); text is the quoted utterance. Transcribe ONLY what is actually heard; do not
invent. Speech is not timestamped. Empty array if silent. If the audio is non-speech (music, noise),
describe it in environment or tags instead.

psychology: an object with exactly:
  awareness: "low" | "medium" | "high"  (required)
  emotion: one or two lowercase keywords (required; "neutral" if none readable)
  note: optional short sentence
"awareness" measures how strongly emotion CAUSALLY drove visible behavior in this 5s scene:
  low    = emotion was background color; behavior was driven by habit or task, not by feeling.
  medium = emotion colored HOW an action was performed (tone, pace, expression, vigor) but did not
           change its direction.
  high   = emotion directly triggered a behavior or a turn/pivot (e.g. surprise -> turning head).
"emotion" picks from: neutral calm relaxed bored focused amused happy excited surprised confused
curious anxious nervous stressed frustrated angry embarrassed sad tired sleepy hungry, or similar.

tags: 5-10 short lowercase keywords (objects, actions, location, people descriptors). MANDATORY,
never empty.

# Rules
- Stay strictly within this 5-second scene; never describe other scenes or invent off-screen events.
- Be concrete and observational; do not speculate beyond what is visible or audible.
- Pick ONE language for all fields based on the dominant spoken language in the scene (Chinese if
  participants speak Chinese, English otherwise). Do NOT translate or duplicate content.
- Output each action, utterance, or fact exactly ONCE; never repeat in another language.
- Output must be valid JSON: double quotes, no trailing commas, no comments."""

USER_TASK_TMPL_ARL = (
    "Annotate this 5-second first-person scene ({clip_id}) and return the JSON object.\n"
    "There are 5 frames at 1fps (frame 0 = scene second 00:00, frame 4 = 00:04) and one 5-second "
    "audio clip. The audio is voice-anonymized: transcribe what is said but do NOT infer speaker "
    "identity, gender, or emotion from the voice quality.\n"
    "Timestamp self_actions and others with the scene-relative MM:SS clock (00:00-00:04)."
)


# ===========================================================================
# ffmpeg helpers (ARL: image downsample + audio slice)
# ===========================================================================

def ffmpeg_downsample_image(src: Path, out: Path, resolution: int, jpeg_q: int) -> None:
    """Downsample one JPEG to resolution x resolution (lanczos). Source is RGB,
    so we square-pad/crop is NOT applied — we just resize to the target square
    to match EgoLife's `scale=W:H:flags=lanczos` (sources are already square 2880x2880)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-vf", f"scale={resolution}:{resolution}:flags=lanczos",
        "-q:v", str(jpeg_q),
        str(out),
    ]
    rc = subprocess.run(cmd, capture_output=True, text=True)
    if rc.returncode != 0:
        raise RuntimeError(f"ffmpeg image downsample failed for {src.name}: {rc.stderr.strip()[:300]}")


def ffmpeg_slice_audio(src: Path, out: Path, start_s: float, duration_s: float,
                       sample_rate: int) -> None:
    """Slice [start_s, start_s+duration_s) out of the source WAV into a 5s mono clip."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_s:.3f}", "-i", str(src), "-t", f"{duration_s:.3f}",
        "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(out),
    ]
    rc = subprocess.run(cmd, capture_output=True, text=True)
    if rc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio slice failed: {rc.stderr.strip()[:300]}")


# ===========================================================================
# Scene enumeration
# ===========================================================================

def enumerate_scenes(data_dir: Path, limit: int | None = None) -> list[dict]:
    """Build one row per 5s scene. Returns a list of clip-row dicts (same shape
    EgoLife expects, adapted for ARL: no day/user, scene-relative clock)."""
    img_dir = data_dir / "images_masked"
    audio_path = data_dir / "audio" / "anonymized.wav"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"images_masked/ not found under {data_dir}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio/anonymized.wav not found under {data_dir}")

    all_imgs = sorted(p for p in img_dir.glob("frame_*.jpg")
                      if not p.name.startswith("._"))
    if not all_imgs:
        raise FileNotFoundError(f"no frame_*.jpg in {img_dir}")
    n_scenes = len(all_imgs) // SCENE_FRAME_COUNT
    if n_scenes * SCENE_FRAME_COUNT != len(all_imgs):
        print(f"WARN: {len(all_imgs)} frames is not a multiple of {SCENE_FRAME_COUNT}; "
              f"using first {n_scenes * SCENE_FRAME_COUNT} ({n_scenes} scenes).", file=sys.stderr)

    scenes = []
    for i in range(n_scenes):
        frame_paths = all_imgs[i * SCENE_FRAME_COUNT:(i + 1) * SCENE_FRAME_COUNT]
        start_s = i * SCENE_DURATION_S
        mm = int(start_s // 60)
        ss = int(start_s % 60)
        scenes.append({
            "scene_idx": i,
            "clip_id": f"ARL_scene_{i:02d}",
            "global_idx": i + 1,
            "day": 0,                       # ARL has no day; 0 placeholder for CaptionRecord
            "user": "ARL",                  # ARL participant tag for CaptionRecord
            "src_file": f"frame_{frame_paths[0].stem.split('_')[1]}..{frame_paths[-1].stem.split('_')[1]}",
            "start_ts": f"{mm:02d}:{ss:02d}:00",   # wall-clock-ish label (not used for ts)
            "start_offset_s": start_s,
            "duration": SCENE_DURATION_S,
            "frame_paths": [str(p) for p in frame_paths],
            "frame_first": frame_paths[0].name,
            "frame_last": frame_paths[-1].name,
            "audio_start_s": start_s,
            "audio_path": str(audio_path),
            "clip_kind": f"{int(SCENE_DURATION_S)}s",
            "is_day_open": (i == 0),
            "is_day_close": (i == n_scenes - 1),
            "status": "pending",
        })
    if limit is not None:
        scenes = scenes[:limit]
    return scenes


# ===========================================================================
# Message construction (10 images + 1 audio + text)
# ===========================================================================

def b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def make_arl_user_message(scene_row: dict, img_dir: Path, audio_path: Path,
                          drop_audio: bool = False) -> dict:
    """Build the user message: 10 image_url blocks + 1 input_audio block + text.

    drop_audio=True omits the audio block (used as a fallback when the combined
    payload is rejected by the API)."""
    clip_id = scene_row["clip_id"]
    content = []
    scene_idx = scene_row["scene_idx"]
    for fp in scene_row["frame_paths"]:
        # frames live under img_dir (the downsampled cache dir); names like frame_00000.jpg
        p = img_dir / Path(fp).name
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_file(p)}"},
        })
    if not drop_audio and audio_path is not None and audio_path.exists():
        content.append({
            "type": "input_audio",
            "input_audio": {"data": f"data:audio/wav;base64,{b64_file(audio_path)}"},
        })
    task = USER_TASK_TMPL_ARL.format(clip_id=clip_id)
    content.append({"type": "text", "text": task})
    return {"role": "user", "content": content}


# ===========================================================================
# Preprocess worker pool (downsample images + slice audio per scene)
# ===========================================================================

class ARLPreprocessJob:
    __slots__ = ("scene_row", "skip_if_exists", "resolution", "jpeg_q",
                 "sample_rate", "img_out_dir", "audio_out_dir")

    def __init__(self, scene_row, *, skip_if_exists, resolution, jpeg_q,
                 sample_rate, img_out_dir, audio_out_dir):
        self.scene_row = scene_row
        self.skip_if_exists = skip_if_exists
        self.resolution = resolution
        self.jpeg_q = jpeg_q
        self.sample_rate = sample_rate
        self.img_out_dir = img_out_dir
        self.audio_out_dir = audio_out_dir


class ARLPreprocessResult:
    __slots__ = ("clip_id", "global_idx", "scene_row", "img_out_dir", "audio_out_path", "error")

    def __init__(self, *, clip_id, global_idx, scene_row, img_out_dir, audio_out_path, error=""):
        self.clip_id = clip_id
        self.global_idx = global_idx
        self.scene_row = scene_row
        self.img_out_dir = img_out_dir
        self.audio_out_path = audio_out_path
        self.error = error


def _arl_preprocess_worker(job_q, out_q, log):
    while True:
        job = job_q.get()
        if job is None:
            job_q.task_done()
            return
        row = job.scene_row
        clip_id = row["clip_id"]
        scene_idx = row["scene_idx"]
        try:
            # Per-scene image output dir: img_out_dir/scene_XX/frame_*.jpg
            scene_img_dir = job.img_out_dir / f"scene_{scene_idx:02d}"
            audio_out = job.audio_out_dir / f"scene_{scene_idx:02d}.wav"

            all_imgs_exist = all((scene_img_dir / Path(fp).name).exists()
                                 for fp in row["frame_paths"])
            audio_ok = audio_out.exists() and audio_out.stat().st_size > 0

            if job.skip_if_exists and all_imgs_exist and audio_ok:
                log.debug(f"[preprocess] {clip_id} cached, skipping")
            else:
                scene_img_dir.mkdir(parents=True, exist_ok=True)
                for fp in row["frame_paths"]:
                    out_p = scene_img_dir / Path(fp).name
                    if job.skip_if_exists and out_p.exists() and out_p.stat().st_size > 0:
                        continue
                    ffmpeg_downsample_image(Path(fp), out_p, job.resolution, job.jpeg_q)
                if not (job.skip_if_exists and audio_ok):
                    ffmpeg_slice_audio(Path(row["audio_path"]), audio_out,
                                       float(row["audio_start_s"]),
                                       float(row["duration"]), job.sample_rate)
            out_q.put(ARLPreprocessResult(
                clip_id=clip_id, global_idx=row["global_idx"], scene_row=row,
                img_out_dir=scene_img_dir, audio_out_path=audio_out))
        except Exception as e:
            log.error(f"[preprocess] {clip_id} failed: {type(e).__name__}: {e}")
            out_q.put(ARLPreprocessResult(
                clip_id=clip_id, global_idx=row["global_idx"], scene_row=row,
                img_out_dir=None, audio_out_path=None,
                error=f"{type(e).__name__}: {e}"))
        finally:
            job_q.task_done()


# ===========================================================================
# API worker (ARL: builds message from scene_row + cached imgs/audio)
# ===========================================================================

def _arl_process_one(client, model, scene_row, img_dir, audio_path,
                     drop_audio, log, limiter, cfg):
    # build_records_from_response (reused from EgoLife) reads slice_path /
    # clip_kind / day / user off the row. Inject the cache paths into a copy so
    # the original scene_row stays clean for parquet persistence.
    row = dict(scene_row)
    row["slice_path"] = str(img_dir)
    messages = [{"role": "system", "content": SYSTEM_MSG_ARL},
                make_arl_user_message(scene_row, img_dir, audio_path, drop_audio=drop_audio)]
    resp, latency, attempt = _call_api_limited(
        client, model, messages, log, scene_row["clip_id"], limiter, cfg)
    usage = extract_usage(resp)
    content = resp.choices[0].message.content or ""
    cap_rec, use_rec = build_records_from_response(
        content, usage, latency, attempt, row, model)
    return cap_rec, use_rec, content


def _arl_api_worker(in_q, out_q, client, model, limiter, cfg, log, worker_id):
    while True:
        job = in_q.get()
        if job is None:
            in_q.task_done()
            return
        row, img_dir, audio_path = job.row, job.img_dir, job.audio_path
        clip_id, gid = row["clip_id"], int(row["global_idx"])
        caption_records, usage_records, failures = [], [], []

        # sanity: all 10 cached images present?
        missing = [Path(fp).name for fp in row["frame_paths"]
                   if not (img_dir / Path(fp).name).exists()]
        if missing:
            log.error(f"[{clip_id}] missing cached images: {missing}")
            failures.append({"clip_id": clip_id, "global_idx": gid,
                             "error": f"missing_images: {missing}"})
            out_q.put(ApiResult(global_idx=gid, clip_id=clip_id, caption_records=[],
                                usage_records=[], failures=failures))
            in_q.task_done()
            continue

        try:
            cap_rec, use_rec, content = _arl_process_one(
                client, model, row, img_dir, audio_path, drop_audio=False, log=log,
                limiter=limiter, cfg=cfg)
            use_rec2 = None
            content2 = ""
            if cap_rec is None:
                reason = use_rec.recovery
                use_rec.recovery = "first_attempt_rejection"
                usage_records.append(use_rec)
                log.info(f"[{clip_id}] first attempt '{reason}'; retrying with audio (cheap via cache)")
                try:
                    cap_rec, use_rec2, content2 = _arl_process_one(
                        client, model, row, img_dir, audio_path, drop_audio=False,
                        log=log, limiter=limiter, cfg=cfg)
                except Exception as e:
                    log.warning(f"[{clip_id}] retry crashed: {type(e).__name__}: {e}")
                    cap_rec, use_rec2 = None, None

            if cap_rec is None:
                reason2 = use_rec2.recovery if use_rec2 is not None else "retry_failed"
                if use_rec2 is not None:
                    usage_records.append(use_rec2)
                # Fallback: drop the audio, retry with images only (audio may be
                # the trigger for safety/size rejection on this short scene).
                log.warning(f"[{clip_id}] retry also failed ({reason2}); "
                            f"trying images-only (no audio)")
                try:
                    cap_rec3, use_rec3, _ = _arl_process_one(
                        client, model, row, img_dir, None, drop_audio=True,
                        log=log, limiter=limiter, cfg=cfg)
                except Exception as e:
                    log.warning(f"[{clip_id}] images-only retry crashed: {type(e).__name__}: {e}")
                    cap_rec3, use_rec3 = None, None
                if cap_rec3 is not None:
                    cap_rec3.recovery = "images_only_fallback"
                    caption_records.append(cap_rec3)
                    usage_records.append(use_rec3)
                    log.info(f"[{clip_id}] images-only fallback succeeded")
                else:
                    if use_rec3 is not None:
                        usage_records.append(use_rec3)
                    failures.append({
                        "clip_id": clip_id, "global_idx": gid,
                        "error": "all_attempts_failed",
                        "reason": reason2,
                        "raw_content_preview": (content2 or content or "")[:200],
                    })
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


class ARLApiJob:
    __slots__ = ("row", "img_dir", "audio_path")

    def __init__(self, row, img_dir, audio_path):
        self.row = row
        self.img_dir = img_dir
        self.audio_path = audio_path


# ===========================================================================
# Probe mode: run one scene end-to-end and print diagnostics
# ===========================================================================

def run_probe(args, client, log):
    """Process scene 0 only, print the raw model output, attempt to parse it.
    Returns 0 on success, non-zero on failure. Does NOT write to the main output."""
    log.info("=== PROBE MODE: scene 0 only ===")
    cfg = ApiCallConfig(args.thinking, args.max_completion_tokens, not args.no_json_mode)
    limiter = SlidingWindowRateLimiter(args.max_rpm)

    data_dir = args.data_dir
    img_src_dir = data_dir / "images_masked"
    audio_src = data_dir / "audio" / "anonymized.wav"
    # probe uses a fixed cache dir under captions/, regardless of --out
    cache_dir = (args.out.parent if args.out is not None else (ROOT / "captions")) / "_cache"
    img_out_dir = cache_dir / IMG_CACHE_NAME / "scene_00"
    audio_out_dir = cache_dir / AUDIO_CACHE_NAME
    audio_out = audio_out_dir / "scene_00.wav"

    # preprocess scene 0
    img_out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    scene0_imgs = sorted(p for p in img_src_dir.glob("frame_*.jpg")
                         if not p.name.startswith("._"))[:SCENE_FRAME_COUNT]
    log.info(f"downsampling {len(scene0_imgs)} frames -> {img_out_dir}")
    for fp in scene0_imgs:
        out_p = img_out_dir / fp.name
        if not out_p.exists():
            ffmpeg_downsample_image(fp, out_p, args.resolution, DEFAULT_JPEG_Q)
    if not audio_out.exists():
        log.info(f"slicing audio [0,{SCENE_DURATION_S:.0f}s] -> {audio_out}")
        ffmpeg_slice_audio(audio_src, audio_out, 0.0, SCENE_DURATION_S, AUDIO_SAMPLE_RATE)

    row = {
        "scene_idx": 0, "clip_id": "ARL_scene_00", "global_idx": 1,
        "start_ts": "00:00:00", "start_offset_s": 0.0, "duration": SCENE_DURATION_S,
        "frame_paths": [str(p) for p in scene0_imgs],
        "frame_first": scene0_imgs[0].name, "frame_last": scene0_imgs[-1].name,
        "audio_start_s": 0.0, "audio_path": str(audio_src), "clip_kind": f"{int(SCENE_DURATION_S)}s",
        "is_day_open": True, "is_day_close": False, "status": "pending",
        "src_file": "AriaRealLife/test", "day": 0, "user": "ARL",
        "slice_path": str(img_out_dir),
    }

    log.info("calling MiMo (10 images + 1 audio)...")
    try:
        cap_rec, use_rec, content = _arl_process_one(
            client, args.model, row, img_out_dir, audio_out, drop_audio=False,
            log=log, limiter=limiter, cfg=cfg)
    except Exception as e:
        log.error(f"PROBE FAILED (exception): {type(e).__name__}: {e}")
        return 1

    print("\n========== PROBE RESULT ==========")
    print(f"usage: {use_rec.usage}")
    print(f"recovery: {use_rec.recovery}")
    print(f"latency: {use_rec.latency_s}s")
    print("\n----- raw model content (first 2000 chars) -----")
    print(content[:2000])
    print("\n----- parsed caption -----")
    if cap_rec is not None:
        print(json.dumps(asdict(cap_rec), ensure_ascii=False, indent=2))
        print(f"\nPROBE OK: {SCENE_FRAME_COUNT}-image + audio combo accepted and parsed.")
        return 0
    else:
        print(f"\nPROBE FAILED to parse: recovery={use_rec.recovery}")
        print("See raw content above. Consider the fallbacks in the plan.")
        return 2


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Mimo captioning pipeline for AriaRealLife/test (10-image + audio scenes).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Selection ---
    ap.add_argument("--data-dir", type=Path, default=ROOT / "test",
                    help="root of the AriaRealLife sample (contains images_masked/ and audio/)")
    ap.add_argument("--probe", action="store_true",
                    help="run scene 0 only and print diagnostics; do not write main output")
    # --- Paths ---
    ap.add_argument("--out", type=Path, default=None,
                    help="output jsonl (default: ./captions/test_full.jsonl)")
    # --- Encoding ---
    ap.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    # --- Model / API ---
    ap.add_argument("--model", default="mimo-v2.5")
    ap.add_argument("--base-url", default="https://api.xiaomimimo.com/v1")
    ap.add_argument("--api-key-env", default="MIMO_API_KEY")
    ap.add_argument("--env-file", type=Path, default=None,
                    help=".env file to load MIMO_API_KEY from (default: search CWD + script dir + EgoLife/)")
    ap.add_argument("--thinking", default=THINKING_DEFAULT, choices=["enabled", "disabled"])
    ap.add_argument("--max-completion-tokens", type=int, default=MAX_COMPLETION_TOKENS)
    ap.add_argument("--no-json-mode", action="store_true")
    # --- Concurrency ---
    ap.add_argument("--max-rpm", type=int, default=90)
    ap.add_argument("--api-workers", type=int, default=4)
    ap.add_argument("--preprocess-workers", type=int, default=2)
    # --- Flow ---
    ap.add_argument("--skip-preprocess", action="store_true",
                    help="caption only, reading cached downsampled images + audio slices")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--reset", action="store_true", help="delete output files before running")
    ap.add_argument("--reset-yes-i-know", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only process first N scenes")
    args = ap.parse_args()

    # --- Load API key ---
    env_paths = []
    if args.env_file:
        env_paths.append(args.env_file)
    env_paths += [Path.cwd() / ".env", ROOT / ".env", _REPO / "EgoLife" / ".env"]
    for ep in env_paths:
        if ep.exists():
            load_dotenv(ep)
            break
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"ERROR: {args.api_key_env} not set. Searched: {[str(p) for p in env_paths]}",
              file=sys.stderr)
        sys.exit(2)

    # --- Resolve paths ---
    if args.out is None:
        out_file = ROOT / "captions" / "test_full.jsonl"
    else:
        out_file = args.out
    out_file.parent.mkdir(parents=True, exist_ok=True)
    captions_dir = out_file.parent
    stem = out_file.stem
    usage_path = captions_dir / f"{stem}_usage.jsonl"
    summary_path = captions_dir / f"{stem}_summary.json"
    log_path = captions_dir / f"{stem}_run.log"
    cache_dir = captions_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_cache_root = cache_dir / IMG_CACHE_NAME
    audio_cache_root = cache_dir / AUDIO_CACHE_NAME
    img_cache_root.mkdir(parents=True, exist_ok=True)
    audio_cache_root.mkdir(parents=True, exist_ok=True)
    clips_parquet = cache_dir / "clips.parquet"

    if args.reset:
        existing = 0
        if out_file.exists():
            try:
                existing = sum(1 for ln in out_file.open(encoding="utf-8") if ln.strip().startswith("{"))
            except Exception:
                pass
        if existing > 10 and not args.reset_yes_i_know:
            print(f"ERROR: --reset would delete {out_file} ({existing} records). "
                  f"Add --reset-yes-i-know to confirm.", file=sys.stderr)
            sys.exit(2)
        for p in [out_file, usage_path, summary_path, log_path]:
            if p.exists():
                p.unlink()

    # --- Logging ---
    log = logging.getLogger("caption_pipeline_arl")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt); log.addHandler(fh)
    if tqdm is not None:
        th = TqdmLoggingHandler(); th.setFormatter(fmt); log.addHandler(th)
    else:
        sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    # --- Probe mode ---
    if args.probe:
        sys.exit(run_probe(args, client, log))

    json_mode = not args.no_json_mode
    cfg = ApiCallConfig(args.thinking, args.max_completion_tokens, json_mode)
    log.info(f"data_dir={args.data_dir} model={args.model} thinking={args.thinking} "
             f"json_mode={json_mode} resolution={args.resolution}")
    log.info(f"output: {out_file}")

    # --- Enumerate scenes ---
    if args.skip_preprocess:
        if not clips_parquet.exists():
            log.error(f"--skip-preprocess but {clips_parquet} not found")
            sys.exit(2)
        scenes_df = pd.read_parquet(clips_parquet)
        all_scenes = scenes_df.to_dict("records")
        # re-resolve frame_paths back to absolute source paths
        img_src_dir = args.data_dir / "images_masked"
        for s in all_scenes:
            if isinstance(s["frame_paths"], str):
                s["frame_paths"] = json.loads(s["frame_paths"])
        log.info(f"--skip-preprocess: loaded {len(all_scenes)} scenes from {clips_parquet}")
    else:
        all_scenes = enumerate_scenes(args.data_dir, limit=args.limit)
        log.info(f"found {len(all_scenes)} scenes "
                 f"({len(all_scenes) * SCENE_FRAME_COUNT} frames, "
                 f"{len(all_scenes) * SCENE_DURATION_S:.0f}s audio)")

    # --- Resume ---
    resume_idx = max_written_global_idx(out_file) if args.skip_existing else 0
    done_ids = load_existing_clip_ids(out_file) if args.skip_existing else set()
    log.info(f"resume: skip_existing={args.skip_existing} already_done={len(done_ids)} max_gid={resume_idx}")

    limiter = SlidingWindowRateLimiter(args.max_rpm)

    produced_q: queue.Queue = queue.Queue()
    api_in_q: queue.Queue = queue.Queue(maxsize=args.api_workers * 2)
    result_q: queue.Queue = queue.Queue()
    writer = OrderedWriter(out_file, usage_path, log, next_idx=resume_idx + 1)

    # --- Stats ---
    all_usage, all_records, all_failures = [], [], []
    outcome_counts = {OUTCOME_OK: 0, OUTCOME_SAFETY: 0, OUTCOME_PARSE_FAILED: 0,
                      OUTCOME_EMPTY: 0, "fallback_sectioned": 0,
                      "images_only_fallback": 0, "first_attempt_rejection": 0}
    skipped_existing = 0
    t_start = time.time()

    # --- Preprocess pool (or skip) ---
    pp_job_q: queue.Queue = queue.Queue()
    pp_threads = []
    if args.skip_preprocess:
        fed = 0
        for s in all_scenes:
            clip_id = s["clip_id"]
            if args.skip_existing and (clip_id in done_ids or int(s["global_idx"]) <= resume_idx):
                skipped_existing += 1
                continue
            scene_idx = int(s["scene_idx"])
            img_dir = img_cache_root / f"scene_{scene_idx:02d}"
            audio_path = audio_cache_root / f"scene_{scene_idx:02d}.wav"
            api_in_q.put(ARLApiJob(row=s, img_dir=img_dir, audio_path=audio_path))
            fed += 1
        expected_total = fed
        api_fed = fed
        for _ in range(args.api_workers):
            api_in_q.put(None)
        pp_done, pp_expected = True, 0
    else:
        for w in range(args.preprocess_workers):
            t = threading.Thread(target=_arl_preprocess_worker, args=(pp_job_q, produced_q, log),
                                 name=f"pp-{w}", daemon=True)
            t.start(); pp_threads.append(t)
        for s in all_scenes:
            if args.skip_existing and (s["clip_id"] in done_ids or int(s["global_idx"]) <= resume_idx):
                skipped_existing += 1
                continue
            pp_job_q.put(ARLPreprocessJob(
                scene_row=s, skip_if_exists=(not args.reset),
                resolution=args.resolution, jpeg_q=DEFAULT_JPEG_Q,
                sample_rate=AUDIO_SAMPLE_RATE,
                img_out_dir=img_cache_root, audio_out_dir=audio_cache_root))
        # count actually enqueued
        enqueued = pp_job_q.qsize()
        expected_total = enqueued
        for _ in range(args.preprocess_workers):
            pp_job_q.put(None)
        pp_done, pp_expected, api_fed = False, enqueued, 0

    # --- Progress bar ---
    pbar = None
    if tqdm is not None and expected_total > 0:
        pbar = tqdm(total=expected_total, unit="scene", desc="captioning",
                    mininterval=0.5, smoothing=0.3)
        pbar.refresh()

    # --- API worker pool ---
    api_threads = []
    for w in range(args.api_workers):
        t = threading.Thread(target=_arl_api_worker,
                             args=(api_in_q, result_q, client, args.model, limiter, cfg,
                                   log, w), name=f"api-{w}", daemon=True)
        t.start(); api_threads.append(t)

    # --- Bridge & main loop ---
    pp_rows_for_parquet = []
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
            if pres.error:
                all_failures.append({"clip_id": pres.clip_id, "global_idx": pres.global_idx,
                                     "error": pres.error})
                continue
            row = pres.scene_row
            # attach cache paths for parquet persistence
            row_for_parquet = dict(row)
            row_for_parquet["frame_paths"] = json.dumps(row["frame_paths"])
            row_for_parquet["slice_path"] = str(pres.img_out_dir)
            row_for_parquet["audio_slice_path"] = str(pres.audio_out_path)
            pp_rows_for_parquet.append(row_for_parquet)
            clip_id = row["clip_id"]
            if args.skip_existing and (clip_id in done_ids or int(row["global_idx"]) <= resume_idx):
                skipped_existing += 1
                continue
            api_in_q.put(ARLApiJob(row=row, img_dir=pres.img_out_dir,
                                   audio_path=pres.audio_out_path))
            api_fed += 1
        pp_received += drained
        if not args.skip_preprocess and pp_received >= pp_expected:
            pp_done = True

    def _update_pbar():
        if pbar is None:
            return
        elapsed_min = max((time.time() - t_start) / 60.0, 1e-9)
        rpm = results_received / elapsed_min
        remaining = max((pbar.total or expected_total) - pbar.n, 0)
        eta_min = remaining / rpm if rpm > 0 else 0.0
        pbar.set_postfix(ok=len(all_records), failed=len(all_failures),
                         rpm=f"{rpm:.1f}", eta=f"{eta_min:.1f}min", refresh=False)
        pbar.update(1)

    while True:
        if not args.skip_preprocess:
            bridge_produced()
            if pbar is not None and pp_done and expected_total > 0:
                pass  # total already correct at startup
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
            if cap.recovery == "images_only_fallback":
                outcome_counts["images_only_fallback"] += 1
                break
        all_failures.extend(res.failures)
        writer.submit(res)
        _update_pbar()
        if pp_done and results_received >= api_fed and api_fed > 0:
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

    if pp_rows_for_parquet:
        try:
            pd.DataFrame(pp_rows_for_parquet).to_parquet(clips_parquet, index=False)
        except Exception as e:
            log.warning(f"could not write {clips_parquet}: {e}")

    total_in = sum(u.usage["prompt_tokens"] for u in all_usage)
    total_out = sum(u.usage["completion_tokens"] for u in all_usage)
    total_cached = sum(u.usage["cached_tokens"] for u in all_usage)
    n_ok = len(all_records)
    summary = {
        "model": args.model, "dataset": "AriaRealLife/test",
        "data_dir": str(args.data_dir), "thinking": args.thinking, "json_mode": json_mode,
        "max_completion_tokens": args.max_completion_tokens,
        "max_rpm": args.max_rpm, "api_workers": args.api_workers,
        "preprocess_workers": args.preprocess_workers,
        "skip_preprocess": args.skip_preprocess, "skip_existing": args.skip_existing,
        "n_scenes_input": expected_total, "n_scenes_ok": n_ok,
        "n_scenes_failed": len(all_failures), "n_skipped_existing": skipped_existing,
        "outcomes": dict(outcome_counts),
        "n_safety_rejection": outcome_counts[OUTCOME_SAFETY],
        "n_parse_failed": outcome_counts[OUTCOME_PARSE_FAILED],
        "n_empty": outcome_counts[OUTCOME_EMPTY],
        "n_images_only_fallback": outcome_counts["images_only_fallback"],
        "n_first_attempt_rejection": outcome_counts["first_attempt_rejection"],
        "elapsed_s": round(elapsed, 2),
        "effective_rpm": round(results_received / max(elapsed / 60, 1e-9), 2),
        "tokens": {"total_in": total_in, "total_out": total_out, "total_cached": total_cached},
        "failures": all_failures,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info(f"done. ok={n_ok} failed={len(all_failures)} skipped={skipped_existing} "
             f"first_rejection={outcome_counts['first_attempt_rejection']} "
             f"safety={outcome_counts[OUTCOME_SAFETY]} parse_failed={outcome_counts[OUTCOME_PARSE_FAILED]} "
             f"empty={outcome_counts[OUTCOME_EMPTY]} img_only={outcome_counts['images_only_fallback']}")
    log.info(f"  elapsed={elapsed:.1f}s ({elapsed/60:.1f}min)  "
             f"effective_rpm={results_received / max(elapsed / 60, 1e-9):.1f}")
    log.info(f"  tokens: in={total_in} out={total_out} cached={total_cached}")
    log.info(f"  captions -> {out_file}")
    log.info(f"  summary  -> {summary_path}")


if __name__ == "__main__":
    main()
