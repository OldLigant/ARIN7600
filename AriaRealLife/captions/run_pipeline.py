#!/usr/bin/env python3
"""POV Dense Video Captioning Pipeline — 统一版

从 HuggingFace 读取 POV 数据，通过 Gemini 生成 v9 三遍 dense caption。

Usage:
  python run_pipeline.py list                           # 列出可用的天和录制
  python run_pipeline.py run --day 2026-05-18           # 处理一天全部录制
  python run_pipeline.py run --tar aria/2026-05-18/...  # 处理单条录制
  python run_pipeline.py run --tar ... --max-clips 3    # 冒烟测试（只跑前 3 个 clip）
  python run_pipeline.py run --tar ... --workers 6      # 指定并发数

合并自 pipelines/ 下的 remote_tar / caption_core / caption_v9 / cloud_worker / merge，
去掉了 HF Job 提交/上传、Watchdog、Batch API 等基础设施，保留全部 v9 caption 核心逻辑。
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import re
import sys
import threading
import time
import wave
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── optional heavy deps ─────────────────────────────────────────────
_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass

_HAS_GCS = False
try:
    from google.cloud import storage as _gcs_mod
    _HAS_GCS = True
except ImportError:
    pass

# ====================================================================
#  Constants
# ====================================================================

# ---- gaze overlay (matches _tools/render_gaze_overlay.py) ----------
GAZE_COLOR = (0, 255, 128)
GAZE_INVALID_COLOR = (255, 60, 60)
TRAIL_COLOR = (255, 200, 0)
POINT_RADIUS = 18          # @2880
TRAIL_POINTS = 5
NATIVE_DIM = 2880

# ---- frame / Gemini ------------------------------------------------
MAX_DIM = 1440
JPEG_QUALITY = 60
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_SERVICE_TIER = "flex"
DEFAULT_REQUEST_TIMEOUT_SEC = 1800
DEFAULT_MAX_WORKERS = 12
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.3
INLINE_LIMIT = 18 << 20
DOWNGRADE_DIMS = [1440, 1200, 1024]

# ---- v9 specific ---------------------------------------------------
FRAME_DIM, FRAME_Q = 2880, 80       # pass1
CROP_DIM, CROP_Q = 2880, 60         # pass2
MAX_INLINE_MB = 20.0
INLINE_RETRIES = 3
MAX_CROPS = 14
QUOTA_RETRIES = 10
QUOTA_BACKOFF = [15, 30, 60, 90, 120, 180, 240, 300, 300, 300]

# ---- tar -----------------------------------------------------------
BLOCK = 512
REPO = "mmm8383/pov-data"

# ---- regex ----------------------------------------------------------
FRAME_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_[A-Z]{2,4}__frame_(\d+)"
    r"\.(?:jpg|jpeg|png|txt)$", re.I)
TR_RE = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: [A-Z]{2,4})? -> "
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: [A-Z]{2,4})?\]\s*(.*)$")
TR_REL_RE = re.compile(
    r"^\[(\d{1,2}):(\d{2})(?::(\d{2}))?\s*[-–—]\s*"
    r"(\d{1,2}):(\d{2})(?::(\d{2}))?\]\s*(.*)$")
TR_REL_BASE = datetime(2000, 1, 1)
OCR_SKIP_PREFIX = ("capture_time", "capture_epoch", "frame_index")
DUR_RE = re.compile(r"_(\d+)m_")
TIME_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")

# ---- pricing (gemini-3.8-flash, 2026-12-31 前引导价) ----------------
PRICE_IN, PRICE_OUT = 0.75, 3.75

# ---- PII audit patterns --------------------------------------------
PII_PATTERNS = {
    "school_plain": r"(?i)\buniversity of hong kong\b|\bHKU\b|港大|香港大学",
    "school_generic": r"(?i)\b[A-Z][a-z]+ (?:University|College)\b",
    "email_plain": r"[A-Za-z0-9._%+-]+@(?!example\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone_plain": r"\b(?:\+?86)?1[3-9]\d{9}\b|\b\d{4}[- ]\d{4}\b",
}
ANON_PATTERNS = r"University X|School X|User [A-Z]\b|email_x@|XXX-XXXX-XXXX|Address X"

# ---- threading locks ------------------------------------------------
_print_lock = threading.Lock()
_silero_lock = threading.Lock()
_gcs_lock = threading.Lock()

# ---- module-level state set by config --------------------------------
_MODEL = DEFAULT_MODEL
_SERVICE_TIER = DEFAULT_SERVICE_TIER
_GCS_BUCKET = "hku-capstone-caption-frames"
_gcs_client = None


def _log(msg):
    with _print_lock:
        print(msg, flush=True)


_SECRET_ENV_NAMES = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def redact_secrets(value, environ=None, limit: int | None = None) -> str:
    """Return a log-safe string without known token/key material."""
    env = os.environ if environ is None else environ
    text = str(value or "")
    for name in _SECRET_ENV_NAMES:
        secret = env.get(name, "")
        if secret and len(secret) >= 4:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;\"']+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)((?:api[_-]?key|key|token)\s*[=:]\s*)[^\s,;\"']+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"AIza[0-9A-Za-z_-]{10,}", "[REDACTED]", text)
    text = re.sub(r"hf_[0-9A-Za-z_-]{10,}", "[REDACTED]", text)
    text = " ".join(text.split())
    return text[:limit] if limit is not None else text


def classify_api_error(error) -> str:
    """Classify a Gemini failure for retry and fallback decisions."""
    msg = str(error or "").lower()
    if any(x in msg for x in (
        "429", "resource_exhausted", "quota", "rate limit", "rate_limit",
        "billing account", "billing is disabled", "insufficient credits",
    )):
        return "quota"
    if any(x in msg for x in (
        "413", "payload too large", "request too large", "request entity too large",
        "content size", "exceeds the maximum size", "exceeded the maximum size",
    )):
        return "payload"
    if any(x in msg for x in (
        "404", "not_found", "model not found", "model is not found",
        "unsupported model", "unknown model",
    )):
        return "model"
    if any(x in msg for x in (
        "default credentials were not found",
        "could not automatically determine credentials",
        "unauthenticated", "permission_denied", "permission denied",
        "api key not valid", "invalid api key", "api key expired",
        "401", "403",
    )):
        return "auth"
    if any(x in msg for x in (
        "500", "502", "503", "504", "internal server error", "unavailable",
        "service unavailable", "empty stream",
    )):
        return "server"
    if any(x in msg for x in (
        "timeout", "timed out", "connection error", "connection reset",
        "connection aborted", "temporary failure", "name resolution",
    )):
        return "network"
    return "other"


def should_retry_api_error(error_type: str) -> bool:
    return error_type in {"quota", "server", "network"}


def should_retry_clip_result(result: dict) -> bool:
    if result.get("ok"):
        return False
    return result.get("error_type") not in {"auth", "model"}


class AIMDController:
    """Small additive-increase/multiplicative-decrease concurrency controller."""

    CONGESTION_ERRORS = {"quota", "server", "network"}

    def __init__(self, initial: int = 3, maximum: int = DEFAULT_MAX_WORKERS):
        self.maximum = max(1, int(maximum))
        self.limit = max(1, min(int(initial), self.maximum))
        self.peak = self.limit
        self.increases = 0
        self.decreases = 0
        self._healthy = 0

    def observe(self, result: dict):
        """Update the window and return (old, new, reason) when it reacts."""
        if result.get("ok"):
            if self.limit >= self.maximum:
                self._healthy = 0
                return None
            self._healthy += 1
            if self._healthy < self.limit:
                return None
            old = self.limit
            self.limit += 1
            self.peak = max(self.peak, self.limit)
            self.increases += 1
            self._healthy = 0
            return old, self.limit, "healthy"

        error_type = result.get("error_type")
        if error_type not in self.CONGESTION_ERRORS:
            return None
        old = self.limit
        self.limit = max(1, self.limit // 2)
        self.decreases += 1
        self._healthy = 0
        return old, self.limit, str(error_type)


def run_dynamic_pool(items, worker, controller: AIMDController, on_change=None):
    """Yield (item, result, error) while keeping only controller.limit active."""
    source = iter(items)
    exhausted = False
    futures = {}

    with ThreadPoolExecutor(controller.maximum, thread_name_prefix="cap") as ex:
        while futures or not exhausted:
            while not exhausted and len(futures) < controller.limit:
                try:
                    item = next(source)
                except StopIteration:
                    exhausted = True
                    break
                futures[ex.submit(worker, item)] = item

            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                item = futures.pop(future)
                error = None
                try:
                    result = future.result()
                except Exception as exc:
                    result = None
                    error = exc

                if error is not None:
                    signal = {
                        "ok": False,
                        "error_type": classify_api_error(error),
                    }
                elif isinstance(result, tuple) and len(result) > 1 and isinstance(result[1], dict):
                    signal = result[1]
                elif isinstance(result, dict):
                    signal = result
                else:
                    signal = {"ok": True}
                change = controller.observe(signal)
                if change and on_change:
                    on_change(change)
                yield item, result, error


def generation_config_kwargs(model: str, max_output_tokens: int) -> dict:
    config = {"max_output_tokens": max_output_tokens}
    if not model.lower().startswith("gemini-3"):
        config["temperature"] = 0.3
    return config


def configure_google_environment(project: str, location: str = "global", environ=None) -> str:
    """Configure the Gen AI SDK exactly as HF Job environment auth expects."""
    env = os.environ if environ is None else environ
    if project:
        env["GOOGLE_CLOUD_PROJECT"] = project
    if location:
        env["GOOGLE_CLOUD_LOCATION"] = location
    env["GOOGLE_GENAI_USE_ENTERPRISE"] = "True"
    return "service-bound-api-key" if env.get("GOOGLE_API_KEY") else "adc"


# ====================================================================
#  Config
# ====================================================================

def load_config(path: str = "config.json") -> dict:
    defaults = {
        "hf_token": "",
        "google_project": "",
        "google_location": "global",
        "model": DEFAULT_MODEL,
        "service_tier": DEFAULT_SERVICE_TIER,
        "request_timeout_sec": DEFAULT_REQUEST_TIMEOUT_SEC,
        "output_dir": "./caption_output",
        "workers": 3,
        "max_workers": DEFAULT_MAX_WORKERS,
        "adaptive_concurrency": False,
        "clip_seconds": 30,
        "rounds": 3,
    }
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            if not k.startswith("_"):
                defaults[k] = v
    # HF Job secrets/environment must take precedence over local config values.
    defaults["hf_token"] = os.environ.get("HF_TOKEN") or defaults["hf_token"]
    defaults["google_project"] = (os.environ.get("GOOGLE_CLOUD_PROJECT")
                                  or defaults["google_project"])
    defaults["google_location"] = (os.environ.get("GOOGLE_CLOUD_LOCATION")
                                   or defaults["google_location"])
    return defaults


# ====================================================================
#  Prompts (v9 — embedded verbatim)
# ====================================================================

PROMPT_V9 = r"""You are a dense video captioning system analyzing first-person (egocentric) video from smart glasses equipped with eye tracking. The green circle on each frame marks where I am actually looking (gaze point), and the yellow trail shows my recent gaze trajectory.

IMPORTANT RULES:

0. Every action I take that changes the state of the external physical world must be described in meticulous detail, ensuring that the original scene can be recreated as closely as possible based on the text.

1. Write ALL captions in FIRST PERSON ("I walk to...", "I tap on...", "I pick up...").

2. APPROPRIATE GRANULARITY. Each segment should describe a meaningful action unit — not too coarse (combining unrelated actions) and not too fine (splitting one continuous motion into micro-steps).
   - If I am performing the SAME continuous action across several frames (e.g., walking, scrolling, typing), describe it ONCE when it starts. Do NOT repeat "I continue walking" or "I keep scrolling" every second.
   - If the scene is mostly static or I'm doing the same thing for several seconds, merge those into ONE segment covering the full duration.
   - Start a new segment only when something MEANINGFULLY changes: a new action begins, I shift attention to something different, a new UI element appears, or someone speaks.
   - BAD: 5 consecutive segments of "I scroll down" — merge into one "I scroll through the menu for 5 seconds."
   - GOOD: "17:57:12–17:57:18: I scroll through the 日式系列 category showing items priced ¥15-¥22."

3. Be EXTREMELY specific about actions and UI interactions:
   - Not "browse menu" but "I scroll down through the 日式系列 category showing items priced ¥15-¥22".
   - For screen/UI: describe exactly which button I tap, which option I select, what text I read, what price I see.
   - For menus / pop-up modals / selection dialogs: LIST THE SPECIFIC OPTIONS shown. Example: "I see a pop-up modal with options: 'Enter to select', 'Tab/Arrow keys to navigate', 'Esc to cancel'. The list shows: 'train.py', 'eval.py', 'config.yaml'." Do NOT write "I examine the options listed" — instead enumerate what the options actually are.
   - For ordering kiosks / food menus: describe each visible item name and price.

4. ENVIRONMENT — EXHAUSTIVE WORLD STATE. The "environment" field trains a world model: it must record
   EVERYTHING the world is presenting to me at that moment, not just what is relevant to my action.
   Write it as a dense description covering ALL of the following that are visible:

   a) PHYSICAL SPACE: room/location type, furniture, walls, lighting, floor, weather/outdoor view,
      other people present (anonymized) and what they are doing.
   b) OBJECTS: every distinguishable object in view — on the desk, in my hands, on shelves, on the
      ground — with color, material, state (open/closed, full/empty, on/off) and rough position
      (left / right / center / foreground / background).
   c) SCREEN CONTENT — describe each screen separately and completely:
      - PHONE: which app, which page/tab, what is displayed (post titles, chat messages, prices,
        buttons, notification badges, status bar indicators), keyboard state, what is scrolled into view.
      - COMPUTER: which app/window, filenames, tab bar contents, sidebar/file-tree entries,
        code or document content, chat panels, status bar, dock/taskbar icons.
      - OTHER DISPLAYS: kiosks, TVs, projectors, e-ink, smartwatch — full listing of options,
        menu items, prices, headings shown.
   d) TEXT IN THE WORLD: signage, posters, labels, packaging with meaningful text, printed pages.
   e) MEDIA BEING PLAYED: when I watch a video or listen to audio, describe the ACTUAL CONTENT:
      - "I hear a voice saying '...' from the video."
      - "The video shows a gameplay scene of Genshin Impact with a character using elemental burst."
      - "A short-video feed shows a person cooking stir-fried noodles in a wok."
   f) AMBIENT AUDIO: background music, announcements, machine noise, other people's conversation.

   HOW MUCH TO WRITE — establishing shot, then deltas only:

   - THE FIRST SEGMENT of the clip carries the FULL establishing description: cover every category
     (a) through (f) above in detail. This is the initial world state an agent gets loaded with,
     so it must be complete and self-contained. Be generous here — this is the one place where
     length is wanted.

   - EVERY LATER SEGMENT describes ONLY WHAT CHANGED since the previous segment. Do not restate
     unchanged state. These deltas are injected into a simulated agent as events, so redundant
     re-description is pure noise.
     - GOOD: "The kiosk screen switches from the 日式系列 category to 娘家碗飯, now listing
       '娘家自選雙餸，白飯 $31.5' and '娘家自選三餸，白飯 $41.8'. A staff member walks past behind me."
     - BAD: repeating the whole room, desk contents and every sidebar item again when only the
       screen scrolled.
     - If genuinely nothing in the world changed, write "No change." — that is a valid, useful value.

   - IGNORE MEANINGLESS CHANGES. My head moves constantly, so objects drift in and out of frame,
     reframe, or change apparent angle without anything actually happening in the world. Do NOT
     report these. Only report changes with real world-state meaning:
     - screen content changing (navigation, new message, video advancing, dialog opening)
     - objects being moved, picked up, put down, opened, closed, switched on/off
     - people entering, leaving, or acting
     - lighting, location, or ambient-audio changes
     - new text becoming readable because content changed, not because my head turned toward it
     - An object merely entering view because I turned my head is NOT a change. If it matters,
       it belongs in the establishing description or in a perception segment.

   - Be concrete and enumerative in whatever you do write. "The desk got messier" is useless;
     name what appeared or moved.
   - Still create separate perception segments ("I see ...", "I notice ...") when I actively shift
     attention to something — the environment field is world state, those segments are attention events.

5. SPEECH HANDLING — EXACT QUOTES WITH JUDGMENT.
   - Quote the EXACT words spoken (Chinese or English as spoken). Every speech event MUST be its own segment with the verbatim quote in both the action and speech fields.
   - The audio transcript may contain recognition errors — homophones (同音字) are common (e.g., "是" vs "试", "在" vs "再"). Use context to correct obvious errors.
   - The transcript may include speech from OTHER people nearby, background TV/radio audio, or ambient noise fragments. Judge whether each utterance is actually ME speaking, someone speaking TO me, or irrelevant background audio. Only include relevant speech.
   - Short noise fragments or unclear mumbles that don't form meaningful words can be omitted.
   - If I speak AND do something physical simultaneously, those are still separate segments.
   - GOOD: "I say '拿个馒头' to the cafeteria staff." / "The cashier says '十三块七'."
   - BAD: "I speak to the staff." (missing the quote)

6. TEXT AND OCR DATA. Include useful readable text from frames and OCR data — signs, labels, prices, screen content.
   - The green gaze circle shows what text I am actually focusing on — prioritize describing text near the gaze point.
   - NOTE: OCR-detected text may contain recognition errors (similar-looking characters misread), and may include trivial environmental text (keyboard labels, product packaging, watermarks) that is not informative. Use your judgment to identify and describe only the MEANINGFUL, informative text — don't transcribe keyboard keys or random packaging text.

7. TIME RANGES MUST USE ABSOLUTE HKT TIMESTAMPS, matching the frame filenames. For example: "17:57:12–17:57:15" not "0s-5s". Note: The audio transcript has coarse 30-second block timestamps. Use visual cues (gestures, mouth movement, context changes between frames) to estimate more precise speech timing within ±2 seconds.

8. PRIVACY MASKING. Replace ALL personally identifiable or sensitive information with anonymized placeholders:
   - Real names / usernames / account names → "User X", "User Y", etc.
   - School names / university names → "School X", "University X"
   - Email addresses → "[email_x@example.com](mailto:email_x@example.com)"
   - Phone numbers → "XXX-XXXX-XXXX"
   - Home addresses → "Address X"
   - API keys / access tokens / passwords / secrets / credentials → replace the entire sensitive value with "X"
   - Any other private authentication or security-related information → "X"
   - Any NSFW / inappropriate content (nudity, explicit material) → describe as "[redacted content]"
   - Keep generic brand names (WeChat, Bilibili, Chrome, VS Code) — only anonymize personal identifiers.

9. TYPING / TEXT INPUT. When I spend time typing or entering text, describe WHAT I type and WHERE — but only the informative, non-obvious parts:
   - GOOD: "I type '如何优化transformer推理速度' into the Doubao AI search box."
   - BAD: "I type on the keyboard." (too vague)
   Focus on WHAT content I'm entering and in WHICH application/field, not the mechanical act of pressing keys.

10. SCREEN CONTENT DETAIL. When I look at a computer or phone screen, describe WHAT is actually visible:
   - What app/website is open? What page/tab am I on?
   - What specific content is displayed? (article titles, code, chat messages, video titles, menu items)
   - For pop-up modals, dropdowns, autocomplete lists: enumerate the visible options/items.
   - GOOD: "I see a VS Code command palette showing options: 'Python: Select Interpreter', 'Python: Run File in Terminal', 'Format Document'."
   - BAD: "I examine the options listed in the pop-up modal." (WHAT options? List them!)

11. GAZE TRACKING. The green circle on each frame shows my exact gaze position. Use this to determine what I am actually looking at vs. what is merely visible in the periphery.

## VERBATIM SPEECH — hard requirement
Whenever I speak, the `action` field MUST carry my words **verbatim, character for character**,
inside quotes. Not paraphrased, not summarised, not truncated, not translated, not cleaned up.
- Keep the original language exactly as spoken (Chinese stays Chinese, English stays English,
  code-switching stays mixed).
- Keep filler, repetition, stutters and self-corrections as spoken ("就是先帮我输入一个默认的
  默认的一个值" keeps both 默认的).
- Never replace any part of an utterance with "..." or "等等" or a description of what I said.
  "I explain the requirements" is WRONG. `I say '<exact words>'` is the only acceptable form.
- The same verbatim quote also goes in the `speech` field. Both fields carry it in full.
- This applies to `action_brief` too: if the segment is a speech act, the quote survives
  compression intact — drop the surrounding scaffolding, never the words themselves.

## action_brief — Condensed Action
Every segment MUST also carry an `action_brief`: `action` with the dead weight removed.

THE RULE — one verb, but keep everything that carries information:
- Exactly ONE main verb. When `action` chains verbs with "and"/"while"/"then", keep the single
  most informative one and drop the others. ("bring ... and set" → "place"; "type ... and send"
  → "send"; "rest my hand while watching" → "monitor".)
- KEEP every information-bearing element, however long that makes it:
  * quoted speech or message text — verbatim, character for character, never paraphrased,
    truncated or elided; see the VERBATIM SPEECH rule above
  * who it is addressed to / who is speaking
  * the specific topic, title, or subject matter
  * app / site / brand names, and the specific object being acted on
- DROP only what carries no information:
  device names ("on my MacBook"), body parts and manner ("with my right hand", "using the
  on-screen keyboard"), posture and location filler ("while sitting at my desk"), screen
  scaffolding ("on the screen", "in the input box"), and any verb already implied by another.
- This is compression, never invention or summarisation. Do not replace a specific noun with a
  generic one — "the Claude Code explanation about TanhTransformedDistribution" must NOT become
  "explanations". No fixed word budget: as short as possible, but not one bit of signal shorter.

CALIBRATION — these four are the standard:
- action: "I read the Claude Code panel explanation about TanhTransformedDistribution and
  change-of-variables log probability calculation."
  action_brief: "I read the Claude Code explanation about TanhTransformedDistribution and
  change-of-variables log probability."
  ← keep the topic; it is the whole point of the segment. Only "panel" and the trailing
    "calculation" go.
- action: "I bring a clear glass bottle to the countertop water dispenser beside the sink and
  set it on the dispenser tray."
  action_brief: "I place a glass bottle."
  ← two verbs → one; the dispenser/sink/tray are scenery, the bottle is the object.
- action: "I type '感觉ai好慢' into the WeChat chat with 'babe' using the on-screen keyboard and
  send it."
  action_brief: "I send '感觉ai好慢' to babe."
  ← keep BOTH the message text and the recipient; drop the keyboard and the app chrome.
- action: "I rest my hand on top of the water dispenser while watching the glass bottle fill
  with water."
  action_brief: "I monitor the water filling."
  ← the hand is incidental; the watching is the action.

BAD (over-compression — loses signal):
- "I read Claude explanations."            ← topic destroyed
- "I send a WeChat message."               ← message text and recipient destroyed
- "I interact with the interface."         ← everything destroyed

## Output Format
Return a JSON object:
{
  "scene_summary": "First-person one-sentence overview",
  "segments": [
    {
      "time_range": "HH:MM:SS–HH:MM:SS",
      "action": "I grab a coke from the fridge.",
      "action_brief": "I grab a coke.",
      "objects": ["specific objects I interact with or look at"],
      "environment": "FIRST segment: full establishing world state per Rule 4 (physical space, every visible object with state and position, complete screen content of every display, world text/signage, media playing, ambient audio). LATER segments: ONLY what changed since the previous segment, ignoring changes caused merely by head movement; 'No change.' is valid.",
      "text_visible": ["meaningful readable text near gaze point — skip trivial keyboard/packaging labels"],
      "speech": "exact quote of what I or others say in this segment, or null",
      "details": "fine-grained context: which hand, which direction, micro-actions"
    }
  ],
  "activity_chain": "comma-separated first-person atomic actions"
}

## Segmentation Rules
- Each segment = one meaningful action unit. Merge continuous/repetitive actions into one segment.
- A new segment starts when: a NEW action begins, I shift attention significantly, a new UI element appears, or someone speaks.
- Speech segments are separate from action segments. Every utterance MUST appear with exact quote — but omit noise/fragments.
- Environmental observation segments ("I see...", "I hear...") are separate from action segments.
- Media content (videos I watch, audio I hear) → environment segments with specific content details.
- Typical: 5-12 segments for a 30-second clip. Don't over-segment static or repetitive scenes."""

PROMPT_V9_NOGAZE = r"""You are a dense video captioning system analyzing first-person (egocentric) video from smart glasses. This recording has NO eye-tracking data, so the frames carry no gaze markers — judge what I am attending to from what is centred, held, manipulated or dwelt on across consecutive frames.

IMPORTANT RULES:

0. Every action I take that changes the state of the external physical world must be described in meticulous detail, ensuring that the original scene can be recreated as closely as possible based on the text.

1. Write ALL captions in FIRST PERSON ("I walk to...", "I tap on...", "I pick up...").

2. APPROPRIATE GRANULARITY. Each segment should describe a meaningful action unit — not too coarse (combining unrelated actions) and not too fine (splitting one continuous motion into micro-steps).
   - If I am performing the SAME continuous action across several frames (e.g., walking, scrolling, typing), describe it ONCE when it starts. Do NOT repeat "I continue walking" or "I keep scrolling" every second.
   - If the scene is mostly static or I'm doing the same thing for several seconds, merge those into ONE segment covering the full duration.
   - Start a new segment only when something MEANINGFULLY changes: a new action begins, I shift attention to something different, a new UI element appears, or someone speaks.
   - BAD: 5 consecutive segments of "I scroll down" — merge into one "I scroll through the menu for 5 seconds."
   - GOOD: "17:57:12–17:57:18: I scroll through the 日式系列 category showing items priced ¥15-¥22."

3. Be EXTREMELY specific about actions and UI interactions:
   - Not "browse menu" but "I scroll down through the 日式系列 category showing items priced ¥15-¥22".
   - For screen/UI: describe exactly which button I tap, which option I select, what text I read, what price I see.
   - For menus / pop-up modals / selection dialogs: LIST THE SPECIFIC OPTIONS shown. Example: "I see a pop-up modal with options: 'Enter to select', 'Tab/Arrow keys to navigate', 'Esc to cancel'. The list shows: 'train.py', 'eval.py', 'config.yaml'." Do NOT write "I examine the options listed" — instead enumerate what the options actually are.
   - For ordering kiosks / food menus: describe each visible item name and price.

4. ENVIRONMENT — EXHAUSTIVE WORLD STATE. The "environment" field trains a world model: it must record
   EVERYTHING the world is presenting to me at that moment, not just what is relevant to my action.
   Write it as a dense description covering ALL of the following that are visible:

   a) PHYSICAL SPACE: room/location type, furniture, walls, lighting, floor, weather/outdoor view,
      other people present (anonymized) and what they are doing.
   b) OBJECTS: every distinguishable object in view — on the desk, in my hands, on shelves, on the
      ground — with color, material, state (open/closed, full/empty, on/off) and rough position
      (left / right / center / foreground / background).
   c) SCREEN CONTENT — describe each screen separately and completely:
      - PHONE: which app, which page/tab, what is displayed (post titles, chat messages, prices,
        buttons, notification badges, status bar indicators), keyboard state, what is scrolled into view.
      - COMPUTER: which app/window, filenames, tab bar contents, sidebar/file-tree entries,
        code or document content, chat panels, status bar, dock/taskbar icons.
      - OTHER DISPLAYS: kiosks, TVs, projectors, e-ink, smartwatch — full listing of options,
        menu items, prices, headings shown.
   d) TEXT IN THE WORLD: signage, posters, labels, packaging with meaningful text, printed pages.
   e) MEDIA BEING PLAYED: when I watch a video or listen to audio, describe the ACTUAL CONTENT:
      - "I hear a voice saying '...' from the video."
      - "The video shows a gameplay scene of Genshin Impact with a character using elemental burst."
      - "A short-video feed shows a person cooking stir-fried noodles in a wok."
   f) AMBIENT AUDIO: background music, announcements, machine noise, other people's conversation.

   HOW MUCH TO WRITE — establishing shot, then deltas only:

   - THE FIRST SEGMENT of the clip carries the FULL establishing description: cover every category
     (a) through (f) above in detail. This is the initial world state an agent gets loaded with,
     so it must be complete and self-contained. Be generous here — this is the one place where
     length is wanted.

   - EVERY LATER SEGMENT describes ONLY WHAT CHANGED since the previous segment. Do not restate
     unchanged state. These deltas are injected into a simulated agent as events, so redundant
     re-description is pure noise.
     - GOOD: "The kiosk screen switches from the 日式系列 category to 娘家碗飯, now listing
       '娘家自選雙餸，白飯 $31.5' and '娘家自選三餸，白飯 $41.8'. A staff member walks past behind me."
     - BAD: repeating the whole room, desk contents and every sidebar item again when only the
       screen scrolled.
     - If genuinely nothing in the world changed, write "No change." — that is a valid, useful value.

   - IGNORE MEANINGLESS CHANGES. My head moves constantly, so objects drift in and out of frame,
     reframe, or change apparent angle without anything actually happening in the world. Do NOT
     report these. Only report changes with real world-state meaning:
     - screen content changing (navigation, new message, video advancing, dialog opening)
     - objects being moved, picked up, put down, opened, closed, switched on/off
     - people entering, leaving, or acting
     - lighting, location, or ambient-audio changes
     - new text becoming readable because content changed, not because my head turned toward it
     - An object merely entering view because I turned my head is NOT a change. If it matters,
       it belongs in the establishing description or in a perception segment.

   - Be concrete and enumerative in whatever you do write. "The desk got messier" is useless;
     name what appeared or moved.
   - Still create separate perception segments ("I see ...", "I notice ...") when I actively shift
     attention to something — the environment field is world state, those segments are attention events.

5. SPEECH HANDLING — EXACT QUOTES WITH JUDGMENT.
   - Quote the EXACT words spoken (Chinese or English as spoken). Every speech event MUST be its own segment with the verbatim quote in both the action and speech fields.
   - The audio transcript may contain recognition errors — homophones (同音字) are common (e.g., "是" vs "试", "在" vs "再"). Use context to correct obvious errors.
   - The transcript may include speech from OTHER people nearby, background TV/radio audio, or ambient noise fragments. Judge whether each utterance is actually ME speaking, someone speaking TO me, or irrelevant background audio. Only include relevant speech.
   - Short noise fragments or unclear mumbles that don't form meaningful words can be omitted.
   - If I speak AND do something physical simultaneously, those are still separate segments.
   - GOOD: "I say '拿个馒头' to the cafeteria staff." / "The cashier says '十三块七'."
   - BAD: "I speak to the staff." (missing the quote)

6. TEXT AND OCR DATA. Include useful readable text from frames and OCR data — signs, labels, prices, screen content.
   - No gaze marker is available — prioritize text that is centred, large, or on the screen/surface I am actively working with.
   - NOTE: OCR-detected text may contain recognition errors (similar-looking characters misread), and may include trivial environmental text (keyboard labels, product packaging, watermarks) that is not informative. Use your judgment to identify and describe only the MEANINGFUL, informative text — don't transcribe keyboard keys or random packaging text.

7. TIME RANGES MUST USE ABSOLUTE HKT TIMESTAMPS, matching the frame filenames. For example: "17:57:12–17:57:15" not "0s-5s". Note: The audio transcript has coarse 30-second block timestamps. Use visual cues (gestures, mouth movement, context changes between frames) to estimate more precise speech timing within ±2 seconds.

8. PRIVACY MASKING. Replace ALL personally identifiable or sensitive information with anonymized placeholders:
   - Real names / usernames / account names → "User X", "User Y", etc.
   - School names / university names → "School X", "University X"
   - Email addresses → "[email_x@example.com](mailto:email_x@example.com)"
   - Phone numbers → "XXX-XXXX-XXXX"
   - Home addresses → "Address X"
   - API keys / access tokens / passwords / secrets / credentials → replace the entire sensitive value with "X"
   - Any other private authentication or security-related information → "X"
   - Any NSFW / inappropriate content (nudity, explicit material) → describe as "[redacted content]"
   - Keep generic brand names (WeChat, Bilibili, Chrome, VS Code) — only anonymize personal identifiers.

9. TYPING / TEXT INPUT. When I spend time typing or entering text, describe WHAT I type and WHERE — but only the informative, non-obvious parts:
   - GOOD: "I type '如何优化transformer推理速度' into the Doubao AI search box."
   - BAD: "I type on the keyboard." (too vague)
   Focus on WHAT content I'm entering and in WHICH application/field, not the mechanical act of pressing keys.

10. SCREEN CONTENT DETAIL. When I look at a computer or phone screen, describe WHAT is actually visible:
   - What app/website is open? What page/tab am I on?
   - What specific content is displayed? (article titles, code, chat messages, video titles, menu items)
   - For pop-up modals, dropdowns, autocomplete lists: enumerate the visible options/items.
   - GOOD: "I see a VS Code command palette showing options: 'Python: Select Interpreter', 'Python: Run File in Terminal', 'Format Document'."
   - BAD: "I examine the options listed in the pop-up modal." (WHAT options? List them!)

11. ATTENTION WITHOUT GAZE. This recording has no eye tracking, so there is no gaze marker on the frames. Infer what I am attending to from frame composition and continuity — what stays centred, what my hands are on, what persists across consecutive frames — and do not claim to know my exact gaze point.

## VERBATIM SPEECH — hard requirement
Whenever I speak, the `action` field MUST carry my words **verbatim, character for character**,
inside quotes. Not paraphrased, not summarised, not truncated, not translated, not cleaned up.
- Keep the original language exactly as spoken (Chinese stays Chinese, English stays English,
  code-switching stays mixed).
- Keep filler, repetition, stutters and self-corrections as spoken ("就是先帮我输入一个默认的
  默认的一个值" keeps both 默认的).
- Never replace any part of an utterance with "..." or "等等" or a description of what I said.
  "I explain the requirements" is WRONG. `I say '<exact words>'` is the only acceptable form.
- The same verbatim quote also goes in the `speech` field. Both fields carry it in full.
- This applies to `action_brief` too: if the segment is a speech act, the quote survives
  compression intact — drop the surrounding scaffolding, never the words themselves.

## action_brief — Condensed Action
Every segment MUST also carry an `action_brief`: `action` with the dead weight removed.

THE RULE — one verb, but keep everything that carries information:
- Exactly ONE main verb. When `action` chains verbs with "and"/"while"/"then", keep the single
  most informative one and drop the others. ("bring ... and set" → "place"; "type ... and send"
  → "send"; "rest my hand while watching" → "monitor".)
- KEEP every information-bearing element, however long that makes it:
  * quoted speech or message text — verbatim, character for character, never paraphrased,
    truncated or elided; see the VERBATIM SPEECH rule above
  * who it is addressed to / who is speaking
  * the specific topic, title, or subject matter
  * app / site / brand names, and the specific object being acted on
- DROP only what carries no information:
  device names ("on my MacBook"), body parts and manner ("with my right hand", "using the
  on-screen keyboard"), posture and location filler ("while sitting at my desk"), screen
  scaffolding ("on the screen", "in the input box"), and any verb already implied by another.
- This is compression, never invention or summarisation. Do not replace a specific noun with a
  generic one — "the Claude Code explanation about TanhTransformedDistribution" must NOT become
  "explanations". No fixed word budget: as short as possible, but not one bit of signal shorter.

CALIBRATION — these four are the standard:
- action: "I read the Claude Code panel explanation about TanhTransformedDistribution and
  change-of-variables log probability calculation."
  action_brief: "I read the Claude Code explanation about TanhTransformedDistribution and
  change-of-variables log probability."
  ← keep the topic; it is the whole point of the segment. Only "panel" and the trailing
    "calculation" go.
- action: "I bring a clear glass bottle to the countertop water dispenser beside the sink and
  set it on the dispenser tray."
  action_brief: "I place a glass bottle."
  ← two verbs → one; the dispenser/sink/tray are scenery, the bottle is the object.
- action: "I type '感觉ai好慢' into the WeChat chat with 'babe' using the on-screen keyboard and
  send it."
  action_brief: "I send '感觉ai好慢' to babe."
  ← keep BOTH the message text and the recipient; drop the keyboard and the app chrome.
- action: "I rest my hand on top of the water dispenser while watching the glass bottle fill
  with water."
  action_brief: "I monitor the water filling."
  ← the hand is incidental; the watching is the action.

BAD (over-compression — loses signal):
- "I read Claude explanations."            ← topic destroyed
- "I send a WeChat message."               ← message text and recipient destroyed
- "I interact with the interface."         ← everything destroyed

## Output Format
Return a JSON object:
{
  "scene_summary": "First-person one-sentence overview",
  "segments": [
    {
      "time_range": "HH:MM:SS–HH:MM:SS",
      "action": "I grab a coke from the fridge.",
      "action_brief": "I grab a coke.",
      "objects": ["specific objects I interact with or look at"],
      "environment": "FIRST segment: full establishing world state per Rule 4 (physical space, every visible object with state and position, complete screen content of every display, world text/signage, media playing, ambient audio). LATER segments: ONLY what changed since the previous segment, ignoring changes caused merely by head movement; 'No change.' is valid.",
      "text_visible": ["meaningful readable text I am attending to — skip trivial keyboard/packaging labels"],
      "speech": "exact quote of what I or others say in this segment, or null",
      "details": "fine-grained context: which hand, which direction, micro-actions"
    }
  ],
  "activity_chain": "comma-separated first-person atomic actions"
}

## Segmentation Rules
- Each segment = one meaningful action unit. Merge continuous/repetitive actions into one segment.
- A new segment starts when: a NEW action begins, I shift attention significantly, a new UI element appears, or someone speaks.
- Speech segments are separate from action segments. Every utterance MUST appear with exact quote — but omit noise/fragments.
- Environmental observation segments ("I see...", "I hear...") are separate from action segments.
- Media content (videos I watch, audio I hear) → environment segments with specific content details.
- Typical: 5-12 segments for a 30-second clip. Don't over-segment static or repetitive scenes."""

REGION_ADDENDUM = """

## ADDITIONAL OUTPUT — TEXT-DENSE REGION BOXES

Besides the caption, you MUST also locate every region containing DENSE READABLE TEXT that
deserves a zoomed-in second look: computer/laptop screens, phone screens, tablets, books,
documents, printed pages, menu boards, kiosk touchscreens, dense signage.

Add a top-level "text_regions" array to your JSON output:

"text_regions": [
  {"frame_index": 0, "hkt": "HH:MM:SS", "label": "laptop screen", "box_2d": [y0, x0, y1, x1]}
]

- frame_index is the 0-based index of the frame in the order given.
- box_2d uses NORMALIZED coordinates 0-1000 as [top, left, bottom, right].
- At most 2 regions per frame — the most text-dense ones.
- Include a region only when text is actually present and worth reading.
- If a frame has no text-dense region, simply omit that frame from the array.

Return the caption fields AND text_regions in the SAME JSON object."""

READ_PROMPT = """You are refining a first-person (egocentric) video caption using zoomed-in crops
of the text-dense regions the wearer looked at. Each crop is a screen, document, menu, or sign,
given in chronological order with its absolute HKT timestamp.

You are given the FIRST-PASS CAPTION below. It was written from downscaled full frames, so its
author could not read these screens. Your job is to read them and then CORRECT AND ENRICH that
caption with what the screens actually say.

## Step 1 — read every crop
For each crop, transcribe ALL meaningful readable text: menu items with prices, code lines,
chat messages, article titles, UI options and button labels, form fields, headings.
Preserve the original language (Chinese stays Chinese). Keep code indentation.
Skip trivial text: individual keyboard key letters, watermarks, generic product packaging,
OS menu-bar clock/battery.

## Step 2 — revise the caption segments
Using what you just read, rewrite each segment of the first-pass caption:
- "action": correct it to what the screen proves I was actually doing. If the first pass said
  "I browse the menu" and the screen shows me on the payment confirmation page, fix it.
  Name the concrete target: which item, which button, which file, which message.
- "environment": keep the first-pass structure — the FIRST segment holds the full establishing
  world state, every LATER segment holds ONLY what changed since the previous segment
  ("No change." is valid). Your job here is to correct and sharpen it with what the screens
  actually say: fix wrong screen content in the establishing shot, and make each delta name the
  real on-screen change (which page it navigated to, which message arrived, which item was
  selected). Do NOT expand the deltas back into full state dumps, and do not report changes that
  are only my head moving.
- "text_visible": the meaningful text actually readable at that moment, from your crop reading.
- Keep "time_range", "speech" and "objects" from the first-pass segment unless the screens prove
  them wrong. Never invent speech.
- Keep the same number of segments and the same time ranges as the first-pass caption.

## Privacy
Apply the same masking as the caption: real names/usernames -> "User X", school/university names
-> "School X"/"University X", emails -> "email_x@example.com", phone numbers -> "XXX-XXXX-XXXX",
addresses -> "Address X". Keep generic brand names.

Return ONLY JSON:
{
  "crops": [
    {"crop_index": 0, "hkt": "HH:MM:SS", "label": "kiosk screen",
     "text": ["line 1", "line 2"],
     "summary": "one sentence: what this screen shows"}
  ],
  "revised_segments": [
    {"time_range": "HH:MM:SS-HH:MM:SS",
     "action": "corrected first-person action naming the concrete on-screen target",
     "objects": ["..."],
     "environment": "exhaustive world state including full screen content",
     "text_visible": ["..."],
     "speech": "unchanged from first pass, or null",
     "details": "...",
     "revision_note": "what this segment changed vs the first pass, or 'unchanged'"}
  ]
}"""

SPEECH_PROMPT = """STEP 1 — Transcribe this audio VERBATIM. Write down only what you actually hear.
Do not invent, complete, or imagine dialogue. This is an isolated field recording from smart
glasses, NOT a scripted conversation. If a fragment is unintelligible, write "[unclear]".
Speech may come from the wearer or from other people nearby. The recording is noisy — background
chatter, machines and footsteps are common, so listen carefully for actual words.
{hint}
STEP 2 — An energy detector found these sound-activity regions. Their TIMINGS are authoritative
(accurate to ~0.03s) — never change them. Some regions contain only noise, not speech:
{windows}

Assign your STEP 1 utterances to these windows in chronological order. If a window contains no
intelligible speech (machine noise, footsteps, door sounds, ambient chatter), set text to
"[non-speech]". Never answer with a refusal such as "I'm not sure" — use "[non-speech]" instead.

Clip starts at {clip_start} HKT, so window offset t seconds means HKT = clip_start + t.

Return ONLY JSON:
{{"step1_raw_transcript": "...",
  "utterances": [{{"window": 0, "start_sec": 0.0, "end_sec": 0.0, "hkt": "HH:MM:SS",
                   "speaker": "wearer|other|unknown", "text": "verbatim words"}}]}}"""

REFUSAL_RE = re.compile(
    r"^\s*(i'?m not sure|i am not sure|unclear|inaudible|no speech|n/?a|none|"
    r"\[non-speech\]|\[unclear\]|cannot|can'?t determine|不确定|听不清)\s*[.。!！]?\s*$",
    re.I)

_FILLER_RE = re.compile(r"^[\s。，、．,.!！?？~～…\-—嗯哦啊呃唔呀哈噢欸诶,]*$")

# ====================================================================
#  5. Remote Tar Access
# ====================================================================


def parse_header(blk: bytes):
    if len(blk) < BLOCK or blk[257:262] != b"ustar":
        return None
    name = blk[0:100].rstrip(b"\0").decode("utf-8", "replace")
    try:
        size = int(blk[124:136].rstrip(b"\0 ").decode() or "0", 8)
    except ValueError:
        return None
    return {"name": name, "size": size, "type": chr(blk[156])}


def _align(n: int) -> int:
    return ((n + BLOCK - 1) // BLOCK) * BLOCK


def scan_headers(buf: bytes, base: int = 0):
    out = []
    for i in range(0, max(0, len(buf) - BLOCK + 1), BLOCK):
        h = parse_header(buf[i:i + BLOCK])
        if h:
            h["offset"] = base + i
            h["data_offset"] = base + i + BLOCK
            out.append(h)
    return out


def resolve_longlink(entries, buf: bytes, base: int):
    out = []
    pending = None
    for h in entries:
        if h["type"] == "L":
            s = h["data_offset"] - base
            pending = buf[s:s + h["size"]].rstrip(b"\0").decode("utf-8", "replace")
            continue
        if pending:
            h = dict(h, name=pending)
            pending = None
        out.append(h)
    return out


class HttpRangeSource:
    seekable = False

    def __init__(self, path_in_repo: str, repo: str = REPO, token: str = None,
                 size: int = None, retries: int = 5):
        import requests
        self.requests = requests
        self.path = path_in_repo
        self.repo = repo
        self.token = token or os.environ.get("HF_TOKEN") or ""
        if not self.token:
            try:
                from huggingface_hub import get_token
                self.token = get_token() or ""
            except Exception:
                pass
        self.retries = retries
        self.base_url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path_in_repo}"
        self._sess = requests.Session()
        self._cdn = None
        self._cdn_at = 0.0
        self.size = size if size is not None else self._head_size()

    def _auth(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _head_size(self):
        r = self._sess.head(self.base_url, headers=self._auth(),
                            allow_redirects=True, timeout=60)
        r.raise_for_status()
        self._cdn, self._cdn_at = r.url, time.time()
        return int(r.headers["Content-Length"])

    def _url(self):
        if self._cdn is None or time.time() - self._cdn_at > 600:
            try:
                r = self._sess.head(self.base_url, headers=self._auth(),
                                    allow_redirects=True, timeout=60)
                self._cdn, self._cdn_at = r.url, time.time()
            except Exception:
                return self.base_url
        return self._cdn

    def read(self, off: int, n: int) -> bytes:
        if n <= 0:
            return b""
        end = min(off + n, self.size) - 1
        if end < off:
            return b""
        last = None
        for attempt in range(self.retries):
            url = self._url()
            hdr = {"Range": f"bytes={off}-{end}"}
            if url == self.base_url or "huggingface.co" in url:
                hdr.update(self._auth())
            try:
                r = self._sess.get(url, headers=hdr, timeout=300)
                if r.status_code in (403, 401):
                    self._cdn = None
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                data = r.content
                if len(data) != end - off + 1:
                    raise RuntimeError(f"short read {len(data)} != {end - off + 1}")
                return data
            except Exception as e:
                last = e
                self._cdn = None
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"range read failed off={off} n={n}: {last}")

    def close(self):
        try:
            self._sess.close()
        except Exception:
            pass


class SeqStream:
    def __init__(self, source, start: int = 0, chunk: int = 8 << 20, ahead: int = 8):
        self.src = source
        self.pos = start
        self.chunk = chunk
        self.ahead = max(1, ahead if not source.seekable else 2)
        self.ex = ThreadPoolExecutor(self.ahead, thread_name_prefix="tarfetch")
        self.futs = {}

    def _cidx(self, p):
        return p // self.chunk

    def _schedule(self):
        c0 = self._cidx(self.pos)
        for c in list(self.futs):
            if c < c0:
                self.futs.pop(c).cancel()
        for c in range(c0, c0 + self.ahead):
            off = c * self.chunk
            if off >= self.src.size:
                break
            if c not in self.futs:
                n = min(self.chunk, self.src.size - off)
                self.futs[c] = self.ex.submit(self.src.read, off, n)

    def read(self, n: int) -> bytes:
        out = bytearray()
        while n > 0 and self.pos < self.src.size:
            self._schedule()
            c = self._cidx(self.pos)
            if c not in self.futs:
                break
            buf = self.futs[c].result()
            s = self.pos - c * self.chunk
            take = min(n, len(buf) - s)
            if take <= 0:
                break
            out += buf[s:s + take]
            self.pos += take
            n -= take
        return bytes(out)

    def skip(self, n: int):
        self.pos += n

    def seek(self, pos: int):
        self.pos = pos

    def close(self):
        for f in self.futs.values():
            f.cancel()
        self.futs.clear()
        self.ex.shutdown(wait=False, cancel_futures=True)


def iter_members(source, start: int = 0, want=None, end: int = None,
                 chunk: int = 8 << 20, ahead: int = 8, with_offset: bool = False):
    limit = source.size if end is None else min(end, source.size)
    st = SeqStream(source, start, chunk=chunk, ahead=ahead)
    try:
        pending_name = None
        while st.pos < limit:
            blk = st.read(BLOCK)
            if len(blk) < BLOCK:
                break
            h = parse_header(blk)
            if h is None:
                if blk.strip(b"\0") == b"":
                    break
                continue
            name, size, typ = h["name"], h["size"], h["type"]
            if typ == "L":
                raw = st.read(_align(size))
                pending_name = raw[:size].rstrip(b"\0").decode("utf-8", "replace")
                continue
            if pending_name:
                name, pending_name = pending_name, None
            padded = _align(size)
            off = st.pos
            if size and want and want(name):
                data = st.read(size)
                st.skip(padded - size)
                yield (name, size, data, off) if with_offset else (name, size, data)
            else:
                st.skip(padded)
                yield (name, size, None, off) if with_offset else (name, size, None)
    finally:
        st.close()


def walk_headers(source, start: int = 0, end: int = None, window: int = 64 << 10):
    limit = source.size if end is None else min(end, source.size)
    pos = start
    base, buf = -1, b""
    pending_name = None

    def ensure(p, need):
        nonlocal base, buf
        if base <= p and p + need <= base + len(buf):
            return True
        base = p
        buf = source.read(base, max(window, need))
        return len(buf) >= need

    while pos < limit:
        if not ensure(pos, BLOCK):
            break
        blk = buf[pos - base:pos - base + BLOCK]
        h = parse_header(blk)
        pos += BLOCK
        if h is None:
            if blk.strip(b"\0") == b"":
                break
            continue
        name, size, typ = h["name"], h["size"], h["type"]
        padded = _align(size)
        if typ == "L":
            if not ensure(pos, padded):
                break
            raw = buf[pos - base:pos - base + size]
            pending_name = raw.rstrip(b"\0").decode("utf-8", "replace")
            pos += padded
            continue
        if pending_name:
            name, pending_name = pending_name, None
        yield name, size, pos
        pos += padded


def read_tail_entries(source, n: int = 8 << 20):
    n = min(n, source.size)
    base = source.size - n
    buf = source.read(base, n)
    return resolve_longlink(scan_headers(buf, base), buf, base), buf, base


def find_boundary(source, pred_left, probe: int = 4 << 20, max_steps: int = 20):
    lo, hi = 0, source.size
    for _ in range(max_steps):
        if hi - lo < probe:
            break
        mid = ((lo + hi) // 2) // BLOCK * BLOCK
        buf = source.read(mid, probe)
        hs = scan_headers(buf, mid)
        if pred_left(hs):
            lo = mid
        else:
            hi = mid
    return lo, hi


def locate_sections(source, probe: int = 4 << 20):
    def still_eye(hs):
        return any("eye_tracking/" in h["name"] and h["name"].endswith(".jpg") for h in hs)

    lo, _ = find_boundary(source, still_eye, probe=probe)
    wav = tr = None
    off = lo
    for _ in range(6):
        buf = source.read(off, probe)
        hs = resolve_longlink(scan_headers(buf, off), buf, off)
        for h in hs:
            if h["name"].endswith("audio/anonymized.wav"):
                wav = h
            elif h["name"].endswith("audio/transcript.txt"):
                tr = h
        if wav:
            break
        off += probe - BLOCK
    if wav is None:
        raise RuntimeError("找不到 audio/anonymized.wav")
    if tr is None:
        after = wav["data_offset"] + _align(wav["size"])
        buf = source.read(after, 256 * 1024)
        hs = resolve_longlink(scan_headers(buf, after), buf, after)
        for h in hs:
            if h["name"].endswith("audio/transcript.txt"):
                tr = h
                break
    pictures_start = wav["data_offset"] + _align(wav["size"])
    return {"wav": wav, "transcript": tr, "pictures_start": pictures_start}


# ====================================================================
#  6. Frame Rendering
# ====================================================================


def frame_time(name: str):
    m = FRAME_RE.search(os.path.basename(name))
    if not m:
        return None
    return datetime.strptime(
        f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}", "%Y-%m-%d %H:%M:%S")


def frame_index(name: str):
    m = FRAME_RE.search(os.path.basename(name))
    return int(m.group(5)) if m else None


def hkt_hms(name: str) -> str:
    t = frame_time(name)
    return t.strftime("%H:%M:%S") if t else ""


def build_trails(gaze_rows: dict):
    out = {}
    hist = []
    for name in sorted(gaze_rows, key=lambda n: (frame_index(n) if frame_index(n) is not None else 0)):
        r = gaze_rows[name]
        valid = bool(r.get("gaze_x") and r.get("in_bounds") == "True")
        if valid:
            hist.append((float(r["gaze_x"]), float(r["gaze_y"])))
            if len(hist) > TRAIL_POINTS + 1:
                del hist[:-(TRAIL_POINTS + 1)]
        out[name] = (list(hist) if valid else [], valid)
    return out


def render_frame(jpg_bytes: bytes, points, valid: bool,
                 size: int = MAX_DIM, quality: int = JPEG_QUALITY,
                 as_image: bool = False):
    im = Image.open(io.BytesIO(jpg_bytes))
    im.draft("RGB", (size, size))
    im = im.convert("RGB")
    if max(im.size) != size:
        im = im.resize((size, size), Image.LANCZOS)
    scale = im.width / NATIVE_DIM
    draw = ImageDraw.Draw(im)
    if valid and points:
        pts = [(x * scale, y * scale) for x, y in points]
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                alpha = (i + 1) / len(pts)
                w = max(1, round(6 * alpha * scale))
                c = tuple(int(v * alpha) for v in TRAIL_COLOR)
                draw.line([pts[i], pts[i + 1]], fill=c, width=w)
        cx, cy = pts[-1]
        r = POINT_RADIUS * scale
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     outline=GAZE_COLOR, width=max(1, round(3 * scale)))
        d = 4 * scale
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], fill=GAZE_COLOR)
    elif points is not None:
        a, b = 10 * scale, 50 * scale
        w = max(1, round(3 * scale))
        draw.rectangle([a, a, b, b], outline=GAZE_INVALID_COLOR, width=w)
        draw.line([a, a, b, b], fill=GAZE_INVALID_COLOR, width=max(1, round(2 * scale)))
        draw.line([b, a, a, b], fill=GAZE_INVALID_COLOR, width=max(1, round(2 * scale)))
    if as_image:
        return im
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


def recompress(jpg_bytes: bytes, size: int, quality: int) -> bytes:
    im = Image.open(io.BytesIO(jpg_bytes))
    im.draft("RGB", (size, size))
    im = im.convert("RGB")
    if max(im.size) != size:
        im = im.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()


def clean_ocr(text: str) -> str:
    lines = [l for l in text.split("\n") if not l.startswith(OCR_SKIP_PREFIX)]
    return "\n".join(lines).strip()


def parse_transcript(text: str):
    out = []
    for line in text.splitlines():
        m = TR_RE.match(line)
        if m:
            a = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            b = datetime.strptime(m.group(2), "%Y-%m-%d %H:%M:%S")
            out.append((a, b, line))
            continue
        m = TR_REL_RE.match(line)
        if m:
            def _sec(h, mi, s):
                return int(h) * 3600 + int(mi) * 60 + int(s or 0)
            a = TR_REL_BASE + timedelta(seconds=_sec(*m.group(1, 2, 3)))
            b = TR_REL_BASE + timedelta(seconds=_sec(*m.group(4, 5, 6)))
            out.append((a, b, line))
    return out


def transcript_for_window(intervals, rel_start: float, dur: float):
    if not intervals:
        return []
    base = intervals[0][0]
    w0 = base + timedelta(seconds=rel_start)
    w1 = w0 + timedelta(seconds=dur)
    return [line for a, b, line in intervals if a < w1 and b > w0]


def build_context_text(sd: dict) -> str:
    first_hkt = sd["frames"][0]["hkt"] if sd["frames"] else "?"
    last_hkt = sd["frames"][-1]["hkt"] if sd["frames"] else "?"
    ocr_block = "\n".join(sd["ocr_texts"]) if sd["ocr_texts"] else "(no text detected)"
    if len(ocr_block) > 30000:
        ocr_block = ocr_block[:30000] + "\n... (OCR truncated, remaining frames omitted)"
    return f"""## Scene: 30-second first-person video clip
- Date: {sd['clip_start'][:10] if sd['clip_start'] else 'unknown'}
- Absolute time range: {sd['clip_start']} to {sd['clip_end']} HKT
- Frame count: {sd['n_frames']} frames at 1fps
- Frame timestamps: {first_hkt} to {last_hkt} HKT
- IMPORTANT: Use these absolute HKT timestamps (HH:MM:SS format) for all time_range fields

## Audio Transcript
{sd['transcript']}

## OCR-detected Text (per-frame, with HKT timestamps)
{ocr_block}

## Video Frames (all {sd['n_frames']} frames, 1 second apart, chronologically ordered)"""


def build_parts_vertex(sd: dict, system_prompt: str):
    from google import genai
    context = build_context_text(sd)
    parts = [genai.types.Part.from_text(text=system_prompt + "\n\n" + context)]
    for frame in sd["frames"]:
        parts.append(genai.types.Part.from_text(text=f"[{frame['hkt']} HKT — {frame['name']}]"))
        parts.append(genai.types.Part.from_bytes(data=frame["jpg"], mime_type="image/jpeg"))
    parts.append(genai.types.Part.from_text(
        text="Now produce the fine-grained, first-person dense temporal caption JSON "
             "with absolute HKT timestamps."))
    return parts


def payload_bytes(sd: dict) -> int:
    return sum(len(f["jpg"]) for f in sd["frames"])


def enforce_inline_limit(sd: dict, limit: int = INLINE_LIMIT, log=print):
    dim = DOWNGRADE_DIMS[0]
    for d in DOWNGRADE_DIMS[1:]:
        if payload_bytes(sd) <= limit:
            break
        dim = d
        log(f"    payload {payload_bytes(sd)/1e6:.1f}MB > 上限，降到 {d}px 重压")
        for f in sd["frames"]:
            f["jpg"] = recompress(f["jpg"], d, JPEG_QUALITY)
    return dim


def make_client(project: str = "", location: str = "global",
                service_tier: str = DEFAULT_SERVICE_TIER,
                request_timeout_sec: int = DEFAULT_REQUEST_TIMEOUT_SEC):
    from google import genai
    from google.genai import types
    configure_google_environment(project, location)
    service_tier = (service_tier or DEFAULT_SERVICE_TIER).lower()
    if service_tier not in {"standard", "flex"}:
        raise ValueError(f"unsupported service tier: {service_tier}")
    if service_tier == "flex" and location != "global":
        raise ValueError("Flex PayGo requires GOOGLE_CLOUD_LOCATION=global")
    timeout_ms = max(1, min(int(request_timeout_sec), 1800)) * 1000
    client_kwargs = {
        "enterprise": True,
        "project": project,
        "location": location,
    }
    headers = {}
    if service_tier == "flex":
        headers = {
            "X-Vertex-AI-LLM-Request-Type": "shared",
            "X-Vertex-AI-LLM-Shared-Request-Type": "flex",
        }
    client_kwargs["http_options"] = types.HttpOptions(
        api_version="v1",
        headers=headers,
        timeout=timeout_ms,
        # The pipeline owns retries/checkpoints. Avoid hidden duplicate work.
        retry_options=types.HttpRetryOptions(attempts=1),
    )
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if api_key:
        # Passing project/location and the service-bound key only through the
        # environment makes google-genai prefer ADC and discard the key.
        client_kwargs["api_key"] = api_key
    return genai.Client(**client_kwargs)


def run_one(client, model: str, parts, retries: int = 3, log=print) -> dict:
    from google import genai
    last_err = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            resp = client.models.generate_content(
                model=model, contents=parts,
                config=genai.types.GenerateContentConfig(
                    temperature=TEMPERATURE, max_output_tokens=MAX_OUTPUT_TOKENS),
            )
            u = resp.usage_metadata
            return {
                "ok": True, "model": model, "api": "adc",
                "time": round(time.time() - t0, 1),
                "in": u.prompt_token_count or 0,
                "out": u.candidates_token_count or 0,
                "think": getattr(u, "thoughts_token_count", 0) or 0,
                "content": resp.text or "",
                "attempts": attempt + 1,
            }
        except Exception as e:
            last_err = e
            msg = str(e)
            transient = any(s in msg for s in
                            ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED",
                             "UNAVAILABLE", "DEADLINE", "Timeout", "timed out",
                             "Connection", "reset"))
            if attempt == retries - 1 or not transient:
                break
            wait = min(2 ** attempt * 5, 60)
            log(f"    重试 {attempt+1}/{retries} ({msg[:80]}) {wait}s 后")
            time.sleep(wait)
    return {"ok": False, "model": model, "api": "adc", "error": str(last_err)[:600]}


def parse_caption_json(text: str):
    if not text:
        return None, "empty"
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1).strip()
    i = s.find("{")
    if i > 0:
        s = s[i:]
    try:
        return json.loads(s), "ok"
    except json.JSONDecodeError:
        pass
    seg = s.find('"segments"')
    if seg >= 0:
        last = s.rfind("}")
        while last > seg:
            cand = s[:last + 1]
            for tail in ("]}", "}]}", '"}]}'):
                try:
                    return json.loads(cand + tail), "salvaged"
                except json.JSONDecodeError:
                    continue
            last = s.rfind("}", 0, last)
    return None, "unparseable"


# ====================================================================
#  7. V9 Caption Logic
# ====================================================================


_silero = None


def _get_silero():
    global _silero
    with _silero_lock:
        if _silero is None:
            if not _HAS_TORCH:
                return None
            try:
                import torch as _torch
                _torch.set_num_threads(1)
                from silero_vad import load_silero_vad, get_speech_timestamps
                _silero = (load_silero_vad(), get_speech_timestamps)
            except ImportError:
                return None
        return _silero


def _read_mono_bytes(wav_bytes_data: bytes):
    w = wave.open(io.BytesIO(wav_bytes_data))
    sr, ch = w.getframerate(), w.getnchannels()
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    w.close()
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def detect_speech_windows(wav_bytes_data: bytes, min_speech=0.25, merge_gap=0.40, max_windows=24):
    try:
        silero = _get_silero()
        if silero is None:
            a, sr = _read_mono_bytes(wav_bytes_data)
            return [], len(a) / sr if sr else 0.0
        model, get_ts = silero
        import torch as _torch
        a, sr = _read_mono_bytes(wav_bytes_data)
        dur = len(a) / sr if sr else 0.0
        if sr != 16000:
            idx = (np.arange(int(len(a) * 16000 / sr)) * sr / 16000).astype(int)
            a = a[np.clip(idx, 0, len(a) - 1)]
        wav = _torch.from_numpy(a / 32768.0).float()
        with _silero_lock:
            try:
                model.reset_states()
            except Exception:
                pass
            with _torch.no_grad():
                stamps = get_ts(wav, model, sampling_rate=16000, return_seconds=True,
                                min_speech_duration_ms=int(min_speech * 1000),
                                min_silence_duration_ms=200, speech_pad_ms=120)
        segs = [[float(s["start"]), float(s["end"])] for s in stamps]
    except Exception as e:
        _log(f"    silero-vad failed ({str(e)[:80]}), no speech windows")
        try:
            a, sr = _read_mono_bytes(wav_bytes_data)
            return [], len(a) / sr if sr else 0.0
        except Exception:
            return [], 0.0

    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    kept = [s for s in merged if s[1] - s[0] >= min_speech]
    if len(kept) > max_windows:
        kept = sorted(sorted(kept, key=lambda s: s[1] - s[0], reverse=True)[:max_windows])
    return kept, dur


def _transcript_is_substantive(transcript):
    body = re.sub(r"\[[^\]]*\]", " ", transcript or "")
    body = re.sub(r"\s+", "", body)
    return len(body) >= 4 and not _FILLER_RE.match(body)


def _energy_windows(wav_bytes_data: bytes, frame_ms=30, min_speech=0.30,
                    merge_gap=0.50, max_windows=12):
    try:
        a, sr = _read_mono_bytes(wav_bytes_data)
    except Exception:
        return []
    n = max(1, int(sr * frame_ms / 1000))
    nf = len(a) // n
    if nf < 2:
        return []
    e = np.array([np.sqrt((a[i * n:(i + 1) * n] ** 2).mean() + 1e-9) for i in range(nf)])
    if e.max() <= 0:
        return []
    med, p95 = float(np.median(e)), float(np.percentile(e, 95))
    thr = med + 0.45 * max(p95 - med, 1e-6)
    segs, st = [], None
    for i, v in enumerate(e > thr):
        if v and st is None:
            st = i
        elif not v and st is not None:
            segs.append([st * frame_ms / 1000, i * frame_ms / 1000])
            st = None
    if st is not None:
        segs.append([st * frame_ms / 1000, nf * frame_ms / 1000])
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    kept = [s for s in merged if s[1] - s[0] >= min_speech]
    if len(kept) > max_windows:
        def loud(s):
            i0, i1 = int(s[0] * 1000 / frame_ms), int(s[1] * 1000 / frame_ms)
            return float(e[i0:i1].mean()) if i1 > i0 else 0.0
        kept = sorted(sorted(kept, key=loud, reverse=True)[:max_windows])
    return kept


def hkt_plus(clip_start_hkt, offset_sec):
    m = re.search(r"(\d{2}):(\d{2}):(\d{2})", clip_start_hkt or "")
    if not m:
        return ""
    base = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    t = int(base + offset_sec) % 86400
    return f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"


def parse_json(text):
    t = text or ""
    if "```json" in t:
        t = t.split("```json")[1].split("```")[0]
    elif "```" in t:
        t = t.split("```")[1].split("```")[0]
    try:
        return json.loads(t.strip())
    except Exception:
        pass
    i = t.rfind("}")
    if i > 0:
        for sfx in ("", "]}", "}]}", "\"}]}", "\"}]}]}"):
            try:
                return json.loads(t[:i + 1].strip() + sfx)
            except Exception:
                continue
    return None


def generate(client, parts, tag, max_out=16384, retries=INLINE_RETRIES, stream=False,
             log=None):
    """Call Gemini with classified retries and secret-safe diagnostics."""
    _l = log or _log
    started = time.time()
    last = None
    last_type = "other"
    attempts = transient_attempts = quota_attempts = 0
    # A 30-minute Flex request must not be silently duplicated. A later Job can
    # resume from checkpoints if it times out; Standard retains short retries.
    transient_limit = 1 if _SERVICE_TIER == "flex" else retries
    quota_limit = 1 if _SERVICE_TIER == "flex" else QUOTA_RETRIES
    _l(f"    {tag}: request start model={_MODEL} parts={len(parts)} "
       f"max_out={max_out} stream={stream}")

    while True:
        attempts += 1
        try:
            from google.genai import types
            cfg = types.GenerateContentConfig(
                **generation_config_kwargs(_MODEL, max_out)
            )
            if stream:
                chunks, u = [], None
                for ch in client.models.generate_content_stream(
                        model=_MODEL, contents=parts, config=cfg):
                    if getattr(ch, "text", None):
                        chunks.append(ch.text)
                    if getattr(ch, "usage_metadata", None):
                        u = ch.usage_metadata
                if not chunks:
                    raise RuntimeError("empty stream")
                content = "".join(chunks)
            else:
                r = client.models.generate_content(
                    model=_MODEL, contents=parts, config=cfg
                )
                u = getattr(r, "usage_metadata", None)
                content = getattr(r, "text", "") or ""

            elapsed = round(time.time() - started, 1)
            usage_in = (getattr(u, "prompt_token_count", 0) or 0) if u else 0
            usage_out = (getattr(u, "candidates_token_count", 0) or 0) if u else 0
            usage_think = (getattr(u, "thoughts_token_count", 0) or 0) if u else 0
            traffic = getattr(u, "traffic_type", None) if u else None
            traffic = getattr(traffic, "value", traffic)
            traffic = str(traffic or "UNKNOWN")
            _l(f"    {tag}: request ok attempt={attempts} elapsed={elapsed}s "
               f"in={usage_in} out={usage_out} think={usage_think} "
               f"traffic={traffic}")
            return {
                "ok": True,
                "time": elapsed,
                "attempts": attempts,
                "in": usage_in,
                "out": usage_out,
                "think": usage_think,
                "traffic_type": traffic,
                "content": content,
            }
        except Exception as e:
            last_type = classify_api_error(e)
            last = redact_secrets(e, limit=800)
            elapsed = round(time.time() - started, 1)
            _l(f"    {tag}: request failed attempt={attempts} type={last_type} "
               f"elapsed={elapsed}s error={last}")

            if not should_retry_api_error(last_type):
                break
            if last_type == "quota":
                quota_attempts += 1
                if quota_attempts >= quota_limit:
                    break
                delay = (QUOTA_BACKOFF[min(quota_attempts - 1, len(QUOTA_BACKOFF) - 1)]
                         + random.uniform(0, 20))
            else:
                transient_attempts += 1
                if transient_attempts >= transient_limit:
                    break
                delay = 4 * transient_attempts
            _l(f"    {tag}: retrying type={last_type} after {delay:.1f}s")
            time.sleep(delay)

    return {
        "ok": False,
        "error": last,
        "error_type": last_type,
        "attempts": attempts,
        "time": round(time.time() - started, 1),
    }


def _get_gcs():
    global _gcs_client
    if not _HAS_GCS:
        return None
    with _gcs_lock:
        if _gcs_client is None:
            _gcs_client = _gcs_mod.Client()
        return _gcs_client


def _upload_gcs(blobs, prefix, workers=8):
    client = _get_gcs()
    if client is None:
        return None
    import concurrent.futures as cf
    bucket = client.bucket(_GCS_BUCKET)

    def up(item):
        i, (name, data) = item
        b = bucket.blob(f"{prefix}/{i:03d}_{name}")
        b.upload_from_string(data, content_type="image/jpeg")
        return f"gs://{_GCS_BUCKET}/{prefix}/{i:03d}_{name}"

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(up, enumerate(blobs)))


def call_with_fallback(client, text_parts, blobs, res_level, tag, gcs_prefix,
                       max_out=16384, log=None):
    from google.genai import types
    _l = log or _log
    total_mb = sum(len(b) for _, b in blobs) / 1024 / 1024

    def build(uris=None):
        parts = []
        for i, (txt, blob) in enumerate(zip(text_parts, blobs)):
            parts.append(types.Part.from_text(text=txt))
            if uris is None:
                parts.append(types.Part.from_bytes(data=blob[1], mime_type="image/jpeg",
                                                   media_resolution=res_level))
            else:
                parts.append(types.Part.from_uri(file_uri=uris[i], mime_type="image/jpeg",
                                                 media_resolution=res_level))
        return parts

    if total_mb <= MAX_INLINE_MB:
        r = generate(client, build(), tag, max_out, log=_l)
        if r.get("ok"):
            r["transport"] = "inline"
            r["payload_mb"] = round(total_mb, 1)
            return r
        r["transport"] = "inline"
        r["payload_mb"] = round(total_mb, 1)
        _l(f"    {tag}: inline failed attempts={r.get('attempts', 1)} "
           f"type={r.get('error_type', 'other')} payload={total_mb:.1f}MB")
        # Resolution/GCS fallbacks can only solve payload-size failures.
        if r.get("error_type") != "payload":
            return r

    # try GCS fallback if available
    if _HAS_GCS:
        if total_mb > MAX_INLINE_MB:
            _l(f"    {tag}: {total_mb:.1f}MB > {MAX_INLINE_MB}MB ceiling -> GCS")
        try:
            t0 = time.time()
            uris = _upload_gcs(blobs, gcs_prefix)
            if uris is not None:
                up = round(time.time() - t0, 1)
                r = generate(client, build(uris), tag, max_out, retries=2, log=_l)
                r["transport"] = "gcs"
                r["upload_time"] = up
                r["payload_mb"] = round(total_mb, 1)
                if r.get("ok") or r.get("error_type") != "payload":
                    return r
        except Exception as e:
            _l(f"    {tag}: GCS fallback failed: {redact_secrets(e, limit=300)}")

    # no GCS — resolution downgrade fallback
    for dim in [1440, 1200, 1024, 768]:
        new_blobs = []
        for name, data in blobs:
            new_blobs.append((name, recompress(data, dim, JPEG_QUALITY)))
        new_mb = sum(len(b) for _, b in new_blobs) / 1024 / 1024
        if new_mb <= MAX_INLINE_MB:
            _l(f"    {tag}: 降到 {dim}px ({new_mb:.1f}MB) 重试 inline")

            def build_low(bs=new_blobs):
                parts = []
                for i, (txt, blob) in enumerate(zip(text_parts, bs)):
                    parts.append(types.Part.from_text(text=txt))
                    parts.append(types.Part.from_bytes(data=blob[1], mime_type="image/jpeg",
                                                       media_resolution=res_level))
                return parts

            r = generate(client, build_low(), tag, max_out, log=_l)
            r["transport"] = f"inline-{dim}px"
            r["payload_mb"] = round(new_mb, 1)
            return r
    return {"ok": False, "error": f"payload too large ({total_mb:.1f}MB) and no fallback succeeded",
            "error_type": "payload", "attempts": 0, "transport": "failed",
            "payload_mb": round(total_mb, 1), "time": 0}


def encode_img(img: Image.Image, max_dim: int, q: int) -> bytes:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        r = max_dim / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=q)
    return b.getvalue()


def build_context_text_v9(sd: dict, use_transcript: bool, use_ocr: bool, speech) -> str:
    frames = sd["frames"]
    head = f"""## Scene: {len(frames)}-second first-person video clip
- Date: {sd['clip_start'][:10] if sd['clip_start'] else 'unknown'}
- Absolute time range: {sd['clip_start']} to {sd['clip_end']} HKT
- Frame count: {len(frames)} frames at 1fps
- Frame timestamps: {frames[0]['hkt'] if frames else '?'} to {frames[-1]['hkt'] if frames else '?'} HKT
- IMPORTANT: Use these absolute HKT timestamps (HH:MM:SS format) for all time_range fields"""
    parts = [head]

    if speech and speech.get("utterances"):
        lines = "\n".join(
            f"  {u['hkt']}–{u['hkt_end']} [{u['speaker']}] \"{u['text']}\""
            for u in speech["utterances"])
        parts.append("## Speech (voice-activity-detected windows, timings are AUTHORITATIVE)\n"
                     "These timings come from acoustic energy analysis and are accurate to ~0.1s.\n"
                     "Use them verbatim for speech segment time_range values.\n" + lines)
    elif use_transcript:
        parts.append("## Audio Transcript\n" + sd["transcript"])

    if use_ocr:
        ocr_block = "\n".join(sd["ocr_texts"]) if sd["ocr_texts"] else "(no text detected)"
        if len(ocr_block) > 30000:
            ocr_block = ocr_block[:30000] + "\n... (OCR truncated)"
        parts.append("## OCR-detected Text (per-frame, with HKT timestamps)\n" + ocr_block)

    if not use_transcript and not use_ocr:
        parts.append("## No transcript or OCR provided\n"
                     "Derive everything — actions, on-screen text, speech — from the frames alone.")

    parts.append(f"## Video Frames (all {len(frames)} frames, 1 second apart, "
                 f"chronologically ordered)")
    return "\n\n".join(parts)


def align_speech(client, wav_bytes_data: bytes, clip_start: str, transcript: str = "",
                 log=None):
    from google.genai import types
    windows, dur = detect_speech_windows(wav_bytes_data)
    source = "silero"
    if not windows and _transcript_is_substantive(transcript):
        windows = _energy_windows(wav_bytes_data)
        source = "energy-fallback"
    if not windows:
        return {"windows": 0, "utterances": [], "duration": round(dur, 1), "source": "silero"}
    wl = "\n".join(f"  window {i}: {s:.2f}s - {e:.2f}s (duration {e-s:.2f}s)"
                   for i, (s, e) in enumerate(windows))
    hint = ""
    if transcript.strip():
        hint = ("\nA separate ASR system produced this rough transcript of the same clip. Its "
                "timings are coarse 30-second blocks and some words may be misheard, but it "
                "tells you which words are likely present. Use it to guide your listening; "
                "correct it where you hear something different:\n"
                + transcript.strip()[:1500] + "\n")
    prompt = SPEECH_PROMPT.format(windows=wl, clip_start=clip_start or "unknown", hint=hint)
    r = generate(client,
                 [types.Part.from_bytes(data=wav_bytes_data, mime_type="audio/wav"),
                  types.Part.from_text(text=prompt)],
                 "speech", max_out=4096, retries=3, log=log)
    if not r.get("ok"):
        return {"windows": len(windows), "utterances": [], "error": r.get("error"),
                "duration": round(dur, 1)}
    j = parse_json(r["content"]) or {}
    utts = []
    for i, u in enumerate(j.get("utterances", [])):
        w = u.get("window", i)
        if not isinstance(w, int) or w >= len(windows):
            w = min(i, len(windows) - 1)
        s, e = windows[w]
        txt = (u.get("text") or "").strip()
        if not txt or REFUSAL_RE.match(txt):
            continue
        utts.append({"hkt": hkt_plus(clip_start, s), "hkt_end": hkt_plus(clip_start, e),
                     "start_sec": round(s, 2), "end_sec": round(e, 2),
                     "speaker": u.get("speaker", "unknown"), "text": txt})
    return {"windows": len(windows), "utterances": utts, "duration": round(dur, 1),
            "source": source, "raw": j.get("step1_raw_transcript", ""),
            "in": r.get("in", 0), "out": r.get("out", 0), "time": r.get("time", 0)}


def process_clip(client, sd: dict, system_prompt: str, wav16k: bytes | None,
                 clip_tag: str = "clip", use_transcript: bool = True, use_ocr: bool = True,
                 use_vad: bool = True, log=None):
    from google.genai import types
    ULTRA = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_ULTRA_HIGH
    HIGH = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
    _l = log or _log
    t_start = time.time()
    frames = sd["frames"]
    out = {"model": _MODEL, "n_frames": len(frames),
           "clip_start": sd.get("clip_start"), "clip_end": sd.get("clip_end")}

    # ── PASS 0: VAD 语音对齐 ──
    speech = None
    if use_vad and use_transcript and wav16k:
        try:
            speech = align_speech(client, wav16k, sd.get("clip_start"),
                                  sd.get("transcript", ""), log=_l)
            out["speech"] = speech
            _l(f"    {clip_tag} VAD {speech['windows']} windows -> "
               f"{len(speech.get('utterances', []))} utterances")
        except Exception as e:
            _l(f"    {clip_tag} VAD failed: {redact_secrets(e, limit=300)}")

    # ── PASS 1: 全量 caption + 区域框 ──
    sys_text = (system_prompt + REGION_ADDENDUM + "\n\n"
                + build_context_text_v9(sd, use_transcript, use_ocr, speech))
    texts, blobs = [], []
    for i, f in enumerate(frames):
        texts.append(f"[frame_index {i} — {f['hkt']} HKT — {f['name']}]")
        blobs.append((f["name"], f["jpg"]))
    texts[0] = sys_text + "\n\n" + texts[0]

    r1 = call_with_fallback(client, texts, blobs, HIGH, f"{clip_tag} pass1",
                            f"p1/{clip_tag}", max_out=16384, log=_l)
    j1 = parse_json(r1["content"]) if r1.get("ok") else None
    out["pass1"] = {k: r1.get(k) for k in
                    ("ok", "time", "in", "out", "think", "transport", "payload_mb",
                     "upload_time", "error", "error_type", "attempts", "traffic_type",
                     "content")}
    out["pass1"]["n_segments"] = len(j1.get("segments", [])) if j1 else 0
    regions = j1.get("text_regions", []) if j1 else []
    out["pass1"]["n_regions"] = len(regions)

    if not r1.get("ok"):
        _l(f"    {clip_tag} pass1 failed type={r1.get('error_type', 'other')}: "
           f"{redact_secrets(r1.get('error'), limit=500)}")
        out["total_time"] = round(time.time() - t_start, 1)
        return out
    _l(f"    {clip_tag} pass1 ok {r1['time']}s [{r1['transport']}] "
       f"segs={out['pass1']['n_segments']} regions={len(regions)} "
       f"payload={r1.get('payload_mb')}MB")

    # ── crop from in-memory 2880px frames ──
    crops = []
    seen = set()
    for reg in regions:
        idx = reg.get("frame_index")
        box = reg.get("box_2d")
        if not isinstance(idx, int) or idx < 0 or idx >= len(frames) or not box or len(box) != 4:
            continue
        src = Image.open(io.BytesIO(frames[idx]["jpg"]))
        W, H = src.size
        y0, x0, y1, x1 = box
        l, t = int(x0 / 1000 * W), int(y0 / 1000 * H)
        rr, b = int(x1 / 1000 * W), int(y1 / 1000 * H)
        pw, ph = int(W * .03), int(H * .03)
        l, t = max(0, l - pw), max(0, t - ph)
        rr, b = min(W, rr + pw), min(H, b + ph)
        if rr - l < 80 or b - t < 80:
            continue
        key = (idx, l // 60, t // 60, rr // 60, b // 60)
        if key in seen:
            continue
        seen.add(key)
        crops.append({"hkt": reg.get("hkt") or frames[idx]["hkt"],
                      "label": reg.get("label", "region"),
                      "frame_index": idx, "img": src.crop((l, t, rr, b))})
    crops = crops[:MAX_CROPS]
    out["n_crops"] = len(crops)
    if not crops:
        _l(f"    {clip_tag}: no crops detected, pass2 skipped")
        out["total_time"] = round(time.time() - t_start, 1)
        return out

    # ── PASS 2: ULTRA_HIGH 精读 + 修订 pass1 ──
    p1_caption = {"scene_summary": j1.get("scene_summary", ""),
                  "segments": [{k: s.get(k) for k in
                                ("time_range", "action", "objects", "environment",
                                 "text_visible", "speech", "details")}
                               for s in j1.get("segments", [])]}
    p1_block = ("## FIRST-PASS CAPTION (to be corrected and enriched)\n"
                + json.dumps(p1_caption, ensure_ascii=False, indent=1))

    texts2, blobs2 = [], []
    for i, c in enumerate(crops):
        w, h = c["img"].size
        texts2.append(f"[crop_index {i} — {c['hkt']} HKT — {c['label']} — native {w}x{h}]")
        blobs2.append((f"c{i:02d}.jpg", encode_img(c["img"], CROP_DIM, CROP_Q)))
    texts2[0] = READ_PROMPT + "\n\n" + p1_block + "\n\n" + texts2[0]

    r2 = call_with_fallback(client, texts2, blobs2, ULTRA, f"{clip_tag} pass2",
                            f"p2/{clip_tag}", max_out=16384, log=_l)
    j2 = parse_json(r2["content"]) if r2.get("ok") else None
    out["pass2"] = {k: r2.get(k) for k in
                    ("ok", "time", "in", "out", "think", "transport", "payload_mb",
                     "upload_time", "error", "error_type", "attempts", "traffic_type",
                     "content")}
    n_text = sum(len(c.get("text", [])) for c in (j2.get("crops", []) if j2 else []))
    out["pass2"]["n_text_items"] = n_text

    if r2.get("ok"):
        _l(f"    {clip_tag} pass2 ok {r2['time']}s [{r2['transport']}] "
           f"crops={len(crops)} text_items={n_text} payload={r2.get('payload_mb')}MB")
    else:
        _l(f"    {clip_tag} pass2 failed type={r2.get('error_type', 'other')}: "
           f"{redact_secrets(r2.get('error'), limit=500)}")

    # ── merge: pass2 revisions overwrite pass1 ──
    if j1 and j2:
        by_hkt = {}
        for c in j2.get("crops", []):
            by_hkt.setdefault(c.get("hkt", ""), []).append(c)

        revised = j2.get("revised_segments") or []
        merged = json.loads(json.dumps(j1))
        merged.pop("text_regions", None)

        if revised:
            by_tr = {}
            for s in revised:
                key = re.sub(r"\s", "", s.get("time_range", ""))
                if key:
                    by_tr[key] = s
            base = merged.get("segments", [])
            new_segs = []
            for i, seg in enumerate(base):
                key = re.sub(r"\s", "", seg.get("time_range", ""))
                r = by_tr.get(key) or (revised[i] if i < len(revised) else None)
                if r:
                    seg = {**seg, **{k: v for k, v in r.items() if v}}
                new_segs.append(seg)
            merged["segments"] = new_segs
            out["pass2"]["n_revised"] = len(revised)

        for seg in merged.get("segments", []):
            m = re.findall(r"(\d{2}:\d{2}:\d{2})", seg.get("time_range", ""))
            if len(m) != 2:
                continue
            lo, hi = m
            detail = [{"hkt": h, "label": c.get("label"), "summary": c.get("summary"),
                       "text": c.get("text", [])}
                      for h, cs in by_hkt.items() if lo <= h <= hi for c in cs]
            if detail:
                seg["screen_text_detail"] = detail
        out["merged"] = merged

    out["total_time"] = round(time.time() - t_start, 1)
    return out


def is_complete(r: dict) -> bool:
    p1ok = bool((r.get("pass1") or {}).get("ok"))
    p2 = r.get("pass2")
    return p1ok and (r.get("n_crops", 0) == 0 or bool(p2 and p2.get("ok")))


# ====================================================================
#  8. Data Extraction
# ====================================================================


def read_tail_adaptive(src, want_frames_from_zero=True):
    n = 16 << 20
    while True:
        entries, buf, base = read_tail_entries(src, n)
        ocr = [e for e in entries if "/picture/ocr_text/" in e["name"]]
        idx = [frame_index(e["name"]) for e in ocr]
        idx = [i for i in idx if i is not None]
        covered = bool(idx) and min(idx) == 0
        if covered or not want_frames_from_zero or n >= src.size or n >= (512 << 20):
            return entries, buf, base
        n = min(n * 2, src.size)


def extract_tail_payload(entries, buf, base):
    gaze_rows, ocr_texts = {}, {}
    for e in entries:
        s = e["data_offset"] - base
        if s < 0 or s + e["size"] > len(buf):
            continue
        blob = buf[s:s + e["size"]]
        if e["name"].endswith("eye_tracking/gaze.csv"):
            for r in csv.DictReader(io.StringIO(blob.decode("utf-8", "replace"))):
                gaze_rows[r["frame_file"]] = r
        elif "/picture/ocr_text/" in e["name"] and e["name"].endswith(".txt"):
            i = frame_index(e["name"])
            if i is not None:
                ocr_texts[i] = (os.path.basename(e["name"])[:-4],
                                clean_ocr(blob.decode("utf-8", "replace")))
    return gaze_rows, ocr_texts


def parse_wav_header(head: bytes):
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise RuntimeError("不是 WAV")
    pos = 12
    fmt = None
    while pos + 8 <= len(head):
        cid = head[pos:pos + 4]
        csize = int.from_bytes(head[pos + 4:pos + 8], "little")
        body = pos + 8
        if cid == b"fmt ":
            fmt = {
                "channels": int.from_bytes(head[body + 2:body + 4], "little"),
                "rate": int.from_bytes(head[body + 4:body + 8], "little"),
                "bits": int.from_bytes(head[body + 14:body + 16], "little"),
            }
        elif cid == b"data":
            if not fmt:
                raise RuntimeError("data 在 fmt 之前")
            fmt["data_offset"] = body
            fmt["data_size"] = csize
            fmt["width"] = fmt["bits"] // 8
            fmt["frame_bytes"] = fmt["width"] * fmt["channels"]
            return fmt
        pos = body + csize + (csize & 1)
    raise RuntimeError("没找到 data 块")


def resample_to_16k(pcm: bytes, rate: int, channels: int, width: int) -> bytes:
    try:
        import audioop
        if channels > 1:
            pcm = audioop.tomono(pcm, width, 0.5, 0.5)
        out, _ = audioop.ratecv(pcm, width, 1, rate, 16000, None)
        return out
    except Exception:
        a = np.frombuffer(pcm, dtype="<i2")
        if channels > 1:
            a = a.reshape(-1, channels).mean(axis=1)
        k = max(1, round(rate / 16000))
        n = (len(a) // k) * k
        a = a[:n].reshape(-1, k).mean(axis=1)
        return a.astype("<i2").tobytes()


def wav_bytes(pcm16k: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16k)
    return buf.getvalue()


class AudioSlicer:
    def __init__(self, src, wav_entry):
        self.src = src
        self.base = wav_entry["data_offset"]
        self.size = wav_entry["size"]
        self.h = parse_wav_header(src.read(self.base, 4096))

    def slice16k(self, start_sec: float, dur_sec: float) -> bytes:
        h = self.h
        off = self.base + h["data_offset"] + int(start_sec * h["rate"]) * h["frame_bytes"]
        n = int(dur_sec * h["rate"]) * h["frame_bytes"]
        n = min(n, self.base + h["data_offset"] + h["data_size"] - off)
        if n <= 0:
            return b""
        pcm = self.src.read(off, n)
        return resample_to_16k(pcm, h["rate"], h["channels"], h["width"])


def locate_sections_safe(src):
    try:
        return locate_sections(src)
    except Exception as e:
        _log(f"locate_sections 失败({type(e).__name__}: {e})，退化到 walk_headers 通用扫描")
        wav = tr = pictures_start = None
        for name, size, off in walk_headers(src):
            if pictures_start is None and "/picture/masked/" in name:
                pictures_start = off
            elif name.endswith("audio/anonymized.wav"):
                wav = {"data_offset": off, "size": size}
            elif name.endswith("audio/transcript.txt"):
                tr = {"data_offset": off, "size": size}
            if wav and tr and pictures_start is not None:
                break
        if pictures_start is None:
            raise RuntimeError("walk_headers 兜底也没找到 picture/masked/") from e
        _log(f"  兜底定位: wav={'有' if wav else '无'} transcript={'有' if tr else '无'} "
             f"pictures_start={pictures_start}")
        return {"wav": wav, "transcript": tr, "pictures_start": pictures_start}


# ====================================================================
#  9. Merge
# ====================================================================


def pii_audit(segments: list) -> dict:
    blob = json.dumps(segments, ensure_ascii=False)
    out = {k: len(re.findall(p, blob)) for k, p in PII_PATTERNS.items()}
    out["anonymized_placeholders"] = len(re.findall(ANON_PATTERNS, blob))
    out["total_hits"] = sum(v for k, v in out.items() if k != "anonymized_placeholders")
    return out


def _seg_start(seg, fallback):
    m = TIME_RE.search(seg.get("time_range") or "")
    return m.group(0) if m else fallback


def merge_recording(clips: list, manifest: dict | None = None) -> dict:
    segments, summaries = [], []
    for c in clips:
        parsed = c.get("parsed") or {}
        base = (c.get("clip_start_hkt") or "")[11:] or "00:00:00"
        summaries.append({
            "clip_id": c.get("clip_id"),
            "time": f"{(c.get('clip_start_hkt') or '')[11:]}–{(c.get('clip_end_hkt') or '')[11:]}",
            "ok": c.get("ok"),
            "summary": parsed.get("scene_summary", ""),
            "activity_chain": parsed.get("activity_chain", ""),
        })
        for s in parsed.get("segments", []) or []:
            s = dict(s)
            s["clip_id"] = c.get("clip_id")
            s["_sort"] = _seg_start(s, base)
            segments.append(s)
    segments.sort(key=lambda s: (s["clip_id"], s["_sort"]))
    for s in segments:
        s.pop("_sort", None)

    ok = [c for c in clips if c.get("ok")]
    bad = [c for c in clips if not c.get("ok")]
    tin = sum((c.get("usage") or {}).get("in") or 0 for c in clips)
    tout = sum(((c.get("usage") or {}).get("out") or 0)
               + ((c.get("usage") or {}).get("think") or 0) for c in clips)
    service_tier = (manifest or {}).get("service_tier", "standard")
    price_multiplier = 0.5 if service_tier == "flex" else 1.0
    date = (clips[0].get("clip_start_hkt") or "")[:10] if clips else ""
    return {
        "recording": (manifest or {}).get("recording") or (clips[0].get("recording") if clips else ""),
        "date": date,
        "clip_count": len(clips),
        "clip_ok": len(ok),
        "clip_failed": len(bad),
        "failed_clip_ids": [c.get("clip_id") for c in bad],
        "segment_count": len(segments),
        "time_start_hkt": clips[0].get("clip_start_hkt") if clips else "",
        "time_end_hkt": clips[-1].get("clip_end_hkt") if clips else "",
        "has_gaze": bool(clips and clips[0].get("has_gaze")),
        "model": clips[0].get("model") if clips else "",
        "service_tier": service_tier,
        "tokens": {"in": tin, "out_billable": tout},
        "est_cost_usd": round(
            (tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT) * price_multiplier,
            4,
        ),
        "pii_audit": pii_audit(segments),
        "manifest": manifest or {},
        "clip_summaries": summaries,
        "segments": segments,
    }


def render_text(merged: dict) -> str:
    L = [f"# {merged['recording']}",
         f"日期 {merged['date']}   {merged['time_start_hkt']} → {merged['time_end_hkt']}",
         f"clip {merged['clip_ok']}/{merged['clip_count']} 成功   "
         f"segments {merged['segment_count']}   "
         f"眼动 {'有' if merged['has_gaze'] else '无'}   模型 {merged['model']}",
         ""]
    by_clip = {}
    for s in merged["segments"]:
        by_clip.setdefault(s.get("clip_id"), []).append(s)
    for cs in merged["clip_summaries"]:
        cid = cs["clip_id"]
        L.append(f"── clip_{cid:04d}  {cs['time']}  {'' if cs['ok'] else '[FAILED]'}")
        if cs["summary"]:
            L.append(f"   « {cs['summary']} »")
        for s in by_clip.get(cid, []):
            L.append(f"   [{s.get('time_range', '')}] {s.get('action', '')}")
            if s.get("speech"):
                L.append(f"       语音: {s['speech']}")
            if s.get("environment"):
                L.append(f"       环境: {s['environment']}")
            tv = s.get("text_visible")
            if tv:
                L.append(f"       文字: {', '.join(tv) if isinstance(tv, list) else tv}")
            ob = s.get("objects")
            if ob:
                L.append(f"       物体: {', '.join(ob) if isinstance(ob, list) else ob}")
            if s.get("details"):
                L.append(f"       细节: {s['details']}")
        L.append("")
    return "\n".join(L)


def write_recording_outputs(outdir: Path, merged: dict):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "captions_full.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2))
    with open(outdir / "captions_full.jsonl", "w") as f:
        for s in merged["segments"]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    (outdir / "captions_full.txt").write_text(render_text(merged))
    report = {k: merged[k] for k in
              ("recording", "date", "clip_count", "clip_ok", "clip_failed",
               "failed_clip_ids", "segment_count", "time_start_hkt", "time_end_hkt",
               "has_gaze", "model", "service_tier", "tokens", "est_cost_usd",
               "pii_audit")}
    (outdir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


class IncompleteRecordingError(RuntimeError):
    """Raised after checkpoints are saved when one or more clips remain failed."""


def load_clip_checkpoints(caps_dir: Path, clip_ids: list[int], recording: str,
                          model: str, pipeline_name: str = "v9-2pass",
                          service_tier: str | None = None):
    """Load only compatible successful clip files; failed/stale clips stay pending."""
    caps_dir = Path(caps_dir)
    completed = {}
    pending = []
    for cid in clip_ids:
        path = caps_dir / f"clip_{cid:04d}.json"
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            pending.append(cid)
            continue
        compatible = (
            isinstance(saved, dict)
            and saved.get("clip_id") == cid
            and saved.get("recording") == recording
            and saved.get("model") == model
            and saved.get("pipeline") == pipeline_name
            and (service_tier is None or saved.get("service_tier") == service_tier)
            and saved.get("ok") is True
            and isinstance(saved.get("parsed"), dict)
        )
        if compatible:
            completed[cid] = saved
        else:
            pending.append(cid)
    return completed, pending


def _start_dt(m):
    try:
        return datetime.strptime(m["time_start_hkt"], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.min


def merge_day(merged_list: list, date: str) -> dict:
    ms = sorted(merged_list, key=_start_dt)
    segs = []
    for m in ms:
        for s in m["segments"]:
            s = dict(s)
            s["recording"] = m["recording"]
            segs.append(s)
    return {
        "date": date,
        "recordings": [{"recording": m["recording"], "start": m["time_start_hkt"],
                        "end": m["time_end_hkt"], "clips": m["clip_count"],
                        "segments": m["segment_count"], "has_gaze": m["has_gaze"]}
                       for m in ms],
        "recording_count": len(ms),
        "segment_count": len(segs),
        "clip_ok": sum(m["clip_ok"] for m in ms),
        "clip_count": sum(m["clip_count"] for m in ms),
        "est_cost_usd": round(sum(m["est_cost_usd"] for m in ms), 4),
        "segments": segs,
    }


def render_day_text(day: dict) -> str:
    L = [f"# {day['date']} 全天 caption",
         f"{day['recording_count']} 条录制   clip {day['clip_ok']}/{day['clip_count']}   "
         f"segments {day['segment_count']}", ""]
    for r in day["recordings"]:
        L.append(f"  · {r['recording']}  {r['start'][11:]}→{r['end'][11:]}  "
                 f"{r['segments']} segs  眼动{'有' if r['has_gaze'] else '无'}")
    L.append("")
    cur = None
    for s in day["segments"]:
        if s.get("recording") != cur:
            cur = s.get("recording")
            L.append(f"\n══ {cur}")
        L.append(f"   [{s.get('time_range', '')}] {s.get('action', '')}")
        if s.get("speech"):
            L.append(f"       语音: {s['speech']}")
    return "\n".join(L)


def write_day_outputs(outdir: Path, day: dict):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"day_{day['date']}.json").write_text(
        json.dumps(day, ensure_ascii=False, indent=2))
    (outdir / f"day_{day['date']}.txt").write_text(render_day_text(day))
    with open(outdir / f"day_{day['date']}.jsonl", "w") as f:
        for s in day["segments"]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


# ====================================================================
#  10. Pipeline CLI
# ====================================================================


def list_tars(hf_token: str = ""):
    from huggingface_hub import HfApi
    a = HfApi(token=hf_token or None)
    info = a.repo_info(REPO, repo_type="dataset", files_metadata=True)
    out = []
    for s in info.siblings:
        if s.rfilename.startswith("aria/") and s.rfilename.endswith(".tar"):
            m = DUR_RE.search(s.rfilename)
            out.append({"path": s.rfilename, "size": s.size or 0,
                        "minutes": int(m.group(1)) if m else 0,
                        "day": s.rfilename.split("/")[1],
                        "rec": Path(s.rfilename).stem})
    return sorted(out, key=lambda r: r["path"])


def cmd_list(args):
    config = load_config(args.config)
    rows = list_tars(config["hf_token"])
    by = {}
    for r in rows:
        by.setdefault(r["day"], []).append(r)
    for d in sorted(by):
        rs = by[d]
        print(f"{d}  {len(rs)} 条  {sum(x['minutes'] for x in rs):>4} 分钟  "
              f"{sum(x['size'] for x in rs)/1e9:>6.1f} GB")
    print(f"\n共 {len(by)} 天 / {len(rows)} 条录制")


def process_recording(tar_path: str, config: dict, max_clips: int = None,
                      use_vad: bool = True, use_ocr: bool = True):
    global _MODEL, _SERVICE_TIER
    _MODEL = config.get("model", DEFAULT_MODEL)
    _SERVICE_TIER = config.get("service_tier", DEFAULT_SERVICE_TIER).lower()
    rec = Path(tar_path).stem
    n_per_clip = config.get("clip_seconds", 30)
    workers = config.get("workers", 3)
    max_workers = config.get("max_workers", DEFAULT_MAX_WORKERS)
    adaptive = bool(config.get("adaptive_concurrency", False))
    rounds = config.get("rounds", 3)
    outdir = Path(config.get("output_dir", "./caption_output"))

    _log(f"=== {rec} ===")
    request_timeout_sec = int(config.get("request_timeout_sec", DEFAULT_REQUEST_TIMEOUT_SEC))
    concurrency_desc = (f"aimd={workers}->{max_workers}" if adaptive
                        else f"workers={workers}")
    _log(f"model={_MODEL} tier={_SERVICE_TIER} timeout={request_timeout_sec}s "
         f"clip={n_per_clip}s {concurrency_desc} max_clips={max_clips}")

    t_all = time.time()
    src = HttpRangeSource(tar_path, token=config.get("hf_token", ""))
    _log(f"源: HTTP Range  {src.size/1e9:.2f} GB")

    # 1) 尾部：gaze.csv + 全部 ocr_text
    t0 = time.time()
    entries, buf, base = read_tail_adaptive(src)
    gaze_rows, ocr_texts = extract_tail_payload(entries, buf, base)
    del buf
    has_gaze = len(gaze_rows) > 0
    _log(f"尾部: gaze={len(gaze_rows)} 行, ocr={len(ocr_texts)} 帧  ({time.time()-t0:.1f}s)")
    trails = build_trails(gaze_rows) if has_gaze else {}

    # 2) audio/transcript 定位
    t0 = time.time()
    sec = locate_sections_safe(src)
    tr_entry = sec["transcript"]
    transcript_text = ""
    if tr_entry:
        transcript_text = src.read(tr_entry["data_offset"], tr_entry["size"]).decode("utf-8", "replace")
    intervals = parse_transcript(transcript_text)
    wav_desc = (f"wav@{sec['wav']['data_offset']} ({sec['wav']['size']/1e6:.0f}MB)"
                if sec["wav"] else "wav=无")
    _log(f"段定位: {wav_desc}, transcript={len(intervals)} 区间  ({time.time()-t0:.1f}s)")
    slicer = AudioSlicer(src, sec["wav"]) if sec["wav"] else None

    # 3) 走 picture/masked/，解码 → 2880px + 注视圈
    t0 = time.time()
    clip_frames: dict[int, list] = {}
    n_bytes = 0
    stop = False

    frames_data: dict[str, bytes] = {}

    def want(name):
        return "/picture/masked/" in name and name.lower().endswith((".jpg", ".jpeg"))

    pool = ThreadPoolExecutor(max(1, min(workers, 4)), thread_name_prefix="render")
    pending = []

    def render_and_store(fname, data):
        pts, valid = trails.get(fname, ([], False)) if has_gaze else (None, False)
        jpg = render_frame(data, pts, valid, size=FRAME_DIM, quality=FRAME_Q)
        return fname, jpg

    for name, size, data in iter_members(src, start=sec.get("pictures_start", 0), want=want):
        if data is None:
            if "/picture/ocr_text/" in name and clip_frames:
                break
            continue
        fname = os.path.basename(name)
        fi = frame_index(fname)
        if fi is None:
            continue
        cid = fi // n_per_clip
        clip_frames.setdefault(cid, []).append(fname)
        n_bytes += size
        pending.append(pool.submit(render_and_store, fname, data))
        if len(pending) >= 16:
            for f in pending[:8]:
                fn, jpg = f.result()
                frames_data[fn] = jpg
            pending = pending[8:]
        if max_clips and len(clip_frames) > max_clips:
            stop = True
            break
    for f in pending:
        fn, jpg = f.result()
        frames_data[fn] = jpg
    pool.shutdown(wait=True)

    if stop:
        newest = min(clip_frames)
        for fn in clip_frames.pop(newest):
            frames_data.pop(fn, None)

    total_frames = sum(len(v) for v in clip_frames.values())
    dt = time.time() - t0
    _log(f"解包: {total_frames} 帧 / {len(clip_frames)} clip, 读 {n_bytes/1e9:.2f} GB, "
         f"{dt:.0f}s ({n_bytes/dt/1e6:.0f} MB/s)" if dt > 0 else
         f"解包: {total_frames} 帧 / {len(clip_frames)} clip")
    if not clip_frames:
        raise RuntimeError("没解出任何帧")

    # 4) 组 clip → caption
    system_prompt = PROMPT_V9 if has_gaze else PROMPT_V9_NOGAZE
    _log(f"prompt: {'v9' if has_gaze else 'v9_nogaze'}（has_gaze={has_gaze}）")
    client = make_client(
        config.get("google_project", ""),
        config.get("google_location", "global"),
        service_tier=_SERVICE_TIER,
        request_timeout_sec=request_timeout_sec,
    )

    _first = min((f for v in clip_frames.values() for f in v), key=frame_index)
    first_index = frame_index(_first)
    base_time = frame_time(_first) - timedelta(seconds=first_index)

    def build_sd(cid):
        names = sorted(clip_frames[cid], key=frame_index)
        frames = []
        for fn in names:
            frames.append({"name": fn, "hkt": hkt_hms(fn), "jpg": frames_data[fn]})
        ocrs = []
        for fn in names:
            fi = frame_index(fn)
            got = ocr_texts.get(fi)
            if got and got[1]:
                ocrs.append(f"[{hkt_hms(fn)} {got[0]}]: {got[1]}")
        rel = frame_index(names[0])
        dur = len(names)
        lines = transcript_for_window(intervals, rel, dur)
        t0_ = frame_time(names[0])
        return {
            "frames": frames,
            "ocr_texts": ocrs,
            "transcript": "\n".join(lines),
            "n_frames": len(frames),
            "clip_start": t0_.strftime("%Y-%m-%d %H:%M:%S"),
            "clip_end": (t0_ + timedelta(seconds=dur)).strftime("%Y-%m-%d %H:%M:%S"),
        }

    todo = sorted(clip_frames)

    # determine output dir from internal date
    date = frame_time(_first).strftime("%Y-%m-%d") if frame_time(_first) else "unknown"
    rec_outdir = outdir / date / rec
    rec_outdir.mkdir(parents=True, exist_ok=True)
    caps_dir = rec_outdir / "clips"
    caps_dir.mkdir(parents=True, exist_ok=True)

    completed, pending = load_clip_checkpoints(
        caps_dir, todo, recording=rec, model=_MODEL, pipeline_name="v9-2pass",
        service_tier=_SERVICE_TIER,
    )
    if completed:
        _log(f"断点续跑: 复用 {len(completed)} 个成功 clip，"
             f"{len(pending)} 个 clip 待处理")
    else:
        _log(f"断点续跑: 未找到可复用的成功 clip，{len(pending)} 个 clip 待处理")

    def do_clip(cid):
        sd = build_sd(cid)
        wav16k = None
        if slicer is not None:
            first_idx = frame_index(sorted(clip_frames[cid], key=frame_index)[0])
            pcm = slicer.slice16k(first_idx, sd["n_frames"])
            if pcm:
                wav16k = wav_bytes(pcm)
        v9 = process_clip(client, sd, system_prompt, wav16k, clip_tag=f"clip_{cid:04d}",
                          use_transcript=True, use_ocr=use_ocr, use_vad=use_vad,
                          log=_log)
        p1, p2, sp = v9.get("pass1") or {}, v9.get("pass2") or {}, v9.get("speech") or {}
        merged_cap = v9.get("merged")
        parsed = merged_cap
        if parsed is None and p1.get("ok"):
            parsed = parse_json(p1.get("content", ""))
        usage = {
            "in": (p1.get("in") or 0) + (p2.get("in") or 0) + (sp.get("in") or 0),
            "out": (p1.get("out") or 0) + (p2.get("out") or 0) + (sp.get("out") or 0),
            "think": (p1.get("think") or 0) + (p2.get("think") or 0),
            "time": round((p1.get("time") or 0) + (p2.get("time") or 0) + (sp.get("time") or 0), 1),
        }
        note = ("v9-merged" if merged_cap is not None else
                ("v9-pass1-only" if parsed is not None else "v9-fail"))
        rec_out = {
            "clip_id": cid,
            "recording": rec,
            "clip_index": f"clip_{cid:04d}",
            "n_frames": sd["n_frames"],
            "frame_dim": FRAME_DIM,
            "clip_start_hkt": sd["clip_start"],
            "clip_end_hkt": sd["clip_end"],
            "payload_bytes": payload_bytes(sd),
            "transcript_lines": len([l for l in sd["transcript"].splitlines() if l.strip()]),
            "ocr_entries": len(sd["ocr_texts"]),
            "has_gaze": has_gaze,
            "ok": bool(is_complete(v9) and parsed is not None),
            "parse_note": note,
            "usage": usage,
            "model": _MODEL,
            "service_tier": _SERVICE_TIER,
            "error": p1.get("error") or p2.get("error"),
            "error_type": p1.get("error_type") or p2.get("error_type"),
            "content_raw": p1.get("content", ""),
            "parsed": parsed,
            "pipeline": "v9-2pass",
            "checkpointed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_crops": v9.get("n_crops", 0),
            "n_regions": p1.get("n_regions", 0),
            "pass1": p1,
            "pass2": p2 or None,
            "speech": sp,
        }
        (caps_dir / f"clip_{cid:04d}.json").write_text(
            json.dumps(rec_out, ensure_ascii=False, indent=2), encoding="utf-8")
        return cid, rec_out

    t0 = time.time()
    best: dict[int, dict] = dict(completed)
    remaining = list(pending)
    aimd = AIMDController(workers, max_workers) if adaptive else None
    for rnd in range(rounds):
        if not remaining:
            break
        w = aimd.limit if aimd else max(1, workers - rnd)
        if rnd:
            worker_desc = (f"AIMD={w}/{aimd.maximum}" if aimd else f"workers={w}")
            _log(f"  --- round {rnd+1}/{rounds}: {len(remaining)} clip 待重跑, "
                 f"{worker_desc} ---")
        pool_context = nullcontext(None) if aimd else ThreadPoolExecutor(
            w, thread_name_prefix="cap"
        )
        with pool_context as ex:
            if aimd:
                completions = run_dynamic_pool(
                    remaining,
                    do_clip,
                    aimd,
                    on_change=lambda change: _log(
                        f"  AIMD: {change[0]} -> {change[1]} "
                        f"({'additive increase' if change[2] == 'healthy' else 'multiplicative decrease: ' + change[2]})"
                    ),
                )
            else:
                futs = {ex.submit(do_clip, cid): cid for cid in remaining}
                completions = ((futs[fut], fut, None) for fut in as_completed(futs))

            for i, (cid, completed_value, completed_error) in enumerate(completions, 1):
                try:
                    if completed_error is not None:
                        raise completed_error
                    if aimd:
                        cid, rec_out = completed_value
                    else:
                        cid, rec_out = completed_value.result()
                except Exception as e:
                    _log(f"  [{i}/{len(remaining)}] clip_{cid:04d} 崩了: "
                         f"{redact_secrets(e, limit=800)}")
                    continue
                cur = best.get(cid)
                better = cur is None or (rec_out["ok"] and not cur.get("ok"))
                if better:
                    best[cid] = rec_out
                segs = len((rec_out["parsed"] or {}).get("segments", []))
                _log(f"  [{i}/{len(remaining)}] clip_{cid:04d} "
                     f"{'✓' if rec_out['ok'] else '✗'} "
                     f"{rec_out['clip_start_hkt'][11:]} segs={segs} "
                     f"in={rec_out['usage'].get('in')} out={rec_out['usage'].get('out')} "
                     f"{rec_out['usage'].get('time')}s")
                if not rec_out["ok"]:
                    _log(f"      error_type={rec_out.get('error_type', 'other')} "
                         f"error={redact_secrets(rec_out.get('error'), limit=800)}")
                _log(f"      checkpoint: {caps_dir / f'clip_{cid:04d}.json'}")
        terminal = [cid for cid in todo
                    if best.get(cid) and not best[cid].get("ok")
                    and not should_retry_clip_result(best[cid])]
        if terminal:
            _log("  本次 Job 不再重试终止型错误: "
                 + ", ".join(f"clip_{cid:04d}" for cid in terminal))
        remaining = [cid for cid in todo
                     if not best.get(cid, {}).get("ok")
                     and should_retry_clip_result(best.get(cid, {}))]
    results = best
    n_ok = sum(1 for v in results.values() if v.get("ok"))
    n_bad = len(todo) - n_ok
    _log(f"caption: {n_ok} ✓ / {n_bad} ✗  ({time.time()-t0:.0f}s)")
    if aimd:
        _log(f"AIMD统计: initial={max(1, min(int(workers), aimd.maximum))} "
             f"final={aimd.limit} peak={aimd.peak}/{aimd.maximum} "
             f"increases={aimd.increases} decreases={aimd.decreases}")
    request_times = []
    traffic_types = set()
    for clip in best.values():
        for phase in ("pass1", "pass2", "speech"):
            call = clip.get(phase) or {}
            if call.get("ok") and call.get("time") is not None:
                request_times.append(float(call["time"]))
            if call.get("traffic_type"):
                traffic_types.add(str(call["traffic_type"]))
    caption_wall = max(time.time() - t0, 0.001)
    if request_times:
        ordered = sorted(request_times)
        median = ordered[len(ordered) // 2]
        _log(f"API统计: requests={len(ordered)} median={median:.1f}s "
             f"max={max(ordered):.1f}s throughput={n_ok / caption_wall * 3600:.1f} clips/h "
             f"traffic={','.join(sorted(traffic_types)) or 'UNKNOWN'}")

    # 5) merge
    clips_list = sorted(results.values(), key=lambda c: c.get("clip_id", 0))
    manifest = {
        "recording": rec,
        "tar": tar_path,
        "clip_seconds": n_per_clip,
        "has_gaze": has_gaze,
        "gaze_rows": len(gaze_rows),
        "ocr_frames": len(ocr_texts),
        "transcript_intervals": len(intervals),
        "recording_start_hkt": base_time.strftime("%Y-%m-%d %H:%M:%S"),
        "frames_processed": total_frames,
        "clips": sorted(clip_frames),
        "frame_dim": FRAME_DIM,
        "model": _MODEL,
        "service_tier": _SERVICE_TIER,
        "request_timeout_sec": request_timeout_sec,
        "adaptive_concurrency": adaptive,
        "initial_workers": workers,
        "max_workers": max_workers,
        "aimd_final_workers": aimd.limit if aimd else None,
        "aimd_peak_workers": aimd.peak if aimd else None,
        "aimd_increases": aimd.increases if aimd else 0,
        "aimd_decreases": aimd.decreases if aimd else 0,
        "max_clips": max_clips,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t_all, 1),
    }
    merged = merge_recording(clips_list, manifest)
    report = write_recording_outputs(rec_outdir, merged)
    _log(f"合并: {report['clip_ok']}/{report['clip_count']} clip, "
         f"{report['segment_count']} segments, 估价 ${report['est_cost_usd']}")
    _log(f"产物: {rec_outdir}")
    if n_bad:
        _log(f"未完整: {n_bad}/{len(todo)} clip 仍失败；已保留成功 clip 和诊断产物，"
             "下次挂载同一 /output 后会自动续跑")
        raise IncompleteRecordingError(
            f"{rec}: {n_bad}/{len(todo)} clips failed; checkpoints saved in {caps_dir}"
        )
    _log(f"完成，总耗时 {time.time()-t_all:.0f}s")
    return merged


def cmd_run(args):
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.workers:
        config["workers"] = args.workers
    if args.max_workers:
        config["max_workers"] = args.max_workers
    if args.adaptive_concurrency:
        config["adaptive_concurrency"] = True
    if args.clip_seconds:
        config["clip_seconds"] = args.clip_seconds
    if args.service_tier:
        config["service_tier"] = args.service_tier
    if args.request_timeout:
        config["request_timeout_sec"] = args.request_timeout
    if args.rounds:
        config["rounds"] = args.rounds

    if not config.get("google_project"):
        sys.exit("错误：config.json 中未设置 google_project（Google Cloud 项目 ID）。\n"
                 "请通过 config.json 或 GOOGLE_CLOUD_PROJECT 提供项目 ID。")

    auth_mode = configure_google_environment(
        config["google_project"], config.get("google_location", "global")
    )
    _log(f"Google auth: mode={auth_mode} "
         f"api_key={'set' if os.environ.get('GOOGLE_API_KEY') else 'missing'} "
         f"project={'set' if config.get('google_project') else 'missing'} "
         f"location={config.get('google_location', 'global')} "
         f"tier={config.get('service_tier', DEFAULT_SERVICE_TIER)} "
         f"timeout={config.get('request_timeout_sec', DEFAULT_REQUEST_TIMEOUT_SEC)}s")

    rows = list_tars(config.get("hf_token", ""))

    if args.tar:
        targets = [args.tar]
    elif args.day:
        targets = [r["path"] for r in rows if r["day"] == args.day]
        if not targets:
            sys.exit(f"找不到 {args.day} 的录制")
    else:
        sys.exit("请指定 --tar 或 --day")

    _log(f"共 {len(targets)} 条录制待处理")
    day_results = []
    failures = 0
    for i, tar_path in enumerate(targets, 1):
        _log(f"\n{'='*60}")
        _log(f"[{i}/{len(targets)}] {Path(tar_path).stem}")
        _log(f"{'='*60}")
        try:
            merged = process_recording(tar_path, config, max_clips=args.max_clips,
                                       use_vad=not args.no_vad, use_ocr=not args.no_ocr)
            day_results.append(merged)
        except Exception as e:
            import traceback
            failures += 1
            _log(f"!! 失败: {type(e).__name__}: {redact_secrets(e, limit=800)}")
            _log(redact_secrets(traceback.format_exc()[-2000:]))

    if len(day_results) > 1 and args.day:
        outdir = Path(config.get("output_dir", "./caption_output")) / args.day
        day = merge_day(day_results, args.day)
        write_day_outputs(outdir, day)
        _log(f"\n整天合并: {outdir}/day_{args.day}.json")

    _log(f"\n全部完成: {len(day_results)} 成功 / {failures} 失败")
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="POV Dense Video Captioning Pipeline")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="列出 HuggingFace 上可用的天和录制")
    p_list.add_argument("--config", default="config.json", help="配置文件路径")

    p_run = sub.add_parser("run", help="对指定录制生成 caption")
    p_run.add_argument("--tar", help="单条录制的 tar 路径（如 aria/2026-05-18/...tar）")
    p_run.add_argument("--day", help="处理一天全部录制（如 2026-05-18）")
    p_run.add_argument("--max-clips", type=int, help="只处理前 N 个 clip（冒烟测试）")
    p_run.add_argument("--workers", type=int, help="caption 并发数（默认 3）")
    p_run.add_argument("--adaptive-concurrency", action="store_true",
                       help="启用 AIMD 动态并发（--workers 为初始值）")
    p_run.add_argument("--max-workers", type=int,
                       help="AIMD 并发上限（默认 12）")
    p_run.add_argument("--service-tier", choices=("standard", "flex"),
                       help="Gemini 服务层级（默认 flex）")
    p_run.add_argument("--request-timeout", type=int,
                       help="Gemini 单请求超时秒数（默认且最高 1800）")
    p_run.add_argument("--rounds", type=int,
                       help="失败 clip 的处理轮数（Flex 实验建议 1）")
    p_run.add_argument("--config", default="config.json", help="配置文件路径")
    p_run.add_argument("--output-dir", help="输出目录（默认 ./caption_output）")
    p_run.add_argument("--no-vad", action="store_true", help="禁用 silero-vad 语音对齐")
    p_run.add_argument("--no-ocr", action="store_true", help="不使用 OCR 文本")
    p_run.add_argument("--clip-seconds", type=int, default=30, help="每个 clip 秒数")

    args = parser.parse_args()
    if args.command == "list":
        cmd_list(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
