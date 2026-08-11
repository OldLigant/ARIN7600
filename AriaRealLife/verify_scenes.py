"""verify_scenes.py — use MiMo to verify whether each 10-frame block is a single
continuous scene, and locate scene cuts within blocks.

Reads the downsampled frames already cached by caption_pipeline_arl.py
(captions/_cache/img1024/scene_XX/frame_*.jpg) so no re-encoding is needed.

For each of the 30 blocks (frame_00000-00009, 00010-00019, ...), sends the 10
frames to MiMo with NO audio and asks:
  - is this block a single continuous scene?
  - if not, between which consecutive frames does the cut happen?

Writes a JSON report + a human-readable markdown report. Does NOT modify the
captioning pipeline or its output.

Usage:
  python AriaRealLife/verify_scenes.py
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

CACHE_DIR = HERE / "captions" / "_cache" / "img1024"
REPORT_JSON = HERE / "captions" / "_test" / "scene_verification.json"
REPORT_MD = HERE / "captions" / "_test" / "scene_verification.md"

SYSTEM_MSG = """You are a video scene-cut detector. You receive 10 frames sampled at 1fps from an egocentric (Aria glasses) clip. The frames are numbered frame_0 through frame_9 and correspond to seconds 0-9 of the clip.

Your job: decide whether ALL 10 frames belong to ONE continuous scene, or whether there is a SCENE CUT somewhere in the middle.

A "scene cut" means a discontinuity in LOCATION or ACTIVITY that cannot be explained by continuous first-person motion within the same scene. Examples of cuts:
  - the wearer was in a kitchen, then suddenly in a gym
  - the wearer was eating, then suddenly walking outdoors
  - a completely different room / set of people appears

NOT a cut (still one continuous scene):
  - the wearer turns their head or body (background changes smoothly)
  - the wearer picks up / puts down an object
  - lighting changes gradually
  - the same people/room remain visible across the transition

Be conservative: only flag a cut when the discontinuity is clear. If unsure, lean toward "single scene"."""

USER_TMPL = """Here are 10 consecutive egocentric frames (frame_0 = second 0, frame_9 = second 9).

Decide whether they form ONE continuous scene or contain a scene cut.

Respond as a SINGLE JSON object, nothing else:
{{
  "single_scene": true | false,
  "cut_after_frame": <int or null>,   // if not single_scene: the index i (0-8) AFTER which the cut happens, i.e. frame_i and frame_(i+1) are in different scenes. null if single_scene is true.
  "scene_summary": "<one sentence describing the scene(s)>",
  "confidence": "high" | "medium" | "low",
  "reasoning": "<one short sentence>"
}}

Do not use markdown fences. Output only the JSON."""


def b64_file(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii")


def load_block(block_idx: int) -> list[Path]:
    """Return the 10 downsampled frames for block block_idx (0-29)."""
    d = CACHE_DIR / f"scene_{block_idx:02d}"
    frames = sorted(d.glob("frame_*.jpg"))
    if len(frames) != 10:
        raise FileNotFoundError(f"block {block_idx}: expected 10 frames in {d}, got {len(frames)}")
    return frames


def make_messages(block_idx: int) -> list[dict]:
    frames = load_block(block_idx)
    content = []
    for i, fp in enumerate(frames):
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_file(fp)}"},
        })
    # annotate which image is which frame index (0-9)
    content.append({"type": "text",
                    "text": USER_TMPL + f"\n\n(Reminder: the images above are frame_0 through frame_9 in order.)"})
    return [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": content}]


def parse_response(content: str) -> dict:
    text = content.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return {"_parse_error": str(e), "_raw": content[:500]}
    return data


def main():
    # --- API key ---
    env_paths = [Path.cwd() / ".env", HERE / ".env", REPO / "EgoLife" / ".env"]
    for ep in env_paths:
        if ep.exists():
            load_dotenv(ep); break
    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key:
        print(f"ERROR: MIMO_API_KEY not set. Searched: {[str(p) for p in env_paths]}", file=sys.stderr)
        sys.exit(2)

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key, base_url="https://api.xiaomimimo.com/v1")

    results = []
    print(f"Verifying 30 blocks via MiMo (mimo-v2.5)...")
    for block_idx in range(30):
        frame_start = block_idx * 10
        frame_end = frame_start + 9
        try:
            messages = make_messages(block_idx)
        except FileNotFoundError as e:
            print(f"  block {block_idx:02d} (frame_{frame_start:05d}-{frame_end:05d}): SKIP ({e})")
            results.append({"block_idx": block_idx, "frame_start": frame_start,
                            "frame_end": frame_end, "error": str(e)})
            continue
        t0 = time.time()
        for attempt in range(1, 4):
            try:
                resp = client.chat.completions.create(
                    model="mimo-v2.5", messages=messages,
                    max_completion_tokens=300,
                    extra_body={"thinking": {"type": "disabled"}},
                    response_format={"type": "json_object"},
                    temperature=1.0,
                )
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        latency = time.time() - t0
        content = resp.choices[0].message.content or ""
        parsed = parse_response(content)
        rec = {
            "block_idx": block_idx,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "latency_s": round(latency, 2),
            "raw_parsed": parsed,
            "scene_summary": parsed.get("scene_summary", ""),
        }
        if "single_scene" in parsed:
            rec["single_scene"] = bool(parsed["single_scene"])
            rec["cut_after_frame"] = parsed.get("cut_after_frame")
            # cut_after_frame is relative to the block (0-8); convert to absolute frame index
            caf = parsed.get("cut_after_frame")
            if caf is not None and not rec["single_scene"]:
                try:
                    rec["cut_after_absolute_frame"] = frame_start + int(caf)
                except (TypeError, ValueError):
                    pass
            rec["confidence"] = parsed.get("confidence", "")
            rec["reasoning"] = parsed.get("reasoning", "")
        else:
            rec["single_scene"] = None
            rec["parse_error"] = parsed.get("_parse_error", "")
        results.append(rec)
        flag = "SINGLE" if rec.get("single_scene") else ("CUT" if rec.get("single_scene") is False else "ERR")
        cut_info = f" cut_after_frame={rec.get('cut_after_absolute_frame','?')}" if flag == "CUT" else ""
        print(f"  block {block_idx:02d} (frame_{frame_start:05d}-{frame_end:05d}): {flag}{cut_info} [{rec.get('confidence','')}] {rec.get('scene_summary','')[:80]}")

    # --- Write JSON report ---
    REPORT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON report -> {REPORT_JSON}")

    # --- Build markdown report ---
    n_single = sum(1 for r in results if r.get("single_scene") is True)
    n_cut = sum(1 for r in results if r.get("single_scene") is False)
    n_err = sum(1 for r in results if r.get("single_scene") is None)
    cuts = [r["cut_after_absolute_frame"] for r in results
            if r.get("single_scene") is False and "cut_after_absolute_frame" in r]

    lines = []
    lines.append("# Scene Verification Report (MiMo, 10-frame blocks)\n")
    lines.append(f"- blocks verified: {len(results)}")
    lines.append(f"- single continuous scene: {n_single}")
    lines.append(f"- contains a scene cut: {n_cut}")
    lines.append(f"- errors: {n_err}")
    if cuts:
        lines.append(f"- detected cut locations (absolute frame index, cut happens AFTER this frame): {cuts}")
    lines.append("")
    lines.append("## Per-block detail\n")
    lines.append("| block | frames | single? | cut after frame | conf | summary |")
    lines.append("|-------|--------|---------|-----------------|------|---------|")
    for r in results:
        blk = r["block_idx"]
        fr = f"{r['frame_start']:05d}-{r['frame_end']:05d}"
        single = "yes" if r.get("single_scene") is True else ("NO" if r.get("single_scene") is False else "err")
        caf = str(r.get("cut_after_absolute_frame", "")) if r.get("single_scene") is False else ""
        conf = r.get("confidence", "")
        summ = (r.get("scene_summary", "") or "").replace("|", "/")
        lines.append(f"| {blk:02d} | frame_{fr} | {single} | {caf} | {conf} | {summ} |")
    lines.append("")
    lines.append("## Reasoning for flagged cuts\n")
    for r in results:
        if r.get("single_scene") is False:
            lines.append(f"### block {r['block_idx']:02d} (frame_{r['frame_start']:05d}-{r['frame_end']:05d})")
            caf_abs = r.get("cut_after_absolute_frame")
            if isinstance(caf_abs, int):
                lines.append(f"- cut after absolute frame: **frame_{caf_abs:05d}**")
            else:
                lines.append(f"- cut after (block-relative): {r.get('cut_after_frame')}")
            lines.append(f"- confidence: {r.get('confidence','')}")
            lines.append(f"- summary: {r.get('scene_summary','')}")
            lines.append(f"- reasoning: {r.get('reasoning','')}")
            lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report -> {REPORT_MD}")
    print(f"\n=== SUMMARY ===")
    print(f"single-scene blocks: {n_single}/{len(results)}")
    print(f"blocks with a cut:   {n_cut}/{len(results)}")
    print(f"errors:              {n_err}/{len(results)}")
    if cuts:
        print(f"detected cut locations (absolute frame idx, cut after): {cuts}")


if __name__ == "__main__":
    main()
