"""overlay_annotations.py — render time-synced annotation overlays onto EgoLife clips.

Reads a re-encoded slice from caption_pipeline.py's _cache/, the matching caption
record from the pipeline's JSONL output, and the ground-truth DenseCaption SRT from
EgoLifeCap/, then burns three annotation blocks into a new video:

  TOP-LEFT  (below the existing timestamp watermark):
              - self_actions  (timestamped, watermark-based)
              - speech        (no timestamps in source -> evenly spread over the clip)
  BOTTOM-LEFT:
              - DenseCaption  (ground truth, SRT-timestamped)

This is a standalone diagnostic/visualization tool — it does NOT touch the captioning
pipeline or any data under captions/. Sample output goes to a user-specified --out.

Usage
-----
  # minimal: defaults find everything for the standard A1_JAKE/DAY1 layout
  python overlay_annotations.py --clip-id DAY1_A1_JAKE_11094208 \
      --out ../out/_test/DAY1_A1_JAKE_11094208_overlay.mp4

  # point at non-default roots
  python overlay_annotations.py --clip-id DAY1_A1_JAKE_11094208 \
      --egolife-root D:/data/EgoLife --out overlay.mp4
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # .../ARIN7600/EgoLife/

# Geometry — chosen to clear the top-left watermark (~280x130 region) and leave a
# readable margin. The slice is 1024x1024 @ 2fps (caption_pipeline.py default).
TOP_BLOCK_Y = 150        # self_action block sits here (below the ~130px watermark)
SPEECH_Y = 230           # speech block sits below self_action
BOTTOM_MARGIN = 24       # DenseCaption distance from bottom edge
LEFT_MARGIN = 24

FONT_SIZE_ACTION = 28
FONT_SIZE_SPEECH = 24
FONT_SIZE_DENSE = 26

# On Windows the Microsoft YaHei TTC covers both Chinese and Latin glyphs.
DEFAULT_FONT = "C:/Windows/Fonts/msyh.ttc"

# ffmpeg encode params mirroring caption_pipeline.py's slice format.
FPS = 2
RESOLUTION = 1024
CRF = 23  # slightly higher quality than pipeline's 28 since text must stay crisp


# ===========================================================================
# Clip id / time parsing
# ===========================================================================

def parse_clip_id(clip_id: str) -> tuple[int, str, int, int, int, int, int]:
    """`DAY1_A1_JAKE_11094208` -> (day=1, participant='A1_JAKE', hh, mm, ss, cs, piece_idx).

    The trailing 8 digits are HH MM SS CC (centiseconds). For clips produced by the
    sub-30s splitter, an optional `_p{N}` suffix follows the timestamp
    (e.g. `DAY4_A1_JAKE_10483000_p2` is the 2nd piece of source 10483000). piece_idx
    is 0 for whole (un-split) clips, or N for `_p{N}`."""
    parts = clip_id.split("_")
    # DAY1_A1_JAKE_11094208      ->  ['DAY1', 'A1', 'JAKE', '11094208']
    # DAY4_A1_JAKE_10483000_p2   ->  ['DAY4', 'A1', 'JAKE', '10483000', 'p2']
    if len(parts) < 4:
        raise ValueError(f"unexpected clip_id format: {clip_id}")
    day = int(parts[0].removeprefix("DAY"))
    # Pop a trailing _p{N} piece suffix if present.
    piece_idx = 0
    if re.fullmatch(r"p\d+", parts[-1]):
        piece_idx = int(parts.pop()[1:])
    participant = "_".join(parts[1:-1])
    ts = parts[-1]
    if len(ts) != 8 or not ts.isdigit():
        raise ValueError(f"clip_id timestamp not 8 digits: {clip_id}")
    hh, mm, ss, cs = int(ts[0:2]), int(ts[2:4]), int(ts[4:6]), int(ts[6:8])
    return day, participant, hh, mm, ss, cs, piece_idx


def clip_start_seconds(clip_id: str, piece_offset_s: float = 0.0) -> float:
    """Absolute wall-clock seconds of the clip's first frame.

    For a split piece (`_p{N}`), the source timestamp names the SOURCE clip's start,
    not the piece's start — so callers must pass `piece_offset_s` = (piece_idx-1) *
    per-piece duration, which main() computes from the slice's ffprobe duration and
    the total source duration (or accepts an explicit --piece-offset)."""
    _, _, hh, mm, ss, cs, _ = parse_clip_id(clip_id)
    return hh * 3600 + mm * 60 + ss + cs / 100.0 + piece_offset_s


# ===========================================================================
# Caption JSONL loading
# ===========================================================================

def load_caption(jsonl_path: Path, clip_id: str) -> dict:
    """Find the caption record for clip_id in the pipeline's JSONL output."""
    if not jsonl_path.exists():
        return {}
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("clip_id") == clip_id:
                return rec
    return {}


def hms_to_seconds(hms: str) -> float | None:
    """`'11:09:42'` -> 40182.0. Returns None if empty/invalid."""
    if not hms:
        return None
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2}):(\d{2})\s*", hms)
    if not m:
        return None
    return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])


# ===========================================================================
# DenseCaption SRT loading (ground truth)
# ===========================================================================

_SRT_TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _srt_timecode_to_seconds(g: re.Match, which: str) -> float:
    """which='start' uses groups 1-4, 'end' uses 5-8."""
    off = 0 if which == "start" else 4
    h, m, s, ms = (int(g.group(off + i)) for i in range(1, 5))
    return h * 3600 + m * 60 + s + ms / 1000.0


def load_dense_caption(srt_dir: Path, participant: str, day: int,
                       clip_start: float, clip_end: float) -> list[dict]:
    """Load ground-truth DenseCaption entries overlapping [clip_start, clip_end].

    SRT filenames look like `A1_JAKE_DAY1_11000000.srt` where `11000000` is the
    hour the segment starts (11:00:00). SRT internal timestamps are offsets from
    that hour. We scan the relevant hourly SRT file(s), convert each entry to an
    absolute wall-clock second, keep those intersecting the clip window, and
    express each as an offset relative to clip_start (t=0 at first frame).
    """
    # Which hourly file(s) can overlap? The clip may straddle an hour boundary.
    entries: list[dict] = []
    start_h = int(clip_start // 3600)
    end_h = int(clip_end // 3600)
    for hour in range(start_h, end_h + 1):
        srt_path = srt_dir / participant / f"DAY{day}" / f"{participant}_DAY{day}_{hour:02d}000000.srt"
        if not srt_path.exists():
            continue
        hour_base = hour * 3600
        text = srt_path.read_text(encoding="utf-8-sig")
        blocks = re.split(r"\r?\n\r?\n", text.strip())
        for blk in blocks:
            lines = blk.strip().splitlines()
            if len(lines) < 2:
                continue
            # find the timecode line (robust to an optional index line before it)
            tc_line = next((ln for ln in lines if "-->" in ln), None)
            if tc_line is None:
                continue
            m = _SRT_TS_RE.search(tc_line)
            if not m:
                continue
            entry_start = hour_base + _srt_timecode_to_seconds(m, "start")
            entry_end = hour_base + _srt_timecode_to_seconds(m, "end")
            # keep if it intersects the clip window
            if entry_end <= clip_start or entry_start >= clip_end:
                continue
            # body = all lines AFTER the timecode line (skip the optional index line
            # that may precede it, and the timecode line itself).
            tc_idx = next(i for i, ln in enumerate(lines) if "-->" in ln)
            body = " ".join(lines[tc_idx + 1:]).strip()
            if not body:
                continue
            entries.append({
                "t_start": max(0.0, entry_start - clip_start),
                "t_end": min(clip_end - clip_start, entry_end - clip_start),
                "text": body,
            })
    return entries


# ===========================================================================
# ffmpeg drawtext filter construction
# ===========================================================================

# Characters that need escaping inside a drawtext text= value. drawtext's escaping
# is fiddly (it goes through libavutil), so we sidestep it entirely by writing each
# label to its own textfile and referencing it with textfile=.
def _write_textfile(tmpdir: Path, idx: int, text: str) -> str:
    """Write `text` to a temp file, return a drawtext-safe escaped path for textfile=."""
    p = tmpdir / f"lbl_{idx}.txt"
    p.write_text(text, encoding="utf-8")
    # drawtext's textfile option needs the path escaped: backslash and single quote.
    # Within our filter string we pass it single-quoted, so escape ' and \ inside.
    escaped = str(p).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return escaped


def _drawtext_node(textfile_escaped: str, *, x: str, y: str, fontsize: int,
                   color: str, enable: str, box_color: str = "black@0.6") -> str:
    """Build one drawtext filter node referencing a textfile."""
    return (
        f"drawtext=fontfile='{{FONT}}':textfile='{textfile_escaped}'"
        f":x={x}:y={y}:fontsize={fontsize}:fontcolor={color}"
        f":box=1:boxcolor={box_color}:boxborderw=8:line_spacing=4"
        f":enable='{enable}'"
    )


def build_filter(slice_path: Path, duration: float, caption: dict,
                 dense_entries: list[dict], font: str, tmpdir: Path) -> str:
    """Assemble the full -vf filtergraph string (fontfile substituted at call time)."""
    nodes: list[str] = []
    tf_idx = 0

    # --- self_actions (timestamped, top-left) ---
    self_actions = caption.get("self_actions") or []
    for act in self_actions:
        # Offsets (seconds from first frame) are precomputed in main() from the
        # watermark HH:MM:SS timestamps and stored as _off_start/_off_end.
        off_s = act.get("_off_start", 0.0)
        off_e = act.get("_off_end", off_s + 1.0)
        text = act.get("text", "").strip()
        if not text:
            continue
        label = f"[动作] {text}"
        tf = _write_textfile(tmpdir, tf_idx, label); tf_idx += 1
        nodes.append(_drawtext_node(
            tf, x=str(LEFT_MARGIN), y=str(TOP_BLOCK_Y),
            fontsize=FONT_SIZE_ACTION, color="white",
            enable=f"between(t,{off_s:.3f},{off_e:.3f})",
        ))

    # --- speech (no timestamps -> evenly spread over the whole clip) ---
    speech = caption.get("speech") or []
    n = len(speech)
    if n > 0 and duration > 0:
        seg = duration / n
        for i, sp in enumerate(speech):
            text = (sp.get("text") or "").strip()
            if not text:
                continue
            sp_label = f"[语音] {text}"
            off_s = i * seg
            off_e = (i + 1) * seg if i < n - 1 else duration  # last runs to end
            tf = _write_textfile(tmpdir, tf_idx, sp_label); tf_idx += 1
            nodes.append(_drawtext_node(
                tf, x=str(LEFT_MARGIN), y=str(SPEECH_Y),
                fontsize=FONT_SIZE_SPEECH, color="#FFD24C",  # warm yellow
                enable=f"between(t,{off_s:.3f},{off_e:.3f})",
                box_color="black@0.6",
            ))

    # --- DenseCaption ground truth (bottom-left) ---
    for de in dense_entries:
        text = de["text"].strip()
        if not text:
            continue
        label = f"[GT] {text}"
        tf = _write_textfile(tmpdir, tf_idx, label); tf_idx += 1
        nodes.append(_drawtext_node(
            tf, x=str(LEFT_MARGIN), y=f"h-th-{BOTTOM_MARGIN}",
            fontsize=FONT_SIZE_DENSE, color="#7CFFB2",  # mint green = ground truth
            enable=f"between(t,{de['t_start']:.3f},{de['t_end']:.3f})",
            box_color="black@0.6",
        ))

    if not nodes:
        return ""
    return ",".join(nodes).replace("{FONT}", font.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'"))


# ===========================================================================
# ffmpeg driver
# ===========================================================================

def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True,
    )
    return float(out.strip())


def render(slice_path: Path, out_path: Path, vf: str) -> None:
    """Run ffmpeg to burn the filtergraph into out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(slice_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart", str(out_path),
    ]
    rc = subprocess.run(cmd, capture_output=True, text=True)
    if rc.returncode != 0:
        sys.stderr.write(rc.stderr[-2000:])
        raise RuntimeError(f"ffmpeg failed (rc={rc.returncode}); see stderr above")


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Burn time-synced annotation overlays onto an EgoLife clip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--clip-id", required=True,
                    help="e.g. DAY1_A1_JAKE_11094208")
    ap.add_argument("--slice-path", type=Path, default=None,
                    help="re-encoded slice mp4 (default: derived from --egolife-root)")
    ap.add_argument("--jsonl", type=Path, default=None,
                    help="caption_pipeline JSONL (default: derived)")
    ap.add_argument("--srt-dir", type=Path, default=None,
                    help="EgoLifeCap/DenseCaption root (default: derived)")
    ap.add_argument("--egolife-root", type=Path, default=ROOT,
                    help="EgoLife project root (contains captions/, EgoLifeCap/, videos/)")
    ap.add_argument("--participant", default=None, help="override participant (default: from clip-id)")
    ap.add_argument("--day", type=int, default=None, help="override day (default: from clip-id)")
    ap.add_argument("--font", default=DEFAULT_FONT)
    ap.add_argument("--piece-offset", type=float, default=None,
                    help="for split pieces (_p{N}): seconds the piece starts into its "
                         "source clip, i.e. (piece_idx-1) * per-piece duration. Auto-"
                         "computed from piece_idx and slice duration if omitted; override "
                         "for non-even splits or to use the exact source-derived offset.")
    ap.add_argument("--out", type=Path, required=True, help="output mp4 path")
    args = ap.parse_args()

    day, participant, hh, mm, ss, cs, piece_idx = parse_clip_id(args.clip_id)
    day = args.day or day
    participant = args.participant or participant

    egolife = args.egolife_root
    # Derive default paths from the standard layout.
    if args.slice_path is None:
        args.slice_path = (egolife / "captions" / participant / f"DAY{day}"
                           / "_cache" / "slices" / f"{args.clip_id}.mp4")
    if args.jsonl is None:
        args.jsonl = egolife / "captions" / participant / f"DAY{day}" / "full.jsonl"
    if args.srt_dir is None:
        args.srt_dir = egolife / "EgoLifeCap" / "DenseCaption"

    if not args.slice_path.exists():
        sys.exit(f"slice not found: {args.slice_path}")
    duration = ffprobe_duration(args.slice_path)

    # For split pieces (_p{N}), the clip_id's timestamp is the SOURCE's start; the
    # piece itself starts (piece_idx-1) slices later. Auto-compute as
    # (piece_idx-1) * this_slice_duration, which is exact for even splits.
    if piece_idx > 0:
        piece_offset = args.piece_offset if args.piece_offset is not None else (piece_idx - 1) * duration
    else:
        piece_offset = 0.0
    clip_start = clip_start_seconds(args.clip_id, piece_offset_s=piece_offset)
    clip_end = clip_start + duration

    caption = load_caption(args.jsonl, args.clip_id)
    if not caption:
        print(f"warning: no caption record for {args.clip_id} in {args.jsonl}", file=sys.stderr)

    # Precompute self_action offsets (watermark HH:MM:SS -> seconds-from-first-frame).
    for act in caption.get("self_actions", []) or []:
        t_s = hms_to_seconds(act.get("time", ""))
        t_e = hms_to_seconds(act.get("time_end", ""))
        act["_off_start"] = max(0.0, (t_s - clip_start)) if t_s is not None else 0.0
        act["_off_end"] = (t_e - clip_start) if t_e is not None else (act["_off_start"] + 1.0)

    dense_entries = load_dense_caption(args.srt_dir, participant, day, clip_start, clip_end)

    n_act = len([a for a in (caption.get("self_actions") or []) if a.get("text", "").strip()])
    n_sp = len([s for s in (caption.get("speech") or []) if s.get("text", "").strip()])
    print(f"clip {args.clip_id} ({duration:.1f}s): "
          f"self_actions={n_act} speech={n_sp} dense_caption={len(dense_entries)}")

    if not Path(args.font).exists():
        sys.exit(f"font not found: {args.font}")

    with tempfile.TemporaryDirectory(prefix="overlay_") as td:
        tmpdir = Path(td)
        vf = build_filter(args.slice_path, duration, caption, dense_entries, args.font, tmpdir)
        if not vf:
            sys.exit("no annotation nodes built (nothing to overlay)")
        render(args.slice_path, args.out, vf)

    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
