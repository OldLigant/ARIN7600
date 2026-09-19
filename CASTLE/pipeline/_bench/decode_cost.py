#!/usr/bin/env python3
"""Measure where CASTLE clip preparation actually spends time.

Separates these costs on a real or synthetic source:
  D0  decode only (select 1 fps, null sink)              <- irreducible
  D1  decode + scale to model size (no encode)
  D2  decode + scale + JPEG q:v 2                        <- current media.py
  D3  decode + scale + pad + drawtext in ffmpeg          <- candidate replacement
  P1  Python: thumbnail + save + stamp at native size
  P2  Python: thumbnail + save + stamp at model size
  P3  Python: stamp only, parallel

Usage:
  python decode_cost.py --source X.mp4 --work _bench/tmp2 --threads 1|2|4|8
Add --fonts DIR to enable drawtext measurements.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SELECT = "select='gte(t,selected_n/1)'"


def run(args):
    return subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)


def measure(label, args, results):
    started = time.perf_counter()
    proc = run(args)
    elapsed = time.perf_counter() - started
    entry = {"label": label, "sec": round(elapsed, 3), "ok": proc.returncode == 0}
    if proc.returncode:
        entry["stderr_tail"] = proc.stderr.decode(errors="replace")[-300:]
    results.append(entry)
    return entry


def stage_native(path, seconds, target):
    with Image.open(path) as image:
        image.thumbnail(target)
        image.save(path, "JPEG", quality=95)
    with Image.open(path) as original:
        with Image.new("RGB", (original.width, original.height + 24), "black") as result:
            result.paste(original, (0, 0))
            ImageDraw.Draw(result).text((6, original.height + 5), f"t={seconds:.3f}s",
                                        fill="white", font=ImageFont.load_default(size=13))
            result.save(path, "JPEG", quality=95)


def find_font(explicit):
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            return candidate
        found = next(iter(candidate.rglob("*.ttf")), None)
        if found:
            return found
    for root in (r"C:\Windows\Fonts", "/usr/share/fonts"):
        p = Path(root)
        if p.exists():
            for pattern in ("arial.ttf", "DejaVuSans.ttf", "**/DejaVuSans.ttf", "**/arial.ttf"):
                found = next(iter(p.glob(pattern)), None)
                if found:
                    return found
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--max-dim", type=int, default=1440)
    ap.add_argument("--native-dim", type=int, default=3840)
    ap.add_argument("--fonts", default=None)
    args = ap.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    results = []
    t = str(args.threads)
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-threads", t, "-filter_threads", t]
    common = ["-ss", "0", "-t", f"{args.seconds:g}", "-noautorotate", "-i", str(args.source),
              "-map", "0:v:0", "-an"]

    # D0: decode only. -f null discards frames with no encoder cost.
    measure("D0_decode_only_null_sink",
            base + common + ["-vf", SELECT, "-fps_mode", "vfr", "-f", "null", "-"],
            results)

    # D1: decode + scale, still no encoder.
    measure("D1_decode_plus_scale_null_sink",
            base + common + ["-vf", f"{SELECT},scale=w='min(iw,{args.max_dim})'"
                                     f":h='min(ih,{args.max_dim - 24})':force_original_aspect_ratio=decrease,setsar=1",
                             "-fps_mode", "vfr", "-f", "null", "-"],
            results)

    # D2: current media.py (JPEG q:v 2 at native size, no footer)
    out2 = args.work / "d2"; out2.mkdir(exist_ok=True)
    for f in out2.glob("*.jpg"):
        f.unlink()
    measure("D2_decode_scale_native_jpeg_qv2",
            base + common + ["-vf", f"{SELECT},scale=w='min(iw,{args.native_dim})'"
                                     f":h='min(ih,{args.native_dim})':force_original_aspect_ratio=decrease,setsar=1",
                             "-fps_mode", "vfr", "-frames:v", str(int(args.seconds)), "-q:v", "2",
                             str(out2 / "f_%06d.jpg")],
            results)

    # D3: model-size JPEG, separate cheap scale-only measurement
    out3 = args.work / "d3"; out3.mkdir(exist_ok=True)
    for f in out3.glob("*.jpg"):
        f.unlink()
    measure("D3_decode_scale_model_jpeg_qv2",
            base + common + ["-vf", f"{SELECT},scale=w='min(iw,{args.max_dim})'"
                                     f":h='min(ih,{args.max_dim - 24})':force_original_aspect_ratio=decrease,setsar=1",
                             "-fps_mode", "vfr", "-frames:v", str(int(args.seconds)), "-q:v", "2",
                             str(out3 / "f_%06d.jpg")],
            results)

    font = find_font(args.fonts)
    if font:
        out4 = args.work / "d4"; out4.mkdir(exist_ok=True)
        for f in out4.glob("*.jpg"):
            f.unlink()
        graph = (f"{SELECT},scale=w='min(iw,{args.max_dim})':h='min(ih,{args.max_dim - 24})'"
                 f":force_original_aspect_ratio=decrease,setsar=1,pad=w=iw:h=ih+24:x=0:y=0:color=black,"
                 f"drawtext=fontfile='{str(font).replace(chr(92), '/').replace(':', chr(92) + ':')}'"
                 f":text='t=%{{pts\\:hms}}s':x=6:y=h-19:fontsize=13:fontcolor=white")
        measure("D4_decode_scale_pad_drawtext_jpeg",
                base + common + ["-vf", graph, "-fps_mode", "vfr", "-frames:v", str(int(args.seconds)),
                                 "-q:v", "2", str(out4 / "f_%06d.jpg")],
                results)
    else:
        results.append({"label": "D4_skipped", "sec": 0, "ok": False, "stderr_tail": "no font found"})

    # Python-side costs on the model-size frames produced by D3.
    frames = sorted(out3.glob("f_*.jpg"))
    results.append({"label": "D3_frame_count", "sec": len(frames), "ok": len(frames) > 0})

    work_p1 = args.work / "p1"; work_p1.mkdir(exist_ok=True)
    for path in frames:
        (work_p1 / path.name).write_bytes(path.read_bytes())
    started = time.perf_counter()
    for index, path in enumerate(sorted(work_p1.glob("f_*.jpg"))):
        stage_native(path, index, (args.native_dim, args.native_dim))
    results.append({"label": "P1_python_thumbnail_save_stamp", "sec": round(time.perf_counter() - started, 3),
                    "ok": True, "frames": len(frames)})

    work_p2 = args.work / "p2"; work_p2.mkdir(exist_ok=True)
    for path in frames:
        (work_p2 / path.name).write_bytes(path.read_bytes())
    started = time.perf_counter()
    for index, path in enumerate(sorted(work_p2.glob("f_*.jpg"))):
        stage_native(path, index, (args.max_dim, args.max_dim - 24))
    results.append({"label": "P2_python_thumbnail_save_stamp_model_size",
                    "sec": round(time.perf_counter() - started, 3), "ok": True, "frames": len(frames)})

    work_p3 = args.work / "p3"; work_p3.mkdir(exist_ok=True)
    for path in frames:
        (work_p3 / path.name).write_bytes(path.read_bytes())

    def stamp_only(pair):
        index, path = pair
        with Image.open(path) as original:
            with Image.new("RGB", (original.width, original.height + 24), "black") as result:
                result.paste(original, (0, 0))
                ImageDraw.Draw(result).text((6, original.height + 5), f"t={index:.3f}s", fill="white",
                                            font=ImageFont.load_default(size=13))
                result.save(path, "JPEG", quality=95)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(stamp_only, enumerate(sorted(work_p3.glob("f_*.jpg")))))
    results.append({"label": "P3_python_stamp_only_parallel8", "sec": round(time.perf_counter() - started, 3),
                    "ok": True, "frames": len(frames)})

    print(json.dumps({"threads": args.threads, "source": str(args.source), "seconds": args.seconds,
                      "cpu_count": os.cpu_count(), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
