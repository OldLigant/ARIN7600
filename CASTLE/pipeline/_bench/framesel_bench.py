#!/usr/bin/env python3
"""Local micro-benchmark for the CASTLE clip-preparation stage.

Answers: of the wall-clock cost of preparing one 30 s clip at 1 fps, how much is
FFmpeg decode, how much is JPEG scaling/encoding, and how much is the Python
footer/re-encode step? Compares the current media.py command against candidate
filter graphs. Reads a synthetic source; requires ffmpeg/ffprobe on PATH.

Not part of the pipeline test suite. Writes only inside its own --work directory.
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

SOFF = "format=yuvj420p"


def run(args, **kw):
    return subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False, **kw)


def timed(label, args, results):
    started = time.perf_counter()
    proc = run(args)
    elapsed = time.perf_counter() - started
    ok = proc.returncode == 0
    results.append({"label": label, "sec": round(elapsed, 3), "ok": ok})
    if not ok:
        results[-1]["stderr_tail"] = proc.stderr.decode(errors="replace")[-400:]
    return elapsed, ok


def footer(path, seconds, height):
    with Image.open(path) as original:
        with Image.new("RGB", (original.width, original.height + height), "black") as result:
            result.paste(original, (0, 0))
            draw = ImageDraw.Draw(result)
            draw.text((6, original.height + 5), f"t={seconds:.3f}s", fill="white",
                      font=ImageFont.load_default(size=13))
            result.save(path, "JPEG", quality=95)


def thumbnail_only(path):
    with Image.open(path) as image:
        image.thumbnail((1440, 1416))
        image.save(path, "JPEG", quality=95)


def stage_image(path, seconds):
    """Current batch path: crop/scale to model size, save, then stamp + re-save."""
    with Image.open(path) as image:
        image.thumbnail((1440, 1416))
        image.save(path, "JPEG", quality=95)
    footer(path, seconds, 24)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--native-dim", type=int, default=3840)
    parser.add_argument("--max-dim", type=int, default=1440)
    parser.add_argument("--make-source", type=Path, help="Generate a synthetic source here and exit")
    parser.add_argument("--source-seconds", type=float, default=60)
    parser.add_argument("--fps", type=float, default=50)
    args = parser.parse_args()

    if args.make_source:
        args.make_source.parent.mkdir(parents=True, exist_ok=True)
        gen = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "lavfi", "-i", f"testsrc2=size=3840x2160:rate={args.fps:g}:duration={args.source_seconds:g}",
               "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={args.source_seconds:g}",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
               "-g", "50", "-c:a", "aac", "-b:a", "128k", "-shortest", str(args.make_source)]
        elapsed, ok = timed("generate_source", gen, [])
        print(json.dumps({"generated": str(args.make_source), "sec": round(elapsed, 1), "ok": ok}))
        return 0 if ok else 1

    args.work.mkdir(parents=True, exist_ok=True)
    results = []
    t = str(args.threads)
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-threads", t, "-filter_threads", t]
    native = args.native_dim
    duration = args.frames  # 1 fps -> N seconds, N frames
    count = args.frames
    ffprobe = ["ffprobe", "-v", "error", "-show_entries",
               "stream=codec_type,width,height,duration:stream_tags=DURATION", "-of", "json", str(args.source)]
    started = time.perf_counter()
    run(ffprobe)
    results.append({"label": "ffprobe", "sec": round(time.perf_counter() - started, 4), "ok": True})

    out = args.work / "a"; out.mkdir(exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()

    # A: exact current media.py video command (select + scale to native dim)
    current = base + ["-ss", "0", "-t", str(duration), "-noautorotate", "-i", str(args.source),
                      "-map", "0:v:0", "-an",
                      "-vf", f"select='gte(t,selected_n/1)',scale=w='min(iw,{native})':h='min(ih,{native})':"
                             "force_original_aspect_ratio=decrease,setsar=1",
                      "-fps_mode", "vfr", "-frames:v", str(count), "-q:v", "2",
                      "-threads", t, str(out / "frame_%06d.jpg")]
    decode_sec, ok = timed("A_ffmpeg_baseline_select_scale_qv2", current, results)

    # B: same decode, model-size scale in ffmpeg, libx264 lossless (cheap encoder)
    out_b = args.work / "b"; out_b.mkdir(exist_ok=True)
    for f in out_b.glob("*.jpg"):
        f.unlink()
    noscale = base + ["-ss", "0", "-t", str(duration), "-noautorotate", "-i", str(args.source),
                      "-map", "0:v:0", "-an", "-vf", "select='gte(t,selected_n/1)',null",
                      "-fps_mode", "vfr", "-frames:v", str(count),
                      "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p",
                      str(out_b / "frame_%06d.mp4")]
    timed("B_ffmpeg_select_encode_only", noscale, results)

    frames = sorted(out.glob("frame_*.jpg"))
    results.append({"label": "frame_count", "sec": len(frames), "ok": len(frames) == count})

    # C: python thumbnail + save + footer (current batch path per frame)
    work_c = args.work / "c"; work_c.mkdir(exist_ok=True)
    for f in list(work_c.glob("*.jpg")):
        f.unlink()
    for index, path in enumerate(frames):
        target = work_c / path.name
        target.write_bytes(path.read_bytes())
    started = time.perf_counter()
    for index, path in enumerate(sorted(work_c.glob("frame_*.jpg"))):
        stage_image(path, index / 1.0)
    results.append({"label": "C_python_thumbnail_save_then_footer", "sec": round(time.perf_counter() - started, 3),
                    "ok": True, "frames": len(frames)})

    # D: same but with a thread pool (GIL release check)
    work_d = args.work / "d"; work_d.mkdir(exist_ok=True)
    for f in list(work_d.glob("*.jpg")):
        f.unlink()
    for path in frames:
        (work_d / path.name).write_bytes(path.read_bytes())
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda p: stage_image(p[1], p[0] / 1.0),
                      enumerate(sorted(work_d.glob("frame_*.jpg")))))
    results.append({"label": "D_python_parallel_8_threads", "sec": round(time.perf_counter() - started, 3),
                    "ok": True, "frames": len(frames)})

    # E: minimize re-encode loss: single save with subsampling matching ffmpeg output
    work_e = args.work / "e"; work_e.mkdir(exist_ok=True)
    for f in list(work_e.glob("*.jpg")):
        f.unlink()
    for path in frames:
        (work_e / path.name).write_bytes(path.read_bytes())
    started = time.perf_counter()
    for index, path in enumerate(sorted(work_e.glob("frame_*.jpg"))):
        with Image.open(path) as original:
            with Image.new("RGB", (original.width, original.height + 24), "black") as result:
                result.paste(original, (0, 0))
                ImageDraw.Draw(result).text((6, original.height + 5), f"t={index:.3f}s", fill="white",
                                            font=ImageFont.load_default(size=13))
                result.save(path, "JPEG", quality=95, subsampling=2)
    results.append({"label": "E_python_footer_only_subsampling2", "sec": round(time.perf_counter() - started, 3),
                    "ok": True, "frames": len(frames)})

    # F: ffmpeg does everything: model-size scaled + padded + drawtext, single pass
    out_f = args.work / "f"; out_f.mkdir(exist_ok=True)
    for f in out_f.glob("*.jpg"):
        f.unlink()
    graph = (f"select='gte(t,selected_n/1)',scale=w='min(iw,{args.max_dim})'"
             f":h='min(ih,{args.max_dim - 24})':force_original_aspect_ratio=decrease,setsar=1,"
             f"pad=w=iw:h=ih+24:x=0:y=0:color=black,"
             f"drawtext=text='t=%{{pts\\:hms}}s':x=6:y=h-19:fontsize=13:fontcolor=white")
    ff = base + ["-ss", "0", "-t", str(duration), "-noautorotate", "-i", str(args.source),
                 "-map", "0:v:0", "-an", "-vf", graph, "-fps_mode", "vfr",
                 "-frames:v", str(count), "-q:v", "2", "-threads", t, str(out_f / "frame_%06d.jpg")]
    timed("F_ffmpeg_select_scale_pad_drawtext_qv2", ff, results)

    sizes = {}
    for name, directory in (("A", out), ("F", out_f)):
        got = sorted(directory.glob("*.jpg"))
        sizes[name] = {"files": len(got), "total_kib": round(sum(p.stat().st_size for p in got) / 1024, 1)}

    report = {"threads": args.threads, "frames": count, "native": native, "max_dim": args.max_dim,
              "device": os.cpu_count(), "results": results, "sizes_kib": sizes}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
