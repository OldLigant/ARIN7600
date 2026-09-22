"""Offline process-tree memory benchmark; requires an explicit fresh output dir.

Example (run from pipeline):
    python tests/benchmark_memory.py --output _test/memory-benchmark-20260916

Each comparison executes in a fresh Python process. All generated media,
checkpoints, logs, and reports stay below the explicitly supplied directory.
The synthetic flat scene understates real egocentric JPEG payload sizes.
"""
import argparse
import base64
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from castle_pipeline.media import probe
from castle_pipeline.runner import MemoryMonitor, Pipeline, RunConfig, rss_bytes


class SerializingOfflineProvider:
    """Read media, retain raw and base64 payloads, serialize JSON, then release."""
    model = "offline-memory-benchmark-no-network"
    service_tier = "offline"

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = []

    def generate(self, prompt, context, images, audio=None, max_output_tokens=32768):
        phase = json.loads(context)["task_phase"]
        started = time.monotonic()
        raw = [path.read_bytes() for _, path in images]
        audio_raw = audio.read_bytes() if audio else b""
        payload = {"prompt": prompt, "context": context,
                   "images": [{"label": label, "data": base64.b64encode(data).decode("ascii")}
                              for (label, _), data in zip(images, raw)],
                   "audio": base64.b64encode(audio_raw).decode("ascii") if audio else None}
        serialized = json.dumps(payload).encode("utf-8")
        # Keep this transient working set alive longer than the monitor's 0.2 s
        # sample interval. This is a payload allocation surrogate, not an SDK.
        time.sleep(0.4)
        record = {"phase": phase, "image_count": len(raw),
                  "jpeg_total_bytes": sum(map(len, raw)),
                  "largest_jpeg_bytes": max(map(len, raw), default=0),
                  "audio_bytes": len(audio_raw), "serialized_payload_bytes": len(serialized),
                  "held_payload_sec": round(time.monotonic() - started, 3)}
        with self.lock:
            self.requests.append(record)
        del serialized, payload, raw, audio_raw
        if phase == "audio":
            data = {"summary": "Synthetic tone.", "utterances": [], "sound_events": [],
                    "uncertainties": []}
        elif phase == "annotation":
            data = {"schema_version": "castle-caption-v1", "scene_summary": "Synthetic flat scene.",
                    "actors": [], "initial_environment": None, "visual_unavailable_intervals": [],
                    "segments": [], "activity_chain": []}
        else:
            raise ValueError("The benchmark disables detail review")
        return {"data": data, "usage": {}, "attempts": 1}


def run_case(source, directory):
    directory.mkdir(parents=True, exist_ok=False)
    provider = SerializingOfflineProvider()
    config = RunConfig(output_dir=directory / "checkpoints", scratch_dir=directory / "scratch",
                       workers=3, fps=1, clip_seconds=30, max_dim=1440, review=False,
                       stamp=True, keep_media=False, memory_soft_limit_gib=12)
    baseline = rss_bytes()
    started = time.monotonic()
    summary = Pipeline(config, provider).run_source(source, {
        "source_id": source.stem, "viewpoint": "egocentric", "synthetic": True,
        "source_path": str(source), "source_bytes": source.stat().st_size})
    gc.collect()
    result = {"source": str(source), "source_probe": probe(source),
              "implementation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in (Path(__file__).resolve().parents[1] / "castle_pipeline").glob("*.py")},
              "source_file_bytes": source.stat().st_size,
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "baseline_process_tree_rss_mib": round(baseline / 2**20, 2),
              "post_gc_process_tree_rss_mib": round(rss_bytes() / 2**20, 2),
              "summary": summary,
              "jpeg_total_bytes": sum(r["jpeg_total_bytes"] for r in provider.requests),
              "jpeg_peak_per_request_bytes": max((r["jpeg_total_bytes"] for r in provider.requests), default=0),
              "largest_jpeg_bytes": max((r["largest_jpeg_bytes"] for r in provider.requests), default=0),
              "peak_serialized_request_bytes": max((r["serialized_payload_bytes"] for r in provider.requests), default=0),
              "requests": provider.requests}
    (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if summary["failed"] or summary["unprocessed"] or summary["reused"]:
        raise RuntimeError("Benchmark case did not complete every fresh clip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="Fresh dedicated test-output directory; must not already exist")
    parser.add_argument("--case-source", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.case_source:
        run_case(args.case_source.resolve(), output)
        return
    output.mkdir(parents=True, exist_ok=False)
    commands = []

    def command(args, logfile):
        commands.append(args)
        print("RUN " + subprocess.list2cmdline(args), flush=True)
        with (output / logfile).open("wb") as stream:
            subprocess.run(args, check=True, stdout=stream, stderr=subprocess.STDOUT)

    seed = output / "seed-4k50-2sec.mkv"
    with MemoryMonitor() as generation_monitor:
        command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi",
                 "-i", "color=c=blue:s=3840x2160:r=50:d=2", "-f", "lavfi", "-i",
                 "sine=frequency=440:sample_rate=48000:duration=2", "-c:v", "libx264",
                 "-preset", "ultrafast", "-tune", "zerolatency", "-threads", "1",
                 "-pix_fmt", "yuv420p", "-g", "50", "-c:a", "pcm_s16le", str(seed)], "seed-generation.log")
        for seconds in (30, 180):
            target = output / f"source-{seconds:03d}s.mkv"
            command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                     "-stream_loop", "-1", "-i", str(seed), "-t", str(seconds),
                     "-map", "0", "-c", "copy", str(target)], f"copy-{seconds:03d}.log")
    cases = []
    for seconds in (30, 180):
        case_dir = output / f"case-{seconds:03d}s"
        command([sys.executable, str(Path(__file__).resolve()), "--output", str(case_dir),
                 "--case-source", str(output / f"source-{seconds:03d}s.mkv")], f"case-{seconds:03d}.log")
        cases.append(json.loads((case_dir / "result.json").read_text(encoding="utf-8")))
    peaks = [case["summary"]["memory"]["peak_process_tree_rss_mib"] for case in cases]
    report = {"platform": platform.platform(), "python": sys.version,
              "configuration": {"source_width": 3840, "source_height": 2160, "source_fps": 50,
                                "workers": 3, "decode_slots": 1, "ffmpeg_threads": 1,
                                "sampling_fps": 1, "clip_seconds": 30, "max_dim": 1440,
                                "timestamp_footer": True, "audio": "mono 16kHz PCM16",
                                "request_payload_hold_seconds": 0.4},
              "invocation": [sys.executable, *sys.argv], "commands": commands,
              "source_generation_memory_excluded_from_comparison": generation_monitor.result(),
              "cases": cases,
              "comparison": {"short_peak_process_tree_rss_mib": peaks[0],
                             "long_peak_process_tree_rss_mib": peaks[1],
                             "long_minus_short_mib": round(peaks[1] - peaks[0], 2),
                             "long_over_short_ratio": round(peaks[1] / peaks[0], 3)},
              "limitations": ["Synthetic flat blue scene understates real egocentric JPEG bytes.",
                              "Ultrafast flat H264 uses a simple reference structure and may understate production decoder memory.",
                              "Offline base64 plus JSON allocation surrogate; no Vertex SDK/network/model invocation.",
                              "A 30-second case has one clip; 180 seconds exercises six clips with three admitted workers.",
                              "Process-tree RSS includes FFmpeg but may double-count shared pages and miss sub-0.2s peaks.",
                              "Windows process RSS does not measure Linux file cache or Hugging Face cgroup memory.",
                              "This local result does not prove a Hugging Face 16GB job succeeds.",
                              "No hard pass/fail memory threshold; compare repeated real-content cloud measurements."]}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    command_text = "\n".join(subprocess.list2cmdline(args) for args in report["commands"]) + "\n"
    (output / "commands.txt").write_text(command_text, encoding="utf-8")
    print(json.dumps({"report": str(output / "report.json"), "comparison": report["comparison"],
                      "completed_clips": [case["summary"]["completed"] for case in cases]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
