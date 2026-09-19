#!/usr/bin/env python3
"""Probe the real CASTLE source video's encoded frame structure using ranged reads.

No huggingface_hub needed: the repo, revision and path are already pinned.
Fetches the head of the file (and the tail if the moov atom is not at the front),
then asks ffprobe about keyframe spacing and frame types. Read-only.
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "CASTLE-Dataset/CASTLE2024"
REVISION = "c8e7b5cd9e9c83d0ff42560fc1169bed7867abd4"
PATH = "main/day1/Bjorn/video/08.mp4"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{PATH}"
SRC = Path(__file__).parent / "src"
CHUNK = 16 * 1024 * 1024


def fetch(byte_range, target):
    request = urllib.request.Request(URL, headers={"Range": f"bytes={byte_range}"})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
        return response.status, response.headers.get("Content-Range"), data


def ffprobe_json(path, entries):
    proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                           "-show_entries", entries, "-of", "json", str(path)],
                          capture_output=True, check=False)
    return proc.stdout.decode(errors="replace")


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    try:
        status, content_range, head = fetch(f"0-{CHUNK - 1}", None)
    except Exception as error:
        print(json.dumps({"error": type(error).__name__, "detail": str(error)[:200]}))
        return 1
    head_path = SRC / "head.mp4"
    head_path.write_bytes(head)
    print(json.dumps({"http_status": status, "content_range": content_range,
                      "bytes": len(head), "saved": str(head_path)}, indent=2))

    print("=== stream (head) ===")
    print(ffprobe_json(head_path, "stream=codec_name,profile,level,width,height,r_frame_rate,"
                                  "avg_frame_rate,bit_rate,nb_frames,pix_fmt,has_b_frames,codec_tag_string"))

    # If the moov atom is at the end, the head alone is undecodable; fetch the tail too.
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "json", str(head_path)], capture_output=True, check=False)
    if probe.returncode != 0 or not probe.stdout.strip():
        print("head not decodable alone; fetching tail range")
        try:
            status, content_range, tail = fetch(f"-{CHUNK}", None)
            tail_path = SRC / "tail.mp4"
            tail_path.write_bytes(tail)
            print(json.dumps({"tail_status": status, "tail_range": content_range,
                              "tail_bytes": len(tail)}, indent=2))
            print("=== tail format ===")
            print(ffprobe_json(tail_path, "format=duration,size,bit_rate"))
        except Exception as error:
            print(json.dumps({"tail_error": type(error).__name__, "detail": str(error)[:200]}))

    print("=== first 40 decoded frames from head ===")
    frames = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "frame=pict_type,pkt_size,key_frame,best_effort_timestamp_time",
                             "-of", "csv=p=0", str(head_path)],
                            capture_output=True, check=False)
    rows = [r for r in frames.stdout.decode(errors="replace").splitlines() if r.strip()]
    for line in rows[:40]:
        print(line)
    keys = [i for i, line in enumerate(rows) if line.split(",")[-1].strip() == "1"]
    print(json.dumps({"frames_read": len(rows), "keyframe_positions": keys[:40]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
