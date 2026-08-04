"""download_video.py — download raw EgoLife videos into the layout
`caption_pipeline.py` expects by default.

Lays files out at ``videos/{participant}/DAY{day}/DAY{day}_{participant}_{HHMMSScc}.mp4``
under this script's directory (the same ROOT caption_pipeline.py resolves), so
caption_pipeline.py finds them with zero extra arguments.

Source: ``lmms-lab/EgoLife`` on Hugging Face. On the repo the tree is
``{participant}/{day}/DAY{day}_{participant}_{ts}.mp4``; we point
``huggingface_hub.snapshot_download`` at ``ROOT/videos`` so that prefix lands
exactly where caption_pipeline.py's default ``--src-dir`` points.

Why huggingface_hub here (and not plain HTTP like download_egolifecap.py)?
Each video is ~10 MB+, and huggingface_hub's per-file fixed overhead (metadata
HEAD + lock + cache check) is amortized by the larger payload. For the tiny
caption text files handled by download_egolifecap.py that same overhead
dominates, so that script uses parallel plain HTTP instead.

Usage
-----
    # default: A1_JAKE / DAY1 (matches caption_pipeline.py defaults)
    python download_video.py
    # pick participant / days
    python download_video.py --participant A1_JAKE A2_ALICE --day 1 2 3
    # everything (6 participants x 7 days, ~hundreds of GB)
    python download_video.py --participant all --day all

Note: a single participant/day is ~10 GB+ and takes on the order of hours.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ROOT = the directory this script lives in (.../ARIN7600/EgoLife/), the same
# ROOT caption_pipeline.py resolves. Layout is laid out under it so the two
# scripts agree on the data location with no extra arguments.
ROOT = Path(__file__).resolve().parent

REPO_ID = "lmms-lab/EgoLife"
REPO_TYPE = "dataset"

PARTICIPANTS = ["A1_JAKE", "A2_ALICE", "A3_TASHA", "A4_LUCIA", "A5_KATRINA", "A6_SHURE"]
DAYS = [f"DAY{i}" for i in range(1, 8)]  # DAY1 .. DAY7

HF_TOKEN = os.environ.get("HF_TOKEN", None)


def _gb(n: float) -> str:
    return f"{n / (1024 ** 3):.2f} GB"


def _count_files(path: Path) -> int:
    n = 0
    if path.exists():
        for _, _, files in os.walk(path):
            n += len(files)
    return n


def _dir_size(path: Path) -> int:
    total = 0
    if path.exists():
        for _, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(_, f))
                except OSError:
                    pass
    return total


def _resolve_participants(arg) -> list[str]:
    if not arg or "all" in arg:
        return list(PARTICIPANTS)
    return arg


def _resolve_days(arg) -> list[str]:
    if not arg or "all" in arg:
        return list(DAYS)
    return [d if str(d).upper().startswith("DAY") else f"DAY{d}" for d in arg]


def download_videos(participants, days, token=None, max_workers=4, out_dir=None):
    """Download raw videos into ``<out_dir>/{participant}/DAY{day}/``.

    out_dir defaults to ``ROOT/videos``, which is where caption_pipeline.py's
    default ``--src-dir`` points. If you override it, also pass
    ``--src-dir <out_dir>/{participant}/DAY{day}`` to caption_pipeline.py."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print("ERROR: `pip install huggingface_hub` is required for video download.",
              file=sys.stderr)
        raise SystemExit(2) from e

    out_dir = Path(out_dir) if out_dir else (ROOT / "videos")
    local_root = out_dir
    local_root.mkdir(parents=True, exist_ok=True)

    tasks = [(p, d) for p in participants for d in days]
    print(f"\n下载 EgoLife 视频")
    print(f"  仓库:   {REPO_ID} ({REPO_TYPE})")
    try:
        show_root = str(local_root.relative_to(ROOT))
    except ValueError:
        show_root = str(local_root)  # outside ROOT (custom --out-dir)
    print(f"  保存到: {show_root}/<participant>/<DAY>/")
    print(f"  布局:   <out_dir>/{{participant}}/DAY{{day}}/DAY{{day}}_{{participant}}_{{HHMMSScc}}.mp4")
    if out_dir is None:
        print(f"  (默认即 caption_pipeline.py 的 --src-dir；改了 --out-dir 则需相应传 --src-dir)")
    else:
        print(f"  ⚠ 自定义位置：跑 caption_pipeline.py 时需加 --src-dir \"{show_root}/{{participant}}/DAY{{day}}\"")
    print(f"  任务: {len(tasks)} 个 ({len(participants)} 人 × {len(days)} 天)")
    print(f"  认证: {'HF_TOKEN / --token' if token else '无 (公开数据集)'}")
    print(f"  ⚠ 单 participant/day ~10 GB+，量级以小时计\n")

    t0 = time.time()
    total_files = 0
    total_bytes = 0
    failed: list[str] = []

    for i, (p, d) in enumerate(tasks, 1):
        pattern = f"{p}/{d}/*"
        dest = local_root / p / d
        print(f"[{i}/{len(tasks)}] {p}/{d} -> {dest}")
        ts = time.time()
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                allow_patterns=pattern,
                local_dir=str(local_root),
                token=token or HF_TOKEN,
                max_workers=max_workers,
            )
            n = _count_files(dest)
            sz = _dir_size(dest)
            total_files += n
            total_bytes += sz
            print(f"   OK  {n} 文件, {_gb(sz)}  ({time.time() - ts:.1f}s)\n")
        except Exception as e:
            print(f"   FAIL  {type(e).__name__}: {e}\n")
            failed.append(f"{p}/{d}")

    print("=" * 70)
    print(f"视频下载完成: {len(tasks) - len(failed)}/{len(tasks)} 成功, "
          f"{total_files} 文件, {_gb(total_bytes)}, 用时 {(time.time() - t0) / 60:.1f} 分钟")
    if failed:
        print("失败:")
        for t in failed:
            print(f"  - {t}")
    print()


def main():
    ap = argparse.ArgumentParser(
        prog="download_video.py",
        description="下载 EgoLife 视频到 videos/{participant}/DAY{day}/（caption_pipeline.py 默认布局）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--participant", nargs="+", choices=PARTICIPANTS + ["all"],
                    default=["A1_JAKE"],
                    help="参与者，或 'all'（可多选）")
    ap.add_argument("--day", nargs="+", default=["DAY1"],
                    help="天数，如 1 / DAY1 / 'all'，可多选（默认 DAY1）")
    ap.add_argument("--token", default=None, help="HuggingFace token（公开数据集可省略）")
    ap.add_argument("--max-workers", type=int, default=4,
                    help="huggingface_hub 并发下载线程数")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="视频保存根目录（默认 ./videos）。改了它，跑 caption_pipeline.py 时"
                         "需相应传 --src-dir <out-dir>/{participant}/DAY{day}")
    args = ap.parse_args()

    participants = _resolve_participants(args.participant)
    days = _resolve_days(args.day)
    download_videos(participants, days, token=args.token, max_workers=args.max_workers,
                    out_dir=args.out_dir)


if __name__ == "__main__":
    main()
