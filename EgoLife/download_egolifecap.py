"""download_egolifecap.py — download EgoLife official captions (DenseCaption
and/or Transcript) from ``lmms-lab/EgoLife`` on Hugging Face.

Lays files out at ``EgoLifeCap/{kind}/{participant}/DAY{day}/*.srt`` under this
script's directory (ROOT). This is deliberately separate from:
  * ``videos/`` — raw video input (handled by download_video.py)
  * ``captions/`` — *our own* pipeline output (written by caption_pipeline.py)
These are the *dataset authors'* official annotations, neither raw input nor
our pipeline's product, so they get their own ``EgoLifeCap/`` tree.

Which kind to fetch is selected by ``--kind``:
  * ``dense``     -> EgoLifeCap/DenseCaption/<participant>/<DAY>/*.srt
  * ``transcript``-> EgoLifeCap/Transcript/<participant>/<DAY>/*.srt
  * ``all``       -> both

Download method (``--method``):
  * ``http`` (default) — parallel plain HTTP against the HF resolve URLs.
    No ``huggingface_hub`` dependency. Benchmark on one participant/day
    (~500 KiB, 10 files): ~3 s vs ~40 s for huggingface_hub. The HF client's
    per-file fixed overhead (metadata HEAD + lock + cache check) is amortized
    by the ~10 MB video files but dominates for these tiny text files, which
    is why download_video.py uses huggingface_hub while this script does not.
  * ``hf`` — huggingface_hub.snapshot_download. Slower for many tiny files
    but kept as a robust fallback (matches the legacy internal script).
  * ``git_lfs`` — not recommended; the repo is ~600 GB so even a sparse
    ``git clone`` spends minutes in tree/commit metadata negotiation before
    touching a byte (the fixed cost has nothing to do with how little you
    actually fetch).

Usage
-----
    # default: DenseCaption, A1_JAKE / DAY1, http
    python download_egolifecap.py
    # Transcript instead
    python download_egolifecap.py --kind transcript
    # both, all participants, all days
    python download_egolifecap.py --kind all --participant all --day all
    # fall back to huggingface_hub
    python download_egolifecap.py --method hf
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ROOT = .../ARIN7600/EgoLife/, same ROOT caption_pipeline.py resolves.
ROOT = Path(__file__).resolve().parent

REPO_ID = "lmms-lab/EgoLife"
REPO_TYPE = "dataset"
HF_TREE_API = f"https://huggingface.co/api/{REPO_TYPE}s/{REPO_ID}/tree/main"
HF_RESOLVE = f"https://huggingface.co/{REPO_TYPE}s/{REPO_ID}/resolve/main"

PARTICIPANTS = ["A1_JAKE", "A2_ALICE", "A3_TASHA", "A4_LUCIA", "A5_KATRINA", "A6_SHURE"]
DAYS = [f"DAY{i}" for i in range(1, 8)]  # DAY1 .. DAY7

# --kind values -> repo subdirectory under EgoLifeCap/
KIND_DIRS = {"dense": "DenseCaption", "transcript": "Transcript"}

HF_TOKEN = os.environ.get("HF_TOKEN", None)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mb(n: float) -> str:
    return f"{n / (1024 ** 2):.2f} MB"


def _count_files(path: Path) -> int:
    n = 0
    if path.exists():
        for _, _, files in os.walk(path):
            n += len(files)
    return n


def _dir_size(path: Path) -> int:
    total = 0
    if path.exists():
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
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


def _resolve_kinds(arg) -> list[str]:
    """--kind values -> list of KIND_DIRS keys (dense / transcript)."""
    if not arg or "all" in arg:
        return list(KIND_DIRS.keys())
    return arg


# ===========================================================================
# HTTP path (default)
# ===========================================================================

def _list_repo_dir(rel_path: str, token: str | None) -> list[dict]:
    """List a directory in the HF repo via the tree API."""
    url = f"{HF_TREE_API}/{quote(rel_path)}"
    req = urllib.request.Request(url, headers={"User-Agent": "download_egolifecap/1.0"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _list_caption_files(kind_dir: str, participant: str, day: str,
                        token: str | None) -> list[str]:
    """Full repo-relative paths under ``EgoLifeCap/{kind_dir}/{p}/{d}/``,
    e.g. ``EgoLifeCap/DenseCaption/A1_JAKE/DAY1/A1_JAKE_DAY1_11000000.srt``."""
    prefix = f"EgoLifeCap/{kind_dir}/{participant}/{day}"
    try:
        items = _list_repo_dir(prefix, token)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise
    return [it["path"] for it in items if it.get("type") == "file"]


def _download_one_file(repo_rel_path: str, dest: Path, token: str | None,
                       ctx: ssl.SSLContext, retries: int = 3) -> int:
    """Download one file via the HF resolve URL. Idempotent: skips if dest
    already exists with non-zero size."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_RESOLVE}/{quote(repo_rel_path)}"
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "download_egolifecap/1.0"})
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
                data = r.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return len(data)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"download failed for {repo_rel_path}: "
                       f"{type(last_err).__name__}: {last_err}")


def _download_kind_http(kind_dir, participants, days, out_root: Path,
                        token=None, max_workers=8) -> tuple[int, int, list[str]]:
    """Parallel HTTP download for one kind. Returns (n_files, n_bytes, missing)."""
    ctx = ssl.create_default_context()
    jobs: list[tuple[str, Path]] = []
    missing: list[str] = []

    for p in participants:
        for d in days:
            files = _list_caption_files(kind_dir, p, d, token)
            if not files:
                missing.append(f"{kind_dir}/{p}/{d}")
                continue
            for rel in files:
                # rel = "EgoLifeCap/<kind>/<p>/<d>/<file>"; strip the
                # "EgoLifeCap/<kind>/" prefix so it lands under
                # out_root/<kind>/<p>/<d>/<file>.
                rel_after_kind = rel[len("EgoLifeCap/") + len(kind_dir) + 1:]
                jobs.append((rel, out_root / kind_dir / rel_after_kind))

    total_bytes = 0
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_download_one_file, rel, dest, token, ctx): rel
                for rel, dest in jobs}
        for fut in as_completed(futs):
            rel = futs[fut]
            try:
                total_bytes += fut.result()
                done += 1
            except Exception as e:
                missing.append(f"{rel} ({type(e).__name__})")
    return done, total_bytes, missing


# ===========================================================================
# huggingface_hub path (fallback)
# ===========================================================================

def _download_kind_hf(kind_dir, participants, days, out_root: Path,
                      token=None) -> tuple[int, int, list[str]]:
    """huggingface_hub.snapshot_download for one kind.

    snapshot_download preserves repo-relative paths under local_dir. The repo
    prefix is ``EgoLifeCap/<kind>/...``; to make files land at
    ``out_root/<kind>/...`` we set ``local_dir = out_root.parent`` so the repo's
    leading "EgoLifeCap/" becomes out_root itself. This requires
    ``out_root.name == "EgoLifeCap"`` (the default). For a custom --out-dir with
    a different name, use ``--method http`` instead, which is unrestricted."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print("ERROR: `--method hf` 需要 huggingface_hub。请 `pip install huggingface_hub`，"
              "或改用 `--method http`（默认，更快且无此依赖）。", file=sys.stderr)
        raise SystemExit(2) from e

    if out_root.name != "EgoLifeCap":
        print(f"ERROR: --method hf 要求 --out-dir 的末尾目录名为 'EgoLifeCap'（得到 "
              f"'{out_root.name}'），因为 HF 仓库路径带 'EgoLifeCap/' 前缀。"
              f"请改用 --method http（默认，无此限制），或把 --out-dir 指向一个叫 "
              f"'EgoLifeCap' 的目录。", file=sys.stderr)
        raise SystemExit(2)

    local_dir = out_root.parent  # repo prefix "EgoLifeCap/..." lands here as out_root/...
    total_files = 0
    total_bytes = 0
    failed: list[str] = []
    for p in participants:
        for d in days:
            pattern = f"EgoLifeCap/{kind_dir}/{p}/{d}/*"
            expected = out_root / kind_dir / p / d
            try:
                snapshot_download(
                    repo_id=REPO_ID,
                    repo_type=REPO_TYPE,
                    allow_patterns=pattern,
                    local_dir=str(local_dir),
                    token=token or HF_TOKEN,
                )
                total_files += _count_files(expected)
                total_bytes += _dir_size(expected)
            except Exception as e:
                failed.append(f"{kind_dir}/{p}/{d} ({type(e).__name__}: {e})")
    return total_files, total_bytes, failed


# ===========================================================================
# git lfs path (not recommended)
# ===========================================================================

def _download_kind_git_lfs(kind_dir, participants, days, out_root: Path) -> None:
    """Optional git-lfs path. NOT recommended: the repo is ~600 GB so even a
    sparse clone spends minutes in tree/commit metadata negotiation before
    fetching anything. The fixed cost is independent of how little you fetch."""
    import shutil
    import subprocess

    tmp = out_root / ".gitlfs_clone"
    if not tmp.exists():
        tmp.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             f"https://huggingface.co/{REPO_TYPE}s/{REPO_ID}", str(tmp)],
            check=True, env=env,
        )
    includes = [f"EgoLifeCap/{kind_dir}/{p}/{d}/*" for p in participants for d in days]
    subprocess.run(
        ["git", "-C", str(tmp), "lfs", "pull", "--include", ",".join(includes)],
        check=True, env=dict(os.environ),
    )
    src = tmp / "EgoLifeCap" / kind_dir
    dst = out_root / kind_dir
    if src.exists():
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.rglob("*"):
            if item.is_file():
                rel = item.relative_to(src)
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)


# ===========================================================================
# driver
# ===========================================================================

def download_captions(kind_keys, participants, days, *, method="http",
                      token=None, max_workers=8, out_dir=None):
    """Download the selected kind(s) into ``<out_dir>/`` (an ``EgoLifeCap/``
    tree). out_dir defaults to ``ROOT/EgoLifeCap``."""
    kind_dirs = [KIND_DIRS[k] for k in kind_keys]
    out_root = Path(out_dir) if out_dir else (ROOT / "EgoLifeCap")
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        show_root = out_root.relative_to(ROOT)
    except ValueError:
        show_root = out_root  # custom --out-dir outside ROOT
    print(f"\n下载 EgoLife caption ({', '.join(kind_dirs)})")
    print(f"  仓库:   {REPO_ID}")
    print(f"  方法:   {method}")
    print(f"  保存到: {show_root}/<kind>/<participant>/<DAY>/")
    print(f"  任务:   {len(participants)} 人 × {len(days)} 天 × {len(kind_dirs)} 类\n")

    t0 = time.time()
    grand_files = 0
    grand_bytes = 0
    all_failed: list[str] = []

    for kind_dir in kind_dirs:
        print(f"-- {kind_dir} --")
        if method == "http":
            n, sz, failed = _download_kind_http(
                kind_dir, participants, days, out_root, token=token,
                max_workers=max_workers)
        elif method == "git_lfs":
            _download_kind_git_lfs(kind_dir, participants, days, out_root)
            n, sz, failed = _count_files(out_root / kind_dir), \
                _dir_size(out_root / kind_dir), []
        else:  # "hf"
            n, sz, failed = _download_kind_hf(
                kind_dir, participants, days, out_root, token=token)
        grand_files += n
        grand_bytes += sz
        all_failed.extend(failed)
        print(f"   {kind_dir}: {n} 文件, {_mb(sz)}\n")

    print("=" * 70)
    print(f"caption 下载完成: {grand_files} 文件, {_mb(grand_bytes)}, "
          f"用时 {time.time() - t0:.1f}s")
    if all_failed:
        print("缺失/失败 (可能是该参与者该天无数据):")
        for f in all_failed:
            print(f"  - {f}")
    print()


def main():
    ap = argparse.ArgumentParser(
        prog="download_egolifecap.py",
        description="下载 EgoLife 官方 caption (DenseCaption / Transcript) 到 EgoLifeCap/。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--kind", nargs="+", default=["dense"],
                    choices=["dense", "transcript", "all"],
                    help="dense=DenseCaption, transcript=Transcript, all=两者都要")
    ap.add_argument("--participant", nargs="+", choices=PARTICIPANTS + ["all"],
                    default=["A1_JAKE"],
                    help="参与者，或 'all'（可多选）")
    ap.add_argument("--day", nargs="+", default=["DAY1"],
                    help="天数，如 1 / DAY1 / 'all'，可多选（默认 DAY1）")
    ap.add_argument("--method", default="http", choices=["http", "hf", "git_lfs"],
                    help="http=并行直连 (默认, 最快, 无依赖); "
                         "hf=huggingface_hub (稳但慢); git_lfs=极慢不推荐")
    ap.add_argument("--token", default=None, help="HuggingFace token（公开数据集可省略）")
    ap.add_argument("--max-workers", type=int, default=8,
                    help="--method http 时的并发线程数")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="caption 保存根目录（默认 ./EgoLifeCap）。注意：--method hf 要求"
                         "该目录末尾名为 'EgoLifeCap'；--method http 无此限制")
    args = ap.parse_args()

    kind_keys = _resolve_kinds(args.kind)
    participants = _resolve_participants(args.participant)
    days = _resolve_days(args.day)
    download_captions(kind_keys, participants, days, method=args.method,
                      token=args.token, max_workers=args.max_workers, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
