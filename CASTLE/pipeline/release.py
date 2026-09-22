#!/usr/bin/env python3
"""Release manifests for the CASTLE pipelines (see docs/release-process.md, R-03/R-06).

A release manifest is the one place that ties together:

* the human-readable release name (a git tag),
* the identity file set that ``batch_pipeline.code_hash()`` hashes,
* the auxiliary files that are published but *not* hashed,
* the prompt files whose text is stored inside run state.

``manifest`` writes one; ``verify`` re-derives everything from a tree and fails on
any drift. Both are stdlib-only on purpose: releasing must not depend on the
runtime dependencies being installed, and importing the pipeline would drag in
Pillow, the Google SDK and the HF client.

Usage:
  python release.py manifest --release castle-batch-v3 --tree . --provenance workspace
  python release.py verify   --release castle-batch-v3 --tree .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
RELEASES = REPO / "releases"

# Identity set: exactly what batch_pipeline.code_hash() hashes. Do not widen this
# without a new hash_scheme, because widening or re-keying it changes the
# fingerprint of already-pinned runs and makes them un-tickable.
IDENTITY_ROOT = ("batch_pipeline.py",)
IDENTITY_GLOB = "castle_pipeline/*.py"

# Published but not hashed: launchers and pinned dependency sets. A change here
# cannot be detected from a run's stored code_hash, which is exactly why it is
# recorded per release.
AUXILIARY = (
    "batch_jobs.py",
    "batch_tick.py",
    "bootstrap_batch.py",
    "jobs.py",
    "run_pipeline.py",
    "bootstrap.py",
    "Dockerfile",
    "requirements.txt",
    "requirements-batch.txt",
)
PROMPT_GLOB = "prompts/*.md"

HASH_SCHEMES = ("basename-v1", "relpath-v1")
DEFAULT_SCHEME = "basename-v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(data) -> str:
    """Byte-identical to castle_pipeline.runner.fingerprint."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def identity_paths(tree: Path) -> list[Path]:
    paths = [tree / name for name in IDENTITY_ROOT]
    paths += sorted(tree.glob(IDENTITY_GLOB))
    return [p for p in paths if p.is_file()]


def code_hash(tree: Path, scheme: str = DEFAULT_SCHEME) -> str:
    if scheme == "basename-v1":
        return fingerprint({p.name: sha256_file(p) for p in identity_paths(tree)})
    if scheme == "relpath-v1":
        return fingerprint(
            {p.relative_to(tree).as_posix(): sha256_file(p) for p in identity_paths(tree)}
        )
    raise ValueError(f"Unknown hash_scheme: {scheme}")


def collect(tree: Path, pattern_or_names, relative: bool = True) -> dict:
    """Map repo-relative path -> sha256 for an explicit name list or a glob."""
    out = {}
    if isinstance(pattern_or_names, str):
        found = sorted(tree.glob(pattern_or_names))
    else:
        found = [tree / name for name in pattern_or_names]
    for path in found:
        if path.is_file():
            out[path.relative_to(tree).as_posix()] = sha256_file(path)
    return out


def git_commit(tree: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(tree), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    commit = proc.stdout.strip()
    return commit if proc.returncode == 0 and commit else None


def git_root(tree: Path) -> Path | None:
    """Resolve the enclosing repository, even when the pipeline is a subdirectory.

    Publishing moved from a standalone repo (where the pipeline directory WAS
    the root) into a monorepo checkout; every git invocation below must anchor
    on the real root because ``git archive`` emits root-relative paths.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(tree), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    root = proc.stdout.strip() if proc.returncode == 0 else ""
    return Path(root) if root else None


def tagged_subtree(extracted: Path, tree: Path, root: Path | None) -> Path | None:
    """Rebase an extracted ``git archive`` tree onto the pipeline subtree."""
    if root is None or tree == root:
        return extracted
    subtree = extracted / tree.relative_to(root)
    return subtree if subtree.is_dir() else None


def build_manifest(release: str, tree: Path, *, scheme: str, provenance: str,
                   published_utc: str | None, notes: str) -> dict:
    identity = {p.relative_to(tree).as_posix(): sha256_file(p) for p in identity_paths(tree)}
    if not identity:
        raise ValueError(f"No identity files found under {tree}; wrong --tree?")
    return {
        "release": release,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "published_utc": published_utc,
        "provenance": {"kind": provenance, "detail": str(tree), "git_commit": git_commit(tree)},
        "hash_scheme": scheme,
        "code_hash": code_hash(tree, scheme),
        "identity_files": identity,
        "auxiliary_files": collect(tree, AUXILIARY),
        "prompt_files": collect(tree, PROMPT_GLOB),
        "notes": notes,
    }


def differences(manifest: dict, tree: Path) -> list[str]:
    problems = []
    scheme = manifest.get("hash_scheme", DEFAULT_SCHEME)
    if scheme not in HASH_SCHEMES:
        problems.append(f"unknown hash_scheme {scheme!r}")
        return problems

    actual = {p.relative_to(tree).as_posix(): sha256_file(p) for p in identity_paths(tree)}
    expected = manifest.get("identity_files") or {}
    for name in sorted(set(expected) | set(actual)):
        if name not in actual:
            problems.append(f"identity file missing from tree: {name}")
        elif name not in expected:
            problems.append(f"identity file not recorded in manifest: {name}")
        elif expected[name] != actual[name]:
            problems.append(f"identity file differs: {name}")

    if not problems:
        recomputed = code_hash(tree, scheme)
        if recomputed != manifest.get("code_hash"):
            problems.append(
                f"code_hash mismatch: manifest {manifest.get('code_hash')} vs tree {recomputed}"
            )

    for key, pattern in (("auxiliary_files", AUXILIARY), ("prompt_files", PROMPT_GLOB)):
        expected = manifest.get(key) or {}
        actual = collect(tree, pattern)
        for name in sorted(set(expected) | set(actual)):
            if name not in actual:
                problems.append(f"{key}: missing from tree: {name}")
            elif name not in expected:
                problems.append(f"{key}: not recorded in manifest: {name}")
            elif expected[name] != actual[name]:
                problems.append(f"{key}: differs: {name}")
    return problems


def load(release: str, releases_dir: Path) -> dict:
    path = releases_dir / f"{release}.json"
    if not path.is_file():
        raise SystemExit(f"No manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_manifest(args) -> int:
    tree = Path(args.tree).resolve()
    manifest = build_manifest(
        args.release, tree, scheme=args.hash_scheme, provenance=args.provenance,
        published_utc=args.published_utc, notes=args.notes,
    )
    args.releases_dir.mkdir(parents=True, exist_ok=True)
    out = args.releases_dir / f"{args.release}.json"
    if out.exists() and not args.force:
        existing = json.loads(out.read_text(encoding="utf-8"))
        if existing.get("code_hash") != manifest["code_hash"]:
            print(json.dumps({
                "ok": False,
                "error": "REFUSE_OVERWRITE",
                "detail": f"{out} already records code_hash {existing.get('code_hash')}; "
                          f"this tree computes {manifest['code_hash']}. "
                          f"A changed tree is a new release, not an edit.",
            }, indent=2))
            return 1
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    print(json.dumps({
        "ok": True, "release": manifest["release"], "code_hash": manifest["code_hash"],
        "hash_scheme": manifest["hash_scheme"],
        "identity_files": len(manifest["identity_files"]),
        "auxiliary_files": len(manifest["auxiliary_files"]),
        "prompt_files": len(manifest["prompt_files"]),
        "path": str(out),
    }, indent=2))
    return 0


def cmd_verify(args) -> int:
    tree = Path(args.tree).resolve()
    manifest = load(args.release, args.releases_dir)
    problems = differences(manifest, tree)
    print(json.dumps({
        "ok": not problems, "release": manifest["release"],
        "code_hash": manifest["code_hash"], "tree": str(tree),
        "problems": problems,
    }, indent=2, ensure_ascii=False))
    return 1 if problems else 0


CODE_BUCKET = "hf://buckets/Ligant/castle-code"


def _tag_exists(tree: Path, tag: str) -> bool:
    proc = subprocess.run(["git", "-C", str(tree), "tag", "--list", tag],
                          capture_output=True, text=True, check=False)
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _tag_tree(tree: Path, tag: str, work: Path) -> Path | None:
    """Extract exactly the tagged bytes, so a publish can never ship the worktree."""
    archive = work / "tag.tar"
    made = subprocess.run(["git", "-C", str(tree), "archive", "--format=tar",
                           "-o", str(archive), tag],
                          capture_output=True, text=True, check=False)
    if made.returncode != 0:
        return None
    target = work / "tree"
    target.mkdir(parents=True, exist_ok=True)
    import tarfile
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    return target


def list_bucket_objects(prefix: str) -> list[str]:
    proc = subprocess.run(["hf", "buckets", "list", prefix, "-R", "--format", "json"],
                          capture_output=True, text=True, check=False, shell=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Could not list {prefix}: {proc.stderr.strip()[:200]}")
    try:
        payload = json.loads(proc.stdout or "[]")
    except ValueError:
        return []
    names = []
    for item in payload if isinstance(payload, list) else payload.get("files", []):
        name = item.get("path") or item.get("name") or ""
        if name:
            names.append(name.split("/", 1)[-1] if name.startswith(prefix.split("/")[-1]) else name)
    return sorted(names)


def cmd_publish(args) -> int:
    """Publish a tagged release: verify, sync, read back, and refuse to overwrite.

    The bytes come from the git tag, never the working tree, because the tag is
    what the manifest's hashes describe. ``--verify-only`` runs every local check
    and skips the network, which is what CI and tests use.
    """
    tree = Path(args.tree).resolve()
    manifest = load(args.release, args.releases_dir)
    target_prefix = args.code_bucket.rstrip("/") + "/" + args.release
    problems = []

    root = git_root(tree)
    if root is None:
        problems.append("not a git repository")
    if not _tag_exists(tree, args.release):
        problems.append(f"tag {args.release} does not exist; publish identity is a tag (R-02)")

    with tempfile.TemporaryDirectory(prefix="release-publish-") as work_dir:
        work = Path(work_dir)
        tag_tree = _tag_tree(root or tree, args.release, work) if not problems else None
        if tag_tree is None and not problems:
            problems.append(f"git archive of tag {args.release} failed")
        if tag_tree is not None:
            tagged = tagged_subtree(tag_tree, tree, root)
            if tagged is None:
                problems.append(f"tag {args.release} does not contain {tree}")
            else:
                drift = differences(manifest, tagged)
                if drift:
                    problems.append(f"tagged tree disagrees with its manifest: {drift[:4]}")
            worktree_drift = differences(manifest, tree)
            if worktree_drift:
                problems.append(
                    "working tree does not match the manifest: "
                    f"{worktree_drift[:4]}; commit and re-tag before publishing")

    existing = []
    if not args.verify_only:
        try:
            existing = list_bucket_objects(target_prefix)
        except Exception as error:
            problems.append(str(error))

    if problems:
        print(json.dumps({"ok": False, "error": "PUBLISH_REFUSED", "release": args.release,
                          "problems": problems}, indent=2, ensure_ascii=False))
        return 1

    if args.verify_only:
        print(json.dumps({"ok": True, "release": args.release, "code_hash": manifest["code_hash"],
                          "verified": True, "published": False,
                          "would_sync": f"{tree} -> {target_prefix}"}, indent=2))
        return 0

    if existing and not args.force:
        print(json.dumps({
            "ok": False, "error": "REFUSE_OVERWRITE",
            "detail": f"{target_prefix} already holds {len(existing)} object(s); a published release "
                      "is immutable. A changed tree is a new release, not an edit.",
            "objects": existing[:5],
        }, indent=2, ensure_ascii=False))
        return 1

    with tempfile.TemporaryDirectory(prefix="release-stage-") as work_dir:
        work = Path(work_dir)
        tag_tree = _tag_tree(git_root(tree) or tree, args.release, work)
        if tag_tree is None:
            print(json.dumps({"ok": False, "error": "TAG_ARCHIVE_FAILED"}, indent=2))
            return 1
        tagged = tagged_subtree(tag_tree, tree, git_root(tree))
        if tagged is None:
            print(json.dumps({"ok": False, "error": "TAGGED_SUBTREE_MISSING",
                              "detail": f"tag {args.release} does not contain {tree}"}, indent=2))
            return 1
        # Publish exactly the manifest's file set. Exclude-patterns proved too easy
        # to get wrong: an earlier publish shipped the whole repo tree (tests,
        # ledger, tools) into an immutable prefix.
        staged = work / "staged"
        declared = sorted(set(manifest["identity_files"]) | set(manifest["auxiliary_files"])
                          | set(manifest["prompt_files"]))
        for name in declared:
            source = tagged / name
            if not source.is_file():
                print(json.dumps({"ok": False, "error": "MANIFEST_FILE_MISSING_FROM_TAG",
                                  "file": name}, indent=2))
                return 1
            destination = staged / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        proc = subprocess.run(["hf", "buckets", "sync", str(staged), target_prefix],
                              capture_output=True, text=True, check=False, shell=True)
        if proc.returncode != 0:
            print(json.dumps({"ok": False, "error": "SYNC_FAILED",
                              "detail": proc.stderr.strip()[-300:]}, indent=2))
            return 1
        expected_files = declared

    # Read back: every declared file must be present, and nothing undeclared may be.
    after = list_bucket_objects(target_prefix)
    missing = [name for name in expected_files if name not in after]
    unexpected = [name for name in after if name not in expected_files]
    print(json.dumps({"ok": not missing and not unexpected, "release": args.release,
                      "code_hash": manifest["code_hash"], "published": True,
                      "objects": len(after), "missing_from_bucket": missing,
                      "unexpected_in_bucket": unexpected,
                      "prefix": target_prefix}, indent=2, ensure_ascii=False))
    return 1 if (missing or unexpected) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("manifest", "verify", "publish"):
        p = sub.add_parser(name)
        p.add_argument("--release", required=True)
        p.add_argument("--tree", default=str(REPO), help="Tree to hash (default: this repo)")
        p.add_argument("--releases-dir", type=Path, default=RELEASES)
        if name == "manifest":
            p.add_argument("--hash-scheme", choices=HASH_SCHEMES, default=DEFAULT_SCHEME)
            p.add_argument("--provenance", default="workspace",
                           help="workspace | git | hf-bucket | other")
            p.add_argument("--published-utc", default=None,
                           help="When this release was pushed to the code bucket")
            p.add_argument("--notes", default="")
            p.add_argument("--force", action="store_true",
                           help="Rewrite an existing manifest even if code_hash differs")
        if name == "publish":
            p.add_argument("--code-bucket", default=CODE_BUCKET)
            p.add_argument("--verify-only", action="store_true",
                           help="Run every local check and skip the network")
            p.add_argument("--force", action="store_true",
                           help="Overwrite an existing release prefix (breaks immutability; avoid)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"manifest": cmd_manifest, "verify": cmd_verify, "publish": cmd_publish}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
