"""Immutable HF source selection; large media downloaded one file at a time."""
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

REPO = 'CASTLE-Dataset/CASTLE2024'
REVISION = 'c8e7b5cd9e9c83d0ff42560fc1169bed7867abd4'
STATIC = {'Kitchen', 'Living1', 'Living2', 'Meeting', 'Reading'}
PATTERN = re.compile(r'^main/(day[1-4])/([A-Za-z0-9_-]+)/video/(0[89]|1[0-9]|20)\.mp4$')


def source_metadata(path, revision):
    match = PATTERN.fullmatch(path)
    if not match:
        raise ValueError('Expected main/day[1-4]/Stream/video/HH.mp4')
    day, stream, hour = match.groups()
    return {'source_id': f'{REPO}@{revision}/{path}', 'repo': REPO, 'revision': revision, 'path': path,
            'day': day, 'date': f'2024-12-{2 + int(day[-1]):02d}', 'stream': stream, 'hour': int(hour),
            'viewpoint': 'exocentric' if stream in STATIC else 'egocentric'}


def select_sources(manifest, day=None, stream=None, viewpoint='ego', shard_index=0, shard_count=1, hour=None):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError('Require 0 <= shard_index < shard_count')
    selected = {}
    with Path(manifest).open(encoding='utf-8-sig') as handle:
        for line in handle:
            row = json.loads(line)
            path = row.get('video', row.get('path'))
            if not path or not PATTERN.fullmatch(path):
                continue
            meta = source_metadata(path, row.get('revision', REVISION))
            if day and meta['day'] != day or stream and meta['stream'] != stream:
                continue
            if hour is not None and meta['hour'] != hour:
                continue
            if viewpoint == 'ego' and meta['viewpoint'] != 'egocentric' or viewpoint == 'exo' and meta['viewpoint'] != 'exocentric':
                continue
            selected[path] = {'path': path, 'bytes': row.get('video_bytes', row.get('bytes', 0)),
                              'revision': row.get('revision')}
    return [selected[path] for i, path in enumerate(sorted(selected)) if i % shard_count == shard_index]


def list_remote(revision=REVISION, day=None, stream=None):
    from huggingface_hub import HfApi, RepoFile
    api = HfApi()
    commit = api.dataset_info(REPO, revision=revision).sha
    prefix = 'main' + (f'/{day}' if day else '') + (f'/{stream}' if day and stream else '')
    result = []
    for entry in api.list_repo_tree(REPO, path_in_repo=prefix, recursive=True, repo_type='dataset', revision=commit):
        if isinstance(entry, RepoFile) and PATTERN.fullmatch(entry.path):
            metadata = source_metadata(entry.path, commit)
            if stream and metadata['stream'] != stream:
                continue
            result.append({'path': entry.path, 'bytes': entry.size, 'revision': commit})
    return sorted(result, key=lambda r: r['path']), commit


def download_source(path, revision, scratch_dir):
    from huggingface_hub import HfApi, hf_hub_download
    # Path validation also prevents a malicious manifest escaping local_dir.
    source_metadata(path, revision)
    entries = HfApi().get_paths_info(REPO, [path], repo_type='dataset', revision=revision)
    if len(entries) != 1 or not getattr(entries[0], 'size', 0):
        raise ValueError('Source file is absent or empty')
    size = entries[0].size
    download_root = Path(scratch_dir) / 'downloads'
    download_root.mkdir(parents=True, exist_ok=True)
    # Each local invocation owns its source, avoiding another run's cleanup.
    directory = Path(tempfile.mkdtemp(prefix=hashlib.sha256((revision + path).encode()).hexdigest()[:12] + '-', dir=download_root))
    expected = directory / path
    additional = 0 if expected.exists() and expected.stat().st_size == size else size
    if shutil.disk_usage(directory).free < additional + 2 * 2**30:
        raise RuntimeError('Insufficient scratch disk for one source video plus a 2 GiB reserve')
    local = Path(hf_hub_download(REPO, path, repo_type='dataset', revision=revision, local_dir=directory))
    if local.stat().st_size != size or not local.resolve().is_relative_to(directory.resolve()):
        raise RuntimeError('Downloaded source has unexpected size or location')
    return local


def remove_downloaded_source(path, scratch_dir):
    """Inventory/check exactly the downloaded MP4; keep SDK metadata and user files."""
    path = Path(path)
    root = (Path(scratch_dir) / 'downloads').resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(root) or path.suffix != '.mp4':
        raise ValueError('Refusing to remove a source outside managed download storage')
    entries = {item.name: item for item in path.parent.iterdir()}
    if path.name in entries and entries[path.name].is_file():
        path.unlink()
