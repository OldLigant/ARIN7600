#!/usr/bin/env python3
"""Create the castle-batch-v4 manifest for an auxiliary-only change (R-05/R-06).

v4 carries the same identity files as v3, so its code_hash is identical: the
launcher change cannot alter any annotation. What differs is recorded under
auxiliary_files, which is exactly why manifests track them separately.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import release  # noqa: E402

TARGET = REPO / 'releases' / 'castle-batch-v4.json'
SOURCE = REPO / 'releases' / 'castle-batch-v3.json'

manifest = json.loads(SOURCE.read_text(encoding='utf-8'))
manifest['release'] = 'castle-batch-v4'
manifest['created_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
manifest['published_utc'] = None
manifest['supersedes'] = 'castle-batch-v3'
manifest['notes'] = ('Launcher-only release: batch_jobs.py gains --code-volume auto, which reads the '
                     "run's pinned code_hash and mounts the matching release, and refuses a "
                     'contradicting --code-volume before submitting (R-05). Identity files are '
                     'byte-identical to castle-batch-v3, so code_hash is unchanged and every run '
                     'pinned to v3 remains tickable with either release.')
manifest['provenance'] = {'kind': 'git', 'detail': str(REPO), 'git_commit': release.git_commit(REPO)}
manifest['auxiliary_files'] = release.collect(REPO, release.AUXILIARY)
manifest['prompt_files'] = release.collect(REPO, release.PROMPT_GLOB)
manifest['identity_files'] = {p.relative_to(REPO).as_posix(): release.sha256_file(p)
                              for p in release.identity_paths(REPO)}
manifest['code_hash'] = release.code_hash(REPO, manifest['hash_scheme'])

assert manifest['code_hash'] == json.loads(SOURCE.read_text(encoding='utf-8'))['code_hash'], \
    'identity hash must not change in an auxiliary-only release'

TARGET.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
                  encoding='utf-8', newline='\n')
print(json.dumps({'ok': True, 'release': manifest['release'],
                  'code_hash': manifest['code_hash'],
                  'identity_files_unchanged': True,
                  'auxiliary_files': len(manifest['auxiliary_files']),
                  'path': str(TARGET)}, indent=2))
