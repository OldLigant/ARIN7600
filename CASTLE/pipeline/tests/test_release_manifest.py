"""Guards for release manifests (docs/release-process.md R-02/R-03/R-06).

Three separate risks are covered here, deliberately kept apart:

* the dependency-free ``release.py`` fingerprint silently drifting from the
  runtime ``fingerprint()`` that actually pins runs;
* a manifest that no longer describes the commit its release tag points at;
* publishing a changed tree under an existing release name.
"""

import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

import pytest

import batch_pipeline
import release
from castle_pipeline.runner import fingerprint

REPO = Path(__file__).resolve().parents[1]


def release_tags():
    """Every manifest that also exists as a tag; the test audits all of them."""
    listed = subprocess.run(['git', '-C', str(REPO), 'tag', '--list'],
                            capture_output=True, text=True, check=False)
    if listed.returncode != 0:
        return []  # not inside any git repository
    tags = set(listed.stdout.split())
    return sorted(str(p.stem) for p in (REPO/'releases').glob('*.json') if p.stem in tags)


def test_inline_fingerprint_matches_runtime_fingerprint():
    """release.py must reproduce batch_pipeline.code_hash() byte for byte."""
    assert release.code_hash(REPO, 'basename-v1') == batch_pipeline.code_hash()
    assert release.fingerprint({'b': 1, 'a': 2}) == fingerprint({'a': 2, 'b': 1})


def test_manifest_code_hash_is_derivable_from_its_own_identity_files():
    """A manifest is self-describing: code_hash must follow from identity_files.

    This catches hand-editing and a wrong hash_scheme without depending on the
    working tree, so normal development against an older release stays green.
    """
    manifests = sorted((REPO/'releases').glob('*.json'))
    assert manifests, 'no release manifests recorded'
    for path in manifests:
        manifest = json.loads(path.read_text(encoding='utf-8'))
        assert manifest['hash_scheme'] == 'basename-v1'
        derived = release.fingerprint(
            {Path(name).name: digest for name, digest in manifest['identity_files'].items()})
        assert derived == manifest['code_hash'], f'{path.name} code_hash does not follow from identity_files'
        assert manifest.get('published_utc') or manifest['provenance'].get('git_commit')


def test_manifest_uses_repository_relative_paths():
    """R-03: manifest keys are repo-relative, never bare basenames.

    Bare basenames collide as soon as a batch subpackage is introduced
    (castle_pipeline/__init__.py vs castle_pipeline/batch/__init__.py), and
    fingerprint() would silently drop one of them.
    """
    for path in sorted((REPO/'releases').glob('*.json')):
        manifest = json.loads(path.read_text(encoding='utf-8'))
        names = list(manifest['identity_files'])
        assert 'batch_pipeline.py' in names, path.name
        assert all('/' in name or name.endswith('.py') for name in names), path.name
        basenames = [Path(name).name for name in names]
        assert len(set(basenames)) == len(basenames), f'{path.name}: identity files share a basename'


@pytest.mark.parametrize('tag', release_tags() or [None])
def test_tagged_release_matches_its_manifest(tag):
    """Every tag and its manifest must describe the same bytes."""
    if tag is None:
        pytest.skip('not a git repository, or no manifest has a tag')
    root = release.git_root(REPO)
    assert root is not None, 'the pipeline directory must live inside a git repository'
    manifest = json.loads((REPO/'releases'/f'{tag}.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='release-tag-') as work:
        work_path = Path(work)
        archive = work_path/'tag.tar'
        made = subprocess.run(['git', '-C', str(root), 'archive', '--format=tar',
                               '-o', str(archive), tag],
                              capture_output=True, text=True, check=False)
        if made.returncode != 0:
            pytest.skip(f'git archive failed: {made.stderr.strip()[:200]}')
        tree = work_path/'tree'
        tree.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(tree, filter='data')  # archive produced locally by git
        # git archive emits root-relative paths; rebase onto the pipeline subtree.
        tagged = release.tagged_subtree(tree, REPO, root)
        assert tagged is not None, f'tag {tag} predates the monorepo layout or lacks the pipeline subtree'
        assert release.differences(manifest, tagged) == []


def test_publishing_a_changed_tree_under_an_existing_release_is_refused(tmp_path):
    """R-06: a changed tree is a new release, not an edit."""
    tree = tmp_path/'tree'
    (tree/'castle_pipeline').mkdir(parents=True)
    (tree/'batch_pipeline.py').write_text('one\n', encoding='utf-8')
    (tree/'castle_pipeline'/'__init__.py').write_text('two\n', encoding='utf-8')
    out = tmp_path/'releases'
    argv = ['manifest', '--release', 'demo', '--tree', str(tree), '--releases-dir', str(out)]
    assert release.main(argv) == 0
    recorded = json.loads((out/'demo.json').read_text(encoding='utf-8'))

    (tree/'batch_pipeline.py').write_text('changed\n', encoding='utf-8')
    assert release.main(argv) == 1
    assert json.loads((out/'demo.json').read_text(encoding='utf-8')) == recorded

    verify = ['verify', '--release', 'demo', '--tree', str(tree), '--releases-dir', str(out)]
    assert release.main(verify) == 1


def test_verify_accepts_the_untouched_tree(tmp_path):
    tree = tmp_path/'tree'
    (tree/'castle_pipeline').mkdir(parents=True)
    (tree/'batch_pipeline.py').write_text('one\n', encoding='utf-8')
    (tree/'castle_pipeline'/'__init__.py').write_text('two\n', encoding='utf-8')
    out = tmp_path/'releases'
    assert release.main(['manifest', '--release', 'demo', '--tree', str(tree),
                         '--releases-dir', str(out)]) == 0
    assert release.main(['verify', '--release', 'demo', '--tree', str(tree),
                         '--releases-dir', str(out)]) == 0


def _init_tagged_repo(tree, tag):
    """Create a real tag, with the user's global git config neutralised."""
    import os
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    def git(*args):
        return subprocess.run(['git', '-C', str(tree), *args], capture_output=True,
                              text=True, check=False, env=env)
    assert git('init', '-q') .returncode == 0
    git('config', 'user.name', 'release-test')
    git('config', 'user.email', 'release-test@local')
    git('config', 'commit.gpgsign', 'false')
    assert git('add', '-A').returncode == 0
    assert git('commit', '-q', '-m', 'fixture').returncode == 0
    assert git('tag', '-a', tag, '-m', 'fixture tag').returncode == 0


def test_publish_verify_only_refuses_a_tree_that_drifted_from_the_manifest(tmp_path):
    """R-06: the publish gate must catch worktree drift before any upload."""
    tree = tmp_path/'tree'
    (tree/'castle_pipeline').mkdir(parents=True)
    (tree/'batch_pipeline.py').write_text('one\n', encoding='utf-8')
    (tree/'castle_pipeline'/'__init__.py').write_text('two\n', encoding='utf-8')
    (tree/'batch_jobs.py').write_text('launcher\n', encoding='utf-8')
    out = tmp_path/'releases'
    assert release.main(['manifest', '--release', 'demo', '--tree', str(tree),
                         '--releases-dir', str(out)]) == 0
    _init_tagged_repo(tree, 'demo')

    args = ['publish', '--release', 'demo', '--tree', str(tree), '--releases-dir', str(out),
            '--verify-only']
    assert release.main(args) == 0, 'the tagged, manifest-consistent tree must pass'

    # An auxiliary file is not part of code_hash, so only the manifest notices.
    (tree/'batch_jobs.py').write_text('launcher changed\n', encoding='utf-8')
    assert release.main(args) == 1
    # Restoring the byte restores the gate, proving it compares content not mtime.
    (tree/'batch_jobs.py').write_text('launcher\n', encoding='utf-8')
    assert release.main(args) == 0


def test_manifest_declares_exactly_the_files_a_publish_ships():
    """R-06: publish stages the declared set only.

    The first publish shipped the whole repository tree into an immutable prefix
    (85 objects for a 22-file manifest). Publishing must be an allow-list.
    """
    for path in sorted((REPO/'releases').glob('*.json')):
        manifest = json.loads(path.read_text(encoding='utf-8'))
        declared = set(manifest['identity_files']) | set(manifest['auxiliary_files']) \
            | set(manifest['prompt_files'])
        for name in declared:
            assert (REPO/name).is_file(), f'{path.name} declares a file that is not in the tree: {name}'
        # Nothing that only exists to support development may be declared.
        for name in declared:
            assert not name.startswith(('_test/', '_bench/', 'tools/', 'tests/', 'ledger/',
                                        'credentials/', 'releases/')), \
                f'{path.name} declares a non-shipping path: {name}'
