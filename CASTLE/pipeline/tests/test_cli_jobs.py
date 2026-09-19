import json
import pytest
from castle_pipeline.inputs import source_metadata, select_sources
from jobs import build_command


def test_manifest_selection_excludes_novideo_and_has_disjoint_shards(tmp_path):
    entries = [{'video': 'main/day1/Allie/video/08.mp4', 'video_bytes': 100},
               {'video': None, 'novideo': True},
               {'video': 'main/day1/Bjorn/video/08.mp4', 'video_bytes': 200},
               {'video': 'main/day1/Kitchen/video/08.mp4', 'video_bytes': 300}]
    path = tmp_path/'hours.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in entries), encoding='utf-8')
    zero = select_sources(path, viewpoint='ego', shard_index=0, shard_count=2)
    one = select_sources(path, viewpoint='ego', shard_index=1, shard_count=2)
    assert [r['path'] for r in zero] == ['main/day1/Allie/video/08.mp4']
    assert [r['path'] for r in one] == ['main/day1/Bjorn/video/08.mp4']
    with pytest.raises(ValueError):
        source_metadata('main/../../stolen.mp4', 'rev')


def test_manifest_preserves_pinned_source_revision(tmp_path):
    path = tmp_path/'manifest.jsonl'
    path.write_text(json.dumps({'path': 'main/day1/Allie/video/08.mp4', 'revision': 'a'*40, 'bytes': 100})+'\n')
    assert select_sources(path)[0]['revision'] == 'a'*40


def test_hour_filter_cannot_expand_a_single_hour_request(tmp_path):
    path = tmp_path/'manifest.jsonl'
    path.write_text('\n'.join(json.dumps({'path': f'main/day1/Allie/video/{hour}.mp4'}) for hour in ['08', '09']))
    assert [r['path'] for r in select_sources(path, hour=9)] == ['main/day1/Allie/video/09.mp4']


def test_jobs_forwards_project_resolved_from_environment(monkeypatch, capsys):
    from jobs import main
    monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'offline-project')
    assert main(['--code-volume', 'hf://buckets/u/code', '--output-volume', 'hf://buckets/u/out', '--name', 'plan',
                 '--', '--source', 'main/day1/Allie/video/08.mp4', '--model', 'offline-model']) == 0
    assert 'GOOGLE_CLOUD_PROJECT=offline-project' in json.loads(capsys.readouterr().out)['argv']


def test_jobs_command_has_persistent_output_and_forwards_secret_names_only():
    command = build_command(code_volume='hf://buckets/u/code', output_volume='hf://buckets/u/results',
                            name='castle-smoke', pipeline_args=['--source', 'main/day1/Allie/video/08.mp4',
                            '--model', 'configured-model', '--max-clips', '2'], secrets=['GOOGLE_API_KEY'])
    assert command[:3] == ['hf', 'jobs', 'run']
    assert 'GOOGLE_API_KEY' in command and not any('GOOGLE_API_KEY=' in v for v in command)
    assert 'hf://buckets/u/results:/output' in command
    shell = command[-1]
    assert '--output-dir /output' in shell and '--scratch-dir /scratch/castle' in shell
    assert '--max-clips 2' in shell
    with pytest.raises(ValueError):
        build_command(code_volume='local-path', output_volume='hf://buckets/u/out', name='x', pipeline_args=[], secrets=[])


def test_jobs_rejects_ephemeral_output_override():
    with pytest.raises(ValueError):
        build_command(code_volume='hf://buckets/u/code', output_volume='hf://buckets/u/out', name='x',
                      pipeline_args=['--output-dir', '/tmp/output'], secrets=[])


def test_jobs_exports_media_tuning_as_environment_defaults():
    """The rendered argv must carry the tuning knobs the container cannot infer."""
    command = build_command(code_volume='hf://buckets/u/code', output_volume='hf://buckets/u/out',
                            name='castle-fast', pipeline_args=['--source', 'p', '--model', 'm'],
                            secrets=[], media_threads=4, footer_workers=4)
    assert 'CASTLE_MEDIA_THREADS=4' in command
    assert 'CASTLE_FOOTER_WORKERS=4' in command
    without = build_command(code_volume='hf://buckets/u/code', output_volume='hf://buckets/u/out',
                            name='castle-default', pipeline_args=['--source', 'p', '--model', 'm'], secrets=[])
    assert not [v for v in without if v.startswith('CASTLE_MEDIA_THREADS=')]
    for bad in (0, -2, True, 'x'):
        with pytest.raises(ValueError):
            build_command(code_volume='hf://buckets/u/code', output_volume='hf://buckets/u/out', name='x',
                          pipeline_args=[], secrets=[], media_threads=bad)
