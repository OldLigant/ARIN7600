import json
from pathlib import Path
import pytest
from castle_pipeline.batch_tasks import (request_row, response_id, parse_response, TaskPlanner,
                                        DEFAULT_MAX_OUTPUT_TOKENS)


def test_vertex_request_uses_gcs_parts_and_echoed_marker_not_line_order():
    row = request_row('audio:c1', 'prompt', {'duration_sec': 30}, [('audio_0', 'gs://bucket/a.wav', 'audio/wav')])
    assert set(row) == {'request'}
    assert row['request']['contents'][0]['parts'][1]['fileData']['fileUri'] == 'gs://bucket/a.wav'
    assert response_id({'request': row['request']}) == 'audio:c1'
    assert row['request']['systemInstruction']['parts'][0]['text'] == 'prompt'


def test_output_token_budget_is_configurable_because_thinking_shares_it():
    """A tight cap truncates annotation JSON; the limit must be explicit and validated."""
    default = request_row('a:c1', 'p', {}, [])
    assert default['request']['generationConfig']['maxOutputTokens'] == DEFAULT_MAX_OUTPUT_TOKENS
    assert DEFAULT_MAX_OUTPUT_TOKENS > 16384
    raised = request_row('a:c1', 'p', {}, [], max_output_tokens=65536)
    assert raised['request']['generationConfig']['maxOutputTokens'] == 65536
    for bad in (0, -1, True, 1.5, 'x'):
        with pytest.raises(ValueError):
            request_row('a:c1', 'p', {}, [], max_output_tokens=bad)


def test_parse_response_handles_per_row_errors_and_excludes_thoughts():
    row = {'status': '', 'response': {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
        {'text': 'not model output', 'thought': True}, {'text': '{"summary":"ok"}'}]}}],
        'usageMetadata': {'promptTokenCount': 12, 'candidatesTokenCount': 4}}}
    result = parse_response(row)
    assert result['data'] == {'summary': 'ok'} and result['usage']['input_tokens'] == 12
    with pytest.raises(ValueError):
        parse_response({'status': 'Bad Request', 'response': {}})


class CloudFiles:
    def __init__(self):
        self.files = {}
        self.output = []
    def upload(self, path, uri):
        self.files[uri] = Path(path).read_bytes()
    def download(self, uri, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(self.files[uri])
    def write_jsonl(self, uri, rows):
        self.files[uri] = ''.join(json.dumps(r)+'\n' for r in rows).encode()
    def iter_jsonl(self, prefix):
        yield from self.output


def test_collector_matches_reordered_rows_and_never_marks_missing_complete(tmp_path):
    cloud=CloudFiles();planner=TaskPlanner(cloud,tmp_path/'scratch',tmp_path/'output')
    items=[]
    for i in ['c1','c2','c3']:
        uri=f'gs://b/run/{i}/meta.json'
        cloud.files[uri]=json.dumps({'clip_id':i,'duration_sec':30,'source':{},'frame_times_sec':[],
                                    'frames':[],'audio_uri':f'gs://b/{i}.wav','start_offset_sec':0}).encode()
        items.append({'clip_id':i,'metadata_uri':uri})
    state={'config':{'gcs_prefix':'gs://b/run','prompts':{'audio':'audio'},'review':False},'eligible_rows':items,
           'batches':{'audio':{'request_ids':['audio:c1','audio:c2','audio:c3']}}}
    data={'summary':'quiet','utterances':[],'sound_events':[],'uncertainties':[]}
    def output(cid):
        return {'request':request_row('audio:'+cid,'',{},[])['request'],'status':'',
                'response':{'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(data)}]}}]}}
    cloud.output=[output('c2'),output('c1')]
    result=planner.collect('audio',{'output_uri':'gs://b/results'},state)
    assert [r['clip_id'] for r in result['eligible_rows']]==['c1','c2']
    assert result['failures'][0]['clip_id']=='c3'
    assert result['failures'][0]['code']=='MISSING_RESPONSE'
    assert result['completed']==[]


def test_nonobject_crop_is_a_validation_error(tmp_path):
    planner=TaskPlanner(CloudFiles(),tmp_path/'scratch',tmp_path/'output')
    with pytest.raises(ValueError,match='object'):
        planner._crops({'native_frames':[]},[None],{})


def test_duplicate_clip_ids_never_reach_jsonl_upload(tmp_path):
    cloud=CloudFiles();planner=TaskPlanner(cloud,tmp_path/'scratch',tmp_path/'out')
    with pytest.raises(ValueError,match='Duplicate'):
        planner.prepare('audio',[{'clip_id':'same'},{'clip_id':'same'}],{'config':{'gcs_prefix':'gs://b/run'}})
    assert cloud.files=={}


def test_planner_writes_the_configured_token_budget_into_every_request(tmp_path):
    """The stored run parameter, not a code constant, decides the request budget."""
    cloud=CloudFiles();planner=TaskPlanner(cloud,tmp_path/'scratch',tmp_path/'out')
    uri='gs://b/run/c1/meta.json'
    cloud.files[uri]=json.dumps({'clip_id':'c1','duration_sec':30,'source':{},'frame_times_sec':[],
                                 'frames':[],'audio_uri':'gs://b/c1.wav','start_offset_sec':0}).encode()
    cfg={'gcs_prefix':'gs://b/run','prompts':{'audio':'audio prompt'},'review':False,
         'max_output_tokens':49152}
    prepared=planner.prepare('audio',[{'clip_id':'c1','metadata_uri':uri}],{'config':cfg})
    lines=[json.loads(line) for line in cloud.files[prepared['input_uri']].decode().splitlines() if line.strip()]
    assert len(lines)==1
    assert lines[0]['request']['generationConfig']['maxOutputTokens']==49152
    # A config that predates the parameter keeps a workable budget rather than 16384.
    cfg.pop('max_output_tokens')
    prepared=planner.prepare('audio',[{'clip_id':'c1','metadata_uri':uri}],{'config':cfg})
    lines=[json.loads(line) for line in cloud.files[prepared['input_uri']].decode().splitlines() if line.strip()]
    assert lines[0]['request']['generationConfig']['maxOutputTokens']==DEFAULT_MAX_OUTPUT_TOKENS


def test_cloud_error_keeps_safe_http_code_without_provider_details():
    from castle_pipeline.batch_cloud import _failure
    error=RuntimeError('private provider payload');error.code=403
    safe=_failure('lookup',error)
    assert safe.code==403 and '403' in str(safe)
    assert 'private' not in str(safe)
