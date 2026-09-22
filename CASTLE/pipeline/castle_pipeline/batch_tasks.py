"""Vertex JSONL requests and row-wise, bounded-memory result collection."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from PIL import Image

from .runner import atomic_json, cleanup_media
from .media import positive_int
from .request_spec import (DEFAULT_MAX_OUTPUT_TOKENS, AUDIO_SOURCE_LABEL, annotation_prompt,
                           annotation_stage_context, audio_stage_context, base_context,
                           crop_label, frame_label, review_prompt, review_stage_context)
from .schema import validate_audio, normalize_annotation, apply_review

MARKER = 'CASTLE_BATCH_ID:'


def request_row(request_id, prompt, context, media, max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    parts = [{'text': MARKER + request_id + '\n' + json.dumps(context, ensure_ascii=False)}]
    for label, uri, mime in media:
        if not uri.startswith('gs://'):
            raise ValueError('Batch media must use persistent GCS URIs')
        parts.append({'fileData': {'fileUri': uri, 'mimeType': mime}})
        parts.append({'text': label})
    return {'request': {'systemInstruction': {'parts': [{'text': prompt}]},
                        'contents': [{'role': 'user', 'parts': parts}],
                        'generationConfig': {'responseMimeType': 'application/json',
                                             'maxOutputTokens': positive_int(max_output_tokens, 'max_output_tokens')}}}


def response_id(row):
    for content in row.get('request', {}).get('contents', []):
        for part in content.get('parts', []):
            value = part.get('text')
            if isinstance(value, str) and value.startswith(MARKER):
                return value.split('\n', 1)[0][len(MARKER):]
    return None


def parse_response(row):
    status = row.get('status')
    if status not in (None, '', {}) and not (isinstance(status, dict) and status.get('code') == 0):
        raise ValueError('ROW_ERROR')
    response = row.get('response') or {}
    candidates = response.get('candidates') or []
    if not candidates or candidates[0].get('finishReason') != 'STOP':
        raise ValueError('INVALID_OUTPUT')
    text = ''.join(p['text'] for p in candidates[0].get('content', {}).get('parts', [])
                   if isinstance(p.get('text'), str) and not p.get('thought'))
    try:
        data = json.loads(text)
    except ValueError:
        raise ValueError('INVALID_OUTPUT') from None
    if not isinstance(data, dict):
        raise ValueError('INVALID_OUTPUT')
    usage = response.get('usageMetadata', {})
    return {'data': data, 'usage': {name: usage.get(key) for name, key in [
        ('input_tokens', 'promptTokenCount'), ('output_tokens', 'candidatesTokenCount'),
        ('thought_tokens', 'thoughtsTokenCount'), ('cached_tokens', 'cachedContentTokenCount'),
        ('total_tokens', 'totalTokenCount')]}}


class TaskPlanner:
    def __init__(self, cloud, scratch_dir, output_dir, log=None):
        self.cloud, self.scratch, self.output = cloud, Path(scratch_dir), Path(output_dir)
        self.log = log
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)

    def emit(self, event, **fields):
        if self.log:
            self.log.emit(event, **fields)

    def get_json(self, uri):
        work = Path(tempfile.mkdtemp(prefix='batch-json-', dir=self.scratch))
        path = work/'data.json'
        try:
            self.cloud.download(uri, path)
            return json.loads(path.read_text(encoding='utf-8'))
        finally:
            cleanup_media(work, [path])

    def put_json(self, uri, data):
        work = Path(tempfile.mkdtemp(prefix='batch-json-', dir=self.scratch))
        path = work/'data.json'
        try:
            atomic_json(path, data)
            self.cloud.upload(path, uri)
        finally:
            cleanup_media(work, [path])
        return uri

    def prepare(self, stage, eligible_rows, state):
        cfg = state['config'];prefix = cfg['gcs_prefix'].rstrip('/')
        if len({item['clip_id'] for item in eligible_rows}) != len(eligible_rows):
            raise ValueError('Duplicate clip IDs would duplicate paid requests')
        ids = []
        # URI binds inputs to precisely the eligible row set, including partial success.
        digest = hashlib.sha256(json.dumps(eligible_rows, sort_keys=True).encode()).hexdigest()[:16]
        uri = f'{prefix}/requests/{stage}-{digest}.jsonl'
        if not eligible_rows or (stage == 'audio' and all(item.get('audio_uri') for item in eligible_rows)):
            return {'input_uri':uri,'output_uri':f'{prefix}/batch-output/{stage}-{digest}/','request_ids':[]}
        def requests():
            for item in eligible_rows:
                if stage == 'audio' and item.get('audio_uri'):
                    continue
                meta = self.get_json(item['metadata_uri'])
                request_id = stage + ':' + item['clip_id']
                context = base_context(stage, item['clip_id'], meta['duration_sec'],
                                       meta['source'], meta['start_offset_sec'])
                if stage == 'audio':
                    prompt = cfg['prompts']['audio']
                    audio_stage_context(context)
                    media = [(AUDIO_SOURCE_LABEL, meta['audio_uri'], 'audio/wav')]
                elif stage == 'annotation':
                    audio = self.get_json(item['audio_uri'])
                    prompt = annotation_prompt(cfg['prompts'], review_enabled=cfg.get('review', True),
                                               max_review_regions=cfg.get('max_review_regions', 4),
                                               exocentric=meta['source'].get('viewpoint') == 'exocentric')
                    annotation_stage_context(context, meta['frame_times_sec'], audio['data'],
                                             bool(meta['audio_uri']))
                    media = [(frame_label(i, t), uri, 'image/jpeg')
                             for i, (t, uri) in enumerate(zip(meta['frame_times_sec'], meta['frames']))]
                elif stage == 'review':
                    first = self.get_json(item['annotation_uri'])
                    review_stage_context(context, first['data'],
                                         [{'source_id': c['source_id'], 'time_sec': c['time_sec']} for c in item['crops']])
                    prompt = review_prompt(cfg['prompts'])
                    media = [(crop_label(c['source_id'], c['time_sec']), c['uri'], 'image/jpeg') for c in item['crops']]
                else:
                    raise ValueError('Unknown batch stage')
                ids.append(request_id)
                yield request_row(request_id, prompt, context, media,
                                  cfg.get('max_output_tokens', DEFAULT_MAX_OUTPUT_TOKENS))
        # write_jsonl consumes the iterator synchronously; it uploads nothing for zero rows.
        self.cloud.write_jsonl(uri, requests())
        return {'input_uri': uri, 'output_uri': f'{prefix}/batch-output/{stage}-{digest}/', 'request_ids': ids}

    def _crops(self, meta, regions, config):
        if not isinstance(regions, list) or len(regions) > config.get('max_review_regions', 4):
            raise ValueError('Invalid crop count')
        crops = []
        for i, region in enumerate(regions):
            if not isinstance(region, dict):
                raise ValueError('Crop region must be an object')
            index, box = region.get('frame_index'), region.get('box_2d')
            if type(index) is not int or not 0 <= index < len(meta['native_frames']):
                raise ValueError('Invalid crop frame index')
            if not isinstance(box, list) or len(box) != 4 or not all(type(v) in (int, float) and math.isfinite(v) for v in box):
                raise ValueError('Invalid crop coordinates')
            top, left, bottom, right = box
            if not (0 <= top < bottom <= 1000 and 0 <= left < right <= 1000):
                raise ValueError('Invalid crop coordinates')
            work = Path(tempfile.mkdtemp(prefix='batch-crop-', dir=self.scratch))
            src, dst = work/'native.jpg', work/'crop.jpg'
            try:
                self.cloud.download(meta['native_frames'][index], src)
                with Image.open(src) as image:
                    with image.crop((math.floor(left*image.width/1000), math.floor(top*image.height/1000),
                                     math.ceil(right*image.width/1000), math.ceil(bottom*image.height/1000))) as crop:
                        crop.thumbnail((config.get('max_dim', 1440),)*2)
                        crop.save(dst, 'JPEG', quality=95)
                uri = f'{config["gcs_prefix"].rstrip("/")}/media/{meta["clip_id"]}/crop-{i}.jpg'
                self.cloud.upload(dst, uri)
                crops.append({'source_id': f'crop_{i}', 'time_sec': meta['frame_times_sec'][index], 'uri': uri})
            finally:
                cleanup_media(work, [src, dst])
        return crops

    def _final(self, item, meta, annotation, first, review, config):
        audio = self.get_json(item['audio_uri'])
        changes = first.get('normalization', [])
        requests = review['data']['resegmentation_requests'] if review else []
        result = {'ok': True, 'execution_mode': 'vertex_batch', 'model': config['model'], 'source': meta['source'],
                  'clip': {'id': meta['clip_id'], 'start_offset_sec': meta['start_offset_sec'], 'duration_sec': meta['duration_sec']},
                  'annotation': annotation, 'normalization': changes,
                  'review_required': bool(requests) or any(c['code'] == 'NONEDGE_ONGOING_TO_UNCERTAIN' for c in changes),
                  'resegmentation_requests': requests,
                  'usage': {'audio': audio.get('usage', {}), 'annotation': first.get('usage', {}),
                            'review': review.get('usage', {}) if review else {}},
                  'batch_run_prefix': config['gcs_prefix']}
        uri = f'{config["gcs_prefix"].rstrip("/")}/final/{meta["clip_id"]}.json'
        self.put_json(uri, result)
        atomic_json(self.output/'final'/f'{meta["clip_id"]}.json', result)
        self.emit('clip_completed',clip_id=meta['clip_id'],final_uri=uri,review_required=result['review_required'])
        return {'clip_id': meta['clip_id'], 'final_uri': uri, 'review_required': result['review_required']}

    def collect(self, stage, job, state):
        cfg = state['config'];prefix = cfg['gcs_prefix'].rstrip('/')
        expected = set(state['batches'][stage]['request_ids'])
        work = Path(tempfile.mkdtemp(prefix='batch-results-', dir=self.scratch))
        paths, duplicates, owned, unknown = {}, set(), [], 0
        result = {'eligible_rows': [], 'failures': [], 'completed': []}
        try:
            for row in self.cloud.iter_jsonl(job['output_uri']):
                rid = response_id(row)
                if rid not in expected:
                    unknown += 1
                    continue
                if rid in paths:
                    duplicates.add(rid)
                    continue
                path = work/(hashlib.sha256(rid.encode()).hexdigest()+'.json')
                owned.append(path);atomic_json(path,row);paths[rid]=path
            if unknown:
                result['failures'].append({'stage': stage, 'code': 'UNMATCHED_RESPONSE', 'count': unknown})
            for item in state['eligible_rows']:
                rid = stage+':'+item['clip_id']
                if rid not in expected:
                    if stage == 'audio' and item.get('audio_uri'):
                        result['eligible_rows'].append(item)
                    else:
                        result['failures'].append({'clip_id':item['clip_id'],'stage':stage,'code':'UNSCHEDULED_ROW'})
                    continue
                code = 'DUPLICATE_RESPONSE' if rid in duplicates else 'MISSING_RESPONSE' if rid not in paths else None
                if code:
                    result['failures'].append({'clip_id': item['clip_id'], 'stage': stage, 'code': code})
                    self.emit('batch_row_failed',request_id=rid,phase=stage,code=code)
                    continue
                raw = json.loads(paths[rid].read_text(encoding='utf-8'))
                raw_uri = f'{prefix}/raw/{stage}/{item["clip_id"]}.json'
                self.put_json(raw_uri, raw)
                try:
                    response = parse_response(raw)
                    self.emit('batch_response_received',request_id=rid,phase=stage,usage=response['usage'])
                    meta = self.get_json(item['metadata_uri'])
                    sources = {'video_0'} | ({'audio_0'} if meta['audio_uri'] else set())
                    if stage == 'audio':
                        validate_audio(response['data'], meta['duration_sec'])
                        uri = self.put_json(f'{prefix}/results/audio/{item["clip_id"]}.json',response)
                        result['eligible_rows'].append({**item,'audio_uri':uri})
                    elif stage == 'annotation':
                        data = response['data'];regions = data.get('review_regions',[]) if cfg.get('review',True) else []
                        normalized, changes = normalize_annotation({k:v for k,v in data.items() if k!='review_regions'},meta['duration_sec'],sources)
                        first = {**response,'raw_data':data,'data':normalized,'normalization':changes}
                        uri = self.put_json(f'{prefix}/results/annotation/{item["clip_id"]}.json',first)
                        crops = self._crops(meta,regions,cfg)
                        if crops:
                            result['eligible_rows'].append({**item,'annotation_uri':uri,'crops':crops})
                        else:
                            result['completed'].append(self._final(item,meta,normalized,first,None,cfg))
                    elif stage == 'review':
                        first = self.get_json(item['annotation_uri'])
                        sources |= {c['source_id'] for c in item['crops']}
                        revised = apply_review(first['data'],response['data'],meta['duration_sec'],sources)
                        self.put_json(f'{prefix}/results/review/{item["clip_id"]}.json',response)
                        result['completed'].append(self._final(item,meta,revised,first,response,cfg))
                    else:
                        raise ValueError('Unknown stage')
                    self.emit('batch_row_success',request_id=rid,phase=stage,usage=response['usage'])
                except (ValueError, KeyError, TypeError) as error:
                    code = str(error) if str(error) in {'ROW_ERROR','INVALID_OUTPUT'} else 'VALIDATION'
                    failure = {'clip_id':item['clip_id'],'stage':stage,'code':code,'raw_uri':raw_uri}
                    status = raw.get('status')
                    if isinstance(status,dict) and isinstance(status.get('code'),(str,int)):
                        failure['provider_code']=status['code']
                    self.put_json(f'{prefix}/errors/{stage}/{item["clip_id"]}.json',failure)
                    result['failures'].append(failure)
                    self.emit('batch_row_failed',request_id=rid,phase=stage,**failure)
            return result
        finally:
            cleanup_media(work,owned)
