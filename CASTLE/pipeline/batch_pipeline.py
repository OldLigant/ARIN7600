#!/usr/bin/env python3
"""Short-lived Vertex Batch workers: prepare/submit, or check/collect/advance once."""
import argparse
import hashlib
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
from PIL import Image

from castle_pipeline.batch_cloud import BatchCloud, parse_gs_uri
from castle_pipeline.batch_engine import BatchEngine
from castle_pipeline.batch_tasks import TaskPlanner, DEFAULT_MAX_OUTPUT_TOKENS
from castle_pipeline.events import EventLog
from castle_pipeline.inputs import REPO, REVISION, source_metadata, download_source, remove_downloaded_source
from castle_pipeline.media import (MediaExtractor, probe, env_decoder_default, env_footer_default,
                                   decoder_memory_warning, positive_int)
from castle_pipeline.runner import atomic_json, cleanup_media, clip_windows, fingerprint


def default_prepare_workers() -> int:
    """Clips prepared at once. Cgroup-blind, so deliberately modest."""
    raw = os.environ.get('CASTLE_PREPARE_WORKERS')
    if raw is not None and raw.strip():
        try:
            value = int(raw.strip())
        except ValueError:
            raise ValueError('Environment variable CASTLE_PREPARE_WORKERS must be a positive integer') from None
        if value < 1:
            raise ValueError('Environment variable CASTLE_PREPARE_WORKERS must be a positive integer')
        return value
    return min(3, max(1, (os.cpu_count() or 1) // 2))


def code_hash():
    root=Path(__file__).parent
    paths=[root/'batch_pipeline.py',*(root/'castle_pipeline').glob('*.py')]
    return fingerprint({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def _prepare_clip(source, metadata, config, planner, extractor, index, start, duration):
    """Persist model-sized stamped frames and native samples for one clip.

    Runs concurrently with sibling clips, so it touches only its own scratch
    directory and content-addressed GCS objects. ``native_uri`` owns the native
    JPEGs as soon as they are uploaded, which is why the paths are stored on the
    PreparedClip instead of being re-derived from frame_paths afterwards.
    """
    native_dim = max(config['source_width'], config['source_height'])
    cid = hashlib.sha256((metadata['source_id'] + ':' + str(index)).encode()).hexdigest()[:24]
    prefix = config['gcs_prefix'].rstrip('/') + '/media/' + cid
    work = Path(tempfile.mkdtemp(prefix='batch-prepare-', dir=planner.scratch))
    owned = []
    try:
        media = extractor.prepare(source, start, duration, work, fps=config['fps'],
                                  max_dim=native_dim, stamp=False)
        owned.extend(media.frame_paths)
        if media.audio_path:
            owned.append(media.audio_path)
        if len(media.frame_paths) != len(media.frame_times):
            raise ValueError('Prepared frame count disagrees with the sampled grid')
        native, model_paths, model_uris = [], [], []
        for i, (path, t) in enumerate(zip(media.frame_paths, media.frame_times)):
            if config.get('review', True):
                uri = f'{prefix}/native-{i:03d}.jpg'
                planner.cloud.upload(path, uri)
                native.append(uri)
            stamped = work / f'model-{i:03d}.jpg'
            owned.append(stamped)
            with Image.open(path) as image:
                image.thumbnail((config['max_dim'], config['max_dim'] - 24))
                image.save(stamped, 'JPEG', quality=95)
            model_paths.append(stamped)
            model_uris.append(f'{prefix}/frame-{i:03d}.jpg')
        # One pinned stamping order for the whole batch, independent of the pool size.
        extractor.stamp_frames(model_paths, media.frame_times, 24)
        for stamped, uri in zip(model_paths, model_uris):
            planner.cloud.upload(stamped, uri)
        audio_uri = None
        if media.audio_path:
            audio_uri = f'{prefix}/audio.wav'
            planner.cloud.upload(media.audio_path, audio_uri)
        meta = {'clip_id': cid, 'source': metadata, 'clip_index': index, 'start_offset_sec': start,
                'duration_sec': duration, 'frame_times_sec': media.frame_times, 'frames': model_uris,
                'native_frames': native, 'audio_uri': audio_uri}
        uri = planner.put_json(f'{prefix}/metadata.json', meta)
        row = {'clip_id': cid, 'metadata_uri': uri}
        if audio_uri is None:
            empty = {'data': {'summary': 'No audio track supplied.', 'utterances': [], 'sound_events': [],
                              'uncertainties': ['No audio evidence available.']}, 'usage': {}, 'skipped': True}
            row['audio_uri'] = planner.put_json(f'{config["gcs_prefix"].rstrip("/")}/results/audio/{cid}.json', empty)
        planner.emit('batch_clip_prepared', clip_id=cid, clip_index=index, frame_count=len(model_uris))
        return row
    finally:
        cleanup_media(work, owned)


def prepare_media(source, metadata, config, planner):
    """Persist both model-sized stamped frames and native samples clip by clip.

    ``prepare_workers`` clips are prepared concurrently (bounded regardless of the
    window count). Row order still follows clip index, so the persisted batch
    request order cannot depend on scheduling.
    """
    info = probe(source)
    prepare_workers = config.get('prepare_workers', 1)
    extractor = MediaExtractor(threads=config.get('media_threads', 1),
                               decode_slots=config.get('decode_slots', 1),
                               footer_workers=config.get('footer_workers', 1))
    total_decoder_mib = decoder_memory_warning(extractor.threads, extractor.decode_slots)
    planner.emit('batch_media_tuning', source=metadata['source_id'],
                 media_threads=extractor.threads, decode_slots=extractor.decode_slots,
                 footer_workers=extractor.footer_workers, prepare_workers=prepare_workers,
                 concurrent_decoder_threads=extractor.threads * extractor.decode_slots,
                 decoder_buffer_budget_mib=round(total_decoder_mib, 1),
                 note='Parallelism and memory change wall-clock cost only; they are not part of the annotation identity.')
    config['source_width'], config['source_height'] = info['width'], info['height']
    windows = list(itertools.islice(clip_windows(info['duration'], config['clip_seconds']),
                                    config['start_clip'], config['start_clip'] + config['max_clips']))
    if not windows:
        return []
    workers = min(positive_int(prepare_workers, 'prepare_workers'), len(windows))
    if workers == 1:
        return [_prepare_clip(source, metadata, config, planner, extractor, *window) for window in windows]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='prepare') as pool:
        return list(pool.map(lambda window: _prepare_clip(source, metadata, config, planner, extractor, *window),
                             windows))


def summary(state):
    return {'status':state['status'],'stage':state['current_stage'],
            'selected':len(state['config'].get('rows',[])),'completed':len(state['completed']),
            'failed_records':len(state['failures']),'eligible_next':len(state['eligible_rows']),
            'batch_jobs':{k:v.get('job',{}).get('name') for k,v in state['batches'].items()},
            'batch_job_states':{k:v.get('job',{}).get('state') for k,v in state['batches'].items()},
            'batch_job_errors':{k:v['job']['error'] for k,v in state['batches'].items() if v.get('job',{}).get('error')},
            'last_error':state.get('last_error'),'last_error_code':state.get('last_error_code'),
            'attention_reason':state.get('attention_reason'),
            'stop_schedule':state['status'] in {'complete','complete_with_errors','needs_attention','paused'}}


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['prepare','start','tick','status','reconcile']:
        p=sub.add_parser(name)
        p.add_argument('--project',default=os.environ.get('GOOGLE_CLOUD_PROJECT',''))
        p.add_argument('--location',default=os.environ.get('GOOGLE_CLOUD_LOCATION','global'))
        p.add_argument('--state-uri',required=True,help='gs://bucket/run-prefix/state.json')
        p.add_argument('--output-dir',type=Path,required=True)
        p.add_argument('--scratch-dir',type=Path,required=True)
        if name=='prepare':
            source=p.add_mutually_exclusive_group(required=True)
            source.add_argument('--source',action='append')
            source.add_argument('--local-video',type=Path)
            p.add_argument('--revision',default=REVISION)
            p.add_argument('--model',default='gemini-3.8-flash')
            p.add_argument('--gcs-prefix',required=True,help='A dedicated, unused gs://bucket/run-prefix')
            p.add_argument('--start-clip',type=int,default=0)
            p.add_argument('--max-clips',type=int,default=3,help='Per source; explicit scope, no full-day default')
            p.add_argument('--max-total-clips',type=int,default=2000)
            p.add_argument('--fps',type=float,default=1)
            p.add_argument('--clip-seconds',type=float,default=30)
            p.add_argument('--max-dim',type=int,default=1440)
            p.add_argument('--no-review',action='store_true')
            p.add_argument('--max-review-regions',type=int,default=4)
            p.add_argument('--max-output-tokens',type=int,default=None,
                           help='Generation budget shared by thinking and visible output; '
                                f'default {DEFAULT_MAX_OUTPUT_TOKENS}. Too low truncates annotation JSON on content-heavy clips')
            p.add_argument('--prepare-workers',type=int,default=None,
                           help='Clips prepared concurrently; default: CASTLE_PREPARE_WORKERS or CPU-derived')
            p.add_argument('--media-threads',type=int,default=None,
                           help='FFmpeg decoder threads per invocation; default: CASTLE_MEDIA_THREADS or CPU-derived')
            p.add_argument('--decode-slots',type=int,default=1,
                           help='Clips decoding concurrently; raise only with RAM to cover threads*slots')
            p.add_argument('--footer-workers',type=int,default=None,
                           help='Footer-stamping threads; default: CASTLE_FOOTER_WORKERS or CPU-derived')
            p.add_argument('--memory-hint-gib',type=float,default=16.,
                           help='Machine RAM this run may use, for the decoder-buffer admission check')
            p.add_argument('--submit',action='store_true',help='Submit initial audio batch after preparing; otherwise uploads/config only')
    return parser


def resolve_media_tuning(args):
    """Validate and resolve the media-tuning knobs for a prepare invocation.

    Pure and network-free so an impossible decoder budget is rejected before any
    credential or GCS client is constructed.
    """
    prepare_workers = args.prepare_workers if args.prepare_workers is not None else default_prepare_workers()
    media_threads = args.media_threads if args.media_threads is not None else env_decoder_default()
    footer_workers = args.footer_workers if args.footer_workers is not None else env_footer_default()
    for value, label, upper in ((prepare_workers, 'prepare-workers', 16), (media_threads, 'media-threads', 32),
                                (args.decode_slots, 'decode-slots', 16), (footer_workers, 'footer-workers', 16)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
            raise ValueError(f'Require --{label} in 1..{upper}')
    if not isinstance(args.memory_hint_gib, (int, float)) or isinstance(args.memory_hint_gib, bool) \
            or not 0 < args.memory_hint_gib:
        raise ValueError('--memory-hint-gib must be positive')
    decoder_mib = decoder_memory_warning(media_threads, args.decode_slots)
    if decoder_mib > args.memory_hint_gib * 1024:
        raise ValueError(f'--media-threads {media_threads} x --decode-slots {args.decode_slots} needs about '
                         f'{decoder_mib/1024:.1f} GiB of decoder buffers, above --memory-hint-gib '
                         f'{args.memory_hint_gib:g}; lower the threads or confirm the larger machine explicitly')
    return prepare_workers, media_threads, footer_workers


def main(argv=None):
    args=build_parser().parse_args(argv)
    parse_gs_uri(args.state_uri)
    tuning=resolve_media_tuning(args) if args.command=='prepare' else None
    args.output_dir.mkdir(parents=True,exist_ok=True);args.scratch_dir.mkdir(parents=True,exist_ok=True)
    log=EventLog();log.attach(args.output_dir/'batch-events.jsonl')
    cloud=BatchCloud(args.project,args.location)
    planner=TaskPlanner(cloud,args.scratch_dir,args.output_dir,log=log)
    engine=BatchEngine(cloud,args.state_uri,planner)
    if args.command=='prepare':
        parse_gs_uri(args.gcs_prefix)
        if args.state_uri!=args.gcs_prefix.rstrip('/')+'/state.json':
            raise ValueError('state-uri must be gcs-prefix/state.json')
        if args.start_clip<0 or not 1<=args.max_clips<=2000 or not 1<=args.max_total_clips<=2000:
            raise ValueError('Invalid bounded batch scope')
        if not 0<args.fps<=4 or not 0<args.clip_seconds<=30 or not 128<=args.max_dim<=2880 or not 0<=args.max_review_regions<=8:
            raise ValueError('Invalid media settings')
        prepare_workers,media_threads,footer_workers=tuning
        max_output_tokens=args.max_output_tokens if args.max_output_tokens is not None else DEFAULT_MAX_OUTPUT_TOKENS
        if isinstance(max_output_tokens,bool) or not isinstance(max_output_tokens,int) or not 1024<=max_output_tokens<=65536:
            raise ValueError('Require --max-output-tokens in 1024..65536')
        previous,_=cloud.read_state(args.state_uri)
        if previous is not None:
            raise ValueError('Run already initialized; use start/tick, never replace its scope')
        paths=args.source or [args.local_video]
        if len(paths)!=len(set(paths)):
            raise ValueError('Duplicate source paths would duplicate paid requests')
        if len(paths)*args.max_clips>args.max_total_clips:
            raise ValueError('Requested maximum exceeds max-total-clips')
        prompts={p.stem:p.read_text(encoding='utf-8') for p in (Path(__file__).parent/'prompts').glob('*.md')}
        cfg={'model':args.model,'project':args.project,'location':args.location,'gcs_prefix':args.gcs_prefix.rstrip('/'),
             'fps':args.fps,'clip_seconds':args.clip_seconds,'max_dim':args.max_dim,
             'start_clip':args.start_clip,'max_clips':args.max_clips,'review':not args.no_review,
             'max_review_regions':args.max_review_regions,'prompts':prompts,'rows':[],
             'prepare_workers':prepare_workers,'media_threads':media_threads,
             'decode_slots':args.decode_slots,'footer_workers':footer_workers,
             'max_output_tokens':max_output_tokens}
        cfg['code_hash']=code_hash()
        if args.source:
            from huggingface_hub import HfApi
            revision=HfApi().dataset_info(REPO,revision=args.revision).sha
        for path in paths:
            if args.source:
                meta=source_metadata(path,revision);source=download_source(path,revision,args.scratch_dir)
            else:
                source=Path(path).resolve(strict=True);s=source.stat()
                meta={'source_id':str(source),'file_size':s.st_size,'mtime_ns':s.st_mtime_ns,'viewpoint':'egocentric'}
            log.emit('batch_media_start',source=meta['source_id'])
            try:cfg['rows'].extend(prepare_media(source,meta,cfg,planner))
            finally:
                if args.source:remove_downloaded_source(source,args.scratch_dir)
            log.emit('batch_media_success',source=meta['source_id'],prepared_clips=len(cfg['rows']))
        if not cfg['rows']:raise ValueError('No clips selected')
        state=engine.initialize(cfg)
        if state['config']!=cfg:
            raise ValueError('Another worker initialized this prefix with a different scope; no submission performed')
        if args.submit:state=engine.start()
    else:
        stored,_=cloud.read_state(args.state_uri)
        if stored is None:raise ValueError('No batch state exists')
        if stored['config']['project']!=args.project or stored['config']['location']!=args.location:
            raise ValueError('Project/location must match the saved run')
        if args.command=='status':state=stored
        else:
            expected=code_hash()
            if stored['config'].get('code_hash')!=expected:
                raise ValueError('Use the pinned code version that initialized this run')
            state={'start':engine.start,'tick':engine.tick,'reconcile':engine.reconcile}[args.command]()
    atomic_json(args.output_dir/'batch-state.snapshot.json',state)
    result=summary(state);atomic_json(args.output_dir/'batch-summary.json',result)
    log.emit('batch_tick',**result)
    return 2 if state['status']=='needs_attention' else 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:raise SystemExit(130)
    except Exception as error:
        code=getattr(error,'code',None)
        print(json.dumps({'ok':False,'error_type':type(error).__name__,
                          'code':code if isinstance(code,int) else None,
                          'message':'Batch worker failed; inspect configuration and persisted state without exposing credentials.'}),file=sys.stderr)
        raise SystemExit(1)
