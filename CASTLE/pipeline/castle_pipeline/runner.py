"""Lazy scheduling and durable, phase-level checkpoints."""
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from dataclasses import dataclass, field
import itertools
import tempfile
import threading
import time
import psutil
import socket
import uuid
from contextlib import contextmanager

from .media import MediaExtractor, probe, env_decoder_default, env_footer_default, decoder_memory_warning
from .request_spec import (DEFAULT_MAX_OUTPUT_TOKENS, annotation_prompt, annotation_stage_context,
                           audio_stage_context, base_context, crop_label, frame_label,
                           review_prompt, review_stage_context)
from .schema import validate_annotation, validate_audio, apply_review, normalize_annotation
from .vertex import ProviderError
from .events import EventLog, ProgressReporter


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@contextmanager
def source_lease(directory):
    """Guard same-run writers on filesystems supporting exclusive creation.

    Never auto-break another process/job's lock. Bucket jobs must additionally
    use disjoint source shards: remote mount consistency varies by backend.
    """
    path = Path(directory) / 'run.lock'
    owner = {'nonce': uuid.uuid4().hex, 'pid': os.getpid(), 'host': socket.gethostname(),
             'job_id': os.environ.get('JOB_ID'), 'created_at': time.time()}
    try:
        with path.open('x', encoding='utf-8') as handle:
            json.dump(owner, handle)
    except FileExistsError:
        raise RuntimeError('Output source is locked; verify the prior owner is stopped before removing its lock') from None
    try:
        yield owner
    finally:
        try:
            # Inventory and owner check before deleting this exact owned file.
            current = json.loads(path.read_text(encoding='utf-8'))
            if current.get('nonce') == owner['nonce']:
                path.unlink()
        except (OSError, ValueError):
            pass


class Checkpoints:
    def __init__(self, directory, fingerprint_value):
        self.directory = Path(directory)
        self.fingerprint = fingerprint_value

    def load(self, phase):
        try:
            record = json.loads((self.directory / f'{phase}.json').read_text(encoding='utf-8'))
            if record.get('fingerprint') == self.fingerprint and record.get('ok') is True:
                return {k: v for k, v in record.items() if k not in ('fingerprint', 'ok')} if phase == 'final' else record['result']
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def save(self, phase, result):
        payload = result if phase == 'final' else {'result': result}
        atomic_json(self.directory / f'{phase}.json', {**payload, 'fingerprint': self.fingerprint, 'ok': True})


def clip_windows(duration, clip_seconds=30):
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(clip_seconds) or clip_seconds <= 0:
        raise ValueError('Positive finite durations required')
    for i in range(math.ceil(duration / clip_seconds)):
        start = i * float(clip_seconds)
        yield i, start, min(float(clip_seconds), duration - start)


def bounded_map(items, worker, workers):
    """At most workers inputs are materialized; yield before consuming another."""
    if workers < 1:
        raise ValueError('workers must be positive')
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='clip') as pool:
        pending = {}
        try:
            for _ in range(workers):
                item = next(iterator, None)
                if item is None:
                    break
                pending[pool.submit(worker, item)] = item
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    item = pending.pop(future)
                    yield future.result()
                    new = next(iterator, None)
                    if new is not None:
                        pending[pool.submit(worker, new)] = new
        finally:
            for future in pending:
                future.cancel()


def memory_estimate(source_seconds=3600, workers=3, width=3840, height=2160,
                    max_dim=1440, fps=1, clip_seconds=30, jpeg_mib=.35):
    """Engineering budget, not an RSS prediction. Source size stays on disk."""
    if min(source_seconds, workers, width, height, max_dim, fps, clip_seconds, jpeg_mib) <= 0:
        raise ValueError('Memory estimate inputs must be positive')
    raw = width * height * 3
    scale = min(1., max_dim / max(width, height))
    scaled_frame_mib = raw * scale**2 / 2**20
    frame_count = math.ceil(fps * clip_seconds)
    jpeg_clip = frame_count * jpeg_mib
    # One decoder (conservative allowance), runtime, and per-inflight SDK copies.
    per_worker = 4 * jpeg_clip + 2 * scaled_frame_mib + 32 + 2 * clip_seconds * 16000 * 2 / 2**20
    return {
        'source_seconds': source_seconds, 'workers': workers, 'frames_per_clip': frame_count,
        'raw_4k_rgb_frame_mib': raw / 2**20,
        'all_raw_50fps_gib': source_seconds * 50 * raw / 2**30,
        'all_raw_1fps_gib': source_seconds * raw / 2**30,
        'jpeg_mib_per_frame_assumption': jpeg_mib,
        'per_worker_mib_budget': round(per_worker, 1),
        'single_decoder_mib_allowance': 768,
        'runtime_mib_allowance': 512,
        'bounded_working_set_mib_estimate': round(1280 + workers * per_worker, 1),
        'note': 'Budget only; excludes file cache, SDK variation and external processes. Measure cgroup memory and process-tree RSS in the actual job.'}


def rss_bytes():
    process = psutil.Process()
    processes = [process] + process.children(recursive=True)
    total = 0
    for child in processes:
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


class MemoryMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.peak_rss = 0
        self.peak_cgroup = 0
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, rss_bytes())
            try:
                value = int(Path('/sys/fs/cgroup/memory.current').read_text())
                self.peak_cgroup = max(self.peak_cgroup, value)
            except (OSError, ValueError):
                pass
            self.stop.wait(.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

    def result(self):
        return {'peak_process_tree_rss_mib': round(self.peak_rss / 2**20, 2),
                'peak_cgroup_current_mib': round(self.peak_cgroup / 2**20, 2) if self.peak_cgroup else None,
                'sample_interval_sec': .2,
                'note': 'RSS sums processes and may double-count shared pages; cgroup includes file cache. Peaks shorter than sampling interval may be missed.'}


def cleanup_media(directory, registered_paths):
    """Delete only explicitly registered generated files after inventory checks."""
    directory = Path(directory).resolve()
    registered = {Path(p).resolve() for p in registered_paths}
    # Directory was created by mkdtemp for this clip. Unknown files are preserved.
    existing = list(directory.rglob('*')) if directory.exists() else []
    for path in existing:
        resolved = path.resolve()
        if path.is_symlink() or not resolved.is_relative_to(directory):
            continue
        if path.is_file() and resolved in registered:
            path.unlink()
    for path in sorted((p for p in existing if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if path.is_symlink():
            continue
        try:
            path.rmdir()  # Only removes empty directories, never a tree.
        except OSError:
            pass
    try:
        directory.rmdir()
    except OSError:
        pass


@dataclass
class RunConfig:
    output_dir: Path
    scratch_dir: Path
    workers: int = 3
    fps: float = 1.
    clip_seconds: float = 30.
    max_dim: int = 1440
    review: bool = True
    max_review_regions: int = 4
    stamp: bool = True
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_clips: int | None = None
    start_clip: int = 0
    memory_soft_limit_gib: float = 12.
    keep_media: bool = False
    progress_interval_sec: float = 600.
    media_threads: int = field(default_factory=env_decoder_default)
    decode_slots: int = 1
    footer_workers: int = field(default_factory=env_footer_default)
    prompts_dir: Path = Path(__file__).resolve().parents[1] / 'prompts'

    def __post_init__(self):
        self.output_dir, self.scratch_dir, self.prompts_dir = map(Path, (self.output_dir, self.scratch_dir, self.prompts_dir))
        if not 1 <= self.workers <= 16 or not 0 < self.clip_seconds <= 30 or not 0 < self.fps <= 4:
            raise ValueError('Require workers 1..16, clip duration (0,30], sampling fps (0,4]')
        if not 128 <= self.max_dim <= 2880 or not 0 <= self.max_review_regions <= 8:
            raise ValueError('Invalid image dimensions or crop budget')
        if isinstance(self.max_output_tokens, bool) or not isinstance(self.max_output_tokens, int) \
                or not 1024 <= self.max_output_tokens <= 65536:
            raise ValueError('Require max_output_tokens in 1024..65536')
        if self.start_clip < 0 or (self.max_clips is not None and self.max_clips < 1):
            raise ValueError('Invalid selected clip range')
        if not math.isfinite(self.memory_soft_limit_gib) or self.memory_soft_limit_gib <= 0:
            raise ValueError('Memory limit must be positive and finite')
        for value, name, upper in ((self.media_threads, 'media_threads', 32),
                                   (self.decode_slots, 'decode_slots', 16),
                                   (self.footer_workers, 'footer_workers', 16)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise ValueError(f'Require {name} in 1..{upper}')
        if not math.isfinite(self.progress_interval_sec) or self.progress_interval_sec <= 0:
            raise ValueError('Progress interval must be positive and finite')


class Pipeline:
    def __init__(self, config, provider, extractor=None, log=None):
        self.config, self.provider, self.log = config, provider, log
        self.events = EventLog(sink=log)
        if hasattr(provider, 'events'):
            provider.events = self.events
        self.extractor = extractor or MediaExtractor(threads=config.media_threads,
                                                     decode_slots=config.decode_slots,
                                                     footer_workers=config.footer_workers)
        self.stopped = threading.Event()
        self.prompts = {name: (config.prompts_dir / f'{name}.md').read_text(encoding='utf-8')
                        for name in ('audio', 'annotation', 'review', 'review_regions')}

    def identity(self, metadata):
        cfg = self.config
        return fingerprint({
            'source': metadata, 'model': self.provider.model,
            'service_tier': getattr(self.provider, 'service_tier', 'offline'),
            'fps': cfg.fps, 'clip_seconds': cfg.clip_seconds, 'max_dim': cfg.max_dim,
            'review': cfg.review, 'max_review_regions': cfg.max_review_regions, 'stamp': cfg.stamp,
            'max_output_tokens': cfg.max_output_tokens,
            'prompts': self.prompts,
            'implementation': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in Path(__file__).parent.glob('*.py')}})

    def _phase(self, checkpoints, name, prompt, context, images, audio, validator, normalizer=None):
        clip_id = context['clip_id']
        cached = checkpoints.load(name)
        if cached is not None:
            try:
                validator(cached['data'])
                self.events.emit('stage_reused', clip_id=clip_id, phase=name)
                return cached
            except (ValueError, KeyError, TypeError):
                pass
        if self.stopped.is_set():
            raise ProviderError('Run stopped before request', category='cancelled', fatal=True)
        started = time.monotonic()
        self.events.emit('stage_start', clip_id=clip_id, phase=name)
        result = self.provider.generate(prompt, json.dumps(context, ensure_ascii=False), images, audio=audio,
                                        max_output_tokens=self.config.max_output_tokens)
        try:
            if normalizer is not None:
                normalized, changes = normalizer(result['data'])
                if changes:
                    result = {**result, 'raw_data': result['data'], 'data': normalized, 'normalization': changes}
                    self.events.emit('annotation_normalized', clip_id=clip_id, phase=name, changes=changes)
            validator(result['data'])
        except (ValueError, KeyError, TypeError) as error:
            # Preserve the paid response for diagnosis, but never mark it reusable.
            atomic_json(checkpoints.directory / f'{name}.invalid.json', {
                'fingerprint': checkpoints.fingerprint, 'ok': False, 'validation_reason': str(error), 'result': result})
            raise
        checkpoints.save(name, result)
        self.events.emit('stage_success', clip_id=clip_id, phase=name,
                         elapsed_sec=round(time.monotonic() - started, 3), usage=result.get('usage', {}))
        return result

    def _clip(self, source, metadata, identity, directory, item):
        index, start, duration = item
        clip_id = f'{metadata["source_id"]}:{index}'
        cfg = self.config
        checkpoint = Checkpoints(directory / 'clips' / f'{index:05d}', identity)
        sources = {'video_0'}
        final = checkpoint.load('final')
        if final is not None:
            try:
                validate_annotation(final['annotation'], duration, set(final['input_sources']))
                return {'index': index, 'ok': True, 'reused': True, 'review_required': final['review_required']}
            except (ValueError, KeyError, TypeError):
                pass
        work = None
        registered = []
        stage = 'prepare'
        try:
            if self.stopped.is_set():
                raise ProviderError('Run stopped', category='cancelled', fatal=True)
            if rss_bytes() > cfg.memory_soft_limit_gib * 2**30:
                raise ProviderError('Process-tree memory admission limit reached', category='memory', fatal=True)
            work = Path(tempfile.mkdtemp(prefix=f'clip-{index:05d}-', dir=cfg.scratch_dir))
            prepare_started = time.monotonic()
            self.events.emit('stage_start', clip_id=clip_id, phase='prepare')
            media = self.extractor.prepare(source, start, duration, work, fps=cfg.fps,
                                           max_dim=cfg.max_dim, stamp=cfg.stamp)
            self.events.emit('stage_success', clip_id=clip_id, phase='prepare',
                             elapsed_sec=round(time.monotonic() - prepare_started, 3),
                             frames=len(media.frame_paths), has_audio=media.audio_path is not None)
            registered.extend(media.frame_paths)
            if media.audio_path:
                registered.append(media.audio_path)
                sources.add('audio_0')
            if not media.frame_paths:
                raise ValueError('No usable sampled frames were decoded')
            exocentric = metadata.get('viewpoint') == 'exocentric'
            stage = 'audio'
            if media.audio_path:
                audio = self._phase(checkpoint, 'audio', self.prompts['audio'],
                                    audio_stage_context(base_context('audio', clip_id, duration, metadata, start)),
                                    [], media.audio_path,
                                    lambda data: validate_audio(data, duration))
            else:
                audio = {'data': {'summary': 'No audio track supplied.', 'utterances': [], 'sound_events': [],
                                  'uncertainties': ['No audio evidence available.']}, 'usage': {}, 'skipped': True}
                checkpoint.save('audio', audio)
                self.events.emit('stage_skipped', clip_id=clip_id, phase='audio', reason='no_audio_track')
            # Request construction is shared with the batch planner: same context
            # keys and order, same prompt suffixes, same media labels, so an
            # online request mirrors its Vertex Batch counterpart byte-for-byte
            # apart from inline data versus GCS URIs.
            context = annotation_stage_context(base_context('annotation', clip_id, duration, metadata, start),
                                               media.frame_times, audio['data'], bool(media.audio_path))
            prompt = annotation_prompt(self.prompts, review_enabled=cfg.review,
                                       max_review_regions=cfg.max_review_regions, exocentric=exocentric)
            images = [(frame_label(i, t), path)
                      for i, (t, path) in enumerate(zip(media.frame_times, media.frame_paths))]
            def validate_first(data):
                base = {key: value for key, value in data.items() if key != 'review_regions'}
                validate_annotation(base, duration, sources)
                regions = data.get('review_regions', [])
                if not isinstance(regions, list) or len(regions) > cfg.max_review_regions:
                    raise ValueError('Invalid review region count')
                for region in regions:
                    if not isinstance(region.get('frame_index'), int) or not 0 <= region['frame_index'] < len(images):
                        raise ValueError('Crop frame reference outside input')
                    box = region.get('box_2d')
                    if not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in box):
                        raise ValueError('Invalid crop coordinates')
                    if not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
                        raise ValueError('Crop coordinates outside original image')
            stage = 'annotation'
            def normalize_first(data):
                base = {key: value for key, value in data.items() if key != 'review_regions'}
                normalized, changes = normalize_annotation(base, duration, sources)
                if 'review_regions' in data:
                    normalized['review_regions'] = data['review_regions']
                return normalized, changes
            first = self._phase(checkpoint, 'annotation', prompt, context, images, None, validate_first, normalize_first)
            annotation = {k: v for k, v in first['data'].items() if k != 'review_regions'}
            regions = first['data'].get('review_regions', []) if cfg.review else []
            review = None
            if regions:
                crop_images = []
                crops = []
                stage = 'crops'
                crop_started = time.monotonic()
                self.events.emit('stage_start', clip_id=clip_id, phase='crops', regions=len(regions))
                for i, region in enumerate(regions):
                    t = media.frame_times[region['frame_index']]
                    path = work / f'review-{i:02d}.jpg'
                    registered.append(path)
                    self.extractor.extract_crop(source, start + t, region['box_2d'], path, max_dim=cfg.max_dim)
                    sid = f'crop_{i}'
                    sources.add(sid)
                    crops.append({'source_id': sid, 'time_sec': t})
                    crop_images.append((crop_label(sid, t), path))
                self.events.emit('stage_success', clip_id=clip_id, phase='crops',
                                 elapsed_sec=round(time.monotonic() - crop_started, 3), regions=len(regions))
                review_context = review_stage_context(base_context('review', clip_id, duration, metadata, start),
                                                      annotation, crops)
                stage = 'review'
                review = self._phase(checkpoint, 'review', review_prompt(self.prompts), review_context, crop_images, None,
                                     lambda data: apply_review(annotation, data, duration, sources))
                annotation = apply_review(annotation, review['data'], duration, sources)
            else:
                self.events.emit('stage_skipped', clip_id=clip_id, phase='review',
                                 reason='disabled' if not cfg.review else 'no_regions')
            result = {'source': metadata, 'clip': {'index': index, 'start_offset_sec': start, 'duration_sec': duration,
                       'frame_times_sec': media.frame_times}, 'input_sources': sorted(sources),
                      'annotation': annotation,
                      'review_required': bool(review and review['data']['resegmentation_requests']) or any(
                          change['code'] == 'NONEDGE_ONGOING_TO_UNCERTAIN' for change in first.get('normalization', [])),
                      'normalization': first.get('normalization', []),
                      'resegmentation_requests': review['data']['resegmentation_requests'] if review else [],
                      'model': self.provider.model, 'usage': {'audio': audio.get('usage', {}),
                       'annotation': first.get('usage', {}), 'review': review.get('usage', {}) if review else {}},
                      'completed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            checkpoint.save('final', result)
            return {'index': index, 'ok': True, 'reused': False, 'review_required': result['review_required']}
        except Exception as error:
            category = getattr(error, 'category', 'validation' if isinstance(error, ValueError) else 'media')
            fatal = bool(getattr(error, 'fatal', False))
            if fatal:
                self.stopped.set()
            diagnostic = {'index': index, 'ok': False, 'stage': stage, 'category': category,
                          'error_type': type(error).__name__, 'fatal': fatal,
                          'code': getattr(error, 'code', category.upper()),
                          'attempts': getattr(error, 'attempts', None), 'usage': getattr(error, 'usage', None)}
            if isinstance(error, ValueError):
                diagnostic['validation_reason'] = str(error)
            self.events.emit('stage_failed', clip_id=clip_id, phase=stage, category=category,
                             code=diagnostic['code'], error_type=type(error).__name__,
                             validation_reason=diagnostic.get('validation_reason'))
            atomic_json(checkpoint.directory / 'error.json', diagnostic)
            return diagnostic
        finally:
            if work and not cfg.keep_media:
                cleanup_media(work, registered)

    def run_source(self, source, metadata):
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        cfg.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.stopped.clear()
        info = probe(source)
        identity = self.identity(metadata)
        safe_id = hashlib.sha256(metadata['source_id'].encode()).hexdigest()[:12]
        directory = cfg.output_dir / identity[:16] / safe_id
        directory.mkdir(parents=True, exist_ok=True)
        with source_lease(directory):
            self.events.attach(directory / 'events.jsonl')
            try:
                return self._run_source_locked(source, metadata, identity, directory, info)
            finally:
                self.events.attach(None)

    def _run_source_locked(self, source, metadata, identity, directory, info):
        cfg = self.config
        atomic_json(directory / 'run.json', {'fingerprint': identity, 'source': metadata, 'probe': info,
                    'model': self.provider.model, 'fps': cfg.fps, 'clip_seconds': cfg.clip_seconds})
        windows = itertools.islice(clip_windows(info['duration'], cfg.clip_seconds), cfg.start_clip,
                                    None if cfg.max_clips is None else cfg.start_clip + cfg.max_clips)
        total = max(0, math.ceil(info['duration'] / cfg.clip_seconds) - cfg.start_clip)
        total = min(total, cfg.max_clips) if cfg.max_clips is not None else total
        if not total:
            raise ValueError('Selected clip range is empty')
        counts = {'selected': total, 'completed': 0, 'failed': 0, 'reused': 0, 'review_required': 0}
        counts_lock = threading.Lock()
        def progress_status():
            with counts_lock:
                state = dict(counts)
            limiter = getattr(self.provider, 'limiter', None)
            return {'source': metadata['source_id'], **state,
                    'pending': total - state['completed'] - state['failed'],
                    'process_tree_rss_mib': round(rss_bytes() / 2**20, 2),
                    'aimd': limiter.snapshot() if limiter else None}
        decoder_mib = decoder_memory_warning(cfg.media_threads, cfg.decode_slots)
        self.events.emit('source_start', source=metadata['source_id'], selected=total,
                         model=self.provider.model, service_tier=getattr(self.provider, 'service_tier', None),
                         progress_interval_sec=cfg.progress_interval_sec,
                         media_threads=cfg.media_threads, decode_slots=cfg.decode_slots,
                         footer_workers=cfg.footer_workers,
                         decoder_buffer_budget_mib=round(decoder_mib, 1))
        if decoder_mib > 0.25 * cfg.memory_soft_limit_gib * 1024:
            self.events.emit('media_tuning_warning', source=metadata['source_id'],
                             media_threads=cfg.media_threads, decode_slots=cfg.decode_slots,
                             decoder_buffer_budget_mib=round(decoder_mib, 1),
                             memory_soft_limit_mib=round(cfg.memory_soft_limit_gib * 1024, 1),
                             reason='concurrent_decoder_buffers_large_relative_to_memory_limit')
        with MemoryMonitor() as monitor, ProgressReporter(self.events, cfg.progress_interval_sec, progress_status):
            results = bounded_map(windows, lambda item: self._clip(source, metadata, identity, directory, item), cfg.workers)
            try:
                for result in results:
                    with counts_lock:
                        counts['completed' if result['ok'] else 'failed'] += 1
                        counts['reused'] += int(result.get('reused', False))
                        counts['review_required'] += int(result.get('review_required', False))
                    self.events.emit('clip_reused' if result.get('reused') else 'clip_completed' if result['ok'] else 'clip_failed',
                                     source=metadata['source_id'], clip_id=f'{metadata["source_id"]}:{result["index"]}', **result)
                    if result.get('fatal'):
                        break
            finally:
                results.close()
        counts.update(memory=monitor.result(), output_dir=str(directory),
                      unprocessed=counts['selected'] - counts['completed'] - counts['failed'], fatal=self.stopped.is_set())
        counts['telemetry'] = self.events.snapshot()
        atomic_json(directory / 'summary.json', counts)
        # Stream one clip at a time; never build an all-day caption object.
        target = directory / 'captions.jsonl'
        temp = target.with_suffix('.jsonl.tmp')
        with temp.open('w', encoding='utf-8') as stream:
            for path in sorted((directory / 'clips').glob('*/final.json')):
                record = Checkpoints(path.parent, identity).load('final')
                if record is not None:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        os.replace(temp, target)
        self.events.emit('source_completed', source=metadata['source_id'], **counts)
        return counts
