"""Disk-backed, bounded FFmpeg clip extraction.

Sampling timestamps are the clip-local requested grid ``k / fps``. A sample uses
the first decoded source frame at or after ``start + k / fps`` (within one
source-frame interval for constant-rate video; variable-rate gaps may be longer).
They are sample labels, not claims of more precise source capture timestamps.
No frame is duplicated to fill a partial tail. Rotation metadata is ignored so
normalized crop coordinates always refer to the same original coded content.

Decoder thread count and the number of clips decoding at once are runtime tuning
knobs: they change wall-clock speed and memory, never the decoded pixels, the
sampled frame, or the footer bytes. Raising them therefore does not invalidate a
configuration fingerprint or any existing checkpoint.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading

from PIL import Image, ImageDraw, ImageFont


# ``os.cpu_count()`` reports the host's cores, not the cgroup quota, so on a small
# container these are deliberately modest rather than "all cores". Operators
# raise them explicitly (CLI flag or CASTLE_MEDIA_THREADS / CASTLE_FOOTER_WORKERS).
DEFAULT_DECODER_THREADS = min(4, max(1, (os.cpu_count() or 1) - 1))
DEFAULT_FOOTER_WORKERS = min(4, max(1, (os.cpu_count() or 1) // 2))
# Per-footprint allowance for 4K H.264 decode buffers, used only to warn.
DECODER_MIB_PER_THREAD = 40


def positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _env_positive_int(name: str, fallback: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return fallback
    try:
        return positive_int(int(raw.strip()), f"environment variable {name}")
    except ValueError:
        raise ValueError(f"Environment variable {name} must be a positive integer") from None


def env_decoder_default() -> int:
    return _env_positive_int("CASTLE_MEDIA_THREADS", DEFAULT_DECODER_THREADS)


def env_footer_default() -> int:
    return _env_positive_int("CASTLE_FOOTER_WORKERS", DEFAULT_FOOTER_WORKERS)


def decoder_memory_warning(decoder_threads: int, decode_slots: int) -> float:
    """Rough budget for concurrent 4K decode buffers, in MiB."""
    return float(decoder_threads * decode_slots * DECODER_MIB_PER_THREAD)


@dataclass
class PreparedClip:
    frame_paths: list[Path]
    frame_times: list[float]
    audio_path: Path | None


def _run(args: list[str], *, capture: bool = False) -> bytes:
    # Do not retain media or unbounded diagnostics in Python, or include command
    # arguments in exceptions: source paths can contain access credentials.
    try:
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(args, stdin=subprocess.DEVNULL,
                                    stdout=output if capture else subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, check=False)
            if result.returncode:
                raise RuntimeError(f"Media subprocess failed (exit {result.returncode})")
            if not capture:
                return b""
            output.seek(0)
            data = output.read(131073)
            if len(data) > 131072:
                raise RuntimeError("Media metadata exceeded the 128 KiB limit")
            return data
    except OSError:
        raise RuntimeError("Could not start media subprocess or access its output") from None


def probe(source: Path) -> dict:
    """Read dimensions, video duration in seconds, and audio availability.

    Reject sources without a finite video-track duration or DURATION tag;
    container duration may be longer because of audio, and is never substituted.
    """
    raw = _run(["ffprobe", "-v", "error", "-show_entries",
                "stream=codec_type,width,height,duration:stream_tags=DURATION",
                "-of", "json", str(source)], capture=True)
    try:
        info = json.loads(raw)
        streams = info.get("streams", [])
        video = next(s for s in streams if s.get("codec_type") == "video")
        duration_value = video.get("duration")
        if duration_value in (None, "N/A"):
            tag = video.get("tags", {}).get("DURATION")
            if tag:
                hours, minutes, seconds = tag.split(":")
                duration_value = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        duration = float(duration_value)
        width, height = int(video["width"]), int(video["height"])
        if not math.isfinite(duration) or duration <= 0 or min(width, height) <= 0:
            raise ValueError
        return {"duration": duration, "width": width, "height": height,
                "has_audio": any(s.get("codec_type") == "audio" for s in streams)}
    except (ValueError, TypeError, KeyError, StopIteration):
        raise RuntimeError("Source has no usable video dimensions or finite video-track duration") from None


def _finite(value: float, name: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return number


def _dimension(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 32:
        raise ValueError("max_dim must be an integer of at least 32")
    return value


def _scale(max_dim: int, footer: int = 0) -> str:
    return (f"scale=w='min(iw,{max_dim})':h='min(ih,{max_dim - footer})':"
            "force_original_aspect_ratio=decrease,setsar=1")


def _cleanup_failed(work: Path, frame_limit: int) -> None:
    """Inventory a call-owned directory and remove only its generated names."""
    known = {f"frame_{index:06d}.jpg" for index in range(1, frame_limit + 1)} | {"audio.wav"}
    try:
        contents = list(work.iterdir())
    except OSError:
        return
    for path in contents:
        if path.name in known and path.is_file() and not path.is_symlink():
            try:
                if path.resolve().parent == work.resolve():
                    path.unlink()
            except OSError:
                pass
    try:
        work.rmdir()  # Only an empty directory; preserve all unknown files.
    except OSError:
        pass


def footer(path: Path, seconds: float, height: int) -> None:
    """Append the clip-local time to the bottom of an existing frame, in place.

    Module-level so concurrent callers (including the Batch planner) can reuse it
    without constructing a MediaExtractor.
    """
    with Image.open(path) as original:
        with Image.new("RGB", (original.width, original.height + height), "black") as result:
            result.paste(original, (0, 0))
            draw = ImageDraw.Draw(result)
            draw.text((6, original.height + 5), f"t={seconds:.3f}s", fill="white",
                      font=ImageFont.load_default(size=13))
            result.save(path, "JPEG", quality=95)


class MediaExtractor:
    """One shared semaphore gates all decode and crop work per instance.

    ``threads`` is the FFmpeg decoder/filter thread count for every invocation:
    on 4K H.264 this is the dominant cost, and it scales close to linearly until
    the source runs out of parallel decode capacity. ``decode_slots`` is how many
    clips may decode at the same time; total concurrent decode threads is roughly
    ``threads * decode_slots``, which is what the memory budget must cover.

    Once frames are on disk, the footer pass is pure Pillow work and runs in a
    pool of ``footer_workers`` threads. Footer output does not depend on that
    count: each frame is read and rewritten by exactly one thread. Duration is
    capped at 30 s and output at 900 images (fps <= 30). Media travels directly
    between subprocesses and disk, never through captured stdout.
    """

    def __init__(self, threads: int = 1, decode_slots: int = 1, footer_workers: int = 1):
        self.threads = positive_int(threads, "threads")
        decode_slots = positive_int(decode_slots, "decode_slots")
        self.footer_workers = positive_int(footer_workers, "footer_workers")
        self.decode_slots = decode_slots
        self._decode = threading.BoundedSemaphore(decode_slots)

    def _command(self) -> list[str]:
        return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-threads", str(self.threads), "-filter_threads", str(self.threads)]

    def prepare(self, source: Path, start: float, duration: float, outdir: Path,
                fps: float = 1, max_dim: int = 1440, stamp: bool = True) -> PreparedClip:
        start = _finite(start, "start")
        duration = _finite(duration, "duration", positive=True)
        fps = _finite(fps, "fps", positive=True)
        max_dim = _dimension(max_dim)
        if duration > 30:
            raise ValueError("duration must not exceed 30 seconds")
        if fps > 30:
            raise ValueError("fps must not exceed 30")
        with self._decode:
            metadata = probe(source)
            if start >= metadata["duration"]:
                raise ValueError("start must precede video end")
            length = min(duration, metadata["duration"] - start)
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix="clip-", dir=outdir))
            try:
                # Input-side seek and duration bound decoding. select retains the
                # original post-seek time base; resetting before selection would
                # offset non-frame-aligned seeks and falsely label the samples.
                footer = 24 if stamp else 0
                filters = (f"select='gte(t,selected_n/{fps:.12g})',"
                           + _scale(max_dim, footer))
                _run(self._command() + ["-ss", f"{start:.12g}", "-t", f"{length:.12g}",
                     "-noautorotate", "-i", str(source), "-map", "0:v:0", "-an",
                     "-vf", filters, "-fps_mode", "vfr", "-frames:v",
                     str(math.ceil(length * fps)), "-q:v", "2", "-threads", str(self.threads),
                     str(work / "frame_%06d.jpg")])
                paths = sorted(work.glob("frame_*.jpg"))
                times = [index / fps for index in range(len(paths))]
                if not paths:
                    raise RuntimeError("Requested clip contains no decodable frames")
                if stamp:
                    self.stamp_frames(paths, times, footer)
                audio = None
                if metadata["has_audio"]:
                    audio = work / "audio.wav"
                    # WAV has no start timestamp. Materialize leading/internal
                    # timestamp gaps as silence, with a silent full-clip reference
                    # so even a clip before the first audio packet has its timeline.
                    audio_filter = ("[0:a:0]aresample=16000:async=1:first_pts=0[aligned];"
                                    "[1:a:0][aligned]amix=inputs=2:duration=first:"
                                    f"dropout_transition=0:normalize=0,atrim=duration={length:.12g}[audio]")
                    _run(self._command() + ["-ss", f"{start:.12g}", "-t", f"{length:.12g}",
                         "-i", str(source), "-f", "lavfi", "-t", f"{length:.12g}", "-i",
                         "anullsrc=r=16000:cl=mono", "-filter_complex_threads", str(self.threads),
                         "-filter_complex", audio_filter, "-map", "[audio]", "-vn", "-ac", "1", "-ar",
                         "16000", "-c:a", "pcm_s16le", "-threads", str(self.threads), str(audio)])
                return PreparedClip(paths, times, audio)
            except BaseException:
                _cleanup_failed(work, math.ceil(length * fps))
                raise

    def stamp_frames(self, paths, times, height: int) -> None:
        """Add the clip-local footer to already-written frames.

        Each frame is owned by exactly one worker, so the result is byte-for-byte
        the same as a serial pass regardless of ``footer_workers``. A failure in
        any frame propagates after the pool drains; the caller's cleanup then
        removes the partially stamped clip.
        """
        jobs = list(zip(paths, times))
        if not jobs:
            return
        if self.footer_workers == 1:
            for path, seconds in jobs:
                footer(path, seconds, height)
            return
        with ThreadPoolExecutor(max_workers=self.footer_workers, thread_name_prefix="footer") as pool:
            list(pool.map(lambda job: footer(job[0], job[1], height), jobs))

    @staticmethod
    def _footer(path: Path, seconds: float, height: int) -> None:
        footer(path, seconds, height)

    def extract_crop(self, source: Path, absolute_sec: float, box: list[float],
                     outpath: Path, max_dim: int = 1440) -> Path:
        """Seek a native frame, crop [top,left,bottom,right] / 1000, then scale.

        Crop edges round outward to native pixels, preserving the full requested
        region. No timestamp footer enters the coordinate calculation.
        """
        absolute_sec = _finite(absolute_sec, "absolute_sec")
        max_dim = _dimension(max_dim)
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("box must contain top, left, bottom, right")
        top, left, bottom, right = [_finite(value, "box coordinate") for value in box]
        if not (0 <= top < bottom <= 1000 and 0 <= left < right <= 1000):
            raise ValueError("box must have positive area inside 0..1000")
        with self._decode:
            metadata = probe(source)
            if absolute_sec >= metadata["duration"]:
                raise ValueError("absolute_sec must precede video end")
            width, height = metadata["width"], metadata["height"]
            x, y = math.floor(left * width / 1000), math.floor(top * height / 1000)
            crop_w = math.ceil(right * width / 1000) - x
            crop_h = math.ceil(bottom * height / 1000) - y
            outpath = Path(outpath)
            outpath.parent.mkdir(parents=True, exist_ok=True)
            _run(self._command() + ["-ss", f"{absolute_sec:.12g}", "-noautorotate",
                 "-i", str(source), "-map", "0:v:0", "-an", "-frames:v", "1",
                 "-vf", f"crop={crop_w}:{crop_h}:{x}:{y}:exact=1," + _scale(max_dim),
                 "-q:v", "2", "-threads", str(self.threads), str(outpath)])
            if not outpath.is_file():
                raise RuntimeError("Requested crop contains no decodable frame")
            return outpath
