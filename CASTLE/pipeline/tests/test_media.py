"""Behavioral FFmpeg tests; all generated artifacts stay in _test/media-* ."""
import importlib
import math
from pathlib import Path
import subprocess
import tempfile
import wave
from array import array
import threading
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import pytest


@pytest.fixture(scope="module")
def api():
    try:
        module = importlib.import_module("castle_pipeline.media")
    except ModuleNotFoundError:
        module = None
    assert module is not None, "bounded media extraction module is missing"
    return module


@pytest.fixture(scope="module")
def media_dir():
    parent = Path(__file__).resolve().parents[1] / "_test"
    parent.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="media-", dir=parent))


@pytest.fixture(scope="module")
def sources(media_dir):
    # Red left half, blue right half: native normalized crop has a clear oracle.
    paths = {}
    for audio in (False, True):
        path = media_dir / ("audio.mp4" if audio else "silent.mp4")
        args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-f", "lavfi", "-i",
                "color=c=red:s=640x360:r=50:d=3.24,drawbox=x=320:y=0:w=320:h=360:color=blue:t=fill"]
        if audio:
            # Audio deliberately outlasts video: container duration is wrong
            # for clip boundaries and must never replace video-track duration.
            args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4.24"]
        args += ["-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p"]
        if audio:
            args += ["-c:a", "aac"]
        subprocess.run(args + [str(path)], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
        paths[audio] = path
    return paths


def test_probe_uses_video_duration_and_detects_audio(api, sources):
    for has_audio, source in sources.items():
        result = api.probe(source)
        assert result["duration"] == pytest.approx(3.24, abs=0.001)
        assert (result["width"], result["height"]) == (640, 360)
        assert result["has_audio"] is has_audio


def test_prepare_bounds_frames_adds_footer_and_extracts_pcm_audio(api, sources, media_dir):
    clip = api.MediaExtractor().prepare(sources[True], 0.2, 2.2,
                                        media_dir / "prepared", max_dim=320)
    assert clip.frame_times == [0.0, 1.0, 2.0]
    assert len(clip.frame_paths) == 3
    for frame in clip.frame_paths:
        with Image.open(frame) as im:
            assert max(im.size) <= 320
            assert im.height > im.width * 360 / 640
            r, g, b = im.convert("RGB").getpixel((im.width // 4, im.height // 3))
            assert r > 200 and g < 30 and b < 30
            footer = im.convert("RGB").crop((0, im.height - 22, im.width, im.height))
            assert max(footer.getpixel((im.width - 5, 15))) < 20
            assert footer.convert("L").getextrema()[1] > 180
    with wave.open(str(clip.audio_path), "rb") as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) == (1, 16000, 2)
        assert wav.getnframes() / wav.getframerate() == pytest.approx(2.2, abs=0.025)


def test_partial_tail_and_silent_clip_do_not_pad_or_invent_frames(api, sources, media_dir):
    clip = api.MediaExtractor().prepare(sources[False], 2.5, 30,
                                        media_dir / "tail", stamp=False)
    assert clip.frame_times == [0.0]
    assert len(clip.frame_paths) == 1
    assert clip.audio_path is None
    with Image.open(clip.frame_paths[0]) as im:
        assert im.size == (640, 360)


def test_fractional_sampling_grid_is_clip_local(api, sources, media_dir):
    clip = api.MediaExtractor().prepare(sources[False], 0.37, 1.3,
                                        media_dir / "fractional", fps=2.5, stamp=False)
    assert clip.frame_times == pytest.approx([0, 0.4, 0.8, 1.2])
    assert len(clip.frame_paths) == 4


def test_crop_uses_original_content_coordinates(api, sources, media_dir):
    output = media_dir / "crop" / "right.jpg"
    result = api.MediaExtractor().extract_crop(sources[False], 1.1,
                                               [0, 500, 1000, 1000], output, max_dim=180)
    assert result == output
    with Image.open(output) as im:
        assert im.size == (160, 180)
        r, g, b = im.convert("RGB").getpixel((80, 90))
        assert b > 200 and r < 30 and g < 30


@pytest.mark.parametrize("start,duration,fps,max_dim", [
    (-1, 1, 1, 320), (0, 0, 1, 320), (0, 30.1, 1, 320),
    (0, 1, 0, 320), (0, 1, float("nan"), 320),
    (float("inf"), 1, 1, 320), (0, 1, 1, 0),
    (0, 1, 30.01, 320),
])
def test_invalid_clip_bounds_rejected(api, sources, media_dir, start, duration, fps, max_dim):
    with pytest.raises(ValueError):
        api.MediaExtractor().prepare(sources[False], start, duration,
                                    media_dir / "invalid", fps=fps, max_dim=max_dim)


@pytest.mark.parametrize("box", [[0, -1, 1000, 1000], [0, 500, 1000, 500],
                                  [500, 0, 0, 1000], [0, 0, 1001, 1000],
                                  [0, 0, math.nan, 1000], [0, 0, 1000]])
def test_invalid_crop_bounds_rejected(api, sources, media_dir, box):
    with pytest.raises(ValueError):
        api.MediaExtractor().extract_crop(sources[False], 0, box, media_dir / "bad.jpg")


def test_out_of_range_seek_rejected(api, sources, media_dir):
    extractor = api.MediaExtractor()
    with pytest.raises(ValueError):
        extractor.prepare(sources[False], 5, 1, media_dir / "after")
    with pytest.raises(ValueError):
        extractor.extract_crop(sources[False], 5, [0, 0, 1000, 1000], media_dir / "after.jpg")


def test_ffmpeg_error_does_not_echo_input_contents(api, media_dir):
    source = media_dir / "credential_SECRET.mp4"
    source.write_bytes(b"not media SECRET")
    with pytest.raises(RuntimeError) as error:
        api.probe(source)
    assert "SECRET" not in str(error.value)


def test_non_aligned_seek_samples_first_source_frame_after_grid(api, media_dir):
    source = media_dir / "frame-levels.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi",
        "-i", "nullsrc=s=64x64:r=50:d=2,geq=r='2*N':g='2*N':b='2*N'",
        "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", "-threads", "1", str(source)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    clip = api.MediaExtractor().prepare(source, 0.37, 1.3,
                                        media_dir / "quantization", fps=2.5, stamp=False)
    levels = []
    for path in clip.frame_paths:
        with Image.open(path) as im:
            levels.append(im.convert("RGB").getpixel((32, 32))[0])
    # Target .37/.77/1.17/1.57 seconds -> source frame 19/39/59/79.
    assert levels == pytest.approx([38, 78, 118, 158], abs=3)


def test_prepare_and_crop_share_decode_semaphore(api, sources, media_dir, monkeypatch):
    real_run = subprocess.run
    lock = threading.Lock()
    active = 0
    peak = 0

    def measured_run(args, **kwargs):
        nonlocal active, peak
        decoding = args[0] == "ffmpeg"
        if decoding:
            with lock:
                active += 1
                peak = max(peak, active)
        try:
            return real_run(args, **kwargs)
        finally:
            if decoding:
                with lock:
                    active -= 1

    monkeypatch.setattr(subprocess, "run", measured_run)
    extractor = api.MediaExtractor()
    barrier = threading.Barrier(3)

    def task(index):
        barrier.wait()
        if index == 2:
            return extractor.extract_crop(sources[False], 0.1, [0, 0, 1000, 1000],
                                          media_dir / "concurrent-crop.jpg")
        return extractor.prepare(sources[True], 0, 3,
                                 media_dir / f"concurrent-{index}")

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(task, range(3)))
    assert peak == 1
    assert len(results[0].frame_paths) == len(results[1].frame_paths) == 3
    assert results[2].is_file()


def test_thirty_second_clip_does_not_decode_rest_of_longer_source(api, media_dir):
    source = media_dir / "longer.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi",
        "-i", "color=c=red:s=64x64:r=50:d=34.24", "-c:v", "libx264", "-threads", "1",
        str(source)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    extractor = api.MediaExtractor()
    clip = extractor.prepare(source, 2, 30, media_dir / "bounded-long", stamp=False)
    assert len(clip.frame_paths) == 30
    assert clip.frame_times == list(range(30))
    tail = extractor.prepare(source, 34, 30, media_dir / "subsecond-tail", stamp=False)
    assert tail.frame_times == [0]
    assert len(tail.frame_paths) == 1


@pytest.fixture(scope="module")
def delayed_audio_source(media_dir):
    source = media_dir / "delayed-audio.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi",
        "-i", "color=c=blue:s=64x64:r=50:d=5", "-itsoffset", "2", "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=48000:duration=3", "-c:v", "libx264", "-threads", "1",
        "-c:a", "aac", str(source)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return source


@pytest.mark.parametrize("start,duration,silent_until", [(0, 5, 1.8), (1, 3, 0.8), (0, 1, 1), (3, 2, 0)])
def test_delayed_audio_stays_on_video_timeline(api, media_dir, delayed_audio_source,
                                              start, duration, silent_until):
    clip = api.MediaExtractor().prepare(delayed_audio_source, start, duration,
                                        media_dir / f"delayed-{start}-{duration}")
    with wave.open(str(clip.audio_path), "rb") as wav:
        samples = array("h", wav.readframes(wav.getnframes()))
    assert len(samples) / 16000 == pytest.approx(duration, abs=0.001)
    assert max(map(abs, samples[:int(silent_until * 16000)]), default=0) < 10
    if duration > silent_until:
        assert max(map(abs, samples[-8000:])) > 1000


def test_failed_audio_removes_only_its_generated_partial_files(api, sources, media_dir, monkeypatch):
    real_run = api._run
    unrelated = []

    def fail_audio(args, **kwargs):
        if str(args[-1]).endswith("audio.wav"):
            path = Path(args[-1])
            path.write_bytes(b"partial wav")
            sentinel = path.parent / "unrelated-user-data.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            unrelated.append(sentinel)
            raise RuntimeError("Induced audio failure")
        return real_run(args, **kwargs)

    monkeypatch.setattr(api, "_run", fail_audio)
    output = media_dir / "partial-failure"
    with pytest.raises(RuntimeError, match="Induced audio failure"):
        api.MediaExtractor().prepare(sources[True], 0, 2, output)
    assert not list(output.rglob("frame_*.jpg"))
    assert not list(output.rglob("audio.wav"))
    assert unrelated[0].read_text(encoding="utf-8") == "preserve"


def test_audio_packet_gap_is_silence_without_shifting_later_sound(api, media_dir):
    source = media_dir / "gapped-audio.mkv"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi",
        "-i", "color=c=blue:s=64x64:r=50:d=4", "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=48000:duration=4", "-af", "aselect='not(between(t,1,2))'",
        "-c:v", "libx264", "-threads", "1", "-c:a", "pcm_s16le", str(source)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    clip = api.MediaExtractor().prepare(source, 0.5, 3, media_dir / "gapped-output")
    with wave.open(str(clip.audio_path), "rb") as wav:
        samples = array("h", wav.readframes(wav.getnframes()))
    assert len(samples) == 48000
    assert max(map(abs, samples[:4000])) > 1000
    assert max(map(abs, samples[11200:19200])) < 10
    assert max(map(abs, samples[28800:])) > 1000


def test_decode_slots_is_stored_and_threads_reach_the_ffmpeg_command(api):
    """The tuning knobs must actually reach FFmpeg, not just be accepted."""
    extractor = api.MediaExtractor(threads=3, decode_slots=2, footer_workers=5)
    command = extractor._command()
    assert command[command.index("-threads") + 1] == "3"
    assert command[command.index("-filter_threads") + 1] == "3"
    assert extractor.decode_slots == 2
    assert api.MediaExtractor().decode_slots == 1


def test_extractor_rejects_non_positive_tuning_values(api):
    for kwargs in ({"threads": 0}, {"threads": True}, {"decode_slots": -1},
                   {"footer_workers": 0}, {"footer_workers": 1.5}, {"threads": "2"}):
        with pytest.raises(ValueError):
            api.MediaExtractor(**kwargs)


def test_footer_output_is_identical_across_worker_counts(api, sources, media_dir):
    """The parallel footer must reproduce the serial footer byte for byte."""
    results = {}
    for workers in (1, 3, 8):
        clip = api.MediaExtractor().prepare(sources[False], 0.2, 2.2, media_dir / f"stamp-src-{workers}",
                                            max_dim=320, stamp=False)
        api.MediaExtractor(footer_workers=workers).stamp_frames(clip.frame_paths, clip.frame_times, 24)
        results[workers] = {path.name: path.read_bytes() for path in clip.frame_paths}
    assert results[1] == results[3] == results[8]
    assert len(results[1]) == 3


def test_prepare_with_configurable_threads_and_footer_still_samples_identically(api, sources, media_dir):
    """Tuning threads changes speed only; frames, times and audio stay the same."""
    serial = api.MediaExtractor(threads=1, decode_slots=1, footer_workers=1).prepare(
        sources[True], 0.2, 2.2, media_dir / "tuned-serial", max_dim=320)
    tuned = api.MediaExtractor(threads=2, decode_slots=2, footer_workers=4).prepare(
        sources[True], 0.2, 2.2, media_dir / "tuned-parallel", max_dim=320)
    assert tuned.frame_times == serial.frame_times == [0.0, 1.0, 2.0]
    assert [p.read_bytes() for p in tuned.frame_paths] == [p.read_bytes() for p in serial.frame_paths]
    assert tuned.audio_path.read_bytes() == serial.audio_path.read_bytes()


def test_two_extractors_share_the_source_file_without_interference(api, sources, media_dir):
    """Concurrent clips of one source is the production shape; results must not blend."""
    extractor = api.MediaExtractor(threads=2, decode_slots=2, footer_workers=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(
            lambda spec: extractor.prepare(sources[False], spec[0], spec[1],
                                           media_dir / f"shared-{spec[0]}", max_dim=320),
            [(0.0, 1.0), (1.0, 1.5)]))
    assert first.frame_times == [0.0]
    assert second.frame_times == [0.0, 1.0]
    assert first.frame_paths[0] != second.frame_paths[0]
    # Every frame is a whole, decodable JPEG: two concurrent decoders of one source
    # must never interleave or truncate each other's output.
    for path in [*first.frame_paths, *second.frame_paths]:
        with Image.open(path) as image:
            image.load()
            assert image.format == "JPEG"
