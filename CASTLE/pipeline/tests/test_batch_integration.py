"""Real Batch planner/engine/crops; only storage and job service are faked."""
from copy import deepcopy
import io
import json
from pathlib import Path

from PIL import Image
import pytest

from castle_pipeline.batch_engine import BatchEngine
from castle_pipeline.batch_tasks import TaskPlanner, response_id
from test_batch_engine import MemoryCloud
from test_batch_tasks import CloudFiles


class OfflineCloud(MemoryCloud, CloudFiles):
    def __init__(self):
        MemoryCloud.__init__(self)
        CloudFiles.__init__(self)
        self.outputs = {}

    def upload(self, path, uri):
        content = Path(path).read_bytes()
        if uri in self.files:
            assert self.files[uri] == content, "Planner attempted to replace immutable content"
        self.files[uri] = content

    def iter_jsonl(self, prefix):
        yield from deepcopy(self.outputs[prefix])

    def requests(self, job_number):
        uri = self.creates[job_number - 1][1]
        return [json.loads(line) for line in self.files[uri].splitlines()]

    def finish(self, job_number, rows, partial=False):
        job = self.jobs[f"jobs/{job_number}"]
        job["state"] = "JOB_STATE_PARTIALLY_SUCCEEDED" if partial else "JOB_STATE_SUCCEEDED"
        # Real jobs add an output subdirectory; consumers must use returned URI.
        job["output_uri"] += "prediction-123/"
        self.outputs[job["output_uri"]] = list(reversed(rows))


def context(request):
    return json.loads(request["request"]["contents"][0]["parts"][0]["text"].split("\n", 1)[1])


def reply(request, data=None, error=False):
    return {"request": deepcopy(request["request"]),
            "status": {"code": 3, "message": "synthetic row failure"} if error else "",
            "response": {} if error else {
                "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(data)}]}}],
                "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18}}}


@pytest.fixture
def setup_run(tmp_path):
    cloud = OfflineCloud()
    prefix = "gs://offline-bucket/batch-integration"
    rows = []
    for index, cid in enumerate(("c1", "c2", "c3")):
        native = tmp_path / f"{cid}-native.jpg"
        preview = tmp_path / f"{cid}-preview.jpg"
        Image.new("RGB", (320, 180), (40 * index, 100, 200)).save(native)
        Image.new("RGB", (32, 18), (40 * index, 100, 200)).save(preview)
        base = f"{prefix}/media/{cid}"
        cloud.upload(native, f"{base}/native.jpg")
        cloud.upload(preview, f"{base}/preview.jpg")
        cloud.files[f"{base}/audio.wav"] = b"offline-audio-placeholder"
        meta = {"clip_id": cid, "duration_sec": 20, "start_offset_sec": index * 20,
                "source": {"source_id": "fixture-source", "viewpoint": "egocentric"},
                "frame_times_sec": [0], "frames": [f"{base}/preview.jpg"],
                "native_frames": [f"{base}/native.jpg"], "audio_uri": f"{base}/audio.wav"}
        cloud.files[f"{base}/metadata.json"] = json.dumps(meta).encode()
        rows.append({"clip_id": cid, "metadata_uri": f"{base}/metadata.json"})
    config = {"model": "offline-model", "gcs_prefix": prefix, "rows": rows, "review": True,
              "max_review_regions": 2, "max_dim": 1440,
              "prompts": {"audio": "Audio contract", "annotation": "Annotation contract",
                          "review_regions": "Up to MAX_REVIEW_REGIONS regions", "review": "Review contract"}}
    planner = TaskPlanner(cloud, tmp_path / "scratch", tmp_path / "out")
    engine = BatchEngine(cloud, f"{prefix}/state.json", planner)
    engine.initialize(config)
    return cloud, engine, config


def test_invalid_crop_entry_does_not_block_other_completed_clips(setup_run):
    cloud,engine,config=setup_run
    engine.start()
    audio={'summary':'quiet','utterances':[],'sound_events':[],'uncertainties':[]}
    cloud.finish(1,[reply(request,audio) for request in cloud.requests(1)])
    engine.tick()
    fixture_path=Path(__file__).resolve().parents[2]/'annotation_design_v1'/'输出示例_假设场景.json'
    fixture=json.loads(fixture_path.read_text(encoding='utf-8'))
    responses=[]
    for request in cloud.requests(2):
        data=deepcopy(fixture)
        data['review_regions']=[None] if context(request)['clip_id']=='c2' else []
        responses.append(reply(request,data))
    cloud.finish(2,responses)
    state=engine.tick()
    assert state['status']=='complete_with_errors'
    assert {r['clip_id'] for r in state['completed']}=={'c1','c3'}
    assert [(r['clip_id'],r['code']) for r in state['failures']]==[('c2','VALIDATION')]
    assert len(cloud.creates)==2


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("interrupt_final_save", [False, True])
def test_three_stages_survive_shuffled_partial_outputs_and_restart(setup_run, tmp_path, partial, interrupt_final_save):
    cloud, engine, config = setup_run
    prefix = config["gcs_prefix"]
    assert engine.start()["current_stage"] == "audio"
    for _ in range(2):
        assert engine.tick()["status"] == "running"
    assert len(cloud.creates) == 1

    audio_requests = cloud.requests(1)
    assert [response_id(row) for row in audio_requests] == ["audio:c1", "audio:c2", "audio:c3"]
    audio_rows = []
    for request in audio_requests:
        cid = context(request)["clip_id"]
        if partial and cid == "c3":
            continue
        audio_rows.append(reply(request, {"summary": f"Audio for {cid}", "utterances": [],
                                          "sound_events": [], "uncertainties": []}))
    cloud.finish(1, audio_rows, partial=partial)
    state = engine.tick()
    assert state["current_stage"] == "annotation"
    assert len(cloud.creates) == 2

    fixture_path = Path(__file__).resolve().parents[2] / "annotation_design_v1" / "输出示例_假设场景.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    annotation_requests = cloud.requests(2)
    expected_ids = ["c1", "c2"] if partial else ["c1", "c2", "c3"]
    assert [context(row)["clip_id"] for row in annotation_requests] == expected_ids
    annotation_rows = []
    for request in annotation_requests:
        ctx = context(request)
        cid = ctx["clip_id"]
        assert ctx["audio_annotation"]["summary"] == f"Audio for {cid}"
        assert ctx["audio_input"] == "absent_in_this_request"
        parts = request["request"]["contents"][0]["parts"]
        assert all(part["fileData"]["mimeType"] == "image/jpeg" for part in parts if "fileData" in part)
        annotation = deepcopy(fixture)
        annotation["scene_summary"] = f"First pass for {cid}"
        annotation["review_regions"] = [{"frame_index": 0, "box_2d": [0, 0, 500, 500],
                                          "label": "detail", "reason": "Inspect native pixels"}]
        annotation_rows.append(reply(request, annotation, error=partial and cid == "c2"))
    cloud.finish(2, annotation_rows, partial=partial)
    # Recreate both real components as a fresh hourly worker would.
    engine = BatchEngine(cloud, f"{prefix}/state.json",
                         TaskPlanner(cloud, tmp_path / "scratch", tmp_path / "out"))
    state = engine.tick()
    assert state["current_stage"] == "review"
    assert len(cloud.creates) == 3
    review_requests = cloud.requests(3)
    survivors = ["c1"] if partial else ["c1", "c2", "c3"]
    assert [context(row)["clip_id"] for row in review_requests] == survivors
    review_rows = []
    for request in review_requests:
        ctx = context(request)
        cid = ctx["clip_id"]
        assert ctx["annotation"]["scene_summary"] == f"First pass for {cid}"
        parts = request["request"]["contents"][0]["parts"]
        uris = [part["fileData"]["fileUri"] for part in parts if "fileData" in part]
        assert uris == [f"{prefix}/media/{cid}/crop-0.jpg"]
        with Image.open(io.BytesIO(cloud.files[uris[0]])) as crop:
            assert crop.size == (160, 90), "Crop must use native pixels, not the 32x18 preview"
        review = {"findings": [{"source_id": "crop_0", "time_sec": 0, "observation": "Detail inspected",
                                "readable_text": [], "limitations": []}],
                  "segment_replacements": [], "initial_environment_replacement": None,
                  "scene_summary_replacement": f"Reviewed {cid}", "resegmentation_requests": []}
        review_rows.append(reply(request, review))
    cloud.finish(3, review_rows)
    if interrupt_final_save:
        def fail_once():
            raise OSError("offline state-save interruption")
        cloud.before_write = fail_once
        with pytest.raises(OSError, match="state-save interruption"):
            engine.tick()
        assert cloud.state["status"] == "running"
        assert cloud.state["completed"] == []
    state = engine.tick()
    assert state["status"] == ("complete_with_errors" if partial else "complete")
    assert [row["clip_id"] for row in state["completed"]] == survivors
    assert [(row["clip_id"], row["stage"], row["code"]) for row in state["failures"]] == (
        [("c3", "audio", "MISSING_RESPONSE"), ("c2", "annotation", "ROW_ERROR")] if partial else [])
    assert len(cloud.creates) == 3
    for _ in range(3):
        assert engine.tick() == state
        assert engine.start() == state
    assert len(cloud.creates) == 3
    assert sorted(path.stem for path in (tmp_path / "out" / "final").glob("*.json")) == survivors
    for cid in survivors:
        saved = json.loads((tmp_path / "out" / "final" / f"{cid}.json").read_text(encoding="utf-8"))
        assert saved == json.loads(cloud.files[f"{prefix}/final/{cid}.json"])
        assert saved["annotation"]["scene_summary"] == f"Reviewed {cid}"
        assert saved["annotation"]["segments"] == fixture["segments"]
        assert saved["ok"] and not saved["review_required"]
        assert saved["clip"]["id"] == cid
        assert saved["usage"]["review"]["input_tokens"] == 11
        assert f"{prefix}/raw/review/{cid}.json" in cloud.files
    assert not list((tmp_path / "scratch").rglob("*.jpg"))
    assert not list((tmp_path / "scratch").rglob("*.json"))
