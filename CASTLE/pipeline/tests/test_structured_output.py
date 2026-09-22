"""The three production stages constrain output before local validation."""
import json

import pytest

from castle_pipeline.request_spec import response_schema_for_stage
from castle_pipeline.batch_tasks import request_row
from castle_pipeline.vertex import VertexProvider


@pytest.mark.parametrize("stage,required", [
    ("audio", {"summary", "utterances", "sound_events", "uncertainties"}),
    ("annotation", {"schema_version", "scene_summary", "actors", "segments", "activity_chain"}),
    ("review", {"findings", "segment_replacements", "initial_environment_replacement",
                "scene_summary_replacement", "resegmentation_requests"}),
])
def test_batch_requests_include_stage_response_schema(stage, required):
    row = request_row(f"{stage}:clip", "prompt", {"task_phase": stage}, [])
    config = row["request"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert required <= set(config["responseSchema"]["required"])
    assert config["responseSchema"] == response_schema_for_stage(stage)


@pytest.mark.parametrize("tier", ["standard", "flex"])
@pytest.mark.parametrize("stage", ["audio", "annotation", "review"])
def test_online_requests_include_the_same_stage_schema(tier, stage):
    class Boundary:
        def __init__(self):
            self.requests = []
            self.models = self
        def __call__(self, **_):
            return self
        def generate_content(self, **request):
            self.requests.append(request)
            candidate = type("Candidate", (), {"finish_reason": "STOP"})()
            return type("Response", (), {"text": "{}", "candidates": [candidate],
                                         "usage_metadata": None})()

    boundary = Boundary()
    provider = VertexProvider("gemini-test", "project-test", service_tier=tier, client_factory=boundary)
    provider.generate("prompt", json.dumps({"task_phase": stage}), [])
    config = boundary.requests[0]["config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_schema"] == response_schema_for_stage(stage)


def test_annotation_schema_includes_optional_review_regions():
    schema = response_schema_for_stage("annotation")
    assert "review_regions" in schema["properties"]
    assert schema["properties"]["review_regions"]["type"] == "ARRAY"
    assert schema["properties"]["segments"]["type"] == "ARRAY"


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="stage"):
        response_schema_for_stage("unknown")
