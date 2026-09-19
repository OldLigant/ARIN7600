"""Offline storage/service boundary tests for the real Batch adapter."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from castle_pipeline import batch_cloud as module
from castle_pipeline.batch_cloud import BatchCloud, CloudError, StateConflict


class RemoteError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__("secret-provider-payload")


class Blob:
    def __init__(self, store, name):
        self.store, self.name = store, name
        self.metadata = None
        self.generation = None

    def reload(self):
        if self.name not in self.store.objects:
            raise RemoteError(404)
        data, metadata, generation = self.store.objects[self.name]
        self.metadata, self.generation = metadata, generation

    def upload_from_file(self, stream, *, if_generation_match, **kwargs):
        previous = self.store.objects.get(self.name)
        generation = previous[2] if previous else 0
        if generation != if_generation_match:
            raise RemoteError(412)
        self.generation = generation + 1
        self.store.objects[self.name] = (stream.read(), self.metadata, self.generation)

    def upload_from_string(self, data, **kwargs):
        self.upload_from_file(io.BytesIO(data.encode() if isinstance(data, str) else data), **kwargs)

    def download_as_bytes(self, *, if_generation_match=None, **kwargs):
        self.reload()
        if if_generation_match is not None and if_generation_match != self.generation:
            raise RemoteError(412)
        return self.store.objects[self.name][0]

    def download_to_filename(self, filename):
        Path(filename).write_bytes(self.download_as_bytes())

    def open(self, mode, **kwargs):
        assert mode == "rt"
        return io.StringIO(self.download_as_bytes().decode())


class Storage:
    def __init__(self):
        self.objects = {}

    def bucket(self, name):
        assert name == "bucket"
        return self

    def blob(self, name):
        return Blob(self, name)

    def list_blobs(self, bucket, prefix):
        assert bucket == "bucket"
        return (self.blob(name) for name in sorted(self.objects) if name.startswith(prefix))


class Jobs:
    def __init__(self):
        self.batches = self
        self.calls = []
        self.jobs = []
        self.error = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return NS(name="projects/p/locations/global/batchPredictionJobs/1", state=NS(value="JOB_STATE_PENDING"), dest=None, output_info=None, error=None)

    def get(self, *, name):
        return self.jobs[0]

    def list(self, *, config):
        self.calls.append(config)
        return iter(self.jobs)


@pytest.fixture
def cloud():
    return BatchCloud("project", storage_client=Storage(), genai_client=Jobs())


def put(cloud, name, data):
    cloud.storage.objects[name] = (data.encode(), {}, 1)


def test_immutable_upload_reuses_identical_bytes_and_rejects_replacement(cloud, tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"original")
    cloud.upload(source, "gs://bucket/media/video.mp4")
    cloud.upload(source, "gs://bucket/media/video.mp4")
    data, metadata, generation = cloud.storage.objects["media/video.mp4"]
    assert data == b"original" and generation == 1
    assert metadata["sha256"] == hashlib.sha256(b"original").hexdigest()
    source.write_bytes(b"replacement")
    with pytest.raises(StateConflict):
        cloud.upload(source, "gs://bucket/media/video.mp4")
    assert cloud.storage.objects["media/video.mp4"][0] == b"original"


def test_download_creates_parent_and_copies_bytes(cloud, tmp_path):
    put(cloud, "video.mp4", "video")
    target = tmp_path / "nested" / "video.mp4"
    cloud.download("gs://bucket/video.mp4", target)
    assert target.read_bytes() == b"video"


def test_jsonl_accepts_one_pass_generator_and_preserves_unicode(cloud):
    cloud.write_jsonl("gs://bucket/input.jsonl", ({"request": {"text": text}} for text in ["你好", "two"]))
    assert [json.loads(line) for line in cloud.storage.objects["input.jsonl"][0].splitlines()] == [
        {"request": {"text": "你好"}}, {"request": {"text": "two"}}]


@pytest.mark.parametrize("limit,rows", [("MAX_JSONL_ROWS", [{}, {}, {}]), ("MAX_JSONL_BYTES", [{"text": "你好"}])])
def test_jsonl_rejects_limits_before_remote_upload(cloud, monkeypatch, limit, rows):
    monkeypatch.setattr(module, limit, 2)
    with pytest.raises(ValueError):
        cloud.write_jsonl("gs://bucket/input.jsonl", iter(rows))
    assert cloud.storage.objects == {}


def test_output_stream_reads_all_jsonl_shards_and_excludes_neighbor_prefix(cloud):
    put(cloud, "output/job/a.jsonl", '{"id":2}\n\n')
    put(cloud, "output/job/b.jsonl", '{"id":1}\n')
    put(cloud, "output/job/readme.txt", "ignore")
    put(cloud, "output/job-other/a.jsonl", '{"id":3}\n')
    assert list(cloud.iter_jsonl("gs://bucket/output/job")) == [{"id": 2}, {"id": 1}]


def test_state_compare_and_swap_prevents_lost_update(cloud):
    assert cloud.read_state("gs://bucket/state.json") == (None, 0)
    assert cloud.write_state("gs://bucket/state.json", {"phase": "prepared"}, 0) == 1
    assert cloud.read_state("gs://bucket/state.json") == ({"phase": "prepared"}, 1)
    assert cloud.write_state("gs://bucket/state.json", {"phase": "submitted"}, 1) == 2
    with pytest.raises(StateConflict):
        cloud.write_state("gs://bucket/state.json", {"phase": "stale"}, 1)
    assert cloud.read_state("gs://bucket/state.json") == ({"phase": "submitted"}, 2)


def test_batch_creation_disables_sdk_retries_and_uses_gcs(cloud):
    job = cloud.create_batch("gemini-3.8-flash", "gs://bucket/input.jsonl", "gs://bucket/output/", "castle-run")
    assert job == {"name": "projects/p/locations/global/batchPredictionJobs/1", "state": "JOB_STATE_PENDING", "output_uri": "gs://bucket/output/"}
    request = cloud.genai.batches.calls[0]
    assert request["src"] == "gs://bucket/input.jsonl"
    assert request["config"]["dest"] == "gs://bucket/output/"
    assert request["config"]["http_options"]["retry_options"]["attempts"] == 1


def test_uncertain_create_is_called_once_and_error_has_no_provider_payload(cloud):
    cloud.genai.error = RemoteError(503)
    with pytest.raises(CloudError) as error:
        cloud.create_batch("model", "gs://bucket/input.jsonl", "gs://bucket/out", "run")
    assert "secret-provider-payload" not in str(error.value)
    assert len(cloud.genai.calls) == 1


def test_get_prefers_actual_output_directory_and_sanitizes_error(cloud):
    cloud.genai.jobs = [NS(name="job", state=NS(value="JOB_STATE_FAILED"), dest=NS(gcs_uri="gs://bucket/requested"),
        output_info=NS(gcs_output_directory="gs://bucket/actual/prediction-123"), error=NS(code=7, message="secret-provider-payload"))]
    job = cloud.get_batch("job")
    assert job["output_uri"] == "gs://bucket/actual/prediction-123"
    assert job["error"]["code"] == 7
    assert "secret-provider-payload" not in json.dumps(job)


def test_find_jobs_filters_exact_display_name_locally_and_remotely(cloud):
    cloud.genai.jobs = [NS(name="match", display_name='run"name', state="JOB_STATE_RUNNING", dest=NS(gcs_uri="gs://bucket/out")),
                        NS(name="wrong", display_name="different", state="JOB_STATE_RUNNING")]
    assert [job["name"] for job in cloud.find_batches('run"name')] == ["match"]
    assert cloud.genai.calls[0]["filter"] == 'display_name="run\\"name"'


@pytest.mark.parametrize("uri", ["https://example.com/x", "gs://bucket", "gs://bucket/x?secret=1", "gs://bucket/x#fragment"])
def test_invalid_gcs_uri_rejected_before_any_cloud_access(cloud, uri):
    with pytest.raises(ValueError):
        cloud.read_state(uri)
    assert cloud.storage.objects == {}


def test_explicit_adc_ignores_ambient_api_key(monkeypatch):
    from google.auth.credentials import AnonymousCredentials
    credentials = AnonymousCredentials()
    monkeypatch.setenv("GOOGLE_API_KEY", "ambient-secret")
    cloud = BatchCloud("project", credentials=credentials, storage_client=Storage())
    try:
        assert cloud.genai._api_client.api_key is None
        assert cloud.genai._api_client._credentials is credentials
        assert cloud.genai._api_client._http_options.retry_options.attempts == 1
    finally:
        cloud.genai.close()


def test_default_credentials_explicitly_request_cloud_scope(monkeypatch):
    import google.auth
    from google.auth.credentials import AnonymousCredentials
    seen = []
    def default(*, scopes):
        seen.append(scopes)
        return AnonymousCredentials(), "ignored-project"
    monkeypatch.setattr(google.auth, "default", default)
    cloud = BatchCloud("project", storage_client=Storage())
    cloud.genai.close()
    assert seen == [["https://www.googleapis.com/auth/cloud-platform"]]


@pytest.mark.parametrize("status", [200, 503])
def test_real_sdk_serialization_and_no_create_retries_with_offline_transport(status):
    """Keep SDK serialization/retries real and replace only HTTP transport."""
    import httpx
    from google import genai
    from google.oauth2.credentials import Credentials
    requests = []
    def respond(request):
        requests.append(request)
        if status == 503:
            return httpx.Response(503, json={"error": {"code": 503, "message": "provider-secret", "status": "UNAVAILABLE"}})
        return httpx.Response(200, json={
            "name": "projects/project/locations/global/batchPredictionJobs/123",
            "state": "JOB_STATE_SUCCEEDED",
            "outputConfig": {"predictionsFormat": "jsonl", "gcsDestination": {"outputUriPrefix": "gs://bucket/output"}},
            "outputInfo": {"gcsOutputDirectory": "gs://bucket/output/prediction-123"}})
    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as http:
        client = genai.Client(vertexai=True, project="project", location="global",
            credentials=Credentials(token="offline-test-token"),
            http_options={"httpx_client": http, "retry_options": {"attempts": 5}})
        cloud = BatchCloud("project", storage_client=Storage(), genai_client=client)
        try:
            if status == 503:
                with pytest.raises(CloudError):
                    cloud.create_batch("gemini-3.8-flash", "gs://bucket/input.jsonl", "gs://bucket/output", "test")
            else:
                job = cloud.create_batch("gemini-3.8-flash", "gs://bucket/input.jsonl", "gs://bucket/output", "test")
                assert job["output_uri"] == "gs://bucket/output/prediction-123"
                assert job["state"] == "JOB_STATE_SUCCEEDED"
            assert len(requests) == 1
            body = json.loads(requests[0].content)
            assert body["inputConfig"] == {"instancesFormat": "jsonl", "gcsSource": {"uris": ["gs://bucket/input.jsonl"]}}
            assert body["outputConfig"] == {"predictionsFormat": "jsonl", "gcsDestination": {"outputUriPrefix": "gs://bucket/output"}}
            assert "x-goog-api-key" not in requests[0].headers
        finally:
            client.close()
