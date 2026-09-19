"""ADC-only Vertex Batch transport and generation-guarded GCS persistence.

Client injection keeps local tests entirely offline. Creating a job is never
retried: an ambiguous response must be reconciled by the durable engine.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
import tempfile
from typing import Iterable, Iterator
from urllib.parse import urlsplit


MAX_JSONL_BYTES = 1_000_000_000
MAX_JSONL_ROWS = 200_000


class CloudError(RuntimeError):
    """A cloud operation failed; the public message excludes provider payloads."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class StateConflict(CloudError):
    """A generation changed or an immutable object has different content."""


def parse_gs_uri(uri: str) -> tuple[str, str]:
    parts = urlsplit(uri)
    if (parts.scheme != "gs" or not parts.netloc or not parts.path.lstrip("/")
            or parts.query or parts.fragment or "@" in parts.netloc or ":" in parts.netloc):
        raise ValueError("Expected a gs://bucket/object URI without query or fragment")
    return parts.netloc, parts.path.lstrip("/")


def _field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _code(exc):
    value = getattr(exc, "code", None)
    return value if isinstance(value, int) else None


def _failure(operation, exc):
    code = _code(exc)
    suffix = f" (HTTP {code})" if code is not None else ""
    return CloudError(f"{operation} failed{suffix}; provider details omitted", code=code)


class BatchCloud:
    def __init__(self, project, location="global", credentials=None,
                 storage_client=None, genai_client=None):
        if not project or not location:
            raise ValueError("Batch requires an explicit Google Cloud project and location")
        self.project, self.location = project, location
        if credentials is None and (storage_client is None or genai_client is None):
            import google.auth
            try:
                credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"])
            except Exception:
                raise CloudError("Batch requires usable Google Application Default Credentials") from None
        if storage_client is None:
            from google.cloud import storage
            storage_client = storage.Client(project=project, credentials=credentials)
        if genai_client is None:
            from google import genai
            # Explicit credentials override GOOGLE_API_KEY in google-genai.
            genai_client = genai.Client(vertexai=True, project=project, location=location,
                credentials=credentials,
                http_options={"retry_options": {"attempts": 1}})
        self.storage, self.genai = storage_client, genai_client

    def _blob(self, uri):
        bucket, name = parse_gs_uri(uri)
        return self.storage.bucket(bucket).blob(name)

    def _upload_stream(self, stream, uri, content_type):
        blob = self._blob(uri)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        blob.metadata = {"sha256": digest}
        try:
            blob.upload_from_file(stream, content_type=content_type, if_generation_match=0)
        except Exception as exc:
            if _code(exc) != 412:
                raise _failure("GCS upload", exc) from None
            try:
                blob.reload()
            except Exception as reload_error:
                raise _failure("GCS immutable upload verification", reload_error) from None
            if (blob.metadata or {}).get("sha256") != digest:
                raise StateConflict("Immutable GCS object already exists with different content") from None
        return uri

    def upload(self, localpath, uri):
        parse_gs_uri(uri)
        with Path(localpath).open("rb") as stream:
            return self._upload_stream(stream, uri,
                mimetypes.guess_type(str(localpath))[0] or "application/octet-stream")

    def download(self, uri, path):
        blob = self._blob(uri)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            blob.download_to_filename(str(path))
        except Exception as exc:
            raise _failure("GCS download", exc) from None
        return path

    def write_jsonl(self, uri, rows: Iterable[dict]):
        """Spool once to disk; enforce Vertex's 1 GB / 200,000-row limits."""
        parse_gs_uri(uri)
        with tempfile.TemporaryFile(mode="w+b", prefix="castle-batch-jsonl-") as stream:
            size = 0
            for count, row in enumerate(rows, 1):
                if count > MAX_JSONL_ROWS:
                    raise ValueError("Batch JSONL exceeds the 200,000-row limit")
                if not isinstance(row, dict):
                    raise ValueError("Each Batch JSONL row must be an object")
                line = (json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
                size += len(line)
                if size > MAX_JSONL_BYTES:
                    raise ValueError("Batch JSONL exceeds the 1 GB limit")
                stream.write(line)
            stream.seek(0)
            return self._upload_stream(stream, uri, "application/jsonl")

    def iter_jsonl(self, prefix) -> Iterator[dict]:
        """Stream every output shard; callers correlate by request ID, not order."""
        bucket, name = parse_gs_uri(prefix)
        name = name.rstrip("/") + "/"
        try:
            for blob in self.storage.list_blobs(bucket, prefix=name):
                if not blob.name.endswith(".jsonl"):
                    continue
                with blob.open("rt", encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            row = json.loads(line)
                            if not isinstance(row, dict):
                                raise ValueError("Batch output row must be an object")
                            yield row
        except Exception as exc:
            raise _failure("GCS Batch output read", exc) from None

    def read_state(self, uri):
        blob = self._blob(uri)
        try:
            blob.reload()
        except Exception as exc:
            if _code(exc) == 404:
                return None, 0
            raise _failure("GCS state lookup", exc) from None
        generation = int(blob.generation)
        try:
            state = json.loads(blob.download_as_bytes(if_generation_match=generation))
            if not isinstance(state, dict):
                raise ValueError("State must be a JSON object")
        except Exception as exc:
            if _code(exc) == 412:
                raise StateConflict("State changed during read; reload before proceeding") from None
            raise _failure("GCS state read", exc) from None
        return state, generation

    def write_state(self, uri, state, expected_generation):
        if not isinstance(state, dict):
            raise ValueError("State must be a JSON object")
        if not isinstance(expected_generation, int) or expected_generation < 0:
            raise ValueError("Expected generation must be a nonnegative integer")
        blob = self._blob(uri)
        payload = json.dumps(state, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        try:
            blob.upload_from_string(payload, content_type="application/json",
                                    if_generation_match=expected_generation)
        except Exception as exc:
            if _code(exc) == 412:
                raise StateConflict("State changed; reload before proceeding") from None
            raise _failure("GCS state write", exc) from None
        return int(blob.generation)

    @staticmethod
    def _job(job, fallback_output=None):
        state = _field(job, "state")
        output = _field(_field(job, "output_info"), "gcs_output_directory")
        dest = _field(job, "dest")
        output = output or (dest if isinstance(dest, str) else _field(dest, "gcs_uri")) or fallback_output
        result = {"name": _field(job, "name"), "state": _field(state, "value", state), "output_uri": output}
        if not result["name"]:
            raise CloudError("Batch response is missing the job name; reconcile submission before retrying")
        error = _field(job, "error")
        if error:
            code = _field(error, "code")
            result["error"] = {"code": code if isinstance(code, int) else None,
                               "message": "Batch provider reported an error; inspect the job in Google Cloud"}
        return result

    def create_batch(self, model, input_uri, output_uri, display_name):
        parse_gs_uri(input_uri)
        parse_gs_uri(output_uri)
        try:
            job = self.genai.batches.create(model=model, src=input_uri, config={
                "dest": output_uri, "display_name": display_name,
                "http_options": {"retry_options": {"attempts": 1}}})
        except Exception as exc:
            raise _failure("Batch create (submission outcome may be unknown)", exc) from None
        return self._job(job, output_uri)

    def get_batch(self, name):
        try:
            job = self.genai.batches.get(name=name)
        except Exception as exc:
            raise _failure("Batch status lookup", exc) from None
        return self._job(job)

    def find_batches(self, display_name):
        try:
            jobs = self.genai.batches.list(config={"filter": f"display_name={json.dumps(display_name)}"})
            return [self._job(job) for job in jobs if _field(job, "display_name") == display_name]
        except CloudError:
            raise
        except Exception as exc:
            raise _failure("Batch reconciliation lookup", exc) from None
