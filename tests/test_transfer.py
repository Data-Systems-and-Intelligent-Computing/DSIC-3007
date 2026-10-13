import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import transfer
from src.artifacts import read_json, seal


class FakeBody(io.BytesIO):
    pass


class FakeR2:
    def __init__(self):
        self.objects = {}
        self.upload_order = []

    def upload_file(self, filename, bucket, key):
        self.objects[(bucket, key)] = Path(filename).read_bytes()
        self.upload_order.append(key)

    def put_object(self, *, Bucket, Key, Body, ContentType):
        self.objects[(Bucket, Key)] = Body
        self.upload_order.append(Key)

    def get_object(self, *, Bucket, Key):
        return {"Body": FakeBody(self.objects[(Bucket, Key)])}


def _make_bundle(tmp_path, kind="prepared-inputs", file_name="documents.jsonl", data=b'{"document_id":"d1"}\n'):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / file_name).write_bytes(data)
    seal(bundle, kind)
    return bundle


def _configure(monkeypatch, tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    monkeypatch.setattr(transfer, "_workspace_root", lambda: workspace)
    monkeypatch.setenv("R2_ACCOUNT_ID", "account-test")
    monkeypatch.setenv("R2_BUCKET", "private-test-bucket")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "test-access")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "test-secret")
    client = FakeR2()

    class Session:
        @staticmethod
        def client(*args, **kwargs):
            assert kwargs["endpoint_url"] == "https://account-test.r2.cloudflarestorage.com"
            assert kwargs["aws_access_key_id"] == "test-access"
            assert kwargs["aws_secret_access_key"] == "test-secret"
            return client

    class FakeBoto:
        session = SimpleNamespace(Session=Session)

    monkeypatch.setitem(__import__("sys").modules, "boto3", FakeBoto)
    return workspace, client


def test_push_and_pull_roundtrip_checks_hashes_and_writes_manifest_last(tmp_path, monkeypatch):
    workspace, client = _configure(monkeypatch, tmp_path)
    source = _make_bundle(tmp_path)

    bundle_id = transfer.push(source, "prepared-inputs")
    prefix = f"prepared-inputs/{bundle_id}"
    assert client.upload_order[-1] == f"{prefix}/manifest.json"
    assert hashlib.sha256(client.objects[("private-test-bucket", f"{prefix}/manifest.json")]).hexdigest() == bundle_id

    destination = workspace / "data" / "prepared" / "bundle-1"
    restored = transfer.pull(bundle_id, "prepared-inputs", destination)
    assert (restored / "documents.jsonl").read_bytes() == (source / "documents.jsonl").read_bytes()
    assert (restored / "manifest.json").read_bytes() == (source / "manifest.json").read_bytes()


def test_pull_rejects_tampered_payload_and_removes_partial_directory(tmp_path, monkeypatch):
    workspace, client = _configure(monkeypatch, tmp_path)
    source = _make_bundle(tmp_path)
    bundle_id = transfer.push(source, "prepared-inputs")
    client.objects[("private-test-bucket", f"prepared-inputs/{bundle_id}/documents.jsonl")] = b"tampered"

    destination = workspace / "data" / "bad-download"
    with pytest.raises(ValueError, match="integrity mismatch"):
        transfer.pull(bundle_id, "prepared-inputs", destination)
    assert not destination.exists()
    assert not list(destination.parent.glob(".bad-download.download-*"))


def test_push_rejects_project_root_and_unlisted_files(tmp_path, monkeypatch):
    workspace, _client = _configure(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="project root"):
        transfer.push(workspace, "prepared-inputs")

    bundle = _make_bundle(tmp_path)
    (bundle / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="unlisted files"):
        transfer.push(bundle, "prepared-inputs")


def test_push_rejects_secret_filename_even_when_bundle_is_sealed(tmp_path, monkeypatch):
    _workspace, _client = _configure(monkeypatch, tmp_path)
    bundle = _make_bundle(tmp_path, file_name=".env")
    with pytest.raises(ValueError, match="unsafe or sensitive"):
        transfer.push(bundle, "prepared-inputs")


def test_push_rejects_manifest_kind_mismatch(tmp_path, monkeypatch):
    _workspace, _client = _configure(monkeypatch, tmp_path)
    bundle = _make_bundle(tmp_path, kind="prepared-inputs")
    with pytest.raises(ValueError, match="wrong kind"):
        transfer.push(bundle, "measurements")


def test_pull_rejects_path_traversal_bundle_id(tmp_path, monkeypatch):
    _workspace, _client = _configure(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        transfer.pull("../manifest", "prepared-inputs", tmp_path / "destination")


def test_inventory_schema_includes_duplicate_and_cross_split_audits(tmp_path, monkeypatch):
    _workspace, _client = _configure(monkeypatch, tmp_path)
    bundle = tmp_path / "inventory"
    bundle.mkdir()
    for name in ("documents.jsonl", "exact-duplicates.jsonl", "cross-split-groups.jsonl"):
        (bundle / name).write_text("{}\n")
    (bundle / "collection.json").write_text("{}\n")
    seal(bundle, "inventory")
    assert transfer._validate_manifest(read_json(bundle / "manifest.json"), "inventory") is None


def test_measurement_schema_accepts_nested_plan_and_frozen_calibration(tmp_path, monkeypatch):
    _workspace, _client = _configure(monkeypatch, tmp_path)
    bundle = tmp_path / "measurements"
    bundle.mkdir()
    plan = bundle / "plan"
    plan.mkdir()
    for name in ("documents.jsonl", "inputs.jsonl", "research.json", "source-review.json", "model-catalog.json"):
        (plan / name).write_text("{}\n")
    seal(plan, "experiment-plan")
    calibration = bundle / "calibration"
    calibration.mkdir()
    (calibration / "frozen-tables.json").write_text("{}\n")
    (calibration / "stability-report.json").write_text("{}\n")
    seal(calibration, "calibration")
    for name in ("identity.json", "events.jsonl", "failures.jsonl", "mechanics.json"):
        (bundle / name).write_text("{}\n")
    manifest = seal(bundle, "measurements")
    transfer._validate_manifest(manifest, "measurements")
