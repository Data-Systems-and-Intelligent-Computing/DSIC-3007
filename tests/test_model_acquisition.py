"""Offline checks for one-copy, pinned model downloads; no real weights or GPU."""

import hashlib

import pytest

from src.artifacts import atomic, canonical
from src import model


@pytest.mark.parametrize("existing", [False, True])
def test_model_download_uses_destination_without_a_second_full_copy(tmp_path, monkeypatch, existing):
    content = b"synthetic model asset"
    catalog = {"repo_id": "fixture/model", "revision": "pinned-revision", "files": [{
        "path": "weights.safetensors", "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }]}
    catalog_path = tmp_path / "catalog.json"
    atomic(catalog_path, canonical(catalog))
    output = tmp_path / "persistent-model"
    output.mkdir()
    if existing:
        (output / "weights.safetensors").write_bytes(content)
    downloads = []

    def download(repo_id, filename, *, revision, local_dir):
        downloads.append((repo_id, filename, revision, local_dir))
        target = local_dir / filename
        target.write_bytes(content)
        return str(target)

    verified = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr(model, "verify_model", lambda directory, manifest: verified.append((directory, manifest)))
    model.acquire_model(catalog_path, output)
    assert downloads == ([] if existing else [(catalog["repo_id"], "weights.safetensors", catalog["revision"], output)])
    assert verified == [(output, catalog)]
    assert list(output.iterdir()) == [output / "weights.safetensors"]


def test_model_download_still_rejects_wrong_bytes(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    atomic(catalog_path, canonical({"repo_id": "fixture/model", "revision": "pinned", "files": [{
        "path": "weights.safetensors", "size_bytes": 4,
        "sha256": hashlib.sha256(b"good").hexdigest(),
    }]}))

    def download(repo_id, filename, *, revision, local_dir):
        target = local_dir / filename
        target.write_bytes(b"bad!")
        return target

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    with pytest.raises(ValueError, match="pinned model bytes"):
        model.acquire_model(catalog_path, tmp_path / "model")
