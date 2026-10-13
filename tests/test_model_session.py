"""A reused model must keep the original executor and parameter identity."""

import pytest
import torch

from src import model as loader


def fixture(monkeypatch):
    model = torch.nn.Linear(2, 2).eval()
    model.config = type("Config", (), {"use_cache": False})()
    calls = []
    monkeypatch.setattr(loader, "implementation_identity", lambda: "implementation")
    monkeypatch.setattr(loader, "runtime_identity", lambda: {"dtype": "float32"})
    monkeypatch.setattr(loader, "load_model", lambda *args: calls.append(args) or model)
    return model, calls, loader.VerifiedModelSession()


def test_session_loads_once_and_rejects_executor_change(monkeypatch, tmp_path):
    model, calls, session = fixture(monkeypatch)
    assert session.acquire(tmp_path, {"revision": "pinned"}, 3007) is model
    assert session.acquire(tmp_path, {"revision": "pinned"}, 3007) is model
    assert len(calls) == 1
    with pytest.raises(ValueError, match="identity changed"):
        session.acquire(tmp_path, {"revision": "different"}, 3007)


def test_session_rejects_changed_parameter(monkeypatch, tmp_path):
    model, calls, session = fixture(monkeypatch)
    session.acquire(tmp_path, {"revision": "pinned"}, 3007)
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(ValueError, match="parameters changed"):
        session.acquire(tmp_path, {"revision": "pinned"}, 3007)


def test_explicit_benchmark_session_skips_weight_verification_and_loads_once(monkeypatch, tmp_path):
    model, _, _ = fixture(monkeypatch)
    calls=[]
    def load(directory, catalog, seed, *, verify_weights):
        calls.append(verify_weights)
        return model
    monkeypatch.setattr(loader,"load_model",load)
    session=loader.VerifiedModelSession(verify_weights=False)
    assert session.acquire(tmp_path,{"revision":"pinned"},3007) is model
    assert session.acquire(tmp_path,{"revision":"pinned"},3007) is model
    assert calls == [False]
