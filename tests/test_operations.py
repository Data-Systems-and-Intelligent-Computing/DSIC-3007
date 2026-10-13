import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

from scripts.common import snapshot


@pytest.mark.parametrize("run_status", [0, 17])
def test_run_releases_compute_after_fetch_failure(tmp_path, run_status):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "bin").mkdir()
    shutil.copy2(root / "scripts/runpod.sh", tmp_path / "scripts/runpod.sh")
    shutil.copy2(root / "config/execution.json", tmp_path / "config/execution.json")
    ssh = tmp_path / "bin/ssh"
    ssh.write_text('#!/bin/sh\nprintf "ssh %s\\n" "$*" >> "$MOCK_LOG"\ncase "$*" in\n *scripts.run_gpu*) exit "$MOCK_RUN_STATUS" ;;\n *scripts.export_results*) exit 9 ;;\nesac\n')
    terraform = tmp_path / "bin/terraform"
    terraform.write_text('#!/bin/sh\nprintf "terraform %s\\n" "$*" >> "$MOCK_LOG"\n')
    for file in [ssh, terraform]:
        file.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path / "bin") + ":" + os.environ["PATH"],
           "PATH_B_HOST": "mock.invalid", "PATH_B_PORT": "22", "PATH_B_SSH_KEY": "mock-key",
           "MOCK_LOG": str(tmp_path / "operations.log"), "MOCK_RUN_STATUS": str(run_status)}
    result = subprocess.run(["bash", str(tmp_path / "scripts/runpod.sh"), "run", "pilot", "--units", "20"], env=env, capture_output=True, text=True)
    assert result.returncode == (run_status or 9), result.stderr
    operations = (tmp_path / "operations.log").read_text()
    assert "terraform" in operations and "enable_compute=false" in operations


def test_snapshot_keeps_lock_and_excludes_state_and_variable_secrets(tmp_path, monkeypatch):
    import scripts.common as common
    for directory in ["scripts", "config", "terraform"]:
        (tmp_path / directory).mkdir()
    (tmp_path / "scripts/measure.py").write_text("# actual source\n")
    (tmp_path / "terraform/main.tf").write_text("# source\n")
    (tmp_path / "terraform/terraform.tfstate.backup").write_text("private state")
    (tmp_path / "terraform/private.auto.tfvars.json").write_text("private variables")
    (tmp_path / "uv.lock").write_text("pinned lock")
    monkeypatch.setattr(common, "ROOT", tmp_path)
    snapshot(tmp_path / "results", {}, {})
    with zipfile.ZipFile(tmp_path / "results/source.zip") as archive:
        names = set(archive.namelist())
    assert "uv.lock" in names and "scripts/measure.py" in names
    assert not any("tfstate" in name or "tfvars" in name for name in names)
