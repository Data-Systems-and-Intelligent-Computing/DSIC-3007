"""Verified original FP32 OLMoE loading and explicit numerical/runtime identity."""

import importlib.metadata
import os
import platform
from pathlib import Path

from .artifacts import digest, file_digest, read_json


def verify_model(directory, catalog):
    directory = Path(directory)
    if catalog["revision"] != "c6f6954f182ffdd9f6ab4f4d772714b7dcd6836b":
        raise ValueError("unexpected base FP32 model revision")
    for asset in catalog["files"]:
        p = directory / asset["path"]
        if p.stat().st_size != asset["size_bytes"] or file_digest(p) != asset["sha256"]:
            raise ValueError(f"model byte identity mismatch: {asset['path']}")
    from safetensors import safe_open
    for asset in catalog["files"]:
        if not asset["path"].endswith(".safetensors"):
            continue
        with safe_open(directory / asset["path"], framework="pt", device="cpu") as f:
            if any(f.get_slice(key).get_dtype() != "F32" for key in f.keys()):
                raise ValueError("original FP32 tensors required; casting BF16 is insufficient")
    config = read_json(directory / "config.json")
    if (config["num_hidden_layers"], config["num_experts"], config["num_experts_per_tok"]) != (16, 64, 8):
        raise ValueError("wrong architecture")
    if config["norm_topk_prob"]:
        raise ValueError("unexpected native coefficient policy")


def runtime_identity():
    import inspect
    import torch
    from transformers.models.olmoe import modeling_olmoe
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {"python": platform.python_version(), "software": {name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy")},
            "modeling_source_sha256": file_digest(inspect.getfile(modeling_olmoe)),
            "cuda": torch.version.cuda, "gpu_name": gpu.name if gpu else None,
            "gpu_memory_bytes": gpu.total_memory if gpu else None,
            "dtype": "float32", "loss_dtype": "float64", "batch_size": 1,
            "attention": "eager", "experts": "eager", "cache": False,
            "image_digest": os.environ.get("RESEARCH_IMAGE_DIGEST"),
            "deterministic_algorithms": True, "tf32": False}


def load_model(directory, catalog, seed, *, verify_weights=True):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from transformers import AutoModelForCausalLM
    if importlib.metadata.version("transformers") != "5.12.1":
        raise ValueError("instrumentation requires inspected Transformers 5.12.1")
    if not torch.cuda.is_available():
        raise ValueError("explicit GPU execution required; no silent CPU/dtype fallback")
    if verify_weights:
        verify_model(directory, catalog)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False, dtype=torch.float32,
        device_map={"": "cuda:0"}, attn_implementation="eager", experts_implementation="eager",
        use_kernels=False)
    model.eval()
    model.config.use_cache = False
    model.requires_grad_(False)
    if any(p.dtype != torch.float32 or p.device.type != "cuda" for p in model.parameters()):
        raise ValueError("entire native FP32 model must be on the GPU")
    return model


def implementation_identity():
    root = Path(__file__).resolve().parent
    names = ("model.py", "interventions.py", "run_experiment.py", "verify.py", "artifacts.py",
             "reuse.py", "observation_contract.py", "optimized_validation.py", "scope.py")
    return digest({name:file_digest(root/name) for name in names})


class VerifiedModelSession:
    """Reuse one verified loaded model while rejecting identity or weight changes."""

    def __init__(self, *, verify_weights=True):
        self._model = None
        self._identity = None
        self._versions = None
        self._verify_weights = verify_weights

    def acquire(self, directory, catalog, seed):
        identity = {"directory": str(Path(directory).resolve()), "catalog": digest(catalog),
                    "seed": seed, "implementation": implementation_identity(),
                    "runtime": runtime_identity()}
        if self._model is None:
            if self._verify_weights:
                self._model = load_model(directory, catalog, seed)
            else:
                self._model = load_model(directory, catalog, seed, verify_weights=False)
            self._identity = identity
            self._versions = {name: parameter._version for name, parameter in self._model.named_parameters()}
        elif identity != self._identity:
            raise ValueError("loaded model session identity changed")
        if self._versions != {name: parameter._version for name, parameter in self._model.named_parameters()}:
            raise ValueError("loaded model parameters changed")
        if self._model.training or self._model.config.use_cache:
            raise ValueError("loaded model session requires eval mode and no KV cache")
        return self._model


def acquire_model(catalog_path, output):
    """Explicit pinned-weight acquisition; never called by collection."""
    from huggingface_hub import hf_hub_download
    catalog = read_json(catalog_path)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    for asset in catalog["files"]:
        target = output/asset["path"]
        if not target.exists():
            tmp = target.with_name(target.name+".tmp")
            if tmp.exists():
                raise ValueError("unfinished model copy requires explicit recovery")
            # Download into the persistent model directory. The Hub manages
            # resumable temporary files and metadata there, without retaining
            # another complete copy in its default cache on container disk.
            downloaded = hf_hub_download(catalog["repo_id"], asset["path"],
                                         revision=catalog["revision"], local_dir=output)
            if Path(downloaded).resolve() != target.resolve():
                raise ValueError("model download did not land in the requested directory")
        if target.stat().st_size != asset["size_bytes"] or file_digest(target) != asset["sha256"]:
            raise ValueError("download differs from pinned model bytes")
    verify_model(output,catalog)
