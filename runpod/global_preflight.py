"""Read-only placement for the selected Global Volume; never fall back to regional storage."""

import math
from pathlib import Path

from runpod import screen
from src.artifacts import read_json
from runpod.global_api import lookup_volume

CONFIG = screen.ROOT / "configs/path-b-l40-storage-decision-v1.json"


def placements(config, catalog, operation="benchmark"):
    """Return all eligible GPU data centers, independent of the volume's location."""
    if config.get("storage_type") != "global" or not config.get("global_volume_id"):
        raise ValueError("explicit Global Volume identity required")
    for key in ("region", "regions", "dataCenter", "dataCenterId", "dataCenterIds"):
        if config.get(key):
            raise ValueError("Global Volume placement must not inherit a volume-region constraint")
    gpu_id = config["gpu"]
    if gpu_id not in ("NVIDIA L40", "NVIDIA L40S"):
        raise ValueError("only an explicitly selected L40/L40S profile is supported")
    card = next((r for r in catalog["gpus"] if r["id"] == gpu_id), None)
    if card is None or not card.get("secure") or card.get("memory", 0) < 48:
        raise ValueError("selected Secure Cloud GPU with >=48GB absent from catalog")
    rate = card["price"]["secure"]
    cap = config["maximum_gpu_rate_usd_per_hour"]
    if (type(rate) not in (int, float) or not math.isfinite(rate) or
            type(cap) not in (int, float) or not math.isfinite(cap) or not 0 < rate <= cap):
        raise ValueError("selected GPU quote exceeds the profile price guard")
    if operation not in ("benchmark", "prepare_model"):
        raise ValueError("unknown global operation")
    floor = config["required_cuda"] if operation == "benchmark" else config["setup_host_cuda_floor"]
    version = lambda value: tuple(int(part) for part in value.split("."))
    available = [r["version"] for r in card.get("cudaVersions", []) if r.get("available")]
    allowed = sorted((v for v in available if version(v) >= version(floor)), key=version)
    if not allowed:
        raise ValueError(f"{operation} requires host CUDA >= {floor}; available: {available}")
    levels = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    rows = [{"data_center_id": r["id"], "availability": r["availability"],
             "gpu": gpu_id, "gpu_usd_per_hour": rate, "allowed_cuda_versions": allowed}
            for r in card.get("dataCenters", []) if r.get("availability") in levels]
    rows.sort(key=lambda r: (levels[r["availability"]], r["data_center_id"]))
    if not rows:
        raise ValueError("no current stock for the selected GPU in any data center")
    return rows


def status(config_path=CONFIG, read_live=False, api=None):
    config = read_json(config_path)
    if config.get("storage_type") != "global" or not config.get("global_volume_id"):
        raise ValueError("explicit Global Volume identity required")
    result = {"status": "global_setup_pending", "gpu": config["gpu"],
              "global_volume_id": config["global_volume_id"],
              "volume_identity_evidence": "user supplied; attachment not independently verified",
              "placement_policy": "any_available_data_center; no volume-region filter",
              "model_path": config["global_volume_model_path"],
              "working_directory": config["local_working_directory"],
              "paid_resources_created": 0, "ready_for_paid_run": False,
              "blockers": ["fresh pinned model download and checksum verification before benchmark"],
              "attachment_backend": "official Runpod SDK GraphQL volumeMounts"}
    try:
        result["cache_setup_receipt"] = cache_ready(config)
        result.update(status="model_cache_ready", blockers=[])
    except ValueError:
        pass
    if read_live:
        reader = api or screen.api
        if api is None:
            result["verified_volume"] = lookup_volume(config)
            result["volume_identity_evidence"] = "live GraphQL ID/name read-back"
        catalog = reader("GET", "/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE")
        try:
            result["model_setup_placements"] = placements(config, catalog, "prepare_model")
        except ValueError as error:
            result["model_setup_placements"] = []
            result["model_setup_blocker"] = str(error)
        try:
            result["available_placements"] = placements(config, catalog)
            result["preferred_data_center"] = result["available_placements"][0]["data_center_id"]
        except ValueError as error:
            result["available_placements"] = []
            result["blockers"].append(str(error))
        result["cuda_capacity_scope"] = "GPU-wide catalog; exact host CUDA still requires pod read-back"
        result["hardware_and_cache_ready"] = not result["blockers"]
    result["paid_authorization_status"] = config.get("paid_authorization", {}).get("benchmark_retry_proposal", {}).get("status", "not_authorized")
    result["ready_for_paid_run"] = False
    return result


def live_profile(config_path=CONFIG, operation="benchmark"):
    config = read_json(config_path)
    volume = lookup_volume(config)
    registry = screen.api("GET", f"/registries/{screen.REGISTRY}")
    if registry.get("id") != screen.REGISTRY or registry.get("name") != screen.REGISTRY_NAME:
        raise ValueError("pinned registry identity differs")
    if screen.account_pods():
        raise ValueError("account has existing pods; preserve running work")
    rows = placements(config, screen.api("GET", "/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE"), operation)
    key = Path.home()/".ssh/id_ed25519.pub"
    if not key.is_file() or not key.with_suffix("").is_file():
        raise ValueError("existing SSH key pair required")
    profile = {k:config[k] for k in ("storage_type", "gpu", "global_volume_id",
               "global_volume_mount_path", "global_volume_model_path")}
    profile.update(eligible_data_centers=[r["data_center_id"] for r in rows],
                   allowed_cuda_versions=rows[0]["allowed_cuda_versions"], operation=operation,
                   volume_identity_evidence="live GraphQL ID/name read-back", verified_volume=volume)
    return profile, rows[0]["gpu_usd_per_hour"], key.read_text().strip()


def cache_ready(config):
    """Require a retrieved, successful setup receipt for this exact volume/catalog."""
    from src.artifacts import digest
    catalog = read_json(screen.ROOT/config["model_catalog"])
    for candidate in (screen.ROOT/"runpod/runs").glob("*/candidate.json"):
        identity = read_json(candidate)
        if (identity.get("hardware_profile", {}).get("global_volume_id") != config["global_volume_id"] or
                identity.get("proposal", {}).get("global_operation") != "prepare_model"):
            continue
        path = candidate.parent/"results/model-preparation.json"
        if path.is_file():
            receipt = read_json(path)
            expected = [{"path":a["path"],"bytes":a["size_bytes"],"sha256":a["sha256"]} for a in catalog["files"]]
            if (receipt.get("status") == "pass" and receipt.get("catalog_sha256") == digest(catalog) and
                    receipt.get("model_path") == config["global_volume_model_path"] and
                    receipt.get("verified_files") == expected):
                return str(path)
    raise ValueError("no complete retrieved model-setup receipt for this Global Volume; run --setup-model first")
