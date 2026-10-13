#!/usr/bin/env bash
# Thin SSH/rsync wrapper. Terraform owns only the pod and durable network storage.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
action=${1:-help}
shift || true

if [[ "$action" == help ]]; then
    echo 'Set PATH_B_HOST, PATH_B_PORT, PATH_B_SSH_KEY from the pod SSH connection.'
    echo 'runpod.sh push | model LOCAL_FP32_DIR | run STAGE... [--units N] | fetch | cleanup'
    exit 0
fi
: "${PATH_B_HOST:?Set the SSH host from RunPod console}"
: "${PATH_B_PORT:?Set the mapped SSH TCP port}"
: "${PATH_B_SSH_KEY:?Set the registered private key path}"
ssh_args=(-i "$PATH_B_SSH_KEY" -p "$PATH_B_PORT" -o BatchMode=yes -o ConnectTimeout=20)
target="root@$PATH_B_HOST"
remote=/workspace/path-b
prepared=$(python -c 'import json; print(json.load(open("config/execution.json"))["prepared_path"])')
results=$(python -c 'import json; print(json.load(open("config/execution.json"))["results_path"])')
[[ "$prepared" =~ ^[a-zA-Z0-9_./-]+$ && "$prepared" != /* && "$prepared" != *..* ]] || { echo 'Use a relative prepared_path for remote transfer' >&2; exit 2; }
[[ "$results" =~ ^[a-zA-Z0-9_./-]+$ && "$results" != /* && "$results" != *..* ]] || { echo 'Use a relative results_path for remote transfer' >&2; exit 2; }
printf -v transport 'ssh -i %q -p %q -o BatchMode=yes -o ConnectTimeout=20' "$PATH_B_SSH_KEY" "$PATH_B_PORT"

fetch() {
    mkdir -p "$root/$results"
    # Online SQLite backup produces self-contained copies even during a live job.
    ssh "${ssh_args[@]}" "$target" "cd $remote && /opt/path-b/.venv/bin/python -m scripts.export_results" || return "$?"
    rsync -az --checksum --protect-args -e "$transport" "$target:$remote/export/" "$root/$results/" || return "$?"
    python -m scripts.export_results --check "$root/$results"
}

case "$action" in
    push)
        ssh "${ssh_args[@]}" "$target" "mkdir -p $remote/$prepared $remote/data/manifests"
        rsync -az --protect-args -e "$transport" --exclude='__pycache__/' "$root/scripts" "$root/config" "$root/pyproject.toml" "$root/uv.lock" "$target:$remote/"
        rsync -az --protect-args -e "$transport" "$root/$prepared/" "$target:$remote/$prepared/"
        rsync -az --protect-args -e "$transport" "$root/data/manifests/model-fp32.json" "$target:$remote/data/manifests/"
        ;;
    model)
        [[ $# == 1 ]] || { echo 'Usage: runpod.sh model LOCAL_FP32_DIR' >&2; exit 2; }
        ssh "${ssh_args[@]}" "$target" 'mkdir -p /workspace/models/OLMoE-fp32'
        rsync -az --checksum --protect-args -e "$transport" "$1/" "$target:/workspace/models/OLMoE-fp32/"
        ;;
    run)
        [[ $# -gt 0 ]] || { echo 'Supply GPU stages' >&2; exit 2; }
        # Arguments are quoted individually; no eval or sourced secret files.
        printf -v args '%q ' "$@"
        seconds=$(python -c 'import json; x=json.load(open("config/execution.json"))["max_seconds"]; print("" if x is None else int(x))')
        timeout_cmd=''
        if [[ -n "$seconds" ]]; then timeout_cmd="timeout --signal=TERM --kill-after=30 ${seconds}s "; fi
        command="cd $remote && export PATH_B_MODEL_PATH=/workspace/models/OLMoE-fp32 CUBLAS_WORKSPACE_CONFIG=:4096:8 && ${timeout_cmd}/opt/path-b/.venv/bin/python -u -m scripts.run_gpu $args"
        mkdir -p "$root/$results"
        set +e
        ssh "${ssh_args[@]}" "$target" "$command" 2>&1 | tee -a "$root/$results/console.log"
        status=${PIPESTATUS[0]}
        fetch
        fetch_status=$?
        # Results and model live on the independently retained network volume.
        terraform -chdir="$root/terraform" apply -auto-approve -var=enable_compute=false
        cleanup_status=$?
        set -e
        if [[ "$status" != 0 ]]; then exit "$status"; fi
        if [[ "$fetch_status" != 0 ]]; then exit "$fetch_status"; fi
        exit "$cleanup_status"
        ;;
    fetch) fetch ;;
    cleanup)
        # Retrieval must succeed before deleting compute. Network volume survives.
        fetch
        terraform -chdir="$root/terraform" apply -var=enable_compute=false
        ;;
    *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
