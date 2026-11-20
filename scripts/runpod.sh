#!/usr/bin/env bash
# Thin SSH/rsync wrapper for a console-created or Terraform-managed pod.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
action=${1:-help}
shift || true

if [[ "$action" == help ]]; then
    echo 'SSH reads runpod.ssh_host, runpod.ssh_port and the private key matching runpod.ssh_public_key_path in execution config.'
    echo 'runpod.sh push | acquire | restore | model LOCAL_FP32_DIR | run STAGE... [--units N] | fetch | cleanup'
    exit 0
fi
execution_file=${PATH_B_EXECUTION_FILE:-config/execution.json}
PATH_B_HOST=${PATH_B_HOST:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["runpod"].get("ssh_host", ""))' "$execution_file")}
PATH_B_PORT=${PATH_B_PORT:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["runpod"].get("ssh_port", ""))' "$execution_file")}
PATH_B_SSH_KEY=$(python -c 'import json,os,sys; r=json.load(open(sys.argv[1]))["runpod"]; print(os.path.expanduser(os.environ.get("PATH_B_SSH_KEY") or r["ssh_public_key_path"].removesuffix(".pub")))' "$execution_file")
: "${PATH_B_HOST:?Set runpod.ssh_host in config/execution.json}"
: "${PATH_B_PORT:?Set runpod.ssh_port in config/execution.json}"
connect_seconds=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["runpod"]["ssh_connect_seconds"])' "$execution_file")
connection_attempts=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["runpod"]["ssh_attempts"])' "$execution_file")
ssh_args=(-i "$PATH_B_SSH_KEY" -p "$PATH_B_PORT" -o BatchMode=yes -o "ConnectTimeout=$connect_seconds" -o "ConnectionAttempts=$connection_attempts" -o StrictHostKeyChecking=accept-new)
if [[ -n "${PATH_B_KNOWN_HOSTS:-}" ]]; then
    ssh_args+=(-o "UserKnownHostsFile=$PATH_B_KNOWN_HOSTS")
fi
if [[ -n "${PATH_B_SSH_CONTROL_PATH:-}" ]]; then
    ssh_args+=(-o ControlMaster=auto -o ControlPersist=60 -o "ControlPath=$PATH_B_SSH_CONTROL_PATH")
fi
target="root@$PATH_B_HOST"
remote=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["runpod"]["remote_root"])' "$execution_file")
managed=$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["runpod"]["terraform_managed"]))' "$execution_file")
model=$(python -c 'import json,os,sys; x=json.load(open(sys.argv[1])); print(os.environ.get("PATH_B_MODEL_PATH", x["model_path"]))' "$execution_file")
[[ "$remote" =~ ^/[a-zA-Z0-9_./-]+$ && "$remote" != / && "$remote" != *..* ]] || { echo 'Use an absolute runpod.remote_root without spaces or parent traversal' >&2; exit 2; }
printf -v quoted_model '%q' "$model"
prepared=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["prepared_path"])' "$execution_file")
results=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["results_path"])' "$execution_file")
[[ "$prepared" =~ ^[a-zA-Z0-9_./-]+$ && "$prepared" != /* && "$prepared" != *..* ]] || { echo 'Use a relative prepared_path for remote transfer' >&2; exit 2; }
[[ "$results" =~ ^[a-zA-Z0-9_./-]+$ && "$results" != /* && "$results" != *..* ]] || { echo 'Use a relative results_path for remote transfer' >&2; exit 2; }
printf -v transport '%q ' ssh "${ssh_args[@]}"

fetch() {
    mkdir -p "$root/$results"
    # Online SQLite backup produces self-contained copies even during a live job.
    ssh "${ssh_args[@]}" "$target" "cd $remote && /opt/path-b/.venv/bin/python -m scripts.export_results" || return "$?"
    rsync -az --checksum --protect-args -e "$transport" "$target:$remote/export/" "$root/$results/" || return "$?"
    python -m scripts.export_results --check "$root/$results"
}

release_compute() {
    if [[ "$managed" == 1 ]]; then
        terraform -chdir="$root/terraform" apply -auto-approve -var=enable_compute=false
    else
        echo 'Results retrieved. This pod was created outside Terraform; terminate it in RunPod when finished.'
    fi
}

case "$action" in
    push)
        ssh "${ssh_args[@]}" "$target" "mkdir -p $remote/$prepared $remote/data/manifests"
        rsync -az --protect-args -e "$transport" --exclude='__pycache__/' "$root/scripts" "$root/config" "$root/README.md" "$root/pyproject.toml" "$root/uv.lock" "$target:$remote/"
        rsync -az --protect-args -e "$transport" "$execution_file" "$target:$remote/config/execution.json"
        rsync -az --protect-args -e "$transport" "$root/$prepared/" "$target:$remote/$prepared/"
        rsync -az --protect-args -e "$transport" "$root/data/manifests/model-fp32.json" "$target:$remote/data/manifests/"
        ;;
    acquire)
        # Separate lightweight preparation environment; never install into GPU runtime.
        acquisition_python=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["runpod"]["acquisition_python"])' "$execution_file")
        hub_version=$(python -c 'import tomllib; print(next(p["version"] for p in tomllib.load(open("uv.lock","rb"))["package"] if p["name"] == "huggingface-hub"))')
        printf -v quoted_acquisition_python '%q' "$acquisition_python"
        ssh "${ssh_args[@]}" "$target" "cd $remote && export PATH_B_MODEL_PATH=$quoted_model &&
            { test -x .acquire-venv/bin/python || $quoted_acquisition_python -m venv .acquire-venv; } &&
            { .acquire-venv/bin/python -c 'import huggingface_hub; assert huggingface_hub.__version__ == \"$hub_version\"' 2>/dev/null ||
              .acquire-venv/bin/python -m pip install 'huggingface_hub==$hub_version'; } &&
            .acquire-venv/bin/python -u -m scripts.acquire_model"
        ;;
    restore)
        # Only restore onto a new pod with no observations yet. Never replace live results.
        snapshot=$(mktemp -d)
        trap 'rm -rf -- "$snapshot"' EXIT
        python -m scripts.export_results --execution "$execution_file" --destination "$snapshot"
        python -m scripts.export_results --check "$snapshot"
        parent=$(dirname "$results")
        ssh "${ssh_args[@]}" "$target" "test ! -e $remote/$results && mkdir -p $remote/$parent $remote/.restore" || { echo 'Restore refused or SSH failed; keep existing remote results' >&2; exit 2; }
        rsync -az --delete --protect-args -e "$transport" "$snapshot/" "$target:$remote/.restore/"
        ssh "${ssh_args[@]}" "$target" "cd $remote && /opt/path-b/.venv/bin/python -m scripts.export_results --check .restore && test ! -e $results && mv .restore $results"
        ;;
    model)
        [[ $# == 1 ]] || { echo 'Usage: runpod.sh model LOCAL_FP32_DIR' >&2; exit 2; }
        ssh "${ssh_args[@]}" "$target" "mkdir -p $quoted_model"
        rsync -az --checksum --protect-args -e "$transport" "$1/" "$target:$model/"
        ;;
    run)
        [[ $# -gt 0 ]] || { echo 'Supply GPU stages' >&2; exit 2; }
        # Arguments are quoted individually; no eval or sourced secret files.
        printf -v args '%q ' "$@"
        seconds=$(python -c 'import json,sys; x=json.load(open(sys.argv[1]))["max_seconds"]; print("" if x is None else int(x))' "$execution_file")
        timeout_cmd=''
        if [[ -n "$seconds" ]]; then timeout_cmd="timeout --signal=TERM --kill-after=30 ${seconds}s "; fi
        command="cd $remote && export PATH_B_MODEL_PATH=$quoted_model CUBLAS_WORKSPACE_CONFIG=:4096:8 && ${timeout_cmd}/opt/path-b/.venv/bin/python -u -m scripts.run_gpu $args"
        mkdir -p "$root/$results"
        set +e
        ssh "${ssh_args[@]}" "$target" "$command" 2>&1 | tee -a "$root/$results/console.log"
        status=${PIPESTATUS[0]}
        fetch
        fetch_status=$?
        cleanup_status=0
        if [[ "$fetch_status" == 0 ]]; then
            release_compute
            cleanup_status=$?
        else
            echo 'Retrieval failed. Keep this pod until its results have been recovered.' >&2
        fi
        set -e
        if [[ "$status" != 0 ]]; then exit "$status"; fi
        if [[ "$fetch_status" != 0 ]]; then exit "$fetch_status"; fi
        exit "$cleanup_status"
        ;;
    fetch) fetch ;;
    cleanup)
        # Retrieval must succeed before deleting compute, including with local pod storage.
        fetch
        release_compute
        ;;
    *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
