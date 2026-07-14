#!/usr/bin/env bash
set -euo pipefail

script_name=${0##*/}
config=${1:?"usage: $script_name CONFIG [REPLICAS]"}
replicas=${2:-1}

if [[ ! "$replicas" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: REPLICAS must be a positive integer" >&2
    exit 2
fi

if (( replicas == 1 )); then
    exec python3 -u -m onpanda.tool.statistics --config "$config"
fi

run_id=$(date +%Y%m%d-%H%M%S)
echo "[statistics] parallel run $run_id with $replicas replicas"
rlaunch_args=(
    -P "$replicas"
    --cpu 2
    --memory 8192
    --backoff-limit 1
    --replica-prefix
    --workdir "$PWD"
)
# Set this only when workers require a provider-specific shared mount.
if [[ -n "${RLAUNCH_MOUNT:-}" ]]; then
    rlaunch_args+=(--mount="$RLAUNCH_MOUNT")
fi

rlaunch "${rlaunch_args[@]}" -- \
    python3 -u -m onpanda.tool.statistics \
    --config "$config" --shard-run-id "$run_id"
python3 -u -m onpanda.tool.statistics \
    --config "$config" --merge-shards "$run_id"
