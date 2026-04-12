#!/bin/bash
# Wait for a GPU with at least N GB free, then launch a command on it.
# Usage: launch_when_gpu_free.sh <min_gb> <command...>
# The selected GPU index is exported as CUDA_VISIBLE_DEVICES_PICKED

set -e

MIN_GB=${1:-15}
shift

LOG_FILE="${LAUNCH_LOG:-/tmp/launch_when_gpu_free.log}"
echo "[$(date)] Waiting for GPU with >= ${MIN_GB} GB free..." | tee -a "$LOG_FILE"

while true; do
    # Get free memory per GPU in MiB
    free_mem=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    picked=""
    while IFS=, read -r idx free; do
        free=$(echo "$free" | tr -d ' ')
        idx=$(echo "$idx" | tr -d ' ')
        free_gb=$((free / 1024))
        if [ "$free_gb" -ge "$MIN_GB" ]; then
            picked="$idx"
            break
        fi
    done <<< "$free_mem"

    if [ -n "$picked" ]; then
        echo "[$(date)] GPU $picked has >= ${MIN_GB} GB free, launching..." | tee -a "$LOG_FILE"
        export CUDA_VISIBLE_DEVICES_PICKED="$picked"
        # Replace --gpu argument if present
        new_args=()
        skip_next=false
        gpu_set=false
        for arg in "$@"; do
            if [ "$skip_next" = "true" ]; then
                new_args+=("$picked")
                skip_next=false
                gpu_set=true
                continue
            fi
            if [ "$arg" = "--gpu" ]; then
                new_args+=("$arg")
                skip_next=true
                continue
            fi
            new_args+=("$arg")
        done
        if [ "$gpu_set" = "false" ]; then
            new_args+=("--gpu" "$picked")
        fi
        exec "${new_args[@]}"
    fi

    sleep 60
done
