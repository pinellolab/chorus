#!/usr/bin/env bash
# Distributed BPNet/CHIP CDF build across 6 GPUs (ml003 + ml007 + ml008,
# 2 GPUs each). Run from a control machine that can ssh to all three.
#
# Splits the 744 unique BPNet TF×cell_type models into 6 equal shards.
# Each GPU runs ~124 models, ~3 min each → ~6.5 hours wall-clock.
# Each shard writes its own interim NPZ; aggregate with --part merge-shards
# after all six finish.
#
# Prereqs on each remote machine:
#   - chorus repo at ~/chorus (or set CHORUS_REPO below)
#   - chorus-chrombpnet env with tensorflow + cuda
#   - genomes/hg38.fa downloaded
#   - HF_TOKEN set if alphagenome models are needed (not for CHIP)
#
# Run as: bash scripts/run_bpnet_cdf_build.sh

set -euo pipefail

CHORUS_REPO="${CHORUS_REPO:-~/chorus}"
LOG_DIR="/tmp/chorus_bpnet_build_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"
echo "Logs: $LOG_DIR"

# Each entry: "host gpu_id shard_id"
declare -a JOBS=(
    "ml003 0 0"
    "ml003 1 1"
    "ml007 0 2"
    "ml007 1 3"
    "ml008 0 4"
    "ml008 1 5"
)

PIDS=()
for entry in "${JOBS[@]}"; do
    read -r host gpu shard <<< "$entry"
    log="$LOG_DIR/${host}_gpu${gpu}_shard${shard}.log"
    echo "Starting shard ${shard}/6 on ${host}:${gpu} → $log"
    ssh "$host" "cd $CHORUS_REPO && CHORUS_NO_TIMEOUT=1 mamba run -n chorus-chrombpnet python scripts/build_backgrounds_chrombpnet.py \
        --part both --assay CHIP --shard $shard --shard-of 6 --gpu $gpu" \
        > "$log" 2>&1 &
    PIDS+=($!)
done

echo "All 6 shards kicked off. Waiting..."
echo "Tail any log with: tail -F $LOG_DIR/<host>_gpu<G>_shard<S>.log"

wait "${PIDS[@]}"
echo "All shards completed."
echo
echo "Now aggregate shards onto one machine:"
echo "  scp ml003:~/.chorus/backgrounds/chrombpnet_*_interim.shard*of6.npz <local>"
echo "  scp ml007:~/.chorus/backgrounds/chrombpnet_*_interim.shard*of6.npz <local>"
echo "  scp ml008:~/.chorus/backgrounds/chrombpnet_*_interim.shard*of6.npz <local>"
echo "  mamba run -n chorus python scripts/build_backgrounds_chrombpnet.py --part merge-shards"
echo
echo "Then push to HF:"
echo "  HF_TOKEN=<write_token> python /tmp/push_chrombpnet_cdf.py"
