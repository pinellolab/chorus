#!/bin/bash
# Launch baseline rebuilds on ml007 (2x A100 GPUs)
# Uses SCREEN cCRE sampling + per-bin collection
cd /PHShome/lp698/chorus

# Enformer baselines on GPU 0
ENV_PREFIX=/data/pinello/SHARED_SOFTWARE/envs/lab_envs/chorus-enformer
nohup env LD_PRELOAD=${ENV_PREFIX}/lib/libstdc++.so.6 \
  /data/pinello/SHARED_SOFTWARE/miniforge3/bin/mamba run -n chorus-enformer \
  python scripts/build_backgrounds_enformer.py --part baselines --gpu 0 \
  > logs/bg_enformer_v2_baselines_ml007.log 2>&1 &
echo "Enformer PID: $!"

# Borzoi baselines on GPU 1
ENV_PREFIX=/data/pinello/SHARED_SOFTWARE/envs/lab_envs/chorus-borzoi
nohup env LD_PRELOAD=${ENV_PREFIX}/lib/libstdc++.so.6 \
  /data/pinello/SHARED_SOFTWARE/miniforge3/bin/mamba run -n chorus-borzoi \
  python scripts/build_backgrounds_borzoi.py --part baselines --gpu 1 \
  > logs/bg_borzoi_v2_baselines_ml007.log 2>&1 &
echo "Borzoi PID: $!"

# ChromBPNet baselines on GPU 0 (small model, fast with batching)
ENV_PREFIX=/data/pinello/SHARED_SOFTWARE/envs/lab_envs/chorus-chrombpnet
nohup env LD_PRELOAD=${ENV_PREFIX}/lib/libstdc++.so.6 \
  /data/pinello/SHARED_SOFTWARE/miniforge3/bin/mamba run -n chorus-chrombpnet \
  python scripts/build_backgrounds_chrombpnet.py --part baselines --gpu 0 \
  > logs/bg_chrombpnet_v2_baselines_ml007.log 2>&1 &
echo "ChromBPNet PID: $!"

# AlphaGenome baselines on GPU 1 (after Borzoi — large model)
ENV_PREFIX=/data/pinello/SHARED_SOFTWARE/envs/lab_envs/chorus-alphagenome
nohup /data/pinello/SHARED_SOFTWARE/miniforge3/bin/mamba run -n chorus-alphagenome \
  python scripts/build_10k_backgrounds.py --part baselines --gpu 1 \
  > logs/bg_alphagenome_v2_baselines_ml007.log 2>&1 &
echo "AlphaGenome PID: $!"

echo "All jobs launched on $(hostname)"
