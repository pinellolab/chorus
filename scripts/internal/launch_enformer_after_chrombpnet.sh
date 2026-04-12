#!/bin/bash
cd /PHShome/lp698/chorus
echo "Waiting for ChromBPNet to finish..."
while ps aux | grep "build_backgrounds_chrombpnet" | grep python | grep -v grep > /dev/null; do
    sleep 60
done
echo "ChromBPNet done. Launching Enformer on GPU 0..."
ENV_PREFIX=/data/pinello/SHARED_SOFTWARE/envs/lab_envs/chorus-enformer
env LD_PRELOAD=${ENV_PREFIX}/lib/libstdc++.so.6 \
  /data/pinello/SHARED_SOFTWARE/miniforge3/bin/mamba run -n chorus-enformer \
  python scripts/build_backgrounds_enformer.py --part baselines --gpu 0 \
  > logs/bg_enformer_v2_baselines_ml007.log 2>&1
echo "Enformer done."

echo "Launching AlphaGenome on GPU 0..."
/data/pinello/SHARED_SOFTWARE/miniforge3/bin/mamba run -n chorus-alphagenome \
  python scripts/build_10k_backgrounds.py --part baselines --gpu 0 \
  > logs/bg_alphagenome_v2_baselines_ml007.log 2>&1
echo "AlphaGenome done."
