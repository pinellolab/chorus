#!/bin/bash
cd /PHShome/lp698/chorus
pkill -f build_backgrounds_chrombpnet 2>/dev/null
sleep 2
ENV_PREFIX=/data/pinello/SHARED_SOFTWARE/envs/lab_envs/chorus-chrombpnet
nohup env LD_PRELOAD=${ENV_PREFIX}/lib/libstdc++.so.6 \
  /data/pinello/SHARED_SOFTWARE/miniforge3/bin/mamba run -n chorus-chrombpnet \
  python scripts/build_backgrounds_chrombpnet.py --part baselines --gpu 0 --batch-size 32 \
  > logs/bg_chrombpnet_v2_baselines_ml007.log 2>&1 &
echo "ChromBPNet PID: $!"
