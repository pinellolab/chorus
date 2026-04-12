#!/bin/bash
cd /PHShome/lp698/chorus
/data/pinello/SHARED_SOFTWARE/miniforge3/bin/mamba run -n chorus-alphagenome \
  python scripts/build_10k_backgrounds.py --part baselines --gpu 0 \
  > logs/bg_alphagenome_v2_baselines.log 2>&1
echo "AlphaGenome done"
