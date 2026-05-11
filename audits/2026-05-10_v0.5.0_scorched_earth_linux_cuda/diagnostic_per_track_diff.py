"""Detailed per-track diff between JAX and PyTorch AlphaGenome backends.

Goal: figure out whether the 0.476 max abs drift at SORT1 is:
- (a) concentrated on a few specific tracks (suggests weight mismatch
      or track-mapping bug, fixable per-layer),
- (b) broad uniform noise (suggests accumulated numerical kernel drift,
      hard to fix without changing torch/jax versions), or
- (c) a global scale or sign offset (could be a simple post-hoc fix).

Run with:
  CUDA_VISIBLE_DEVICES=1 CHORUS_NO_TIMEOUT=1 \
  HF_TOKEN=$(cat ~/.token_chorus_audit) \
  mamba run -n chorus python audits/2026-05-10_v0.5.0_scorched_earth_linux_cuda/diagnostic_per_track_diff.py
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

REPO_ROOT = Path("/PHShome/lp698/chorus")
sys.path.insert(0, str(REPO_ROOT))

# Match the test's coordinates + assays exactly.
GENOME_FASTA = REPO_ROOT / "genomes" / "hg38.fa"
SORT1_WINDOW = ("chr1", 108_774_968, 109_774_968)  # ~1 Mb around rs12740374

# All assays the equivalence test exercises (per the test source).
import chorus

# Get the test's ASSAY_IDS list from the test module directly.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "eqv_test",
    REPO_ROOT / "tests" / "test_alphagenome_backends_equivalence.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ASSAY_IDS = mod.ASSAY_IDS

print(f"Comparing {len(ASSAY_IDS)} tracks at SORT1 ({SORT1_WINDOW})\n")
for aid in ASSAY_IDS:
    print(f"  - {aid}")
print()

# Load both backends, run same window through both.
print("=" * 70)
print("Loading JAX backend...")
oracle_jax = chorus.create_oracle(
    "alphagenome", use_environment=True, reference_fasta=str(GENOME_FASTA),
)
oracle_jax.load_pretrained_model()
print("JAX model loaded.\n")
print("Running JAX prediction...")
preds_jax = oracle_jax.predict(SORT1_WINDOW, ASSAY_IDS)
print("JAX prediction done.\n")

print("=" * 70)
print("Loading PyTorch backend...")
oracle_pt = chorus.create_oracle(
    "alphagenome_pt", use_environment=True, reference_fasta=str(GENOME_FASTA),
)
oracle_pt.load_pretrained_model()
print("PT model loaded.\n")
print("Running PT prediction...")
preds_pt = oracle_pt.predict(SORT1_WINDOW, ASSAY_IDS)
print("PT prediction done.\n")

# Detailed per-track analysis.
print("=" * 70)
print("Per-track diff analysis\n")
print(f"{'track':<60} {'max|d|':>9} {'mean|d|':>9} {'corr':>7} {'jax_mean':>10} {'pt_mean':>10} {'ratio':>7}")
print("-" * 120)

global_max = 0.0
diffs = {}
for aid in ASSAY_IDS:
    a = preds_jax[aid].values
    b = preds_pt[aid].values
    if a.shape != b.shape:
        print(f"{aid[:60]:<60} SHAPE MISMATCH jax={a.shape} pt={b.shape}")
        continue
    d = np.abs(a - b)
    max_d = float(d.max())
    mean_d = float(d.mean())
    # pearson r
    if a.std() > 0 and b.std() > 0:
        corr = float(np.corrcoef(a, b)[0, 1])
    else:
        corr = float("nan")
    jax_mean = float(a.mean())
    pt_mean = float(b.mean())
    ratio = pt_mean / jax_mean if abs(jax_mean) > 1e-9 else float("nan")
    diffs[aid] = (max_d, mean_d, corr, jax_mean, pt_mean, ratio)
    global_max = max(global_max, max_d)
    name = aid[:60]
    print(f"{name:<60} {max_d:>9.4f} {mean_d:>9.4f} {corr:>7.4f} {jax_mean:>10.4f} {pt_mean:>10.4f} {ratio:>7.4f}")

print()
print("=" * 70)
print(f"Global max abs diff: {global_max:.4f}  (PR #62 baseline <0.05; test tolerance 0.1)")
print()

# Check whether diffs concentrate at high vs low signal positions
print("Pattern probes — for the worst-drifting track:")
worst = max(diffs.items(), key=lambda kv: kv[1][0])[0]
a = preds_jax[worst].values
b = preds_pt[worst].values
d = np.abs(a - b)
print(f"  worst track: {worst}")
print(f"  signal range:  jax=[{a.min():.4f}, {a.max():.4f}]  pt=[{b.min():.4f}, {b.max():.4f}]")
print(f"  signal mean:   jax={a.mean():.4f}  pt={b.mean():.4f}")
print(f"  diff range:    [{d.min():.4f}, {d.max():.4f}]  mean={d.mean():.4f}")
# Where are the largest diffs? At high-signal positions or low?
top_diff_idx = np.argsort(d)[-10:][::-1]
print(f"  top-10 highest-diff positions:")
for i in top_diff_idx:
    print(f"    bin {i:>6}: jax={a[i]:>8.4f}  pt={b[i]:>8.4f}  |diff|={d[i]:>7.4f}")
