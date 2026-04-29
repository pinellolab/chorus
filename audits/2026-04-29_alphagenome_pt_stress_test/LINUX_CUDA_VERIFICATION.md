# Linux/CUDA verification for PR #62 — instructions for the agent

You're picking up the Linux/CUDA spot-check that gates merge of
[chorus PR #62](https://github.com/pinellolab/chorus/pull/62)
("AlphaGenome PyTorch backend, opt-in spike"). The branch is
`feat/alphagenome-pytorch-backend`. macOS/MPS verification is done; we
need the same equivalence + speed numbers measured on Linux + an
NVIDIA CUDA GPU before flipping the PR to ready.

## Context — what's already verified (macOS)

- **Functional equivalence (Tier 1)**: chorus public API on 524 kb
  SORT1 window — JAX vs PT match within 1–2% per-assay across all 7
  chorus-exposed assays.
- **Functional equivalence (Tier 2)**: full Fig 3f region-swap layer
  scores agree within 0.02 log₂ on 6 tracks.
- **Speed (M3 Ultra)**: PT MPS is 13.8× faster than JAX CPU on a
  524 kb head pass, 8.9× faster on the full 1 MB region-swap pipeline.
- **GPU on-die cache cliff**: at ~768→896 kb on Apple Silicon, MPS
  regresses sharply. The routing helper uses 600 kb as a safe-zone.
  We don't expect this cliff on CUDA — that's part of what you're
  measuring.
- **F2 (P0)** caught and fixed: `organism_index` must be a torch.Tensor,
  not a Python int (twin call sites in the subprocess template AND the
  direct-load oracle method, both now fixed; static-text regressions
  in `tests/test_alphagenome_pt_predict_template.py`).

## Goals (in priority order)

1. **(Required for ready-flip)** Confirm the chorus-API equivalence
   number from Tier 1 carries on CUDA — same tolerances expected.
2. **(Required for ready-flip)** Confirm `chorus setup --oracle alphagenome_pt`
   builds the conda env cleanly on Linux + the oracle loads + a single
   smoke prediction succeeds.
3. **(Strongly recommended)** Run the integration equivalence test we
   never executed on Mac:
   `pytest -m integration tests/test_alphagenome_backends_equivalence.py`
4. **(Bonus)** Measure speed numbers (PT CUDA vs JAX CPU vs JAX CUDA
   if the JAX env has CUDA support installed) at 128 kb / 524 kb / 1 MB
   to fill the "Linux + CUDA" row in the routing helper's audit.
5. **(Bonus)** Test the full 1 MB context — does CUDA show a cliff
   anywhere like Mac MPS does at 896 kb? If it scales smoothly to 1 MB,
   we may want to update the routing helper to recommend `alphagenome_pt`
   on CUDA without a window-size restriction.

## Setup

```bash
git clone https://github.com/pinellolab/chorus
cd chorus
git checkout feat/alphagenome-pytorch-backend  # current head: 138a10f or later
git log --oneline -5  # should include b82cfb6 (F2), 138a10f (F2 twin)

# Build base env if you don't have it
mamba env create -f environment.yml  # creates `chorus` env
mamba run -n chorus pip install -e .

# Build the new PT env — this is what we're testing CLI-wise
mamba run -n chorus chorus setup --oracle alphagenome_pt --hf-token "$HF_TOKEN"
# Or directly:
mamba env create -f environments/chorus-alphagenome_pt.yml
# Note: the env file currently has channels [conda-forge, bioconda] only —
# if conda-forge's pip-installable torch wheel doesn't pick up CUDA on your
# distro automatically, you may need to install torch from the pytorch
# channel explicitly. If that's the case, document the deviation in the
# audit you write.

# Build the JAX env if not already present
mamba env create -f environments/chorus-alphagenome.yml

# HF_TOKEN setup — JAX path needs gated google/alphagenome-all-folds;
# PT path is public (gtca/alphagenome_pytorch).
export HF_TOKEN=hf_...

# Reference genome
chorus genome download hg38   # or use an existing hg38.fa
```

## Step 1 — Smoke load + predict on CUDA

```bash
# Inside the chorus base env:
mamba run -n chorus python <<'PY'
import chorus
oracle = chorus.create_oracle("alphagenome_pt", use_environment=True, device="cuda",
                              reference_fasta="genomes/hg38.fa")
oracle.load_pretrained_model()
result = oracle.predict(("chr1", 109_274_968 - 100_000, 109_274_968 + 100_000))
tracks = dict(result.items())
print(f"Predicted {len(tracks)} tracks; sample mean: {next(iter(tracks.values())).values.mean():.4f}")
PY
```

Expected: clean load, finite outputs. If you see
`embedding(): argument 'indices' must be Tensor, not int`, F2 is back —
re-pull the branch.

If torch.cuda.is_available() returns False inside the
`chorus-alphagenome_pt` env, the conda-forge torch wheel probably
isn't CUDA-enabled. Fix by adding the `pytorch-cuda` package or
installing torch from the pytorch channel inside the env, then
retry.

## Step 2 — Tier 1 chorus-API equivalence

The audit scripts hardcode the macOS author's FASTA path. Either:

- (a) **Quick patch the path**: edit the four `run_*.py` files in
  `audits/2026-04-29_alphagenome_pt_stress_test/` and replace
  `/Users/lp698/Projects/chorus.bak-2026-04-28/genomes/hg38.fa` with
  your absolute hg38 path. Don't commit the path edits — they're a
  local override.
- (b) **Symlink** to make the macOS path resolve:
  `mkdir -p /Users/lp698/Projects/chorus.bak-2026-04-28/genomes &&
   ln -sf $(realpath your-hg38.fa) /Users/lp698/.../genomes/hg38.fa`
   (probably overkill on Linux).

Then run from the repo root:

```bash
mamba run -n chorus python audits/2026-04-29_alphagenome_pt_stress_test/run_chorus_api.py \
  2>&1 | tee audits/2026-04-29_alphagenome_pt_stress_test/api_run_linux.log
```

This calls `oracle._predict(seq, assay_ids)` for both backends on a
524 kb SORT1 window with one assay per head, then writes
`chorus_api_compare.json` next to the script with per-track relative
diffs.

**Pass criterion**: every assay reports
`mean_rel_diff < 0.02` and `sum_rel_diff < 0.02`. The macOS run got
1–2% per-assay; CUDA should be at least as tight (often tighter
because no CPU fallback paths).

## Step 3 — Tier 2 application-level region-swap

```bash
mamba run -n chorus python audits/2026-04-29_alphagenome_pt_stress_test/run_app_swap.py \
  2>&1 | tee audits/2026-04-29_alphagenome_pt_stress_test/app_swap_linux.log
```

Compares the v30 GATA1 region-swap pipeline (full 1 MB window, 6
tracks, 4 layer scores). Pass criterion: every track score within
**0.02 log₂** of the JAX baseline.

## Step 4 — Integration equivalence test

```bash
mamba run -n chorus pytest -m integration \
  tests/test_alphagenome_backends_equivalence.py -v
```

This is the test we never ran on Mac. Asserts
`max(|pt - jax|) < 0.1` and `mean rel diff < 5%` on DNase outputs at
the SORT1 window. Should pass cleanly on CUDA in ~5 min.

## Step 5 — CUDA speed sweep (bonus)

If you have time, run the same length sweep we ran on M3 Ultra to see
if CUDA shows any cliff:

```bash
mamba run -n chorus-alphagenome_pt python <<'PY'
import time, math, torch, numpy as np
import huggingface_hub
from alphagenome_pytorch import AlphaGenome

# Apply chorus rope-shim (no-op on CUDA but confirms env wiring)
import chorus.oracles.alphagenome_pt_source._mps_compat  # noqa

w = huggingface_hub.hf_hub_download("gtca/alphagenome_pytorch", "model_all_folds.safetensors")
device = torch.device("cuda")
model = AlphaGenome.from_pretrained(w, device=device); model.eval()
org = torch.tensor([0], dtype=torch.long, device=device)

print("| L (kb) | mean (s) | mem (MB) |")
print("|---:|---:|---:|")
for L in [65536, 131072, 262144, 524288, 786432, 1048576]:
    rng = np.random.default_rng(0)
    arr = np.zeros((L, 4), dtype=np.float32)
    arr[np.arange(L), rng.integers(0, 4, L)] = 1.0
    x = torch.from_numpy(arr).unsqueeze(0).to(device)
    # 1 warm + 3 timed
    with torch.no_grad():
        _ = model(x, organism_index=org, heads=("atac", "dnase"))
    torch.cuda.synchronize()
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x, organism_index=org, heads=("atac", "dnase"))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    mem = torch.cuda.memory_allocated() / 2**20
    print(f"| {L//1024} | {sum(times)/3:.2f} | {mem:.0f} |")
    del x; torch.cuda.empty_cache()
PY
```

Save the table somewhere reproducible. We expect smooth O(n) scaling
with no cliff at 1 MB — if so, we'll update the routing helper to
recommend `alphagenome_pt` on CUDA at every window size with high
confidence (currently medium).

## Step 6 — JAX CUDA comparison (only if jax[cuda] is set up)

If the chorus-alphagenome env's JAX has CUDA support installed (some
JAX builds do, some don't), repeat the same length sweep with the
JAX backend:

```bash
mamba run -n chorus-alphagenome python -c "
import jax; print('jax devices:', jax.devices())
"
```

If CUDA shows up there, bench JAX too. Otherwise skip — JAX CPU is
fine for the comparison reference.

## Writing up the verification

Add a section to `audits/2026-04-29_alphagenome_pt_stress_test/report.md`
called "Linux/CUDA verification (YYYY-MM-DD)" with:

- Hardware (GPU model, RAM, distro)
- Tier 1, Tier 2, integration test results (pass/fail + numbers)
- Speed table from Step 5
- Anything that broke during setup (env build, CUDA detection,
  FASTA path resolution, etc.)
- Recommendation: ready to flip to merge, or what's blocking

Commit the audit log files (api_run_linux.log etc.) and the report
update on a new branch (suggest `audit/linux-cuda-pr62`), then either
merge into `feat/alphagenome-pytorch-backend` directly or open a draft
PR to it. **Don't push to main.**

## What to do if something is broken

- Equivalence violated by >5%: capture the diffs, write up, and post
  a comment on PR #62. Don't merge. Most likely cause would be a JAX
  vs PT weight conversion drift or a bug in chorus's slicing.
- F2-style regression: re-pull the branch (the existing static-text
  test should have caught it; if it didn't, that's also a finding).
- Conda env build fails on Linux: the env file may need a
  `pytorch-cuda` add — note the diff and post it.
- 1 MB MPS-style cliff on CUDA: unexpected but possible. Document and
  measure where the cliff is so the routing helper can be tightened
  for CUDA hosts too.

## Branch state at handoff

```
138a10f fix(alphagenome_pt): F2 twin in _predict_direct + upstream PR draft
f5a263b test(alphagenome_pt): direct rope-shim equivalence vs upstream apply_rope
b82cfb6 fix(alphagenome_pt): F2 — predict_template was passing organism_index as int (crash); equivalence + 8.9× speed audit
325b86f feat: AlphaGenome backend-routing helper + function-mapping audit
57c80b3 diag: MPS rope patch + cliff root-cause investigation
2b69fd6 feat: AlphaGenome PyTorch backend (opt-in spike)
```

Fast suite at 138a10f: 358 passed, 1 skipped (env-gated rope-shim),
0 failed. Linux CI green at last push.
