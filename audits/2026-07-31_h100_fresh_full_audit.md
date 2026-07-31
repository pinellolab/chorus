# H100 fresh full audit — 2026-07-31

Fresh install from nothing on a Linux box with **8× H100 80GB HBM3**
(driver 595.71.05, CUDA 13.2, 208 cores, 1.8 TB RAM) — no conda, no
chorus, no caches on the machine at the start. Cloned `main` at #112
(`57f121d`, macOS Cherimoya adaptation), bootstrapped a toolchain, built
the base env + every per-oracle env that does not need credentials,
downloaded weights and hg38, ran the fast suite and the integration
suite, verified **Cherimoya on the real Triton/CUDA path** (this box is
the first place it can run at full speed), fixed the ChromBPNet
`exp`→`expm1` count-inversion bug behind PR #113, and started the
`chrombpnet_pertrack.npz` rebuild against the corrected transform.

Two things did not finish and are called out plainly below: the
**AlphaGenome pair is not installed** and the **rebuilt NPZ is not
uploaded**, both blocked on a HuggingFace write token that was not
usable in this environment.

## What was actually run

### Install — Miniforge had to be bootstrapped first

The README's install section assumes `mamba` exists. On this box nothing
did: no `mamba`, `conda`, `micromamba`, `python` (only `/usr/bin/python3`),
and no `hf`/`huggingface-cli`. Installed Miniforge3 to
`/home/nvidia/miniforge3` (mamba 2.5.0, conda 26.3.2) and pointed
`pkgs_dirs` at `/ephemeral/conda_pkgs` to keep the 193 GB root volume
clear.

```bash
bash Miniforge3-Linux-x86_64.sh -b -p /home/nvidia/miniforge3
mamba env create -f environment.yml          # clean, incl. the pinned coolbox git SHA
mamba run -n chorus python -m pip install -e .   # chorus 0.5.6
```

`chorus list` → **7 of 9 installed**, `chorus health` → all 7 Healthy:

| oracle | env | status |
|---|---|---|
| borzoi, cherimoya, chrombpnet, enformer, epinformerseq, legnet, sei | `chorus-<name>` | ✓ Installed / Healthy |
| alphagenome, alphagenome_pt | — | ✗ **not installed — HF token** |

Note the "9 oracles vs 8 models" question resolves cleanly: 9 rows
because AlphaGenome ships two backends (JAX + PyTorch); Cherimoya is the
8th distinct model. `tests/test_mcp.py:310-318` says the same.

hg38 downloaded and verified real: 3,273,481,150 B, 455 sequences,
`chr1` = 248,956,422 bp, `.fai` complete at 455 lines.

### Cherimoya — full CUDA/Triton path confirmed

The load-bearing check for this box. Probed inside `chorus-cherimoya`:

```
torch 2.13.0+cu130   cuda avail: True   H100 80GB HBM3   capability (9, 0)
triton 3.7.1
cherimoya.cheri.HAS_TRITON = True
```

So Linux/CUDA takes the Triton path, **not** the macOS CPU fallback that
#112 added. `tests/test_cherimoya_integration.py -m integration` →
**9 passed, 1 skipped** in 2:10, including both tests the brief flags as
load-bearing (`test_predict_matches_direct_window_scoring`,
`test_variant_effect_runs_end_to_end`) and
`test_counts_are_recovered_with_expm1`. The single skip is
`test_geometry_guard_rejects_a_mismatched_checkpoint` (needs a second,
deliberately-mismatched checkpoint).

### Tests

| suite | result |
|---|---|
| `pytest -m "not integration"` | **474 passed, 4 skipped, 1 error** |
| `tests/test_cherimoya_integration.py -m integration` | **9 passed, 1 skipped** |
| `tests/test_integration.py` + `test_error_recovery.py -m integration` | **3 passed, 1 skipped** |
| `tests/test_alphagenome_backends_equivalence.py` | **blocked** (no AlphaGenome env) |

The one fast-suite error is `TestSmokeAlphagenome::test_predict` —
needs `chorus-alphagenome`, which cannot be built without the token. The
two integration skips are both the same root cause:
`test_mcp_e2e_list_oracles_and_analyze_variant` skips on
`HF_TOKEN not set — AlphaGenome is gated` (`test_integration.py:173`).

Before the oracle envs existed the same fast suite gave 457 passed with
6 errors in `test_smoke_predict.py`; those 6 are purely an
envs-not-built artefact, so **run `chorus setup` before reading anything
into a smoke-test failure**. 474 = 457 + 5 smoke tests that now pass + 9
new tests added by PR #113, minus skip reshuffling.

### ChromBPNet `exp` → `expm1` (PR #113)

Confirmed the bug from first principles rather than taking it on faith.
ChromBPNet's count head is trained on `log(1 + count)` — upstream
`chrombpnet/training/data_generators/batchgen_generator.py` feeds
`np.log(1+batch_cts.sum(-1, keepdims=True))` — so the inverse is
`expm1`. Chorus used `np.exp` at exactly three sites; the profile-softmax
`np.exp` calls at `:577`, `:801`, `:347` are a different transform and
were left alone. Diff is three functional lines.

Measured on one forward pass (`DNASE:K562` fold 0, `chr11:5226000-5228114`),
computing both inversions from the same model heads:

```
count head (log1p space)       : 4.655128
total counts, exp   (old/buggy): 105.122625
total counts, expm1 (new/fixed): 104.122625
absolute difference            : 1.000000      <- exactly one count
ratio old/new                  : 1.009604
predicted (c+1)/c              : 1.009604      <- matches
501bp window sum old -> new    : 77.830338 -> 77.089966
window difference              : 0.740372  == the window softmax mass w
```

So the correction is exactly `-1` count on the total, i.e. `new = old - w`
on any window sum. Negligible at a peak, up to 100 % at a dead site —
which is the regime the activity CDFs are drawn from.

New `tests/test_chrombpnet_counts.py` (9 tests) was verified to **fail
9/9 against the pre-fix code and pass 9/9 after**, by reintroducing the
bug by hand and reverting. It includes a builder↔oracle consistency
assertion, so the CDF builder and the query path cannot silently drift
apart again — the failure mode `cherimoya_source/scoring.py` warns about.

### `chrombpnet_pertrack.npz` rebuild — scope pinned, in flight

Read the shipped file rather than guessing its scope
(sha256 `526beb2ce8310f6fdb331f766eac55ce3262b67f1a43416532d8bad8f83183eb`,
82,350,909 B): **786 tracks = 22 ATAC + 20 DNASE + 744 CHIP**, ordered
ATAC → DNASE → CHIP; three CDFs at `(786, 10000)`; `effect_counts=18672`,
`summary_counts=34004`, `perbin_counts=1088128`, uniform across all
tracks; `signed_flags` all False. That is the complete slim-mirror
catalogue (42 + 744, `chrombpnet.py:31-33`), so the build needs
`--assay all` — **not** the default `--assay ATAC_DNASE`.

Every sampling seed is a hardcoded constant in the builder (12345, 999,
42, 43, 44, 456, 789, 111, 567) with no `--seed` flag, so identical flags
are deterministic by construction. `NORMALIZATION_GUIDE.md:212-218`
independently documents the same targets (`18672` = 9,609 random SNPs +
9,063 DHS-proximal; `34004` = 15,000 random + 11,500 cCRE + 3,000 TSS,
29,004 usable, + 5,000 DHS summits).

**The shipped row order proves the original shard layout**: rows 0-41 are
the 42 ATAC/DNASE sequentially, then CHIP appears as even-then-odd
enumerator indices with exactly one parity flip. That is a 42-model
single-process build plus `--assay CHIP --shard {0,1} --shard-of 2`
merged onto it. This matters because `rng_bins =
np.random.RandomState(999)` is consumed once per baseline position
*inside* the per-model loop, so a track's perbin bin selection depends on
its index within its process — an 8-way shard would change perbin rows
for a reason **unrelated to expm1**. The rebuild therefore replicates the
original layout (1 ATAC/DNASE process + 2 CHIP shards) so `expm1` is the
only intended difference, at the cost of wall-clock.

ATAC/DNASE half is **done and validated** (42/42 models):

| check | result |
|---|---|
| rows / assays | 42 = 22 ATAC + 20 DNASE |
| track_id order vs shipped prefix | **identical** |
| `effect_counts` / `summary_counts` / `perbin_counts` | **18672 / 34004 / 1088128**, uniform |
| monotone rows | 42/42 in all three CDFs |
| all-zero rows | 0 (no silently-failed models) |
| `effect_cdfs` | min 0, **zero negatives** — matches shipped |
| `summary_cdfs` | 16 negatives, min −0.531 — **expected** |
| `perbin_cdfs` | 17 negatives, min −0.00242 — **expected** |
| `effect_cdfs` mean(new−old) | **+0.00049** (max \|Δ\| 0.138) → rankings stable |
| `summary_cdfs` mean(new−old) | **−0.512** ≈ the window softmax mass |
| `summary_cdfs` median old/new where old>1 | **1.0122** → counts corrected by (c+1)/c |

The negatives are correct, not a defect: a near-dead window gives
`log(count+1) < 0`, so `expm1 < 0`. `NORMALIZATION_GUIDE.md:220-224`
documents exactly this for Cherimoya and deliberately leaves it
unclamped so the builder and `oracle.predict()` agree. ChromBPNet now
inherits the same property and the guide has been updated to say so.

The 744 CHIP rows are still building on GPUs 5/6 (~0.6 min/model,
65/372 per shard at the time of writing, ETA ≈ 05:50). Remaining steps
are recorded in "Handed off" below.

### `AUDIT_CHECKLIST.md` — worked end to end

All 18 sections were attempted. Every reported failure and every claimed
stale gate was then independently re-run by a separate reconciliation
pass before being written down here, so nothing below is inherited
unverified. Coverage caveats: §7's selenium/Chrome method is not runnable
on a documented install (see below), §6 cannot execute notebooks as
written (kernelspec), and everything AlphaGenome-dependent is BLOCKED.

Verified green: determinism (§13) — all **7 installed oracles bitwise
identical** back-to-back, `max_abs_diff = 0`; offline (§15) — cached
oracles work under `HF_HUB_OFFLINE=1`, and `analyze_gene_expression`
resolves 6 GATA1 TSS from the local GENCODE v48 GTF; dependencies (§17) —
`pip-audit` on the base env reports **no known vulnerabilities** across
266 packages.

**26 checklist gates are themselves stale** and manufacture failures if
run verbatim. The most load-bearing:

| section | expects | actual |
|---|---|---|
| §4, §5, §8, §13 | "all 6 oracles" | **9 registered**, 7 installed; the loops skip cherimoya, epinformerseq, alphagenome_pt |
| §4, §10 | ChromBPNet "24 CDFs per-model" | **786 per-track** (predates the per-track scheme) |
| §8 | "exactly 6 oracles", "exactly 22 tools" | **9 oracles**, **24 tools** |
| §11 | "≥ 334 pass, ≤ 1 skip" | **469 pass, 4 skip** in the CI selection; `≤1 skip` is unreachable now |
| §11 | its fast-suite command | collects 488 incl. 15 integration tests — needs `-m "not integration"`, which CI does use |
| §4/§10 | AlphaGenome "5,168" (§88) vs "5,731" (§183) | the checklist contradicts itself |
| §7 | `Ref%ile` column, "Glossary", "collapsible" | column is `Effect %ile`; block is `How to read this report` and is **not** collapsible |
| §14 | `extract_sequence` returns lowercase; `.upper()` at `core/base.py:325` | it upper-cases (`utils/sequence.py:135`); the check moved to `:460` |
| §15 | `oracle.analyze_gene_expression('GATA1')` | wrong arity — needs `(predictions, gene_name)` |
| §1 | `CHORUS_DOWNLOAD_DIR` override | not implemented anywhere |
| §2, §9 | two `test_error_recovery.py` node IDs | both renamed (tests exist and pass) |
| §12 | regen output "byte-identical" | impossible — every regen stamps a fresh UTC timestamp |

## New P0 / P1 findings, unrelated to the `expm1` work

These came out of the checklist pass. **None is caused by PR #113** and
none is fixed here; each needs its own decision. Listing the ones that
change what a user gets:

**P0 — Enformer silently returns track index 0 for any non-ENCFF track
id** (`enformer.py:413`). `_get_assay_indices(['CNhs10608']) -> [0]`
while `get_metadata().get_track_by_identifier('CNhs10608') -> 4675`, and
`_validate_assay_ids` accepts the id. That is all **638 FANTOM CAGE
`CNhs*` tracks** silently returning the wrong track's signal in the
default `use_environment=False` path.

**P0 — Borzoi variant scores silently come back `None`**
(`borzoi.py:217`). `prediction_interval` is built with
`query_interval.extend(self.output_size)`, but `Interval.extend` is a
no-op when the requested length is smaller than the interval, leaving an
interval 2.67× wider than the values array (`n_values 6144 × 32 bp =
196,608` covered vs `pred_interval len 524,288`). Reproduced:
`score_variant_effect -> {'ref_score': None, 'alt_score': None,
'effect': None}`.

**P0 — Borzoi is completely broken in direct mode** (`borzoi.py:302`),
which is the `create_oracle` default: `AttributeError: 'numpy.ndarray'
object has no attribute 'numpy'`.

Both Borzoi P0s and the Enformer P0 survive green CI because there is
**zero fast-suite coverage of the direct (`use_environment=False`)
predict path** for either oracle (`tests/test_prediction_methods.py:465`).

**P1 — effect percentiles are inflated ~4.1 %**
(`normalization.py:456`). `_get_denominator` divides by the raw sample
count (9,606) while the CDF grid is interpolated to full width (10,000),
so the value at grid index 5000 reports the 0.5206 quantile (ratio
1.0412) and the top ~3.9 % is pinned to the 100th percentile. This is
adjacent to the CDF work in this audit and worth fixing in the same
neighbourhood.

**P1 — Enformer and ChromBPNet are silently CPU-only when their env is
used directly** (`runner.py:99`). The CUDA libs live in `nvidia-*-cu11`
wheels and nothing puts them on the loader path except
`EnvironmentRunner._prepare_env`, which only applies to subprocesses it
spawns. So `mamba run -n chorus-enformer python …` — the exact
invocation `CLAUDE.md` tells users and agents to use — reports
`tf devs: ['/physical_device:CPU:0']`. I hit this myself: the
count-inversion spot-check above printed "Skipping registering GPU
devices" and ran on CPU. On an 8×H100 box that is silent CPU inference
with no warning. An `activate.d` hook would fix it on every path.

**P1 — `device='cuda:N'` escapes the caller's `CUDA_VISIBLE_DEVICES`
mask** (`chrombpnet_source/templates/load_template.py:31`, and
cherimoya's at `:22`). The templates assign
`os.environ['CUDA_VISIBLE_DEVICES'] = device.split(':')[1]`, so the bare
ordinal *replaces* the outer mask: with outer `CUDA_VISIBLE_DEVICES=3`,
the allocation landed on physical GPU 1. On a shared or SLURM node a
user passing `device='cuda:1'` runs on a GPU they were not granted. The
PyTorch oracles do it correctly. Separately, **Cherimoya
`device='cuda:N'` for N ≥ 1 is unconditionally broken** — it masks the
device list then passes the original `'cuda:1'` to torch:
`Attempting to deserialize object on CUDA device 1 but
torch.cuda.device_count() is 1`.

**P1 — MCP drift**: `html_report_path` returns the **entire HTML
document** instead of a path (`server.py:1136`); `load_oracle('legnet',
cell_type=…)` raises `TypeError` although the tool docstring and
`list_tracks` both tell Claude to pass it (`state.py:79`); 24 tools are
registered but 22 advertised everywhere, with `score_ism` undocumented
(`server.py:1939`); the system prompt says "7 oracles" and never
mentions Cherimoya (`server.py:28`).

**P1 — notebooks are unrunnable as documented**: every notebook declares
kernelspec `chorus`, which no setup step registers, so the documented
`nbconvert` command fails with `NoSuchKernel` (16 notebooks repo-wide).

**P1 — edge cases**: `predict_variant_effect` crashes for any variant
within half a model window of a chromosome **end**
(`InvalidRegionError: End position 248956978 exceeds chromosome chr1
length 248956422`); insertions crash with a raw numpy broadcast error
*after* the model has already run (`operands could not be broadcast
together with shapes (2116,) (2114,)`).

**P1 — attribution/licensing**: `docs/THIRD_PARTY.md` still says "six
oracles" and has no Cherimoya/CATv1 or EPInformer-seq entry, though
Cherimoya ships third-party CC-BY-4.0 weights and the
`cherimoya==0.2.0` package; bundled `igv.min.js` has no upstream license
header and no adjacent NOTICE.

**P1 — `CHANGELOG.md` has no entries for any 0.5.x release** although the
package declares 0.5.6.

**P2 (abbreviated)**: report tables duplicate the cell type in 106/111
rows; two walkthrough READMEs publish numbers no artefact in their own
directory reproduces; a committed notebook output ships the
ref-allele-mismatch WARNING that §5 designates a P0 regression
signature; `chorus-legnet` pins CUDA-11.8 torch with no sm_90 kernels so
every H100 LegNet call warns; `setup.py` `data_files` misses three env
specs; selenium/Chrome are prescribed by §7 but declared nowhere;
`alleles=[]` raises a bare `IndexError`; missing-genome errors never
mention `chorus genome download hg38`.

Security note: the base env is clean, but `chorus-chrombpnet`'s
TensorFlow 2.8.0 carries 25+ advisories and `chorus-enformer`'s Keras
2.13.1 carries 16 (including the Keras deserialization-RCE class, which
matters because chorus loads `.h5`/`.keras` artefacts fetched from
ENCODE/HF). Both pins are deliberate and documented; this is a
known-risk acceptance item, not a quick fix.

## Fixed in this PR (#113)

1. **`np.exp` → `np.expm1`** at the three count-inversion sites
   (`oracles/chrombpnet.py:579`, `:802`,
   `scripts/build_backgrounds_chrombpnet.py:348`).
2. **`tests/test_chrombpnet_counts.py`** — 9 regression tests incl. a
   builder↔oracle consistency guard.
3. **Stale documentation that asserted the buggy behaviour**:
   `cherimoya_source/scoring.py` (claimed chorus "recovers its counts
   with `np.exp` in three places", and cited `:800` where the count line
   is `:802`); `docs/NORMALIZATION_GUIDE.md:203-210`;
   `build_backgrounds_chrombpnet.py:327` (`predict_profiles_batch`
   docstring still said `softmax × exp(counts)`);
   `tests/test_cherimoya.py:226`.
4. **`README.md:1228`** — the ChromBPNet CDF sample sizes were listed as
   10,000 / 31,500; the shipped NPZ carries 18,672 / 34,004.
5. **`CHANGELOG.md`** — `[Unreleased] / Fixed` entry.

## Findings NOT fixed here

### P0 — `merge_shards` silently discards a full rebuild

`build_backgrounds_chrombpnet.py:789-811` appends onto any existing
`~/.chorus/backgrounds/chrombpnet_pertrack.npz` and de-dups by
`track_id`, keeping only ids **not** already present. A full 786-track
rebuild collides on every id, so `keep_mask` is all-False, all 786
rebuilt rows are dropped, and it logs the reassuring
`DONE — merged NPZ has 786 tracks (786 existing + 0 new from 8 shards)`.
Since `get_pertrack_normalizer('chrombpnet')` auto-downloads the shipped
NPZ into exactly that path — triggered by the examples, `pytest -m
integration`, any walkthrough regeneration, or an MCP `load_oracle` —
the natural failure mode is **re-uploading the old file believing it is
the fix**. Mitigated here by letting the ATAC/DNASE `--part both` merge
overwrite the path first, and by gating on sha256 ≠ `526beb2c…`.

### P0 — silent partial builds exit 0

Model-load failures (`:524`) and per-batch failures (`:554`, `:581`) are
caught, logged at WARNING, and skipped; the process still exits 0 and the
merge writes those rows with `counts == 0` and all-zero CDFs.
`_has_samples` then makes every lookup return `None`, so percentiles
silently vanish rather than erroring. Any rebuild must gate on
`counts == 18672/34004/1088128` for all 786 rows — a green exit code
means nothing.

### P1 — `chorus setup` hard-gates all 9 oracles on the AlphaGenome token

`_setup_all.py:71-82` resolves the HF token before any env build and
halts with "Nothing was downloaded" if it fails. The intent is good (do
not burn 10+ GB then fail), but 7 of the 9 oracles need no credential at
all, so a tokenless user gets **zero** oracles from the documented
command. Per-oracle `chorus setup --oracle <name>` bypasses it, which is
how this box was provisioned. Worth a `--skip-gated` or an
"install what we can" path.

### P1 — concurrent per-oracle setup collides on GPU 0 and on hg38

The weight-prefetch step actually loads each model, and TensorFlow claims
the whole device: with 7 setups running, GPU 0 sat at 79,551/81,559 MiB
and enformer/sei/epinformerseq failed with
`Dst tensor is not initialized` / `Could not create cudnn handle`. All
three succeeded immediately when pinned to distinct GPUs with
`CUDA_VISIBLE_DEVICES`. Separately, they race on the shared
`genomes/hg38.fa` download (legnet failed once, succeeded on retry).
Either serialise the prefetch or document the pinning.

### P1 — `--setup-timeout` is silently ignored by bare `chorus setup`

Forwarded in the single-oracle path (`main.py:85-86`, `:109`) but not in
the all-oracles path (`_setup_all.py:96`, `:106-113`), so the flag looks
accepted and does nothing.

### P1 — the 744 CHIP CDF rows measure a different quantity than `predict()`

Pre-existing and **not** an `expm1` issue, but it undercuts any claim
that this rebuild restores builder↔query consistency for the whole file.
The builder collapses the two strands *before* the softmax
(`probabilities.sum(axis=-1)` on logits at `:341`, softmax at `:347`),
while the oracle softmaxes each strand separately and emits
`CHIP:{cell}:{TF}:+` / `:-`, each carrying the full count mass
(`chrombpnet.py:659-667`); the normalizer maps both strands onto the one
CDF row via the `rsplit(':', 1)` fallback
(`normalization.py:500-501`). Summing logits is a geometric mean of the
two strand profiles. So consistency is restored for the 42 ATAC/DNASE
rows only. **Decide before the next rebuild** — fixing it later costs
another full-catalogue build.

### P2 — assorted

- **`scripts/run_bpnet_cdf_build.sh` is unusable as shipped**: ssh's to
  hardcoded hosts `ml003/ml007/ml008` with `--shard-of 6`, and covers
  only `--assay CHIP` (not the 42 ATAC/DNASE).
- **Shard log collision**: every process opens
  `logs/bg_chrombpnet_{part}.log` with `mode='w'` (`:116`) — no shard
  index — so parallel shards truncate each other. Redirect stdout
  per-shard instead.
- **`audits/2026-04-29_chrombpnet_cdf_rebuild/HANDOFF.md:96-99, :134`**
  claims phase 2 is `--force`-gated and that the inner `--gpu 0` is a
  no-op when the env var is set. Neither is true: there is no `--force`
  in the parser, both interim writes are bare `np.savez_compressed`, and
  `:233` sets `CUDA_VISIBLE_DEVICES` unconditionally.
- **TF 2.8 in `chorus-chrombpnet` has no sm_90 cubins**
  (`cuda_compute_capabilities` stops at `compute_80`), so every process
  JIT-compiles from PTX with a "could take 30 minutes" warning. It does
  work (cuDNN 8906 loads, ~0.6 min/model). Also: a bare
  `python -c "import tensorflow"` in that env sees **no GPU** — the
  device only appears because the builder ctypes-preloads
  `site-packages/nvidia/*/lib/*.so*` (`:236`). Any hand-rolled
  verification script must do the same or it will falsely report CPU.
- **EPInformer-seq has the same class of bug**, unfixed:
  `10**log_count` in `epinformerseq_source/model_usage.py:110`.
- **cCRE source is unpinned**:
  `https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed`
  (`annotations.py:629`) with no checksum, and `_DHS_VOCAB_SHA256` is
  declared but never used. Verified undrifted today only because the
  rebuild reproduced 18,672/34,004/1,088,128 exactly.

### `AUDIT_CHECKLIST.md` is itself stale

Running it verbatim as a gate manufactures failures. Measured against
this box:

| section | checklist expects | actual |
|---|---|---|
| §4 CDFs | ChromBPNet "24 CDFs" | **786** |
| §8 MCP | "exactly 6 oracles", "22 tools" | **9 oracles**, **24 tools** |
| §11 Tests | "≥ 334 pass, ≤ 1 skip" | **474 pass, 4 skip** |
| §4 | cites `core/base.py:325` for a `.upper()` | lives at `:460` |
| §11 | four pytest node IDs | do not exist |
| CI | `.github/workflows/tests.yml:3` "303-test" | ~469 in the CI selection |

## Delivered

- 8×H100 fresh install: base env + 7 oracle envs, all Healthy; hg38 and
  all non-gated weights + backgrounds cached.
- Cherimoya verified on the real Triton/CUDA path with integration green.
- PR **#113** — the `expm1` fix, 9 regression tests, and the docs that
  asserted the old behaviour. **Open, not merged** (see below).
- ATAC/DNASE half of the corrected `chrombpnet_pertrack.npz`, validated
  against every acceptance criterion.
- This report.

## Handed off / blocked

1. **PR #113 is deliberately not merged.** The brief assigns this item
   tentatively to Lorenzo and says to confirm with Luca first; no
   competing PR exists (latest was #112), but the merge decision is not
   mine to make.
2. **HuggingFace write token unusable in this environment**, which blocks
   (a) `chorus-alphagenome` + `chorus-alphagenome_pt`, (b) the
   `AlphaGenome` smoke/MCP/backend-equivalence tests, and (c) **the
   upload of the rebuilt NPZ**. Everything else was routed around it —
   the backgrounds dataset repo is public and needed no credential.
3. **Finish the rebuild** (CHIP shards ETA ≈ 05:50), then, in this order:

   ```bash
   # 1. append the 744 CHIP rows onto the 42-row base -> 786
   mamba run -n chorus python scripts/build_backgrounds_chrombpnet.py --part merge-shards
   # 2. gate BEFORE uploading anything
   mamba run -n chorus python /tmp/validate_rebuild.py     # all checks must pass
   # 3. upload (needs a write token)
   hf auth whoami            # must say lucapinello
   hf upload lucapinello/chorus-backgrounds \
       ~/.chorus/backgrounds/chrombpnet_pertrack.npz chrombpnet_pertrack.npz \
       --repo-type dataset \
       --commit-message "ChromBPNet CDFs rebuilt with expm1 count inversion"
   ```

   Reference copy of the pre-rebuild file for before/after work:
   `/ephemeral/chorus_rebuild_ref/chrombpnet_pertrack.SHIPPED.npz`.
4. **Phase 6 regeneration is larger than "regenerate affected examples"**
   and needs scoping: 22 files under `examples/` reference ChromBPNet;
   two shipped notebook cells still contain the bug itself
   (`klf1_validated_enhancer_profiles.ipynb:684`, `:797`) and need
   editing plus GPU re-execution; there are four regeneration entry
   points, not two; no `*_variant_report.pkl` is committed so
   `regenerate_multioracle.py --consolidate` degrades to "JSON only (no
   IGV predictions)" unless all three oracles re-run — and AlphaGenome
   is not installed here. Also
   `examples/walkthroughs/variant_analysis/SORT1_chrombpnet/example_output.md`
   is **already** stale from the 2026-06-17 windowing fix; do not
   attribute that drift to `expm1`.
