# Scorched-earth fresh-install audit — 2026-05-09

**Branch**: `fix/post-v040-followups` HEAD `0cd9f2f`
**Host**: macOS 15.7.4, Apple M3 Ultra (96 GB), `mamba` from miniforge
**Goal**: Validate the README's documented install path **as a brand-new
user**, with all chorus envs / caches / weights / genome / annotations
wiped before starting. Catch first-run bugs that warm-state pytest
sweeps miss (per CLAUDE.md memory: "v27 caught a P0 that 5 warm audits
missed").

This is the audit Lorenzo will see referenced from the PR description.

## Pre-state (clean slate)

```
mamba env list | grep chorus    →  (empty)
ls ~/.chorus/                    →  (no such dir)
ls genomes/ annotations/         →  (empty)
git status --short               →  (clean)
git log -1 --oneline             →  0cd9f2f docs(audit): close the DHS-CDF deferral...
```

Wiped via `chorus cleanup --all` (deletes 7 oracle envs + weights +
backgrounds + genomes), then `mamba env remove -n chorus -y` to drop
the base env too, then `rm -rf ~/.chorus/ annotations/*` for the
residue. `chorus` CLI no longer in PATH from the chorus env (only
the leftover one in miniforge base).

## Step 1 — README "Install (5 minutes)"

```bash
git clone https://github.com/pinellolab/chorus.git && cd chorus       # already cloned
mamba env create -f environment.yml
mamba activate chorus
python -m pip install -e .
```

| Sub-step | Wall-clock | Result |
|---|---|---|
| `mamba env create -f environment.yml` | **1m 52s** | ✓ env built |
| `python -m pip install -e .` | **2s** | ✓ chorus 0.4.0 installed |
| `chorus --version` | < 1s | ✓ `chorus 0.4.0` |

README claim: "5 minutes". **Actual: ~2 min on M3 Ultra with warm
conda package cache.** Cold conda cache would be slower (network
download dominates).

## Step 2 — README "Get every oracle, weight, and reference"

```bash
chorus setup --hf-token <hf token with read access>
```

(README also mentions LDlink token — optional, skipped here.)

**Wall-clock: 18 min 35 s** — README claim "55–75 min" is
conservative; this Mac with a warm conda package cache + good network
came in well under. Real first-time-on-this-host might be 2–3× slower
to download from conda-forge, but still much less than the README
upper bound.

Per-oracle phases captured from `logs/fresh_install_setup.log`:

| Oracle | env build | weights | backgrounds | total |
|---|---|---|---|---|
| alphagenome   | 44s | 1m 26s | (incl. genome ~10 min) | 11m 56s |
| alphagenome_pt| 39s | 22s    | 0s (404 → alias to alphagenome) ✓  | 1m 0s |
| borzoi        | 37s | 15s    | 10s | 1m 2s |
| chrombpnet    | 33s | 31s    | 5s ← **the new uniform-DHS NPZ, 78.5 MB** | 1m 9s |
| enformer      | 32s | 24s    | 11s | 1m 7s |
| legnet        | 30s | 24s    | 1s  | 0m 55s |
| sei           | 31s | 28s    | 1s  | 1m 0s |
| **TOTAL**     | — | — | — | **18m 35s** |

### chrombpnet NPZ round-trip ✓

The big artifact in this audit: the new uniform-DHS-augmented
chrombpnet NPZ uploaded today (sha
`526beb2ce8310f6fdb331f766eac55ce3262b67f1a43416532d8bad8f83183eb`,
78.5 MB).  The fresh `chorus setup` auto-pulled it from
`huggingface.co/datasets/lucapinello/chorus-backgrounds`:

```
$ shasum -a 256 ~/.chorus/backgrounds/chrombpnet_pertrack.npz
526beb2ce8310f6fdb331f766eac55ce3262b67f1a43416532d8bad8f83183eb
```

Matches the upload sha. **First-install round-trip: confirmed.**

### Disk footprint after setup

```
chorus envs:       ~11 GB (8 envs)
downloads:          6.4 GB (oracle weights)
~/.chorus:          1.6 GB (CDFs)
~/.cache/huggingface: 8.7 GB (HF download cache + alphagenome weights)
hg38 genome:       ~3 GB
```

README claim: "~28 GB free disk". Actual: ~32 GB end-state, includes
HF cache duplication. README understates by ~10–15% — defensible if
you assume `~/.cache/huggingface` evictions.

## Step 3 — README "Your first prediction" (β-globin SNP via Enformer)

```python
import chorus
from chorus.utils import get_genome

oracle = chorus.create_oracle(
    'enformer', use_environment=True,
    reference_fasta=str(get_genome('hg38')),
)
oracle.load_pretrained_model()

wt = oracle.predict(('chr11', 5247000, 5248000), ['ENCFF413AHU'])
print(f"WT mean signal: {wt['ENCFF413AHU'].values.mean():.3f}")
# → WT mean signal: 0.468

effects = oracle.predict_variant_effect(
    'chr11:5247000-5248000', 'chr11:5247500',
    ['C', 'A', 'G', 'T'], ['ENCFF413AHU'],
)
n_alts = len(effects['predictions']) - 1
print(f"Variant result: scored {n_alts} alt alleles "
      f"({list(effects['predictions'].keys())})")
# → Variant result: scored 3 alt alleles (['reference', 'alt_1', 'alt_2', 'alt_3'])
```

**Wall-clock**: 49 s end-to-end (model load + prefetch metadata +
WT predict + 3-alt variant scan).  Output matches README exactly.

## Tests (cold)

```
mamba run -n chorus pytest tests/ -q -m "not integration"
```

**376 passed, 1 skipped, 5 deselected — 5m 39s wall.**  Same result as
the warm sweep yesterday.  No new failures introduced by the
fresh-install path.

## End-to-end walkthroughs (cold)

The canonical SORT1 multi-oracle pipeline, run cold against the
freshly-installed envs:

```
mamba run -n chorus-chrombpnet  python scripts/regenerate_multioracle.py --oracle chrombpnet
mamba run -n chorus-legnet      python scripts/regenerate_multioracle.py --oracle legnet
mamba run -n chorus-alphagenome python scripts/regenerate_multioracle.py --oracle alphagenome
mamba run -n chorus-alphagenome python scripts/regenerate_multioracle.py --consolidate
```

**Wall-clock: 6m 52s.**  `example_output.md` cross-oracle consensus
table:

| Layer | chrombpnet | legnet | alphagenome | Agreement |
|---|---|---|---|---|
| Chromatin accessibility (log2FC) | +1.241 · DNASE:HepG2 | — | +1.336 · DNASE:HepG2 | all ↑ |
| Promoter activity (Δ alt−ref)    | — | +0.000 · LentiMPRA:HepG2 | — | only ↑ (n=1) |
| TF binding (log2FC)              | — | — | +2.777 · CHIP:CEBPA:HepG2 | only ↑ (n=1) |
| Histone modifications (log2FC)   | — | — | +1.267 · CHIP:H3K27ac:HepG2 | only ↑ (n=1) |
| TSS activity (log2FC)            | — | — | +1.522 · CAGE:HepG2 | only ↑ (n=1) |

ChromBPNet and AlphaGenome agree on direction (both ↑ chromatin
accessibility), magnitudes within 8% of each other.  This is the
expected cell-line-correct result for the rs12740374 SORT1 enhancer
in HepG2.

### All 18 walkthrough HTMLs IGV-inspected

```
walkthrough                                                  panels with_data verdict
batch_scoring/batch_sort1_locus_scoring.html                      -         -  no-IGV
causal_prioritization/SORT1_locus/...locus_causal_report.html    18        18      OK
discovery/SORT1_cell_type_screen/..._HepG2_report.html           30        30      OK
discovery/SORT1_cell_type_screen/..._MCF_10A_report.html          6         6      OK
discovery/SORT1_cell_type_screen/..._left_lobe_of_liver.html      6         6      OK
sequence_engineering/integration_simulation/...                   3         3      OK
sequence_engineering/region_swap/...                              4         4      OK
validation/SORT1_rs12740374_multioracle/..._alphagenome.html      5         5      OK
validation/SORT1_rs12740374_multioracle/..._chrombpnet.html       1         1      OK
validation/SORT1_rs12740374_multioracle/..._legnet.html           1         1      OK
validation/SORT1_rs12740374_multioracle/..._multioracle.html      7         7      OK
validation/SORT1_rs12740374_with_CEBP/..._CEBP_validation.html    6         6      OK
validation/TERT_chr5_1295046/..._TERT_alphagenome.html           18        18      OK
variant_analysis/BCL11A_rs1427407/..._BCL11A_alphagenome.html     6         6      OK
variant_analysis/FTO_rs1421085/..._FTO_alphagenome.html           6         6      OK
variant_analysis/SORT1_chrombpnet/..._chrombpnet_report.html      1         1      OK
variant_analysis/SORT1_enformer/..._enformer_report.html         48        48      OK
variant_analysis/SORT1_rs12740374/..._alphagenome.html            6         6      OK

0 need attention
```

## Friction / surprises / fixes

**(none P0/P1)**.  Three minor notes for future doc polish:

1. **README "5 minutes" install claim** is conservative-ish — actual
   ~2 min on this hardware with warm conda cache.  No fix needed
   (the upper bound is the right one to advertise).
2. **README "55–75 min" setup claim** also conservative — actual
   ~19 min here.  Variance dominated by network throughput on the
   alphagenome weights + hg38 genome.  No fix.
3. **README "~28 GB free disk"** slightly optimistic — actual ~32 GB
   end-state because the HF cache duplicates the alphagenome
   weights.  Could clarify "~28 GB plus ~8 GB transient HF cache".
   Low priority.

No P0/P1 install bugs.  Every README command worked verbatim.  Every
oracle env created cleanly.  Every weight + CDF + genome + GENCODE
download succeeded on the first try.

## Sign-off

Branch `fix/post-v040-followups` (HEAD `0cd9f2f`) is **safe to merge
into `main`** based on this scorched-earth audit:

- ✅ Fresh-install path works exactly as documented (Steps 1–3)
- ✅ HF auto-fetch works for the new uniform-DHS chrombpnet NPZ
  (sha-verified end-to-end)
- ✅ HF auto-fetch works for the DHS vocabulary
  (separate test earlier today)
- ✅ 376 tests pass cold
- ✅ All 18 walkthrough HTMLs render with peaks
- ✅ Multi-oracle SORT1 produces sensible cross-oracle consensus
  (chrombpnet + alphagenome agree on direction, magnitudes within 8%)

Recommended next step: open the merge PR
`fix/post-v040-followups` → `main`, link this audit doc + the two
preceding ones (2026-05-08, 2026-05-09 DHS rebuild), and close
Lorenzo's PR #79 with a comment pointing at our merge commit.
