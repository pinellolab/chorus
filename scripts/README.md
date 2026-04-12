# Per-Track Background Distribution Scripts

These scripts build the **per-track CDFs** consumed by `PerTrackNormalizer`
for variant effect interpretation and IGV visualization. Each oracle gets a
single `~/.chorus/backgrounds/{oracle}_pertrack.npz` file containing three
CDF matrices:

| CDF | Shape | Used for |
|-----|-------|----------|
| `effect_cdfs` | `(n_tracks, 10000)` | Variant effect %ile (table column) |
| `summary_cdfs` | `(n_tracks, 10000)` | Activity %ile (table column) |
| `perbin_cdfs` | `(n_tracks, 10000)` | IGV per-bin visualization |

For scalar-output oracles (Sei, LegNet), only `effect_cdfs` and
`summary_cdfs` are stored — `perbin_cdfs` is omitted.

## Quick start

```bash
# Build all backgrounds for one oracle (multi-GPU recommended)
mamba run -n chorus-enformer python scripts/build_backgrounds_enformer.py --part variants  --gpu 0
mamba run -n chorus-enformer python scripts/build_backgrounds_enformer.py --part baselines --gpu 1
mamba run -n chorus              python scripts/build_backgrounds_enformer.py --part merge
```

The `--part variants` and `--part baselines` steps run in parallel on
separate GPUs, then `--part merge` (no GPU needed) combines the interim
files into the final `enformer_pertrack.npz`.

Use `--part all` to run everything sequentially on a single GPU.

## Available scripts

| Script | Oracle | Tracks | Env | Runtime (single GPU) |
|--------|--------|--------|-----|---------------------|
| `build_backgrounds_enformer.py` | Enformer | 5,313 | `chorus-enformer` | ~4h variants + ~7h baselines |
| `build_backgrounds_borzoi.py` | Borzoi | 7,612 | `chorus-borzoi` | ~4h + ~9h |
| `build_backgrounds_alphagenome.py` | AlphaGenome | ~5,168 | `chorus-alphagenome` | ~10h + ~12h |
| `build_backgrounds_chrombpnet.py` | ChromBPNet | 24 (per-model) | `chorus-chrombpnet` | ~25 min total |
| `build_backgrounds_sei.py` | Sei | 40 classes | `chorus-sei` | ~6h + ~8h |
| `build_backgrounds_legnet.py` | LegNet | 3 cell types | `chorus-legnet` | ~16h (CPU) |

ChromBPNet's "tracks" are individual cell-type-specific models. Sei's
"tracks" are regulatory classes. LegNet's "tracks" are MPRA cell types.

## Other public scripts

| Script | Purpose |
|--------|---------|
| `regenerate_examples.py` | Regenerate all variant_analysis/validation example outputs (AlphaGenome, Enformer, ChromBPNet). Run inside the matching conda env; model loads once. |
| `regenerate_remaining_examples.py` | Regenerate discovery, causal, batch, sequence_engineering examples. Run in chorus-alphagenome env. |
| `screenshot_report.py` | Take a full-page screenshot of an HTML variant report using headless Chrome (requires selenium). |

Internal/maintenance scripts (not for end users) live in `scripts/internal/`.

## How positions are sampled

The build scripts use a shared sampling strategy that approximates the
genome-wide distribution. Each baseline build uses ~31,500 positions:

| Position type | Count | Purpose |
|--------------|-------|---------|
| Random intergenic | 15,000 | Genome-wide null (most genome is silent) |
| SCREEN cCREs (per-category) | ~11,500 | PLS, dELS, pELS, CA-CTCF, CA-TF, TF, CA-H3K4me3, CA |
| Protein-coding TSSs | 3,000 | Sharp signals: CAGE, H3K4me3, promoter activity |
| Gene body midpoints (>10kb genes) | 2,000 | RNA-seq, H3K36me3, broad gene-body marks |

For the **perbin CDF**, each position contributes 32 random bins from the
full output window of the prediction. This captures the full per-bin
distribution at each track's native resolution.

**CAGE summary routing**: CAGE tracks skip cCRE positions for the
summary CDF (since CAGE biology lives at TSSs, not regulatory elements).

**RNA-seq exon-precise sampling** (Borzoi, AlphaGenome): RNA tracks only
collect bins overlapping protein-coding exons (loaded from GENCODE v48
basic). Implementation uses a per-chromosome merged exon interval list.

## Variant effect sampling

10,000 random SNPs across chr1–22, well away from chromosome edges. For
each SNP:
1. Predict ref + alt allele full output
2. For each track, score the variant effect using its layer formula
   (`log2fc` for unsigned signals, `logfc`/`diff` for signed layers)
3. Add abs effect (unsigned) or raw effect (signed) to the track's
   reservoir

## Output file structure

```
~/.chorus/backgrounds/
  enformer_pertrack.npz       # 5,313 tracks, ~550 MB
  borzoi_pertrack.npz         # 7,612 tracks
  alphagenome_pertrack.npz    # ~5,168 tracks
  chrombpnet_pertrack.npz     # 24 models, 2.4 MB
  sei_pertrack.npz            # 40 classes
  legnet_pertrack.npz         # 3 cell types
```

Each `.npz` contains:
- `track_ids`: Unicode array, one identifier per track row
- `effect_cdfs`: `(n_tracks, 10000)` float32 — sorted variant effects per track
- `summary_cdfs`: `(n_tracks, 10000)` float32 — sorted window-sum signals per track
- `perbin_cdfs`: `(n_tracks, 10000)` float32 — sorted per-bin values per track (omitted for scalar oracles)
- `effect_counts`, `summary_counts`, `perbin_counts`: actual sample counts per track
- `signed_flags`: bool array — True for signed layers (RNA, MPRA, Sei)

## Reservoir sampling

To keep memory bounded, each track's data is collected via Algorithm R
reservoir sampling with capacity 50,000. After collection, the reservoir
is compacted to 10,000 evenly-spaced percentile points for storage.

## Auto-discovery

After building, files are auto-discovered by:

```python
from chorus.analysis.normalization import get_pertrack_normalizer

norm = get_pertrack_normalizer("enformer")
# norm is a PerTrackNormalizer ready for use in variant analysis
```

Or via the MCP state manager (used by chorus MCP server tools):

```python
from chorus.mcp.state import OracleStateManager
state = OracleStateManager()
state._auto_load_normalizer("enformer")
norm = state.get_normalizer("enformer")  # PerTrackNormalizer
```

## Reproducibility

All scripts use the same random seeds:
- Seed 42: random SNP generation
- Seed 456: cCRE category sampling
- Seed 789: random intergenic positions
- Seed 111: TSS sampling
- Seed 222: gene body midpoint sampling
- Seed 999 (numpy): per-position random bin selection
- Seed 12345: reservoir sampler tie-breaking

This ensures the same genomic positions are scored across all oracles
(modulo each oracle's input window size).

## Adding a new oracle

To build per-track CDFs for a new oracle:

1. Copy the closest existing script as a template (Borzoi for multi-track,
   ChromBPNet for per-model, Sei for scalar-output)
2. Update model loading and prediction to match the new oracle's API
3. Build a `track_info` list with `{idx, identifier, layer, window, formula,
   pseudocount, signed}` per track
4. Use `ReservoirSampler` for memory-bounded collection
5. Save interim NPZ files for variants and baselines separately
6. The merge step calls `PerTrackNormalizer.build_and_save()` to produce
   the final `{oracle}_pertrack.npz`

## Validation

After building, verify with the SORT1 rs12740374 variant:

```python
from chorus.oracles.enformer import EnformerOracle
from chorus.analysis.normalization import get_pertrack_normalizer
from chorus.analysis.discovery import discover_variant_effects

oracle = EnformerOracle(use_environment=True, reference_fasta='hg38.fa')
oracle.load_pretrained_model()
norm = get_pertrack_normalizer('enformer')

result = discover_variant_effects(
    oracle, oracle_name='enformer',
    variant_position='chr1:109274968', alleles=['G', 'T'],
    gene_name='SORT1', normalizer=norm,
    output_path='/tmp/sort1_validation/',
)
```

The HTML report uses the layer-aware floor rescale for the IGV browser.
Pass `igv_raw=True` to use raw autoscale instead.
