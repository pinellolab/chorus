# Per-track background distribution scripts

These scripts build the **per-track CDFs** consumed by `PerTrackNormalizer` for variant
effect interpretation and IGV visualization.

**This file documents the mechanics: which script, which env, which flag. It is not the
authority on the design.** For which regions are sampled and why, which SNPs, how each
layer's statistic is defined, the guard inventory, and the dated decision log, read
[`docs/BACKGROUND_NULL_PROTOCOL.md`](../docs/BACKGROUND_NULL_PROTOCOL.md) — and update it
in the same commit as any change it describes.

> **Rewritten 2026-08-10.** The previous revision predated the August rebuild and had
> drifted badly: it listed 6 oracles of 8, gave ChromBPNet as "24 models, 2.4 MB" when it
> is 753 tracks and 79.5 MB, described the baseline as one 31,500-position mixture when
> there are three, said the effect null was "10,000 random SNPs" when it is a versioned
> stratified set, and — worst — documented capacity-50,000 reservoir subsampling as the
> design when that was the defect the rebuild existed to fix. Since README.md and
> docs/API_DOCUMENTATION.md both point here for "the full pipeline", it was the most
> load-bearing stale document in the repo.

## Where the output goes

Bulk data follows one switch, resolved in this order:
`CHORUS_DATA_DIR` → `<install>/chorus_data_dir.txt` → the install directory →
`~/.chorus` (only if the install tree is not writable). It is **not** `$HOME` by default
any more. `chorus config data-dir` prints what resolved and why.

```
$CHORUS_DATA_DIR/backgrounds/{oracle}_pertrack.npz
```

## The eight builds

| Script | Oracle | Tracks | NPZ | Env |
|---|---|---|---|---|
| `build_backgrounds_alphagenome.py` | AlphaGenome | 5,168 | 279.0 MB | `chorus-alphagenome` |
| `build_backgrounds_borzoi.py` | Borzoi | 7,611 | 803.9 MB | `chorus-borzoi` |
| `build_backgrounds_enformer.py` | Enformer | 5,313 | 556.7 MB | `chorus-enformer` |
| `build_backgrounds_cherimoya.py` | Cherimoya (CATv1) | 1,518 | 161.7 MB | `chorus-cherimoya` |
| `build_backgrounds_chrombpnet.py` | ChromBPNet | 753 | 79.5 MB | `chorus-chrombpnet` |
| `build_backgrounds_sei.py` | Sei | 40 | 2.9 MB | `chorus-sei` |
| `build_backgrounds_epinformerseq_v2_percell.py` | EPInformer-seq | 33 | 2.3 MB | `chorus-epinformerseq` |
| `build_backgrounds_legnet.py` | LegNet | 3 | 0.2 MB | `chorus-legnet` |

Note the EPInformer-seq script does **not** follow the
`build_backgrounds_<oracle>.py` pattern the other docs quote. ChromBPNet's 753 are
per-track CDFs (9 human ATAC/DNASE + 744 ChIP); its 33 mouse mm10 models were dropped on
2026-08-01 because their backgrounds had been built on hg38. Sei's are regulatory
classes, LegNet's are MPRA cell types.

Approximate cost for the full fleet is ~63 GPU-hours, dominated by AlphaGenome (~39 h).
ChromBPNet is ~8 h; Cherimoya, Sei, LegNet and EPInformer-seq are ~1 h between them.

## Quick start

```bash
mamba run -n chorus-enformer python scripts/build_backgrounds_enformer.py --part variants  --gpu 0
mamba run -n chorus-enformer python scripts/build_backgrounds_enformer.py --part baselines --gpu 1
mamba run -n chorus          python scripts/build_backgrounds_enformer.py --part merge
```

`variants` and `baselines` run in parallel on separate GPUs; `merge` needs no GPU and
combines the interim files into `{oracle}_pertrack.npz`.

### Flags that differ per script, and the traps

* **`--part`** — `variants` / `baselines` / `merge` / `both` / `all`. Most scripts accept
  `both`; **Cherimoya defaults to `all`** and additionally offers `merge-incremental` and
  `merge-shards`. Check `--help` rather than assuming.
* **`--gpu N`** — pass it explicitly on the five scripts that take it. They used to
  *overwrite* `CUDA_VISIBLE_DEVICES` with the `--gpu` default of 0, so two processes
  launched with different env values both landed on GPU 0: the first took 78 GB and the
  second failed every forward pass with `Attempting to perform BLAS operation using
  StreamExecutor without BLAS support`, silently dropping all 5,968 positions. An explicit
  env var now wins, but `--gpu` is unambiguous. **LegNet and EPInformer-seq have only
  `--device`**, so they do need the env var.
* **`--fold`** — Cherimoya and ChromBPNet ship cross-validation folds. Cherimoya's default
  is `CATV1_DEFAULT_FOLD`, i.e. the 5-fold **ensemble**, deliberately the same default the
  oracle uses. A null built on fold 0 under a query path that ensembles is not a null; see
  protocol §7b.
* **`--shard` / `--shard-of`** — position sharding for the long builds. Merge from a
  staging directory, not in place.
* **`conda run` buffers stdout**, so a 14-hour job's log stays empty until it exits. Use
  `conda run --no-capture-output ... python -u` when you need progress, and key failure
  detection off the exit code rather than off log contents.
* **Run each build inside its oracle's env.** Cherimoya's builder imports the `cherimoya`
  package, which exists only in `chorus-cherimoya`; elsewhere it logs `Failed to load
  <track>` once per track and carries on, so a 1,518-track run spent 75 minutes loading
  nothing before dying at the provenance step.

## What lands in the NPZ

| Array | Shape | Used for |
|---|---|---|
| `track_ids` | `(n_tracks,)` unicode | row identity; the key the query path looks up |
| `effect_cdfs` | `(n_tracks, 10000)` | variant effect percentile |
| `summary_cdfs` | `(n_tracks, 10000)` | activity percentile |
| `perbin_cdfs` | `(n_tracks, 10000)` | IGV per-bin display scaling |
| `{layer}_counts` | `(n_tracks,)` | offered sample count per track |
| `{layer}_retained` | `(n_tracks,)` | **retained** count — the thinning check reads this |
| `perbin_tail_k` | scalar | size of the exact tail buffer |
| `signed_flags` | `(n_tracks,)` bool | signed layers (RNA, MPRA, Sei) |
| `build_config` | JSON string | provenance, schema 4 |

`perbin_cdfs` is omitted for the three oracles with no per-bin profile: **Sei, LegNet and
EPInformer-seq**.

## Retention — exact, or capped with an exact tail

This is the part the old revision got wrong, so it is stated plainly.

A percentile is a rank against the sampled null, and it clamps at the largest sampled
value. **A uniform *m*-of-*N* reservoir subsample retains the population maximum with
probability exactly *m*/*N***, so capping the reservoir silently discarded the statistic
the clamp is computed against — measured up to **8.3×** understated on AlphaGenome's
`gene_expression` ceilings while p99 stayed right to 0.6%.

| Layer | Retention now |
|---|---|
| `effect` | **exact** — every offered value kept |
| `summary` | **exact** |
| `perbin` | capped at 50,000 **plus an exact top/bottom `tail_k`** — exact retention would be ~244 GB for Borzoi alone |

`tail_k` is derived, never picked:
`ceil(MIN_EXACT_TAIL_SLOTS * N_expected / n_points)`. `build_and_save` raises unless a
layer is exact or its exact-tail slots are sufficient, and a builder that omits the
`sampling=` argument logs an error by name — silence is how the original defect shipped
past a guard that already existed.

## Seeds

Region sampling `42`; DHS pools `43`; DHS summits in the activity population `567`;
reservoir `DEFAULT_SEED = 12345`; baseline sub-populations `789` (random), `111` (TSS),
`222` (gene body). All fixed: **a rebuild of an oracle whose inputs have not changed must
be bit-identical**, verified 2026-08-06 on Cherimoya, 1,518/1,518 rows.

## Positions: three activity populations, three effect families

Not one mixture. The populations are versioned artefacts in
`reference_sets/chorus_reference_positions_v1.npz`, each with its own sha256, and the
stamper *derives* which one an oracle used rather than assuming:

| Activity population | Positions | Oracles |
|---|---|---|
| `regions_genome_dominated` | 31,500 | alphagenome, borzoi, enformer |
| `…_minus_gene_body` | 29,500 | sei, legnet |
| `…_minus_gene_body_plus_dhs` | 34,500 | chrombpnet, cherimoya, epinformerseq |

Effect nulls come in three families, also stratified: `gene_anchored` (enformer, borzoi,
alphagenome, sei, epinformerseq), `accessibility` (chrombpnet, cherimoya — `dhs` 9,063 +
`random` 9,609), and `promoter` (legnet — `tss_promoter` 7,200, `ccre_pls` 5,400,
`ccre_pels` 2,700, `random` 2,505). Protocol §3 and §4 give the reasoning; §9 records
what was measured before each composition was changed.

## Provenance

Stamp after building:

```bash
python scripts/stamp_provenance_v4.py
```

Appends in place, no rebuild. Schema 4 carries one `build_id` across all eight oracles,
the per-layer `sampling` block (`mode`, `offered`, `retained`, `thinned_tracks`,
`tail_k`), the reference-set hashes, and the derived activity population.
`tests/test_npz_provenance.py` reads it back; `tests/test_no_reservoir_thinning.py`
asserts the retention claim from the artefact alone.

## Publishing

```bash
python -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='<file>', path_in_repo='<file>',
                    repo_id='lucapinello/chorus-backgrounds', repo_type='dataset')"
```

Verify by comparing the remote LFS sha256 against the local file rather than trusting the
upload. Users auto-download on first use; a cached copy that has been superseded is now
moved aside and refetched, so publishing a fix does reach existing installs.

## Other public scripts

| Script | Purpose |
|---|---|
| `regenerate_examples.py` | walkthrough outputs for alphagenome / enformer / chrombpnet (`--oracle`, `--gpu`, `--dry-run`) |
| `regenerate_remaining_examples.py` | discovery, causal, region_swap, integration, batch, TERT (`--only`) |
| `regenerate_multioracle.py` | per-oracle passes (`--oracle`) and the unified IGV (`--consolidate`, which re-renders from the pickles) |
| `generate_walkthrough_notebooks.py` | codegen for `examples/walkthroughs/*/notebook.ipynb` |
| `rerender_examples.py` | re-render HTML from stored JSON — **lossy for IGV**, it drops the per-bin arrays |
| `stamp_provenance_v4.py` | provenance, append-in-place |

Internal/maintenance scripts live in `scripts/internal/`.

## Validating a build

Cheapest first, and none of these needs a GPU:

```bash
python -m pytest tests/test_npz_provenance.py tests/test_no_reservoir_thinning.py \
                 tests/test_background_grid_integrity.py -q -m integration
```

Then check that real effects still land where they should — `tests/test_release_gates.py`
pins the measured percentiles — and that every shipped track is reachable through the
query path (`tests/test_every_shipped_track_is_reachable.py`, which is what caught Sei's
40 rows being unreachable). Protocol §8 step 7 has the full pre-ship list.
