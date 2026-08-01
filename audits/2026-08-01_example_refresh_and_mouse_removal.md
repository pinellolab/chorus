# 2026-08-01 — example refresh, mouse-model removal, and eight latent defects

**Scope.** Started as "regenerate the examples after #119". Ended up covering two
shipped-artefact corrections and eight defects, seven of them pre-existing and
**none caught by the test suite**.

**Result.** `pytest -m "not integration"`: **507 passed, 4 skipped, 0 failed**.
Two PRs ([#121](https://github.com/pinellolab/chorus/pull/121) and the example
refresh), one HuggingFace dataset revision, eight issues
([#122](https://github.com/pinellolab/chorus/issues/122)–[#129](https://github.com/pinellolab/chorus/issues/129)).

**The through-line.** Six of the eight are the same shape: **two sides of one
contract with nothing comparing them.** A registry with no organism field vs a
builder that opens hg38. A builder writing `HepG2` vs an oracle asking for
`LentiMPRA:HepG2`. A builder classifying layers from `description` vs a query path
reading `identifier`. A generator writing `.json` but not the `.tsv` beside it.
Two scripts independently declaring the same variant's alleles. In every case both
halves were individually plausible and no test compared them.

---

## 1. ChromBPNet shipped 33 mouse models scored against hg38 — FIXED (#121)

ENCODE publishes a mouse developmental atlas of ChromBPNet models (embryonic
forebrain / midbrain / hindbrain / limb / liver / heart / neural tube / facial
prominence, E11.5–E14.5). **33 of the registry's 42 ATAC/DNASE entries were
those mouse models.** Every code path around them assumes hg38:

- `scripts/build_backgrounds_chrombpnet.py` opens `genomes/hg38.fa` and draws its
  DHS-anchored positions from the **hg38** DHS vocabulary. So the per-track CDFs
  shipped for those 33 rows were built by pushing *human* sequence through *mouse*
  models.
- `chrombpnet_globals.py` recorded no organism, so there was no field to filter or
  assert on. That is what let it ship.

The 9 surviving human models are **exactly** the set
`scripts/generate_catv1_defaults.py` already listed as
`CHROMBPNET_HUMAN_ANNOTATIONS`, with the comment *"CATv1 is GRCh38-only, so those
have no counterpart and are omitted."* The filter existed in one place and had
never propagated to the registry or the builder.

### The name-collision trap

`liver`, `heart`, `brain` and `forebrain` are all **human** ENCODE CHIP biosamples
in `chrombpnet_JASPAR_metadata.tsv` *and* mouse ATAC/DNASE names. A name-only
filter silently deletes human CHIP rows. The removal keys on
`(assay, cell_type)`; the validator asserts no CHIP row is dropped, and **16
human CHIP rows with colliding names were correctly kept.**

### Background republished

`chrombpnet_pertrack.npz`: **786 → 753 rows** (9 human ATAC/DNASE + 744 human
CHIP) by pure row subsetting — no model re-run, no CDF recomputed.

| gate | result |
|---|---|
| rows | 753 = 744 CHIP + 9 ATAC/DNASE |
| the 9 are exactly the human allow-list | pass |
| rows removed | exactly 33, **zero CHIP** |
| 9 surviving accessibility rows | bit-identical, `max abs diff = 0.000e+00` |
| sample counts | 18,672 / 34,004 / 1,088,128 unchanged |
| CHIP counts | 37,344 / 68,008 / 2,176,256 |
| CDF health | all finite, all monotone, no all-zero rows |

Published to `lucapinello/chorus-backgrounds` (dataset PR #3, merged),
sha256 `76f267dc862edc86052f2b25a2a8520e960dd193ad39c4ecd19e32b8a8546553`.
Verified by fresh **unauthenticated** download and through
`PerTrackNormalizer`: surviving tracks normalise, the 33 removed tracks return
`None` (graceful — no crash, no false 100th), and `CHIP:liver:CTCF` still
resolves.

This revision also carries the #120 CHIP transform correction (joint softmax +
`exp(C) − n_tracks`): median effect ratio 1.86×, summary 2.50×.

**Not fixed, filed as #124:** species consistency is enforced *by accident*
everywhere else. Enformer (1,643 mouse tracks) and Borzoi (2,608) are human-only
only because a `*_human_targets.txt` was chosen; nothing asserts that against the
FASTA the builders open. And `AlphaGenomeOracle(organism="mouse")` is accepted,
stored on `self.organism`, and **never read**.

---

## 2. Eleven of thirteen examples were three months stale — FIXED

11 of 13 committed examples were last regenerated **2026-04-21..04-29** and so
predated two correctness fixes:

- **#92 T1 (2026-06-17)**: variants were scored on a 1 bp region, which
  fixed-input oracles map into an N-padded output window, collapsing the effect.
  This moved the **raw** predictions.
- **#119**: the percentile denominator was the raw sample count instead of the CDF
  grid width. AlphaGenome stores 1,697–1,909 samples against a 10,000-point grid,
  so its percentiles were divided by ~1,909 — inflated up to ~5×.

The committed examples had a median `|percentile|` of **exactly 1.0000**.

The cleanest single proof that they were pre-#119 is a numerical fingerprint:
`FTO_rs1421085`'s committed `DNASE:HepG2` quantile is
`0.38082765845992667`, which is **exactly** `727 / 1909` — 1,909 being that
track's `effect_counts`. Under #119 the same rank gives `727 / 10000 = 0.0727`.
The committed value therefore encodes the sample count as the denominator instead
of the grid width, an inflation of **5.24×** on that row.

| example | median \|pct\| | ≥95th with \|effect\| < 0.1 |
|---|---|---|
| variant_analysis/SORT1_rs12740374 | 1.0000 → **0.7935** | 80 → **2** |
| validation/TERT_chr5_1295046 | 1.0000 → 0.9646 | 104 → 42 |
| sequence_engineering/region_swap | 1.0000 → 0.9988 | 16 → 6 |
| .../integration_simulation | 1.0000 → 0.9984 | 16 → 10 |
| variant_analysis/SORT1_enformer | 0.9995 → 0.9587 | 10 → 4 |

226 → 64 contradictory rows. **The residual 64 are the genuine #83 problem** and
concentrate in RNA_SEQ and CAGE — exactly the layers measured to be unfixable by
better position sampling (RNA-seq background `p0 = 1.000` under all eight
candidate region sets). Those need the magnitude floor, not a rebuild.

`batch_scoring` is the starkest case: all 5 SORT1-locus variants previously read
`max_quantile = 1.0000`, four of them for `|effect| < 0.07`. They now spread
0.9059–1.0000, and the four LD proxies land at 0.18–0.91 — a real gradient.

The flagship SORT1 conclusion **survives**: effects grew 3–7× under #92
(`DNASE:HepG2` +0.449 → +1.332, `CEBPB` +0.274 → +3.046), so the `≥99th` is now
earned rather than an artefact.

### One regression the fix introduced

`discovery/SORT1_cell_type_screen` swaps its 3rd cell type,
`left lobe of liver` → `amniotic epithelial cell`. **Not** a percentile artefact —
that screen ranks by `alt_value × |effect|` (raw values), so this is #92's
windowing fix. The new pick has `ref_value` 53.0 vs 184.9 and `|effect|` 2.90 vs
1.84: a bigger fold-change on a 3.5× quieter baseline. For a canonical *liver*
variant that is a worse answer, and it argues directly for the low-activity
qualifier proposed in #83 — `_is_low_baseline` requires `|effect| > 1.5` **and**
`alt_value < 5`, so it does not catch this.

`variant_analysis/SORT1_chrombpnet` was deliberately **not** touched: it scores
only `DNASE:HepG2`, bit-identical across both of tonight's rebuilds, so
regenerating it changed nothing but a timestamp.

### Two artefacts had to be left stale — they cannot be committed at all

The regenerated LegNet and consolidated multi-oracle HTMLs exceed GitHub's hard
100 MiB per-file limit, so `git push` is rejected by the pre-receive hook:

| artefact | committed | regenerated | factor |
|---|---|---|---|
| `rs12740374_SORT1_legnet_report.html` | 1.29 MB | **131.0 MB** | ~101× |
| `rs12740374_SORT1_multioracle_report.html` | 9.46 MB | **138.8 MB** | ~15× |

Both were restored to their previous bytes, so those two files are now frozen
while their sibling `example_output.json`/`.md` moved — the same drift class as the
orphaned TSVs fixed above, and it cannot be fixed from this side.

`audits/AUDIT_CHECKLIST.md:172` already documents this as **P0** with the exact
guard (`find examples -name '*.html' -size +50M` before regenerating). **I did not
run it, and hit the wall on push.** That is the argument for automating it: a
manual pre-flight check does not survive a regeneration sweep. Filed as
[#129](https://github.com/pinellolab/chorus/issues/129).

LegNet predicts a *single scalar* per sequence, so a 131 MB IGV payload for a
one-number oracle is the anomaly — the suspect is #99's "CDF fallback for models
without per-bin distributions", tiling that scalar across every 1 bp position.

---

## 3. Three shipped examples could not be regenerated at all — FIXED

`BCL11A_rs1427407`, `FTO_rs1421085` and `SORT1_rs12740374_with_CEBP` had their
generator entries **commented out** in `scripts/regenerate_examples.py`. Their
outputs still shipped, so they were frozen at 2026-04-21 with no way to refresh
them.

It was collateral, not a decision: `fc38632` (2026-05-08) is titled *"fix: support
mixed-resolution tracks and per-track normalization in IGV…"* and does three
unrelated things to this file — adds ATAC to `HEPG2_TRACKS`, flips the ChromBPNet
assay to DNASE, and comments out these three dicts, with **no prose added**.
Contrast the `ENFORMER_EXAMPLES` block twelve lines below, where a deliberate
removal carries a full paragraph ending *"Do not re-introduce them."*

Worth noting for future archaeology: `git log -S '"name": "BCL11A rs1427407…"'`
points at `96649ee`, the commit that *added* the entry — because `-S` counts
occurrences and commenting a line out does not change the count. That is part of
why this drifted unnoticed.

They are live documentation, not dead weight:

- `README.md:149` links `validation/SORT1_rs12740374_with_CEBP` as the answer to
  *"Replicate a published regulatory variant finding."*
- `examples/walkthroughs/README.md:108` promises *"Four worked examples: SORT1
  (HepG2 liver), BCL11A (K562 erythroid), FTO (metabolic), TERT promoter
  (K562)."*
- each ships a reproduction `notebook.ipynb`.

Re-enabled after verifying all 13 track identifiers they reference still exist in
`alphagenome_tracks.json`, and regenerated.

An earlier audit had already flagged this as an open decision —
`audits/2026-05-08_post_pr79_merge_audit.md:133`: *"Lorenzo's PR commented these
out. The HTMLs are still in the repo but the regen script won't refresh them.
Decide: re-enable, or remove the stale HTMLs."* It had sat undecided for three
months. Decided here: re-enable.

### The blocker found while re-enabling: BCL11A scored a sequence that does not exist

`BCL11A_rs1427407` declared `ref="G"` at `chr2:60490908`. **hg38 has `T`** there
(context `AAACA[T]TTCCC`). `chorus/core/base.py:464-470` detects this and *warns,
then proceeds*:

```
WARNING - Provided reference allele 'G' does not match the genome at
chr2:60490908 (genome='T'). Chorus will substitute the provided reference
allele into the prediction interval — verify your coordinates and genome build.
```

So the "ref" arm was a **synthetic non-reference sequence** and the "alt" arm was
the actual reference base — the effect was inverted *and* measured against a
sequence that does not exist. The warning fired on every regeneration, including
mine, and was invisible in a multi-thousand-line log.

`G>T` is the literature's *ancestral* orientation. Corrected to the hg38-oriented
`T>G` in `regenerate_examples.py`, `generate_walkthrough_notebooks.py:179` (which
had independently declared the same wrong allele) and the three prose references in
`variant_analysis/README.md`. **Every effect sign flips** relative to any earlier
BCL11A output. Regenerated with zero ref-mismatch warnings.

Checked all four declared examples against the FASTA; 1 of 4 was wrong:

| example | position | declared | hg38 | |
|---|---|---|---|---|
| SORT1 rs12740374 | chr1:109274968 | G | G | ok |
| **BCL11A rs1427407** | chr2:60490908 | **G** | **T** | **mismatch** |
| FTO rs1421085 | chr16:53767042 | T | T | ok |
| TERT | chr5:1295046 | T | T | ok |

Filed as [#128](https://github.com/pinellolab/chorus/issues/128) — the
warn-and-substitute default, and the absence of any ref-mismatch marker in the
output, are what let this ship.

---

## 4. Two committed TSVs were never regenerated — FIXED

`regenerate_examples.py` wrote `example_output.tsv` only in the AlphaGenome path.
The Enformer and ChromBPNet paths rewrote `example_output.json` **in the same
directory** and left the `.tsv` alone, so the two drifted apart:

- `SORT1_enformer/example_output.tsv` listed `ENCFF571HTM` … at `quantile 1.0`
- `SORT1_enformer/example_output.json` (regenerated) listed `ENCFF430NNH` … at
  `quantile 0.9605`

Different tracks, contradictory values, one directory. Confirmed by mtime: after a
full regeneration run every covered artefact was stamped `08-01 03:2x` while those
two TSVs kept checkout time. The TSV is documented in
`SORT1_enformer/README.md:61`, so it must be generated, not deleted. Extracted
`_write_tsv()` and called it from all three paths.

---

## 5. Two READMEs documented the bug as intended behaviour — FIXED

`variant_analysis/SORT1_rs12740374/README.md` carried:

> **Why all `≥99th`?** Each effect percentile is computed against ~10,000 random
> SNPs; Chorus collapses the top bucket to `≥99th` rather than rendering a
> spurious gradient in a CDF tail that doesn't have enough samples…

Three things wrong. The background is **1,697–1,909** random genomic positions
with a random alternate allele — not ~10,000, not SNPs, and **not gnomAD** (no
code samples gnomAD; `docs/NORMALIZATION_GUIDE.md:389` claims otherwise). And the
uniform `≥99th` was not a display choice at all: it was #119's denominator bug.
Rewritten with measured background statistics (95.1% of `DNASE:HepG2`'s background
below `|log2FC| = 0.1`, median 0.0126) and an explicit note about what the earlier
revision got wrong.

`batch_scoring/README.md` was worse: besides the same "~10K random SNPs" claim and
conflating effect percentile with regional activity, its **input** VCF snippet
listed `rs629301` at chr1:109275684 C>T while its **output** table listed
`rs1626484` at the same position G>T — and `rs629301`, `rs12037222` and
`rs2228603` are not scored by the generator at all. All three variant lists
realigned to `scripts/regenerate_remaining_examples.py:430-434`.

---

## 6. The examples are not reproducible run-to-run — FILED (#127)

Two independent runs of the same example, 14 minutes apart with identical code,
GPU and background, disagree on **454 numeric fields** across 63 tracks.

| layer | n | median \|Δ raw\| | p90 | max | median \|effect\| |
|---|---|---|---|---|---|
| chromatin_accessibility | 4 | 0.003446 | 0.003446 | 0.003446 | 1.328484 |
| histone_marks | 2 | 0.001537 | 0.001537 | 0.001537 | 1.255372 |
| tf_binding | 4 | 0.002579 | 0.002579 | 0.002579 | 3.046299 |
| **tss_activity** | **116** | **0.005385** | **0.009467** | **0.015477** | **0.005761** |

**For `tss_activity` the run-to-run noise exceeds the effect being reported**, and
**36 of 126 raw scores flip sign between runs** (e.g. `+0.000600 → −0.008847`).
Relative difference: median 0.42%, p90 122%, max 2145%.

This matters for #83 more than the background argument does. For those rows the
effect is not merely small, it is **not reproducible** — so no normalisation can
rescue them, and any magnitude floor should be at least the pipeline's own
reproducibility scale (~0.01 for AlphaGenome CAGE). It also means the sign-only
consensus vote (`multi_oracle_report.py:247`) is a coin flip on those tracks.

The large-effect layers are unaffected in relative terms (0.0034 on a 1.33 effect
is 0.26%); the problem is confined to the near-zero regime.

---

## 7. Filed, not fixed

| # | finding |
|---|---|
| [#126](https://github.com/pinellolab/chorus/issues/126) | **LegNet's percentiles have never worked.** Builder writes bare `HepG2`/`K562`/`WTC11` (`build_backgrounds_legnet.py:225,326`); oracle asks for `LentiMPRA:HepG2` (`legnet.py:50`). Measured: `LentiMPRA:HepG2` → `None`, `HepG2` → 0.9967. All 3 shipped rows are unreachable, and the multi-oracle report renders an em dash that reads like deliberate suppression. |
| [#122](https://github.com/pinellolab/chorus/issues/122) | **AlphaGenome histone tracks: background at 501 bp, scored at 2001 bp.** The builder classifies CHIP layers from `description` (`CHIP:<cell type>`, no mark name) so all 2,733 CHIP rows were built with the `tf_binding` window; the query path reads `identifier` (which carries the mark) and returns `histone_marks` for 1,075. **20.8% of AlphaGenome tracks compare a 2001 bp statistic to a 501 bp null.** |
| [#123](https://github.com/pinellolab/chorus/issues/123) | **Enformer tracks are not ranked against the same variant set.** `effect_counts` takes 7 distinct values (9,600–9,606) because one per-variant `try/except` wraps the whole per-track loop. |
| [#124](https://github.com/pinellolab/chorus/issues/124) | Genome/species consistency enforced by accident; inert `organism=` parameter. |
| [#125](https://github.com/pinellolab/chorus/issues/125) | Shared sampler/transform extraction — 8 `ReservoirSampler` copies, 6+4 one-hot implementations, 8 `get_sequence` copies with divergent N thresholds. All three ChromBPNet CHIP bugs fixed on 2026-07-31 were two copies of the same arithmetic disagreeing. |
| [#127](https://github.com/pinellolab/chorus/issues/127) | **Example outputs are not reproducible run-to-run** — see §6. |
| [#129](https://github.com/pinellolab/chorus/issues/129) | **The LegNet report HTML is ~101x oversized** (1.29 MB -> 131 MB) and cannot be committed at all, so two artefacts stay frozen. Known as P0 in AUDIT_CHECKLIST.md:172 with a manual guard that was forgotten. |
| [#128](https://github.com/pinellolab/chorus/issues/128) | **A ref-allele/genome mismatch only warns**, then scores a synthetic sequence with no marker in the output — how the BCL11A bug shipped. 1 of 4 checked examples was wrong. |

Also noted while auditing:

- The **6 library notebooks** under `examples/notebooks/` carry executed outputs
  with numbers and are regenerated by **no script** — only a manual
  `nbconvert --execute`. `single_oracle_quickstart.ipynb` (Enformer, outputs from
  2026-04-27) was re-executed here; the other 5 were refreshed 2026-07-30/31.
- All 13 per-walkthrough `notebook.ipynb` files store **zero** output cells (pure
  codegen), so none held stale numbers.
- `advanced_multi_oracle_analysis.ipynb` cell 62 referenced
  `images/MA0139.1.svg`, which resolves to a non-existent
  `examples/notebooks/images/`. The file is at `examples/images/`. Fixed to
  `../images/`.

## Method note

One agent claim was **checked and refuted**: that all 13 examples carried
pre-#119 percentiles including the two refreshed at `707badb`. Recomputing each
committed `quantile_score` through the current normalizer while replicating
production semantics exactly (`variant_report.py:775-786` — `signed` from
`LAYER_CONFIGS`, `abs()` when unsigned, `None` below `NOISE_FLOOR_RAW_SCORE`)
gives **30 MATCH, 5 legitimately `None`, 0 stale**. Dropping any one of those
three details produces spurious "stale" verdicts, which is the likely origin of
the claim.
