# Background null models: protocol

**Status: LIVING DOCUMENT — updated as the 2026-08-06 rebuild converges.**
Last updated 2026-08-07. Sections marked ⚠️ are not yet final.

Every percentile chorus reports is a rank against a *background null* — a per-track
distribution of what the same statistic looks like at positions the variant is not at.
This document is the protocol: what the nulls are, which regions and SNPs go into them,
how they are computed, and what a new oracle must do to get one. It exists because the
reasoning was previously spread across commit messages and could not be followed by
anyone who had not written it.

If you are adding an oracle, skip to [§8](#8-adding-a-new-oracle).

---

## 1. The three nulls, and the question each answers

One `{oracle}_pertrack.npz` per oracle holds three `(n_tracks, 10000)` matrices. They are
**different reference classes answering different questions** and must not be conflated.

| array | question | reference class | consumed by |
|---|---|---|---|
| `effect_cdfs` | "is this variant's effect unusual among variants in comparable regulatory regions?" | peak-anchored positions (§3) | `effect_percentile`, `effect_exceedance` |
| `summary_cdfs` | "is this locus active for this track, **genome-wide**?" | genome-dominated positions (§4) | `activity_percentile` |
| `perbin_cdfs` | display only — rescales per-bin values for IGV colour scales | same positions as `summary` | `perbin_percentile_batch` |

**Why the effect and baseline reference classes must differ.** If they were the same
positions, then "median activity percentile of the effect null's REF windows" would be
0.5 identically, for any track — the statistic measures the *offset between two
populations*, not quality. Re-anchoring the baseline on peaks would also make "98th
percentile accessibility" stop meaning *top 2% of the genome* and start meaning *top 2%
of cCREs*, silently, in every report and IGV colour bar.

Companion arrays: `track_ids`, `signed_flags`, `layers_per_row` (per-row layer),
`{layer}_counts` (values **offered**), `{layer}_retained` (values **kept** — see §6),
`{layer}_tail_k`, `build_config` (provenance JSON).

---

## 2. Is the null stratified? **No — and this matters**

Positions are **stratified during sampling** to control the mixture (§3). The resulting
CDF is **not stratified**: `effect_cdfs` is one row per track pooling every stratum, and
no per-stratum array is stored.

Consequences, both load-bearing:

* You **cannot** ask "what percentile is this among cCRE positions specifically". Only
  the pooled mixture is available.
* Therefore **the mixture composition *is* the reference-class definition.** Changing a
  stratum's share silently redefines every percentile that oracle reports. This is why
  §3 fixes absolute counts rather than fractions, and why composition changes require
  the measurements in §9.

A per-stratum null is possible (the same positions, tagged, could fill separate rows) but
is **not implemented** and would multiply the artefact size by the stratum count.

---

## 3. Effect-null regions: which, and why

### 3.1 The gene-anchored family — enformer, borzoi, alphagenome, sei, epinformerseq

`DEFAULT_REGION_STRATA` in `chorus/utils/annotations.py`, sampled by
`sample_gene_anchored_positions`, at `DEFAULT_N_EFFECT_POSITIONS = 18_000`:

| stratum | fraction | count | what it is | why |
|---|---|---|---|---|
| `tss_near` | 0.100 | 1,800 | ±1 kb of a protein-coding TSS | CAGE/promoter signal lives here; a uniform position has none |
| `tss_far` | 0.100 | 1,800 | 1–10 kb from a TSS | proximal regulatory, off the peak |
| `junction` | 0.165 | 2,970 | ±100 bp of an exon/intron boundary | the only positions where splice statistics are non-trivial |
| `gene_body` | 0.060 | 1,080 | inside a PC gene, elsewhere | exonic/intronic context for RNA |
| `random` | 0.075 | 1,350 | uniform | **load-bearing**: without near-zero mass, small real effects get artificially LOW percentiles — the mirror of saturation |
| `ccre` | 0.500 | 9,000 | inside an ENCODE SCREEN cCRE | accessibility/TF/histone signal; includes the CA-TF, CA-CTCF, CA-H3K4me3 and TF categories |

### 3.2 LegNet — promoter, deliberately different

LegNet is a 200 bp promoter-MPRA model: the sampled position **is** the whole modelled
sequence, so a uniform window is almost entirely non-promoter.
`PROMOTER_REGION_STRATA`, `sample_promoter_anchored_positions`, also n = 18,000:

| stratum | fraction | count |
|---|---|---|
| `tss_promoter` (±250 bp of a PC TSS) | 0.40 | 7,200 |
| `ccre_pls` (SCREEN promoter-like) | 0.30 | 5,400 |
| `ccre_pels` (proximal enhancer-like) | 0.15 | 2,700 |
| `random` | 0.15 | 2,700 |

Deliberately **not** the generic cCRE mix: SCREEN is 62% distal enhancer-like against 2%
promoter-like, so the generic mixture would give a promoter model a null made of
enhancers — right family, wrong member.

### 3.3 ChromBPNet and Cherimoya — accessibility, unchanged

`random 10,000 ∪ DHS-summit 10,000` (Meuleman index, summit ±150 bp), assay-appropriate
and **not** modified by the 2026-08 rebuild. Their effect layers were never thinned; their
gain from the rebuild is the `perbin` tail (§6).

### 3.4 Two composition decisions, both settled by measurement

**DHS was added to the gene-anchored and promoter mixtures, then removed.** The a-priori
case was good — DHS summits concentrate TF footprints. Three Sei builds, medians over 40
tracks:

| | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| 12,000 no DHS → 18,000 **+DHS** | 0.971 | 0.937 | 0.954 | 0.936 | **1.000** |
| 12,000 no DHS → 18,000 **more cCRE+gene** | 1.035 | 1.030 | 1.042 | 0.992 | **1.261** |

`max = 1.000` is decisive: across all 40 tracks **not one DHS position beat the best
cCRE- or gene-anchored position already present**. Tested again on the layer the idea
targeted — enformer `tf_binding`, n=6,000/arm — DHS diluted it **worst of any layer**
(p99 0.858; 744 of 2,101 tracks gained a ceiling, 1,217 lost one). Likely cause:
redundancy, since SCREEN already carries the TF-ish categories, plus DHS being 3.6%
TSS-proximal at a median 68.7 kb.

An argument to retire with it: *"an additive union cannot hurt because
`max(union) = max(max_a, max_b)`"* protects **only the maximum**. A percentile is a
quantile of the mixture, so adding smaller-effect positions lowers the whole upper body.

**N grew 12,000 → 18,000 with proportions held.** That is the lever that worked: +26–31%
ceiling with the body unmoved. Re-dividing a fixed N *dilutes* — measured when the cCRE
half was first tried that way, TF saturation went 25% → 92%.

---

## 4. Baseline (summary / perbin) regions

**Composition is frozen; only scale was harmonised.** Genome-dominated, because the
question is genome-wide activity:

| oracle | composition | positions |
|---|---|---|
| enformer, borzoi | random 15,000 ∪ cCRE 11,500 ∪ TSS 3,000 ∪ gene-body 2,000 | 31,500 |
| **alphagenome** | same proportions | **10,500 → 31,500** (harmonised 2026-08-06) |
| chrombpnet | random ∪ cCRE ∪ DHS | unchanged |
| cherimoya | random ∪ cCRE ∪ TSS ∪ DHS | unchanged |
| sei, legnet | random 15,000 ∪ cCRE | unchanged |
| epinformerseq | random ∪ cCRE ∪ TSS ∪ DHS | unchanged |

AlphaGenome's baseline was a third the size of the two oracles printed beside it in
multi-oracle reports, so its activity percentiles were ranked against a smaller null. The
proportions were already identical, so this is a pure count increase.

Baseline positions out of margin are **dropped and counted** in `drop_reasons`, not
clamped — see §5.

---

## 5. How regions are sampled, exactly

1. **Chromosome eligibility.** `usable = {c: L for c, L in chrom_sizes.items() if L > 2 *
   margin_bp and c in protein-coding chroms}`, `margin_bp = 5_000_000` (so a 1 Mb
   prediction window always fits).
2. **Source populations are filtered to the usable *interval*, not just the chromosome.**
   ⚠️ This was a defect until 2026-08-06: 2,515 of 20,083 PC TSS (12.5%) sit within 5 Mb
   of a contig end, passed the chromosome test, and were then **clamped onto the margin
   boundary** — up to 5 Mb from the TSS they were labelled as within 1 kb of. Measured
   over 6,000 positions before the fix: 12.1–14.6% of each stratum on a boundary, only
   5,265 of 6,000 positions distinct, `chr16:5,000,000` appearing **64 times**. Duplicate
   positions give identical effect values, padding the sample count and manufacturing
   tied CDF runs. After: 0.0–0.1% on a boundary, 6,000/6,000 distinct.
3. **Per-stratum draw**, `int(round(n * fraction))` positions each, each stratum with its
   **own cursor** into its pool. Indexing a shared pool by total-emitted (the pre-2026-08
   behaviour) makes every stratum's draw depend on every other stratum's size.
4. **Unknown stratum name → `ValueError`.** The dispatch used to end in a bare `else`
   drawing a uniform position, which doubled as the `random` handler, the empty-pool
   fallback *and* the catch-all. Adding a stratum without a branch would have emitted
   uniformly random positions *tagged and stamped* with that stratum's name.
5. **Empty source population → `ValueError`**, never silent substitution.
6. **Every position carries its stratum tag**, so composition is recoverable from
   provenance.
7. **Verification (§7) round-trips the tags against the annotations they name** —
   ≥99% of `dhs` within 150 bp of a Meuleman summit, `ccre` inside a cCRE interval,
   `tss_near` within 1 kb of a PC TSS. This check is what found defect (2).

**Seeds.** Region sampling `seed=42`; DHS pools `seed=43`; reservoir `DEFAULT_SEED =
12345`; baseline sub-populations `789` (random), `111` (TSS), `222` (gene body). All
fixed. A rebuild of an oracle whose inputs have not changed **must be bit-identical** —
verified 2026-08-06 on cherimoya, 1,518/1,518 rows.

---

## 6. Which SNPs, and how the CDFs are computed

### 6.1 Position → variant

For each sampled position: read the reference base from the FASTA
(`ref.fetch(chrom, pos-1, pos)`), **reject if not in ACGT** (N-masked), and choose the
alt uniformly from the three other bases. One SNP per position, ref allele always the
true reference. Yield is ~99.5% (18,000 positions → ~17,900 SNPs); rejects are counted
by reason.

### 6.2 The statistic — per layer, not per oracle

Both alleles are predicted, and the layer decides the statistic
(`LAYER_CONFIGS`, `chorus/analysis/scorers.py`). **The builder and the query must use the
same one**; that is what `chorus/analysis/background_sampling.py` exists to guarantee.

| layer | window | aggregation | formula | pseudocount | signed |
|---|---|---|---|---|---|
| chromatin_accessibility | 501 | sum | log2fc | 1.0 | no |
| tf_binding | 501 | sum | log2fc | 1.0 | no |
| histone_marks | **2001** | sum | log2fc | 1.0 | no |
| tss_activity | 501 | sum | log2fc | 1.0 | no |
| splicing | 501 | sum | log2fc | 1.0 | no |
| gene_expression | gene mask | mean | logfc | **0.001** | **yes** |
| promoter_activity | whole | mean | diff | 0.0 | **yes** |
| regulatory_classification | whole | mean | diff | 0.0 | **yes** |
| enhancer_activity | whole | mean | log2fc | 1.0 | no |

Unsigned layers store `|effect|`; signed layers store the signed effect, so their rows run
negative and **both** ends of the support are live (100% of Sei and LegNet rows, 12.9% of
AlphaGenome's, 20.3% of Borzoi's).

### 6.3 Retention — exact, or capped with an exact tail

Values are accumulated per track in a `ReservoirSampler`. **A uniform *m*-of-*N*
subsample retains the population maximum with probability exactly *m/N*** — and the
maximum is what `effect_percentile` clamps against. So retention policy is per layer:

| layer | policy | why |
|---|---|---|
| `effect` | **exact** (`capacity` ≥ max offered) | worst case AlphaGenome 222,551/track ≈ 37 GB, affordable |
| `summary` | **exact** | worst case AlphaGenome 319,642/track ≈ 52 GB, affordable |
| `perbin` | **capped 50,000 + exact top/bottom `tail_k`** | up to 2,176,256/track; exact retention would be ~244 GB for borzoi alone |

`tail_k` is **derived, never picked**:
`tail_k = ceil(MIN_EXACT_TAIL_SLOTS * N_expected / n_points)` with
`MIN_EXACT_TAIL_SLOTS = 200` (the top 2% of a 10,000-point grid — where percentiles
saturate and where `effect_exceedance` divides). A single fixed `tail_k = 20,000` gives
AlphaGenome 202 exact slots but ChromBPNet only **91** and Cherimoya **183**. Derived
values: AG 19,740 · borzoi 19,832 · enformer 19,844 · chrombpnet 43,526 · cherimoya
21,763 — each yielding exactly 200.

`N_expected = n_positions × fan_out` is known **before** the first forward pass, which is
what makes `sampler_preflight()` a preflight.

### 6.4 Grid construction

`to_cdf_matrix` projects each track's sorted values onto a 10,000-point grid:

* `n ≥ 10,000` → evenly-spaced **order statistics**; the last slot is the true maximum.
* `n < 10,000` → **interpolated** onto the full grid (never padded).
* thinned + exact tail → spliced by **population rank**: the top/bottom `K` slots are the
  population's own order statistics, the interior is the uniform estimate. Degenerates
  **bit-identically** to the plain path when nothing was thinned.

The percentile denominator is the **grid width (10,000), not the sample count** — see
issue #119; dividing by the count inflated every AlphaGenome percentile ~5×.

⚠️ **Above the ceiling the percentile is exhausted, and we do not model past it.**
Measured: a GPD fit overshoots the far tail **3.8×**, an exponential undershoots
**0.27×**, the plain empirical maximum is within **13%**. `effect_exceedance` reports the
ratio to the ceiling instead — a fact about the sample rather than a modelling
assumption.

---

## 7. Guards, and what each one catches

Every one of these exists because something passed all the *other* checks.

| guard | catches | where |
|---|---|---|
| `cdf_grid_violations` | a grid `to_cdf_matrix` could not have produced (the padded-to-10,000 Enformer defect) | write time |
| `thinning_violations` | a row whose top slots came from a thinned sample. **Independent of the above**, which is fed *offered* counts while checking geometry set by *retained* counts, and skips every row with `n ≥ n_points` — so it was structurally blind | write time + merge |
| `yield_violations` | a build where <50% of tracks produced samples (an all-zero background merges cleanly and then silently disables the oracle) | write time |
| `scope_violations` | a build covering far fewer tracks than the background it replaces | **preflight** |
| `sampler_preflight` | a retention config that would thin the tail | **preflight** |
| `abort_if_nothing_loads` | a per-track loader that has attempted 25 models and loaded none | build loop |
| stratum-name `ValueError` | a stratum with no sampler branch | sampling |
| annotation round-trip | positions that do not match the annotation their tag names | test |
| `verify_rebuilt_backgrounds.py` | track-set changes, body drift, falling ceilings, missing retention, and **more real effects pinning than before** | before swap |

**Where the guards were insufficient**, recorded because the pattern recurs: ChromBPNet
once built 9 of 753 tracks and passed *every* quality gate — `rc=0`, 100% yield, exact
retention, 400 exact tail slots, zero load failures. A flawless build of 1.2% of the job,
caught only by comparing against the shipped track set. `yield_violations` asks "did the
attempted tracks produce samples?" and cannot ask "were the right tracks attempted?".

---

## 8. Adding a new oracle

Decide by **what the model outputs** and **which layers its tracks map to**.

### Step 1 — layers

Every track must classify to a `LAYER_CONFIGS` key via `classify_track_layer`. If it
returns `"other"`, `LAYER_CONFIGS.get("other")` is `None` and **every score becomes
`None` silently** — Sei shipped 40 built, verified, unreachable rows this way for months.
Add an `assay_type` branch; do not invent a layer without a statistic.

### Step 2 — effect-null region set

| if the model predicts… | use | because |
|---|---|---|
| binned profiles genome-wide (accessibility / TF / histone / CAGE / RNA / splice) | `DEFAULT_REGION_STRATA`, n = 18,000 | covers every layer's signal in one position set; one forward pass per position serves all layers |
| a fixed short promoter window (MPRA) | `PROMOTER_REGION_STRATA` | a generic mixture would be mostly enhancers |
| accessibility only, peak-centric | `random ∪ DHS-summit` (ChromBPNet/Cherimoya pattern) | assay-appropriate; do not "upgrade" it to the gene-anchored mix without measuring |
| something else | **measure before choosing** — build two arms differing in one thing and compare p50/p90/p99/max per layer, as in §3.4 | every composition guess in this project that was not measured was wrong |

### Step 3 — baseline region set

Use the genome-dominated mixture (§4) at 31,500 positions unless the assay demands
otherwise. **Do not** reuse the effect positions (§1).

### Step 4 — retention

Compute `N_expected = n_positions × fan_out` per layer, then:

* `N_expected` fits in memory → **exact** (`capacity ≥ N_expected`)
* otherwise → `capacity = 50_000`, `tail_k = derive_tail_k(N_expected)`
* call `sampler_preflight(...)` and let it refuse a bad config **before** the GPU time
* write `{layer}_retained` beside `{layer}_counts` in every interim **and** the final file

### Step 5 — wire the guards

Pass `sampling=sampling_block(...)` to `build_and_save`. Omitting it logs an error and
disables the thinning check entirely.

### Step 6 — verify before shipping

`scripts/verify_rebuilt_backgrounds.py --strict-retention`, then §7's expectations:
body ratios ≈ 1.0, ceiling not falling, 0 thinned on exact layers, ≥200 exact slots on
hybrid layers, and no increase in real-effect pinning.

---

## 9. Decision log

| date | decision | evidence |
|---|---|---|
| 2026-08-03 | cCRE half added as an additive union, not a re-weighting | fixed-N re-division took TF saturation 25% → 92% |
| 2026-08-06 | merge retains **exactly**; `capacity` keyword-only and required | AlphaGenome RNA thinned 2.97× at the merge; ceilings understated median 1.33×, worst **8.34×**, while p99 stayed correct to 0.02% |
| 2026-08-06 | anchored source populations filtered to the usable interval | 12.5% of PC TSS were clamped onto margins; `chr16:5,000,000` appeared 64× |
| 2026-08-06 | **DHS rejected** for both mixtures | added nothing to any ceiling (max ratio 1.000 on Sei); diluted enformer `tf_binding` worst of all layers (p99 0.858) |
| 2026-08-06 | N 12,000 → 18,000, proportions held | +26–31% ceiling, body unmoved |
| 2026-08-06 | AlphaGenome baseline 10,500 → 31,500 | it was ⅓ the size of the oracles printed beside it |
| 2026-08-06 | `perbin` capped + derived exact tail; `effect`/`summary` exact | perbin was thinned 16–43× on five oracles, `summary` up to 5.2× on three |
| 2026-08-06 | percentiles stay **strictly empirical** | GPD overshoots the far tail 3.8×, exponential undershoots 0.27×, empirical max within 13% |
| 2026-08-06 | vectorising Algorithm R **not** done | measured 2.4M values/s ≈ 19% of a pass; saves <1 h fleet-wide while changing every retained sample |
| 2026-08-07 | motif-anchored ChIP null **deferred** | cost is per-TF scoring (240 TFs × ~6,000 passes), not motif lookup; and peak/attribution-derived positions cannot contain motif-*creating* variants, which is the saturating case |

## 10. ⚠️ Open

* The 2026-08-06 rebuild is **not complete**: AlphaGenome's effect pass is outstanding.
  Numbers here for AlphaGenome `effect` still describe the 2026-08-05 exact re-merge.
* Nothing has been swapped into the live backgrounds; staged in
  `/data/chorus_data/rebuild_2026-08-06/`, originals in
  `/data/chorus_data/pre_unified_rebuild/`.
* LegNet declares `resolution = 50` over a 200 bp window while holding **one** value, so
  every sub-region score returns `None`. The tools now explain it; the geometry is not
  fixed, because that would move its background and every committed artefact.
* AlphaGenome `histone_marks` and Enformer `tf_binding` remain ~20%/25% pinned on
  motif-creating variants. Irreducible with an empirical ceiling; see §9's last row.
* `build_config` is still absent on ChromBPNet and a different schema on Cherimoya;
  unifying it to `schema_version 4` with one `build_id` across all 8 is not yet done.
