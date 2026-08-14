# Background null models: protocol

**Status: CONVERGED for the 2026-08-06/07 rebuild, and still living.**
All 8 oracles rebuilt, verified and swapped into place on 2026-08-07; provenance stamped at
schema 4. §10 records the final state and what remains open. Update this document in the
same commit as any change it describes.

⚠️ **The stamped activity population was corrected on 2026-08-09 and the shipped
backgrounds must be re-stamped** (`python scripts/stamp_provenance_v4.py`) before that
correction reaches a reader. The stamper had written `regions_genome_dominated`, 31,500
positions, plus that artefact's sha256 into all eight files unconditionally; it is true for
three of them. There is **one activity population per builder, not one for the fleet** —
see §4 and §10.

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
question is genome-wide activity. There are **three mixtures, not one** — read off each
builder's sampling block, and each one confirmed against the summary sample count the
corresponding NPZ carries:

| oracle | composition | positions | offered per track |
|---|---|---|---|
| enformer, borzoi | random 15,000 ∪ cCRE 11,500 ∪ TSS 3,000 ∪ gene-body 2,000 | 31,500 | 31,005 / 30,986 |
| **alphagenome** | same, same proportions | **10,500 → 31,500** (harmonised 2026-08-06) | 30,868 |
| chrombpnet, cherimoya | random 15,000 ∪ cCRE 11,500 ∪ TSS 3,000 ∪ **DHS 5,000**, no gene body | 34,500 | 34,004 |
| epinformerseq | same | 34,500 | 34,002 |
| sei, legnet | random 15,000 ∪ cCRE 11,500 ∪ TSS 3,000, **no gene body, no DHS** | 29,500 | 29,004 / 29,002 |

The rightmost column is `max(summary_counts)` from the shipped NPZ, and it is the reason the
mixtures cannot be conflated: a reservoir offered 34,004 samples per track did not draw them
from a 31,500-position set. (ChromBPNet's 753 tracks offer up to 68,008 because its profile
head is two-stranded and both strands are scored; AlphaGenome's RNA tracks offer 319,642
because they emit one row per gene in the window. See §7 for the guard.)

AlphaGenome's baseline was a third the size of the two oracles printed beside it in
multi-oracle reports, so its activity percentiles were ranked against a smaller null. The
proportions were already identical, so this is a pure count increase.

Baseline positions out of margin are **dropped and counted** in `drop_reasons`, not
clamped — see §5.

**Why three mixtures and not one.** Nobody chose this: each builder grew its own sampling
block and no builder reads `reference_sets/` (§4b) — they all resample from the same seeds,
so a difference in the block is a difference in the population, silently. It is recorded
rather than harmonised because harmonising it means rebuilding five backgrounds, and
**no composition change ships without a two-arm measurement** (§9). What is fixed is that
each file now names the mixture it actually used.

---

## 4b. The reference sets — versioned populations

`reference_sets/chorus_reference_positions_v1.npz`, built by
`scripts/build_reference_position_sets.py`. Three SNP families, each with a content
sha256, plus provenance (generator git sha, `fai` sha256, FASTA prefix hash, every seed,
requested **and** realised strata):

| family | oracles | SNPs | composition |
|---|---|---|---|
| `gene_anchored` | enformer, borzoi, alphagenome, sei, epinformerseq | 17,909 | §3.1 |
| `promoter` | legnet | 17,805 | §3.2 |
| `accessibility` | chrombpnet, cherimoya | 18,672 | random 9,609 ∪ DHS 9,063 |

**Why an artefact rather than a seed.** A seed reproduces positions only while nothing
upstream moves, and the GTF, cCRE BED, DHS index and FASTA are all updatable without
anyone noticing the reference class moved with them. Since the composition *is* the
reference-class definition (§2), that would silently redefine every percentile.

**Verification.** `--verify-against ORACLE [--backgrounds-dir DIR]` checks a built
background reproduces its family's population. The **retained subset is
oracle-specific**: a window whose N content exceeds `max_n_fraction` is rejected, and
windows differ by orders of magnitude — measured shortfall Sei **0**, Borzoi **1**,
Enformer **2** of 17,909 (≤0.011%). So percentiles across oracles rank against *nearly*
the same population, not identically the same, and the tolerance makes that checkable
rather than assumed. It also flags a consecutive run of `*_counts` as the #123
partial-credit fingerprint.

**Two defects it caught on its first run**, both of which had already passed the
distributional verifier:

* **epinformerseq built on 10,000 positions, not 18,000** — its builder's
  `--n-variants` defaults to 10,000 and the fleet driver never passed it. Its null was
  inconsistent with the rest of its family. Rebuilt; shortfall now 0.
* **81.7% of the promoter `random` stratum was on non-primary contigs** — 2,206 of 2,700
  positions across 109 unplaced scaffolds and alt haplotypes, 12.3% of LegNet's whole
  null. The promoter sampler filtered contigs on margin alone (100 kb, so anything
  >200 kb qualified) while the gene-anchored sampler also requires protein-coding genes
  present. Alt contigs are redundant copies of primary sequence and scaffolds are largely
  repetitive, so that stratum was not a uniform genomic background. Fixed to the same
  rule: 0% non-primary, 24 contigs.

### The region set

`regions_genome_dominated`, **31,500 positions** — random 15,000 · cCRE 11,500 · TSS
3,000 · gene-body 2,000, with its own four RNG streams (789 / 456 / 111 / 222) and a
**10 Mb** random-position margin, wider than the SNP sets' 5 Mb. Reproduces what the
enformer, borzoi and AlphaGenome baselines logged exactly.

It is the **only** region set in the artefact, and only those three builders sample all of
it (§4). The other five are recorded as **derivations** of it, so their populations still
have a content hash a reader can recompute from shipped inputs:

| activity population | derivation | positions | sha256 |
|---|---|---|---|
| `regions_genome_dominated` | the set, verbatim | 31,500 | `86ea592d46c4` |
| `regions_genome_dominated_minus_gene_body` | drop `gene_body` | 29,500 | `ddbc4b246ab3` |
| `regions_genome_dominated_minus_gene_body_plus_dhs` | drop `gene_body`, add 5,000 DHS summits (`sample_dhs_positions`, seed 567) | 34,500 | `ec3070d6a361` |

The first two are derivable from the artefact alone; the third additionally needs
`annotations/dhs_vocabulary_hg38.txt.gz`, and without it the stamper records **no** activity
hash and says why, rather than claiming one. The derivation is checked against itself before
any hash is published: recomputing the full set's sha256 from the artefact's own rows must
reproduce the sha256 the artefact records, or nothing is stamped. Verified 2026-08-09 by
replaying Sei's sampling block position for position — 29,500 rows hashing `ddbc4b246ab3`,
bit-identical to the artefact minus `gene_body`.

Uniform positions are the largest stratum **on purpose**: most of the genome is silent for
most tracks, and that is what makes a real peak land as a high percentile. Tests assert
`random` is the largest stratum and >45% of the set, and that the region and SNP
populations remain **distinct** (<5% overlap) — because unifying them makes the acceptance
criterion "median activity percentile of the effect null's REF windows" equal 0.5
identically for any track (§1).

Verification compares the **maximum** `summary_counts`, not the minimum: that array has a
genuine per-track spread (enformer 19,549–31,005 of 31,500) because a track can lack a
usable value at a position even when the forward pass succeeded. The maximum tracks the
size of the population; the minimum tracks something else.

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

### 6.5 From `perbin_cdfs` to a colour — the display scale

`perbin_cdfs` is the only null whose output is a *picture*, so it is the only one where
being statistically right is not sufficient — it also has to be legible. Two decisions sit
between the CDF and the rendered track, and both used to be hardcoded per oracle **name**:

**The band.** `perbin_floor_rescale_batch` maps a raw value to
`clip((v − floor) / (peak − floor), 0, 4)`, with `floor = cdf[p95]` and `peak = cdf[p99]`,
so 1.0 always means "genome-wide p99 for this track" and the ceiling is 3.0. That band
assumes signal decays smoothly out of the background. It is right for accessibility and
wrong for base-resolution TSS assays, where the distribution is a huge near-zero mass plus a
tiny population of enormous peaks: AlphaGenome CAGE has p95 = 0.0050 and p99 = 0.0405
against a maximum of **852**, so every real TSS from strength 1 to 3000 rendered at exactly
3.00, with 13.1% of the panel's bins pinned at the ceiling.

A track is re-rendered on a log band (`log1p`, anchored p99.5/p99.9) when the linear band is
measured to clip more than 4% of the bins it will draw. **The trigger is the rendered panel,
not the CDF** — `perbin_cdfs` cannot answer this question, and four attempts to make it
answer are recorded here because each looked plausible:

| candidate statistic | must-log | must-stay-linear | verdict |
|---|---|---|---|
| `max / p99.9` | p5 697, min 172 | p95 20.5, **max 4212** (ChromBPNet ChIP) | overlaps |
| `p99.9 / p99` | p5 5.7 | p95 15.6 | overlaps |
| `p99 / p95` | p5 3.0 | p95 10.0 | overlaps |
| predicted clip fraction | p5 0.0028 | p95 0.0045 | overlaps |

`max/p99.9` at a threshold of 50 looked clean at 41× separation until ChromBPNet's ChIP
tracks were added to the protected set; it would have log-scaled 130 tracks in total -- 102
ChromBPNet ChIP, 10 Enformer and 8 Borzoi CAGE, **7 AlphaGenome TF-ChIP**, 2 ChromBPNet DNase
and 1 Cherimoya DNase. Populations: 1,296 must-log against 19,070 must-stay, of 20,366 tracks
with a CDF. Note also that on a 10,000-point grid
`int(0.9999 × n)` is the **last slot**, so "p99.99" is the track maximum and that statistic is
a ratio to a single extreme order statistic — §6.4's warning applies to it directly.

The limit (7.5%) is calibrated on the **corpus**, not on one panel. Across all 346 subtracks
of the 19 committed IGV panels there is a gap with nothing in it:

| rank | saturation | track |
|---|---|---|
| #20–22 | **0.0899** | alphagenome `DNASE:HepG2`, SORT1 panels |
| | *— gap —* | |
| #23–24 | 0.0656 | alphagenome `ATAC:HepG2`, FTO panel |
| #25–26 | 0.0625 | enformer CAGE substantia nigra, SORT1 enformer panel |

The 22 subtracks above the gap are exactly the AlphaGenome panels this work exists to fix. An
earlier 4% limit sat *below* the gap and would have escalated 45 subtracks (13%), including
seven Enformer CAGE tracks that render acceptably — invalidating committed panels nobody had
regenerated. Three properties of the measurement matter:

* **It is taken as drawn, not natively.** Pooling is what creates saturation. CAGE's native
  clip rate is 0.005–0.014, indistinguishable from the ChIP tracks at 0.001–0.008; CAGE is
  1 bp and collapses 349 native bins per display bin where ChIP is 128 bp and collapses 2.
  Only after pooling do they separate, 0.131 against ≤0.013.
* **Acceptance is two-sided, and the improvement has to be real.** The log band is kept only
  if the strongest feature still reaches 1.0 *and* the band either clears the limit or halves
  the clipping. Without the peak test, p99.9/p99.99 anchors "fixed" CAGE by dropping its peak
  to 1.24 of 3.0 — zero saturation, no track. Without the halving test, a 0.550 → 0.500 change
  counts as a fix. A collapsed band (both log anchors mapping to the same value, reachable in
  shipped data) renders a two-level barcode that passes both tests because clipping guarantees
  the peak, so it is rejected outright.
* **Signed tracks are excluded from both decisions.** They have no floor at zero, so "does max
  lift the floor" is meaningless, and max over a bin holding a strong repression and a weak
  activation returns the activation — the repressive half of the panel disappears. Measured on
  borzoi `ENCFF734OLC+`, the measured choice flips mean → max and takes saturation 0.000 →
  0.138. 2,253 shipped tracks are signed (borzoi 1,543, alphagenome 667, sei 40, legnet 3).

So a wrong trigger cannot damage a track: it either changes nothing or it demonstrably
improves the panel.

**There are two pooling stages, and they take opposite defaults.** The feature stage above
reduces ~349 native bins into each display bin, so max can lift a floor and the choice is
measured per track. igv.js then reduces those already-pooled features to pixels — about 3:1 on
a 1 Mb panel — and there `max` is right for everything, because 3 chances cannot meaningfully
promote background while mean still dilutes a sharp peak. Measured at the browser's ratio, the
peak height mean costs: LegNet 2.33×, AlphaGenome DNase 1.56×, ChromBPNet 1.38×, CAGE 1.31×,
Cherimoya 1.14×. It costs *unequally*, which is the disqualifying part — 1.38× against 1.14×
is a 1.2× relative distortion between the two tracks the cross-oracle panel compares — and it
cancels signed tracks against themselves. Mirroring the feature stage's per-track choice is
also wrong here: it would send AlphaGenome DNase and the ChIP tracks to mean for floor
protection a 3:1 collapse does not need. A re-rendered track is labelled `(log scale)`, because its 1.0 is p99.9
rather than p99 — two same-assay panels in one report can legitimately differ (BCL11A's two
CAGE:K562 tracks measured 0.053 and 0.036 and only the first escalated). This is disclosure,
not a new inconsistency: the display axis (0–4 since v0.7.2, 0–3 before) has always
been per-track.

The escalation is deliberately confined to the IGV render paths. The matplotlib and CoolBox
figures share `rescale_for_display` but mean-smooth instead of max-pooling, so they cannot
manufacture ceiling bins, and native CAGE saturation is already under the limit. In practice it fires on AlphaGenome's 1 bp CAGE and splice layers, and
the reason is **resolution, not assay**: Enformer (128 bp) and Borzoi (32 bp) pre-bin CAGE,
which smooths the spike away, so their panels never clip enough to trigger. A track with no
CDF keeps the linear band, because it cannot be rescaled at all.

**The pooling operator.** IGV caps a track at 4,000 features, so a 1 Mb panel draws ~349 bp
display bins and each one must reduce ~349 native values to one number. Max-pooling can
never lose a peak; mean-pooling can never lift a floor. Which risk is real is a property of
the *track*, not the oracle — AlphaGenome needs opposite answers for its own 1 bp and 128 bp
layers — so `choose_aggregation` max-pools, measures the median of the result, and falls
back to mean only if the floor actually rose. Five cheaper predictors were measured first
and every one of them gets the sign backwards on at least one oracle, including two read
straight off the artefact (perbin `max/p99`, and signal mass above p99).

**Read the saturated fraction, not the ink fraction.** A panel is unreadable when a large
share of its bins *clip*, not when many of them are non-empty. On the shipped SORT1 panel
Cherimoya inks 41% of its display bins and reads correctly, because only 1.3% of them
saturate; CAGE at 13.1% saturated was a solid block. An ink-fraction criterion was tried
for pooling and flipped Cherimoya and ChromBPNet to mean, re-creating the 5.5× dilution it
was meant to fix. The two concerns stay separate: **pooling protects the floor, the band
protects the peaks.**

Scope is pinned by `tests/test_display_scale_is_measured_not_declared.py`, including the
check that all three duplicated render paths (`_igv_report`, `multi_oracle_report`,
`causal`) go through the measured decision — patching one of three is how a change here
came back reporting byte-identical output.

---

---

### 6.6 One oracle, two nulls — Cherimoya's folds

A percentile is a rank against a null, so the null must come from the *same model* that made
the prediction. CATv1 ships five cross-validation folds and chorus exposes two modes, so it
ships two nulls:

| artefact | built on | used when |
|---|---|---|
| `cherimoya_pertrack.npz` | fold 0 | the default |
| `cherimoya_ensemble_pertrack.npz` | the 5-fold mean | `fold="ensemble"` |

They are **not** interchangeable. On `DNASE:ENCSR149XIL` at chr1:109,274,968 the five fold
peaks are 8.24 / 15.47 / 15.34 / 11.08 / 7.65 against an ensemble peak of 11.10 — a 2.02×
spread, with any single fold landing between 0.69× and 1.39× of the ensemble. Ranking a fold-0
prediction against the ensemble's null does not return an approximation; it returns a number
that looks entirely normal and is wrong.

Both nulls are built on the **same reference sets** — each reproduces `effect_counts=18672`
and `summary_counts=34004`, the counts the published ChromBPNet CDFs use, which is the
builder's own proof that the variant and region sets are shared. The two differ *only* in which
model scored them.

Selection is automatic: every Cherimoya `TrackPrediction` stamps `metadata["fold"]`, and
`normalization.normalization_key()` resolves the CDF key from it. Two guards back that up —
a load-time check that refuses an artefact whose stamped `fold` disagrees with the key asking
for it (`BackgroundFoldMismatch`, deliberately *not* swallowed by the legacy fallback, because
every other load failure means "no percentiles" while this one means "wrong percentiles"), and
`tests/test_fold_selects_its_own_null.py`, which enumerates the percentile call sites so a new
one cannot pass a bare oracle name. A fold with no null raises rather than being approximated.

**Why fold 0 is the default.** It matches ChromBPNet's default fold, and with both nulls on
shared reference sets that makes the two oracles comparable *at the percentile level* — 0.9325
for ChromBPNet against 0.9550 for Cherimoya at the SORT1 locus. Decided with CATv1's author
(2026-08-11), whose view is that handling five models complicates and slows most analyses, so
fold 0 suits an interactive tool with the ensemble one argument away.

⚠️ **Raw magnitudes are NOT comparable between the two oracles, and that is about bias, not
folds.** chorus loads ChromBPNet as `chrombpnet_nobias` — the "TF Model" predicting the
*bias-corrected* accessibility profile, since ChromBPNet trains a Tn5/DNase bias model first
and regresses its effect out. CATv1 does no such correction: `n_control_tracks: 0` in the
shipped checkpoints, `controls = None` in its training config, GC-matched negatives in place of
a control track. Measured over four shared DNase experiments, fold 0 both sides:

| CATv1 vs | window sum | peak | peak/sum |
|---|---|---|---|
| `chrombpnet_nobias` (what chorus loads) | 1.32× | **3.40×** | **2.19×** |
| `chrombpnet` (bias-aware) | 1.14× | **1.02×** | **0.80×** |

CATv1 tracks the bias-*aware* model almost exactly. Profile shape agrees either way (rank
correlation 0.95, peaks 18 bp apart) — it is height and sharpness that separate, and both
collapse when the bias model is left in. Percentiles are unaffected because each oracle is
ranked against its own null, which is why the cross-oracle panel reads consistently. The
asymmetry is pre-existing rather than introduced by the fold change; switching the panel to
bias-aware ChromBPNet is deliberately left as separate work.

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
| `test_count_head_copies_agree` | a builder and its oracle computing the count head differently. Feeds identical heads to every copy and compares outputs — including the torch path, which stays duplicated for speed, and the pre-extraction expressions verbatim | test |
| `require_reference_assembly` | a builder opening a FASTA that is not the assembly its model was trained on. Checks the oracle's declared `training_genome` against the reference's chr1 length | **preflight**, all 8 builders |
| `BackgroundGenomeMismatch` | an artefact declaring a genome chorus does not rank against. On the `BackgroundArtefactMismatch` contract, so it is **not** absorbed by `get_normalizer`'s legacy fallback | load time |
| stratum-name `ValueError` | a stratum with no sampler branch | sampling |
| annotation round-trip | positions that do not match the annotation their tag names | test |
| `check_counts_fit_the_population` | a stamped activity population too small to have produced the samples the file carries: `max(counts) ≤ n_positions × per-position × fan-out`, per layer | stamp time + test |
| `verify_rebuilt_backgrounds.py` | track-set changes, body drift, falling ceilings, missing retention, and **more real effects pinning than before** | before swap |

`check_counts_fit_the_population` is one-sided, and that limit is worth knowing: it catches a
population declared too **small** (chrombpnet's 68,008 summary samples against 31,500 × 2 =
63,000; cherimoya's 34,004 and epinformerseq's 34,002 against 31,500) but not one declared
too **large**. Sei and legnet drew a strict *subset* of what they claimed, 29,004 under
31,500, and no inequality can see that. That side is held by `ACTIVITY_POPULATIONS` naming
what each builder samples, which is why the two mechanisms ship together.

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

### Step 1b — you can develop without our HuggingFace dataset

Before the region-set decision, the practical question: **where does your null live while you work on
it?** The canonical CDFs are in `lucapinello/chorus-backgrounds`, which an outside contributor cannot
write to, and reading §8 top-to-bottom leaves the impression that a contribution is blocked on that.
It is not. Two supported overrides:

* `CHORUS_BACKGROUNDS_REPO=your-username/your-backgrounds` — the dataset repo is read from that
  environment variable, defaulting to the canonical one.
* `get_pertrack_normalizer("mymodel", cache_dir=...)` — a local directory is checked **before** any
  download, so `mymodel_pertrack.npz` on your disk is used for an oracle name the canonical dataset
  has never heard of.

Neither is a workaround; both are in `chorus/analysis/normalization.py` for exactly this purpose.
Mirroring a finished null into the canonical dataset is a maintainer step, and it does not have to
block the PR that adds the oracle.

### Step 2 — effect-null region set

| if the model predicts… | use | because |
|---|---|---|
| binned profiles genome-wide (accessibility / TF / histone / CAGE / RNA / splice) | `DEFAULT_REGION_STRATA`, n = 18,000 | covers every layer's signal in one position set; one forward pass per position serves all layers |
| a fixed short promoter window (MPRA) | `PROMOTER_REGION_STRATA` | a generic mixture would be mostly enhancers |
| accessibility only, peak-centric | `random ∪ DHS-summit` (ChromBPNet/Cherimoya pattern) | assay-appropriate; do not "upgrade" it to the gene-anchored mix without measuring |
| something else | **measure before choosing** — build two arms differing in one thing and compare p50/p90/p99/max per layer, as in §3.4 | every composition guess in this project that was not measured was wrong |

### Step 2b — if it is a BPNet-family model, do NOT write the head arithmetic again

A profile head plus a count head is four operations — centre the logits, softmax, invert the
count head, scale — and they live in exactly one place:
[`chorus/core/count_head.py`](../chorus/core/count_head.py). Call
`expected_counts_profile(logits, log_counts, n_tracks=...)` from both the oracle and the
builder. Do not reimplement it, and in particular do not reach for a bare `expm1`.

**The rule this enforces: a CDF is only meaningful if it was built from the quantity
`predict()` returns.** If the builder and the oracle combine the heads differently, every
percentile that oracle produces is ranked against a distribution of something else — and it
will look completely normal. That is not hypothetical; three separate defects in this project
were two copies of these four operations disagreeing:

| what disagreed | what it cost |
|---|---|
| `exp` vs `expm1` on the count head, four call sites, fixed one at a time | +1 read — ~0.1% at a peak, up to **100%** at a quiet site, which is the regime a null is built from |
| per-strand vs one joint softmax over the flattened both-strand vector | the two emitted tracks together claimed **2.00x** the predicted counts |
| a count bias hardcoded `(N, 1)` where the model declares `(None, 2)`, silently broadcast by Keras | every predicted log-count shifted by 0.5885 — 1.80x low at a peak, 3.04x at a quiet site |

**The count head has three conventions and they are not interchangeable.** Get this from the
model's training target, not from a sibling oracle:

| convention | inverse | who |
|---|---|---|
| `log1p` per track | `expm1(C)` | ChromBPNet ATAC/DNASE, CATv1 — 1,560 tracks |
| `log1p` per track, pooled across a task's tracks with `logsumexp` | `exp(C) - n_tracks` | BPNet CHIP — 744 models |
| `log10` | `10 ** C` | EPInformer-seq |

EPInformer-seq is deliberately **not** routed through the shared helper for that reason: at a
log-count of 2.5 the log10 and log1p conventions differ by **26x**, so "unifying" them would
silently rescale every value. `tests/test_count_head_copies_agree.py` pins that distinction
along with the equivalences.

### Step 3 — baseline region set

Use the genome-dominated mixture (§4) at 31,500 positions unless the assay demands
otherwise. **Do not** reuse the effect positions (§1).

If the assay does demand otherwise, say so where the stamper can read it: add an entry to
`ACTIVITY_POPULATIONS` in `scripts/stamp_provenance_v4.py` describing your mixture as a
derivation of `regions_genome_dominated` (§4b). It is the only place the population is
recorded — no builder reads the reference artefact — and a missing entry is a hard error
rather than an inherited default, because inheriting somebody else's population is exactly
how five oracles shipped a false one.

### Step 4 — retention

Compute `N_expected = n_positions × fan_out` per layer, then:

* `N_expected` fits in memory → **exact** (`capacity ≥ N_expected`)
* otherwise → `capacity = 50_000`, `tail_k = derive_tail_k(N_expected)`
* call `sampler_preflight(...)` and let it refuse a bad config **before** the GPU time
* write `{layer}_retained` beside `{layer}_counts` in every interim **and** the final file

### Step 5 — wire the guards

Pass `sampling=sampling_block(...)` to `build_and_save`. Omitting it logs an error and
disables the thinning check entirely.

Declare `training_genome` on the oracle class and call
`require_reference_assembly(fasta, YourOracle, label=...)` immediately before the builder
opens its FASTA. The declaration is deliberately **not** inherited — `OracleBase` says
`None` and the guard refuses that — because chorus's human-only property came from a
metadata-file choice rather than an assertion, and an inherited `"hg38"` would be another
such choice wearing an assertion's clothes. If your model is not hg38 you cannot simply
declare that and proceed: the reference class itself is hg38-specific (§4b), so read §11's
mouse entry first.

### Step 6 — register the reference family

Add the oracle to `ORACLE_SNP_SET` in `scripts/build_reference_position_sets.py`, choosing
the family that matches Step 2:

| family | size | who uses it |
|---|---|---|
| `snps_gene_anchored` | 17,909 | enformer, borzoi, alphagenome, sei, epinformerseq |
| `snps_promoter` | 17,805 | legnet |
| `snps_accessibility` | 18,672 | chrombpnet, cherimoya |

This is not bookkeeping. It is what lets
`build_reference_position_sets.py --verify-against ORACLE --backgrounds-dir DIR [--strict]`
compare the built file with its family. Be precise about what that compares, because it
was over-claimed here until 2026-08-09: it checks the **cardinality** of the offered SNPs
(always), and the file's `build_config["reference_sets"]` stamp against the artefact's
recomputed content hash and strata (when stamped — `--strict` refuses a file that is not).
The stamp is copied out of the artefact by `stamp_provenance_v4.py`, so a match pins the
artefact *revision* the file was stamped against; nothing in a background records the
positions it scored, so this is **not** proof of population identity, and the closing line
names which comparisons actually ran.

A ChromBPNet run in this cycle exited 0, reported 100% yield and
exact retention, and had built **9 of 753 tracks** — a population comparison was the only
check that caught it. If the new oracle needs a family that does not exist, add it to the
builder and give it a sha256, so the same comparison is possible for the next one.

### Step 7 — verify before shipping

`scripts/verify_rebuilt_backgrounds.py --strict-retention`, then §7's expectations:
body ratios ≈ 1.0, ceiling not falling, 0 thinned on exact layers, ≥200 exact slots on
hybrid layers, and no increase in real-effect pinning.

Two failure modes this cycle that a plain "did it finish" check misses, both now guarded
but worth knowing about when the guard is the thing you are changing:

* **A build that produces nothing still writes a valid file.** All 5,968 positions were
  dropped (two processes on one GPU, cuBLAS OOM) and the result loaded fine.
  `yield_violations` exists for this. Never key success off the exit code alone —
  `conda run` also buffers stdout, so a 14-hour job's log stays empty until it exits.
* **Arrays get lost between interim and final.** `layers_per_row` twice, `build_config`
  once. The first version of that guard only checked arrays whose first dimension equals
  the track count, so it could not see `build_config` (file-level, shape `(1,)`) at all. If
  you add a new array, confirm the preservation check covers its shape class.

### Step 7b — if the model ships cross-validation folds, decide once and wire it BOTH sides

Applies to any oracle distributed as several replicates of the same experiment (CATv1 has
five; ChromBPNet has folds too). Two rules, both learned the hard way on Cherimoya:

**Whatever you choose, the builder and the query path must do the same thing.** A null built
on fold 0 under a query path that ensembles is not a null — the numerator and the
denominator are different statistics, and nothing in the artefact reveals it. Make the
builder's `--fold` default *be* the oracle's default (`CATV1_DEFAULT_FOLD`) rather than a
literal, so the two cannot drift, and verify parity on one sequence before spending GPU
hours: score the same window through the builder's fast path and through
`oracle.predict()`, and require agreement to ~1e-6.

**Averaging happens on the model's OUTPUT, not its heads.** For a BPNet-family model the
output is the expected-counts profile; the heads combine non-linearly (softmax and `expm1`),
so a mean over heads is meaningless. A mean over per-fold *effects* is also a different
number from a mean over predictions — measured 1.4849 vs 1.4576 on one variant. Read the
upstream model card for which it recommends; do not infer it.

Cost to budget: *k* folds is *k*× the forward passes for the build and, in
`use_environment=True` mode, *k* subprocess calls per query. Check the dispatch works in
**both** execution modes — env mode is the user default and is the easy one to miss.

### Step 8 — stamp provenance

`scripts/stamp_provenance_v4.py` — schema 4, one `build_id`, read **from the artefacts
rather than the build logs**. Logs describe what a run intended; artefacts describe what it
produced, and AlphaGenome once shipped a stamped claim contradicted by a stale measurement
sitting in the same file.

Read from the artefacts, but *checked* against them too — a hardcoded constant is neither.
The activity population is derived per oracle (§4b) and `check_counts_fit_the_population`
refuses to write a stamp the file's own count arrays contradict (§7). A refusal fails that
oracle and leaves it unstamped, with a non-zero exit code; the other seven still stamp.

`genome` is measured from the reference's chromosome lengths rather than stated, for the
same reason: the loader now refuses an artefact whose declared assembly is not the one
chorus ranks against, and that check compares two constants if the declaration is the
stamper's own assumption. An unidentifiable reference fails the stamp rather than being
guessed at.

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
| 2026-08-10 | display **pooling** measured per track, not declared per oracle | Cherimoya (1 bp, BPNet family, absent from the hardcoded list) rendered its peak at 0.547 against ChromBPNet's 3.000 on the same axis — 5.5×, display-only. Five cheaper predictors each get the sign wrong on ≥1 oracle |
| 2026-08-10 | display **band** log-scaled when the *rendered panel* is measured to clip >4% of its bins | AlphaGenome CAGE p99 = 0.0405 against max 852: 13.1% of display bins pinned at the ceiling, against 0.0–1.3% on the panels that read well. Fixed to 1.3%, peak still 3.00 |
| 2026-08-12 | Cherimoya default fold 0, with a second null for the ensemble | folds disagree 2.02x on the same sequence, so one null cannot rank the other's predictions; fold 0 matches ChromBPNet's default, making percentiles comparable (0.9325 vs 0.9550). Agreed with CATv1's author |
| 2026-08-12 | ChromBPNet↔Cherimoya compared at percentile level only | CATv1 has no bias model (`n_control_tracks: 0`, `controls = None`); it tracks bias-*aware* ChromBPNet at 1.02x on peak and differs from `chrombpnet_nobias` by 3.40x |
| 2026-08-11 | the trigger is **not** a genome-wide CDF statistic | all four candidates overlap between must-change and must-not-move; `max/p99.9 > 50` would have log-scaled 130 tracks: 102 ChromBPNet ChIP, 10 Enformer + 8 Borzoi CAGE, 7 AlphaGenome TF-ChIP, 2 ChromBPNet DNase, 1 Cherimoya DNase. `CHIP:K562:ZBTB11`, which that statistic ranked above CAGE, measures 0.000 saturation as drawn |
| 2026-08-13 | the BPNet-family count-head arithmetic has **one** implementation (`chorus/core/count_head.py`), shared by the oracles and the builders | it existed in five places and all three of the 2026-07-31 defects were two copies disagreeing (exp vs expm1 at four sites; per-strand vs joint softmax, 2.00x; a broadcast count bias, 0.5885 shift). Extraction verified **bit-identical** — the pre-extraction expressions compared with `array_equal`, not `allclose`, in float32 and float64 — and the two examples that exercise it regenerated to timestamp-only diffs. The torch builder path stays duplicated for accelerator speed, with equivalence pinned at 1.25e-07 |
| 2026-08-12 | the assembly is **checked**, not inherited from a filename | human-only held by accident: Enformer/Borzoi via `*_human_targets.txt`, AlphaGenome via a hardcoded `HOMO_SAPIENS`, ChromBPNet via nothing — which is why 33 mm10 models were scored against hg38 sequence (#121). `build_config.genome` was itself a literal in the stamper, so the field restated an assumption; it is now read off the FASTA. No artefact changed: all nine already said `hg38`, and all nine are |
| 2026-08-09 | activity population **recorded per builder** (three mixtures), not harmonised to one | harmonising means rebuilding five backgrounds, and the three mixtures are 29,500 / 31,500 / 34,500 positions — a composition change, so §9's two-arm rule applies. Recording it is free and makes the difference legible; the stamp had asserted one 31,500-position set for all eight, false for five |

## 10. The converged state (2026-08-07)

All 8 oracles rebuilt onto one reference population per family, verified, and swapped in
atomically with read-back and a manifest. Backups at
`/data/chorus_data/pre_unified_rebuild/`; `swap_in_rebuilt_backgrounds.py --rollback`
restores them, applying the same read-back check to the backup.

### What each oracle now draws from

One effect population per family — and, as of the 2026-08-09 correction, **three** activity
populations, because the builders' baseline blocks were never harmonised the way their
effect blocks were (§4):

| oracle | effect population | activity population | retention |
|---|---|---|---|
| enformer, borzoi, alphagenome | `snps_gene_anchored` 17,909 | `regions_genome_dominated` 31,500 | effect+summary exact, perbin capped + exact tail |
| sei | `snps_gene_anchored` 17,909 | `…_minus_gene_body` **29,500** | effect+summary exact |
| epinformerseq | `snps_gene_anchored` 17,909 | `…_minus_gene_body_plus_dhs` **34,500** | effect+summary exact |
| legnet | `snps_promoter` 17,805 | `…_minus_gene_body` **29,500** | exact |
| chrombpnet, cherimoya | `snps_accessibility` 18,672 | `…_minus_gene_body_plus_dhs` **34,500** | effect+summary exact, perbin capped + exact tail |

Every file records the **content sha256** of both populations, so "which reference class is
this?" is answerable from the artefact. Tests assert the stamp matches the artefact on
disk, that all gene-anchored oracles share one effect hash, and that oracles built by the
same baseline block share one activity hash **and that the three blocks' hashes differ** —
because until 2026-08-09 they were asserted to be identical, and that assertion passed only
because the stamper wrote one hash into all eight regardless of what each had sampled. An
oracle carrying no hashed activity population now fails the test rather than dropping out of
it.

### The measured outcome

Body unchanged, ceilings up. Medians of **per-track** ratios (new/old):

| oracle | layer | p50 | p99 | max | % tracks with a higher ceiling |
|---|---|---|---|---|---|
| alphagenome | effect | 1.022 | 1.026 | 1.110 | 68% |
| alphagenome | perbin | 1.000 | 1.011 | **1.682** | 98% |
| borzoi | effect | 1.033 | 1.049 | 1.251 | 70% |
| enformer | effect | 1.028 | 1.067 | 1.114 | 68% |
| enformer | perbin | 1.000 | 1.001 | 1.195 | 95% |
| chrombpnet | perbin | 1.000 | 1.002 | **2.350** | 97% |
| cherimoya | perbin | 1.000 | 0.998 | 1.827 | 97% |
| sei | effect | 1.021 | 1.031 | 1.308 | 85% |

Real-effect **pinning** on committed artefacts — the user-facing measure:
enformer **9.5% → 3.6%**, alphagenome **2.5% → 2.0%**.

And the mechanism, confirmed on the **six layers whose position population is unchanged**,
so retention is the only variable. 1 − *m/N* predicts the share of tracks whose ceiling
rises; worst deviation **1.4 points** over predictions spanning 26% to 98% and a 32-fold
range of thinning, with **0 of borzoi's 6,068 unthinned tracks** moving.

AlphaGenome's three layers are **deliberately excluded** from that check. Its position
count grew in the same rebuild (effect 148,367 → 225,253), so retention is not the only
variable and the identity does not apply — measured 70.8% against a predicted 86.5% for
`effect`, and 97.0% against 80.8% for `summary`. Those two numbers are not evidence for
or against the mechanism; quoting them as agreement would be dishonest, and quoting them
as disagreement would be equally wrong. The six controlled layers are the evidence. See
`audits/2026-08-06_null_model_rebuild.md`.

### One layer got narrower, and it is recorded rather than smoothed over

LegNet was **never thinned** — 3 tracks, well under capacity. It moved to the shared
`snps_promoter` reference class (11,913 → 17,805 positions), and that composition change
*narrowed* one of its three ceilings:

| track | n before | n after | max before | max after | ratio | p99 ratio |
|---|---|---|---|---|---|---|
| K562 | 11,913 | 17,805 | 0.9057 | 0.7887 | **0.871** | 0.954 |
| HepG2 | 11,913 | 17,805 | 0.9879 | 0.9788 | 0.991 | 1.034 |
| WTC11 | 11,913 | 17,805 | 1.3651 | 1.4353 | 1.051 | 1.067 |

A narrower ceiling *raises* percentiles — the opposite of this rebuild's intent — and it
happened with 49% **more** positions offered, which is only possible because the population
itself changed. Verified consequence-free on what ships (0 of LegNet's committed rows pin,
checked in `validation/SORT1_rs12740374_multioracle`), but it is a composition effect on
n=3 and should be re-measured if LegNet ever gains tracks or a committed walkthrough with
strong promoter effects.

## 10b. What changed after the 2026-08-07 converged state

§10 froze a snapshot; the release continued for three more days. Recorded here so the
document does not describe a state that has been superseded — the failure mode this whole
release was about.

| date | change | evidence |
|---|---|---|
| 2026-08-08 | Cherimoya switched to the 5-fold CATv1 **ensemble**, nulls rebuilt to match (§7b) | three-way HepG2 DNase spread 2.52–3.47 → **2.52–2.75** |
| 2026-08-09 | Activity provenance **derived** per oracle instead of stamped uniformly | three populations, 31,500 / 29,500 / 34,500, each with its own sha256 (§4b) |
| 2026-08-09 | All 8 artefacts re-stamped: `schema_version 4`, one `build_id` (`2026-08-06 unified rebuild`), per-layer `sampling` block | verified present on all 8; `build_config` now exists on ChromBPNet and matches Cherimoya's schema |
| 2026-08-09 | LegNet's declared `resolution` computed from the array it actually holds | `chorus/oracles/legnet.py`; sub-region scores no longer return `None` |
| 2026-08-10 | Artefacts published to `lucapinello/chorus-backgrounds` | remote LFS sha256 verified against local on 8/8 |

Retention as shipped, read from the artefacts rather than asserted: `effect` and `summary`
are **exact** for all eight oracles; `perbin` is capped with an exact tail. No layer of any
oracle is thinned.

---

## 11. ⚠️ Open

* AlphaGenome's `perbin` carries **199** exact tail slots against an intended 200,
  because `n_expected` was estimated 0.08% low (986,976 against 987,776). The floor
  tolerates 1% for exactly this, and `derive_tail_k` now carries a 2% margin so future
  builds land at 204. Recorded rather than fixed: correcting one grid slot would cost a
  14 GPU-hour rebuild for the difference between p98.00 and p98.01.
* The **effect** and **summary/perbin** layers of an oracle are built by separate passes,
  so a partial rebuild can still put them on different populations. `unified_build: true`
  plus the two sha256 fields make that detectable; nothing yet *prevents* it.
* **Activity percentiles are ranked against three different populations** (29,500 / 31,500 /
  34,500 positions, §4), so a `summary` percentile is only strictly comparable across
  oracles that share a mixture. The mixtures overlap heavily — two are subsets of the third
  in all but the DHS stratum — so the practical effect is expected to be small, but it has
  **not been measured**, and §9's two-arm rule means it should be before anyone harmonises
  them or claims they are equivalent.
* AlphaGenome `histone_marks` and Enformer `tf_binding` remain ~20%/25% pinned on
  motif-creating variants. Irreducible with an empirical ceiling; see §9's last row.
* **Mouse is refused, not supported, and the reference class is why.** hg38 is now asserted
  at three points (§7), so a non-hg38 model can no longer reach a percentile by accident.
  What the assertion does *not* do is make mouse cheap: SCREEN publishes an mm10 cCRE
  registry but the Meuleman DHS index has **no** mouse equivalent, so the effect-null and
  baseline populations of §3–§4 would each need an mm10 variant plus its own validation that
  the `p0` gain transfers. Exposing the mouse heads of Enformer + Borzoi + AlphaGenome is
  ~4,300 further tracks, each needing a full background pass — the largest single compute
  item outstanding, and it should be costed on its own merits rather than folded into an
  oracle addition. Also note the collision trap from #121: mouse tissue names (`liver`,
  `heart`, `brain`, `forebrain`) are all *human* ENCODE CHIP biosamples, so any species
  filter must key on `(assay, cell_type)` — a name-only filter deletes 16 human rows.
* **One AlphaGenome MCP end-to-end test never runs on an authenticated machine.**
  `tests/test_integration.py:196` gates on the `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`
  environment variables, but `huggingface_hub` also authenticates from the stored token on
  disk — so on a host that is logged in (`whoami` resolves, write scope) the test still
  skips, reporting "HF_TOKEN not set", which reads as "no credentials available" when there
  are. Either gate on `HfApi().whoami()` or say "not set in the environment"; until then
  that path is unverified and the integration count is one lower than it looks.

---

## 12. Cross-oracle comparison: what is and is not comparable

Two oracles can be pointed at the same assay, the same biosample and the same ENCODE
experiment and still disagree substantially. This section records what was measured when
that happened, because the instinct — "same data, so one of them must be wrong" — sends you
looking for a bug that is not there.

### The case

At rs12740374, HepG2 DNase accessibility, all three of these are the **same** ENCODE
experiment `ENCSR149XIL` (ChromBPNet's mirror manifest resolves `DNASE/HepG2/fold_0` to
`model.chrombpnet_nobias.fold_0.ENCSR149XIL.h5`; Cherimoya loads
`models/ENCSR149XIL/cherimoya.fold_0.torch`):

| oracle | ref | alt | linear ratio | log2FC | percentile |
|---|---|---|---|---|---|
| Cherimoya | 603.3 | 2093.2 | 3.469 | +1.793 | 0.9999 |
| ChromBPNet | 287.2 | 746.9 | 2.600 | +1.376 | 0.9995 |
| AlphaGenome | 660.2 | 1666.3 | 2.524 | +1.334 | 0.9964 |

**Nothing on the chorus side accounts for it.** Verified: byte-identical model input (one
md5 for both 2,114 bp windows, variant at index 1057, forward strand); the same 501 bp span
`values[808:1309]` through one shared `LAYER_CONFIGS['chromatin_accessibility']` and one
call site; `expm1` on both count heads; the same fold-0 chromosome partition with **chr1
held out for both models**.

### Three rules that follow

**1. Never compare `ref_value` / `alt_value` across oracles.** They are model-specific
depth-normalised scales. Cherimoya's ref is 2.1× ChromBPNet's for the identical sequence,
and neither is "right". Only `raw_score` (log2FC) and `quantile_score` are cross-comparable,
and here all three agree: every one places the variant above the 99.6th percentile of its
own null.

**2. The aggregation window is doing more work than it looks.** The gap is a monotone
function of `window_bp`:

| window | Cherimoya | ChromBPNet | AlphaGenome |
|---|---|---|---|
| 51 bp | 3.62 | 3.57 | 2.51 |
| **501 bp** (shipped) | **3.47** | **2.60** | **2.52** |
| 1001 bp | 3.21 | 2.11 | 2.42 |

At 51 bp the two BPNet-family models agree to 1.6% and **both** disagree with AlphaGenome;
the curves cross at 47 bp. So "ChromBPNet and AlphaGenome corroborate each other, Cherimoya
is the outlier" is an artefact of where the curves happen to intersect at 501 bp, **not**
evidence about which model is right. Do not draw the outlier conclusion from a single
window width.

The gap decomposes exactly into a count-head term and a profile-shape term:

| | count-head FC | × shape term | = reported |
|---|---|---|---|
| Cherimoya | 3.211 | 1.081 | 3.469 |
| ChromBPNet | 2.114 | 1.230 | 2.600 |

The count heads disagree by **52%**; the shape term pulls 14% the other way. ChromBPNet's
reference profile is broader (64% of its mass inside the central 501 bp vs Cherimoya's 80%),
so widening the window inflates its denominator faster than its numerator.

**Do not narrow `window_bp` to close a gap.** Going 501 → 51 closes 95% of this one, and it
(a) invalidates every 501 bp background CDF in `*_pertrack.npz`, and (b) moves both
BPNet-family models *away* from AlphaGenome. It is curve-fitting to one variant.

**3. A single-fold checkpoint is a sample, not the model — and Cherimoya now ensembles.** Cherimoya `ENCSR149XIL` across
its own five cross-validation folds gives ratios **3.469** (fold 0, which chorus ships),
2.393, 2.716, 2.765, 2.768 — and ChromBPNet's 2.600 sits inside that range. Absolute
reference counts vary **2.49×** across folds for the identical sequence. The 5-fold
ensemble, which CATv1's own README recommends, gives 2.749 and closes **80%** of the gap.

**Resolved 2026-08-08: Cherimoya ships the 5-fold ensemble.** CATv1's model card offers
both usages — "use a single fold (e.g. `fold_0`), or average the predictions of all five
folds for a more robust estimate" — and chorus now takes the second. `CATV1_DEFAULT_FOLD`
is the sentinel `CATV1_ENSEMBLE`; pass `fold=0..4` for one fold explicitly.

Three things about that change are load-bearing, and all three are the kind of detail that
silently produces a wrong null if got wrong.

**The mean is over the expected-counts PROFILES.** Not over the two raw heads: both enter
`expected_counts_profile` non-linearly (softmax across positions, `expm1` on the count
head), so averaging heads computes a different quantity. Not over per-fold log2FCs either.
Measured at rs12740374/ENCSR149XIL the three give **1.4576** (averaging predictions — what
the card describes and what ships), — , and **1.4849** (averaging per-fold effects). The
averaged profile is mapped back onto equivalent heads by
`scoring.heads_equivalent_to_profile`, which round-trips to ~1e-15, so every caller
downstream keeps its two-head contract unchanged.

**The builder and the query path must both ensemble, or percentiles mean nothing.** The
builder called `model(batch)` directly and would have scored fold 0 while queries scored
five — a null and a numerator that are not the same statistic. `forward_window_sums` now
takes a **list** of models, and its signature is plural specifically so that mistake cannot
recur. Verified: builder path 783.983032 vs query path 783.983066 on the same sequence,
relative difference 4e-8. Averaging profiles then summing the window is identical to
averaging per-fold window sums (the sum is linear), but `compute_effect` is a log ratio and
is **not**, so ref and alt must be averaged separately and the effect taken of the averages.

**It must work in both execution modes.** An earlier version dispatched on `self._models`,
which only the in-process loader populates — so with `use_environment=True`, *the user
default*, an ensemble request silently returned fold 0 with no warning. Dispatch now keys
off `model_paths`, which both modes set. Verified bit-identical across modes:
ref 782.9413, alt 2152.1508, log2FC +1.457632.

Effect on the gap this section is about: ChromBPNet 2.600 vs Cherimoya **3.469 → 2.749**,
i.e. log2 gap 0.4174 → 0.0820, **80% closed**. Cost: 5× the forward passes (17.9 GPU-hours
for the 1,518-track rebuild, ~3h across four GPUs) and 5 subprocess calls per env-mode
query. Checkpoints are 2.4 MB each so disk is not a concern.

### Is a given disagreement unusual?

Measure it against the reference set rather than reasoning about it. Over all 18,672
`snps_accessibility` variants, ChromBPNet vs Cherimoya: mean signed difference **−0.001
log2**, Pearson **r = 0.888**, and Cherimoya is systematically *quieter* (|log2FC| ratio
0.736 at the median, 0.934 at q99). rs12740374's +33% sits at the **83rd percentile** of the
|log2FC| ≥ 0.5 stratum, in which **18–22% of loci disagree by more than 33%** — and it runs
*opposite* to the systematic trend. So it is inside the normal spread, and there is nothing
locus-specific to fix.

Full derivation, including the three adversarial verification passes that failed to refute
it, in `audits/2026-08-06_null_model_rebuild.md` addendum E.
