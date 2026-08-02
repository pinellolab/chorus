"""Modality-specific scoring strategies for variant effect analysis.

Each regulatory layer requires a different scoring approach — different
window sizes, aggregation methods, and effect formulas.  This module
defines those strategies and applies them to raw oracle predictions.

Variant Effect Scoring
======================

| Layer                | Window  | Aggregation | Formula                          |
|----------------------|---------|-------------|----------------------------------|
| Chromatin (DNASE/ATAC)| 501 bp | sum         | log2[(sum_alt+1)/(sum_ref+1)]    |
| TF binding (ChIP-TF) | 501 bp | sum         | log2[(sum_alt+1)/(sum_ref+1)]    |
| Histone marks         | 2001 bp| sum         | log2[(sum_alt+1)/(sum_ref+1)]    |
| TSS activity (CAGE)   | 501 bp | sum         | log2[(sum_alt+1)/(sum_ref+1)]    |
| Gene expression (RNA)  | exons | mean        | log[(mean_alt+0.001)/(mean_ref+0.001)] |
| Promoter activity      | full  | mean        | alt - ref                        |
| Regulatory class (Sei) | full  | mean        | alt - ref                        |
| Splicing               | 501 bp| sum         | log2[(sum_alt+1)/(sum_ref+1)]    |

Baseline Signal Interpretation
==============================

Baseline background distributions (built from ~25K genomic positions)
contextualise raw predicted signal levels.  The activity percentile
indicates where a region's signal ranks genome-wide.

| Layer                | Signal Measured              | High Percentile Means            |
|----------------------|------------------------------|----------------------------------|
| chromatin_accessibility| 501 bp sum of DNASE/ATAC    | Active regulatory element        |
| tf_binding            | 501 bp sum of ChIP-TF       | Transcription factor is bound    |
| histone_marks         | 2001 bp sum of histone ChIP  | Strong histone mark deposition   |
| tss_activity          | 501 bp sum of CAGE/PRO-CAP  | Active promoter / TSS            |
| gene_expression       | Mean exon RNA-seq coverage   | Highly expressed gene            |
| splicing              | 501 bp sum of splice scores  | Active splice junction           |
| promoter_activity     | Mean MPRA activity score     | Strong synthetic promoter        |
| regulatory_classification | Mean Sei class probability | Confident regulatory class call  |
| enhancer_activity     | sqrt(DNase × H3K27ac) scalar | Strong enhancer in this cell type|

The baselines are built by the ``scripts/build_backgrounds_*.py`` scripts
using the same window/aggregation as variant scoring, scored at 20K
protein-coding TSSs and 5K random positions per oracle.  Files are stored
as ``~/.chorus/backgrounds/{oracle}_{layer}_baseline.npy`` and auto-loaded
by ``chorus.analysis.normalization.get_normalizer()``.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Known histone mark patterns in track identifiers
_HISTONE_PATTERNS = frozenset({
    "H2AFZ", "H2AZ",
    "H3K4me1", "H3K4me2", "H3K4me3",
    "H3K9ac", "H3K9me1", "H3K9me2", "H3K9me3",
    "H3K14ac",
    "H3K27ac", "H3K27me3",
    "H3K36me3",
    "H3K79me2",
    "H4K20me1",
})


@dataclass(frozen=True)
class LayerConfig:
    """Scoring parameters for a regulatory layer."""

    name: str
    description: str
    window_bp: int | None  # None → full output or gene exons
    aggregation: str  # scoring strategy passed to score_region()
    pseudocount: float
    formula: str  # 'log2fc', 'logfc', or 'diff'
    signed: bool = True  # True → quantile in [-1,1]; False → [0,1]


LAYER_CONFIGS: dict[str, LayerConfig] = {
    "chromatin_accessibility": LayerConfig(
        name="chromatin_accessibility",
        description="Chromatin accessibility (DNASE/ATAC)",
        window_bp=501,
        aggregation="sum",
        pseudocount=1.0,
        formula="log2fc",
        signed=False,  # [0,1] — magnitude of effect matters
    ),
    "tf_binding": LayerConfig(
        name="tf_binding",
        description="Transcription factor binding (ChIP-TF)",
        window_bp=501,
        aggregation="sum",
        pseudocount=1.0,
        formula="log2fc",
        signed=False,  # [0,1]
    ),
    "histone_marks": LayerConfig(
        name="histone_marks",
        description="Histone modifications (ChIP-Histone)",
        window_bp=2001,
        aggregation="sum",
        pseudocount=1.0,
        formula="log2fc",
        signed=False,  # [0,1]
    ),
    "tss_activity": LayerConfig(
        name="tss_activity",
        description="TSS activity (CAGE/PRO-CAP)",
        window_bp=501,
        aggregation="sum",
        pseudocount=1.0,
        formula="log2fc",
        signed=False,  # [0,1]
    ),
    "gene_expression": LayerConfig(
        name="gene_expression",
        description="Gene expression (RNA-seq)",
        window_bp=None,
        aggregation="mean",
        pseudocount=0.001,
        formula="logfc",
        signed=True,  # [-1,1] — direction of expression change matters
    ),
    "promoter_activity": LayerConfig(
        name="promoter_activity",
        description="Promoter activity (MPRA)",
        window_bp=None,
        aggregation="mean",
        pseudocount=0.0,
        formula="diff",
        signed=True,  # [-1,1] — activation vs repression
    ),
    "splicing": LayerConfig(
        name="splicing",
        description="Splicing (splice sites)",
        window_bp=501,
        aggregation="sum",
        pseudocount=1.0,
        formula="log2fc",
        signed=False,  # [0,1]
    ),
    "regulatory_classification": LayerConfig(
        name="regulatory_classification",
        description="Regulatory element classification (Sei)",
        window_bp=None,
        aggregation="mean",
        pseudocount=0.0,
        formula="diff",
        signed=True,  # [-1,1] — class shift direction matters
    ),
    "enhancer_activity": LayerConfig(
        name="enhancer_activity",
        description="Scalar enhancer activity (EPInformer-seq sqrt(DNase × H3K27ac))",
        window_bp=None,           # scalar oracle — score full output
        aggregation="mean",
        pseudocount=1.0,          # matches build_backgrounds_epinformerseq_v2_percell.py
        formula="log2fc",
        signed=False,             # [0,1] — magnitude of accessibility change
    ),
}


# ---------------------------------------------------------------------------
# Track → layer classification
# ---------------------------------------------------------------------------

def classify_chip_layer(assay_id: str, description: str = "") -> str:
    """Is this CHIP track a histone mark or a TF? ``histone_marks`` decides the
    2001 bp window, ``tf_binding`` the 501 bp one.

    **The single implementation, called by both the query path and
    ``scripts/build_backgrounds_alphagenome.py``.** They each had their own, and
    the two read different fields, which is #122: the builder passed
    AlphaGenome's ``description`` — which reads ``"CHIP:<cell type>"`` and carries
    **no mark name** — so 0 of 2,733 CHIP tracks were built as histone, while the
    query read ``assay_id``, which does carry the mark, and scored 1,075 of them
    at 2001 bp against a 501 bp null.

    Two rules, in order:

    1. **AlphaGenome's own output-type prefix, when present.** It already states
       the distinction, so nothing needs inferring. This is also strictly more
       correct than pattern matching: the prefix classifies all **1,116**
       CHIP_HISTONE tracks, where the 15-mark ``_HISTONE_PATTERNS`` list matches
       only 1,075 and misses **41** acetylation marks (H2AK5ac, H2BK5ac,
       H2BK12ac, H2BK120ac, H3K18ac, ...). It wrongly matches 0 CHIP_TF tracks,
       so the fallback is safe where the prefix is absent.
    2. **Mark-name patterns**, for the oracles whose identifiers carry no prefix
       (enformer, borzoi, chrombpnet). Unchanged behaviour for them —
       deliberately, since widening ``_HISTONE_PATTERNS`` would move ~105
       enformer and ~105 borzoi tracks from 501 to 2001 bp and stale two more
       backgrounds. That is filed separately, not smuggled in here.

    Defaults to ``tf_binding``, the narrower window, when nothing is decidable.
    """
    if assay_id.startswith("CHIP_HISTONE/"):
        return "histone_marks"
    if assay_id.startswith("CHIP_TF/"):
        return "tf_binding"

    upper_text = f"{assay_id} {description}".upper()
    for pattern in _HISTONE_PATTERNS:
        if pattern.upper() in upper_text:
            return "histone_marks"
    return "tf_binding"


def classify_track_layer(track) -> str:
    """Classify a prediction track into a regulatory layer.

    Uses the track's ``assay_type`` and ``assay_id`` to determine which
    layer it belongs to.  CHIP tracks are split into *tf_binding* vs
    *histone_marks* by :func:`classify_chip_layer`.
    """
    assay_type = getattr(track, "assay_type", "")
    assay_id = getattr(track, "assay_id", "")

    if assay_type in ("DNASE", "ATAC"):
        return "chromatin_accessibility"
    if assay_type == "CHIP":
        description = ""
        metadata = getattr(track, "metadata", None)
        if metadata and isinstance(metadata, dict):
            description = metadata.get("description", "") or ""
        return classify_chip_layer(assay_id, description)
    if assay_type in ("CAGE", "PRO_CAP"):
        return "tss_activity"
    if assay_type == "RNA":
        return "gene_expression"
    if assay_type == "LentiMPRA":
        return "promoter_activity"
    if assay_type in (
        "Enhancer_DNase",
        "Enhancer_H3K27ac",
        "Enhancer_H3K27ac_DNase",
    ):
        # EPInformer-seq scalar enhancer-activity tracks (DNase / H3K27ac /
        # composite). Unsigned log2fc with pseudocount=1.0, matching the
        # background CDF builder.
        return "enhancer_activity"
    if assay_type == "SPLICE_SITES":
        return "splicing"
    if assay_type == "sequence-class":
        return "regulatory_classification"
    return "other"


# ---------------------------------------------------------------------------
# Effect formulas
# ---------------------------------------------------------------------------

def _compute_effect(
    ref_value: float,
    alt_value: float,
    formula: str,
    pseudocount: float,
) -> float:
    """Compute effect score using the specified formula."""
    if formula == "log2fc":
        return math.log2((alt_value + pseudocount) / (ref_value + pseudocount))
    if formula == "logfc":
        return math.log((alt_value + pseudocount) / (ref_value + pseudocount))
    if formula == "diff":
        return alt_value - ref_value
    raise ValueError(f"Unknown formula: {formula}")


# ---------------------------------------------------------------------------
# Single-track scoring
# ---------------------------------------------------------------------------

def score_track_effect(
    ref_track,
    alt_track,
    variant_chrom: str,
    variant_pos: int,
    layer_config: Optional[LayerConfig] = None,
    gene_exons: Optional[list[dict]] = None,
) -> dict | None:
    """Score a single ref/alt track pair using modality-specific scoring.

    Args:
        ref_track: Reference allele OraclePredictionTrack.
        alt_track: Alternate allele OraclePredictionTrack.
        variant_chrom: Chromosome of the variant.
        variant_pos: Position of the variant.
        layer_config: Scoring config.  If ``None``, auto-detected from
            the track type.
        gene_exons: ``[{chrom, start, end}, ...]`` for RNA exon scoring.

    Returns:
        ``{layer, ref_value, alt_value, raw_score}`` or ``None`` if
        scoring is not possible.
    """
    layer_name = classify_track_layer(ref_track)

    if layer_config is None:
        layer_config = LAYER_CONFIGS.get(layer_name)
        if layer_config is None:
            return None

    # Gene expression → exon-based scoring
    if layer_name == "gene_expression":
        if gene_exons is None:
            return None
        # Mean over the exon MASK, i.e. divided by the number of bins actually
        # summed — not by the number of exon intervals.
        #
        # AlphaGenome's reference implementation
        # (alphagenome_research/model/variant_scoring/gene_mask.py:53-56) is
        #     ln(sum(pred * mask) / gene_mask.sum() + 1e-3)  difference,
        # where the divisor is the EXTENT of the mask. chorus divided by
        # len(gene_exons) instead — the interval count — which on real genes is
        # 251-1,736x too small (median 347x: CELSR2 34 exons / 10,981 bp,
        # BCL11A 8 / 13,884). Because the 1e-3 pseudocount is fixed it does not
        # cancel in the log ratio, so the too-small denominator left the effect
        # UNDER-damped: chorus overstated RNA effects rather than understating
        # them. Instance 4 of #144.
        #
        # Counting BINS rather than exonic BASES is what makes this correct at
        # every resolution, and it is not interchangeable:
        #   * at AlphaGenome's resolution 1 a bin IS a base, so this equals
        #     gene_mask.sum() exactly — the reference formula, unchanged;
        #   * at Borzoi's 32 bp it matches that builder's own
        #     np.mean(pred[rna_exon_bins]) units, so Borzoi's 1,543 RNA
        #     background rows stay valid. A bases denominator would replace
        #     today's overstatement with a fresh 32x understatement;
        #   * it counts what was actually summed, so exons clipped by the
        #     prediction window and bin-phase over-counting are handled rather
        #     than silently mis-weighted.
        # scorers.py reads no other resolution-dependent quantity, so this is the
        # one place the distinction bites.
        ref_total = 0.0
        alt_total = 0.0
        n_bins = 0
        for exon in gene_exons:
            ref_s = ref_track.score_region(
                exon["chrom"], exon["start"], exon["end"], "sum",
            )
            alt_s = alt_track.score_region(
                exon["chrom"], exon["start"], exon["end"], "sum",
            )
            bins = ref_track.region_bin_count(
                exon["chrom"], exon["start"], exon["end"],
            )
            if ref_s is not None and alt_s is not None and bins:
                ref_total += ref_s
                alt_total += alt_s
                n_bins += bins
        if n_bins == 0:
            return None
        ref_value = ref_total / n_bins
        alt_value = alt_total / n_bins

    elif layer_config.window_bp is not None:
        # Window-based scoring, in bin space against the SAME definition of
        # "centred window" the builders use (#144 instance 2).
        #
        # This used to build genomic coordinates and hand them to score_region,
        # which floor/ceil-expanded them back to bins. Two defects followed:
        # the span depended on where the variant fell within its bin (4 OR 5 bins
        # for window_bp=501 at 128 bp resolution), so identical settings summed
        # different spans; and that span was wider than the null's (3 bins), so
        # the percentile compared a wider statistic against a narrower reference.
        #
        # At resolution=1 this returns exactly the same bins as before for odd
        # windows, so ChromBPNet's numbers are unchanged.
        ref_value = ref_track.score_centered_window(
            variant_chrom, variant_pos, layer_config.window_bp,
            layer_config.aggregation,
        )
        alt_value = alt_track.score_centered_window(
            variant_chrom, variant_pos, layer_config.window_bp,
            layer_config.aggregation,
        )
        if ref_value is None or alt_value is None:
            return None

    else:
        # Full output scoring (LentiMPRA, Sei, etc.)
        ref_value = float(ref_track.score(layer_config.aggregation))
        alt_value = float(alt_track.score(layer_config.aggregation))

    raw_score = _compute_effect(
        ref_value, alt_value, layer_config.formula, layer_config.pseudocount,
    )

    return {
        "layer": layer_name,
        "ref_value": float(ref_value),
        "alt_value": float(alt_value),
        "raw_score": float(raw_score),
    }


# ---------------------------------------------------------------------------
# Multi-layer scoring
# ---------------------------------------------------------------------------

def score_variant_multilayer(
    variant_result: dict,
    gene_name: str | None = None,
) -> dict:
    """Score a variant across all regulatory layers.

    Takes the output of ``oracle.predict_variant_effect()`` and scores
    each track using the appropriate window, aggregation, and formula
    for its modality.

    Args:
        variant_result: Return value of ``oracle.predict_variant_effect()``.
        gene_name: Optional gene name for RNA expression scoring.

    Returns:
        ``{allele_name: {assay_id: {layer, ref_value, alt_value, raw_score}}}``.
    """
    predictions = variant_result["predictions"]
    variant_info = variant_result["variant_info"]

    pos_str = variant_info["position"]
    var_chrom, var_pos_str = pos_str.split(":")
    var_pos = int(var_pos_str)

    # Get gene exons for RNA scoring
    gene_exons = None
    if gene_name:
        try:
            from chorus.utils.annotations import get_gene_exons

            exon_df = get_gene_exons(gene_name)
            if len(exon_df) > 0:
                gene_exons = exon_df[["chrom", "start", "end"]].to_dict("records")
        except Exception as exc:
            logger.warning("Could not get exons for %s: %s", gene_name, exc)

    ref_pred = predictions["reference"]
    results: dict = {}

    for allele_name, alt_pred in predictions.items():
        if allele_name == "reference":
            continue

        allele_scores: dict = {}
        for assay_id in ref_pred.keys():
            ref_track = ref_pred[assay_id]
            alt_track = alt_pred[assay_id]

            score = score_track_effect(
                ref_track, alt_track, var_chrom, var_pos,
                gene_exons=gene_exons,
            )
            if score is not None:
                allele_scores[assay_id] = score
            else:
                layer = classify_track_layer(ref_track)
                note = None
                if layer == "gene_expression" and not gene_name:
                    note = "Requires gene_name for exon-based scoring"
                allele_scores[assay_id] = {
                    "layer": layer,
                    "ref_value": None,
                    "alt_value": None,
                    "raw_score": None,
                    "note": note,
                }

        results[allele_name] = allele_scores

    return results
