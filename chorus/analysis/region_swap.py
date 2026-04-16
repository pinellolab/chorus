"""Region swap analysis — predict effects of replacing a genomic region.

Reuses the multi-layer variant report infrastructure: a region replacement
produces wild-type vs replaced predictions, structurally identical to
reference vs alternate allele scoring.
"""

import logging
from typing import Optional

from .normalization import QuantileNormalizer
from .variant_report import VariantReport, build_variant_report

logger = logging.getLogger(__name__)


def analyze_region_swap(
    oracle,
    region: str,
    replacement_sequence: str,
    assay_ids: list[str],
    gene_name: str | None = None,
    normalizer: QuantileNormalizer | None = None,
    oracle_name: str | None = None,
) -> VariantReport:
    """Predict effects of replacing a genomic region with a custom sequence.

    Compares wild-type predictions against replacement predictions across
    all regulatory layers using the same scoring infrastructure as variant
    analysis.

    Args:
        oracle: A loaded Chorus oracle.
        region: Genomic region to replace as "chr1:1000000-1001000".
        replacement_sequence: DNA sequence to insert in place of the region.
        assay_ids: List of assay identifiers.
        gene_name: Optional gene name for expression scoring.
        normalizer: Optional quantile normalizer.

    Returns:
        A :class:`VariantReport` with wild-type as "reference" and
        replacement as the alternate allele.
    """
    # Parse region to get position info
    chrom, coords = region.split(":")
    start_str, end_str = coords.split("-")
    start, end = int(start_str), int(end_str)
    midpoint = (start + end) // 2

    # Get wild-type prediction using (chrom, start, end) tuple so oracle
    # fetches genomic sequence, not interprets region string as raw DNA.
    wt_pred = oracle.predict((chrom, start, end), assay_ids)

    # Get replacement prediction
    swap_result = oracle.predict_region_replacement(
        genomic_region=region,
        seq=replacement_sequence,
        assay_ids=assay_ids,
    )
    swap_pred = swap_result["raw_predictions"]

    # Build a pseudo variant_result dict compatible with build_variant_report
    variant_result = {
        "predictions": {
            "reference": wt_pred,
            "replacement": swap_pred,
        },
        "variant_info": {
            "position": f"{chrom}:{midpoint}",
            "alleles": ["wt", "replacement"],
        },
    }

    report = build_variant_report(
        variant_result,
        oracle_name=oracle_name or getattr(oracle, "name", None) or oracle.__class__.__name__.lower().replace("oracle", ""),
        gene_name=gene_name,
        normalizer=normalizer,
    )
    report.report_title = "Region Swap Analysis Report"
    report.modification_region = (start, end)
    report.modification_description = (
        f"Replaced {end - start:,} bp region ({chrom}:{start+1:,}-{end:,}) "
        f"with a {len(replacement_sequence):,} bp custom sequence"
    )

    return report
