"""Integration simulation — predict effects of inserting a construct.

Nearly identical to region swap analysis, but uses
``oracle.predict_region_insertion_at()`` to insert a sequence at a
specific position rather than replacing a region.
"""

import logging
from typing import Optional

from .normalization import QuantileNormalizer
from .variant_report import VariantReport, build_variant_report

logger = logging.getLogger(__name__)


def simulate_integration(
    oracle,
    position: str,
    construct_sequence: str,
    assay_ids: list[str],
    gene_name: str | None = None,
    normalizer: QuantileNormalizer | None = None,
    oracle_name: str | None = None,
) -> VariantReport:
    """Predict effects of inserting a construct at a genomic position.

    Compares wild-type predictions against insertion predictions across
    all regulatory layers.

    Args:
        oracle: A loaded Chorus oracle.
        position: Insertion point as "chr1:1050000".
        construct_sequence: DNA sequence to insert.
        assay_ids: List of assay identifiers.
        gene_name: Optional gene name for expression scoring.
        normalizer: Optional quantile normalizer.

    Returns:
        A :class:`VariantReport` with wild-type as "reference" and
        insertion as the alternate allele.
    """
    chrom, pos_str = position.split(":")
    pos = int(pos_str)

    # Get wild-type prediction using (chrom, start, end) tuple so oracle
    # fetches genomic sequence, not interprets string as raw DNA.
    wt_pred = oracle.predict((chrom, pos, pos + 1), assay_ids)

    # Get insertion prediction
    ins_result = oracle.predict_region_insertion_at(
        genomic_position=position,
        seq=construct_sequence,
        assay_ids=assay_ids,
    )
    ins_pred = ins_result["raw_predictions"]

    # Build pseudo variant_result for build_variant_report
    variant_result = {
        "predictions": {
            "reference": wt_pred,
            "insertion": ins_pred,
        },
        "variant_info": {
            "position": position,
            "alleles": ["wt", "insertion"],
        },
    }

    report = build_variant_report(
        variant_result,
        oracle_name=oracle_name or getattr(oracle, "name", None) or oracle.__class__.__name__.lower().replace("oracle", ""),
        gene_name=gene_name,
        normalizer=normalizer,
    )
    report.report_title = "Integration Simulation Report"
    report.modification_region = (pos, pos + len(construct_sequence))
    report.modification_description = (
        f"Inserted {len(construct_sequence):,} bp construct at "
        f"{chrom}:{pos+1:,}"
    )

    return report
