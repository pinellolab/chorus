"""Cell-type discovery mode for variant analysis.

Predicts a variant across ALL available tracks in a single oracle call,
then ranks cell types by the largest effect magnitude.  Returns the
top cell types with all their tracks for full multi-layer analysis.

This enables "discovery mode" where you don't know which cell type is
relevant — the model tells you where the variant has the strongest impact.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CellTypeHit:
    """A cell type with a strong variant effect."""
    cell_type: str
    best_track: str
    best_layer: str
    effect: float          # log2FC or diff
    abs_effect: float      # |effect|
    track_ids: list[str] = field(default_factory=list)  # all tracks for this cell type


def discover_cell_types(
    oracle,
    variant_position: str,
    alleles: list[str],
    reference_fasta: Optional[str] = None,
    top_n: int = 5,
    min_effect: float = 0.2,
    scout_types: tuple[str, ...] = ("DNASE", "ATAC"),
) -> list[CellTypeHit]:
    """Discover which cell types are most affected by a variant.

    Predicts the variant effect on all DNASE/ATAC tracks (one per cell
    type), scores each, and returns the top cell types ranked by effect
    magnitude.

    Args:
        oracle: A loaded Chorus oracle (must support all-track prediction).
        variant_position: Variant position as "chr1:1050000".
        alleles: [ref_allele, alt_allele, ...].
        reference_fasta: Path to reference FASTA (if not set on oracle).
        top_n: Number of top cell types to return.
        min_effect: Minimum |log2FC| to include a cell type.
        scout_types: Track types to use for initial screening.

    Returns:
        List of CellTypeHit sorted by descending |effect|.
    """
    from .scorers import classify_track_layer, LAYER_CONFIGS, _compute_effect

    # Get all available scout track IDs
    try:
        from chorus.oracles.alphagenome_source.alphagenome_metadata import get_metadata
        meta = get_metadata()
    except ImportError:
        logger.warning("Discovery mode requires AlphaGenome metadata")
        return []

    scout_ids = []
    for scout_type in scout_types:
        df = meta.search_tracks(scout_type)
        if len(df) > 0:
            for _, row in df.iterrows():
                aid = row["identifier"]
                # Only take the unstranded DNASE/ATAC tracks (one per cell type)
                if aid.endswith("/."):
                    scout_ids.append(aid)

    if not scout_ids:
        logger.warning("No scout tracks found for types: %s", scout_types)
        return []

    logger.info("Screening %d cell types with %s tracks...",
                len(scout_ids), "/".join(scout_types))

    # Parse variant position
    chrom, pos_str = variant_position.split(":")
    pos = int(pos_str)

    # Predict variant effect on ALL scout tracks
    try:
        variant_result = oracle.predict_variant_effect(
            genomic_region=f"{variant_position}-{pos + 1}",
            variant_position=variant_position,
            alleles=alleles,
            assay_ids=scout_ids,
        )
    except Exception as exc:
        logger.error("Scout prediction failed: %s", exc)
        return []

    # Score each scout track
    predictions = variant_result["predictions"]
    ref_pred = predictions["reference"]
    # Get the first alt allele
    alt_key = [k for k in predictions if k != "reference"][0]
    alt_pred = predictions[alt_key]

    hits: list[CellTypeHit] = []

    for aid in scout_ids:
        if aid not in ref_pred.tracks or aid not in alt_pred.tracks:
            continue

        ref_track = ref_pred[aid]
        alt_track = alt_pred[aid]

        layer = classify_track_layer(ref_track)
        cfg = LAYER_CONFIGS.get(layer)
        if cfg is None:
            continue

        # Score in the standard window
        half = (cfg.window_bp or 501) // 2
        ref_v = ref_track.score_region(chrom, pos - half, pos + half + 1, cfg.aggregation)
        alt_v = alt_track.score_region(chrom, pos - half, pos + half + 1, cfg.aggregation)

        if ref_v is None or alt_v is None:
            continue

        effect = _compute_effect(ref_v, alt_v, cfg.formula, cfg.pseudocount)

        # Extract cell type from track identifier
        # Format: DNASE/<cell_type_id> DNase-seq/.
        parts = aid.split("/", 1)
        if len(parts) < 2:
            continue
        ct_part = parts[1].split(" ")[0]  # e.g. "EFO:0001187"

        # Look up the cell type name
        ct_name = ct_part
        ct_df = meta.search_tracks(ct_part)
        if len(ct_df) > 0 and "cell_type" in ct_df.columns:
            names = ct_df["cell_type"].dropna().unique()
            if len(names) > 0:
                ct_name = names[0]

        # Get ALL tracks for this cell type
        all_ct_tracks = ct_df["identifier"].tolist() if len(ct_df) > 0 else [aid]

        hits.append(CellTypeHit(
            cell_type=ct_name,
            best_track=aid,
            best_layer=layer,
            effect=effect,
            abs_effect=abs(effect),
            track_ids=all_ct_tracks,
        ))

    # Sort by |effect| descending
    hits.sort(key=lambda h: h.abs_effect, reverse=True)

    # Filter by minimum effect and take top N
    filtered = [h for h in hits if h.abs_effect >= min_effect][:top_n]

    if filtered:
        logger.info("Top cell types by variant effect:")
        for h in filtered:
            direction = "+" if h.effect > 0 else ""
            logger.info("  %s: %s%0.3f log2FC (%s, %d total tracks)",
                        h.cell_type, direction, h.effect,
                        h.best_track, len(h.track_ids))

    return filtered


def discover_and_report(
    oracle,
    variant_position: str,
    alleles: list[str],
    gene_name: Optional[str] = None,
    top_n: int = 3,
    min_effect: float = 0.15,
    output_path: Optional[str] = None,
):
    """Discovery mode: find top cell types, then build full reports.

    Args:
        oracle: A loaded Chorus oracle.
        variant_position: Variant position as "chr1:1050000".
        alleles: [ref_allele, alt_allele].
        gene_name: Optional gene to focus expression analysis on.
        top_n: Number of top cell types to analyze in detail.
        min_effect: Minimum |log2FC| to consider a cell type.
        output_path: Directory to write HTML reports to.

    Returns:
        Dict with discovery results and per-cell-type reports.
    """
    from .variant_report import build_variant_report

    chrom, pos_str = variant_position.split(":")
    pos = int(pos_str)

    # Step 1: Discover top cell types
    logger.info("Step 1: Screening all cell types for variant effect...")
    hits = discover_cell_types(
        oracle, variant_position, alleles,
        top_n=top_n, min_effect=min_effect,
    )

    if not hits:
        logger.warning("No cell types with effect >= %s found", min_effect)
        return {"hits": [], "reports": {}}

    # Step 2: For each top cell type, predict with all tracks and build report
    reports = {}
    for i, hit in enumerate(hits):
        logger.info("Step 2.%d: Full analysis in %s (%d tracks)...",
                     i + 1, hit.cell_type, len(hit.track_ids))

        # Limit tracks to keep prediction manageable (max 30)
        track_ids = hit.track_ids[:30]

        try:
            variant_result = oracle.predict_variant_effect(
                genomic_region=f"{variant_position}-{pos + 1}",
                variant_position=variant_position,
                alleles=alleles,
                assay_ids=track_ids,
            )

            report = build_variant_report(
                variant_result,
                oracle_name=getattr(oracle, "name", "oracle"),
                gene_name=gene_name,
            )

            reports[hit.cell_type] = report

            if output_path:
                import os
                # Build descriptive filename: variant_gene_celltype_oracle_report.html
                ct_safe = hit.cell_type.replace(" ", "_").replace("/", "_")[:50]
                base_name = report.default_filename("html")
                # Insert cell type before _report.html
                base_name = base_name.replace("_report.html", f"_{ct_safe}_report.html")
                html_path = os.path.join(output_path, base_name)
                report.to_html(output_path=html_path)
                logger.info("  Report saved to %s", html_path)

        except Exception as exc:
            logger.error("  Failed for %s: %s", hit.cell_type, exc)

    return {
        "hits": [
            {
                "cell_type": h.cell_type,
                "effect": h.effect,
                "abs_effect": h.abs_effect,
                "best_track": h.best_track,
                "n_tracks": len(h.track_ids),
            }
            for h in hits
        ],
        "reports": reports,
    }
