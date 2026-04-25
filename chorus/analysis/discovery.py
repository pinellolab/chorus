"""Variant effect discovery across all tracks and cell types.

Predicts a variant across ALL available tracks in a single oracle call,
scores each using modality-specific formulas, and ranks to reveal which
cell types and regulatory layers are most affected.

Supports two strategies:
- **All-at-once** (Enformer, Borzoi, AlphaGenome, Sei): single prediction
  pass with all tracks, score and rank.
- **Iterate-models** (ChromBPNet, LegNet): loop over all available
  (assay, cell_type) combinations, load each model, predict, score.

The unified entry point is :func:`discover_variant_effects`.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .analysis_request import AnalysisRequest

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Oracle classification: which strategy to use
# ---------------------------------------------------------------------------

_MULTI_TRACK_ORACLES = {"enformer", "borzoi", "alphagenome", "sei"}
_PER_MODEL_ORACLES = {"chrombpnet", "legnet"}


@dataclass
class CellTypeHit:
    """A cell type with a strong variant effect.

    ``best_track`` carries the human-readable display name of the scout
    track (e.g. ``DNASE:HepG2``) so tables / markdown / logs read cleanly.
    The raw AlphaGenome catalog identifier is preserved on
    ``best_track_assay_id`` for traceability and programmatic lookups.
    """
    cell_type: str
    best_track: str
    best_layer: str
    effect: float          # log2FC or diff
    abs_effect: float      # |effect|
    track_ids: list[str] = field(default_factory=list)  # all tracks for this cell type
    best_track_assay_id: str = ""  # raw catalog id for traceability


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

        # Derive a clean display name for the track ("DNASE:HepG2") so the
        # rendered markdown/table shows human-readable labels rather than
        # the raw AlphaGenome catalog identifier
        # ("DNASE/EFO:0001187 DNase-seq/.").  Keep the raw id around for
        # traceability via ``best_track_assay_id``.
        from .variant_report import _track_description
        display = _track_description(ref_track) or aid

        hits.append(CellTypeHit(
            cell_type=ct_name,
            best_track=display,
            best_track_assay_id=aid,
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
    normalizer=None,
    igv_raw: bool = False,
    oracle_name: Optional[str] = None,
    user_prompt: Optional[str] = None,
    tool_name: str = "discover_variant_cell_types",
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
        user_prompt: If provided, stamped onto each per-cell-type
            report's AnalysisRequest so the HTML renders the prompt
            the first time it's written (avoids a post-hoc rewrite).
        tool_name: AnalysisRequest.tool_name for each sub-report.

    Returns:
        Dict with discovery results and per-cell-type reports.
    """
    from .variant_report import build_variant_report
    from .analysis_request import AnalysisRequest

    chrom, pos_str = variant_position.split(":")
    pos = int(pos_str)

    resolved_oracle_name = (
        oracle_name
        or getattr(oracle, "name", None)
        or oracle.__class__.__name__.lower().replace("oracle", "")
    )

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

            analysis_request = None
            if user_prompt is not None:
                analysis_request = AnalysisRequest(
                    user_prompt=user_prompt,
                    tool_name=tool_name,
                    oracle_name=resolved_oracle_name,
                    cell_types=[hit.cell_type],
                    tracks_requested=f"top {len(track_ids)} tracks for {hit.cell_type}",
                )

            report = build_variant_report(
                variant_result,
                oracle_name=resolved_oracle_name,
                gene_name=gene_name,
                normalizer=normalizer,
                igv_raw=igv_raw,
                analysis_request=analysis_request,
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


# ---------------------------------------------------------------------------
# Unified variant discovery
# ---------------------------------------------------------------------------

@dataclass
class TrackEffect:
    """Variant effect on a single track."""

    assay_id: str
    layer: str
    cell_type: str
    description: str
    raw_score: float
    abs_score: float
    ref_value: float
    alt_value: float
    effect_pctile: float | None = None
    activity_pctile: float | None = None


def _score_all_tracks(
    variant_result: dict,
    oracle_name: str,
    normalizer=None,
) -> list[TrackEffect]:
    """Score every track in a variant result using modality-specific formulas.

    Returns a flat list of :class:`TrackEffect` sorted by |effect| descending.
    """
    from .scorers import classify_track_layer, LAYER_CONFIGS, _compute_effect
    from .normalization import QuantileNormalizer, PerTrackNormalizer

    predictions = variant_result["predictions"]
    ref_pred = predictions["reference"]
    alt_key = [k for k in predictions if k != "reference"][0]
    alt_pred = predictions[alt_key]

    var_info = variant_result.get("variant_info", {})
    pos_str = var_info.get("position", "chr1:0")
    chrom, pos_s = pos_str.split(":")
    pos = int(pos_s)

    effects: list[TrackEffect] = []

    for assay_id in ref_pred.keys():
        ref_track = ref_pred[assay_id]
        alt_track = alt_pred[assay_id]

        layer = classify_track_layer(ref_track)
        cfg = LAYER_CONFIGS.get(layer)
        if cfg is None:
            continue

        # Score using the layer's window and aggregation
        if cfg.window_bp is not None:
            half = cfg.window_bp // 2
            ref_v = ref_track.score_region(chrom, pos - half, pos + half + 1, cfg.aggregation)
            alt_v = alt_track.score_region(chrom, pos - half, pos + half + 1, cfg.aggregation)
        else:
            # Full output (MPRA, Sei classes)
            ref_v = float(np.mean(ref_track.values)) if len(ref_track.values) > 0 else None
            alt_v = float(np.mean(alt_track.values)) if len(alt_track.values) > 0 else None

        if ref_v is None or alt_v is None:
            continue

        raw_score = _compute_effect(ref_v, alt_v, cfg.formula, cfg.pseudocount)

        # Extract cell type and description from track metadata
        from .variant_report import _track_description, _parse_cell_type_from_description

        cell_type = getattr(ref_track, "cell_type", "unknown")
        at = getattr(ref_track, "assay_type", "")
        description = _track_description(ref_track) or f"{at}:{cell_type}"

        # Fix cell type from full description (handles CHIP:mark:cell, CAGE quirks)
        cell_type = _parse_cell_type_from_description(description, at)

        te = TrackEffect(
            assay_id=assay_id,
            layer=layer,
            cell_type=cell_type,
            description=description,
            raw_score=raw_score,
            abs_score=abs(raw_score),
            ref_value=ref_v,
            alt_value=alt_v,
        )

        # Add normalization if available
        if normalizer is not None:
            use_signed = cfg.signed
            raw_for_norm = abs(raw_score) if not use_signed else raw_score

            if isinstance(normalizer, PerTrackNormalizer):
                te.effect_pctile = normalizer.effect_percentile(
                    oracle_name, assay_id, raw_for_norm, signed=use_signed,
                )
                te.activity_pctile = normalizer.activity_percentile(
                    oracle_name, assay_id, ref_v,
                )
            else:
                bg_key = QuantileNormalizer.background_key(oracle_name, layer)
                te.effect_pctile = normalizer.normalize(bg_key, raw_for_norm, signed=use_signed)
                te.activity_pctile = normalizer.normalize_baseline(oracle_name, layer, ref_v)

        effects.append(te)

    effects.sort(key=lambda e: e.abs_score, reverse=True)
    return effects


def _rank_and_select(
    effects: list[TrackEffect],
    top_n_per_layer: int = 3,
    min_effect_pctile: float = 0.80,
) -> tuple[list[TrackEffect], dict[str, list[TrackEffect]]]:
    """Rank effects and select top tracks per layer.

    Returns:
        (selected_tracks, layer_rankings) where layer_rankings maps
        layer_name → sorted list of TrackEffect.
    """
    # Filter out control/reference samples (not cell-type-specific)
    _CONTROL_PATTERNS = {"universal rna control", "universal rna", "unknown"}

    by_layer: dict[str, list[TrackEffect]] = defaultdict(list)
    for te in effects:
        if te.cell_type.lower() in _CONTROL_PATTERNS:
            continue
        by_layer[te.layer].append(te)

    selected: list[TrackEffect] = []
    for layer, layer_effects in by_layer.items():
        layer_effects.sort(key=lambda e: e.abs_score, reverse=True)
        # Take top N, but only if they have meaningful effects
        for te in layer_effects[:top_n_per_layer]:
            if te.effect_pctile is not None and te.effect_pctile < min_effect_pctile:
                continue  # Skip weak effects when we have percentiles
            if te.effect_pctile is None and te.abs_score < 0.01:
                continue  # Skip negligible effects
            selected.append(te)

    return selected, dict(by_layer)


def _rank_cell_types(
    effects: list[TrackEffect],
    top_n: int = 10,
) -> list[dict]:
    """Group effects by cell type and rank by best effect."""
    by_ct: dict[str, list[TrackEffect]] = defaultdict(list)
    for te in effects:
        by_ct[te.cell_type].append(te)

    ct_ranking = []
    for ct, ct_effects in by_ct.items():
        best = max(ct_effects, key=lambda e: e.abs_score)
        ct_ranking.append({
            "cell_type": ct,
            "best_layer": best.layer,
            "best_track": best.description,
            "best_effect": best.raw_score,
            "best_abs_effect": best.abs_score,
            "effect_pctile": best.effect_pctile,
            "n_tracks": len(ct_effects),
            "layers_affected": list({e.layer for e in ct_effects if e.abs_score > 0.01}),
        })

    ct_ranking.sort(key=lambda r: r["best_abs_effect"], reverse=True)
    return ct_ranking[:top_n]


def discover_variant_effects(
    oracle,
    oracle_name: str,
    variant_position: str,
    alleles: list[str],
    top_n_per_layer: int = 3,
    top_n_cell_types: int = 10,
    gene_name: str | None = None,
    normalizer=None,
    output_path: str | None = None,
    output_filename: str | None = None,
    igv_raw: bool = False,
    analysis_request: "AnalysisRequest | None" = None,
) -> dict:
    """Discover which cell types and regulatory layers are most affected.

    Predicts variant effect across ALL available tracks, ranks by effect
    magnitude, and returns the top hits with a full report.

    Works with any oracle:
    - **Multi-track** (Enformer, Borzoi, AlphaGenome, Sei): single prediction
      pass with all tracks.
    - **Per-model** (ChromBPNet, LegNet): iterates all available models.

    Args:
        oracle: A loaded Chorus oracle.
        oracle_name: Oracle name string.
        variant_position: Variant position as "chr1:109274968".
        alleles: [ref_allele, alt_allele, ...].
        top_n_per_layer: Number of top tracks per regulatory layer.
        top_n_cell_types: Number of top cell types in ranking.
        gene_name: Optional gene for expression scoring.
        normalizer: Optional QuantileNormalizer for percentiles.
        output_path: Directory to write HTML report.
        output_filename: If provided, the HTML is written at
            ``output_path/output_filename``. Otherwise the report's
            default filename is used (``chr*_*_report.html``).
        analysis_request: Optional AnalysisRequest stamped onto the
            report so the HTML is rendered with the user prompt on
            first write (avoids a post-hoc ``to_html`` rewrite).

    Returns:
        Dict with cell_type_ranking, layer_rankings, selected tracks,
        and a VariantReport for the top tracks.
    """
    from .variant_report import build_variant_report
    from ..core.result import OraclePrediction

    chrom, pos_str = variant_position.split(":")
    pos = int(pos_str)
    name = oracle_name.lower()

    # ── Step 1: Get variant predictions for ALL tracks ──────────────
    if name in _MULTI_TRACK_ORACLES:
        logger.info("Discovery: scoring ALL tracks in single pass (%s)...", name)
        all_ids = oracle.get_all_assay_ids()
        logger.info("  %d tracks to score", len(all_ids))

        variant_result = oracle.predict_variant_effect(
            genomic_region=f"{variant_position}-{pos + 1}",
            variant_position=variant_position,
            alleles=alleles,
            assay_ids=all_ids,
        )

        all_effects = _score_all_tracks(variant_result, oracle_name, normalizer)
        logger.info("  Scored %d tracks", len(all_effects))

    elif name in _PER_MODEL_ORACLES:
        logger.info("Discovery: iterating all models for %s...", name)
        all_effects = []
        variant_result = None  # will build from best tracks later

        if name == "chrombpnet":
            from ..oracles.chrombpnet_source.chrombpnet_globals import iter_unique_models
            # Dedupe by ENCFF — the registry has aliases (`"limb"` and
            # `"limb_E12.5"` point to the same model); without dedup
            # discover_variant_effects would load the same weights twice.
            models = [
                {"assay": assay, "cell_type": cell_type}
                for assay, cell_type, _encff in iter_unique_models()
            ]
        elif name == "legnet":
            from ..oracles.legnet_source.legnet_globals import LEGNET_AVAILABLE_CELLTYPES
            models = [{"cell_type": ct} for ct in LEGNET_AVAILABLE_CELLTYPES]
        else:
            models = []

        for i, model_params in enumerate(models):
            ct = model_params.get("cell_type", "?")
            assay = model_params.get("assay", "")
            label = f"{assay}:{ct}" if assay else ct
            logger.info("  Model %d/%d: %s", i + 1, len(models), label)

            try:
                oracle.load_pretrained_model(**model_params)
                result = oracle.predict_variant_effect(
                    genomic_region=f"{variant_position}-{pos + 1}",
                    variant_position=variant_position,
                    alleles=alleles,
                    assay_ids=None,
                )
                effects = _score_all_tracks(result, oracle_name, normalizer)
                all_effects.extend(effects)

                # Keep the last result for report building
                if variant_result is None:
                    variant_result = result
            except Exception as exc:
                logger.warning("  Failed %s: %s", label, str(exc)[:100])

        all_effects.sort(key=lambda e: e.abs_score, reverse=True)
        logger.info("  Scored %d tracks across %d models", len(all_effects), len(models))

    else:
        logger.warning("Unknown oracle type: %s", name)
        return {"error": f"Unsupported oracle: {name}"}

    if not all_effects:
        return {"cell_type_ranking": [], "layer_rankings": {}, "report": None}

    # ── Step 2: Rank and select top tracks ──────────────────────────
    selected, layer_rankings = _rank_and_select(
        all_effects, top_n_per_layer=top_n_per_layer,
    )
    cell_type_ranking = _rank_cell_types(all_effects, top_n=top_n_cell_types)

    logger.info("Discovery results:")
    logger.info("  %d tracks selected across %d layers", len(selected), len(layer_rankings))
    for ct in cell_type_ranking[:5]:
        logger.info("  %s: %+.3f (%s)", ct["cell_type"], ct["best_effect"], ct["best_layer"])

    # ── Step 3: Build report with selected tracks ───────────────────
    report = None
    if variant_result is not None and selected:
        # Filter predictions to only selected tracks
        selected_ids = [te.assay_id for te in selected]
        ref_pred = variant_result["predictions"]["reference"]
        alt_key = [k for k in variant_result["predictions"] if k != "reference"][0]
        alt_pred = variant_result["predictions"][alt_key]

        filtered_ref = OraclePrediction()
        filtered_alt = OraclePrediction()
        for tid in selected_ids:
            if tid in ref_pred.tracks:
                filtered_ref.add(tid, ref_pred[tid])
            if tid in alt_pred.tracks:
                filtered_alt.add(tid, alt_pred[tid])

        if filtered_ref.tracks:
            filtered_effects = {
                alt_key: {
                    tid: variant_result["effect_sizes"][alt_key][tid]
                    for tid in selected_ids
                    if tid in variant_result["effect_sizes"].get(alt_key, {})
                }
            }
            filtered_result = {
                "predictions": {"reference": filtered_ref, alt_key: filtered_alt},
                "effect_sizes": filtered_effects,
                "variant_info": variant_result["variant_info"],
            }

            report = build_variant_report(
                filtered_result,
                oracle_name=oracle_name,
                gene_name=gene_name,
                normalizer=normalizer,
                igv_raw=igv_raw,
                analysis_request=analysis_request,
            )

            if output_path:
                import os
                if output_filename is not None:
                    html_path = os.path.join(output_path, output_filename)
                else:
                    html_path = output_path
                report.to_html(output_path=html_path)
                logger.info("Report saved to %s", html_path)

    return {
        "cell_type_ranking": cell_type_ranking,
        "layer_rankings": {
            layer: [
                {
                    "assay_id": te.assay_id,
                    "description": te.description,
                    "cell_type": te.cell_type,
                    "effect": te.raw_score,
                    "abs_effect": te.abs_score,
                    "effect_pctile": te.effect_pctile,
                    "activity_pctile": te.activity_pctile,
                }
                for te in sorted(effects, key=lambda e: e.abs_score, reverse=True)[:top_n_per_layer * 3]
            ]
            for layer, effects in layer_rankings.items()
        },
        "total_tracks_scored": len(all_effects),
        "selected_tracks": len(selected),
        "report": report,
    }
