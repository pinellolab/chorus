"""Multi-layer variant analysis report.

Orchestrates modality-specific scoring across regulatory layers and
produces formatted reports (dict, DataFrame, markdown) suitable for
both programmatic use and Claude-driven analysis via MCP.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .scorers import (
    LAYER_CONFIGS,
    LayerConfig,
    classify_track_layer,
    score_track_effect,
)
from .normalization import QuantileNormalizer, PerTrackNormalizer
from .analysis_request import AnalysisRequest

logger = logging.getLogger(__name__)

# Display order for regulatory layers
LAYER_ORDER = [
    "chromatin_accessibility",
    "tf_binding",
    "histone_marks",
    "tss_activity",
    "gene_expression",
    "promoter_activity",
    "regulatory_classification",
    "splicing",
]


@dataclass
class TrackScore:
    """Score for a single track from modality-specific scoring.

    For expression tracks (RNA, CAGE), multiple TrackScore rows are
    created per track — one per gene and (for CAGE) one for the local
    variant site to capture transcribed enhancers.  The ``region_label``
    field distinguishes these rows.
    """

    assay_id: str
    assay_type: str
    cell_type: str
    layer: str
    ref_value: float | None
    alt_value: float | None
    raw_score: float | None
    quantile_score: float | None = None
    ref_signal_percentile: float | None = None
    description: str | None = None  # human-readable track description
    note: str | None = None
    region_label: str | None = None  # e.g. "SORT1 (exons)", "variant site"

    def to_dict(self) -> dict:
        d: dict = {
            "assay_id": self.assay_id,
            "assay_type": self.assay_type,
            "cell_type": self.cell_type,
            "layer": self.layer,
            "ref_value": self.ref_value,
            "alt_value": self.alt_value,
            "raw_score": self.raw_score,
        }
        if self.description is not None:
            d["description"] = self.description
        if self.quantile_score is not None:
            d["quantile_score"] = self.quantile_score
        if self.ref_signal_percentile is not None:
            d["ref_signal_percentile"] = self.ref_signal_percentile
        if self.note is not None:
            d["note"] = self.note
        if self.region_label is not None:
            d["region_label"] = self.region_label
        return d


@dataclass
class VariantReport:
    """Multi-layer, multi-oracle variant analysis report."""

    chrom: str
    position: int
    ref_allele: str
    alt_alleles: list[str]
    oracle_name: str
    gene_name: str | None
    allele_scores: dict[str, list[TrackScore]] = field(default_factory=dict)
    nearby_genes: list[str] = field(default_factory=list)
    # Descriptive title for the report — defaults to "Multi-Layer Variant
    # Effect Report" but should be overridden for region swaps, insertions,
    # discovery reports, etc.
    report_title: str = "Multi-Layer Variant Effect Report"
    # For region swaps / insertions: the full affected region in the genome.
    # When set, the IGV browser highlights this entire region (not just a
    # point-variant marker). Format: (start, end) in 0-based coordinates.
    modification_region: tuple[int, int] | None = None
    # Human-readable description of the modification (what was inserted/replaced,
    # how long, what it represents). Rendered in the report header.
    modification_description: str | None = None
    # Optional: preserve the user's original prompt + how the report was made
    analysis_request: AnalysisRequest | None = None
    # Optional: raw predictions for track plots (set by build_variant_report)
    _predictions: dict | None = field(default=None, repr=False)
    # Optional: normalizer for percentile-mode visualizations
    _normalizer: object | None = field(default=None, repr=False)
    # When True, IGV browser uses raw signal with autoscale instead of
    # the layer-aware floor rescale (default).  Table scores are unaffected.
    _igv_raw: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dict."""
        result: dict = {
            "variant": {
                "chrom": self.chrom,
                "position": self.position,
                "ref_allele": self.ref_allele,
                "alt_alleles": self.alt_alleles,
            },
            "oracle": self.oracle_name,
            "gene_name": self.gene_name,
            "nearby_genes": self.nearby_genes,
            "alleles": {},
        }
        if self.analysis_request is not None:
            result["analysis_request"] = self.analysis_request.to_dict()
        if self.modification_description:
            result["modification_description"] = self.modification_description
        if self.modification_region:
            result["modification_region"] = list(self.modification_region)
        for allele, scores in self.allele_scores.items():
            by_layer: dict[str, list[dict]] = {}
            for ts in scores:
                by_layer.setdefault(ts.layer, []).append(ts.to_dict())
            result["alleles"][allele] = {
                "scores_by_layer": by_layer,
                "all_scores": [ts.to_dict() for ts in scores],
            }
        return result

    def to_dataframe(self):
        """Convert to a ``pandas.DataFrame`` with one row per track per allele."""
        import pandas as pd

        rows = []
        for allele, scores in self.allele_scores.items():
            for ts in scores:
                rows.append({
                    "allele": allele,
                    "layer": ts.layer,
                    "assay_id": ts.assay_id,
                    "region_label": ts.region_label,
                    "assay_type": ts.assay_type,
                    "cell_type": ts.cell_type,
                    "ref_value": ts.ref_value,
                    "alt_value": ts.alt_value,
                    "raw_score": ts.raw_score,
                    "quantile_score": ts.quantile_score,
                    "ref_signal_percentile": ts.ref_signal_percentile,
                    "note": ts.note,
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Filename helpers
    # ------------------------------------------------------------------

    def default_filename(self, ext: str = "html") -> str:
        """Generate a descriptive default filename for this report.

        Format: ``<variant_id>_<gene>_<oracle>_report.<ext>``

        Examples:
            - ``rs12740374_SORT1_alphagenome_report.html``
            - ``chr5_1295228_G_A_TERT_chrombpnet_report.html``
        """
        # Build variant part
        alleles_str = f"{self.ref_allele}_{'_'.join(self.alt_alleles)}"
        var_part = f"{self.chrom}_{self.position}_{alleles_str}"

        # Gene part
        gene_part = self.gene_name or ""

        # Oracle part
        oracle_part = self.oracle_name.replace(" ", "_")

        parts = [p for p in [var_part, gene_part, oracle_part] if p]
        name = "_".join(parts) + f"_report.{ext}"
        # Sanitize
        name = name.replace(":", "_").replace("/", "_").replace(" ", "_")
        return name

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def to_html(self, output_path: str | None = None) -> str:
        """Generate a self-contained HTML report with color-coded tables.

        Args:
            output_path: If provided, write the HTML to this file path.
                If a directory is given (no file extension), the default
                filename is used.

        Returns:
            HTML string.
        """
        html = _build_html_report(self)
        if output_path is not None:
            from pathlib import Path

            path = Path(output_path)
            # If output_path looks like a directory, append default filename
            if path.suffix == "" or path.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                path = path / self.default_filename()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)

            path.write_text(html, encoding="utf-8")
            logger.info("HTML report written to %s", path)
        return html

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------

    def to_markdown(self, max_rows_per_layer: int = 10) -> str:
        """Generate a markdown report with tables organised by layer.

        Args:
            max_rows_per_layer: Cap the number of rows shown per regulatory
                layer. When a layer has more scored tracks than this, the
                report shows the top-*N* by effect magnitude and a footer
                row "(showing top N of M tracks — see example_output.json
                for the full set)". Default 10 keeps reports scannable;
                pass ``None`` to show everything.
        """
        lines: list[str] = []
        if self.analysis_request is not None:
            lines.append(self.analysis_request.to_markdown())
        lines.append(f"## {self.report_title}")
        lines.append("")
        lines.append(
            f"**Variant**: {self.chrom}:{self.position} "
            f"{self.ref_allele}>{','.join(self.alt_alleles)}"
        )
        lines.append(f"**Oracle**: {self.oracle_name}")
        if self.gene_name:
            lines.append(f"**Gene**: {self.gene_name}")
            if self.nearby_genes and len(self.nearby_genes) > 1:
                others = [g for g in self.nearby_genes if g != self.gene_name][:4]
                lines.append(f"**Other nearby genes**: {', '.join(others)}")
        if self.modification_description:
            lines.append(f"**Modification**: {self.modification_description}")
        if self.modification_region:
            s, e = self.modification_region
            lines.append(f"**Modified region**: {self.chrom}:{s+1:,}-{e:,} ({e-s:,} bp)")
        lines.append("")

        # Summary
        summary = _build_summary(self.allele_scores)
        lines.append(f"**Summary**: {summary}")
        lines.append("")

        has_quantile = False
        has_baseline = False
        for allele, scores in self.allele_scores.items():
            if len(self.alt_alleles) > 1:
                lines.append(f"### Allele: {allele}")
                lines.append("")

            # Group by layer
            by_layer: dict[str, list[TrackScore]] = {}
            for ts in scores:
                by_layer.setdefault(ts.layer, []).append(ts)

            # Order layers
            ordered = [l for l in LAYER_ORDER if l in by_layer]
            for l in by_layer:
                if l not in ordered:
                    ordered.append(l)

            has_quantile = any(ts.quantile_score is not None for ts in scores)
            has_baseline = any(ts.ref_signal_percentile is not None for ts in scores)

            for layer_name in ordered:
                layer_scores_full = _sort_layer_scores(by_layer[layer_name])
                total_in_layer = len(layer_scores_full)
                # Cap the rendered table so reports stay scannable. Tracks
                # after the cap are still in JSON/TSV for programmatic use.
                if max_rows_per_layer is not None and total_in_layer > max_rows_per_layer:
                    layer_scores = layer_scores_full[:max_rows_per_layer]
                    truncated = True
                else:
                    layer_scores = layer_scores_full
                    truncated = False
                cfg = LAYER_CONFIGS.get(layer_name)
                display = cfg.description if cfg else layer_name

                lines.append(f"#### {display}")
                lines.append("")

                # Build table header dynamically
                header_cols = ["Track", "Ref", "Alt", "Effect"]
                if has_quantile:
                    header_cols.append("Effect %ile")
                if has_baseline:
                    header_cols.append("Activity %ile")
                header_cols.append("Interpretation")
                lines.append("| " + " | ".join(header_cols) + " |")
                lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")

                for ts in layer_scores:
                    # Track label: include region_label if present
                    track_label = ts.description or ts.assay_id
                    if ts.region_label:
                        track_label += f" — {ts.region_label}"

                    if ts.raw_score is None:
                        note = ts.note or "Not scored"
                        n_dash = len(header_cols) - 2  # track + note
                        dash = "| — " * n_dash
                        lines.append(f"| {track_label} {dash}| {note} |")
                        continue

                    ref_str = f"{ts.ref_value:.3g}"
                    alt_str = f"{ts.alt_value:.3g}"
                    sign = "+" if ts.raw_score >= 0 else ""
                    score_str = f"{sign}{ts.raw_score:.3f}"
                    interp = _interpret_score(
                        ts.raw_score, ts.quantile_score, layer_name,
                    )

                    cols = [track_label, ref_str, alt_str, score_str]
                    if has_quantile:
                        cols.append(
                            _fmt_percentile(ts.quantile_score)
                        )
                    if has_baseline:
                        cols.append(
                            f"{ts.ref_signal_percentile:.3f}"
                            if ts.ref_signal_percentile is not None
                            else "—"
                        )
                    cols.append(interp)
                    lines.append("| " + " | ".join(cols) + " |")

                if truncated:
                    lines.append(
                        f"| _…showing top {len(layer_scores)} of "
                        f"{total_in_layer} — see `example_output.json` for "
                        f"the full set_ |" + " |" * (len(header_cols) - 1)
                    )
                lines.append("")

        # Add explanation of normalization columns
        if has_quantile or has_baseline:
            lines.append("---")
            lines.append("**Score guide:**")
            if has_quantile:
                lines.append(
                    "- **Effect %ile**: Variant effect ranked against ~10K random SNPs. "
                    "0.95 = stronger than 95% of random variants."
                )
            if has_baseline:
                lines.append(
                    "- **Activity %ile**: Reference signal ranked genome-wide against "
                    "ENCODE SCREEN cCREs + random regions. "
                    "0.95 = more active than 95% of genomic positions."
                )
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(allele_scores: dict[str, list["TrackScore"]]) -> str:
    """Build a plain-English summary of the top effects across all layers."""
    # Collect best effect per layer across all alleles
    best_per_layer: dict[str, tuple[float, str, str]] = {}  # layer → (score, track, interp)
    for allele, scores in allele_scores.items():
        for ts in scores:
            if ts.raw_score is None:
                continue
            layer = ts.layer
            abs_score = abs(ts.raw_score)
            if layer not in best_per_layer or abs_score > abs(best_per_layer[layer][0]):
                interp = _interpret_score(ts.raw_score, ts.quantile_score, layer)
                best_per_layer[layer] = (ts.raw_score, ts.assay_id, interp)

    if not best_per_layer:
        return "No scorable effects detected."

    # Sort layers by |effect| descending
    ranked = sorted(best_per_layer.items(), key=lambda x: abs(x[1][0]), reverse=True)

    # Build summary
    strong_effects = []
    for layer, (score, track, interp) in ranked:
        if abs(score) >= 0.1:
            cfg = LAYER_CONFIGS.get(layer)
            layer_name = cfg.description if cfg else layer
            sign = "+" if score >= 0 else ""
            strong_effects.append(f"{layer_name}: {interp.lower()} ({sign}{score:.2f})")

    if not strong_effects:
        return "No strong regulatory effects detected across any layer."

    return "; ".join(strong_effects) + "."


def _sort_layer_scores(layer_scores: list["TrackScore"]) -> list["TrackScore"]:
    """Sort track scores within a layer by |raw_score| descending."""
    return sorted(
        layer_scores,
        key=lambda ts: abs(ts.raw_score) if ts.raw_score is not None else -1,
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------


def _fmt_percentile(q: float | None) -> str:
    """Format a quantile score for display, avoiding misleading precision."""
    if q is None:
        return "—"
    if q >= 0.99:
        return "≥99th"
    if q <= 0.01:
        return "≤1st"
    return f"{q:.2f}"


def _interpret_score(
    raw_score: float,
    quantile_score: float | None,
    layer: str,
) -> str:
    """Generate a human-readable interpretation of a score.

    Magnitude comes from the effect percentile when available, but we ALSO
    gate on the absolute raw score so near-zero effects can never be
    labeled "very strong" just because their quantile ranks high against a
    mostly-zero background. A tiny raw effect with quantile=1.0 is still a
    tiny raw effect.
    """
    abs_raw = abs(raw_score)

    # Hard-override: a genuinely tiny raw effect is Minimal regardless of
    # what the background distribution says. Layer-specific cutoffs mirror
    # the thresholds in the README "Quick guide to magnitudes" table.
    _MIN_MAGNITUDE = {
        "chromatin_accessibility": 0.1,
        "tf_binding": 0.1,
        "histone_marks": 0.1,
        "tss_activity": 0.1,
        "gene_expression": 0.05,
        "promoter_activity": 0.05,
        "splicing": 0.05,
    }
    if abs_raw < _MIN_MAGNITUDE.get(layer, 0.1):
        return "Minimal effect"

    # Determine strength label via quantile (preferred) or raw thresholds
    if quantile_score is not None:
        mag = abs(quantile_score)
        if mag < 0.5:
            strength = "Minimal"
        elif mag < 0.75:
            strength = "Moderate"
        elif mag < 0.9:
            strength = "Strong"
        else:
            strength = "Very strong"
    else:
        if abs_raw < 0.3:
            strength = "Moderate"
        elif abs_raw < 0.7:
            strength = "Strong"
        else:
            strength = "Very strong"

    # Cap strength by raw magnitude: you can't be "very strong" at 0.15 log2FC.
    if abs_raw < 0.3 and strength in ("Strong", "Very strong"):
        strength = "Moderate"
    elif abs_raw < 0.7 and strength == "Very strong":
        strength = "Strong"

    if strength == "Minimal":
        return "Minimal effect"

    _POS_DIRECTION = {
        "chromatin_accessibility": "opening",
        "tf_binding": "binding gain",
        "histone_marks": "mark gain",
        "tss_activity": "increase",
        "gene_expression": "increase",
        "promoter_activity": "activation",
        "splicing": "increase",
    }
    _NEG_DIRECTION = {
        "chromatin_accessibility": "closing",
        "tf_binding": "binding loss",
        "histone_marks": "mark loss",
        "tss_activity": "decrease",
        "gene_expression": "decrease",
        "promoter_activity": "repression",
        "splicing": "decrease",
    }

    if raw_score > 0:
        direction = _POS_DIRECTION.get(layer, "increase")
    else:
        direction = _NEG_DIRECTION.get(layer, "decrease")

    return f"{strength} {direction}"


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _track_description(track) -> str | None:
    """Extract a human-readable description from a track's metadata.

    For CHIP tracks, ensures the TF name or histone mark is included
    (e.g. ``CHIP:CTCF:K562`` instead of just ``CHIP:K562``).
    """
    meta = getattr(track, "metadata", None)
    if meta and isinstance(meta, dict):
        desc = meta.get("description", "")
        name = meta.get("name", "")

        # For CHIP tracks: if the description lacks TF/mark info but
        # the name has it (common in AlphaGenome metadata), extract it.
        if desc and desc.startswith("CHIP:") and ":" not in desc[5:]:
            # Description is just "CHIP:cell_type" — missing TF/mark
            # Try to extract from name (e.g. "CL:0000062 TF ChIP-seq CTCF")
            if name:
                import re
                # Look for TF name after "TF ChIP-seq " or mark after "Histone ChIP-seq "
                tf_match = re.search(r'TF ChIP-seq\s+(\S+)', name)
                hist_match = re.search(r'Histone ChIP-seq\s+(\S+)', name)
                if tf_match:
                    tf_name = tf_match.group(1)
                    cell = desc.split(":", 1)[1] if ":" in desc else ""
                    return f"CHIP:{tf_name}:{cell}"
                elif hist_match:
                    mark = hist_match.group(1)
                    cell = desc.split(":", 1)[1] if ":" in desc else ""
                    return f"CHIP:{mark}:{cell}"

        if desc:
            return desc

    # Fallback: assay_type:cell_type
    at = getattr(track, "assay_type", "")
    ct = getattr(track, "cell_type", "")
    if at and ct:
        return f"{at}:{ct}"
    return None


def _parse_cell_type_from_description(description: str, assay_type: str = "") -> str:
    """Extract a proper cell type name from a track description.

    Handles various formats:
    - ``DNASE:K562`` → ``K562``
    - ``CHIP:H3K27ac:GM12878`` → ``GM12878``
    - ``CHIP:HNF4A:liver male adult (32 years)`` → ``liver``
    - ``CAGE:Clontech Human Universal...`` → ``Universal RNA control``
    - ``CAGE:adipose tissue, adult, pool1`` → ``adipose tissue``
    """
    parts = description.split(":")
    if len(parts) >= 3 and parts[0] == "CHIP":
        # CHIP:mark:cell_type or CHIP:TF:cell_type
        ct = parts[2].strip()
        # Take first word/phrase before comma
        ct = ct.split(",")[0].strip()
        return ct
    if len(parts) >= 2:
        ct = parts[1].strip()
        # Skip known commercial/control names
        control_words = {"clontech", "sabiosciences", "universal rna", "xpressref"}
        if any(w in ct.lower() for w in control_words):
            return "Universal RNA control"
        # Take first meaningful part (before comma)
        ct = ct.split(",")[0].strip()
        return ct
    return description


# When |raw_score| is below this threshold, the variant effect is
# indistinguishable from numerical noise in the oracle's forward pass. The
# per-track effect-CDFs are very densely clustered near zero (most random
# SNPs have near-zero effects), so a 1-2% raw-score drift can swing
# quantile_score by 0.5+ in that regime.  Suppress the quantile_score to
# None so readers don't over-interpret near-zero raw effects — see the
# 2026-04-16 deep audit, finding #6.
#
# Value chosen empirically: for log2fc-based layers (chromatin, TF,
# histone, CAGE, splicing) the bottom ~10% of the abs-effect CDF sits
# below 1e-3 on most tracks. For logfc gene_expression (pseudocount
# 0.001) the noise floor is ~2x smaller. A single conservative
# threshold of 1e-3 covers both without hiding real effects.
NOISE_FLOOR_RAW_SCORE = 1e-3


def _apply_normalization(
    ts: TrackScore, normalizer, oracle_name: str, layer: str,
    assay_id: str | None = None,
):
    """Apply quantile normalization to a TrackScore in place.

    Sets both ``quantile_score`` (variant effect vs background variants)
    and ``ref_signal_percentile`` (reference signal vs genome-wide baseline).

    Supports both :class:`PerTrackNormalizer` (preferred, per-track CDFs)
    and legacy :class:`QuantileNormalizer` (per-layer CDFs).

    When ``|raw_score| < NOISE_FLOOR_RAW_SCORE`` the quantile is suppressed
    to ``None`` — the CDF is too densely packed near zero to produce a
    stable percentile.
    """
    if normalizer is None:
        return

    layer_cfg = LAYER_CONFIGS.get(layer)
    use_signed = layer_cfg.signed if layer_cfg else True

    in_noise_floor = (
        ts.raw_score is not None and abs(ts.raw_score) < NOISE_FLOOR_RAW_SCORE
    )

    if isinstance(normalizer, PerTrackNormalizer) and assay_id is not None:
        # Per-track normalization
        if ts.raw_score is not None and not in_noise_floor:
            raw_for_norm = ts.raw_score if use_signed else abs(ts.raw_score)
            ts.quantile_score = normalizer.effect_percentile(
                oracle_name, assay_id, raw_for_norm, signed=use_signed,
            )
        # else: leave ts.quantile_score at its default None

        if ts.ref_value is not None:
            pctile = normalizer.activity_percentile(oracle_name, assay_id, ts.ref_value)
            if pctile is not None:
                ts.ref_signal_percentile = pctile
    else:
        # Legacy per-layer normalization
        if ts.raw_score is not None and not in_noise_floor:
            bg_key = QuantileNormalizer.background_key(oracle_name, layer)
            raw_for_norm = abs(ts.raw_score) if not use_signed else ts.raw_score
            ts.quantile_score = normalizer.normalize(
                bg_key, raw_for_norm, signed=use_signed,
            )

        if ts.ref_value is not None:
            pctile = normalizer.normalize_baseline(oracle_name, layer, ts.ref_value)
            if pctile is not None:
                ts.ref_signal_percentile = pctile


def _find_nearby_genes(
    chrom: str, pos: int, prediction_window: tuple[int, int] | None = None,
) -> list[str]:
    """Find protein-coding genes near a variant within the prediction window.

    Returns gene names sorted by distance to the variant (nearest first).
    """
    try:
        from chorus.utils.annotations import get_genes_in_region

        # Use prediction window if available, otherwise ±500kb
        if prediction_window:
            start, end = prediction_window
        else:
            start = max(0, pos - 500_000)
            end = pos + 500_000

        df = get_genes_in_region(chrom, start, end)
        if len(df) == 0:
            return []

        # Filter to protein-coding genes if feature type / gene_type available
        if "gene_type" in df.columns:
            coding = df[df["gene_type"] == "protein_coding"]
            if len(coding) > 0:
                df = coding

        # Sort by distance to variant (use gene midpoint)
        if "start" in df.columns and "end" in df.columns:
            df = df.copy()
            df["_dist"] = ((df["start"] + df["end"]) / 2 - pos).abs()
            df = df.sort_values("_dist")

        # Return gene names
        name_col = "gene_name" if "gene_name" in df.columns else "name"
        if name_col in df.columns:
            return df[name_col].dropna().unique().tolist()
        return []
    except Exception as exc:
        logger.debug("Could not find nearby genes: %s", exc)
        return []


def _describe_normalizer(normalizer) -> str:
    """One-line human-readable description of a normalizer for AnalysisRequest."""
    if normalizer is None:
        return "none"
    if isinstance(normalizer, PerTrackNormalizer):
        return "per-track background CDFs"
    if isinstance(normalizer, QuantileNormalizer):
        return "per-layer quantile background"
    return type(normalizer).__name__


def build_variant_report(
    variant_result: dict,
    oracle_name: str,
    gene_name: str | None = None,
    normalizer: QuantileNormalizer | None = None,
    igv_raw: bool = False,
    analysis_request: AnalysisRequest | None = None,
) -> VariantReport:
    """Build a multi-layer variant report from oracle predictions.

    Scores each track in *variant_result* using modality-specific
    strategies and optionally normalises against background distributions.

    When *gene_name* is not provided and RNA tracks are present, nearby
    genes within the prediction window are auto-detected and scored.

    Args:
        variant_result: Return value of ``oracle.predict_variant_effect()``.
        oracle_name: Name of the oracle that produced the predictions.
        gene_name: Optional gene name for RNA expression scoring.  When
            ``None``, nearby genes are auto-detected for non-coding
            variant interpretation.
        normalizer: Optional :class:`QuantileNormalizer` with pre-computed
            background distributions.  Used for table scores AND IGV
            visualization (rescaled view) unless ``igv_raw=True``.
        igv_raw: When ``True``, the IGV browser shows raw signal values
            with per-track autoscale instead of the layer-aware
            floor-rescale view.  Table scores still use the normalizer.

    Returns:
        A :class:`VariantReport` with per-track, per-layer scores.
    """
    predictions = variant_result["predictions"]
    variant_info = variant_result["variant_info"]

    pos_str = variant_info["position"]
    var_chrom, var_pos_str = pos_str.split(":")
    var_pos = int(var_pos_str)

    # Handle both oracle formats:
    # Real oracle: variant_info has 'ref' and 'alts'
    # MCP/test: variant_info has 'alleles' = [ref, alt1, alt2, ...]
    if "alleles" in variant_info:
        alleles = variant_info["alleles"]
        ref_allele = alleles[0] if alleles else ""
        alt_alleles = alleles[1:] if len(alleles) > 1 else []
    else:
        ref_allele = variant_info.get("ref", "")
        alt_alleles = variant_info.get("alts", [])

    # Find all protein-coding genes in the prediction window for
    # per-gene CAGE/RNA scoring.  Always do this regardless of gene_name.
    ref_pred = predictions["reference"]
    first_track = next(iter(ref_pred.values()))
    pred_start = first_track.prediction_interval.reference.start
    pred_end = first_track.prediction_interval.reference.end

    nearby_genes = _find_nearby_genes(
        var_chrom, var_pos, (pred_start, pred_end),
    )
    resolved_gene_name = gene_name
    if not resolved_gene_name and nearby_genes:
        resolved_gene_name = nearby_genes[0]
    # Make sure the user-specified gene is included and first in the list
    if resolved_gene_name and resolved_gene_name not in nearby_genes:
        nearby_genes.insert(0, resolved_gene_name)
    elif resolved_gene_name and resolved_gene_name in nearby_genes:
        nearby_genes.remove(resolved_gene_name)
        nearby_genes.insert(0, resolved_gene_name)

    logger.info(
        "Genes in prediction window: %s",
        ", ".join(nearby_genes[:10]) if nearby_genes else "none",
    )

    # Build per-gene exon maps and TSS positions for all genes
    all_gene_exons: dict[str, list[dict]] = {}
    all_gene_tss: dict[str, list[int]] = {}
    for gn in nearby_genes:
        if not gn:
            continue
        try:
            from chorus.utils.annotations import get_gene_exons as _get_exons
            edf = _get_exons(gn)
            if len(edf) > 0:
                all_gene_exons[gn] = edf[["chrom", "start", "end"]].to_dict("records")
        except Exception:
            pass
        try:
            from chorus.utils.annotations import get_gene_tss as _get_tss
            tdf = _get_tss(gn)
            if len(tdf) > 0:
                all_gene_tss[gn] = tdf["tss"].tolist()
        except Exception:
            pass

    # gene_exons for the primary gene (used by non-expression tracks if needed)
    gene_exons = all_gene_exons.get(resolved_gene_name)

    allele_scores: dict[str, list[TrackScore]] = {}

    for allele_name, alt_pred in predictions.items():
        if allele_name == "reference":
            continue

        scores: list[TrackScore] = []
        for assay_id in ref_pred.keys():
            ref_track = ref_pred[assay_id]
            alt_track = alt_pred[assay_id]

            layer = classify_track_layer(ref_track)
            at = getattr(ref_track, "assay_type", "")
            ct = getattr(ref_track, "cell_type", "")
            desc = _track_description(ref_track)

            # Fix cell type from description (handles CHIP:mark:cell, CAGE quirks)
            if desc:
                ct = _parse_cell_type_from_description(desc, at)

            # --- RNA tracks: one row per gene (exon-based scoring) ---
            if layer == "gene_expression":
                scored_any = False
                for gn, exons in all_gene_exons.items():
                    result = score_track_effect(
                        ref_track, alt_track, var_chrom, var_pos,
                        gene_exons=exons,
                    )
                    ts = TrackScore(
                        assay_id=assay_id, assay_type=at, cell_type=ct, description=desc,
                        layer=layer,
                        ref_value=result["ref_value"] if result else None,
                        alt_value=result["alt_value"] if result else None,
                        raw_score=result["raw_score"] if result else None,
                        region_label=f"{gn} (exons)",
                    )
                    if result is None:
                        ts.note = f"{gn} exons outside prediction window"
                    _apply_normalization(ts, normalizer, oracle_name, layer, assay_id=assay_id)
                    scores.append(ts)
                    scored_any = True
                if not scored_any:
                    scores.append(TrackScore(
                        assay_id=assay_id, assay_type=at, cell_type=ct, description=desc,
                        layer=layer, ref_value=None, alt_value=None,
                        raw_score=None,
                        note="No nearby genes found for exon-based scoring",
                    ))
                continue

            # --- CAGE tracks: per-gene TSS rows + variant site row ---
            if layer == "tss_activity":
                # 1) Variant site (captures transcribed enhancer)
                result = score_track_effect(
                    ref_track, alt_track, var_chrom, var_pos,
                )
                ts = TrackScore(
                    assay_id=assay_id, assay_type=at, cell_type=ct, description=desc,
                    layer=layer,
                    ref_value=result["ref_value"] if result else None,
                    alt_value=result["alt_value"] if result else None,
                    raw_score=result["raw_score"] if result else None,
                    region_label="variant site",
                )
                if result is None:
                    ts.note = "Outside scoring window"
                _apply_normalization(ts, normalizer, oracle_name, layer, assay_id=assay_id)
                scores.append(ts)

                # 2) Per-gene TSS rows (pick the most active TSS per gene)
                cage_cfg = LAYER_CONFIGS["tss_activity"]
                half_w = (cage_cfg.window_bp or 501) // 2
                from .scorers import _compute_effect
                for gn, tss_list in all_gene_tss.items():
                    best_ts = None
                    best_ref = -1.0
                    for tss_pos in tss_list:
                        tss_start = tss_pos - half_w
                        tss_end = tss_pos + half_w + 1
                        ref_v = ref_track.score_region(
                            var_chrom, tss_start, tss_end, cage_cfg.aggregation,
                        )
                        alt_v = alt_track.score_region(
                            var_chrom, tss_start, tss_end, cage_cfg.aggregation,
                        )
                        if ref_v is not None and alt_v is not None:
                            raw = _compute_effect(
                                ref_v, alt_v, cage_cfg.formula, cage_cfg.pseudocount,
                            )
                            # Keep the TSS with highest reference signal
                            if ref_v > best_ref:
                                best_ref = ref_v
                                best_ts = TrackScore(
                                    assay_id=assay_id, assay_type=at, cell_type=ct, description=desc,
                                    layer=layer,
                                    ref_value=float(ref_v),
                                    alt_value=float(alt_v),
                                    raw_score=float(raw),
                                    region_label=f"{gn} TSS",
                                )
                    if best_ts is not None:
                        _apply_normalization(best_ts, normalizer, oracle_name, layer, assay_id=assay_id)
                        scores.append(best_ts)
                continue

            # --- All other tracks: single row with standard scoring ---
            result = score_track_effect(
                ref_track, alt_track, var_chrom, var_pos,
                gene_exons=gene_exons,
            )

            ts = TrackScore(
                assay_id=assay_id, assay_type=at, cell_type=ct, description=desc,
                layer=layer,
                ref_value=result["ref_value"] if result else None,
                alt_value=result["alt_value"] if result else None,
                raw_score=result["raw_score"] if result else None,
            )
            if result is None:
                ts.note = "Outside scoring window"
            _apply_normalization(ts, normalizer, oracle_name, layer, assay_id=assay_id)
            scores.append(ts)

        allele_scores[allele_name] = scores

    # Use the resolved gene name (auto-detected or user-provided)
    report_gene = resolved_gene_name or gene_name

    # If the caller didn't build an AnalysisRequest themselves, synthesize a
    # minimal one so every report still carries oracle / normalizer / timestamp.
    if analysis_request is None:
        analysis_request = AnalysisRequest(
            oracle_name=oracle_name,
            normalizer_name=_describe_normalizer(normalizer),
        )
    else:
        if analysis_request.oracle_name is None:
            analysis_request.oracle_name = oracle_name
        if analysis_request.normalizer_name is None:
            analysis_request.normalizer_name = _describe_normalizer(normalizer)

    report = VariantReport(
        chrom=var_chrom,
        position=var_pos,
        ref_allele=ref_allele,
        alt_alleles=alt_alleles,
        oracle_name=oracle_name,
        gene_name=report_gene,
        allele_scores=allele_scores,
        analysis_request=analysis_request,
        _normalizer=normalizer,
        _igv_raw=igv_raw,
    )

    # Attach nearby genes info for the report
    if nearby_genes:
        report.nearby_genes = nearby_genes

    # Stash raw predictions for HTML track plots
    # The discovery pipeline filters to ~20 selected tracks before calling
    # build_variant_report, so this threshold must accommodate that.
    n_tracks = len(ref_pred.tracks)
    if n_tracks <= 50:
        report._predictions = predictions

    return report


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------

_CSS = """\
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
       Helvetica, Arial, sans-serif; background: #f8f9fa; color: #212529;
       padding: 2rem; max-width: 1200px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin-bottom: .3rem; }
h2 { font-size: 1.15rem; margin: 1.8rem 0 .6rem; color: #495057;
     border-bottom: 2px solid #dee2e6; padding-bottom: .3rem; }
.meta { color: #6c757d; font-size: .92rem; margin-bottom: .2rem; }
.meta b { color: #343a40; }
table { width: 100%; border-collapse: collapse; margin-bottom: 1rem;
        font-size: .88rem; }
th { background: #343a40; color: #fff; text-align: left; padding: .45rem .6rem;
     font-weight: 600; }
td { padding: .4rem .6rem; border-bottom: 1px solid #dee2e6; }
tr:hover td { background: #e9ecef; }
.bar-cell { position: relative; }
.bar { position: absolute; top: 2px; bottom: 2px; left: 0; opacity: .18;
       border-radius: 3px; }
.bar-pos { background: #28a745; }
.bar-neg { background: #dc3545; }
.badge { display: inline-block; padding: .15rem .45rem; border-radius: .25rem;
         font-size: .78rem; font-weight: 600; }
.badge-minimal  { background: #e9ecef; color: #6c757d; }
.badge-moderate { background: #fff3cd; color: #856404; }
.badge-strong   { background: #f8d7da; color: #721c24; }
.badge-vstrong  { background: #dc3545; color: #fff; }
.note { color: #6c757d; font-style: italic; }
.footer { margin-top: 2rem; font-size: .78rem; color: #adb5bd;
          border-top: 1px solid #dee2e6; padding-top: .8rem; }
"""


def _score_color_class(interpretation: str):
    """Return CSS class for interpretation badge, derived from the label itself.

    This ensures the badge color always matches the text — e.g. "Minimal effect"
    never shows a red badge even when the percentile is high.
    """
    lower = interpretation.lower()
    if "very strong" in lower:
        return "badge-vstrong"
    if "strong" in lower:
        return "badge-strong"
    if "moderate" in lower:
        return "badge-moderate"
    return "badge-minimal"


def _bar_width(raw_score, max_abs=2.0):
    """Bar width as percentage (0–100) for inline bar chart."""
    if raw_score is None:
        return 0
    return min(abs(raw_score) / max_abs * 100, 100)


def _make_track_svg(
    ref_values,
    alt_values,
    variant_idx: int | None = None,
    label: str = "",
    coord_start: int = 0,
    coord_end: int = 0,
    width: int = 700,
    height: int = 120,
) -> str:
    """Create an inline SVG with two sub-panels: signal overlay + diff track.

    Top panel: ref (grey line) and alt (blue line) signal.
    Bottom panel: alt−ref difference (green=gain, red=loss fill).
    Red dashed vertical line marks the variant position.
    Y-axis tick labels on the left.

    Args:
        ref_values: 1-D array of reference signal (zoomed window).
        alt_values: 1-D array of alternate signal (same length).
        variant_idx: Variant bin index within the array.
        label: Track label.
        coord_start: Genomic start coordinate for x-axis.
        coord_end: Genomic end coordinate for x-axis.
        width: SVG pixel width.
        height: SVG pixel height (both panels combined).
    """
    import numpy as np

    n = len(ref_values)
    if n == 0:
        return ""

    ref = np.asarray(ref_values, dtype=np.float64)
    alt = np.asarray(alt_values, dtype=np.float64)
    diff = alt - ref

    # Smooth for cleaner display
    def _smooth(arr, w=5):
        if len(arr) <= w:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="same")

    # Downsample
    max_pts = min(n, width)
    if n > max_pts:
        idx = np.linspace(0, n - 1, max_pts, dtype=int)
        ref_s = _smooth(ref)[idx]
        alt_s = _smooth(alt)[idx]
        diff_s = _smooth(diff)[idx]
        var_frac = variant_idx / n if variant_idx is not None else None
    else:
        ref_s = _smooth(ref)
        alt_s = _smooth(alt)
        diff_s = _smooth(diff)
        var_frac = variant_idx / n if variant_idx is not None else None

    # Layout
    margin_l = 48  # left margin for y-axis labels
    margin_r = 8
    margin_top = 18  # label
    gap = 6
    signal_h = (height - margin_top - gap - 8) * 0.6
    diff_h = (height - margin_top - gap - 8) * 0.4
    plot_w = width - margin_l - margin_r

    # Signal panel y-scaling
    all_signal = np.concatenate([ref_s, alt_s])
    sig_max = float(np.max(all_signal)) if np.max(all_signal) > 0 else 1.0

    # Diff panel y-scaling (symmetric)
    diff_abs_max = max(float(np.max(np.abs(diff_s))), 1e-6)

    def _x(i):
        return margin_l + i * plot_w / max(len(ref_s) - 1, 1)

    def _sig_y(v):
        return margin_top + signal_h - (v / sig_max) * signal_h

    def _diff_y(v):
        base = margin_top + signal_h + gap + diff_h / 2
        return base - (v / diff_abs_max) * (diff_h / 2)

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
           f'height="{height}" style="display:block;margin:4px 0 8px">'
           f'<rect width="{width}" height="{height}" fill="#fdfdfe" '
           f'rx="5" stroke="#d1d5db" stroke-width="0.5"/>')

    # --- Signal panel: line traces ---
    def _polyline(vals, y_fn, colour, sw="1.5"):
        pts = " ".join(f"{_x(i):.1f},{y_fn(v):.1f}" for i, v in enumerate(vals))
        return f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="{sw}"/>'

    svg += _polyline(ref_s, _sig_y, "#9ca3af", "1.2")  # ref grey
    svg += _polyline(alt_s, _sig_y, "#2563eb", "1.5")   # alt blue

    # --- Diff panel: filled area ---
    base_y = _diff_y(0)
    for i in range(len(diff_s) - 1):
        x1, x2 = _x(i), _x(i + 1)
        y1, y2 = _diff_y(diff_s[i]), _diff_y(diff_s[i + 1])
        colour = "#16a34a" if diff_s[i] >= 0 else "#dc2626"  # green/red
        svg += (f'<path d="M{x1:.1f},{base_y:.1f} L{x1:.1f},{y1:.1f} '
                f'L{x2:.1f},{y2:.1f} L{x2:.1f},{base_y:.1f} Z" '
                f'fill="{colour}" opacity="0.4"/>')
    # Zero line
    svg += (f'<line x1="{margin_l}" y1="{base_y:.1f}" '
            f'x2="{width - margin_r}" y2="{base_y:.1f}" '
            f'stroke="#9ca3af" stroke-width="0.5" stroke-dasharray="2,2"/>')

    # --- Variant marker ---
    if var_frac is not None:
        vx = margin_l + var_frac * plot_w
        svg += (f'<line x1="{vx:.1f}" y1="{margin_top}" x2="{vx:.1f}" '
                f'y2="{height - 4}" stroke="#ef4444" stroke-width="1.5" '
                f'stroke-dasharray="4,2" opacity="0.7"/>')

    # --- Y-axis labels ---
    # Signal panel
    svg += (f'<text x="{margin_l - 4}" y="{margin_top + 4}" text-anchor="end" '
            f'font-size="8" fill="#6b7280">{sig_max:.1f}</text>')
    svg += (f'<text x="{margin_l - 4}" y="{margin_top + signal_h}" text-anchor="end" '
            f'font-size="8" fill="#6b7280">0</text>')
    # Diff panel
    svg += (f'<text x="{margin_l - 4}" y="{_diff_y(diff_abs_max) + 3:.0f}" '
            f'text-anchor="end" font-size="8" fill="#16a34a">+{diff_abs_max:.2f}</text>')
    svg += (f'<text x="{margin_l - 4}" y="{_diff_y(-diff_abs_max) + 3:.0f}" '
            f'text-anchor="end" font-size="8" fill="#dc2626">-{diff_abs_max:.2f}</text>')

    # --- X-axis coordinates ---
    if coord_end > coord_start:
        for frac, anchor in [(0, "start"), (0.5, "middle"), (1.0, "end")]:
            cx = margin_l + frac * plot_w
            coord = int(coord_start + frac * (coord_end - coord_start))
            svg += (f'<text x="{cx:.0f}" y="{height - 1}" text-anchor="{anchor}" '
                    f'font-size="7" fill="#9ca3af">{coord:,}</text>')

    # --- Label & legend ---
    svg += (f'<text x="{margin_l}" y="{margin_top - 5}" font-size="10" '
            f'font-family="sans-serif" font-weight="600" fill="#1f2937">'
            f'{label}</text>')

    lx = width - 160
    svg += (f'<line x1="{lx}" y1="6" x2="{lx + 14}" y2="6" stroke="#9ca3af" stroke-width="1.2"/>'
            f'<text x="{lx + 17}" y="9" font-size="8" fill="#6b7280">ref</text>'
            f'<line x1="{lx + 35}" y1="6" x2="{lx + 49}" y2="6" stroke="#2563eb" stroke-width="1.5"/>'
            f'<text x="{lx + 52}" y="9" font-size="8" fill="#2563eb">alt</text>'
            f'<rect x="{lx + 70}" y="2" width="8" height="8" fill="#16a34a" opacity="0.4" rx="1"/>'
            f'<text x="{lx + 81}" y="9" font-size="8" fill="#6b7280">diff</text>'
            f'<line x1="{lx + 100}" y1="2" x2="{lx + 100}" y2="10" '
            f'stroke="#ef4444" stroke-width="1.2" stroke-dasharray="2,1"/>'
            f'<text x="{lx + 103}" y="9" font-size="8" fill="#ef4444">var</text>')

    svg += "</svg>"
    return svg


def _build_html_report(report: "VariantReport") -> str:
    """Build the full HTML string for a VariantReport."""
    import html as html_mod

    parts: list[str] = []
    parts.append("<!DOCTYPE html><html lang='en'><head>")
    parts.append("<meta charset='utf-8'>")
    title_parts = [report.report_title or "Variant Report"]
    if report.gene_name:
        title_parts.append(report.gene_name)
    title_parts.append(f"{report.chrom}:{report.position}")
    parts.append(f"<title>{' — '.join(title_parts)}</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append("</head><body>")

    # Header
    parts.append(f"<h1>{report.report_title}</h1>")
    if report.analysis_request is not None:
        parts.append(report.analysis_request.to_html_fragment())
    parts.append(f'<p class="meta"><b>Variant:</b> {report.chrom}:{report.position} '
                 f'{html_mod.escape(report.ref_allele)}&gt;'
                 f'{html_mod.escape(",".join(report.alt_alleles))}</p>')
    parts.append(f'<p class="meta"><b>Oracle:</b> {html_mod.escape(report.oracle_name)}</p>')
    if report.gene_name:
        parts.append(f'<p class="meta"><b>Gene:</b> {html_mod.escape(report.gene_name)}</p>')
    if report.nearby_genes and len(report.nearby_genes) > 1:
        others = [g for g in report.nearby_genes if g != report.gene_name][:4]
        parts.append(f'<p class="meta"><b>Other nearby genes:</b> '
                     f'{html_mod.escape(", ".join(others))}</p>')
    if report.modification_description:
        parts.append(f'<p class="meta"><b>Modification:</b> '
                     f'{html_mod.escape(report.modification_description)}</p>')
    if report.modification_region:
        s, e = report.modification_region
        parts.append(f'<p class="meta"><b>Modified region:</b> '
                     f'{report.chrom}:{s+1:,}-{e:,} ({e-s:,} bp)</p>')

    # Summary
    summary = _build_summary(report.allele_scores)
    parts.append(f'<p class="meta" style="margin-top:0.5rem;padding:0.6rem 0.8rem;'
                 f'background:#f0f7ff;border-left:3px solid #3b82f6;border-radius:4px">'
                 f'<b>Summary:</b> {html_mod.escape(summary)}</p>')

    # Per-allele sections
    has_quantile = False
    has_baseline = False
    for allele, scores in report.allele_scores.items():
        if len(report.alt_alleles) > 1:
            parts.append(f"<h2>Allele: {html_mod.escape(allele)}</h2>")

        # Group by layer
        by_layer: dict[str, list[TrackScore]] = {}
        for ts in scores:
            by_layer.setdefault(ts.layer, []).append(ts)

        ordered = [l for l in LAYER_ORDER if l in by_layer]
        for l in by_layer:
            if l not in ordered:
                ordered.append(l)

        has_quantile = any(ts.quantile_score is not None for ts in scores)
        has_baseline = any(ts.ref_signal_percentile is not None for ts in scores)
        # Compute max |score| for bar scaling
        raw_scores = [abs(ts.raw_score) for ts in scores if ts.raw_score is not None]
        max_abs = max(raw_scores) if raw_scores else 1.0

        for layer_name in ordered:
            layer_scores = _sort_layer_scores(by_layer[layer_name])
            cfg = LAYER_CONFIGS.get(layer_name)
            display = cfg.description if cfg else layer_name

            parts.append(f"<h2>{html_mod.escape(display)}</h2>")
            parts.append("<table><thead><tr>")
            parts.append("<th>Track</th><th>Cell Type</th>"
                         "<th>Ref</th><th>Alt</th><th>Effect</th>")
            if has_quantile:
                parts.append('<th title="Variant effect percentile vs random SNPs">Effect %ile</th>')
            if has_baseline:
                parts.append('<th title="Reference signal activity percentile genome-wide">Activity %ile</th>')
            parts.append("<th>Interpretation</th></tr></thead><tbody>")

            for ts in layer_scores:
                parts.append("<tr>")
                track_label = ts.description or ts.assay_id
                if ts.region_label:
                    track_label += f' <span style="color:#6b7280;font-weight:normal">— {ts.region_label}</span>'
                parts.append(f"<td>{track_label}</td>")
                parts.append(f"<td>{html_mod.escape(ts.cell_type)}</td>")

                if ts.raw_score is None:
                    note = ts.note or "Not scored"
                    cols = 3 + (1 if has_quantile else 0) + (1 if has_baseline else 0)
                    parts.append(f'<td colspan="{cols}" class="note">'
                                 f'{html_mod.escape(note)}</td>')
                    parts.append("</tr>")
                    continue

                parts.append(f"<td>{ts.ref_value:.3g}</td>")
                parts.append(f"<td>{ts.alt_value:.3g}</td>")

                sign = "+" if ts.raw_score >= 0 else ""
                bw = _bar_width(ts.raw_score, max_abs)
                bar_cls = "bar-pos" if ts.raw_score >= 0 else "bar-neg"
                parts.append(
                    f'<td class="bar-cell">'
                    f'<span class="bar {bar_cls}" style="width:{bw:.0f}%"></span>'
                    f'{sign}{ts.raw_score:.3f}</td>'
                )

                if has_quantile:
                    q_str = _fmt_percentile(ts.quantile_score)
                    parts.append(f"<td>{q_str}</td>")

                if has_baseline:
                    rsp_str = f"{ts.ref_signal_percentile:.3f}" if ts.ref_signal_percentile is not None else "—"
                    parts.append(f"<td>{rsp_str}</td>")

                interp = _interpret_score(ts.raw_score, ts.quantile_score, layer_name)
                badge_cls = _score_color_class(interp)
                parts.append(f'<td><span class="badge {badge_cls}">'
                             f'{html_mod.escape(interp)}</span></td>')
                parts.append("</tr>")

            parts.append("</tbody></table>")

        # --- Track signal figure (matplotlib) ---
        _render_track_figure(parts, report, allele)

    # Explanation of normalization columns
    if has_quantile or has_baseline:
        parts.append('<div style="margin-top:2rem;padding:1rem;background:#f1f3f5;'
                     'border-radius:6px;font-size:.82rem;color:#495057">')
        parts.append('<b>Score interpretation guide</b><br>')
        if has_quantile:
            parts.append(
                '<b>Effect %ile</b>: How unusual is this variant\'s effect? '
                'Compared to ~10,000 random SNPs scored genome-wide. '
                'A value of 0.95 means the effect is larger than 95% of random variants. '
                'Range [0,1] for unsigned layers (chromatin, TF, histone, TSS, splicing); '
                '[-1,1] for signed layers (gene expression, MPRA).<br>'
            )
        if has_baseline:
            parts.append(
                '<b>Activity %ile</b>: How active is this region genome-wide? '
                'Reference signal ranked against ~26,000 positions sampled from '
                'ENCODE SCREEN cCREs (promoters, enhancers, CTCF sites) and random regions. '
                'A value of 0.95 means the predicted signal here is higher than 95% of '
                'genome-wide positions — indicating a highly active regulatory element.'
            )
        parts.append('</div>')

    parts.append(f'<div class="footer">Generated by Chorus '
                 f'<code>analyze_variant_multilayer</code></div>')
    parts.append("</body></html>")

    return "\n".join(parts)


def _render_track_figure(
    parts: list[str],
    report: "VariantReport",
    allele: str,
) -> None:
    """Embed an interactive IGV.js genome browser showing ref vs alt tracks.

    The browser shows all tracks with ref (grey) and alt (coloured)
    signals overlaid.  Gene annotations come from hg38 automatically.
    A red stripe marks the variant position.  The user can zoom and pan.
    """
    predictions = report._predictions
    if predictions is None:
        return

    ref_pred = predictions.get("reference")
    alt_pred = predictions.get(allele)
    if ref_pred is None or alt_pred is None:
        return

    if not list(ref_pred.keys()):
        return

    try:
        from ._igv_report import build_igv_html

        alt_alleles_str = ",".join(report.alt_alleles) if report.alt_alleles else allele
        # Pass normalizer=None when igv_raw is set so the IGV uses raw
        # autoscale instead of the rescaled view (table scores still use
        # the normalizer, only IGV is affected).
        igv_normalizer = None if report._igv_raw else report._normalizer
        igv_html = build_igv_html(
            ref_pred, alt_pred,
            variant_chrom=report.chrom,
            variant_pos=report.position,
            ref_allele=report.ref_allele,
            alt_allele=alt_alleles_str,
            gene_name=report.gene_name,
            normalizer=igv_normalizer,
            oracle_name=report.oracle_name,
            modification_region=report.modification_region,
        )

        if igv_html:
            parts.append('<h2>Interactive Genome Browser</h2>')
            parts.append(
                '<p style="font-size:.85rem;color:#6b7280">'
                'Ref signal in grey, alt signal in colour. '
                'Red stripe marks the variant. '
                'Zoom in/out and pan to explore. '
                'Gene track (RefSeq) loaded automatically.</p>'
            )
            # Scale legend
            from .normalization import PerTrackNormalizer
            if report._igv_raw:
                parts.append(
                    '<p style="font-size:.85rem;color:#6b7280;margin-top:-.5rem">'
                    'Signal shown as <b>raw model output</b> with per-track '
                    'autoscale. Each track has its own Y-axis range — peak '
                    'heights are NOT comparable across tracks or cell types.</p>'
                )
            elif isinstance(report._normalizer, PerTrackNormalizer):
                parts.append(
                    '<p style="font-size:.85rem;color:#6b7280;margin-top:-.5rem">'
                    'Signal rescaled using each track\'s genome-wide noise '
                    'floor (p95) and peak threshold (p99): '
                    '<b>0</b> = noise floor, <b>1.0</b> = top 1% of bins '
                    'genome-wide. Peak shape preserved; tracks comparable '
                    'across cell types.</p>'
                )
            parts.append(igv_html)

    except Exception as exc:
        logger.debug("Could not render IGV browser: %s", exc)
