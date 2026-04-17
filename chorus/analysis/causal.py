"""Causal variant prioritization from LD-based fine-mapping.

Given a sentinel GWAS variant and its LD proxies, scores each variant
across all regulatory layers and ranks by a composite causal score that
rewards convergent multi-layer evidence.
"""

import base64
import io
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .analysis_request import AnalysisRequest
from .normalization import QuantileNormalizer
from .variant_report import TrackScore, VariantReport, build_variant_report, _describe_normalizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CausalWeights:
    """Configurable weights for the composite score formula."""

    max_effect: float = 0.35
    n_layers: float = 0.25
    convergence: float = 0.20
    ref_activity: float = 0.20

    layer_effect_threshold: float = 0.1
    activity_layers: tuple[str, ...] = (
        "chromatin_accessibility", "histone_marks",
    )


@dataclass
class CausalVariantScore:
    """Composite causal prioritization score for a single variant."""

    variant_id: str
    chrom: str
    position: int
    ref: str
    alt: str
    r2: float
    is_sentinel: bool

    # Component scores
    max_effect: float
    n_layers_affected: int
    convergence_score: float
    ref_activity: float

    # Composite
    composite: float

    # Per-layer breakdown
    per_layer_scores: dict = field(default_factory=dict)
    per_layer_directions: dict = field(default_factory=dict)
    top_layer: str = ""
    top_track: str = ""

    # Context fields
    gene_name: str = ""
    cell_type: str = ""

    # Per-track detail — keyed by assay_id, preserves raw + percentile per track
    track_scores: dict[str, "TrackScore"] = field(default_factory=dict)

    # Back-reference (not serialized)
    _variant_report: VariantReport | None = field(default=None, repr=False)


@dataclass
class CausalResult:
    """Result of causal variant prioritization."""

    sentinel_id: str
    population: str
    n_variants: int
    scores: list[CausalVariantScore]
    weights: CausalWeights
    oracle_name: str
    gene_name: str | None
    cell_types: list[str] = field(default_factory=list)
    nearby_genes: list[str] = field(default_factory=list)
    analysis_request: AnalysisRequest | None = None

    def top_candidate(self) -> CausalVariantScore:
        """Return the highest-ranked variant."""
        return self.scores[0] if self.scores else None

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd

        rows = []
        for s in self.scores:
            row = {
                "rank": rows.__len__() + 1,
                "variant_id": s.variant_id,
                "chrom": s.chrom,
                "position": s.position,
                "ref": s.ref,
                "alt": s.alt,
                "r2": s.r2,
                "is_sentinel": s.is_sentinel,
                "gene_name": s.gene_name,
                "cell_type": s.cell_type,
                "max_effect": s.max_effect,
                "n_layers_affected": s.n_layers_affected,
                "convergence": s.convergence_score,
                "ref_activity": s.ref_activity,
                "composite": s.composite,
                "top_layer": s.top_layer,
                "top_track": s.top_track,
            }
            for layer, score in s.per_layer_scores.items():
                row[f"layer_{layer}"] = score
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        result = {
            "sentinel": self.sentinel_id,
            "oracle": self.oracle_name,
            "gene_name": self.gene_name,
            "population": self.population,
            "cell_types": self.cell_types,
            "n_variants": self.n_variants,
            "weights": {
                "max_effect": self.weights.max_effect,
                "n_layers": self.weights.n_layers,
                "convergence": self.weights.convergence,
                "ref_activity": self.weights.ref_activity,
            },
            "top_candidate": {
                "variant_id": self.scores[0].variant_id,
                "composite": self.scores[0].composite,
                "max_effect": self.scores[0].max_effect,
                "n_layers_affected": self.scores[0].n_layers_affected,
                "convergence": self.scores[0].convergence_score,
                "top_layer": self.scores[0].top_layer,
            } if self.scores else None,
            "rankings": [
                {
                    "variant_id": s.variant_id,
                    "chrom": s.chrom,
                    "position": s.position,
                    "ref": s.ref,
                    "alt": s.alt,
                    "r2": s.r2,
                    "is_sentinel": s.is_sentinel,
                    "gene_name": s.gene_name,
                    "cell_type": s.cell_type,
                    "composite": round(s.composite, 4),
                    "max_effect": round(s.max_effect, 4),
                    "n_layers_affected": s.n_layers_affected,
                    "convergence": round(s.convergence_score, 4),
                    "ref_activity": round(s.ref_activity, 4),
                    "top_layer": s.top_layer,
                    "top_track": s.top_track,
                    "per_layer_scores": {
                        k: round(v, 4) for k, v in s.per_layer_scores.items()
                    },
                }
                for s in self.scores
            ],
        }
        if self.analysis_request is not None:
            result["analysis_request"] = self.analysis_request.to_dict()
        return result

    def to_html(self, output_path: str | None = None) -> str:
        """Generate a self-contained HTML report.

        Args:
            output_path: File path or directory. If directory, uses
                default filename.
        """
        html = _build_causal_html(self)
        if output_path is not None:
            from pathlib import Path

            path = Path(output_path)
            if path.suffix == "" or path.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                sentinel_safe = self.sentinel_id.replace(":", "_")
                gene = self.gene_name or "locus"
                fname = f"{sentinel_safe}_{gene}_{self.oracle_name}_causal_report.html"
                path = path / fname
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
            logger.info("Causal report written to %s", path)
        return html

    def to_markdown(self) -> str:
        """Generate a markdown summary."""
        lines: list[str] = []
        if self.analysis_request is not None:
            lines.append(self.analysis_request.to_markdown())
        lines += [
            "## Causal Variant Prioritization Report",
            "",
            f"**Sentinel**: {self.sentinel_id}",
            f"**Oracle**: {self.oracle_name}",
        ]
        if self.cell_types:
            lines.append(f"**Cell type(s)**: {', '.join(self.cell_types)}")
        if self.gene_name:
            lines.append(f"**Gene**: {self.gene_name}")
        lines.append(f"**Variants scored**: {self.n_variants}")
        lines.append("")

        # Summary
        top = self.top_candidate()
        if top:
            lines.append(
                f"**Top candidate**: {top.variant_id} "
                f"(composite={top.composite:.3f}, max_effect={top.max_effect:+.3f}, "
                f"{top.n_layers_affected} layers affected, "
                f"convergence={top.convergence_score:.2f})"
            )
            if top.is_sentinel:
                lines.append("The sentinel SNP itself is the top candidate.")
            lines.append("")

        # Ranked table — per-track columns if available
        has_tracks = any(s.track_scores for s in self.scores)
        if has_tracks:
            lines.append(self._per_track_table())
        else:
            lines.append(self._summary_table())
        lines.append("")
        return "\n".join(lines)

    def _summary_table(self) -> str:
        """Legacy table without per-track detail."""
        rows = ["| Rank | Variant | r² | Max Effect | Layers | Convergence | Composite |",
                "|------|---------|-----|-----------|--------|-------------|-----------|"]
        for i, s in enumerate(self.scores, 1):
            mark = " ★" if s.is_sentinel else ""
            sign = "+" if s.max_effect >= 0 else ""
            rows.append(
                f"| {i} | {s.variant_id}{mark} | {s.r2:.2f} "
                f"| {sign}{s.max_effect:.3f} | {s.n_layers_affected} "
                f"| {s.convergence_score:.2f} | {s.composite:.3f} |"
            )
        return "\n".join(rows)

    def _per_track_table(self) -> str:
        """Per-track columns: each scored track gets its own column."""
        from .batch_scoring import _track_display_name

        # Ordered columns from the union of all variants' track_scores
        col_order: list[str] = []
        col_labels: dict[str, str] = {}
        seen: set[str] = set()
        for s in self.scores:
            for tid, ts in s.track_scores.items():
                if tid not in seen:
                    col_order.append(tid)
                    col_labels[tid] = _track_display_name(ts)
                    seen.add(tid)

        # Build header
        header = "| Rank | Variant | r² |"
        sep = "|------|---------|-----|"
        for tid in col_order:
            header += f" {col_labels[tid]} |"
            sep += "---|"
        header += " Composite |"
        sep += "-----------|"

        rows = [header, sep]
        for i, s in enumerate(self.scores, 1):
            mark = " ★" if s.is_sentinel else ""
            row = f"| {i} | {s.variant_id}{mark} | {s.r2:.2f} |"
            for tid in col_order:
                ts = s.track_scores.get(tid)
                if ts and ts.raw_score is not None:
                    sign = "+" if ts.raw_score >= 0 else ""
                    cell = f"{sign}{ts.raw_score:.3f}"
                    if ts.quantile_score is not None:
                        cell += f" ({ts.quantile_score:.0%})"
                    row += f" {cell} |"
                else:
                    row += " — |"
            row += f" {s.composite:.3f} |"
            rows.append(row)

        rows.append("")
        rows.append("Each cell: **raw effect** (effect percentile). "
                     "Composite score combines effect magnitude, layer convergence, and baseline activity.")
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def prioritize_causal_variants(
    oracle,
    lead_variant: dict,
    ld_variants: list,
    assay_ids: list[str] | None,
    gene_name: str | None = None,
    oracle_name: str | None = None,
    weights: CausalWeights | None = None,
    normalizer: QuantileNormalizer | None = None,
    analysis_request: AnalysisRequest | None = None,
) -> CausalResult:
    """Score and rank LD variants by composite causal evidence.

    Each variant is scored multi-layer, then ranked by a composite score
    combining: (1) max effect magnitude across layers,
    (2) number of layers with meaningful effect,
    (3) directional convergence across layers (same sign = more causal),
    and (4) baseline activity at the variant site (active regions weigh more).

    Args:
        oracle: A loaded Chorus oracle.
        lead_variant: Dict with ``chrom``, ``pos``, ``ref``, ``alt``, ``id``.
        ld_variants: List of :class:`LDVariant` objects in LD with the sentinel.
        assay_ids: Track identifiers for scoring. Pass ``None`` to let the
            oracle score all tracks (recommended for AlphaGenome).
        gene_name: Target gene for expression scoring.
        weights: Scoring weights (default: CausalWeights()).
        normalizer: Optional quantile normalizer.
        analysis_request: Optional :class:`AnalysisRequest` with the user's
            prompt; rendered at the top of the report for traceability.

    Returns:
        :class:`CausalResult` with variants ranked by composite score.
        Supports ``to_markdown()``, ``to_dict()``, ``to_dataframe()``, and
        ``to_html()``. The top-ranked variant is the most likely causal SNP;
        use the per-layer scores in ``result.scores[0].per_layer_scores``
        to understand the mechanism.
    """
    if weights is None:
        weights = CausalWeights()

    if oracle_name is None:
        oracle_name = getattr(oracle, "name", None) or oracle.__class__.__name__.lower()

    sentinel_id = lead_variant.get("id", f"{lead_variant['chrom']}:{lead_variant['pos']}")

    # Score each variant
    raw_scores: list[CausalVariantScore] = []
    nearby_genes: list[str] = []
    cell_types: set[str] = set()

    for i, ldv in enumerate(ld_variants):
        logger.info(
            "Scoring variant %d/%d: %s (r²=%.2f)",
            i + 1, len(ld_variants), ldv.variant_id, ldv.r2,
        )

        try:
            position = f"{ldv.chrom}:{ldv.position}"
            region = f"{ldv.chrom}:{ldv.position}-{ldv.position + 1}"

            variant_result = oracle.predict_variant_effect(
                genomic_region=region,
                variant_position=position,
                alleles=[ldv.ref, ldv.alt],
                assay_ids=assay_ids,
            )

            report = build_variant_report(
                variant_result,
                oracle_name=oracle_name,
                gene_name=gene_name,
                normalizer=normalizer,
            )

            # Capture nearby genes from first report
            if not nearby_genes and report.nearby_genes:
                nearby_genes = report.nearby_genes
            # Record ONLY the cell type of each variant's strongest-effect
            # track — collecting all 600+ cell types scored per variant
            # would make the rendered report unreadable.
            _best_abs = 0.0
            _best_ct = ""
            for allele_scores in report.allele_scores.values():
                for ts in allele_scores:
                    if ts.raw_score is None:
                        continue
                    if abs(ts.raw_score) > _best_abs:
                        _best_abs = abs(ts.raw_score)
                        _best_ct = ts.cell_type or ""
            if _best_ct:
                cell_types.add(_best_ct)

            # Extract per-layer scores
            cs = _extract_component_scores(ldv, report, weights)
            raw_scores.append(cs)

        except Exception as exc:
            logger.error("Failed to score %s: %s", ldv.variant_id, exc, exc_info=True)
            raw_scores.append(CausalVariantScore(
                variant_id=ldv.variant_id,
                chrom=ldv.chrom,
                position=ldv.position,
                ref=ldv.ref,
                alt=ldv.alt,
                r2=ldv.r2,
                is_sentinel=ldv.is_sentinel,
                max_effect=0.0,
                n_layers_affected=0,
                convergence_score=0.0,
                ref_activity=0.0,
                composite=0.0,
                top_layer="error",
                top_track=str(exc),
            ))

    # Compute composite scores with min-max normalization
    _compute_composites(raw_scores, weights)

    # Sort by composite descending
    raw_scores.sort(key=lambda s: s.composite, reverse=True)

    # Ensure every CausalResult carries provenance metadata
    if analysis_request is None:
        analysis_request = AnalysisRequest(
            tool_name="fine_map_causal_variant",
            oracle_name=oracle_name,
            normalizer_name=_describe_normalizer(normalizer),
            tracks_requested=(
                "all oracle tracks" if not assay_ids else f"{len(assay_ids)} tracks"
            ),
            cell_types=sorted(cell_types),
        )
    else:
        if analysis_request.oracle_name is None:
            analysis_request.oracle_name = oracle_name
        if analysis_request.normalizer_name is None:
            analysis_request.normalizer_name = _describe_normalizer(normalizer)
        if not analysis_request.cell_types:
            analysis_request.cell_types = sorted(cell_types)

    return CausalResult(
        sentinel_id=sentinel_id,
        population="",
        cell_types=sorted(cell_types),
        n_variants=len(raw_scores),
        scores=raw_scores,
        weights=weights,
        oracle_name=oracle_name,
        gene_name=gene_name,
        nearby_genes=nearby_genes,
        analysis_request=analysis_request,
    )


def _extract_component_scores(
    ldv,
    report: VariantReport,
    weights: CausalWeights,
) -> CausalVariantScore:
    """Extract the 4 component scores from a VariantReport."""
    # Get first allele's scores
    allele_key = None
    for k in report.allele_scores:
        allele_key = k
        break
    track_scores = report.allele_scores.get(allele_key, [])

    # Per-layer max scores
    per_layer: dict[str, float] = {}
    per_layer_dir: dict[str, int] = {}
    top_layer = ""
    top_track = ""
    max_effect = 0.0
    ref_activity_vals: list[float] = []

    for ts in track_scores:
        if ts.raw_score is None:
            continue

        layer = ts.layer
        abs_score = abs(ts.raw_score)

        # Track max per layer
        if layer not in per_layer or abs_score > abs(per_layer[layer]):
            per_layer[layer] = ts.raw_score
            per_layer_dir[layer] = 1 if ts.raw_score > 0 else (-1 if ts.raw_score < 0 else 0)

        # Overall max
        if abs_score > abs(max_effect):
            max_effect = ts.raw_score
            top_layer = layer
            top_track = ts.assay_id

        # Ref activity for chromatin/histone
        if layer in weights.activity_layers and ts.ref_value is not None:
            ref_activity_vals.append(ts.ref_value)

    # n_layers_affected
    n_layers = sum(
        1 for score in per_layer.values()
        if abs(score) >= weights.layer_effect_threshold
    )

    # Convergence score
    signs = [
        d for layer, d in per_layer_dir.items()
        if d != 0 and abs(per_layer[layer]) >= weights.layer_effect_threshold
    ]
    if signs:
        convergence = abs(sum(signs)) / len(signs)
    else:
        convergence = 0.0

    # Ref activity
    ref_activity = float(np.mean(ref_activity_vals)) if ref_activity_vals else 0.0

    # Extract gene name from the report
    gene = report.gene_name or ""

    # Collect per-track scores keyed by assay_id
    per_track: dict[str, TrackScore] = {}
    for ts in track_scores:
        per_track[ts.assay_id] = ts

    return CausalVariantScore(
        variant_id=ldv.variant_id,
        chrom=ldv.chrom,
        position=ldv.position,
        ref=ldv.ref,
        alt=ldv.alt,
        r2=ldv.r2,
        is_sentinel=ldv.is_sentinel,
        max_effect=max_effect,
        n_layers_affected=n_layers,
        convergence_score=convergence,
        ref_activity=ref_activity,
        composite=0.0,  # computed later
        per_layer_scores=per_layer,
        per_layer_directions=per_layer_dir,
        top_layer=top_layer,
        top_track=top_track,
        gene_name=gene,
        cell_type="",
        track_scores=per_track,
        _variant_report=report,
    )


def _compute_composites(
    scores: list[CausalVariantScore],
    weights: CausalWeights,
) -> None:
    """Compute composite scores with min-max normalization in-place."""
    if not scores:
        return

    # Collect raw component values
    effects = [abs(s.max_effect) for s in scores]
    layers = [float(s.n_layers_affected) for s in scores]
    convs = [s.convergence_score for s in scores]
    refs = [s.ref_activity for s in scores]

    def _minmax(vals):
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12:
            return [0.5] * len(vals)
        return [(v - lo) / (hi - lo) for v in vals]

    norm_effects = _minmax(effects)
    norm_layers = _minmax(layers)
    norm_convs = _minmax(convs)
    norm_refs = _minmax(refs)

    for i, s in enumerate(scores):
        s.composite = (
            weights.max_effect * norm_effects[i]
            + weights.n_layers * norm_layers[i]
            + weights.convergence * norm_convs[i]
            + weights.ref_activity * norm_refs[i]
        )


# ---------------------------------------------------------------------------
# Locus plot
# ---------------------------------------------------------------------------

def render_locus_plot(result: CausalResult, dpi: int = 150) -> str:
    """Render a locus plot as base64-encoded PNG.

    Two panels: scatter plot (composite vs position, colored by r²)
    and gene track below.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    scores = result.scores
    if not scores:
        return ""

    positions = [s.position for s in scores]
    composites = [s.composite for s in scores]
    r2s = [s.r2 for s in scores]

    # LocusZoom-style r² colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "r2", ["#357EBD", "#46B8DA", "#5CB85C", "#F0AD4E", "#D43F3A"], N=256,
    )
    norm = mcolors.Normalize(vmin=0, vmax=1)

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 4.5), gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax = axes[0]
    # Plot all variants as circles
    sc = ax.scatter(
        positions, composites, c=r2s, cmap=cmap, norm=norm,
        s=50, edgecolors="#333", linewidths=0.5, zorder=3,
    )

    # Highlight sentinel (diamond)
    for s in scores:
        if s.is_sentinel:
            ax.scatter(
                [s.position], [s.composite], c=[s.r2], cmap=cmap, norm=norm,
                s=120, marker="D", edgecolors="black", linewidths=1.5, zorder=5,
            )

    # Highlight top candidate (star, if different from sentinel)
    top = scores[0]
    if not top.is_sentinel:
        ax.scatter(
            [top.position], [top.composite], c=[top.r2], cmap=cmap, norm=norm,
            s=200, marker="*", edgecolors="goldenrod", linewidths=1.5, zorder=5,
        )

    ax.set_ylabel("Composite Causal Score", fontsize=10)
    ax.set_title(
        f"Causal Variant Prioritization — {result.sentinel_id} "
        f"({result.gene_name or 'locus'})",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("r² with sentinel", fontsize=9)

    # Gene track
    ax2 = axes[1]
    _draw_simple_gene_track(ax2, scores, result.gene_name)

    plt.tight_layout()

    # Encode as base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _draw_simple_gene_track(ax, scores, gene_name=None):
    """Draw a simplified gene track below the locus plot."""
    positions = [s.position for s in scores]
    min_pos = min(positions) - 5000
    max_pos = max(positions) + 5000

    chrom = scores[0].chrom

    ax.set_xlim(min_pos, max_pos)
    ax.set_ylim(-0.5, 0.5)
    ax.set_ylabel("Genes", fontsize=9)
    ax.set_xlabel(f"Position on {chrom}", fontsize=9)

    # Try to fetch genes
    try:
        from chorus.utils.annotations import get_genes_in_region
        df = get_genes_in_region(chrom, min_pos, max_pos)
        if "gene_type" in df.columns:
            coding = df[df["gene_type"] == "protein_coding"]
            if len(coding) > 0:
                df = coding

        name_col = "gene_name" if "gene_name" in df.columns else "name"

        drawn = set()
        for _, row in df.iterrows():
            gn = row.get(name_col, "")
            if gn in drawn:
                continue
            drawn.add(gn)

            start = row.get("start", min_pos)
            end = row.get("end", max_pos)
            strand = row.get("strand", "+")

            color = "#D43F3A" if gn == gene_name else "#666"
            fontweight = "bold" if gn == gene_name else "normal"

            y = 0.1 if strand == "+" else -0.1
            ax.plot([start, end], [y, y], color=color, linewidth=3, solid_capstyle="butt")
            mid = (start + end) / 2
            ax.text(mid, y + 0.2, gn, ha="center", va="bottom", fontsize=7,
                    color=color, fontweight=fontweight)

    except Exception as exc:
        logger.debug("Could not draw genes: %s", exc)

    ax.axhline(0, color="#ddd", linewidth=0.5)
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CAUSAL_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 1rem; background: #f8f9fa; }
h1 { font-size: 1.5rem; margin-bottom: 0.3rem; }
h2 { font-size: 1.15rem; border-bottom: 1px solid #dee2e6; padding-bottom: 0.3rem;
     margin-top: 1.5rem; }
.meta { font-size: 0.92rem; margin: 0.15rem 0; color: #495057; }
.summary-box { margin: 0.8rem 0; padding: 0.7rem 1rem; background: #f0f7ff;
               border-left: 3px solid #3b82f6; border-radius: 4px; font-size: 0.95rem; }
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-bottom: 1rem; }
thead { background: #343a40; color: white; }
th, td { padding: 0.4rem 0.6rem; text-align: left; border-bottom: 1px solid #dee2e6; }
th { cursor: pointer; user-select: none; }
th:hover { background: #495057; }
tbody tr:hover { background: #e9ecef; }
.badge-sentinel { background: #ffc107; color: #333; padding: 0.15rem 0.4rem;
                  border-radius: 3px; font-size: 0.8rem; font-weight: 600; }
.badge-top { background: #28a745; color: white; padding: 0.15rem 0.4rem;
             border-radius: 3px; font-size: 0.8rem; font-weight: 600; }
.r2-cell { font-weight: 600; }
.r2-high { color: #D43F3A; }
.r2-med { color: #F0AD4E; }
.r2-low { color: #357EBD; }
details { margin: 0.5rem 0; border: 1px solid #dee2e6; border-radius: 4px; }
details summary { padding: 0.5rem 0.8rem; background: #f8f9fa; cursor: pointer;
                  font-size: 0.92rem; }
details summary:hover { background: #e9ecef; }
details[open] summary { border-bottom: 1px solid #dee2e6; }
.detail-card { padding: 0.6rem 0.8rem; }
.detail-card table { font-size: 0.85rem; }
.detail-card th { background: #6c757d; }
.locus-plot { text-align: center; margin: 1rem 0; }
.locus-plot img { max-width: 100%; border: 1px solid #dee2e6; border-radius: 4px; }
.footer { margin-top: 2rem; padding-top: 0.5rem; border-top: 1px solid #dee2e6;
          font-size: 0.8rem; color: #6c757d; }
"""

_SORT_JS = """
document.querySelectorAll('.sortable th').forEach(th => {
  th.addEventListener('click', function() {
    const table = this.closest('table');
    const idx = Array.from(this.parentNode.children).indexOf(this);
    const rows = Array.from(table.querySelectorAll('tbody tr'));
    const asc = this.dataset.sort !== 'asc';
    rows.sort((a, b) => {
      let va = a.children[idx]?.textContent?.trim() || '';
      let vb = b.children[idx]?.textContent?.trim() || '';
      let na = parseFloat(va), nb = parseFloat(vb);
      if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
      return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    });
    this.dataset.sort = asc ? 'asc' : 'desc';
    const tbody = table.querySelector('tbody');
    rows.forEach(r => tbody.appendChild(r));
  });
});
"""


_IGV_CDN = "https://cdn.jsdelivr.net/npm/igv@3.1.1/dist/igv.min.js"

# r² to RGB for IGV variant annotations
def _r2_to_rgb(r2: float) -> str:
    if r2 >= 0.8: return "212,63,58"     # red
    if r2 >= 0.6: return "240,173,78"    # orange
    if r2 >= 0.4: return "92,184,92"     # green
    if r2 >= 0.2: return "70,184,218"    # light blue
    return "53,126,189"                   # blue


def _build_causal_igv(result: CausalResult) -> str:
    """Build an IGV.js browser with composite score + signal tracks."""
    import json as json_mod

    scores = result.scores
    if not scores:
        return ""

    chrom = scores[0].chrom
    positions = [s.position for s in scores]
    # Show wider region to include nearby genes (±500kb or prediction window)
    center = (min(positions) + max(positions)) // 2
    half_window = max(500_000, (max(positions) - min(positions)) // 2 + 50_000)
    min_pos = center - half_window
    max_pos = center + half_window

    # Find top 3 candidates with predictions for signal tracks
    top_with_preds_list: list = []
    for s in scores:
        if s._variant_report and s._variant_report._predictions:
            top_with_preds_list.append(s)
            if len(top_with_preds_list) >= 3:
                break
    top_with_preds = top_with_preds_list[0] if top_with_preds_list else None

    tracks = []

    # Track 1: Composite causal score as wig (bar chart)
    score_features = []
    for s in scores:
        score_features.append({
            "chr": s.chrom,
            "start": s.position - 50,
            "end": s.position + 50,
            "value": round(s.composite, 4),
        })
    tracks.append({
        "name": "Composite Causal Score",
        "type": "wig",
        "height": 60,
        "color": "rgb(59,130,246)",
        "min": 0,
        "max": 1.0,
        "features": score_features,
    })

    # Track 2: Max effect magnitude per variant
    effect_features = []
    for s in scores:
        effect_features.append({
            "chr": s.chrom,
            "start": s.position - 50,
            "end": s.position + 50,
            "value": round(abs(s.max_effect), 4),
        })
    tracks.append({
        "name": "Max |Effect|",
        "type": "wig",
        "height": 50,
        "color": "rgb(220,53,69)",
        "min": 0,
        "features": effect_features,
    })

    # Track 3: LD variant annotations with r² coloring
    variant_features = []
    for s in scores:
        label = s.variant_id
        if s.is_sentinel:
            label += " (sentinel)"
        rgb = _r2_to_rgb(s.r2)
        variant_features.append({
            "chr": s.chrom,
            "start": s.position - 1,
            "end": s.position + max(len(s.ref), 1),
            "name": f"{label} r²={s.r2:.2f} composite={s.composite:.3f}",
            "color": f"rgb({rgb})",
        })
    tracks.append({
        "name": "LD Variants (colored by r²)",
        "type": "annotation",
        "displayMode": "EXPANDED",
        "height": 40,
        "features": variant_features,
    })

    # Track 4+: Signal tracks from top 3 candidates' predictions
    _TOP_VARIANT_COLORS = [
        {"r": "59,130,246", "label": "#1"},   # blue
        {"r": "245,158,11", "label": "#2"},    # amber
        {"r": "139,92,246", "label": "#3"},    # purple
    ]
    for vi, top_s in enumerate(top_with_preds_list):
        if not top_s._variant_report or not top_s._variant_report._predictions:
            continue
        preds = top_s._variant_report._predictions
        ref_pred = preds.get("reference")
        alt_key = next((k for k in preds if k != "reference"), None)
        alt_pred = preds.get(alt_key) if alt_key else None

        if ref_pred and alt_pred:
            from .scorers import classify_track_layer
            from ._igv_report import _downsample_to_features, _LAYER_COLORS, _REF_COLOR

            assay_ids = list(ref_pred.keys())
            first_track = ref_pred[assay_ids[0]]
            pred_start = first_track.prediction_interval.reference.start
            pred_end = first_track.prediction_interval.reference.end
            window_bp = pred_end - pred_start
            bin_size = max(1, window_bp // 3000)
            alt_rgb = _TOP_VARIANT_COLORS[vi]["r"] if vi < len(_TOP_VARIANT_COLORS) else "70,130,180"

            for aid in assay_ids:
                ref_t = ref_pred[aid]
                alt_t = alt_pred[aid]
                t_start = ref_t.prediction_interval.reference.start
                t_res = ref_t.resolution

                ref_feats = _downsample_to_features(
                    ref_t.values, chrom, t_start, t_res, bin_size,
                )
                alt_feats = _downsample_to_features(
                    alt_t.values, chrom, t_start, t_res, bin_size,
                )

                group_id = f"{aid}_{top_s.variant_id}".replace(":", "_").replace(" ", "_")
                rank_label = _TOP_VARIANT_COLORS[vi]["label"] if vi < len(_TOP_VARIANT_COLORS) else f"#{vi+1}"
                tracks.append({
                    "name": f"{aid} ({rank_label} {top_s.variant_id})",
                    "type": "merged",
                    "height": 60,
                    "tracks": [
                        {
                            "type": "wig",
                            "name": f"{aid} ref",
                            "color": f"rgb({_REF_COLOR})",
                            "autoscale": True,
                            "autoscaleGroup": group_id,
                            "features": ref_feats,
                        },
                        {
                            "type": "wig",
                            "name": f"{aid} alt",
                            "color": f"rgb({alt_rgb})",
                            "autoscale": True,
                            "autoscaleGroup": group_id,
                            "features": alt_feats,
                        },
                    ],
                })

    # Sentinel ROI
    sentinel = next((s for s in scores if s.is_sentinel), scores[0])
    roi = [{
        "name": "Sentinel",
        "color": "rgba(255, 0, 0, 0.12)",
        "features": [{
            "chr": sentinel.chrom,
            "start": sentinel.position - 1,
            "end": sentinel.position + max(len(sentinel.ref), 1),
        }],
    }]

    igv_options = {
        "genome": "hg38",
        "locus": f"{chrom}:{min_pos}-{max_pos}",
        "showRuler": True,
        "showNavigation": True,
        "showCenterGuide": True,
        "roi": roi,
        "tracks": tracks,
    }

    options_json = json_mod.dumps(igv_options, separators=(",", ":"))

    # Prefer the locally-cached igv.min.js (populated on first report
    # generation by _igv_report._ensure_igv_local) so the rendered HTML is
    # self-contained offline. Fall back to the CDN tag if the cache is
    # unavailable for any reason.
    from chorus.analysis._igv_report import _ensure_igv_local
    _local_igv = _ensure_igv_local()
    _igv_script_tag = (
        f"<script>{_local_igv.read_text()}</script>" if _local_igv is not None
        else f'<script src="{_IGV_CDN}"></script>'
    )

    return f"""
<div id="igv-causal" style="margin: 1rem 0; min-height: 400px;"></div>
{_igv_script_tag}
<script>
(async function() {{
    try {{
        const browser = await igv.createBrowser(
            document.getElementById("igv-causal"),
            {options_json}
        );
        console.log("IGV causal browser created");
    }} catch(e) {{
        console.error("IGV error:", e);
        document.getElementById("igv-causal").innerHTML =
            '<p style="color:red;padding:1rem">Error loading IGV browser: ' + e.message + '</p>';
    }}
}})();
</script>
"""


def _build_causal_html(result: CausalResult) -> str:
    """Build the full HTML string for a CausalResult."""
    import html as html_mod

    from .scorers import LAYER_CONFIGS

    p = []
    p.append("<!DOCTYPE html><html lang='en'><head>")
    p.append("<meta charset='utf-8'>")
    p.append(f"<title>Causal Prioritization — {result.sentinel_id}</title>")
    p.append(f"<style>{_CAUSAL_CSS}</style>")
    p.append("</head><body>")

    # Header
    p.append("<h1>Causal Variant Prioritization Report</h1>")
    if result.analysis_request is not None:
        p.append(result.analysis_request.to_html_fragment())
    p.append(f'<p class="meta"><b>Sentinel:</b> {html_mod.escape(result.sentinel_id)}</p>')
    p.append(f'<p class="meta"><b>Oracle:</b> {html_mod.escape(result.oracle_name)}</p>')
    if result.cell_types:
        p.append(f'<p class="meta"><b>Cell type(s):</b> {html_mod.escape(", ".join(result.cell_types))}</p>')
    if result.gene_name:
        p.append(f'<p class="meta"><b>Gene:</b> {html_mod.escape(result.gene_name)}</p>')
    if result.nearby_genes:
        others = [g for g in result.nearby_genes if g != result.gene_name][:5]
        if others:
            p.append(f'<p class="meta"><b>Nearby genes:</b> {html_mod.escape(", ".join(others))}</p>')
    p.append(f'<p class="meta"><b>Variants scored:</b> {result.n_variants}</p>')

    # Summary
    top = result.top_candidate()
    if top:
        layers_desc = []
        for layer, score in sorted(top.per_layer_scores.items(),
                                    key=lambda x: abs(x[1]), reverse=True):
            if abs(score) >= 0.1:
                cfg = LAYER_CONFIGS.get(layer)
                name = cfg.description if cfg else layer
                sign = "+" if score >= 0 else ""
                layers_desc.append(f"{name}: {sign}{score:.2f}")

        summary_parts = [f"<b>Top candidate:</b> {html_mod.escape(top.variant_id)} "
                         f"(composite={top.composite:.3f})"]
        if layers_desc:
            summary_parts.append(". ".join(layers_desc))
        if top.is_sentinel:
            summary_parts.append("The sentinel SNP is the top candidate.")

        p.append(f'<div class="summary-box">{"<br>".join(summary_parts)}</div>')

    # IGV genome browser with composite score track + signal tracks
    try:
        igv_html = _build_causal_igv(result)
        if igv_html:
            p.append('<h2>Interactive Genome Browser</h2>')
            p.append(
                '<p style="font-size:.85rem;color:#6b7280">'
                'Top track: composite causal score per variant (bar height = score). '
                'Signal tracks show ref (grey) vs alt (colored) for the top candidate. '
                'Red stripe marks the sentinel. Zoom and pan to explore.</p>'
            )
            p.append(igv_html)
    except Exception as exc:
        logger.debug("Could not render IGV browser: %s", exc)

    # Note: static locus plot removed — the IGV browser above provides
    # the same information interactively (composite score track + variant
    # annotations + gene track + signal overlays).

    # Collect all layer names present across variants for per-layer columns
    all_layers: list[str] = []
    seen_layers: set[str] = set()
    for s in result.scores:
        for layer in s.per_layer_scores:
            if layer not in seen_layers:
                seen_layers.add(layer)
                all_layers.append(layer)

    # Ranked table
    p.append('<h2>Variant Rankings</h2>')
    p.append('<table class="sortable"><thead><tr>')
    p.append('<th>Rank</th><th>Variant</th><th>r²</th>')
    p.append('<th>Gene</th><th>Cell Type</th>')
    p.append('<th>Max Effect</th><th>Layers</th><th>Convergence</th>')
    for layer in all_layers:
        cfg = LAYER_CONFIGS.get(layer)
        col_name = cfg.description if cfg else layer
        p.append(f'<th>{html_mod.escape(col_name)}</th>')
    p.append('<th>Composite ▾</th>')
    p.append('</tr></thead><tbody>')

    for i, s in enumerate(result.scores, 1):
        p.append("<tr>")
        p.append(f"<td>{i}</td>")

        # Variant cell with badges
        badges = ""
        if s.is_sentinel:
            badges += ' <span class="badge-sentinel">sentinel</span>'
        if i == 1:
            badges += ' <span class="badge-top">top</span>'
        p.append(f"<td>{html_mod.escape(s.variant_id)}{badges}</td>")

        # r² colored
        r2_cls = "r2-high" if s.r2 >= 0.8 else ("r2-med" if s.r2 >= 0.5 else "r2-low")
        p.append(f'<td class="r2-cell {r2_cls}">{s.r2:.2f}</td>')

        # Gene and Cell Type
        gene = html_mod.escape(s.gene_name) if s.gene_name else "—"
        ct = html_mod.escape(s.cell_type) if s.cell_type else "—"
        p.append(f"<td>{gene}</td>")
        p.append(f"<td>{ct}</td>")

        sign = "+" if s.max_effect >= 0 else ""
        p.append(f"<td>{sign}{s.max_effect:.3f}</td>")
        p.append(f"<td>{s.n_layers_affected}</td>")
        p.append(f"<td>{s.convergence_score:.2f}</td>")

        # Per-layer score columns with color coding
        for layer in all_layers:
            score = s.per_layer_scores.get(layer)
            if score is None:
                p.append('<td style="color:#adb5bd">—</td>')
            else:
                if abs(score) < 0.05:
                    color = "#6c757d"  # grey for minimal
                elif score > 0:
                    color = "#28a745"  # green for positive
                else:
                    color = "#dc3545"  # red for negative
                layer_sign = "+" if score >= 0 else ""
                p.append(f'<td style="color:{color};font-weight:600">'
                         f'{layer_sign}{score:.3f}</td>')

        p.append(f"<td><b>{s.composite:.3f}</b></td>")
        p.append("</tr>")

    p.append("</tbody></table>")

    # Per-variant detail cards
    p.append('<h2>Variant Details</h2>')

    for i, s in enumerate(result.scores, 1):
        open_attr = " open" if i == 1 else ""
        badges = ""
        if s.is_sentinel:
            badges += " ★ sentinel"
        if i == 1:
            badges += " — Top candidate"

        p.append(f'<details{open_attr}>')
        p.append(f'<summary><b>{html_mod.escape(s.variant_id)}</b> — '
                 f'composite: {s.composite:.3f}{badges}</summary>')
        p.append('<div class="detail-card">')

        # Extract cell type from this variant's report
        var_cell_types = set()
        if s._variant_report:
            for allele_scores in s._variant_report.allele_scores.values():
                for ts in allele_scores:
                    if ts.cell_type:
                        var_cell_types.add(ts.cell_type)
        ct_str = f' | <b>Cell type:</b> {html_mod.escape(", ".join(sorted(var_cell_types)))}' if var_cell_types else ""

        p.append(f'<p><b>Position:</b> {s.chrom}:{s.position:,} {s.ref}>{s.alt} '
                 f'| <b>r²:</b> {s.r2:.2f} '
                 f'| <b>Top layer:</b> {s.top_layer}{ct_str}</p>')

        # Per-layer table
        if s.per_layer_scores:
            p.append('<table><thead><tr><th>Layer</th><th>Effect</th>'
                     '<th>Direction</th></tr></thead><tbody>')
            sorted_layers = sorted(s.per_layer_scores.items(),
                                   key=lambda x: abs(x[1]), reverse=True)
            for layer, score in sorted_layers:
                cfg = LAYER_CONFIGS.get(layer)
                name = cfg.description if cfg else layer
                sign = "+" if score >= 0 else ""
                direction = "▲" if score > 0 else ("▼" if score < 0 else "—")
                color = "#28a745" if score > 0 else ("#dc3545" if score < 0 else "#6c757d")
                p.append(f'<tr><td>{html_mod.escape(name)}</td>'
                         f'<td>{sign}{score:.3f}</td>'
                         f'<td style="color:{color};font-weight:bold">{direction}</td></tr>')
            p.append('</tbody></table>')

        p.append('</div></details>')

    p.append(f'<div class="footer">Generated by Chorus '
             f'<code>fine_map_causal_variant</code></div>')
    p.append(f"<script>{_SORT_JS}</script>")
    p.append("</body></html>")

    return "\n".join(p)
