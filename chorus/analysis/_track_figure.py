"""Matplotlib-based track figures for variant effect HTML reports.

Produces two figures for a variant analysis:

1. **Zoom-in** (~2kb around variant): chromatin accessibility, TF binding,
   histone marks — shows *what the variant alters locally* (peak shape,
   binding site creation/disruption).

2. **Zoom-out** (full prediction window): CAGE at TSS, RNA-seq across exons,
   gene annotation — shows *how the variant affects target gene expression*
   at a distance.

Both are returned as base64-encoded PNGs for embedding in self-contained HTML.
"""

import base64
import io
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Track colours matching coolbox_params from OraclePredictionTrack subclasses
_LAYER_COLORS = {
    "chromatin_accessibility": "#1f77b4",  # blue  (DNASE/ATAC)
    "tf_binding": "#d62728",              # red   (ChIP-TF)
    "histone_marks": "#e377c2",           # pink  (ChIP-Histone)
    "tss_activity": "#ff7f0e",            # orange (CAGE)
    "gene_expression": "#9467bd",         # purple (RNA)
    "promoter_activity": "#ff7f0e",       # orange (LentiMPRA)
    "splicing": "#8c564b",               # brown
    "regulatory_classification": "#17becf",  # cyan (Sei)
}

# Layers that go in the zoom-in figure (local regulatory context)
_ZOOM_IN_LAYERS = {
    "chromatin_accessibility", "tf_binding", "histone_marks",
    "splicing", "promoter_activity", "regulatory_classification",
}

# Layers that go in the zoom-out figure (distal gene expression effects)
_ZOOM_OUT_LAYERS = {
    "tss_activity", "gene_expression",
}


def render_track_figures(
    ref_pred,
    alt_pred,
    variant_chrom: str,
    variant_pos: int,
    gene_name: Optional[str] = None,
    zoom_in_bp: int = 2000,
    dpi: int = 150,
    normalizer=None,
    oracle_name: Optional[str] = None,
    normalize: bool = True,
) -> dict[str, str]:
    """Render zoom-in and zoom-out track figures as base64-encoded PNGs.

    By default (``normalize=True``) the panels are CDF-rescaled to the
    canonical 0–3.0 (unsigned) or ±3.0 (signed) scale used by the IGV /
    CoolBox / notebook paths, so every renderer shares one source of
    truth: ``1.0 = genome-wide p99``.  The normalizer is auto-loaded
    from ``~/.chorus/backgrounds/`` (or the HuggingFace mirror) using
    the first track's ``source_model`` as the oracle name.

    Pass ``normalize=False`` to opt out and get autoscaled raw values
    (legacy behaviour).  Pass an explicit ``normalizer=`` /
    ``oracle_name=`` to override the auto-load.
    """
    from .scorers import classify_track_layer

    assay_ids = list(ref_pred.keys())
    if not assay_ids:
        return {}

    # Classify tracks into zoom-in vs zoom-out
    zoom_in_ids = []
    zoom_out_ids = []
    for aid in assay_ids:
        layer = classify_track_layer(ref_pred[aid])
        if layer in _ZOOM_OUT_LAYERS:
            zoom_out_ids.append(aid)
        else:
            zoom_in_ids.append(aid)

    # If no natural split, put everything in zoom-in
    if not zoom_out_ids:
        zoom_in_ids = assay_ids

    # Get prediction window bounds
    first = ref_pred[assay_ids[0]]
    pred_start = first.prediction_interval.reference.start
    pred_end = first.prediction_interval.reference.end

    # Auto-load normalizer + oracle_name from the track's source_model
    # so default usage matches the IGV / CoolBox panels without the
    # caller having to pass anything.  Pass ``normalize=False`` to opt out.
    if normalize:
        if oracle_name is None:
            oracle_name = getattr(first, "source_model", None)
        if normalizer is None and oracle_name:
            try:
                from chorus.analysis.normalization import get_normalizer
                normalizer = get_normalizer(oracle_name)
            except Exception:
                normalizer = None

    results = {}

    # --- Zoom-in figure ---
    if zoom_in_ids:
        b64 = _render_figure(
            ref_pred, alt_pred, zoom_in_ids,
            variant_chrom, variant_pos, gene_name,
            x_start=variant_pos - zoom_in_bp // 2,
            x_end=variant_pos + zoom_in_bp // 2,
            pred_start=pred_start, pred_end=pred_end,
            title=f"Zoom-in: local regulatory context ({variant_chrom}:{variant_pos:,})",
            show_genes=True,
            dpi=dpi,
            normalizer=normalizer,
            oracle_name=oracle_name,
        )
        if b64:
            results["zoom_in"] = b64

    # --- Zoom-out figure ---
    if zoom_out_ids:
        b64 = _render_figure(
            ref_pred, alt_pred, zoom_out_ids,
            variant_chrom, variant_pos, gene_name,
            x_start=pred_start, x_end=pred_end,
            pred_start=pred_start, pred_end=pred_end,
            title=f"Zoom-out: gene expression effects ({variant_chrom}:{variant_pos:,})",
            show_genes=True,
            show_exons=True,
            dpi=dpi,
            normalizer=normalizer,
            oracle_name=oracle_name,
        )
        if b64:
            results["zoom_out"] = b64

    return results


# Keep old single-figure API for backwards compat
def render_track_figure(ref_pred, alt_pred, variant_chrom, variant_pos,
                        gene_name=None, zoom_bp=2000, dpi=150,
                        normalizer=None, oracle_name=None,
                        normalize: bool = True):
    """Render track figures. Returns the zoom-in PNG (or first available)."""
    figs = render_track_figures(
        ref_pred, alt_pred, variant_chrom, variant_pos,
        gene_name=gene_name, zoom_in_bp=zoom_bp, dpi=dpi,
        normalizer=normalizer, oracle_name=oracle_name, normalize=normalize,
    )
    return figs.get("zoom_in", figs.get("zoom_out", ""))


# ---------------------------------------------------------------------------
# Core rendering
# ---------------------------------------------------------------------------

def _render_figure(
    ref_pred, alt_pred, assay_ids,
    variant_chrom, variant_pos, gene_name,
    x_start, x_end, pred_start, pred_end,
    title="", show_genes=True, show_exons=False, dpi=150,
    normalizer=None, oracle_name=None,
) -> str:
    """Render a multi-panel figure for a subset of tracks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    from .scorers import classify_track_layer, LAYER_CONFIGS

    n_tracks = len(assay_ids)
    if n_tracks == 0:
        return ""

    # Gene annotation
    genes_df = _get_genes_for_region(variant_chrom, pred_start, pred_end)
    has_genes = show_genes and genes_df is not None and len(genes_df) > 0

    # Exons for the highlighted gene
    exons = None
    if show_exons and gene_name:
        exons = _get_exons_for_gene(gene_name)

    # Layout
    n_panels = n_tracks + (1 if has_genes else 0)
    track_h = 1.6
    gene_h = 0.9
    height_ratios = [track_h] * n_tracks + ([gene_h] if has_genes else [])
    fig_height = track_h * n_tracks + (gene_h if has_genes else 0) + 0.8

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(10, fig_height), sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.05},
    )
    if n_panels == 1:
        axes = [axes]

    # Clamp display range
    x_start = max(x_start, pred_start)
    x_end = min(x_end, pred_end)

    for i, assay_id in enumerate(assay_ids):
        ax = axes[i]
        ref_track = ref_pred[assay_id]
        alt_track = alt_pred[assay_id]

        layer = classify_track_layer(ref_track)
        cfg = LAYER_CONFIGS.get(layer)
        color = _LAYER_COLORS.get(layer, "#2563eb")

        _plot_track_panel(
            ax, ref_track, alt_track,
            x_start, x_end, variant_pos,
            assay_id=assay_id, color=color,
            show_legend=(i == 0),
            show_exon_shading=(show_exons and exons is not None),
            exons=exons,
            normalizer=normalizer,
            oracle_name=oracle_name,
        )

    # Gene annotation panel
    if has_genes:
        ax_gene = axes[-1]
        _draw_gene_panel(ax_gene, genes_df, x_start, x_end,
                         variant_pos, gene_name, exons)

    # X-axis on bottom panel
    axes[-1].tick_params(axis="x", labelsize=7, length=3)
    axes[-1].xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    axes[-1].set_xlabel(variant_chrom, fontsize=8)

    # Set shared xlim
    axes[0].set_xlim(x_start, x_end)

    fig.suptitle(title, fontsize=10, fontweight="bold", y=0.99)

    try:
        plt.tight_layout(rect=[0.08, 0, 1, 0.97])
    except Exception:
        pass

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _plot_track_panel(
    ax, ref_track, alt_track,
    x_start, x_end, variant_pos,
    assay_id, color, show_legend=False,
    show_exon_shading=False, exons=None,
    normalizer=None, oracle_name=None,
):
    """Plot a single ref-vs-alt signal panel."""
    import matplotlib.ticker as ticker

    from .scorers import classify_track_layer, LAYER_CONFIGS
    from ._igv_report import rescale_for_display

    t_res = ref_track.resolution
    t_start = ref_track.prediction_interval.reference.start
    layer = classify_track_layer(ref_track)
    layer_cfg = LAYER_CONFIGS.get(layer)
    is_signed = bool(layer_cfg and layer_cfg.signed)

    # Extract the visible window from the arrays
    bin_s = max(0, (x_start - t_start) // t_res)
    bin_e = min(len(ref_track.values), (x_end - t_start + t_res - 1) // t_res)

    if bin_e <= bin_s:
        ax.set_visible(False)
        return

    x_coords = np.arange(bin_s, bin_e) * t_res + t_start
    ref_vals = ref_track.values[bin_s:bin_e].astype(np.float64)
    alt_vals = alt_track.values[bin_s:bin_e].astype(np.float64)

    # Unified rescale (same path IGV/CoolBox/notebooks use).  When a
    # normalizer is supplied, ref/alt are CDF-rescaled to [0, 3] (unsigned)
    # or [-3, +3] (signed) and the y-axis snaps to those limits.  Without
    # a normalizer we fall through to the legacy autoscale (with the
    # signed-layer symmetric y-axis fix below).
    rescaled_cfg = None
    if normalizer is not None and oracle_name is not None:
        ref_resc, cfg_r = rescale_for_display(
            ref_vals, layer, normalizer=normalizer,
            oracle_name=oracle_name, assay_id=assay_id,
        )
        alt_resc, cfg_a = rescale_for_display(
            alt_vals, layer, normalizer=normalizer,
            oracle_name=oracle_name, assay_id=assay_id,
        )
        if cfg_r["rescaled"] and cfg_a["rescaled"]:
            ref_vals = np.asarray(ref_resc)
            alt_vals = np.asarray(alt_resc)
            rescaled_cfg = cfg_r
            is_signed = cfg_r["signed"]

    # Smooth
    n = len(ref_vals)
    if n > 50:
        w = max(3, n // 150)
        kernel = np.ones(w) / w
        ref_vals = np.convolve(ref_vals, kernel, mode="same")
        alt_vals = np.convolve(alt_vals, kernel, mode="same")

    # Exon shading (for zoom-out RNA tracks)
    if show_exon_shading and exons is not None:
        for exon in exons:
            e_s, e_e = exon["start"], exon["end"]
            if e_e > x_start and e_s < x_end:
                ax.axvspan(max(e_s, x_start), min(e_e, x_end),
                           alpha=0.08, color=color, zorder=0)

    # Ref signal
    ax.fill_between(x_coords, ref_vals, alpha=0.20, color="#9ca3af", linewidth=0)
    ax.plot(x_coords, ref_vals, color="#9ca3af", linewidth=0.6, alpha=0.7,
            label="ref" if show_legend else None)

    # Alt signal
    ax.fill_between(x_coords, alt_vals, alpha=0.30, color=color, linewidth=0)
    ax.plot(x_coords, alt_vals, color=color, linewidth=0.9,
            label="alt" if show_legend else None)

    # Variant marker
    ax.axvline(variant_pos, color="#ef4444", linestyle="--",
               linewidth=1.2, alpha=0.7, zorder=10)

    # Track label
    ax.set_ylabel(assay_id, fontsize=8, fontweight="bold", rotation=0,
                  labelpad=65, ha="right", va="center")

    # Y-axis: when CDF-rescaled, snap to the canonical [0,3] (unsigned)
    # or [-3,+3] (signed) range so the panel is comparable to the IGV/
    # CoolBox/notebook renders of the same track.  When NOT rescaled
    # (no normalizer), fall back to autoscale with the signed-layer
    # symmetric range so the negative tail of RNA/Sei/MPRA tracks stays
    # visible.
    if rescaled_cfg is not None:
        ax.set_ylim(bottom=rescaled_cfg["ymin"], top=rescaled_cfg["ymax"])
        if rescaled_cfg["signed"]:
            ax.axhline(0, color="#9ca3af", linewidth=0.4, linestyle=":",
                       alpha=0.6, zorder=1)
    elif is_signed:
        vmax = max(float(np.max(ref_vals)), float(np.max(alt_vals)), 0.0)
        vmin = min(float(np.min(ref_vals)), float(np.min(alt_vals)), 0.0)
        span = max(vmax - vmin, 0.01) * 1.05
        center = (vmax + vmin) / 2.0
        ax.set_ylim(bottom=center - span / 2.0, top=center + span / 2.0)
        ax.axhline(0, color="#9ca3af", linewidth=0.4, linestyle=":",
                   alpha=0.6, zorder=1)
    else:
        ymax = max(float(np.max(ref_vals)), float(np.max(alt_vals)), 0.01) * 1.05
        ax.set_ylim(bottom=0, top=ymax)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, integer=False))
    ax.tick_params(axis="y", labelsize=7, length=2)
    ax.tick_params(axis="x", labelsize=0, length=0)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.3)
    ax.spines["left"].set_linewidth(0.5)

    if show_legend:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7,
                  edgecolor="none", ncol=2)


# ---------------------------------------------------------------------------
# Gene annotation panel
# ---------------------------------------------------------------------------

def _draw_gene_panel(ax, genes_df, region_start, region_end,
                     variant_pos, highlight_gene=None, exons=None):
    """Draw gene models: thin intron line, thick exon blocks, TSS arrow.

    For the highlighted target gene, exon blocks are drawn as filled
    rectangles on a thin intron line, with a bent TSS arrow indicating
    the transcription start site and direction.  Other genes get a
    simpler representation.
    """
    from matplotlib.patches import FancyArrowPatch, Rectangle

    ax.set_ylim(-0.5, 2.2)
    ax.set_xlim(region_start, region_end)
    ax.axis("off")

    name_col = "gene_name" if "gene_name" in genes_df.columns else "name"
    strand_col = "strand" if "strand" in genes_df.columns else None

    # Fetch exons for ALL visible genes (not just the target)
    gene_exons_cache: dict[str, list[dict] | None] = {}
    if highlight_gene and exons is not None:
        gene_exons_cache[highlight_gene] = exons

    seen: set = set()
    y_offset = 0
    drawn = 0

    for _, row in genes_df.iterrows():
        gname = str(row.get(name_col, ""))
        if not gname or gname in seen:
            continue
        seen.add(gname)

        g_start = max(int(row.get("start", region_start)), region_start)
        g_end = min(int(row.get("end", region_end)), region_end)
        strand = row.get(strand_col, ".") if strand_col else "."

        is_target = (gname == highlight_gene)
        y = 0.4 + (y_offset % 3) * 0.6
        y_offset += 1

        color = "#4338ca" if is_target else "#6366f1"
        alpha = 0.8 if is_target else 0.5

        # Fetch exons for this gene if not cached
        if gname not in gene_exons_cache:
            gene_exons_cache[gname] = _get_exons_for_gene(gname)
        g_exons = gene_exons_cache[gname]

        # Thin intron line (full gene body)
        ax.plot([g_start, g_end], [y, y], color=color,
                linewidth=0.8 if is_target else 0.5,
                solid_capstyle="butt", alpha=alpha)

        # Exon blocks
        exon_height = 0.25 if is_target else 0.15
        if g_exons is not None:
            for exon in g_exons:
                e_s = max(exon["start"], region_start)
                e_e = min(exon["end"], region_end)
                if e_e > e_s:
                    rect = Rectangle(
                        (e_s, y - exon_height / 2), e_e - e_s, exon_height,
                        facecolor=color, edgecolor="none", alpha=alpha,
                    )
                    ax.add_patch(rect)
        else:
            # No exon data — draw gene as a single thick bar
            rect = Rectangle(
                (g_start, y - exon_height / 2), g_end - g_start, exon_height,
                facecolor=color, edgecolor="none", alpha=alpha * 0.5,
            )
            ax.add_patch(rect)

        # TSS arrow (bent arrow showing transcription direction)
        arrow_len = (region_end - region_start) * 0.008
        arrow_h = 0.3
        if strand == "+":
            tss = g_start
            # Vertical stem up, then horizontal arrow right
            ax.plot([tss, tss], [y, y + arrow_h], color=color,
                    linewidth=1.2 if is_target else 0.8, alpha=alpha)
            ax.annotate("", xy=(tss + arrow_len, y + arrow_h),
                       xytext=(tss, y + arrow_h),
                       arrowprops=dict(arrowstyle="-|>", color=color,
                                       lw=1.2 if is_target else 0.8,
                                       mutation_scale=8))
        elif strand == "-":
            tss = g_end
            ax.plot([tss, tss], [y, y + arrow_h], color=color,
                    linewidth=1.2 if is_target else 0.8, alpha=alpha)
            ax.annotate("", xy=(tss - arrow_len, y + arrow_h),
                       xytext=(tss, y + arrow_h),
                       arrowprops=dict(arrowstyle="-|>", color=color,
                                       lw=1.2 if is_target else 0.8,
                                       mutation_scale=8))

        # Gene name
        label_x = (g_start + g_end) / 2
        ax.text(label_x, y - exon_height / 2 - 0.12, gname,
                fontsize=7.5 if is_target else 6.5,
                fontstyle="italic",
                fontweight="bold" if is_target else "normal",
                color=color, ha="center", va="top")

        drawn += 1
        if drawn >= 10:
            break

    # Variant marker
    ax.axvline(variant_pos, color="#ef4444", linestyle="--",
               linewidth=1.2, alpha=0.7, zorder=10)

    ax.text(region_start, -0.35, "Genes", fontsize=7, fontweight="bold",
            color="#374151", ha="left")


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _get_genes_for_region(chrom, start, end):
    """Get gene annotations for a region. Returns DataFrame or None."""
    try:
        from chorus.utils.annotations import get_genes_in_region
        df = get_genes_in_region(chrom, start, end)
        return df if len(df) > 0 else None
    except Exception:
        return None


def _get_exons_for_gene(gene_name):
    """Get merged exon coordinates for a gene. Returns list of dicts or None."""
    try:
        from chorus.utils.annotations import get_gene_exons
        df = get_gene_exons(gene_name)
        if len(df) > 0:
            return df[["chrom", "start", "end"]].to_dict("records")
    except Exception:
        pass
    return None
