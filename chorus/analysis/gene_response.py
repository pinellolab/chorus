"""Which genes does a variant move, and in which cell types? A per-track dot plot of RNA-seq effects.

This is the "Which liver cell type responds?" panel. It reads the RNA gene tables written by discovery
(``*_rna_gene_percentiles.csv`` and its companions, see :mod:`chorus.analysis.layer_ranking`) and draws, for the
RNA-seq tracks of one tissue or body system, which the caller must choose (the RNA-seq column of
:func:`~chorus.analysis.layer_ranking.rank_variant_tissues` says which respond most):

* one **box per track** over every gene's log2 fold change (IQR, whiskers = 1.5 IQR);
* a **grey dot** for every other boxplot outlier;
* a **coloured dot, named once in the left margin**, for each *responding gene*: a gene that is a boxplot
  outlier on at least ``min_track_fraction`` of the tracks (default 0.75) and whose median |effect| clears the
  noise floor. Genes are joined by a line across the tracks of one cell type.

The rule is deliberately about being an outlier, not about clearing the floor on every track: on the SORT1 liver
data PSRC1 clears the floor on 12 of 12 tracks, CELSR2 and MYBPHL on 11 and SORT1 on 8, yet all four are outliers on
11-12 of 12 and no other gene is an outlier anywhere. ``notes`` says where a named gene dips inside the floor.

Why raw log2 fold change and not the percentile: the RNA effect null is narrow (AlphaGenome's tops out at
0.0417 ln = 0.060 log2, ``chorus/utils/annotations.py``), so almost every real effect saturates at percentile 1.0
and genes cannot be told apart by it. The raw effect against that floor can. The floor is a measured property of
the null, not a p-value, and it is specific to the oracle: it is applied only where it is known.

A gene is read only on unstranded tracks and on the stranded track of its own strand
(``strand_mode="matched"``); the two strands of a track are one track.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .layer_ranking import GROUP_LEVELS, RNA_LAYER, load_annotation, load_rna_tables

logger = logging.getLogger(__name__)

#: Top of the measured effect null of each oracle's RNA tracks, in natural log (the unit of ``raw_score``).
#: Only AlphaGenome has been measured (chorus/utils/annotations.py); other oracles get no floor unless one is given.
NOISE_FLOOR_LN = {"AlphaGenome": 0.0417, "AlphaGenomePT": 0.0417}

#: Colours for the named genes, in order of decreasing median |effect| (Okabe-Ito, then two extra hues).
GENE_COLOURS = ["#0072B2", "#009E73", "#CC79A7", "#D55E00", "#E69F00", "#56B4E9", "#7B3294", "#8C6D31"]
_INK, _GREY, _LIGHT, _NEUTRAL = "#1F2933", "#5F6B7A", "#E3E8EE", "#B8C0CC"
_ASSAY_COL = {"polyA+": "#DCE6F2", "total": "#F3DDB8", "other": "#E6E6E6"}
MAX_TRACKS_DEFAULT = 40


# ---------------------------------------------------------------------------
# Selecting tracks and finding the responding genes
# ---------------------------------------------------------------------------

def _available_groups(tracks: pd.DataFrame, group_map: pd.Series) -> pd.Series:
    g = tracks.cell_type_key.map(group_map)
    return g.value_counts()


def select_tracks(long: pd.DataFrame, tissue: str, group_level: str, annotation, max_tracks: int):
    """Restrict the long RNA table to the RNA-seq tracks of one tissue / system (required; there is no default)."""
    level = GROUP_LEVELS[group_level]
    ann = annotation if isinstance(annotation, pd.DataFrame) else load_annotation(annotation)
    gm = ann.drop_duplicates("cell_type_key").set_index("cell_type_key")[level]
    keys = long.drop_duplicates("track_id")[["track_id", "cell_type_key"]]
    counts = _available_groups(keys, gm)
    if not tissue or not str(tissue).strip():
        raise ValueError(f"tissue is required: choose a {level} (tracks): "
                         + ", ".join(f"{g} ({n})" for g, n in counts.head(30).items()))
    by_lower = {str(g).lower(): g for g in counts.index}
    if tissue.lower() not in by_lower:
        raise ValueError(f"{level} {tissue!r} has no RNA-seq tracks in this table. Available: "
                         + ", ".join(f"{g} ({n})" for g, n in counts.head(30).items()))
    tissue = by_lower[tissue.lower()]
    long = long[long.cell_type_key.isin(set(gm[gm == tissue].index))]
    n = long.track_id.nunique()
    if n > max_tracks:
        raise ValueError(f"{tissue!r} has {n} RNA-seq tracks, more than fit on one panel (limit {max_tracks}); "
                         f"choose a narrower group (group_level='tissue' rather than 'system') or raise max_tracks")
    return long, tissue


def find_responders(long: pd.DataFrame, floor_log2: float, min_track_fraction: float, max_named: int):
    """Flag outliers per track and pick the genes that respond.

    ``long`` needs ``gene``, ``track_id`` and ``log2fc``. A track's outliers are the genes beyond 1.5 IQR of that
    track's own distribution over genes. A gene is a responder when it is an outlier on at least
    ``min_track_fraction`` of the tracks it is read on and its median |log2fc| is at least ``floor_log2``.
    At most ``max_named`` responders are kept, those with the largest median |log2fc|.

    ``fraction_above_floor`` (the stricter "outlier and above the floor", per track) is reported but does not
    decide, so a gene that is an outlier everywhere but dips inside the floor on a few tracks is still named.

    Returns ``(long_with_flags, per_gene_table, named_genes)``.
    """
    long = long.copy()
    q = long.groupby("track_id").log2fc.quantile([0.25, 0.75]).unstack()
    iqr = q[0.75] - q[0.25]
    lo = long.track_id.map(q[0.25] - 1.5 * iqr)
    hi = long.track_id.map(q[0.75] + 1.5 * iqr)
    long["is_outlier"] = (long.log2fc > hi) | (long.log2fc < lo)
    long["above_floor"] = long.log2fc.abs() >= floor_log2
    long["responds"] = long.is_outlier & long.above_floor
    long["abs_log2fc"] = long.log2fc.abs()
    per = long.groupby("gene").agg(n_tracks=("track_id", "nunique"), n_outlier=("is_outlier", "sum"),
                                   n_above_floor=("above_floor", "sum"), n_responding=("responds", "sum"),
                                   median_log2fc=("log2fc", "median"), median_abs_log2fc=("abs_log2fc", "median"),
                                   max_abs_log2fc=("abs_log2fc", "max"))
    per["fraction_outlier"] = per.n_outlier / per.n_tracks
    per["fraction_above_floor"] = per.n_above_floor / per.n_tracks
    per["responder"] = (per.fraction_outlier >= min_track_fraction - 1e-12) & (per.median_abs_log2fc >= floor_log2)
    named = per[per.responder].sort_values("median_abs_log2fc", ascending=False).index[:max_named].tolist()
    per["named"] = per.index.isin(named)
    per["colour"] = pd.Series({g: GENE_COLOURS[i] for i, g in enumerate(named)})
    per = per.sort_values(["named", "fraction_outlier", "median_abs_log2fc"], ascending=[False, False, False])
    return long, per, named


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _kind(track_id: str) -> str:
    return "polyA+" if "polyA" in track_id else ("total" if "total" in track_id else "other")


def _tick(track_id: str) -> str:
    k = {"polyA+": "polyA+", "total": "total", "other": "track"}[_kind(track_id)]
    return k + ("\nunstr." if track_id.endswith("/.") else "")


def _cell_label(name: str) -> str:
    name = name[:1].upper() + name[1:]
    if len(name) <= 12 or " " not in name:
        return name
    if " of " in name:
        return name.replace(" of ", "\nof ", 1)
    cuts = [i for i, ch in enumerate(name) if ch == " "]
    i = min(cuts, key=lambda c: abs(c - len(name) / 2))
    return name[:i] + "\n" + name[i + 1:]


def _spread(ys: Sequence[float], gap: float) -> list[float]:
    """Label heights for points ``ys`` (high to low): each at its own height unless within ``gap`` of the one above."""
    out, last = [], np.inf
    for y in ys:
        yl = min(y, last - gap)
        out.append(yl)
        last = yl
    return out


def draw_gene_response(long: pd.DataFrame, per_gene: pd.DataFrame, named: list, *, floor_log2: float, title: str,
                       subtitle: str, ylabel: str, out_stem: str | Path, height_in: float = 1.9,
                       width_in: Optional[float] = None, panel_letter: Optional[str] = None,
                       cell_type_order: Optional[Sequence[str]] = None,
                       formats: Sequence[str] = ("png", "pdf")):
    """The compact "B" layout of the Liver RNA-seq panel. Returns the written paths."""
    import matplotlib as mpl
    mpl.use("Agg", force=False)
    import matplotlib.pyplot as plt

    colour = per_gene.colour.to_dict()
    tracks = long.drop_duplicates("track_id").set_index("track_id")[["cell_type", "cell_type_key"]]
    val = {t: s.set_index("gene").log2fc for t, s in long.groupby("track_id")}

    # cell types in order of how strongly the named genes respond there (all genes' extremes if none is named)
    def strength(ct):
        ts = tracks.index[tracks.cell_type == ct]
        pool = named if named else list(per_gene.index)
        return float(np.mean([np.mean([abs(val[t].get(g, 0.0)) for g in pool]) for t in ts]))
    cts = sorted(tracks.cell_type.unique(), key=lambda c: -strength(c))
    if cell_type_order:                                       # explicit order first (case-insensitive), the rest as above
        rank = {str(c).lower(): i for i, c in enumerate(cell_type_order)}
        cts = sorted(cts, key=lambda c: (rank.get(c.lower(), len(rank)), cts.index(c)))
    order = sorted(tracks.index, key=lambda t: (cts.index(tracks.cell_type[t]), "total" in t, t))

    pos, x = {}, 0.0
    for i, t in enumerate(order):
        if i and tracks.cell_type[t] != tracks.cell_type[order[i - 1]]:
            x += 0.55
        pos[t] = x
        x += 1.0
    xlim = (-2.1, x - 0.2)
    span_panel = 16.65                                           # the published 12-track panel, 7.4 in wide
    width = width_in or max(4.6, 7.4 * (xlim[1] - xlim[0]) / span_panel)
    h = height_in
    vmin, vmax = float(long.log2fc.min()), float(long.log2fc.max())
    ylim = (min(-0.1, vmin * 1.08, -1.6 * floor_log2), max(vmax * 1.08, 0.2))
    gap = 0.0615 * (ylim[1] - ylim[0])

    rc = {"font.family": "DejaVu Sans", "font.size": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
          "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
          "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 300,
          "savefig.dpi": 300, "axes.facecolor": "white", "figure.facecolor": "white"}
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(width, h), layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0, 1, 1 - 0.40 / h))
        ax.set_ylim(*ylim)
        ax.set_xlim(*xlim)
        ax.axhline(0, color=_GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
        x0 = (pos[order[0]] - 0.6 - xlim[0]) / (xlim[1] - xlim[0])        # the band covers the tracks, not the label margin
        if floor_log2 > 0:
            ax.axhspan(-floor_log2, floor_log2, xmin=x0, color=_LIGHT, alpha=0.55, lw=0, zorder=0)
        for t in order:
            kind = _kind(t)
            ax.boxplot(val[t], positions=[pos[t]], widths=0.55, vert=True, showfliers=False, patch_artist=True,
                       zorder=3, boxprops=dict(facecolor=_ASSAY_COL[kind], edgecolor=_GREY, lw=0.6),
                       medianprops=dict(color=_INK, lw=0.8), whiskerprops=dict(color=_GREY, lw=0.5),
                       capprops=dict(color=_GREY, lw=0.5))
            flagged = long[(long.track_id == t) & long.is_outlier & ~long.gene.isin(named)]
            ax.scatter(np.full(len(flagged), pos[t]), flagged.log2fc.values, s=3.5, color=_NEUTRAL, lw=0, zorder=4)
        for ct in cts:                                                       # named genes joined within a cell type
            ts = [t for t in order if tracks.cell_type[t] == ct]
            for g in named:
                if len(ts) > 1:
                    ax.plot([pos[t] for t in ts], [val[t][g] for t in ts], color=colour[g], lw=0.5, alpha=0.55, zorder=4)
        for t in order:
            for g in named:
                ax.scatter(pos[t], val[t][g], s=12, color=colour[g], edgecolor="white", lw=0.4, zorder=6)
        if named:                                                            # names once, at the leftmost track's heights
            first = order[0]
            ranked = sorted(named, key=lambda g: -val[first][g])
            for g, yl in zip(ranked, _spread([val[first][g] for g in ranked], gap)):
                ax.annotate(g, (pos[first], val[first][g]), xytext=(pos[first] - 0.65, yl), textcoords="data",
                            ha="right", va="center", fontsize=5.6, color=colour[g], fontweight="bold",
                            arrowprops=dict(arrowstyle="-", color=colour[g], lw=0.4, shrinkA=0.5, shrinkB=2))
        ax.set_xticks([pos[t] for t in order])
        ax.set_xticklabels([_tick(t) for t in order], fontsize=5.0, linespacing=1.0)
        ax.tick_params(axis="x", length=0, pad=1.5)
        ax.tick_params(axis="y", labelsize=5.4)
        for ct in cts:
            xs = [pos[t] for t in order if tracks.cell_type[t] == ct]
            ax.annotate(_cell_label(ct), (np.mean(xs), 0), xycoords=("data", "axes fraction"), xytext=(0, -14),
                        textcoords="offset points", ha="center", va="top", fontsize=5.4, fontweight="bold", color=_INK,
                        linespacing=1.05, annotation_clip=False)
        ax.set_ylabel(ylabel, fontsize=6.0)
        tx = 0.012 * 7.4 / width
        if panel_letter:
            fig.text(tx, 1 - 0.07 / h, panel_letter, fontsize=11, fontweight="bold", color=_INK, va="top", ha="left")
            tx += 0.055 * 7.4 / width
        fig.text(tx, 1 - 0.07 / h, title, fontsize=7.8, fontweight="bold", color=_INK, va="top", ha="left")
        fig.text(tx, 1 - 0.25 / h, subtitle, fontsize=5.8, color=_GREY, va="top", ha="left")
        paths = []
        for fmt in formats:
            p = f"{out_stem}.{fmt}"
            fig.savefig(p, facecolor="white")
            paths.append(p)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def plot_variant_gene_response(
    rna_table: str | Path,
    output_dir: str | Path,
    *,
    tissue: str,
    group_level: str = "tissue",
    annotation: str | Path | pd.DataFrame | None = None,
    noise_floor_log2: Optional[float] = None,
    min_track_fraction: float = 0.75,
    max_named: int = 8,
    max_tracks: int = MAX_TRACKS_DEFAULT,
    variant_label: Optional[str] = None,
    title: Optional[str] = None,
    height_in: float = 1.9,
    width_in: Optional[float] = None,
    panel_letter: Optional[str] = None,
    cell_type_order: Optional[Sequence[str]] = None,
    formats: Sequence[str] = ("png", "pdf"),
) -> dict:
    """Dot plot of a variant's RNA-seq effect on every gene, per track, naming the genes that respond.

    Args:
        rna_table: The ``rna_gene_percentiles_csv`` written by ``discover_variant`` (its two companion
            files are found next to it).
        output_dir: Where the figure and CSVs are written.
        tissue: Required. Tissue (``group_level="tissue"``, e.g. ``"Liver"``) or body system
            (``group_level="system"``) whose RNA-seq tracks are shown. Pick the one the ranking says responds;
            an unknown name raises an error listing the choices with their track counts.
        group_level: ``"tissue"`` or ``"system"``; how ``tissue`` is matched against the annotation.
        annotation: Cell-type -> tissue/system mapping; default is the copy shipped with chorus.
        noise_floor_log2: |log2 fold change| below which an effect is indistinguishable from the RNA null.
            Default: 0.0417 / ln 2 = 0.060 for AlphaGenome (measured); for an oracle without a measured floor
            the default is no floor, with a note.
        min_track_fraction: A gene is named if it is a boxplot outlier on at least this fraction of the tracks
            and its median |log2FC| is above the floor (default 0.75; 1.0 = outlier on every track).
        max_named: Name at most this many genes, the ones with the largest median |effect|.
        variant_label: Subtitle prefix; default read from the table header.
        title: Default ``"Which <tissue> cell type responds?"``.
        height_in / width_in: Figure size; the width grows with the number of tracks by default.
        panel_letter: Optional bold letter before the title.
        cell_type_order: Cell types, left to right. Default: by how strongly the named genes respond there.

    Returns:
        A dict with ``files``, ``tracks``, ``named_genes`` (with colour and how many tracks they respond on),
        ``noise_floor_log2`` and ``notes``.
    """
    if group_level not in GROUP_LEVELS:
        raise ValueError(f"group_level must be 'tissue' or 'system', got {group_level!r}")
    if not 0 < min_track_fraction <= 1:
        raise ValueError("min_track_fraction must be in (0, 1]")
    long, summary, label = load_rna_tables(rna_table, strand_mode="matched")
    notes: list[str] = []
    if long.empty:
        raise ValueError(f"{rna_table} holds no RNA-seq scores")
    long, tissue = select_tracks(long, tissue, group_level, annotation, max_tracks)
    if long.empty:
        raise ValueError("no RNA-seq track is left after selecting the tissue")
    long["log2fc"] = long.raw_score / np.log(2)
    long = long[long.log2fc.notna()]

    models = sorted(long.model.dropna().unique())
    model = models[0] if len(models) == 1 else "RNA-seq"
    if noise_floor_log2 is not None:
        floor = float(noise_floor_log2)
    elif len(models) == 1 and models[0] in NOISE_FLOOR_LN:
        floor = NOISE_FLOOR_LN[models[0]] / np.log(2)
    else:
        floor = 0.0
        notes.append(f"no measured RNA noise floor for {models}: only the boxplot-outlier rule was applied; pass "
                     f"noise_floor_log2 to add one")

    flagged, per_gene, named = find_responders(long, floor, min_track_fraction, max_named)
    n_resp = int(per_gene.responder.sum())
    if n_resp > max_named:
        notes.append(f"{n_resp} genes qualify; the {max_named} with the largest median |effect| are named")
    for g in named:
        dip = int(per_gene.n_tracks[g] - per_gene.n_above_floor[g])
        if dip:
            notes.append(f"{g} is inside the noise floor on {dip} of {int(per_gene.n_tracks[g])} tracks "
                         f"(an outlier on {int(per_gene.n_outlier[g])})")
    if not named:
        best = per_gene.fraction_outlier.max()
        notes.append(f"no gene is a boxplot outlier on {min_track_fraction:.0%} of the tracks with a median effect "
                     f"above the noise floor (best outlier fraction: {best:.0%}); lower min_track_fraction to name some")

    n_tracks = flagged.track_id.nunique()
    n_genes = flagged.gene.nunique()
    unit = f"{tissue} "
    when = "on every track" if min_track_fraction >= 1 else f"on at least {min_track_fraction:.0%} of tracks"
    sub = (f"{variant_label or label or 'variant'} · {model} RNA-seq · {n_tracks} {unit}tracks · "
           + (f"coloured dots: the {len(named)} gene{'s' if len(named) != 1 else ''} that are outliers {when}"
              if named else f"no gene is an outlier {when} above the noise floor"))
    alleles = re.search(r"([ACGT]+)>([ACGT]+)", variant_label or label or "")
    ylab = f"log2 fold-change ({alleles.group(2)} / {alleles.group(1)})" if alleles else "log2 fold-change (alt / ref)"
    title = title or f"Which {tissue.lower()} cell type responds?"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"figure": draw_gene_response(flagged, per_gene, named, floor_log2=floor, title=title, subtitle=sub,
                                          ylabel=ylab, out_stem=out / "gene_response", height_in=height_in,
                                          width_in=width_in, panel_letter=panel_letter,
                                          cell_type_order=cell_type_order, formats=formats)}
    cols = ["gene", "cell_type", "track_id", "track_strand", "log2fc", "raw_score", "signed_percentile",
            "is_outlier", "above_floor", "responds"]
    table = flagged[cols].assign(named=flagged.gene.isin(named)).sort_values(["gene", "cell_type", "track_id"])
    table.to_csv(out / "gene_response_by_track.csv", index=False, float_format="%.6g")
    per_gene.reset_index().to_csv(out / "gene_response_by_gene.csv", index=False, float_format="%.6g")
    files["by_track_csv"], files["by_gene_csv"] = str(out / "gene_response_by_track.csv"), str(out / "gene_response_by_gene.csv")
    summary_json = {
        "tissue": tissue, "group_level": GROUP_LEVELS[group_level], "noise_floor_log2": round(floor, 5),
        "min_track_fraction": min_track_fraction, "n_tracks": int(n_tracks), "n_genes": int(n_genes),
        "model": model, "named_genes": named, "notes": notes,
    }
    (out / "run_summary.json").write_text(json.dumps(summary_json, indent=2))
    files["run_summary"] = str(out / "run_summary.json")
    return {
        **summary_json, "files": files,
        "genes": [{"gene": g, "colour": per_gene.colour[g], "tracks_as_outlier": int(per_gene.n_outlier[g]),
                   "tracks_above_floor": int(per_gene.n_above_floor[g]), "of_tracks": int(per_gene.n_tracks[g]),
                   "median_log2fc": round(float(per_gene.median_log2fc[g]), 4)} for g in named],
        "tracks": sorted(flagged.drop_duplicates("track_id").track_id.tolist()),
    }
