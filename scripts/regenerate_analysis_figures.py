#!/usr/bin/env python
"""Regenerate the article's analysis panels from Chorus outputs (corrected numbers).

Each panel reads a JSON produced by a Chorus analysis on GPU (ml007) and renders a
clean matplotlib figure in the chorus house palette. Panels:
  1. cell-type ranking (SORT1)        — ChromBPNet/AlphaGenome/EPInformer
  2. TF scan (SORT1)                  — AlphaGenome TF-ChIP   [this file: render_tf_scan]
  3. gene-expression (SORT1)          — AlphaGenome + Borzoi
  4. fine-map Top-N (rs9504151)       — AlphaGenome + ChromBPNet
  5. ATF4 ISM (rs9504151)             — saturation_mutagenesis, multi-oracle (logomaker)

Run per-panel from the gathered JSONs (no GPU needed for plotting).
"""
from __future__ import annotations
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Chorus house palette (matches the figures in chorus-article)
INK = "#16202B"; MUTED = "#7f7f7f"; LINE = "#E3E7EE"
FAMILY_COLORS = {
    "C/EBP": "#CF5430", "FOX/HNF": "#2B5FA0", "nuclear receptor": "#7A47A0",
    "ATF/CREB": "#C98A22", "other": "#9099A8",
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c4c4c4"})


def _tf_family(name: str) -> str:
    n = name.upper()
    if n.startswith("CEBP") or "C/EBP" in n:
        return "C/EBP"
    if n.startswith(("FOX", "HNF", "HLF")):
        return "FOX/HNF"
    if n.startswith(("RAR", "RXR", "NR", "PPAR", "ESR", "VDR", "THR")):
        return "nuclear receptor"
    if n.startswith(("ATF", "CREB", "JUN", "FOS", "XBP")):
        return "ATF/CREB"
    return "other"


def _tf_name(description: str) -> str:
    # descriptions look like "CHIP:CEBPA:HepG2"
    parts = description.split(":")
    return parts[1] if len(parts) >= 2 else description


def render_tf_scan(discovery_json: str, out_png: str, *, variant_label: str,
                   gene: str, cell: str, top_n: int = 16, title_extra: str = ""):
    """Horizontal family-colored bar of top TF-binding effects from a discovery run."""
    d = json.load(open(discovery_json))
    tf = d["layer_rankings"]["tf_binding"]
    # Restrict to the article's cell line (the scan is cell-type-specific); otherwise
    # the max-over-all-cells dedup would surface non-HepG2 tracks (e.g. CEBPB:IMR-90).
    tf = [e for e in tf if e.get("cell_type") == cell]
    # dedupe by TF name, keep the strongest |effect|
    best: dict[str, dict] = {}
    for e in tf:
        nm = _tf_name(e.get("description", e.get("assay_id", "")))
        if nm not in best or abs(e["effect"]) > abs(best[nm]["effect"]):
            best[nm] = e
    rows = sorted(best.values(), key=lambda e: e["effect"], reverse=True)[:top_n]
    names = [_tf_name(e.get("description", "")) for e in rows]
    effects = [e["effect"] for e in rows]
    fams = [_tf_family(n) for n in names]
    colors = [FAMILY_COLORS[f] for f in fams]

    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=200)
    y = range(len(names))
    ax.barh(list(y), effects, color=colors, edgecolor="white", height=0.74)
    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("TF-binding effect — log2 fold-change (alt vs ref)", fontsize=10.5)
    ax.set_title(f"Transcription-factor scan — {variant_label} ({gene})\n"
                 f"AlphaGenome ChIP-TF, {cell}{title_extra}", fontsize=12, fontweight="bold", color=INK)
    ax.axvline(0, color="#c4c4c4", lw=1)
    for i, (eff, nm) in enumerate(zip(effects, names)):
        ax.text(eff + (0.05 if eff >= 0 else -0.05), i, f"{eff:+.2f}",
                va="center", ha="left" if eff >= 0 else "right", fontsize=9, color=INK)
    # legend
    seen = []
    for f in ["C/EBP", "ATF/CREB", "FOX/HNF", "nuclear receptor", "other"]:
        if f in fams and f not in seen:
            seen.append(f)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in seen]
    ax.legend(handles, seen, fontsize=9, frameon=False, loc="lower right", title="TF family")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.margins(x=0.18)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_png}  (top TF: {names[0]} {effects[0]:+.2f})")


def render_ism_panel(ism_specs, out_png, *, variant_label: str, gene: str,
                     motif: str = "ATF4", variant_offset: int | None = None):
    """Stacked per-oracle ISM motif logos (importance of the reference base).

    ism_specs: list of (oracle_label, json_path) produced by saturation_mutagenesis.
    The reference base at each position is drawn at a height equal to its disruption
    importance, so functional positions (the motif) spell out as tall letters.
    """
    import pandas as pd
    import numpy as np
    import logomaker

    n = len(ism_specs)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.7 * n + 0.6), dpi=200, sharex=True)
    if n == 1:
        axes = [axes]
    width = None
    for ax, (label, path) in zip(axes, ism_specs):
        d = json.load(open(path))
        seq = d["ref_seq"]; imp = d["importance"]; width = len(seq)
        mat = pd.DataFrame(0.0, index=range(width), columns=["A", "C", "G", "T"])
        for i, (b, v) in enumerate(zip(seq, imp)):
            if b in "ACGT":
                mat.loc[i, b] = max(float(v), 0.0)
        logomaker.Logo(mat, ax=ax, color_scheme="classic", show_spines=False)
        ax.set_ylabel(label, fontsize=10.5, fontweight="bold", rotation=0,
                      ha="right", va="center", labelpad=18, color=INK)
        ax.set_yticks([]); ax.set_xticks([])
        center = width // 2 if variant_offset is None else variant_offset
        ax.axvline(center, color="#CF5430", lw=1.4, ls="--", alpha=0.8, zorder=0)
    # x labels on the bottom axis: the reference sequence
    d0 = json.load(open(ism_specs[0][1]))
    axes[-1].set_xticks(range(width))
    axes[-1].set_xticklabels(list(d0["ref_seq"]), fontsize=8, family="monospace")
    axes[0].set_title(
        f"In-silico saturation mutagenesis — {variant_label} ({gene})\n"
        f"every oracle's reference-base importance converges on the {motif} motif",
        fontsize=12, fontweight="bold", color=INK, pad=10)
    fig.text(0.5, 0.005, "position (reference sequence; dashed line = variant)",
             ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_png}  ({n} oracles, window {width})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "ism":
        # args: ism <out.png> label1=path1 label2=path2 ...
        specs = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in sys.argv[3:]]
        render_ism_panel(specs, sys.argv[2], variant_label="rs9504151", gene="CDYL")
    elif cmd == "tf_scan":
        render_tf_scan(sys.argv[2], sys.argv[3], variant_label="rs12740374",
                       gene="SORT1", cell="HepG2")
    else:
        print("usage: regenerate_analysis_figures.py tf_scan <discovery.json> <out.png>")
