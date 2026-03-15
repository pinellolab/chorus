#!/usr/bin/env python
"""Generate variant effect reports with plots for 3 canonical GWAS variants.

Uses the Chorus Python API directly (not MCP) to produce:
- Per-variant multi-panel figures (DNASE + CAGE tracks, ref vs alt, effect)
- Gene annotation overlay
- A combined markdown report
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Variant definitions ──────────────────────────────────────────────

VARIANTS = [
    {
        "rsid": "rs1427407",
        "chrom": "chr2", "pos": 60490908,
        "ref": "T", "alt": "G",
        "gene": "BCL11A",
        "trait": "Fetal hemoglobin (HbF) levels / Sickle cell disease",
        "description": (
            "Located in a BCL11A intronic erythroid enhancer, this variant disrupts "
            "GATA1 binding and reduces BCL11A expression, de-repressing fetal hemoglobin. "
            "This mechanism is the basis for sickle cell gene therapies (Casgevy)."
        ),
        "assay_ids": ["ENCFF413AHU", "CNhs11250"],
        "assay_labels": ["DNASE K562", "CAGE K562"],
        "genes_in_region": [
            {"name": "BCL11A", "start": 60450520, "end": 60554467, "strand": "-"},
            {"name": "MIR4432HG", "start": 60311286, "end": 60439841, "strand": "-"},
        ],
    },
    {
        "rsid": "rs12740374",
        "chrom": "chr1", "pos": 109274968,
        "ref": "G", "alt": "T",
        "gene": "SORT1",
        "trait": "LDL cholesterol / Coronary artery disease",
        "description": (
            "Creates a C/EBP binding site in the CELSR2-PSRC1-SORT1 locus, increasing "
            "hepatic SORT1 expression. SORT1 promotes VLDL secretion from hepatocytes, "
            "directly linking this variant to plasma LDL levels."
        ),
        "assay_ids": ["ENCFF413AHU", "CNhs11250", "CNhs10624"],
        "assay_labels": ["DNASE K562", "CAGE K562", "CAGE Liver"],
        "genes_in_region": [
            {"name": "CELSR2", "start": 109249539, "end": 109275751, "strand": "+"},
            {"name": "PSRC1", "start": 109279556, "end": 109283186, "strand": "-"},
            {"name": "SORT1", "start": 109309568, "end": 109397918, "strand": "-"},
            {"name": "SARS1", "start": 109213918, "end": 109238182, "strand": "+"},
        ],
    },
    {
        "rsid": "rs1421085",
        "chrom": "chr16", "pos": 53767042,
        "ref": "T", "alt": "C",
        "gene": "FTO / IRX3",
        "trait": "Body mass index (BMI) / Obesity",
        "description": (
            "Despite residing in FTO intron 1, this variant disrupts an ARID5B repressor "
            "binding site, leading to increased IRX3/IRX5 expression in adipocyte "
            "precursors. This shifts the thermogenic program from energy-dissipating "
            "beige adipocytes to energy-storing white adipocytes."
        ),
        "assay_ids": ["ENCFF413AHU", "CNhs11250"],
        "assay_labels": ["DNASE K562", "CAGE K562"],
        "genes_in_region": [
            {"name": "FTO", "start": 53701692, "end": 54158512, "strand": "+"},
            {"name": "RPGRIP1L", "start": 53598153, "end": 53703938, "strand": "-"},
        ],
    },
]


def load_and_predict(variant):
    """Run Enformer variant effect prediction via chorus API."""
    from chorus import create_oracle
    from chorus.utils.genome import GenomeManager

    gm = GenomeManager()
    fasta = str(gm.get_genome_path("hg38"))

    oracle = create_oracle("enformer", use_environment=True, reference_fasta=fasta)
    oracle.load_pretrained_model()

    v = variant
    region = f"{v['chrom']}:{v['pos']}-{v['pos']+1}"
    alleles = [v["ref"], v["alt"]]

    result = oracle.predict_variant_effect(
        genomic_region=region,
        variant_position=f"{v['chrom']}:{v['pos']}",
        alleles=alleles,
        assay_ids=v["assay_ids"],
    )
    return oracle, result


def plot_variant(variant, result, oracle):
    """Create multi-panel figure for a variant."""
    ref_pred = result["predictions"]["reference"]
    alt_pred = result["predictions"]["alt_1"]
    n_assays = len(variant["assay_ids"])

    fig, axes = plt.subplots(n_assays * 2, 1, figsize=(14, 3.5 * n_assays * 2),
                              gridspec_kw={"hspace": 0.35})
    if n_assays * 2 == 2:
        axes = [axes[0], axes[1]]

    panel_idx = 0
    effect_summaries = {}

    for i, (assay_id, label) in enumerate(zip(variant["assay_ids"], variant["assay_labels"])):
        ref_track = ref_pred[assay_id]
        alt_track = alt_pred[assay_id]
        positions = ref_track.positions / 1e6  # Mb
        var_pos_mb = variant["pos"] / 1e6

        # ── Panel 1: Ref vs Alt overlay ──
        ax = axes[panel_idx]
        ax.fill_between(positions, ref_track.values, alpha=0.35, color="#1f77b4", label="Reference")
        ax.fill_between(positions, alt_track.values, alpha=0.35, color="#d62728", label="Alternate")
        ax.axvline(var_pos_mb, color="black", ls="--", lw=1.2, alpha=0.7, label=f"{variant['rsid']}")
        ax.set_ylabel(f"{label}\nSignal")
        ax.set_title(f"{label} — Ref ({variant['ref']}) vs Alt ({variant['alt']})", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(positions[0], positions[-1])

        # Gene annotations
        ymin, ymax = ax.get_ylim()
        gene_y = ymax * 0.92
        for g in variant["genes_in_region"]:
            gs, ge = g["start"] / 1e6, g["end"] / 1e6
            if gs < positions[-1] and ge > positions[0]:
                gs_c = max(gs, positions[0])
                ge_c = min(ge, positions[-1])
                ax.axhspan(ymax * 0.88, ymax * 0.96, xmin=(gs_c - positions[0])/(positions[-1]-positions[0]),
                           xmax=(ge_c - positions[0])/(positions[-1]-positions[0]),
                           alpha=0.15, color="green")
                mid = (gs_c + ge_c) / 2
                ax.text(mid, gene_y, g["name"], ha="center", va="center", fontsize=8,
                        fontstyle="italic", color="darkgreen", fontweight="bold")
        panel_idx += 1

        # ── Panel 2: Effect (alt - ref) ──
        ax2 = axes[panel_idx]
        effect = alt_track.values - ref_track.values
        colors = np.where(effect >= 0, "#d62728", "#1f77b4")
        ax2.bar(positions, effect, width=(positions[1]-positions[0])*0.9,
                color=colors, alpha=0.7, linewidth=0)
        ax2.axhline(0, color="gray", lw=0.8)
        ax2.axvline(var_pos_mb, color="black", ls="--", lw=1.2, alpha=0.7)
        ax2.set_ylabel(f"Effect\n(Alt - Ref)")
        ax2.set_xlabel("Genomic position (Mb)" if panel_idx == len(axes)-1 else "")
        ax2.set_title(f"{label} — Variant effect", fontsize=11)
        ax2.set_xlim(positions[0], positions[-1])

        # Compute summary
        abs_max_idx = np.argmax(np.abs(effect))
        effect_summaries[label] = {
            "mean_effect": float(np.mean(effect)),
            "abs_max_effect": float(np.max(np.abs(effect))),
            "abs_max_position": f"{variant['chrom']}:{int(ref_track.positions[abs_max_idx])}",
            "std_effect": float(np.std(effect)),
        }
        panel_idx += 1

    fig.suptitle(f"{variant['rsid']} ({variant['ref']}>{variant['alt']}) — {variant['gene']} locus\n{variant['trait']}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    fig_path = OUTPUT_DIR / f"{variant['rsid']}_enformer_report.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    return effect_summaries


def generate_markdown(all_results):
    """Write the combined markdown report."""
    lines = [
        "# Chorus MCP: GWAS Variant Effect Analysis Report",
        "",
        "**Oracle:** Enformer (DeepMind) | **Genome:** hg38 | **Generated by:** Chorus MCP Server",
        "",
        "---",
        "",
        "## Overview",
        "",
        "This report demonstrates end-to-end variant effect prediction using the Chorus MCP server ",
        "with Enformer on 3 canonical GWAS variants that have well-characterized molecular mechanisms.",
        "",
    ]

    for v, effects in all_results:
        lines += [
            f"---",
            f"",
            f"## {v['rsid']} — {v['gene']} ({v['trait']})",
            f"",
            f"**Position:** `{v['chrom']}:{v['pos']}` | **Alleles:** {v['ref']} > {v['alt']}",
            f"",
            f"**Biological mechanism:** {v['description']}",
            f"",
            f"### Enformer Predictions",
            f"",
            f"![{v['rsid']} tracks]({v['rsid']}_enformer_report.png)",
            f"",
            f"| Track | Mean Effect | Abs Max Effect | Peak Position | Std |",
            f"|-------|-----------|---------------|---------------|-----|",
        ]
        for label, e in effects.items():
            lines.append(
                f"| {label} | {e['mean_effect']:.6f} | {e['abs_max_effect']:.6f} | "
                f"`{e['abs_max_position']}` | {e['std_effect']:.6f} |"
            )
        lines += ["", ""]

    lines += [
        "---",
        "",
        "## How to Run Your Own Variant Analysis",
        "",
        "The Chorus MCP server makes it easy to analyze any variant of interest. Here is a ",
        "step-by-step workflow:",
        "",
        "### 1. Discover available tracks",
        "```",
        'list_tracks("enformer", query="DNASE")',
        'list_tracks("enformer", query="CAGE")',
        "```",
        "This returns track identifiers (e.g., `ENCFF413AHU` for DNASE:K562) that you'll use in predictions.",
        "",
        "### 2. Load an oracle",
        "```",
        'load_oracle("enformer")  # ~10s, cached for reuse',
        "```",
        "Other oracles: `borzoi` (32bp resolution), `alphagenome` (1bp resolution, 1Mb window), ",
        "`chrombpnet` (base-resolution TF binding).",
        "",
        "### 3. Predict variant effect",
        "```",
        'predict_variant_effect(',
        '    oracle_name="enformer",',
        '    position="chr2:60490908",     # variant position',
        '    ref_allele="T",',
        '    alt_alleles=["G"],',
        '    assay_ids=["ENCFF413AHU", "CNhs11250"],',
        '    # region is auto-centered on the variant',
        ')',
        "```",
        "Returns per-track summary stats and saves bedgraph files for genome browser viewing.",
        "",
        "### 4. Score at the variant site",
        "```",
        'score_variant_effect_at_region(',
        '    oracle_name="enformer",',
        '    position="chr2:60490908",',
        '    ref_allele="T", alt_alleles=["G"],',
        '    assay_ids=["ENCFF413AHU"],',
        '    at_variant=True,',
        '    window_bins=5,  # +/- 5 bins (640bp) around variant',
        '    scoring_strategy="mean",',
        ')',
        "```",
        "",
        "### 5. Assess impact on gene expression",
        "```",
        'predict_variant_effect_on_gene(',
        '    oracle_name="enformer",',
        '    position="chr2:60490908",',
        '    ref_allele="T", alt_alleles=["G"],',
        '    gene_name="BCL11A",',
        '    assay_ids=["CNhs11250"],  # CAGE track for expression',
        ')',
        "```",
        "Returns fold change, log2FC, and absolute change in predicted expression.",
        "",
        "### Tips",
        "- **Multi-oracle comparison:** Load multiple oracles and compare predictions across models.",
        "- **Cell-type specificity:** Use tissue-relevant tracks (e.g., liver CAGE for lipid variants, ",
        "  erythroid tracks for hemoglobin variants).",
        "- **ChromBPNet for TF binding:** Use `chrombpnet` with specific TFs to predict binding changes ",
        "  at base resolution.",
        "- **Region parameter is optional:** When omitted, the region is auto-centered on the variant, ",
        "  correctly sized for each oracle's output window.",
        "",
    ]

    report_path = OUTPUT_DIR / "GWAS_variant_report.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport saved: {report_path}")


def main():
    print("Loading Enformer...")
    oracle = None
    all_results = []

    for v in VARIANTS:
        print(f"\n{'='*60}")
        print(f"Analyzing {v['rsid']} ({v['gene']}) — {v['chrom']}:{v['pos']}")
        print(f"{'='*60}")

        if oracle is None:
            oracle, result = load_and_predict(v)
        else:
            # Reuse loaded oracle
            alleles = [v["ref"], v["alt"]]
            region = f"{v['chrom']}:{v['pos']}-{v['pos']+1}"
            result = oracle.predict_variant_effect(
                genomic_region=region,
                variant_position=f"{v['chrom']}:{v['pos']}",
                alleles=alleles,
                assay_ids=v["assay_ids"],
            )

        effects = plot_variant(v, result, oracle)
        all_results.append((v, effects))

        # Print summary
        for label, e in effects.items():
            print(f"  {label}: mean={e['mean_effect']:.6f}, abs_max={e['abs_max_effect']:.6f}")

    generate_markdown(all_results)
    print("\nDone!")


if __name__ == "__main__":
    main()
