"""Regenerate the non-variant_analysis application examples.

Runs inside chorus-alphagenome env so the AlphaGenome model is loaded
ONCE and reused across all predictions. Covers:
  - discovery/SORT1_cell_type_screen
  - causal_prioritization/SORT1_locus
  - sequence_engineering/region_swap
  - sequence_engineering/integration_simulation
  - batch_scoring

Each example writes: example_output.md, example_output.json,
example_output.tsv, and an HTML report (when applicable).

Usage:
    mamba run -n chorus-alphagenome python scripts/regenerate_remaining_examples.py --gpu 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--only", choices=["discovery", "causal", "region_swap", "integration", "batch", "tert_chr5", "all", "skip_discovery"], default="all")
args = parser.parse_args()

os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE = os.path.join(REPO_ROOT, "examples/applications")

# ══════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════

def _variant_report_tsv_rows(report) -> list[dict]:
    """Flatten a VariantReport into TSV rows (one per track per allele)."""
    rows: list[dict] = []
    for allele, scores in report.allele_scores.items():
        seen: set[tuple] = set()
        for ts in scores:
            key = (allele, ts.assay_id, ts.layer)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "allele": allele,
                "layer": ts.layer,
                "assay_id": ts.assay_id,
                "assay_type": ts.assay_type,
                "cell_type": ts.cell_type,
                "description": ts.region_label or "",
                "ref_value": ts.ref_value,
                "alt_value": ts.alt_value,
                "raw_score": ts.raw_score,
                "quantile_score": ts.quantile_score,
                "ref_signal_percentile": ts.ref_signal_percentile,
                "note": ts.note,
            })
    return rows


def _write_tsv(rows: list[dict], out_path: str) -> None:
    import csv
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_variant_report(report, out_dir: str, html_name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/example_output.md", "w") as fh:
        fh.write(report.to_markdown())
    with open(f"{out_dir}/example_output.json", "w") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=str)
    _write_tsv(_variant_report_tsv_rows(report), f"{out_dir}/example_output.tsv")
    html_path = f"{out_dir}/{html_name}"
    report.to_html(output_path=html_path)
    size_kb = os.path.getsize(html_path) / 1024
    logger.info("  ✓ wrote MD/JSON/TSV/HTML (%s, %.0f KB)", html_name, size_kb)


# ══════════════════════════════════════════════════════════════════════
# Regeneration functions
# ══════════════════════════════════════════════════════════════════════

def regen_discovery(oracle, norm):
    """discovery/SORT1_cell_type_screen — screen all cell types for rs12740374."""
    from chorus.analysis.discovery import discover_and_report

    out_dir = f"{BASE}/discovery/SORT1_cell_type_screen"
    logger.info("=== DISCOVERY: SORT1 rs12740374 cell-type screen ===")

    # Remove old HTMLs
    import glob
    for old in glob.glob(f"{out_dir}/*.html"):
        os.remove(old)
        logger.info("  removed stale %s", os.path.basename(old))

    os.makedirs(out_dir, exist_ok=True)
    user_prompt = (
        "Screen all cell types for variant rs12740374 (chr1:109274968 G>T) "
        "using AlphaGenome. Find which cell types show the strongest "
        "chromatin and regulatory effects. Gene is SORT1."
    )
    result = discover_and_report(
        oracle,
        variant_position="chr1:109274968",
        alleles=["G", "T"],
        gene_name="SORT1",
        top_n=3,
        min_effect=0.15,
        output_path=out_dir,
        normalizer=norm,
        user_prompt=user_prompt,
    )

    # Write discovery_summary.json (list of top hits)
    with open(f"{out_dir}/discovery_summary.json", "w") as fh:
        json.dump(result.get("hits", []), fh, indent=2, default=str)

    hits = result.get("hits", [])
    reports = result.get("reports", {})

    # Build a combined markdown that lists top cell types and best tracks
    md_lines = [
        "## Discovery: SORT1 rs12740374 cell-type screen",
        "",
        f"**Variant**: chr1:109274968 G>T (rs12740374)",
        f"**Oracle**: alphagenome",
        f"**Gene**: SORT1",
        f"**Top cell types**: {len(hits)}",
        "",
        "| Rank | Cell type | Best effect | Best track | N tracks |",
        "|------|-----------|-------------|------------|----------|",
    ]
    for i, h in enumerate(hits, 1):
        md_lines.append(
            f"| {i} | {h.get('cell_type','')} | {h.get('effect',0):+.3f}"
            f" | {h.get('best_track','')} | {h.get('n_tracks','')} |"
        )
    md_lines.append("")
    for ct, report in reports.items():
        md_lines.append(f"### {ct}")
        md_lines.append("")
        md_lines.append(report.to_markdown())
        md_lines.append("")
    with open(f"{out_dir}/example_output.md", "w") as fh:
        fh.write("\n".join(md_lines))

    # Combined JSON
    combined = {
        "variant": {"chrom": "chr1", "position": 109274968, "ref": "G", "alt": "T", "id": "rs12740374"},
        "oracle": "alphagenome",
        "gene": "SORT1",
        "cell_type_ranking": hits,
        "reports": {ct: rpt.to_dict() for ct, rpt in reports.items()},
    }
    with open(f"{out_dir}/example_output.json", "w") as fh:
        json.dump(combined, fh, indent=2, default=str)

    # Combined TSV (union of tracks across reports, tagged with cell_type focus)
    all_rows = []
    for ct, rpt in reports.items():
        for row in _variant_report_tsv_rows(rpt):
            row["focus_cell_type"] = ct
            all_rows.append(row)
    if all_rows:
        import csv
        fieldnames = list(all_rows[0].keys())
        with open(f"{out_dir}/example_output.tsv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
    logger.info("  ✓ discovery: %d cell types, %d reports", len(hits), len(reports))


def regen_causal(oracle, norm):
    """causal_prioritization/SORT1_locus — fine-map 11 LD variants around rs12740374."""
    from chorus.analysis.causal import prioritize_causal_variants
    from chorus.utils.ld import LDVariant

    out_dir = f"{BASE}/causal_prioritization/SORT1_locus"
    logger.info("=== CAUSAL: SORT1 fine-mapping (11 variants) ===")

    ld_variants = [
        LDVariant(variant_id="rs12740374", chrom="chr1", position=109274968, ref="G", alt="T", r2=1.0),
        LDVariant(variant_id="rs4970836",  chrom="chr1", position=109279175, ref="G", alt="A", r2=0.907),
        LDVariant(variant_id="rs1624712",  chrom="chr1", position=109275908, ref="C", alt="T", r2=1.0),
        LDVariant(variant_id="rs660240",   chrom="chr1", position=109275216, ref="T", alt="C", r2=0.9509),
        LDVariant(variant_id="rs142678968",chrom="chr1", position=109275536, ref="C", alt="T", r2=0.9509),
        LDVariant(variant_id="rs1626484",  chrom="chr1", position=109275684, ref="G", alt="T", r2=1.0),
        LDVariant(variant_id="rs7528419",  chrom="chr1", position=109274570, ref="A", alt="G", r2=1.0),
        LDVariant(variant_id="rs56960352", chrom="chr1", position=109278685, ref="G", alt="T", r2=0.907),
        LDVariant(variant_id="rs1277930",  chrom="chr1", position=109279521, ref="G", alt="A", r2=0.907),
        LDVariant(variant_id="rs599839",   chrom="chr1", position=109279544, ref="G", alt="A", r2=0.907),
        LDVariant(variant_id="rs602633",   chrom="chr1", position=109278889, ref="T", alt="G", r2=0.8582),
    ]
    # Mark the sentinel
    ld_variants[0].is_sentinel = True

    from chorus.analysis.analysis_request import AnalysisRequest

    ar = AnalysisRequest(
        user_prompt=(
            "Fine-map the SORT1 LDL cholesterol GWAS locus. Sentinel is "
            "rs12740374 with 11 LD variants (r²≥0.85). Score each variant "
            "across HepG2 DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE. "
            "Rank by composite causal evidence. Gene is SORT1."
        ),
        tool_name="fine_map_causal_variant",
        oracle_name="alphagenome",
        tracks_requested=f"{len(HEPG2_TRACKS)} HepG2 tracks",
    )

    result = prioritize_causal_variants(
        oracle,
        lead_variant={"id": "rs12740374", "chrom": "chr1", "pos": 109274968, "ref": "G", "alt": "T"},
        ld_variants=ld_variants,
        assay_ids=HEPG2_TRACKS,
        gene_name="SORT1",
        oracle_name="alphagenome",
        normalizer=norm,
        analysis_request=ar,
    )

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/example_output.md", "w") as fh:
        fh.write(result.to_markdown())
    with open(f"{out_dir}/example_output.json", "w") as fh:
        json.dump(result.to_dict(), fh, indent=2, default=str)
    try:
        df = result.to_dataframe()
        df.to_csv(f"{out_dir}/example_output.tsv", sep="\t", index=False)
    except Exception as exc:
        logger.warning("  tsv write failed: %s", exc)
    try:
        html_path = f"{out_dir}/rs12740374_SORT1_locus_causal_report.html"
        result.to_html(output_path=html_path)
        logger.info("  ✓ HTML: %s (%.0f KB)", os.path.basename(html_path), os.path.getsize(html_path)/1024)
    except Exception as exc:
        logger.warning("  html write failed: %s", exc)

    top = result.top_candidate()
    logger.info("  ✓ causal: top=%s composite=%.3f", top.variant_id if top else "?", top.composite if top else 0)


def regen_region_swap(oracle, norm):
    """sequence_engineering/region_swap — replace a small region with a strong K562 promoter."""
    from chorus.analysis.region_swap import analyze_region_swap

    out_dir = f"{BASE}/sequence_engineering/region_swap"
    logger.info("=== SEQUENCE ENG: region_swap ===")

    # Existing example uses chr1:1000500-1001500 + strong promoter
    # K562 DNASE/H3K27ac/H3K4me3/CAGE tracks
    replacement = (
        "GCCACCATGGCCACCATGGCCACCATGGCCACCATGGCCACCATGGCCACCATGGCCACCATG"
        "CGAATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGAC"
        "GGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGC"
        "AAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTG"
        "ACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGAC"
        "TTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGAC"
        "GGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAG"
        "CTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTAC"
        "AACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAG"
        "ATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCC"
    )
    # Real AlphaGenome K562 track identifiers
    assay_ids = [
        "DNASE/EFO:0002067 DNase-seq/.",
        "CHIP_HISTONE/EFO:0002067 Histone ChIP-seq H3K27ac/.",
        "CHIP_HISTONE/EFO:0002067 Histone ChIP-seq H3K4me3/.",
        "CAGE/hCAGE EFO:0002067/+",
    ]

    from chorus.analysis.analysis_request import AnalysisRequest

    # Use the SORT1 enhancer region (well-assembled, real signal in K562)
    report = analyze_region_swap(
        oracle,
        region="chr1:109274500-109275500",
        replacement_sequence=replacement,
        assay_ids=assay_ids,
        gene_name="SORT1",
        normalizer=norm,
        oracle_name="alphagenome",
    )
    report.analysis_request = AnalysisRequest(
        user_prompt=(
            "Replace the SORT1 enhancer region chr1:109274500-109275500 with a "
            f"{len(replacement)} bp GFP/reporter construct sequence and predict "
            "effects on K562 DNASE, H3K27ac, H3K4me3, and CAGE."
        ),
        tool_name="analyze_region_swap",
        oracle_name="alphagenome",
        tracks_requested=f"{len(assay_ids)} K562 tracks",
    )
    save_variant_report(report, out_dir, "region_swap_SORT1_K562_report.html")


def regen_integration(oracle, norm):
    """sequence_engineering/integration_simulation — insert a CMV construct at chr19:55115000."""
    from chorus.analysis.integration import simulate_integration

    out_dir = f"{BASE}/sequence_engineering/integration_simulation"
    logger.info("=== SEQUENCE ENG: integration_simulation ===")

    # CMV promoter + reporter (simplified)
    construct = (
        "TAGTTATTAATAGTAATCAATTACGGGGTCATTAGTTCATAGCCCATATATGGAGTTCCGCGT"
        "TACATAACTTACGGTAAATGGCCCGCCTGGCTGACCGCCCAACGACCCCCGCCCATTGACGTC"
        "AATAATGACGTATGTTCCCATAGTAACGCCAATAGGGACTTTCCATTGACGTCAATGGGTGGA"
        "GTATTTACGGTAAACTGCCCACTTGGCAGTACATCAAGTGTATCATATGCCAAGTACGCCCCC"
        "TATTGACGTCAATGACGGTAAATGGCCCGCCTGGCATTATGCCCAGTACATGACCTTATGGGA"
        "CTTTCCTACTTGGCAGTACATCTACGTATTAGTCATCGCTATTACCATGGTGATGCGGTTTTG"
    )
    assay_ids = [
        "DNASE/EFO:0002067 DNase-seq/.",
        "CHIP_HISTONE/EFO:0002067 Histone ChIP-seq H3K27ac/.",
        "CAGE/hCAGE EFO:0002067/+",
    ]

    # Insert construct at a well-assembled locus on chr19 (GAPDH region ~ chr12 but
    # for K562 we pick PPP1R12C on chr19 which is assembled and transcribed)
    from chorus.analysis.analysis_request import AnalysisRequest

    report = simulate_integration(
        oracle,
        position="chr19:55115000",
        construct_sequence=construct,
        assay_ids=assay_ids,
        gene_name="PPP1R12C",
        normalizer=norm,
        oracle_name="alphagenome",
    )
    report.analysis_request = AnalysisRequest(
        user_prompt=(
            f"Insert a {len(construct)} bp CMV promoter construct at "
            "chr19:55115000 (PPP1R12C locus / AAVS1 safe harbour) and predict "
            "local disruption in K562 using DNASE, H3K27ac, and CAGE tracks."
        ),
        tool_name="simulate_integration",
        oracle_name="alphagenome",
        tracks_requested=f"{len(assay_ids)} K562 tracks",
    )
    save_variant_report(report, out_dir, "integration_CMV_PPP1R12C_report.html")


def regen_tert_chr5(oracle, norm):
    """validation/TERT_chr5_1295046 — re-run with TF/mark fix."""
    from chorus.analysis.discovery import discover_variant_effects
    from chorus.analysis.analysis_request import AnalysisRequest

    out_dir = f"{BASE}/validation/TERT_chr5_1295046"
    logger.info("=== RE-RUN: TERT chr5:1295046 (CHIP label fix) ===")

    ar = AnalysisRequest(
        user_prompt=(
            "Validate the TERT chr5:1295046 T>G variant from the AlphaGenome "
            "paper. Score across all tracks in discovery mode. Gene is TERT."
        ),
        tool_name="discover_variant",
        oracle_name="alphagenome",
        tracks_requested="all tracks (discovery mode)",
    )

    html_name = "chr5_1295046_T_G_TERT_alphagenome_report.html"
    result = discover_variant_effects(
        oracle, oracle_name="alphagenome",
        variant_position="chr5:1295046",
        alleles=["T", "G"],
        gene_name="TERT",
        normalizer=norm,
        output_path=out_dir,
        output_filename=html_name,
        top_n_per_layer=3, top_n_cell_types=5,
        analysis_request=ar,
    )
    report = result.get("report")
    if report is None:
        logger.warning("  no report returned")
        return
    with open(f"{out_dir}/example_output.md", "w") as fh:
        fh.write(report.to_markdown())
    with open(f"{out_dir}/example_output.json", "w") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=str)
    _write_tsv(_variant_report_tsv_rows(report), f"{out_dir}/example_output.tsv")

    target = f"{out_dir}/{html_name}"
    logger.info("  ✓ %s (%.0f KB)", os.path.basename(target), os.path.getsize(target)/1024)


HEPG2_TRACKS = [
    "DNASE/EFO:0001187 DNase-seq/.",
    "CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA genetically modified (insertion) using CRISPR targeting H. sapiens CEBPA/.",
    "CHIP_TF/EFO:0001187 TF ChIP-seq CEBPB/.",
    "CHIP_HISTONE/EFO:0001187 Histone ChIP-seq H3K27ac/.",
    "CAGE/hCAGE EFO:0001187/+",
    "CAGE/hCAGE EFO:0001187/-",
]


def regen_batch(oracle, norm):
    """batch_scoring — rank 5 SORT1-locus variants in HepG2 liver tracks."""
    from chorus.analysis.batch_scoring import score_variant_batch

    out_dir = f"{BASE}/batch_scoring"
    logger.info("=== BATCH SCORING: SORT1 locus (5 variants × HepG2 tracks) ===")

    variants = [
        {"chrom": "chr1", "pos": 109274968, "ref": "G", "alt": "T", "id": "rs12740374"},
        {"chrom": "chr1", "pos": 109275684, "ref": "G", "alt": "T", "id": "rs1626484"},
        {"chrom": "chr1", "pos": 109275216, "ref": "T", "alt": "C", "id": "rs660240"},
        {"chrom": "chr1", "pos": 109279175, "ref": "G", "alt": "A", "id": "rs4970836"},
        {"chrom": "chr1", "pos": 109274570, "ref": "A", "alt": "G", "id": "rs7528419"},
    ]

    from chorus.analysis.analysis_request import AnalysisRequest

    ar = AnalysisRequest(
        user_prompt=(
            "Score 5 SORT1-locus GWAS variants in HepG2 liver cells using "
            "DNASE, CEBPA/CEBPB ChIP, H3K27ac, and CAGE tracks. Rank by "
            "regulatory effect. Gene is SORT1."
        ),
        tool_name="score_variant_batch",
        oracle_name="alphagenome",
        tracks_requested=f"{len(HEPG2_TRACKS)} HepG2 tracks",
    )

    result = score_variant_batch(
        oracle,
        variants=variants,
        assay_ids=HEPG2_TRACKS,
        gene_name="SORT1",
        normalizer=norm,
        oracle_name="alphagenome",
        analysis_request=ar,
    )

    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/example_output.md", "w") as fh:
        fh.write(result.to_markdown(display_mode="by_assay"))
    with open(f"{out_dir}/example_output.json", "w") as fh:
        json.dump(result.to_dict(), fh, indent=2, default=str)
    result.to_tsv(f"{out_dir}/example_output.tsv")
    result.to_html(
        output_path=f"{out_dir}/batch_sort1_locus_scoring.html",
        display_mode="by_assay",
    )
    logger.info("  ✓ batch: %d variants × %d tracks", len(result.scores), len(HEPG2_TRACKS))


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    from chorus.oracles.alphagenome import AlphaGenomeOracle
    from chorus.analysis.normalization import get_pertrack_normalizer

    logger.info("Loading AlphaGenome (direct, in-process)...")
    t0 = time.time()
    oracle = AlphaGenomeOracle(
        use_environment=False,
        reference_fasta=os.path.join(REPO_ROOT, "genomes/hg38.fa"),
        device=f"cuda:{args.gpu}",
    )
    oracle.load_pretrained_model()
    norm = get_pertrack_normalizer("alphagenome")
    logger.info("Model loaded in %.0fs; normalizer loaded=%s", time.time() - t0, norm is not None)

    tasks = []
    if args.only == "discovery":
        tasks.append(("discovery", regen_discovery))
    elif args.only == "region_swap":
        tasks.append(("region_swap", regen_region_swap))
    elif args.only == "integration":
        tasks.append(("integration", regen_integration))
    elif args.only == "batch":
        tasks.append(("batch", regen_batch))
    elif args.only == "causal":
        tasks.append(("causal", regen_causal))
    elif args.only == "tert_chr5":
        tasks.append(("tert_chr5", regen_tert_chr5))
    elif args.only == "skip_discovery":
        tasks.extend([
            ("region_swap", regen_region_swap),
            ("integration", regen_integration),
            ("batch", regen_batch),
            ("causal", regen_causal),
            ("tert_chr5", regen_tert_chr5),
        ])
    else:  # all
        tasks.extend([
            ("discovery", regen_discovery),
            ("region_swap", regen_region_swap),
            ("integration", regen_integration),
            ("batch", regen_batch),
            ("causal", regen_causal),
            ("tert_chr5", regen_tert_chr5),
        ])

    for name, fn in tasks:
        t = time.time()
        try:
            fn(oracle, norm)
        except Exception as exc:
            logger.error("✗ %s failed: %s", name, exc, exc_info=True)
            continue
        logger.info("%s done in %.0fs\n", name, time.time() - t)

    logger.info("=" * 60)
    logger.info("ALL DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
