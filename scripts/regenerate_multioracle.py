"""Regenerate the multi-oracle validation example for SORT1 rs12740374.

The report combines three oracles that look at the same variant from very
different angles:

    ChromBPNet      — chromatin accessibility (ATAC/DNase), HepG2
    LegNet          — promoter activity (LentiMPRA), HepG2
    AlphaGenome     — generalist: ChIP, histone marks, CAGE, etc.

Each oracle runs inside its own conda env:

    mamba run -n chorus-chrombpnet python scripts/regenerate_multioracle.py --oracle chrombpnet
    mamba run -n chorus-legnet     python scripts/regenerate_multioracle.py --oracle legnet
    mamba run -n chorus-alphagenome python scripts/regenerate_multioracle.py --oracle alphagenome

Then the consolidator — which has no GPU requirement and runs in any env:

    mamba run -n chorus python scripts/regenerate_multioracle.py --consolidate

produces a single ``rs12740374_SORT1_multioracle_report.html`` along with
a consolidated ``example_output.md``/``.json``.

The per-oracle JSON files are written to
``examples/applications/validation/SORT1_rs12740374_multioracle/`` and can
be re-consolidated at any time — e.g. after refreshing a single oracle.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = os.path.join(
    REPO_ROOT,
    "examples/applications/validation/SORT1_rs12740374_multioracle",
)

# Reference genome (shared by every oracle here).
GENOME_REF = os.path.join(REPO_ROOT, "genomes/hg38.fa")

# ---------------------------------------------------------------------------
# Variant and request description (shared across all oracles)
# ---------------------------------------------------------------------------

VARIANT = {
    "chrom": "chr1",
    "position": 109274968,
    "ref": "G",
    "alt": "T",
    "id": "rs12740374",
    "gene": "SORT1",
}

USER_PROMPT = (
    "Validate rs12740374 (the classic SORT1 LDL-cholesterol causal variant) "
    "by scoring it with three independent deep-learning oracles: ChromBPNet "
    "for chromatin accessibility, LegNet for MPRA promoter activity, and "
    "AlphaGenome as a generalist model covering ChIP, histones and CAGE. "
    "A new user should be able to see at a glance whether the three oracles "
    "agree on direction, and which assay/cell type drove each call."
)


# ---------------------------------------------------------------------------
# Per-oracle runners
# ---------------------------------------------------------------------------

def _build_variant_report(oracle, oracle_name: str, assay_ids=None):
    """Score the SORT1 variant with the given oracle and return the report."""
    from chorus.analysis.variant_report import build_variant_report
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.normalization import get_normalizer

    normalizer = None
    try:
        normalizer = get_normalizer(oracle_name=oracle_name)
    except Exception as exc:
        logger.warning("No normalizer for %s: %s — percentile columns absent.",
                       oracle_name, exc)

    logger.info("Predicting variant effect with %s ...", oracle_name)
    # Provide a small genomic region centred on the variant. Most oracles
    # only look ±half-window-size around the variant position; passing a 2bp
    # region here keeps the API contract satisfied without wasting compute.
    position_str = f"{VARIANT['chrom']}:{VARIANT['position']}"
    region_str = f"{VARIANT['chrom']}:{VARIANT['position']}-{VARIANT['position'] + 1}"
    result = oracle.predict_variant_effect(
        genomic_region=region_str,
        variant_position=position_str,
        alleles=[VARIANT["ref"], VARIANT["alt"]],
        assay_ids=assay_ids if assay_ids else [oracle.assay_id],
        genome=GENOME_REF,
    )
    ar = AnalysisRequest(
        user_prompt=USER_PROMPT,
        tool_name="analyze_variant_multilayer",
        oracle_name=oracle_name,
        normalizer_name="chorus per-track v1" if normalizer else "(none)",
        tracks_requested=(
            f"{len(assay_ids)} tracks" if assay_ids else "all oracle tracks"
        ),
    )
    logger.info("Building variant report ...")
    return build_variant_report(
        result, oracle_name=oracle_name, gene_name=VARIANT["gene"],
        normalizer=normalizer, analysis_request=ar,
    )


def _save_oracle_artefacts(report, oracle_name: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, f"{oracle_name}_variant_report.json")
    with open(json_path, "w") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=str)
    html_path = os.path.join(
        OUT_DIR, f"{VARIANT['id']}_{VARIANT['gene']}_{oracle_name}_report.html"
    )
    report.to_html(output_path=html_path)
    logger.info("  ✓ wrote %s and %s",
                os.path.basename(json_path), os.path.basename(html_path))
    return json_path, html_path


def run_chrombpnet():
    from chorus.oracles.chrombpnet import ChromBPNetOracle
    oracle = ChromBPNetOracle()
    oracle.load_pretrained_model(assay="ATAC", cell_type="HepG2", fold=0)
    report = _build_variant_report(oracle, oracle_name="chrombpnet")
    return _save_oracle_artefacts(report, "chrombpnet")


def run_legnet():
    from chorus.oracles.legnet import LegNetOracle
    oracle = LegNetOracle(cell_type="HepG2", assay="LentiMPRA")
    oracle.load_pretrained_model()
    report = _build_variant_report(oracle, oracle_name="legnet")
    return _save_oracle_artefacts(report, "legnet")


# HepG2 tracks for AlphaGenome — kept small & focused so the consensus
# matrix highlights the multi-layer picture rather than being dominated by
# hundreds of near-zero tracks.
ALPHAGENOME_TRACKS = [
    "DNASE/EFO:0001187 DNase-seq/.",
    "CHIP/hCAGE EFO:0001187 CEBPA/.",
    "CHIP/hCAGE EFO:0001187 CEBPB/.",
    "CHIP/hCAGE EFO:0001187 H3K27ac/.",
    "CAGE/hCAGE EFO:0001187/-",
    "CAGE/hCAGE EFO:0001187/+",
]


def run_alphagenome():
    from chorus.oracles.alphagenome import AlphaGenome
    oracle = AlphaGenome()
    report = _build_variant_report(
        oracle, oracle_name="alphagenome", assay_ids=ALPHAGENOME_TRACKS,
    )
    return _save_oracle_artefacts(report, "alphagenome")


# ---------------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------------

def consolidate():
    """Assemble the multi-oracle HTML from per-oracle JSONs in OUT_DIR."""
    from chorus.analysis import MultiOracleReport
    from chorus.analysis.analysis_request import AnalysisRequest

    per_oracle = {}
    paths = []
    for oracle_name in ("chrombpnet", "legnet", "alphagenome"):
        jp = os.path.join(OUT_DIR, f"{oracle_name}_variant_report.json")
        if os.path.isfile(jp):
            paths.append(jp)
            # Prefer a relative link for portability.
            html_fname = (
                f"{VARIANT['id']}_{VARIANT['gene']}_{oracle_name}_report.html"
            )
            if os.path.isfile(os.path.join(OUT_DIR, html_fname)):
                per_oracle[oracle_name] = html_fname
        else:
            logger.warning("Missing per-oracle JSON for %s — skipped.", oracle_name)

    if not paths:
        raise SystemExit(
            "No per-oracle JSONs found. Run --oracle chrombpnet/legnet/"
            "alphagenome first."
        )

    ar = AnalysisRequest(
        user_prompt=USER_PROMPT,
        tool_name="MultiOracleReport",
        oracle_name=", ".join(os.path.splitext(os.path.basename(p))[0]
                              .replace("_variant_report", "") for p in paths),
        normalizer_name="per-oracle chorus per-track v1",
        tracks_requested="assay_ids as listed in each per-oracle request",
    )
    moracle = MultiOracleReport.from_json_files(
        paths,
        variant_id=VARIANT["id"],
        analysis_request=ar,
        per_oracle_report_paths=per_oracle,
    )

    html_path = os.path.join(OUT_DIR, f"{VARIANT['id']}_{VARIANT['gene']}_multioracle_report.html")
    moracle.to_html(output_path=html_path)

    md_path = os.path.join(OUT_DIR, "example_output.md")
    with open(md_path, "w") as fh:
        fh.write(moracle.to_markdown())

    json_path = os.path.join(OUT_DIR, "example_output.json")
    with open(json_path, "w") as fh:
        json.dump(moracle.to_dict(), fh, indent=2, default=str)

    logger.info("  ✓ wrote %s", os.path.basename(html_path))
    logger.info("  ✓ wrote %s", os.path.basename(md_path))
    logger.info("  ✓ wrote %s", os.path.basename(json_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oracle",
        choices=["chrombpnet", "legnet", "alphagenome"],
        help="Run a single oracle and save its VariantReport JSON to OUT_DIR.",
    )
    parser.add_argument(
        "--consolidate", action="store_true",
        help="Read per-oracle JSONs from OUT_DIR and write the multi-oracle HTML.",
    )
    args = parser.parse_args()

    if args.oracle == "chrombpnet":
        run_chrombpnet()
    elif args.oracle == "legnet":
        run_legnet()
    elif args.oracle == "alphagenome":
        run_alphagenome()

    if args.consolidate:
        consolidate()

    if not (args.oracle or args.consolidate):
        parser.error("pass --oracle <name> and/or --consolidate")


if __name__ == "__main__":
    main()
