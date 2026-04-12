"""Regenerate all application example outputs with current normalization.

Runs inside chorus-alphagenome env for AlphaGenome examples (loads model
once), then switches to chorus env for Enformer/ChromBPNet examples.

Usage:
  # AlphaGenome examples (most of them):
  mamba run -n chorus-alphagenome python scripts/regenerate_examples.py --oracle alphagenome --gpu 1

  # Enformer examples:
  mamba run -n chorus python scripts/regenerate_examples.py --oracle enformer --gpu 1

  # ChromBPNet examples:
  mamba run -n chorus python scripts/regenerate_examples.py --oracle chrombpnet --gpu 1
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--oracle", choices=["alphagenome", "enformer", "chrombpnet", "all"], default="all")
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--dry-run", action="store_true", help="List what would be regenerated without running")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE = os.path.join(REPO_ROOT, 'examples/applications')

# ══════════════════════════════════════════════════════════════════
# Example definitions
# ══════════════════════════════════════════════════════════════════

ALPHAGENOME_EXAMPLES = [
    # variant_analysis
    {
        "name": "SORT1 rs12740374 (AlphaGenome)",
        "dir": f"{BASE}/variant_analysis/SORT1_rs12740374",
        "type": "variant",
        "position": "chr1:109274968",
        "ref": "G", "alt": "T",
        "gene": "SORT1",
        "html_name": "rs12740374_SORT1_alphagenome_report.html",
    },
    {
        "name": "BCL11A rs1427407",
        "dir": f"{BASE}/variant_analysis/BCL11A_rs1427407",
        "type": "variant",
        "position": "chr2:60490908",
        "ref": "G", "alt": "T",
        "gene": "BCL11A",
        "html_name": "rs1427407_BCL11A_alphagenome_report.html",
    },
    {
        "name": "FTO rs1421085",
        "dir": f"{BASE}/variant_analysis/FTO_rs1421085",
        "type": "variant",
        "position": "chr16:53767042",
        "ref": "T", "alt": "C",
        "gene": "FTO",
        "html_name": "rs1421085_FTO_alphagenome_report.html",
    },
    {
        "name": "TERT promoter (C228T)",
        "dir": f"{BASE}/variant_analysis/TERT_promoter",
        "type": "variant",
        "position": "chr5:1295228",
        "ref": "G", "alt": "A",
        "gene": "TERT",
        "html_name": "TERT_promoter_alphagenome_report.html",
    },
    # validation
    {
        "name": "SORT1 with CEBP (validation)",
        "dir": f"{BASE}/validation/SORT1_rs12740374_with_CEBP",
        "type": "variant",
        "position": "chr1:109274968",
        "ref": "G", "alt": "T",
        "gene": "CELSR2",
        "html_name": "chr1_109274968_G_T_CELSR2_alphagenome_report.html",
    },
    {
        "name": "TERT chr5:1295046 (validation)",
        "dir": f"{BASE}/validation/TERT_chr5_1295046",
        "type": "variant",
        "position": "chr5:1295046",
        "ref": "T", "alt": "G",
        "gene": "TERT",
        "html_name": "chr5_1295046_T_G_TERT_alphagenome_report.html",
    },
    {
        "name": "HBG2 HPFH (validation)",
        "dir": f"{BASE}/validation/HBG2_HPFH",
        "type": "variant",
        "position": "chr11:5254983",
        "ref": "G", "alt": "C",
        "gene": "HBG2",
        "html_name": "chr11_5254983_G_C_HBG2_alphagenome_report.html",
    },
]

ENFORMER_EXAMPLES = [
    {
        "name": "SORT1 rs12740374 (Enformer)",
        "dir": f"{BASE}/variant_analysis/SORT1_enformer",
        "type": "discovery",
        "position": "chr1:109274968",
        "ref": "G", "alt": "T",
        "gene": "SORT1",
        "html_name": "rs12740374_SORT1_enformer_report.html",
    },
    {
        "name": "SORT1 Enformer validation (rescale)",
        "dir": f"{BASE}/validation/SORT1_rs12740374_with_CEBP",
        "type": "discovery",
        "position": "chr1:109274968",
        "ref": "G", "alt": "T",
        "gene": "SORT1",
        "html_name": "chr1_109274968_G_T_SORT1_enformer_report.html",
    },
    {
        "name": "SORT1 Enformer validation (raw autoscale)",
        "dir": f"{BASE}/validation/SORT1_rs12740374_with_CEBP",
        "type": "discovery_raw",
        "position": "chr1:109274968",
        "ref": "G", "alt": "T",
        "gene": "SORT1",
        "html_name": "chr1_109274968_G_T_SORT1_enformer_RAW_autoscale.html",
    },
]

CHROMBPNET_EXAMPLES = [
    {
        "name": "SORT1 ChromBPNet (ATAC HepG2)",
        "dir": f"{BASE}/variant_analysis/SORT1_chrombpnet",
        "type": "chrombpnet",
        "position": "chr1:109274968",
        "ref": "G", "alt": "T",
        "gene": "SORT1",
        "assay": "ATAC", "cell_type": "HepG2",
        "html_name": "rs12740374_SORT1_chrombpnet_report.html",
    },
]


def regenerate_variant_alphagenome(oracle, norm, example):
    """Regenerate a variant analysis example using AlphaGenome."""
    from chorus.analysis.discovery import discover_variant_effects
    import glob

    out_dir = example["dir"]
    logger.info("Regenerating: %s → %s", example["name"], out_dir)

    result = discover_variant_effects(
        oracle, oracle_name="alphagenome",
        variant_position=example["position"],
        alleles=[example["ref"], example["alt"]],
        gene_name=example["gene"],
        normalizer=norm,
        output_path=out_dir,
        top_n_per_layer=10, top_n_cell_types=8,
    )

    report = result.get("report")
    if report is None:
        logger.warning("  No report generated for %s", example["name"])
        return False

    # Save markdown + JSON
    md = report.to_markdown()
    with open(f'{out_dir}/example_output.md', 'w') as f:
        f.write(md)
    d = report.to_dict()
    with open(f'{out_dir}/example_output.json', 'w') as f:
        json.dump(d, f, indent=2, default=str)

    # Rename the auto-generated HTML to the expected name
    htmls = sorted(glob.glob(f'{out_dir}/chr*.html'), key=os.path.getmtime)
    if htmls:
        target = f'{out_dir}/{example["html_name"]}'
        os.rename(htmls[-1], target)
        logger.info("  ✓ HTML: %s (%s)", example["html_name"],
                     f"{os.path.getsize(target)/1024:.0f} KB")

    logger.info("  ✓ %s: %d tracks scored, %d selected",
                example["name"], result["total_tracks_scored"], result["selected_tracks"])
    return True


def regenerate_enformer_discovery(oracle, norm, example, igv_raw=False):
    """Regenerate an Enformer discovery example."""
    from chorus.analysis.discovery import discover_variant_effects
    import glob

    out_dir = example["dir"]
    logger.info("Regenerating: %s → %s (igv_raw=%s)", example["name"], out_dir, igv_raw)

    result = discover_variant_effects(
        oracle, oracle_name="enformer",
        variant_position=example["position"],
        alleles=[example["ref"], example["alt"]],
        gene_name=example["gene"],
        normalizer=norm,
        output_path=out_dir,
        top_n_per_layer=12, top_n_cell_types=15,
        igv_raw=igv_raw,
    )

    report = result.get("report")
    if report is None:
        logger.warning("  No report generated")
        return False

    md = report.to_markdown()
    with open(f'{out_dir}/example_output.md', 'w') as f:
        f.write(md)
    d = report.to_dict()
    with open(f'{out_dir}/example_output.json', 'w') as f:
        json.dump(d, f, indent=2, default=str)

    htmls = sorted(glob.glob(f'{out_dir}/chr*.html'), key=os.path.getmtime)
    if htmls:
        target = f'{out_dir}/{example["html_name"]}'
        os.rename(htmls[-1], target)
        logger.info("  ✓ HTML: %s", example["html_name"])

    logger.info("  ✓ Done")
    return True


def regenerate_chrombpnet(oracle, norm, example):
    """Regenerate a ChromBPNet example."""
    from chorus.analysis.variant_report import build_variant_report

    out_dir = example["dir"]
    logger.info("Regenerating: %s → %s", example["name"], out_dir)

    oracle.load_pretrained_model(assay=example["assay"], cell_type=example["cell_type"], fold=0)
    result = oracle.predict_variant_effect(
        genomic_region=f'{example["position"].split(":")[0]}:{int(example["position"].split(":")[1])-100}-{int(example["position"].split(":")[1])+100}',
        variant_position=example["position"],
        alleles=[example["ref"], example["alt"]],
        assay_ids=[],
    )

    report = build_variant_report(result, oracle_name="chrombpnet",
                                   gene_name=example["gene"], normalizer=norm)

    md = report.to_markdown()
    with open(f'{out_dir}/example_output.md', 'w') as f:
        f.write(md)
    d = report.to_dict()
    with open(f'{out_dir}/example_output.json', 'w') as f:
        json.dump(d, f, indent=2, default=str)

    html_path = report.to_html(output_path=f'{out_dir}/{example["html_name"]}')
    logger.info("  ✓ HTML: %s", html_path)
    return True


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if args.oracle in ("alphagenome", "all"):
    if args.dry_run:
        print(f"Would regenerate {len(ALPHAGENOME_EXAMPLES)} AlphaGenome examples")
        for ex in ALPHAGENOME_EXAMPLES:
            print(f"  {ex['name']}: {ex['position']} {ex['ref']}>{ex['alt']}")
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
        logger.info("=" * 60)
        logger.info("ALPHAGENOME EXAMPLES (%d)", len(ALPHAGENOME_EXAMPLES))
        logger.info("=" * 60)

        # Use the oracle env-runner wrapper (loads model in chorus-alphagenome
        # subprocess). Each prediction reloads the model (~2 min per call)
        # but this is the only way to call from the chorus env.
        from chorus.oracles.alphagenome import AlphaGenomeOracle
        from chorus.analysis.normalization import get_pertrack_normalizer

        logger.info("Loading AlphaGenome oracle (via env runner)...")
        oracle = AlphaGenomeOracle(
            use_environment=True,
            reference_fasta=os.path.join(REPO_ROOT, 'genomes/hg38.fa'),
            device=f'cuda:{args.gpu}',
            predict_timeout=900, model_load_timeout=1200,
        )
        oracle.load_pretrained_model()
        norm = get_pertrack_normalizer('alphagenome')

        t0 = time.time()
        for i, ex in enumerate(ALPHAGENOME_EXAMPLES):
            logger.info("--- Example %d/%d ---", i + 1, len(ALPHAGENOME_EXAMPLES))
            try:
                regenerate_variant_alphagenome(oracle, norm, ex)
            except Exception as exc:
                logger.error("  ✗ Failed: %s", str(exc)[:200])
            logger.info("")

        logger.info("AlphaGenome examples done in %.1f min", (time.time() - t0) / 60)

if args.oracle in ("enformer", "all"):
    if args.dry_run:
        print(f"Would regenerate {len(ENFORMER_EXAMPLES)} Enformer examples")
    else:
        logger.info("=" * 60)
        logger.info("ENFORMER EXAMPLES (%d)", len(ENFORMER_EXAMPLES))
        logger.info("=" * 60)

        from chorus.oracles.enformer import EnformerOracle
        from chorus.analysis.normalization import get_pertrack_normalizer

        oracle = EnformerOracle(
            use_environment=True,
            reference_fasta=os.path.join(REPO_ROOT, 'genomes/hg38.fa'),
            device=f'cuda:{args.gpu}',
            predict_timeout=600, model_load_timeout=900,
        )
        oracle.load_pretrained_model()
        norm = get_pertrack_normalizer('enformer')

        for ex in ENFORMER_EXAMPLES:
            try:
                igv_raw = ex["type"] == "discovery_raw"
                regenerate_enformer_discovery(oracle, norm, ex, igv_raw=igv_raw)
            except Exception as exc:
                logger.error("  ✗ Failed: %s", str(exc)[:200])

if args.oracle in ("chrombpnet", "all"):
    if args.dry_run:
        print(f"Would regenerate {len(CHROMBPNET_EXAMPLES)} ChromBPNet examples")
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
        logger.info("=" * 60)
        logger.info("CHROMBPNET EXAMPLES (%d)", len(CHROMBPNET_EXAMPLES))
        logger.info("=" * 60)

        from chorus.oracles.chrombpnet import ChromBPNetOracle
        from chorus.analysis.normalization import get_pertrack_normalizer

        oracle = ChromBPNetOracle(
            use_environment=True,
            reference_fasta=os.path.join(REPO_ROOT, 'genomes/hg38.fa'),
            device=f'cuda:{args.gpu}',
            model_load_timeout=600, predict_timeout=300,
        )
        norm = get_pertrack_normalizer('chrombpnet')

        for ex in CHROMBPNET_EXAMPLES:
            try:
                regenerate_chrombpnet(oracle, norm, ex)
            except Exception as exc:
                logger.error("  ✗ Failed: %s", str(exc)[:200])

logger.info("=" * 60)
logger.info("ALL DONE")
logger.info("=" * 60)
