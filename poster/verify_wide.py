"""Rerun ChromBPNet (ATAC, DNase, CEBPA) and LegNet on rs12740374 with the
WIDE genomic_region used by the canonical regenerate_multioracle.py on v0.5.6.

This matches the latest chorus convention where ChromBPNet / LegNet use
``predict_sliding`` to extend signal across the full window.
"""
from __future__ import annotations
import json, logging, os, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("verify_wide")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
GENOME = os.path.join(REPO_ROOT, "genomes/hg38.fa")

VARIANT = dict(chrom="chr1", position=109274968, ref="G", alt="T",
               id="rs12740374", gene="SORT1")
POS = f"{VARIANT['chrom']}:{VARIANT['position']}"


def _max_window():
    from chorus.oracles.chrombpnet import ChromBPNetOracle
    from chorus.oracles.legnet import LegNetOracle
    from chorus.oracles.alphagenome import AlphaGenomeOracle
    return max(
        ChromBPNetOracle().output_size,
        LegNetOracle().sequence_length,
        AlphaGenomeOracle().output_size,
    )


def _wide_region():
    half = _max_window() // 2
    s = VARIANT["position"] - half
    e = VARIANT["position"] + half
    return f"{VARIANT['chrom']}:{s}-{e}"


def run(which):
    region = _wide_region()
    log.info("WIDE region: %s (width %s bp)", region, _max_window())

    from chorus.analysis.variant_report import build_variant_report
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.normalization import get_normalizer

    def save(name, report):
        path = os.path.join(OUT_DIR, f"_verified_{name}_wide_full.json")
        with open(path, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2, default=str)
        log.info("wrote %s", path)
        # also print the per-allele scores succinctly
        d = report.to_dict()
        for a, b in d.get("alleles", {}).items():
            for s in b.get("all_scores", []):
                rs = s.get("raw_score"); qs = s.get("quantile_score")
                log.info("  %s : raw=%s q=%s ref=%s alt=%s assay=%s",
                         a, rs, qs, s.get("ref_value"), s.get("alt_value"), s.get("assay_id"))

    try:
        n = get_normalizer(oracle_name="chrombpnet" if which.startswith("chromb") else which)
    except Exception:
        n = None

    if which == "chrombpnet_atac":
        from chorus.oracles.chrombpnet import ChromBPNetOracle
        o = ChromBPNetOracle(use_environment=False)
        o.load_pretrained_model(assay="ATAC", cell_type="HepG2", fold=0)
        r = o.predict_variant_effect(genomic_region=region, variant_position=POS,
                                     alleles=[VARIANT["ref"], VARIANT["alt"]],
                                     assay_ids=["ATAC:HepG2"], genome=GENOME)
        rep = build_variant_report(r, oracle_name="chrombpnet",
                                   gene_name=VARIANT["gene"], normalizer=n,
                                   analysis_request=AnalysisRequest(
                                       user_prompt="chrombpnet ATAC wide", tool_name="x",
                                       oracle_name="chrombpnet",
                                       normalizer_name="chorus per-track v1" if n else "(none)",
                                       tracks_requested="ATAC:HepG2"))
        save("chrombpnet_atac", rep)
    elif which == "chrombpnet_dnase":
        from chorus.oracles.chrombpnet import ChromBPNetOracle
        o = ChromBPNetOracle(use_environment=False)
        o.load_pretrained_model(assay="DNASE", cell_type="HepG2", fold=0)
        r = o.predict_variant_effect(genomic_region=region, variant_position=POS,
                                     alleles=[VARIANT["ref"], VARIANT["alt"]],
                                     assay_ids=["DNASE:HepG2"], genome=GENOME)
        rep = build_variant_report(r, oracle_name="chrombpnet",
                                   gene_name=VARIANT["gene"], normalizer=n,
                                   analysis_request=AnalysisRequest(
                                       user_prompt="chrombpnet DNase wide", tool_name="x",
                                       oracle_name="chrombpnet",
                                       normalizer_name="chorus per-track v1" if n else "(none)",
                                       tracks_requested="DNASE:HepG2"))
        save("chrombpnet_dnase", rep)
    elif which == "legnet":
        from chorus.oracles.legnet import LegNetOracle
        o = LegNetOracle(cell_type="HepG2", assay="LentiMPRA", use_environment=False)
        o.load_pretrained_model()
        r = o.predict_variant_effect(genomic_region=region, variant_position=POS,
                                     alleles=[VARIANT["ref"], VARIANT["alt"]],
                                     assay_ids=["LentiMPRA:HepG2"], genome=GENOME)
        rep = build_variant_report(r, oracle_name="legnet",
                                   gene_name=VARIANT["gene"], normalizer=None,
                                   analysis_request=AnalysisRequest(
                                       user_prompt="legnet wide", tool_name="x",
                                       oracle_name="legnet", normalizer_name="(none)",
                                       tracks_requested="LentiMPRA:HepG2"))
        save("legnet", rep)
    else:
        raise SystemExit("unknown: " + which)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True,
                    choices=["chrombpnet_atac", "chrombpnet_dnase", "legnet"])
    args = ap.parse_args()
    run(args.which)
