"""One-off verification script for the poster panel.

Reruns rs12740374 (chr1:109,274,968 G>T) at SORT1 in HepG2 with:

1. AlphaGenome   — DNase, ChIP-CEBPA, H3K27ac, CAGE  + new: RNA-seq HepG2
2. ChromBPNet    — ATAC HepG2  + new: CEBPA HepG2 ChIP (BPNet TF model)
3. LegNet        — LentiMPRA HepG2

Writes one JSON per oracle to poster/_verified_<oracle>.json.

Run each block in the matching mamba env, e.g.
    mamba run -n chorus-alphagenome python poster/_verify_predictions.py --oracle alphagenome
    mamba run -n chorus-chrombpnet  python poster/_verify_predictions.py --oracle chrombpnet
    mamba run -n chorus-legnet      python poster/_verify_predictions.py --oracle legnet
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
log = logging.getLogger("verify")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
GENOME = os.path.join(REPO_ROOT, "genomes/hg38.fa")

VARIANT = dict(
    chrom="chr1", position=109274968, ref="G", alt="T",
    id="rs12740374", gene="SORT1",
)
REGION = f"{VARIANT['chrom']}:{VARIANT['position']}-{VARIANT['position']+1}"
POS    = f"{VARIANT['chrom']}:{VARIANT['position']}"


def _summarise(report, oracle_name):
    """Pull per-track rows via VariantReport.to_dataframe (one row per allele per track)."""
    try:
        df = report.to_dataframe()
    except Exception as exc:
        log.warning("to_dataframe failed on %s: %s", oracle_name, exc)
        return []
    rows = []
    for r in df.to_dict(orient="records"):
        rows.append({
            "allele":      r.get("allele"),
            "assay_id":    r.get("assay_id"),
            "layer":       r.get("layer"),
            "assay_type":  r.get("assay_type"),
            "cell_type":   r.get("cell_type"),
            "ref_value":   r.get("ref_value"),
            "alt_value":   r.get("alt_value"),
            "raw_score":   r.get("raw_score"),
            "quantile":    r.get("quantile_score"),
            "ref_pctile":  r.get("ref_signal_percentile"),
        })
    return rows


def _save_full(oracle_name, report):
    """Also dump the full to_dict so nothing is lost."""
    path = os.path.join(OUT_DIR, f"_verified_{oracle_name}_full.json")
    with open(path, "w") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=str)
    log.info("wrote %s", path)


def _save(oracle_name, payload):
    path = os.path.join(OUT_DIR, f"_verified_{oracle_name}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    log.info("wrote %s", path)


def run_alphagenome():
    from chorus.oracles.alphagenome import AlphaGenomeOracle
    from chorus.analysis.variant_report import build_variant_report
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.normalization import get_normalizer

    tracks = [
        # Existing 5 tracks
        "DNASE/EFO:0001187 DNase-seq/.",
        "CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA genetically modified (insertion) using CRISPR targeting H. sapiens CEBPA/.",
        "CHIP_HISTONE/EFO:0001187 Histone ChIP-seq H3K27ac/.",
        "CAGE/hCAGE EFO:0001187/-",
        "CAGE/hCAGE EFO:0001187/+",
        # New: 5 RNA-seq HepG2 tracks
        "RNA_SEQ/EFO:0001187 polyA plus RNA-seq/+",
        "RNA_SEQ/EFO:0001187 polyA plus RNA-seq/-",
        "RNA_SEQ/EFO:0001187 polyA plus RNA-seq/.",
        "RNA_SEQ/EFO:0001187 total RNA-seq/+",
        "RNA_SEQ/EFO:0001187 total RNA-seq/-",
    ]
    log.info("Loading AlphaGenome ...")
    oracle = AlphaGenomeOracle(use_environment=False, reference_fasta=GENOME)
    oracle.load_pretrained_model()
    log.info("Predicting ...")
    res = oracle.predict_variant_effect(
        genomic_region=REGION, variant_position=POS,
        alleles=[VARIANT["ref"], VARIANT["alt"]],
        assay_ids=tracks, genome=GENOME,
    )
    try:
        norm = get_normalizer(oracle_name="alphagenome")
    except Exception as exc:
        log.warning("no normalizer for alphagenome: %s", exc)
        norm = None
    ar = AnalysisRequest(
        user_prompt="Verify SORT1 expression change via RNA-seq + CAGE",
        tool_name="analyze_variant_multilayer", oracle_name="alphagenome",
        normalizer_name="chorus per-track v1" if norm else "(none)",
        tracks_requested=f"{len(tracks)} HepG2 tracks (DNase + CEBPA + H3K27ac + CAGE + RNA-seq)",
    )
    report = build_variant_report(res, oracle_name="alphagenome",
                                  gene_name=VARIANT["gene"], normalizer=norm,
                                  analysis_request=ar)
    rows = _summarise(report, "alphagenome")
    _save_full("alphagenome", report)
    _save("alphagenome", {
        "variant": VARIANT, "n_tracks_requested": len(tracks),
        "n_tracks_scored": len(rows), "tracks": rows,
    })


def run_chrombpnet():
    from chorus.oracles.chrombpnet import ChromBPNetOracle
    from chorus.analysis.variant_report import build_variant_report
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.normalization import get_normalizer

    out = {"variant": VARIANT, "runs": []}

    # Run 1: ATAC HepG2 (the existing example)
    log.info("ChromBPNet ATAC:HepG2 ...")
    o1 = ChromBPNetOracle(use_environment=False)
    o1.load_pretrained_model(assay="ATAC", cell_type="HepG2", fold=0)
    r1 = o1.predict_variant_effect(
        genomic_region=REGION, variant_position=POS,
        alleles=[VARIANT["ref"], VARIANT["alt"]],
        assay_ids=["ATAC:HepG2"], genome=GENOME,
    )
    try:
        n = get_normalizer(oracle_name="chrombpnet")
    except Exception:
        n = None
    rep1 = build_variant_report(r1, oracle_name="chrombpnet",
                                gene_name=VARIANT["gene"], normalizer=n,
                                analysis_request=AnalysisRequest(
                                    user_prompt="ATAC verify", tool_name="x",
                                    oracle_name="chrombpnet",
                                    normalizer_name="chorus per-track v1" if n else "(none)",
                                    tracks_requested="ATAC:HepG2",
                                ))
    _save_full("chrombpnet_atac", rep1)
    out["runs"].append({"assay": "ATAC:HepG2", "tracks": _summarise(rep1, "chrombpnet")})

    # Run 2: CEBPA HepG2 ChIP (new BPNet TF model BP001094)
    log.info("ChromBPNet CHIP:CEBPA:HepG2 ...")
    o2 = ChromBPNetOracle(use_environment=False)
    try:
        # The TF API may differ; try common method names
        loaded = False
        for meth, kw in [
            ("load_pretrained_TF",  dict(tf="CEBPA", cell_type="HepG2")),
            ("load_pretrained_tf",  dict(tf="CEBPA", cell_type="HepG2")),
            ("load_pretrained_model", dict(assay="ChIP-seq", tf="CEBPA", cell_type="HepG2", fold=0)),
            ("load_pretrained_model", dict(assay="CHIP", tf="CEBPA", cell_type="HepG2", fold=0)),
            ("load_pretrained_model", dict(assay="CHIP-seq", tf="CEBPA", cell_type="HepG2")),
        ]:
            if hasattr(o2, meth):
                try:
                    getattr(o2, meth)(**kw)
                    log.info("  loaded via %s(%s)", meth, kw)
                    loaded = True
                    break
                except TypeError as e:
                    log.info("  %s(%s) -> TypeError %s", meth, kw, e)
                except Exception as e:
                    log.info("  %s(%s) -> %s", meth, kw, e)
        if not loaded:
            raise RuntimeError("No load method accepted (tf=CEBPA, cell_type=HepG2)")
        r2 = o2.predict_variant_effect(
            genomic_region=REGION, variant_position=POS,
            alleles=[VARIANT["ref"], VARIANT["alt"]],
            assay_ids=["CHIP:CEBPA:HepG2"], genome=GENOME,
        )
        rep2 = build_variant_report(r2, oracle_name="chrombpnet_cebpa",
                                    gene_name=VARIANT["gene"], normalizer=n,
                                    analysis_request=AnalysisRequest(
                                        user_prompt="CEBPA verify", tool_name="x",
                                        oracle_name="chrombpnet",
                                        normalizer_name="chorus per-track v1" if n else "(none)",
                                        tracks_requested="CHIP:CEBPA:HepG2",
                                    ))
        _save_full("chrombpnet_cebpa", rep2)
        out["runs"].append({"assay": "CHIP:CEBPA:HepG2", "tracks": _summarise(rep2, "chrombpnet_cebpa")})
    except Exception as exc:
        log.warning("ChromBPNet CEBPA failed: %s", exc)
        out["runs"].append({"assay": "CHIP:CEBPA:HepG2", "error": str(exc)})

    _save("chrombpnet", out)


def run_legnet():
    from chorus.oracles.legnet import LegNetOracle
    from chorus.analysis.variant_report import build_variant_report
    from chorus.analysis.analysis_request import AnalysisRequest

    log.info("LegNet LentiMPRA:HepG2 ...")
    o = LegNetOracle(cell_type="HepG2", assay="LentiMPRA", use_environment=False)
    o.load_pretrained_model()
    r = o.predict_variant_effect(
        genomic_region=REGION, variant_position=POS,
        alleles=[VARIANT["ref"], VARIANT["alt"]],
        assay_ids=["LentiMPRA:HepG2"], genome=GENOME,
    )
    rep = build_variant_report(r, oracle_name="legnet",
                               gene_name=VARIANT["gene"], normalizer=None,
                               analysis_request=AnalysisRequest(
                                   user_prompt="LegNet verify", tool_name="x",
                                   oracle_name="legnet",
                                   normalizer_name="(none)",
                                   tracks_requested="LentiMPRA:HepG2",
                               ))
    _save_full("legnet", rep)
    _save("legnet", {"variant": VARIANT, "tracks": _summarise(rep, "legnet")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True,
                    choices=["alphagenome", "chrombpnet", "legnet"])
    args = ap.parse_args()
    {"alphagenome": run_alphagenome,
     "chrombpnet":  run_chrombpnet,
     "legnet":      run_legnet}[args.oracle]()
