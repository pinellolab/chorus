"""rs2853669 (TERT promoter) verification script.

Variant: chr5:1,295,234 A→G (hg38), in the TERT promoter.
Cell type: K562.

Layers:
- AlphaGenome  : DNase, ChIP-CEBPB-ish (we use GABPA), H3K27ac, CAGE, RNA-seq @ TERT
- ChromBPNet   : DNase K562
- ChromBPNet TF: GABPA K562 (BP000563 / ENCSR000BLO)
- LegNet       : LentiMPRA K562 (narrow window)

Usage:
    mamba run -n chorus-alphagenome python poster/_verify_rs2853669.py --oracle alphagenome
    mamba run -n chorus-chrombpnet  python poster/_verify_rs2853669.py --oracle chrombpnet_dnase
    mamba run -n chorus-chrombpnet  python poster/_verify_rs2853669.py --oracle chrombpnet_gabpa
    mamba run -n chorus-legnet      python poster/_verify_rs2853669.py --oracle legnet_narrow
"""
from __future__ import annotations
import argparse, json, logging, os, sys, numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("verify_rs2853669")

OUT = os.path.dirname(os.path.abspath(__file__))
GENOME = os.path.join(REPO_ROOT, "genomes/hg38.fa")

V = dict(chrom="chr5", position=1295234, ref="A", alt="G",
         id="rs2853669", gene="TERT")
POS    = f"{V['chrom']}:{V['position']}"
NARROW = f"{V['chrom']}:{V['position']}-{V['position']+1}"


def save_full(name, report):
    p = os.path.join(OUT, f"_verified_rs2853669_{name}_full.json")
    with open(p, "w") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=str)
    log.info("wrote %s", p)
    # Print headline numbers
    d = report.to_dict()
    for a, b in d.get("alleles", {}).items():
        for s in b.get("all_scores", [])[:30]:
            log.info("  %s : raw=%s q=%s layer=%s region=%s assay=%s",
                     a, s.get("raw_score"), s.get("quantile_score"),
                     s.get("layer"), s.get("region_label"),
                     (s.get("assay_id") or "")[:60])


def run_alphagenome():
    from chorus.oracles.alphagenome import AlphaGenomeOracle
    from chorus.analysis.variant_report import build_variant_report
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.normalization import get_normalizer

    tracks = [
        "DNASE/EFO:0002067 DNase-seq/.",
        "CHIP_HISTONE/EFO:0002067 Histone ChIP-seq H3K27ac/.",
        "CHIP_TF/EFO:0002067 TF ChIP-seq GABPA stably expressing C-terminal eGFP-tagged GABPA/.",
        "CHIP_TF/EFO:0002067 TF ChIP-seq GABPB1/.",
        "CAGE/hCAGE EFO:0002067/+",
        "CAGE/hCAGE EFO:0002067/-",
        "RNA_SEQ/EFO:0002067 polyA plus RNA-seq/+",
        "RNA_SEQ/EFO:0002067 polyA plus RNA-seq/-",
        "RNA_SEQ/EFO:0002067 total RNA-seq/+",
        "RNA_SEQ/EFO:0002067 total RNA-seq/-",
    ]

    o = AlphaGenomeOracle(use_environment=False, reference_fasta=GENOME)
    o.load_pretrained_model()
    log.info("AG: predicting at %s ref=%s alt=%s", POS, V["ref"], V["alt"])
    r = o.predict_variant_effect(
        genomic_region=NARROW, variant_position=POS,
        alleles=[V["ref"], V["alt"]],
        assay_ids=tracks, genome=GENOME,
    )
    try:
        n = get_normalizer(oracle_name="alphagenome")
    except Exception:
        n = None
    rep = build_variant_report(r, oracle_name="alphagenome",
                               gene_name=V["gene"], normalizer=n,
                               analysis_request=AnalysisRequest(
                                   user_prompt="rs2853669 AG verify",
                                   tool_name="x", oracle_name="alphagenome",
                                   normalizer_name="chorus per-track v1" if n else "(none)",
                                   tracks_requested=f"{len(tracks)} K562 tracks"))
    save_full("alphagenome", rep)


def _max_window():
    from chorus.oracles.chrombpnet import ChromBPNetOracle
    from chorus.oracles.legnet import LegNetOracle
    from chorus.oracles.alphagenome import AlphaGenomeOracle
    return max(ChromBPNetOracle().output_size,
               LegNetOracle().sequence_length,
               AlphaGenomeOracle().output_size)


def run_chrombpnet_dnase():
    from chorus.oracles.chrombpnet import ChromBPNetOracle
    from chorus.analysis.variant_report import build_variant_report
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.normalization import get_normalizer

    half = _max_window() // 2
    wide = f"{V['chrom']}:{V['position']-half}-{V['position']+half}"

    o = ChromBPNetOracle(use_environment=False)
    o.load_pretrained_model(assay="DNASE", cell_type="K562", fold=0)
    r = o.predict_variant_effect(genomic_region=wide, variant_position=POS,
                                 alleles=[V["ref"], V["alt"]],
                                 assay_ids=["DNASE:K562"], genome=GENOME)
    try:
        n = get_normalizer(oracle_name="chrombpnet")
    except Exception:
        n = None
    rep = build_variant_report(r, oracle_name="chrombpnet",
                               gene_name=V["gene"], normalizer=n,
                               analysis_request=AnalysisRequest(
                                   user_prompt="rs2853669 ChromBPNet DNase K562",
                                   tool_name="x", oracle_name="chrombpnet",
                                   normalizer_name="chorus per-track v1" if n else "(none)",
                                   tracks_requested="DNASE:K562"))
    save_full("chrombpnet_dnase", rep)


def run_chrombpnet_gabpa():
    """Score with the BPNet TF head for GABPA in K562 (BP000563)."""
    from chorus.oracles.chrombpnet import ChromBPNetOracle

    o = ChromBPNetOracle(use_environment=False)
    o.load_pretrained_model(assay="CHIP", TF="GABPA", cell_type="K562", fold=0)
    log.info("BPNet GABPA K562 loaded")
    r = o.predict_variant_effect(
        genomic_region=NARROW, variant_position=POS,
        alleles=[V["ref"], V["alt"]],
        assay_ids=["CHIP:GABPA:K562"], genome=GENOME,
    )
    # build_variant_report crashes on BPNet TF metadata; extract directly.
    preds = r.get("predictions", {})
    eff   = r.get("effect_sizes", {}).get("alt_1", {})
    out = {"variant": V, "effect_sizes": {}, "predictions": {}}
    for tname, arr in eff.items():
        a = np.asarray(arr)
        out["effect_sizes"][tname] = {
            "sum": float(a.sum()), "max": float(a.max()), "min": float(a.min()), "n": int(a.size)
        }
        log.info("  effect %s : sum=%+.4f max=%+.4f", tname, float(a.sum()), float(a.max()))

    def to_array(op):
        tracks = getattr(op, "tracks", None) or {}
        mats = []
        for tname, t in tracks.items():
            v = getattr(t, "values", t)
            if callable(v): v = v()
            arr = np.asarray(v)
            mats.append(arr)
            log.info("    %s pred shape=%s sum=%.4g", tname, arr.shape, float(arr.sum()))
        return np.stack(mats) if mats else None

    ref_arr = to_array(preds.get("reference"))
    alt_arr = to_array(preds.get("alt_1"))
    if isinstance(ref_arr, np.ndarray) and isinstance(alt_arr, np.ndarray):
        ref_sum = float(ref_arr.sum()); alt_sum = float(alt_arr.sum())
        log2fc = float(np.log2(alt_sum / ref_sum)) if ref_sum > 0 else None
        out["log2fc_sum"] = log2fc
        out["ref_sum"]    = ref_sum
        out["alt_sum"]    = alt_sum
        log.info("GABPA K562  ref_sum=%.4g  alt_sum=%.4g  log2FC=%+.4f", ref_sum, alt_sum, log2fc)

    p = os.path.join(OUT, "_verified_rs2853669_chrombpnet_gabpa.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    log.info("wrote %s", p)


def run_legnet_narrow():
    """LegNet at the SNP site (narrow window) in K562."""
    from chorus.oracles.legnet import LegNetOracle
    from chorus.analysis.variant_report import build_variant_report
    from chorus.analysis.analysis_request import AnalysisRequest

    o = LegNetOracle(cell_type="K562", assay="LentiMPRA", use_environment=False)
    o.load_pretrained_model()
    log.info("LegNet K562 LentiMPRA loaded, seq_len=%s", o.sequence_length)
    r = o.predict_variant_effect(
        genomic_region=NARROW, variant_position=POS,
        alleles=[V["ref"], V["alt"]],
        assay_ids=["LentiMPRA:K562"], genome=GENOME,
    )
    rep = build_variant_report(r, oracle_name="legnet",
                               gene_name=V["gene"], normalizer=None,
                               analysis_request=AnalysisRequest(
                                   user_prompt="rs2853669 LegNet narrow K562",
                                   tool_name="x", oracle_name="legnet",
                                   normalizer_name="(none)",
                                   tracks_requested="LentiMPRA:K562"))
    save_full("legnet_narrow", rep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True,
                    choices=["alphagenome", "chrombpnet_dnase",
                             "chrombpnet_gabpa", "legnet_narrow"])
    args = ap.parse_args()
    {
        "alphagenome":      run_alphagenome,
        "chrombpnet_dnase": run_chrombpnet_dnase,
        "chrombpnet_gabpa": run_chrombpnet_gabpa,
        "legnet_narrow":    run_legnet_narrow,
    }[args.oracle]()
