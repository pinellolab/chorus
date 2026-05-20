"""Score rs12740374 with ChromBPNet's CEBPA HepG2 ChIP-seq TF model (BP001094)."""
from __future__ import annotations

import json
import logging
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("verify_cebpa")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
GENOME = os.path.join(REPO_ROOT, "genomes/hg38.fa")

VARIANT = dict(
    chrom="chr1", position=109274968, ref="G", alt="T",
    id="rs12740374", gene="SORT1",
)
REGION = f"{VARIANT['chrom']}:{VARIANT['position']}-{VARIANT['position']+1}"
POS    = f"{VARIANT['chrom']}:{VARIANT['position']}"


def main():
    from chorus.oracles.chrombpnet import ChromBPNetOracle
    from chorus.analysis.variant_report import build_variant_report
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.normalization import get_normalizer

    log.info("Loading ChromBPNet CHIP CEBPA HepG2 ...")
    o = ChromBPNetOracle(use_environment=False)
    o.load_pretrained_model(assay="CHIP", TF="CEBPA", cell_type="HepG2", fold=0)
    log.info("Predicting rs12740374 ...")
    r = o.predict_variant_effect(
        genomic_region=REGION, variant_position=POS,
        alleles=[VARIANT["ref"], VARIANT["alt"]],
        assay_ids=["CHIP:CEBPA:HepG2"], genome=GENOME,
    )
    import numpy as np
    log.info("dict keys: %s", list(r.keys()))
    rows = {"raw_keys": list(r.keys())}
    # Pull effect_sizes (per-bin alt - ref deltas) and predictions
    effect = r.get("effect_sizes", {}).get("alt_1", {})
    log.info("effect tracks: %s", list(effect.keys()))
    eff_sums = {}
    for tname, arr in effect.items():
        a = np.asarray(arr)
        eff_sums[tname] = {"sum": float(a.sum()), "max": float(a.max()),
                           "min": float(a.min()), "n": int(a.size)}
        log.info("  %s : sum=%+.4f  max=%+.4f  min=%+.4f", tname, *map(eff_sums[tname].get, ("sum","max","min")))
    rows["effect_sizes"] = eff_sums
    # Pull OraclePrediction arrays to compute log2FC from totals
    preds = r.get("predictions", {})
    log.info("preds keys: %s; attrs of ref: %s",
             list(preds.keys()),
             [a for a in dir(preds.get("reference") or object()) if not a.startswith('_')][:30])
    def to_array(op):
        if op is None: return None
        # OraclePrediction.tracks is a dict[name -> Track]; each Track has .values (ndarray)
        tracks = getattr(op, "tracks", None)
        if isinstance(tracks, dict):
            mats = []
            for tname, t in tracks.items():
                v = getattr(t, "values", t)
                if callable(v): v = v()
                a = np.asarray(v)
                log.info("    track %s -> shape %s sum=%.4g max=%.4g", tname, a.shape, float(a.sum()), float(a.max()))
                mats.append(a)
            return np.stack(mats) if mats else None
        return None
    ref_arr = to_array(preds.get("reference"))
    alt_arr = to_array(preds.get("alt_1"))
    log.info("ref type: %s, alt type: %s",
             type(ref_arr).__name__,
             type(alt_arr).__name__)
    if isinstance(ref_arr, np.ndarray) and isinstance(alt_arr, np.ndarray):
        ref_sum = float(ref_arr.sum()); alt_sum = float(alt_arr.sum())
        log2fc = float(np.log2(alt_sum / ref_sum)) if ref_sum > 0 else None
        rows["log2fc_sum"] = log2fc
        rows["ref_sum"] = ref_sum
        rows["alt_sum"] = alt_sum
        log.info("CEBPA HepG2  ref_sum=%.4g  alt_sum=%.4g  log2FC=%+.4f", ref_sum, alt_sum, log2fc)
    out = {"variant": VARIANT, "tracks": rows}
    path = os.path.join(OUT_DIR, "_verified_chrombpnet_cebpa.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    log.info("wrote %s", path)
    # `rows` is a flat dict (log2fc_sum + ref_sum + alt_sum + per-track
    # effect_sizes dict), not a list. Print the headline numbers + each
    # per-track summary directly instead of treating dict keys as rows.
    if rows.get("log2fc_sum") is not None:
        log.info("  CEBPA log2fc_sum=%+.4f  ref_sum=%.4g  alt_sum=%.4g",
                 rows["log2fc_sum"], rows["ref_sum"], rows["alt_sum"])
    for tname, tdict in (rows.get("effect_sizes") or {}).items():
        if isinstance(tdict, dict):
            log.info("  %s  sum=%.4g  max=%.4g  min=%.4g  n=%d",
                     tname, tdict.get("sum", float("nan")),
                     tdict.get("max", float("nan")),
                     tdict.get("min", float("nan")),
                     tdict.get("n", 0))


if __name__ == "__main__":
    main()
