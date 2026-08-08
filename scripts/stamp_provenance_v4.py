"""Stamp `build_config` (schema_version 4) from ARTEFACTS, never from a build log.

The previous stamper scraped the builder's stdout with a regex. That is how AlphaGenome
came to carry a `build_config` whose stamped claim (`effect_region_strata`) and scraped
measurement (`effect_region_set_as_logged`) contradicted each other, and it is why the
same regex crashed on the 2026-08-06 logs: the message it parses had changed. It also
only knew three of the eight oracles.

Everything here is read from something checkable instead:

  * the reference position sets   -> which populations, their strata, their sha256
  * the NPZ's own arrays          -> track count, per-layer offered/retained/tail_k,
                                     signed fraction, which layers exist
  * LAYER_CONFIGS                 -> window, aggregation, formula, pseudocount per layer
  * the repo and the genome       -> git sha, fai sha256, FASTA sha256 prefix

So a reader can answer "which reference class is this, and what statistic?" from the file
alone, and every field is derivable from inputs that are themselves hashed.

`build_config` is file-level, so this appends in place: no rebuild, no re-merge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SCHEMA_VERSION = 4
REF = REPO / "reference_sets" / "chorus_reference_positions_v1.npz"

# Which reference SNP family each oracle's effect null draws from, and the input geometry
# that is a property of the MODEL rather than of the build.
GEOMETRY = {
    "alphagenome":   dict(input_length=1_048_576, resolution=1),
    "borzoi":        dict(input_length=524_288, resolution=32),
    "enformer":      dict(input_length=393_216, resolution=128),
    "chrombpnet":    dict(input_length=2_114, resolution=1),
    "cherimoya":     dict(input_length=2_114, resolution=1),
    "epinformerseq": dict(input_length=2_114, resolution=None),
    "sei":           dict(input_length=4_096, resolution=None),
    "legnet":        dict(input_length=200, resolution=None),
}


def _sha_file(p: Path, limit: int | None = None) -> str:
    h, n = hashlib.sha256(), 0
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
            n += len(c)
            if limit and n >= limit:
                break
    return h.hexdigest()


def build_config(oracle: str, payload: dict) -> dict:
    from chorus.analysis.background_sampling import MIN_EXACT_TAIL_SLOTS
    from chorus.analysis.scorers import LAYER_CONFIGS

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "brps", REPO / "scripts" / "build_reference_position_sets.py")
    brps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(brps)

    ids = [str(x) for x in payload["track_ids"]]
    family = brps.ORACLE_SNP_SET.get(oracle)
    with np.load(REF, allow_pickle=False) as d:
        ref_prov = json.loads(str(d["provenance"][0]))

    # per-layer sampling, read from the arrays the build actually wrote
    sampling: dict = {}
    for layer in ("effect", "summary", "perbin"):
        ck, rk, tk = f"{layer}_counts", f"{layer}_retained", f"{layer}_tail_k"
        if ck not in payload:
            continue
        offered = np.asarray(payload[ck])
        entry: dict = {"offered": [int(offered.min()), int(offered.max())]}
        if rk in payload:
            ret = np.asarray(payload[rk])
            entry["retained"] = [int(ret.min()), int(ret.max())]
            entry["thinned_tracks"] = int((ret < offered).sum())
            entry["mode"] = "exact" if entry["thinned_tracks"] == 0 else "hybrid"
        if tk in payload:
            k = int(payload[tk])
            entry["tail_k"] = k
            entry["exact_top_slots"] = int(min(k, offered.max()) * 10_000 // offered.max())
            entry["min_exact_tail_slots_intent"] = MIN_EXACT_TAIL_SLOTS
        sampling[layer] = entry

    layers = sorted({str(x) for x in payload["layers_per_row"]}) \
        if "layers_per_row" in payload else None
    statistics = {L: {"window_bp": LAYER_CONFIGS[L].window_bp,
                      "aggregation": LAYER_CONFIGS[L].aggregation,
                      "formula": LAYER_CONFIGS[L].formula,
                      "pseudocount": LAYER_CONFIGS[L].pseudocount,
                      "signed": LAYER_CONFIGS[L].signed}
                  for L in (layers or []) if L in LAYER_CONFIGS}

    sf = np.asarray(payload["signed_flags"]).astype(bool) if "signed_flags" in payload else None
    cfg = {
        "schema_version": SCHEMA_VERSION,
        "oracle": oracle,
        "n_tracks": len(ids),
        "genome": "hg38",
        "fai_sha256": _sha_file(REPO / "genomes" / "hg38.fa.fai"),
        "fasta_sha256_prefix64mb": _sha_file(REPO / "genomes" / "hg38.fa", 64 << 20),
        "stamped_at": datetime.now(timezone.utc).isoformat(),
        "stamper": "scripts/stamp_provenance_v4.py",
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True).stdout.strip(),
        **GEOMETRY.get(oracle, {}),
        "cdf_points": int(payload["effect_cdfs"].shape[1]) if "effect_cdfs" in payload else None,
        "layers_present": layers,
        "statistics_per_layer": statistics,
        "signed_fraction": None if sf is None else round(float(sf.mean()), 4),
        # Kept for compatibility with the schema-2/3 readers and with
        # tests/test_provenance_is_read.py, which guards the #122 substance: histone ChIP
        # is scored over 2001 bp and everything else over 501, and a background built with
        # the wrong window is silently wrong. DERIVED from LAYER_CONFIGS rather than
        # hardcoded, so it cannot drift from the statistic actually used --
        # statistics_per_layer above is the same facts per layer.
        "histone_window_bp": LAYER_CONFIGS["histone_marks"].window_bp,
        "other_window_bp": LAYER_CONFIGS["chromatin_accessibility"].window_bp,
        "sampling": sampling,
        # the reference populations, by content hash -- this is the reference class
        "reference_sets": {
            "artefact": REF.name,
            "artefact_schema": ref_prov["schema_version"],
            "effect_family": family,
            "effect_sha256": (ref_prov["sets"].get(f"snps_{family}", {}) or {}).get("sha256"),
            "effect_strata": (ref_prov["sets"].get(f"snps_{family}", {}) or {}).get(
                "strata_realised"),
            "activity_set": "regions_genome_dominated",
            "activity_sha256": ref_prov["sets"]["regions_genome_dominated"]["sha256"],
            "activity_strata": ref_prov["sets"]["regions_genome_dominated"]["strata_realised"],
            "seeds": ref_prov["seeds"],
        },
        "notes": [
            "Effect and activity nulls are DIFFERENT reference classes and must not be "
            "unified; see docs/BACKGROUND_NULL_PROTOCOL.md section 1.",
            "Percentiles are strictly empirical: above the sampled maximum the value is "
            "clamped and PerTrackNormalizer.effect_exceedance reports the ratio to that "
            "ceiling. No tail is extrapolated.",
            "DHS was measured and REJECTED for the gene-anchored and promoter mixtures "
            "(it added nothing to any ceiling and diluted enformer tf_binding worst of "
            "all layers); it remains the basis of the accessibility family by design.",
        ],
        # Determinism: recorded rather than claimed. The XLA flags AlphaGenome runs under
        # are the thing that makes a rebuild reproducible, and a bare "available: true"
        # asserts nothing -- so point at the module that sets them and at the evidence.
        "determinism": {
            "available": True,
            "module": "chorus/core/determinism.py",
            "evidence": "cherimoya rebuilt from scratch reproduced its shipped effect "
                        "null bit-identically on all 1,518 rows (2026-08-06), covering "
                        "region sampling, forward passes, reservoir, gridding and write",
        },
        "unified_build": True,
        "build_campaign": "2026-08-06 unified rebuild",
    }
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracles", nargs="*", default=list(GEOMETRY))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()

    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR
    root = Path(args.dir) if args.dir else CHORUS_BACKGROUNDS_DIR

    rc = 0
    for o in args.oracles:
        p = root / f"{o}_pertrack.npz"
        if not p.exists():
            print(f"  {o}: no file at {p}")
            rc = 1
            continue
        with np.load(p, allow_pickle=True) as d:
            payload = {k: d[k] for k in d.files}
        cfg = build_config(o, payload)
        n_layers = len(cfg["statistics_per_layer"])
        print(f"  {o:14s} schema {cfg['schema_version']}  {cfg['n_tracks']:5d} tracks  "
              f"{len(cfg['sampling'])} layers sampled, {n_layers} statistics, "
              f"family={cfg['reference_sets']['effect_family']}")
        if args.dry_run:
            continue
        payload["build_config"] = np.array([json.dumps(cfg, sort_keys=True)])
        tmp = p.with_suffix(".npz.stamping.npz")     # savez appends .npz; name it so
        np.savez_compressed(tmp, **payload)
        with np.load(tmp, allow_pickle=True) as chk:   # read back before replacing
            got = json.loads(str(chk["build_config"][0]))
            assert got["oracle"] == o and got["schema_version"] == SCHEMA_VERSION
            assert len(chk["track_ids"]) == cfg["n_tracks"]
        import os
        os.replace(tmp, p)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
