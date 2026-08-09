"""Stamp `build_config` (schema_version 4) from ARTEFACTS, never from a build log.

The previous stamper scraped the builder's stdout with a regex. That is how AlphaGenome
came to carry a `build_config` whose stamped claim (`effect_region_strata`) and scraped
measurement (`effect_region_set_as_logged`) contradicted each other, and it is why the
same regex crashed on the 2026-08-06 logs: the message it parses had changed. It also
only knew three of the eight oracles.

Everything here is read from something checkable instead:

  * the reference position sets   -> which populations, their strata, their sha256
                                     (the ACTIVITY population per oracle, derived -- see
                                     ACTIVITY_POPULATIONS; the artefact carries one region
                                     set and only three builders sample all of it)
  * the NPZ's own arrays          -> track count, per-layer offered/retained/tail_k,
                                     signed fraction, which layers exist, and a ceiling
                                     on the activity population they can have come from
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

# The 5,000-position DHS-summit stratum the three accessibility-family builders add on top
# of the shared region mixture. It is NOT one of the reference artefact's sets, so it is
# reproduced here from the same sampler, file and seed the builders call.
DHS_STRATUM = dict(
    n=5_000, seed=567,
    sampler="chorus.utils.annotations.sample_dhs_positions",
    source="annotations/dhs_vocabulary_hg38.txt.gz",
)

# Which ACTIVITY (summary/perbin) population each oracle's builder actually samples.
#
# No builder reads reference_sets/ -- each resamples from the same seeds -- so the
# population is a property of the builder's own sampling block, and those blocks differ.
# Until 2026-08-09 this file stamped `regions_genome_dominated` (31,500 positions) into all
# eight unconditionally. That was false for five of them, and arithmetically impossible for
# three from the artefacts alone: chrombpnet and cherimoya offered 34,004 summary samples
# per track against a claimed 31,500-position set, epinformerseq 34,002, while sei and
# legnet offered 29,004/29,002 -- a strict SUBSET, missing the gene_body stratum.
#
# Each entry is therefore a DERIVATION of the one region set the artefact does carry, so
# the population still has a content hash a reader can recompute from shipped inputs:
#   drop -- strata of `regions_genome_dominated` this builder never samples
#   add  -- strata it samples that the artefact does not carry
# Verified by replaying each builder's sampling block against the artefact: sei/legnet's
# 29,500 positions hash ddbc4b246ab3..., bit-identical to the artefact minus gene_body,
# and the accessibility trio's 34,500 hash ec3070d6a361... .
_GENOME_DOMINATED = dict(name="regions_genome_dominated", drop=(), add={})
_NO_GENE_BODY = dict(name="regions_genome_dominated_minus_gene_body",
                     drop=("gene_body",), add={})
_NO_GENE_BODY_PLUS_DHS = dict(name="regions_genome_dominated_minus_gene_body_plus_dhs",
                              drop=("gene_body",), add={"dhs": DHS_STRATUM})
ACTIVITY_POPULATIONS = {
    # build_backgrounds_{enformer,borzoi}.py and _alphagenome.py: random + cCRE + TSS +
    # gene-body midpoints, which is what the artefact's set was built to replicate.
    "alphagenome":   _GENOME_DOMINATED,
    "borzoi":        _GENOME_DOMINATED,
    "enformer":      _GENOME_DOMINATED,
    # build_backgrounds_{sei,legnet}.py: the same three streams, no gene-body block.
    "sei":           _NO_GENE_BODY,
    "legnet":        _NO_GENE_BODY,
    # build_backgrounds_{chrombpnet,cherimoya}.py and _epinformerseq_v2_percell.py: those
    # three streams plus DHS summits, no gene-body block.
    "chrombpnet":    _NO_GENE_BODY_PLUS_DHS,
    "cherimoya":     _NO_GENE_BODY_PLUS_DHS,
    "epinformerseq": _NO_GENE_BODY_PLUS_DHS,
}

# Rows offered per activity POSITION, per track: one window statistic for `summary`, and
# PERBIN_BINS_PER_POSITION random bins for `perbin` (32 in every builder that writes one).
OBS_PER_POSITION = {"summary": 1, "perbin": 32}

# ...times a fan-out, for the three builders that emit more than one row per position.
# Everything not listed here is 1, which makes the ceiling below TIGHT for it.
FAN_OUT = {
    # ChromBPNet's profile head is (L, n_strands) and BOTH strands are scored
    # (`for strand in range(prof.shape[-1])`), so 744 of its 753 tracks see 2 rows.
    ("chrombpnet", None): {"summary": 2, "perbin": 2},
    # AlphaGenome and Borzoi emit one RNA summary row per (GENE, track), matching a query
    # that emits an RNA row per gene near the variant (#144 inst. 3). The multiplier is
    # the mean number of genes with a TSS in the input window -- measured 10.35 at 1 Mb
    # and 2.42 at 524 kb, so these bounds carry ~55%/65% headroom. `perbin` stays POOLED
    # across genes and capped at PERBIN_BINS_PER_POSITION, so it does not fan out.
    ("alphagenome", "gene_expression"): {"summary": 16, "perbin": 1},
    ("borzoi", "gene_expression"): {"summary": 4, "perbin": 1},
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


_DHS_CACHE: dict = {}


def _dhs_rows(spec: dict) -> list:
    """The added DHS stratum, from the same sampler/file/seed the builders call.

    Cached: it reads a 90 MB vocabulary and costs ~6.5 s, and three oracles want it.
    """
    key = (spec["n"], spec["seed"])
    if key not in _DHS_CACHE:
        from chorus.utils.annotations import sample_dhs_positions
        src = REPO / spec["source"]
        if not src.exists():
            raise FileNotFoundError(src)
        _DHS_CACHE[key] = [(str(c), int(p), "dhs") for c, p in
                           sample_dhs_positions(spec["n"], dhs_path=str(src),
                                                seed=spec["seed"])]
    return _DHS_CACHE[key]


def activity_population(oracle: str, brps, ref_arrays: dict, ref_prov: dict) -> dict:
    """The activity population this oracle's builder sampled, content-hashed.

    Returns the `activity_*` half of `reference_sets`. The hash is recomputed over the
    derived (chrom, pos, stratum) tuples with the reference generator's own `_sha256_of`,
    so it is comparable with the artefact's hashes and reproducible by any reader holding
    the same inputs.

    The derivation is checked against itself first: recomputing the hash of the FULL
    artefact set must reproduce the sha256 the artefact records. If that fails, the
    hashing convention has drifted and no derived hash from it means anything, so nothing
    is stamped rather than something unverifiable.
    """
    spec = ACTIVITY_POPULATIONS.get(oracle)
    if spec is None:
        raise ValueError(f"{oracle} has no entry in ACTIVITY_POPULATIONS; add one "
                         f"describing what its builder samples")

    base = "regions_genome_dominated"
    rows = [(str(c), int(p), str(s)) for c, p, s in ref_arrays[base]]
    if brps._sha256_of(rows) != ref_prov["sets"][base]["sha256"]:
        raise ValueError(
            f"recomputing {base} from the artefact's own rows does not reproduce its "
            f"recorded sha256 -- the hashing convention moved, so no derived population "
            f"hash is trustworthy")

    rows = [r for r in rows if r[2] not in spec["drop"]]
    unavailable = None
    for name, add in spec["add"].items():
        try:
            rows += _dhs_rows(add)
        except FileNotFoundError as exc:
            # State nothing rather than state it falsely: without the source we cannot
            # reproduce the stratum, so the mixture has no hash we are entitled to claim.
            unavailable = (f"{name} stratum needs {add['source']}, absent here ({exc}); "
                           f"the composition below is from the builder, unhashed")

    strata: dict = {}
    for _c, _p, s in rows:
        strata[s] = strata.get(s, 0) + 1
    if unavailable:
        for name, add in spec["add"].items():
            strata[name] = add["n"]

    out = {
        "activity_set": spec["name"],
        "activity_sha256": None if unavailable else brps._sha256_of(rows),
        "activity_strata": strata,
        "activity_derivation": {
            "from": base,
            "from_sha256": ref_prov["sets"][base]["sha256"],
            "drop_strata": list(spec["drop"]),
            "add_strata": {k: dict(v) for k, v in spec["add"].items()},
            "hash": "sorted (chrom, pos, stratum) tuples, "
                    "scripts/build_reference_position_sets.py:_sha256_of",
        },
    }
    if unavailable:
        out["activity_sha256_unavailable"] = unavailable
    return out


def check_counts_fit_the_population(oracle: str, payload: dict, n_positions: int) -> None:
    """A reservoir cannot be offered more samples than the build had positions to offer.

    `max(counts) <= n_positions * rows_per_position` is the one inequality that catches a
    misdeclared activity population from the artefact alone, and it is not tautological:
    both sides come from different files. It is what the false `regions_genome_dominated`
    stamp tripped on -- chrombpnet's 68,008 summary samples against 31,500 * 2 = 63,000,
    cherimoya's 34,004 and epinformerseq's 34,002 against 31,500 * 1.

    One-sided on purpose. A build that sampled FEWER positions than declared satisfies it
    (sei and legnet's 29,004/29,002 sat under the claimed 31,500 for two months); that side
    is pinned by ACTIVITY_POPULATIONS naming what each builder samples, not by arithmetic.
    """
    lay = np.asarray(payload["layers_per_row"]) if "layers_per_row" in payload else None
    for stat, per_position in OBS_PER_POSITION.items():
        ck = f"{stat}_counts"
        if ck not in payload:
            continue
        counts = np.asarray(payload[ck])
        if lay is not None and len(lay) == len(counts):
            groups = [(L, counts[lay == L]) for L in sorted({str(x) for x in lay.tolist()})]
        else:
            groups = [(None, counts)]
        for layer, c in groups:
            if c.size == 0:
                continue
            # layer-specific entry, else the oracle-wide one, else no fan-out. The fallback
            # matters: chrombpnet ships no `layers_per_row` today, and gaining one must not
            # silently drop its two-strand multiplier.
            fan = (FAN_OUT.get((oracle, layer))
                   or FAN_OUT.get((oracle, None)) or {}).get(stat, 1)
            ceiling = n_positions * per_position * fan
            if int(c.max()) > ceiling:
                raise ValueError(
                    f"{oracle}: {stat}_counts.max()={int(c.max()):,} for "
                    f"{layer or 'all tracks'} exceeds {n_positions:,} activity positions "
                    f"x {per_position} per position x {fan} fan-out = {ceiling:,}. The "
                    f"stamped activity population cannot be the one this build sampled")


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
        ref_arrays = {k: d[k] for k in d.files if k != "provenance"}

    activity = activity_population(oracle, brps, ref_arrays, ref_prov)
    check_counts_fit_the_population(oracle, payload,
                                    sum(activity["activity_strata"].values()))

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

    # Carry forward the checkpoint identity the BUILDER recorded. The stamp replaces
    # build_config wholesale, so anything the builder knew and the stamper does not is
    # lost -- and "which checkpoints produced this null" is exactly what provenance is
    # for. Cherimoya's first stamp after the 5-fold ensemble swap dropped
    # fold="ensemble", leaving an artefact that could not say whether it was built from
    # one fold or five, which is a 33% difference in the statistic it ranks against.
    prior: dict = {}
    if "build_config" in payload:
        try:
            prior = json.loads(str(payload["build_config"][0]))
        except Exception:
            prior = {}
    carried = {k: prior[k] for k in ("fold", "folds", "ensemble", "model_variant",
                                     "checkpoint_template")
               if k in prior and prior[k] is not None}

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
        # Builder-recorded checkpoint identity, preserved across the stamp.
        **carried,
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
            **activity,
            "seeds": ref_prov["seeds"],
        },
        "notes": [
            "Effect and activity nulls are DIFFERENT reference classes and must not be "
            "unified; see docs/BACKGROUND_NULL_PROTOCOL.md section 1.",
            "The activity population is PER BUILDER: only enformer, borzoi and "
            "alphagenome sample the whole of regions_genome_dominated. The rest are "
            "recorded as derivations of it (reference_sets.activity_derivation), because "
            "no builder reads the reference artefact -- each resamples from the seeds.",
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
        try:
            cfg = build_config(o, payload)
        except ValueError as exc:
            # Refuse this one and carry on: a stamp that contradicts the file is worse
            # than no stamp, but it is no reason to leave the other seven unstamped.
            print(f"  {o:14s} NOT STAMPED -- {exc}")
            rc = 1
            continue
        rs, n_layers = cfg["reference_sets"], len(cfg["statistics_per_layer"])
        print(f"  {o:14s} schema {cfg['schema_version']}  {cfg['n_tracks']:5d} tracks  "
              f"{len(cfg['sampling'])} layers sampled, {n_layers} statistics, "
              f"family={rs['effect_family']}, activity={rs['activity_set']} "
              f"({sum(rs['activity_strata'].values()):,} positions, "
              f"{(rs['activity_sha256'] or 'UNHASHED')[:12]})")
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
