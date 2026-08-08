"""Measure how far real causal variants exceed their tracks' null maxima.

This decides one open question with evidence rather than argument: **does any layer
need its effect null rebuilt against a motif-anchored reference set, or is exposing
the exceedance ratio (read-side) sufficient?**

The reasoning it replaces. ChIP percentiles pin at 1.0 on variants that create a
complete transcription-factor motif, because a null drawn from *random* positions --
DHS-anchored, cCRE-anchored or gene-anchored alike -- contains few single-base
changes that make or break a specific factor's full motif. That is true of the
DHS-anchored null ChromBPNet already uses for its 744 ChIP tracks: at rs12740374 it
pins ``CHIP:HepG2:CEBPA:+`` (raw +1.865 vs null max 1.682). So "adopt Cherimoya's
background for ChIP" cannot be the fix -- it is already in use, and it saturates.

The decision rule, fixed before looking at the numbers:

* exceedances clustered just past 1.0 (say < 1.5x) mean the null's ceiling is in
  roughly the right place and only *resolution* was missing -- the read-side ratio
  is then the whole fix, and rebuilding would change what a ChIP percentile means
  ("stronger than X% of motif-altering variants" rather than "...of random
  regulatory variants") for no measurement gain;
* a substantial mass at >3x means that layer's null is genuinely mis-scaled and a
  motif-anchored *union* is warranted (union at 2N is provably never worse than
  either component, since max(union) = max(max_a, max_b)).

Why a motif-anchored null cannot simply be built from the model's own sensitivity:
selecting positions where the model responds strongly is selection on the outcome,
and the result would be an upper envelope of model behaviour, not a null
distribution. It needs a PWM defined independently of the model -- which is why this
measurement runs first.

Coordinates are validated against hg38 before scoring; 5 of an initial 12 recalled
variants were rejected there, which is the point of doing it before spending GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT = Path("/tmp/exceedance")

# hg38, 1-based, each with a published regulatory mechanism. ``gene`` is the gene
# the locus is usually attributed to and only steers gene-anchored layers.
PANEL = [
    {"id": "rs12740374", "chrom": "chr1", "pos": 109274968, "ref": "G", "alt": "T",
     "gene": "SORT1", "note": "creates C/EBP motif"},
    {"id": "rs4988235", "chrom": "chr2", "pos": 135851076, "ref": "G", "alt": "A",
     "gene": "MCM6", "note": "lactase persistence"},
    {"id": "rs1421085", "chrom": "chr16", "pos": 53767042, "ref": "T", "alt": "C",
     "gene": "FTO", "note": "disrupts ARID5B motif"},
    {"id": "rs6801957", "chrom": "chr3", "pos": 38767315, "ref": "T", "alt": "C",
     "gene": "SCN10A", "note": "TBX5 motif"},
    {"id": "TERT-124", "chrom": "chr5", "pos": 1295113, "ref": "G", "alt": "A",
     "gene": "TERT", "note": "creates ETS/GABP motif"},
    {"id": "rs2168101", "chrom": "chr11", "pos": 8252911, "ref": "G", "alt": "T",
     "gene": "LMO1", "note": "GATA motif"},
    {"id": "rs17293632", "chrom": "chr15", "pos": 67163292, "ref": "C", "alt": "T",
     "gene": "SMAD3", "note": "AP-1 motif"},
]


def verify_panel() -> list[dict]:
    """Reject any variant whose REF does not match hg38. Runs before any GPU work."""
    import pyfaidx

    from chorus.core.globals import CHORUS_DATA_DIR

    fa_path = CHORUS_DATA_DIR / "genomes" / "hg38.fa"
    fa = pyfaidx.Fasta(str(fa_path), as_raw=True, sequence_always_upper=True)
    ok = []
    for v in PANEL:
        got = fa[v["chrom"]][v["pos"] - 1: v["pos"] - 1 + len(v["ref"])]
        if got == v["ref"]:
            ok.append(v)
        else:
            print(f"  REJECT {v['id']}: expected {v['ref']} got {got!r}", flush=True)
    return ok


def _assay_ids(oracle_name: str, oracle=None) -> list[str]:
    """Every track that HAS a background row, read from the shipped artefact.

    Track enumeration is not uniform across oracles -- Enformer/Borzoi/Cherimoya
    expose ``list_tracks``, Sei only a private ``_get_all_assay_ids``, and the MCP
    server resolves it with a per-oracle switch. The background's own ``track_ids``
    sidesteps all of that and is the *right* population regardless: a track without
    a background row has no null to be measured against.

    ChromBPNet is capped because it loads a separate model per track (753 would be
    753 loads per variant, where the sequence-to-profile models return every track
    from a single pass). The cap is printed, never silent.
    """
    import numpy as np

    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR

    path = CHORUS_BACKGROUNDS_DIR / f"{oracle_name}_pertrack.npz"
    with np.load(path, allow_pickle=True) as d:
        ids = [str(x) for x in d["track_ids"]]

    if oracle_name == "legnet":
        # A LegNet instance refuses any assay but the one it was constructed with
        # ("Instantiated LegNet oracle can only predict for assay LentiMPRA:HepG2").
        # All three cell types are now discoverable via list_cell_types(), but
        # predicting the other two needs a separate instance, so scope to this one.
        cell = getattr(oracle, "cell_type", None)
        keep = [x for x in ids if cell and cell in x]
        print(f"  legnet: scoring {len(keep)} of {len(ids)} tracks (instance is "
              f"cell_type={cell!r}; other cell types need their own instance)",
              flush=True)
        return keep

    if oracle_name != "chrombpnet":
        return ids

    acc = [t for t in ids if t.startswith(("ATAC:", "DNASE:"))]
    chip = [t for t in ids if t.startswith("CHIP:")]
    pref = [t for t in chip if ":HepG2:" in t or ":K562:" in t]
    rest = [t for t in chip if t not in set(pref)]
    take = acc + pref[:80] + rest[:20]
    print(f"  chrombpnet CAP: scoring {len(take)} of {len(ids)} tracks "
          f"({len(acc)} accessibility + {len(pref[:80])}/{len(pref)} HepG2-K562 CHIP "
          f"+ {len(rest[:20])}/{len(rest)} other CHIP) -- per-track model loads",
          flush=True)
    return take


def _score_one(oracle, oracle_name: str, v: dict) -> tuple[list[dict], dict]:
    """Score one variant through the SHARED scorer, keeping each track's null bounds.

    ``score_variant_multilayer`` is what ``build_variant_report`` calls internally, so
    the raw score here is the shipped statistic. Going through the report instead adds
    gene fan-out and window logic that is irrelevant to this measurement -- and on Sei
    actively drops every track with "Outside scoring window".

    Returns (rows, coverage) where *coverage* counts tracks the scorer could not place
    in a layer. That count is a finding in its own right: Sei's 40 tracks all come back
    ``layer='other'`` with ``raw_score=None``, and Sei appears in no committed example
    output, so its variant path through the shared scorer is unexercised.
    """
    from chorus.analysis.normalization import PerTrackNormalizer
    from chorus.analysis.scorers import score_variant_multilayer
    from chorus.analysis.variant_report import LAYER_CONFIGS
    from chorus.core.globals import CHORUS_DATA_DIR

    norm = PerTrackNormalizer()
    pos = f"{v['chrom']}:{v['pos']}"
    res = oracle.predict_variant_effect(
        genomic_region=f"{pos}-{v['pos'] + 1}",
        variant_position=pos,
        alleles=[v["ref"], v["alt"]],
        assay_ids=_assay_ids(oracle_name, oracle),
        # A path, not the name "hg38" -- the oracles hand this straight to pyfaidx
        # and a bare assembly name raises "file `hg38` not found".
        genome=str(CHORUS_DATA_DIR / "genomes" / "hg38.fa"),
    )
    scored = score_variant_multilayer(res, gene_name=v.get("gene"))

    rows = []
    cov = {"total": 0, "no_raw_score": 0, "layer_other": 0}
    for allele, tracks in scored.items():
        for track_key, f in tracks.items():
            cov["total"] += 1
            layer = f.get("layer")
            if layer in (None, "other"):
                cov["layer_other"] += 1
            raw = f.get("raw_score")
            if raw is None:
                cov["no_raw_score"] += 1
                continue
            cfg = LAYER_CONFIGS.get(layer)
            signed = bool(getattr(cfg, "signed", False))
            value = raw if signed else abs(raw)
            support = norm.effect_null_support(oracle_name, track_key)
            rows.append({
                "variant": v["id"], "oracle": oracle_name, "layer": layer,
                "track": track_key, "allele": allele, "raw": float(raw),
                "signed": signed,
                "pctile": norm.effect_percentile(oracle_name, track_key, value,
                                                 signed=signed),
                "exceedance": norm.effect_exceedance(oracle_name, track_key, value,
                                                    signed=signed),
                "null_lo": support[0] if support else None,
                "null_hi": support[1] if support else None,
            })
    return rows, cov


def build(oracle_name: str):
    if oracle_name == "alphagenome":
        from chorus.oracles.alphagenome import AlphaGenomeOracle
        return AlphaGenomeOracle()
    if oracle_name == "enformer":
        from chorus.oracles.enformer import EnformerOracle
        return EnformerOracle()
    if oracle_name == "borzoi":
        from chorus.oracles.borzoi import BorzoiOracle
        return BorzoiOracle()
    if oracle_name == "chrombpnet":
        from chorus.oracles.chrombpnet import ChromBPNetOracle
        return ChromBPNetOracle()
    if oracle_name == "cherimoya":
        from chorus.oracles.cherimoya import CherimoyaOracle
        return CherimoyaOracle()
    if oracle_name == "sei":
        from chorus.oracles.sei import SeiOracle
        return SeiOracle()
    if oracle_name == "legnet":
        from chorus.oracles.legnet import LegNetOracle
        return LegNetOracle()
    if oracle_name == "epinformerseq":
        from chorus.oracles.epinformerseq import EPInformerSeqOracle
        return EPInformerSeqOracle()
    raise SystemExit(f"unknown oracle {oracle_name}")


def build_and_load(oracle_name: str):
    """Construct, then load weights.

    ``load_pretrained_model()`` takes NO positional argument. Its signature is
    ``(weights=None)``, so passing a cell type is silently interpreted as a weights
    path -- which is how an earlier run downloaded 41 MB into the repo root. Call it
    bare. ChromBPNet loads per-track models lazily and has no such method.
    """
    oracle = build(oracle_name)
    loader = getattr(oracle, "load_pretrained_model", None)
    if callable(loader):
        loader()
    return oracle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    OUT.mkdir(parents=True, exist_ok=True)

    panel = verify_panel()
    oracle = build_and_load(args.oracle)
    rows, failed, cover = [], [], []
    for v in panel:
        try:
            got, cov = _score_one(oracle, args.oracle, v)
            rows.extend(got)
            cov["variant"] = v["id"]
            cover.append(cov)
            print(f"  {args.oracle}/{v['id']}: {len(got)} scored of {cov['total']} "
                  f"(no_raw={cov['no_raw_score']}, layer_other={cov['layer_other']})",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad locus must not lose the rest
            failed.append({"variant": v["id"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {args.oracle}/{v['id']} FAILED: {exc}", flush=True)

    path = OUT / f"{args.oracle}.json"
    path.write_text(json.dumps(
        {"rows": rows, "failed": failed, "coverage": cover}, indent=1))
    print(f"wrote {path} ({len(rows)} rows, {len(failed)} failures)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
