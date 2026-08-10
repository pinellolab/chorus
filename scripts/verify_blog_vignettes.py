"""Do the blog post's two vignettes still recapitulate on the rebuilt backgrounds?

The 2026-08-05 rebuild changed every oracle's effect reference population, so every
percentile in both vignettes moved. The numbers were always expected to move; the
question is whether the **biological findings** survive:

Vignette 1 — rs12740374 / SORT1 (Musunuru et al., Nature 2010)
  V1.1  the alternate allele OPENS chromatin, and HepG2 (liver) is among the
        strongest cell types ChromBPNet covers
  V1.2  the C/EBP family tops AlphaGenome's HepG2 TF panel, with binding GAINED
  V1.3  SORT1 is the top gene by predicted expression change, and it is ~118 kb away
        — outside every short-context oracle's window
  V1.4  H3K27ac gained, CAGE (eRNA) gained, LegNet MPRA alt > ref
  V1.5  the SORT1 magnitude is UNDER-predicted (well under 2-fold against >12-fold
        measured) — a stated limitation, so it must stay true

Vignette 2 — rs9504151 / CDYL (Sniff preprint)
  V2.1  rs9504151 ranks 1st of its ~56-proxy credible set in AlphaGenome
  V2.2  accessibility and active histone marks DROP on the alternate allele
  V2.3  RNA predicts no significant change
  V2.4  ATF4 / CEBPB top the ALL-TF panel with binding LOST
  V2.5  rs9504151 also ranks 1st with ChromBPNet on lung fibroblast (IMR-90)

V2.1-V2.3 are already answerable from the committed walkthrough and are checked
there. This script covers the parts that need a fresh forward pass: the panels that
range over ALL tracks rather than a hand-picked handful.

It deliberately drives the oracle the way the shipped report path does —
``predict_variant_effect`` then the shared ``scorers`` — so a pass here is evidence
about chorus, not about a bespoke script.

Usage:
  python scripts/verify_blog_vignettes.py --part alphagenome --gpu 0
  python scripts/verify_blog_vignettes.py --part chrombpnet --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT = Path("/data/chorus_data/blog_vignette_check")

SORT1 = dict(chrom="chr1", pos=109_274_968, ref="G", alt="T", gene="SORT1",
             cell="EFO:0001187", label="rs12740374 (SORT1, HepG2)")
CDYL = dict(chrom="chr6", pos=4_577_675, ref="T", alt="A", gene="CDYL",
            cell="CL:0002553", label="rs9504151 (CDYL, lung fibroblast)")


def _report(oracle, v, assay_ids=None):
    """Score a variant through the shipped report path."""
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.normalization import PerTrackNormalizer
    from chorus.analysis.variant_report import build_variant_report

    pos = f"{v['chrom']}:{v['pos']}"
    res = oracle.predict_variant_effect(
        genomic_region=f"{pos}-{v['pos'] + 1}",
        variant_position=pos,
        alleles=[v["ref"], v["alt"]],
        assay_ids=assay_ids,
    )
    return build_variant_report(
        res, oracle_name=oracle.name if hasattr(oracle, "name") else "alphagenome",
        gene_name=v["gene"], normalizer=PerTrackNormalizer(),
        analysis_request=AnalysisRequest(
            user_prompt="blog vignette verification",
            tool_name="analyze_variant_multilayer",
            oracle_name="alphagenome", tracks_requested="all"),
    )


def _rows(report):
    d = report.to_dict()
    return [s for p in (d.get("alleles") or {}).values()
            for s in (p.get("all_scores") or [])]


def run_alphagenome(gpu: str) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    from chorus.oracles.alphagenome import AlphaGenomeOracle

    oracle = AlphaGenomeOracle(
        use_environment=False,
        reference_fasta=str(REPO / "genomes" / "hg38.fa"),
    )
    oracle.load_pretrained_model()
    OUT.mkdir(parents=True, exist_ok=True)

    for v in (SORT1, CDYL):
        print(f"\n{'=' * 72}\n{v['label']}\n{'=' * 72}", flush=True)
        report = _report(oracle, v)          # assay_ids=None -> every track
        rows = _rows(report)
        print(f"  scored {len(rows)} rows", flush=True)
        (OUT / f"ag_{v['gene']}.json").write_text(json.dumps(rows, default=str))

        by_layer = defaultdict(list)
        for s in rows:
            by_layer[s.get("layer")].append(s)

        # every layer's top hit, by |raw|
        print(f"\n  {'layer':26} {'top track':44} {'raw':>8} {'pctile':>8}")
        for layer in sorted(by_layer):
            best = max(by_layer[layer], key=lambda s: abs(float(s.get("raw_score") or 0)))
            print(f"  {layer:26} {str(best.get('assay_id'))[:44]:44} "
                  f"{float(best['raw_score']):+8.3f} {best.get('quantile_score')}")

        # the TF panel, cell-matched and global, which is what the vignettes claim on
        tf = [s for s in by_layer.get("tf_binding", [])]
        for scope, subset in (("cell-matched", [s for s in tf if v["cell"] in str(s.get("assay_id"))]),
                              ("ALL cell types", tf)):
            subset = sorted(subset, key=lambda s: -abs(float(s.get("raw_score") or 0)))
            print(f"\n  TF panel, {scope} ({len(subset)} tracks) — top 8 by |raw|:")
            for s in subset[:8]:
                aid = str(s.get("assay_id"))
                print(f"    {float(s['raw_score']):+8.3f}  q={str(s.get('quantile_score')):8} {aid[:60]}")

        # per-gene expression, for the "which gene moves" claim
        ge = sorted(by_layer.get("gene_expression", []),
                    key=lambda s: -abs(float(s.get("raw_score") or 0)))
        print(f"\n  gene_expression rows ({len(ge)}) — top 8 by |raw|:")
        for s in ge[:8]:
            print(f"    {float(s['raw_score']):+8.4f}  {str(s.get('region_label')):26} "
                  f"{str(s.get('assay_id'))[:44]}")


def run_chrombpnet(gpu: str) -> None:
    """V1.1: which cell types open, and where HepG2 ranks."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    from chorus.analysis.normalization import PerTrackNormalizer
    from chorus.oracles.chrombpnet import ChromBPNetOracle

    norm = PerTrackNormalizer()
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    oracle = ChromBPNetOracle(reference_fasta=str(REPO / "genomes" / "hg38.fa"))
    cells = oracle.list_cell_types()   # signature is (self) -> List[str]
    print(f"  ChromBPNet DNASE cell types available: {len(cells)}", flush=True)
    for cell in cells:
        try:
            oracle.load_pretrained_model(assay="DNASE", cell_type=cell, fold=0)
            v = SORT1
            pos = f"{v['chrom']}:{v['pos']}"
            res = oracle.predict_variant_effect(
                genomic_region=f"{pos}-{v['pos'] + 1}",
                variant_position=pos, alleles=[v["ref"], v["alt"]], assay_ids=None)
            from chorus.analysis.scorers import score_variant_multilayer
            # signature is (variant_result, gene_name=None); the normalizer is
            # applied by the report layer, not here
            scored = score_variant_multilayer(res)
            # Shape is {allele: {track_key: {field: value}}} -- a dict of plain
            # dicts, not objects. Verified by inspection rather than assumed; two
            # earlier guesses at this structure produced zero rows silently.
            for allele, tracks in (scored or {}).items():
                for key, fields in (tracks or {}).items():
                    raw = (fields or {}).get("raw_score")
                    if raw is None:
                        continue
                    results.append(dict(cell_type=cell, track=str(key),
                                        raw=float(raw),
                                        pctile=(fields or {}).get("quantile_score"),
                                        ref=(fields or {}).get("ref_value"),
                                        alt=(fields or {}).get("alt_value")))
        except Exception as exc:
            print(f"    {cell}: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
    (OUT / "chrombpnet_celltypes.json").write_text(json.dumps(results, default=str))
    results.sort(key=lambda r: -r["raw"])
    print(f"\n  ChromBPNet DNASE, rs12740374, ranked by raw log2FC ({len(results)} models):")
    for i, r in enumerate(results[:12], 1):
        mark = "  <-- HepG2" if "hepg2" in r["cell_type"].lower() else ""
        print(f"    {i:2d}. {r['raw']:+7.3f}  q={str(r['pctile']):8} {r['cell_type'][:40]}{mark}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["alphagenome", "chrombpnet"])
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    if args.part == "alphagenome":
        run_alphagenome(args.gpu)
    else:
        run_chrombpnet(args.gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
