"""Does AlphaGenome predict large gene-level RNA effects for REAL eQTLs?

This decides whether an eQTL-anchored effect null is worth building (#83).

The problem: AlphaGenome's RNA effect null is drawn from gene-anchored random
variants and tops out at 0.0106-0.0417 depending on the track, so any real effect
above ~0.05 saturates at exactly 1.0000 and the percentile stops discriminating.
An eQTL null would put mass where real regulatory effects live — but only if the
model actually *predicts* large effects for known eQTLs.

Two possible outcomes, and they point opposite ways:

* **eQTLs score large** (say median |effect| 0.1-0.5 against a null max of 0.01):
  the mass exists, an eQTL-anchored null would discriminate, and it is worth
  ~10 GPU-hours to build.
* **eQTLs score ~0.01 too**: the model does not produce large gene-level RNA
  effects for *anything*, including variants with measured expression effects. No
  reference class can rescue the percentile, and that is a far more important
  finding than the calibration question — gene-level RNA scoring would simply be
  low-dynamic-range in this model.

Deliberately stacked in favour of the first outcome: only eQTLs with large |slope|
and strong p-values are sampled, each scored against **its own eGene** in a
tissue-matched track. If strong liver eQTLs in a liver track do not move, weaker
ones will not either.

The statistic is identical to the builder's and the query's — mean over the eGene's
merged exon mask in bins, natural log, pseudocount 1e-3 — so the numbers are
directly comparable to the stored null.

Usage:
  python scripts/probe_eqtl_effect_scale.py --gpu 2 --n 150 --tissue Liver
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

EQTL_DIR = Path("/data/chorus_data/eqtl/GTEx_Analysis_v8_eQTL")
OUT = Path("/data/chorus_data/eqtl_effect_probe.json")
INPUT_LENGTH = 1_048_576

# GTEx tissue -> ontology substrings that appear in AlphaGenome RNA track ids.
TISSUE_ONTOLOGY = {
    "Liver": ("UBERON:0002107", "CL:0000182"),          # liver, hepatocyte
    "Whole_Blood": ("UBERON:0000178",),                  # blood
    "Cells_Cultured_fibroblasts": ("CL:0000057", "CL:0002553"),
}


def load_eqtls(tissue: str, n: int, seed: int = 0,
               max_tss_distance: int | None = None) -> list[dict]:
    """Strong eQTLs only — this test should favour the optimistic outcome."""
    path = EQTL_DIR / f"{tissue}.v8.signif_variant_gene_pairs.txt.gz"
    if not path.exists():
        raise SystemExit(f"missing {path}; extract it from the GTEx tar first")
    rows = []
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            try:
                slope = float(f[idx["slope"]])
                maf = float(f[idx["maf"]])
                pval = float(f[idx["pval_nominal"]])
            except (ValueError, KeyError):
                continue
            # strong effect, common enough to be well powered, highly significant
            if abs(slope) < 0.5 or maf < 0.05 or pval > 1e-10:
                continue
            if max_tss_distance is not None:
                try:
                    if abs(int(f[idx["tss_distance"]])) > max_tss_distance:
                        continue
                except (ValueError, KeyError):
                    continue
            vid = f[idx["variant_id"]]
            parts = vid.split("_")
            if len(parts) < 5 or parts[0] == "chrX":
                continue
            chrom, pos, ref, alt = parts[0], int(parts[1]), parts[2], parts[3]
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
                continue  # SNVs only, matching how the null is built
            rows.append({
                "chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
                "gene_id": f[idx["gene_id"]].split(".")[0],
                "slope": slope, "maf": maf, "pval": pval,
            })
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--tissue", default="Liver")
    ap.add_argument("--layer", choices=["RNA_SEQ", "CAGE"], default="RNA_SEQ",
                    help="RNA_SEQ scores the eGene's exon mask; CAGE scores a "
                         "501 bp window centred on the variant")
    ap.add_argument("--max-tss-distance", type=int, default=None,
                    help="restrict to eQTLs this close to the eGene TSS. CAGE is "
                         "a localised promoter peak, so only TSS-proximal eQTLs "
                         "are a fair positive set for it")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    from chorus.utils.annotations import (
        build_transcript_exon_index,
        exon_bins_for_gene,
        genes_with_tss_in_window,
        get_annotation_manager,
    )

    eqtls = load_eqtls(args.tissue, args.n,
                       max_tss_distance=args.max_tss_distance)
    print(f"[eqtl] {len(eqtls)} strong {args.tissue} eQTLs "
          f"(|slope|>=0.5, maf>=0.05, p<=1e-10)", flush=True)

    # ENSG -> gene symbol, so the eGene can be found in the exon index
    manager = get_annotation_manager()
    gtf = manager.get_annotation_path("gencode_v48_basic")
    genes_df = manager._get_genes_df(gtf)
    ensg_to_symbol = {}
    for row in genes_df.itertuples():
        gid = str(getattr(row, "gene_id", "")).split(".")[0]
        if gid:
            ensg_to_symbol[gid] = str(row.gene_name)
    print(f"[eqtl] mapped {len(ensg_to_symbol)} ENSG ids to symbols", flush=True)

    index = build_transcript_exon_index()

    from chorus.oracles.alphagenome import AlphaGenomeOracle
    from alphagenome.models.dna_output import OutputType
    from chorus.oracles.alphagenome_source.alphagenome_metadata import get_metadata

    oracle = AlphaGenomeOracle(use_environment=False)
    oracle.load_pretrained_model()
    metadata = get_metadata()

    # tissue-matched RNA tracks
    wanted = TISSUE_ONTOLOGY.get(args.tissue, ())
    rna_local = []
    # iter_tracks() yields DICTS. Iterating one yields its keys, which is what
    # made an earlier version of this look like a header row followed by tuples.
    for row in metadata.iter_tracks():
        if str(row.get("output_type")) != args.layer:
            continue
        identifier = str(row.get("identifier", ""))
        if any(w in identifier for w in wanted):
            rna_local.append((int(row["index"]), int(row["local_index"]), identifier))
    print(f"[eqtl] {len(rna_local)} tissue-matched {args.layer} tracks for {args.tissue}",
          flush=True)
    if not rna_local:
        raise SystemExit("no tissue-matched RNA track; widen TISSUE_ONTOLOGY")

    import pysam
    ref_fa = pysam.FastaFile(str(REPO / "genomes" / "hg38.fa"))

    results = []
    for k, q in enumerate(eqtls):
        symbol = ensg_to_symbol.get(q["gene_id"])
        if symbol is None:
            continue
        half = INPUT_LENGTH // 2
        start, end = q["pos"] - half, q["pos"] + half
        if start < 0 or end > ref_fa.get_reference_length(q["chrom"]):
            continue
        seq_ref = ref_fa.fetch(q["chrom"], start, end).upper()
        if len(seq_ref) != INPUT_LENGTH:
            continue
        offset = half - 1
        if seq_ref[offset] != q["ref"]:
            continue  # GTEx ref disagrees with hg38 here; skip rather than force
        seq_alt = seq_ref[:offset] + q["alt"] + seq_ref[offset + 1:]

        # the eGene's own mask, exactly as builder and query construct it
        spans = dict(genes_with_tss_in_window(index, q["chrom"], start, end)).get(symbol)
        if not spans:
            continue

        try:
            want = OutputType[args.layer]
            out_ref = oracle._model.predict_sequence(
                seq_ref, requested_outputs=[want], ontology_terms=None)
            out_alt = oracle._model.predict_sequence(
                seq_alt, requested_outputs=[want], ontology_terms=None)
        except Exception as exc:
            print(f"[eqtl] skip {q['chrom']}:{q['pos']}: {type(exc).__name__}", flush=True)
            continue

        attr = "rna_seq" if args.layer == "RNA_SEQ" else "cage"
        ref_arr = np.asarray(getattr(out_ref, attr).values)
        alt_arr = np.asarray(getattr(out_alt, attr).values)
        n_bins = ref_arr.shape[0]
        if args.layer == "RNA_SEQ":
            bins = exon_bins_for_gene(spans, start, end, n_bins, 1)
        else:
            # CAGE is scored as a 501 bp window centred on the variant, via the
            # same shared definition the builder and query use (#144 instance 2) —
            # not over the exon mask, which is an RNA-only construct.
            from chorus.analysis.background_sampling import centered_bin_span
            lo, hi = centered_bin_span(n_bins, 501, 1, centre_bin=n_bins // 2)
            bins = np.arange(lo, hi, dtype=np.int64)
        if len(bins) == 0:
            continue

        for _t_i, li, ident in rna_local:
            if li >= ref_arr.shape[1]:
                continue
            if args.layer == "RNA_SEQ":
                # mean over the mask, ln, 1e-3 — AlphaGenome's GeneMaskLFC
                r = float(np.mean(ref_arr[bins, li]))
                a = float(np.mean(alt_arr[bins, li]))
                effect = float(np.log(a + 1e-3) - np.log(r + 1e-3))
            else:
                # sum over the window, log2, +1 — AlphaGenome's CenterMaskScorer
                # DIFF_LOG2_SUM at width 501
                r = float(np.sum(ref_arr[bins, li]))
                a = float(np.sum(alt_arr[bins, li]))
                effect = float(np.log2((a + 1.0) / (r + 1.0)))
            results.append({
                "chrom": q["chrom"], "pos": q["pos"], "gene": symbol,
                "gtex_slope": q["slope"], "maf": q["maf"],
                "track": ident, "ref_mean": r, "alt_mean": a, "effect": effect,
            })
        if (k + 1) % 10 == 0:
            eff = np.abs([x["effect"] for x in results])
            print(f"[eqtl] {k+1}/{len(eqtls)} variants, {len(results)} rows, "
                  f"median |effect| so far {np.median(eff):.6f}", flush=True)

    OUT.write_text(json.dumps(results, indent=1))
    eff = np.abs(np.array([r["effect"] for r in results]))
    print(f"\n[eqtl] {len(results)} (eQTL, track) rows over "
          f"{len({(r['chrom'], r['pos']) for r in results})} variants")
    if eff.size:
        for lab, v in (("p50", 50), ("p90", 90), ("p99", 99)):
            print(f"   |effect| {lab} = {np.percentile(eff, v):.6f}")
        print(f"   |effect| max = {eff.max():.6f}")
    print(f"[eqtl] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
