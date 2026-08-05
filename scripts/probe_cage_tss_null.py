"""What would a TSS-only CAGE null give, before spending ~11 h building one?

CAGE is a localised promoter peak, so the natural reference class for it is
"variants at annotated TSSs" rather than the stratified mixture the shared region
set uses (20 % within 1 kb of a TSS, 20 % at 1-10 kb, 33 % junction-proximal, 12 %
gene body, 15 % uniform — fractions that were a labelled guess).

Measured so far, with strong TSS-proximal liver eQTLs as the positive set:

  uniform-random null   CAGE eQTL percentile p50 = 0.857
  gene-anchored null    CAGE eQTL percentile p50 = 0.659

The gene-anchored null is already *harder* than the old one, because 40 % of its
positions sit near TSSs where CAGE responds most. A TSS-ONLY null is harder still,
and the open question is by how much — hence this probe rather than a rebuild.

It builds a small empirical CAGE null from N annotated protein-coding TSSs, then
reports where the same eQTL effects fall against it. Cost is ~2N forward passes,
roughly 30 minutes at N=300, against ~11 hours for the real build.

Worth stating the consequence of the design, whatever the number: against a
TSS-only null, a variant far from any TSS gets a percentile near 0 — correctly, but
it means the CAGE percentile largely tracks "is this variant at a TSS", which the
user already knows. That is the real cost of a perfectly matched reference class,
and it is a judgement call rather than a bug.

Usage:
  python scripts/probe_cage_tss_null.py --gpu 4 --n 300
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

INPUT_LENGTH = 1_048_576
OUT = Path(os.environ.get("CAGE_PROBE_OUT", "/data/chorus_data/cage_tss_null_probe.json"))
LIVER = ("UBERON:0002107", "CL:0000182")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--window", type=int, default=501)
    ap.add_argument("--offset-uniform-bp", type=int, default=None, metavar="N",
                    help="Draw the variant's distance from the TSS uniformly over "
                         "[-N, +N]. This is the reference class with NO selection "
                         "effect: 'a random SNV within N bp of an annotated "
                         "protein-coding TSS'. Preferred over --offset-like-eqtl, "
                         "whose distribution partly reflects GTEx's testing window "
                         "and discovery power rather than biology.")
    ap.add_argument("--offset-like-eqtl", metavar="TISSUE", default=None,
                    help="Instead of placing the variant exactly AT the TSS, draw "
                         "its distance from the empirical tss_distance distribution "
                         "of this tissue's significant eQTLs. Placing it at the peak "
                         "maximum makes the null systematically more perturbing than "
                         "any real variant population — measured, p50 0.323 vs the "
                         "gene-anchored null's 0.659.")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    from chorus.analysis.background_sampling import centered_bin_span
    from chorus.utils.annotations import get_annotation_manager, load_chrom_sizes

    manager = get_annotation_manager()
    gtf = manager.get_annotation_path("gencode_v48_basic")
    genes = manager._get_genes_df(gtf)
    pc = genes[genes["gene_type"] == "protein_coding"]
    sizes = load_chrom_sizes(str(REPO / "genomes" / "hg38.fa.fai"))

    # every annotated protein-coding TSS, strand-aware, with room for a 1 Mb window
    tss = []
    for row in pc.itertuples():
        chrom = str(row.chrom)
        if chrom not in sizes or sizes[chrom] < 2 * 5_000_000:
            continue
        pos = int(row.start) if row.strand == "+" else int(row.end)
        if 5_000_000 <= pos <= sizes[chrom] - 5_000_000:
            tss.append((chrom, pos))
    print(f"[cage] {len(tss)} annotated PC TSSs with room for a 1 Mb window",
          flush=True)
    random.Random(0).shuffle(tss)
    tss = tss[:args.n]

    from alphagenome.models.dna_output import OutputType
    from chorus.oracles.alphagenome import AlphaGenomeOracle
    from chorus.oracles.alphagenome_source.alphagenome_metadata import get_metadata

    oracle = AlphaGenomeOracle(use_environment=False)
    oracle.load_pretrained_model()
    metadata = get_metadata()
    cage = [(int(r["index"]), int(r["local_index"]), str(r["identifier"]))
            for r in metadata.iter_tracks()
            if str(r.get("output_type")) == "CAGE"
            and any(w in str(r.get("identifier", "")) for w in LIVER)]
    print(f"[cage] {len(cage)} liver CAGE tracks", flush=True)
    if not cage:
        raise SystemExit("no liver CAGE track found")

    import pysam
    ref_fa = pysam.FastaFile(str(REPO / "genomes" / "hg38.fa"))
    rng = random.Random(1)

    # The offset pool. Empty means "variant exactly at the TSS".
    offsets: list[int] = []
    if args.offset_uniform_bp:
        n = int(args.offset_uniform_bp)
        offsets = list(range(-n, n + 1))
        print(f"[cage] uniform offsets over +/-{n} bp "
              f"(mean |d| {n / 2:.0f} bp)", flush=True)
    elif args.offset_like_eqtl:
        import gzip
        path = (Path("/data/chorus_data/eqtl/GTEx_Analysis_v8_eQTL")
                / f"{args.offset_like_eqtl}.v8.signif_variant_gene_pairs.txt.gz")
        with gzip.open(path, "rt") as fh:
            idx = {n: i for i, n in enumerate(fh.readline().rstrip("\n").split("\t"))}
            for line in fh:
                f = line.rstrip("\n").split("\t")
                try:
                    d = int(f[idx["tss_distance"]])
                except (ValueError, KeyError):
                    continue
                if abs(d) <= 10_000:      # the range the TSS strata cover
                    offsets.append(d)
        print(f"[cage] {len(offsets)} eQTL tss_distances within 10 kb; "
              f"median |d| {np.median(np.abs(offsets)):.0f} bp", flush=True)
        if not offsets:
            raise SystemExit("no eQTL offsets loaded")

    null: dict[str, list[float]] = {ident: [] for _, _, ident in cage}
    for k, (chrom, pos) in enumerate(tss):
        if offsets:
            pos = pos + rng.choice(offsets)
        half = INPUT_LENGTH // 2
        start, end = pos - half, pos + half
        seq_ref = ref_fa.fetch(chrom, start, end).upper()
        if len(seq_ref) != INPUT_LENGTH or seq_ref.count("N") > INPUT_LENGTH * 0.5:
            continue
        offset = half - 1
        base = seq_ref[offset]
        if base not in "ACGT":
            continue
        alt = rng.choice([b for b in "ACGT" if b != base])
        seq_alt = seq_ref[:offset] + alt + seq_ref[offset + 1:]
        try:
            o_ref = oracle._model.predict_sequence(
                seq_ref, requested_outputs=[OutputType.CAGE], ontology_terms=None)
            o_alt = oracle._model.predict_sequence(
                seq_alt, requested_outputs=[OutputType.CAGE], ontology_terms=None)
        except Exception as exc:
            print(f"[cage] skip {chrom}:{pos}: {type(exc).__name__}", flush=True)
            continue
        ra = np.asarray(o_ref.cage.values)
        aa = np.asarray(o_alt.cage.values)
        n_bins = ra.shape[0]
        lo, hi = centered_bin_span(n_bins, args.window, 1, centre_bin=n_bins // 2)
        for _t_i, li, ident in cage:
            if li >= ra.shape[1]:
                continue
            r = float(np.sum(ra[lo:hi, li]))
            a = float(np.sum(aa[lo:hi, li]))
            null[ident].append(abs(float(np.log2((a + 1.0) / (r + 1.0)))))
        if (k + 1) % 25 == 0:
            got = sum(len(v) for v in null.values())
            print(f"[cage] {k+1}/{len(tss)} TSSs, {got} null samples", flush=True)

    OUT.write_text(json.dumps(null))
    print(f"\n[cage] wrote {OUT}")

    # where do the eQTL effects fall against this TSS-only null?
    probe = json.load(open("/data/chorus_data/eqtl_effect_probe.json"))
    print(f"\n{'track':46} {'null p50':>9} {'null max':>9} {'eQTL p50 pct':>13}")
    all_pcts = []
    for ident, vals in null.items():
        if not vals:
            continue
        arr = np.sort(np.array(vals))
        eff = [abs(r["effect"]) for r in probe if r["track"] == ident]
        if not eff:
            continue
        pcts = [float(np.searchsorted(arr, e, side="right") / len(arr)) for e in eff]
        all_pcts += pcts
        print(f"{ident[:46]:46} {np.median(arr):9.4f} {arr.max():9.4f} "
              f"{np.median(pcts):13.3f}")
    if all_pcts:
        a = np.array(all_pcts)
        print(f"\nTSS-only CAGE null: eQTL percentile p10 {np.percentile(a,10):.3f}  "
              f"p50 {np.percentile(a,50):.3f}  p90 {np.percentile(a,90):.3f}  "
              f"saturated {100*(a>=1.0).mean():.1f}%")
        print("compare: uniform-random p50 0.857 | gene-anchored p50 0.659")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
