"""Application-level equivalence: GATA1 region swap with both backends.

Reproduces the v30 audit's Fig 3f swap (chrX:48,782,929-48,783,129 with
the HepG2 DNA-Diffusion 99598_GENERATED_HEPG2 sequence) using
analyze_region_swap from each backend, then diffs the layer-aggregated
effect scores.

Run from chorus base env:
    mamba run -n chorus python audits/.../run_app_swap.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import chorus
from chorus.analysis.region_swap import analyze_region_swap

AUDIT_DIR = Path(__file__).parent

REGION = "chrX:48782929-48783129"
REPLACEMENT = (
    "ATAGAACCTTGGACCTCTCTTTGATCTGTTTGTATTTGTCCTGTTTACGCAATTGTCTTT"
    "GAACTATTTCCACACAGACCTAGGCAGGGGACCCGGCCAATGCATTAATCAGGGGGGAAA"
    "TATTGCTCCATCCAGTTAGTCACTGGACTCTGGACAAATACTTATGCAAGAAGCCACAGA"
    "GTTCACTGTTCTATCCTCAT"
)
ASSAY_IDS = [
    "DNASE/EFO:0002067 DNase-seq/.",   # K562
    "DNASE/EFO:0001187 DNase-seq/.",   # HepG2
    "DNASE/EFO:0002784 DNase-seq/.",   # GM12878
    "CAGE/hCAGE EFO:0002067/+",        # K562 + strand
    "CAGE/hCAGE EFO:0001187/+",        # HepG2 + strand
    "CAGE/hCAGE EFO:0002784/+",        # GM12878 + strand
]


def run_one(oracle_name: str) -> tuple[dict, float]:
    o = chorus.create_oracle(
        oracle_name,
        use_environment=True,
        reference_fasta="/Users/lp698/Projects/chorus.bak-2026-04-28/genomes/hg38.fa",
    )
    o.load_pretrained_model()
    t0 = time.time()
    report = analyze_region_swap(
        oracle=o,
        region=REGION,
        replacement_sequence=REPLACEMENT,
        assay_ids=ASSAY_IDS,
        gene_name="GATA1",
        oracle_name=oracle_name,
    )
    dt = time.time() - t0

    # Pull per-track effects from the swap allele's TrackScores
    rows = []
    swap_scores = report.allele_scores.get("replacement", [])
    if not swap_scores:
        # fallback to first non-reference allele key
        for k, v in report.allele_scores.items():
            if k.lower() != "reference":
                swap_scores = v
                break
    for ts in swap_scores:
        rows.append({
            "track_id": ts.assay_id,
            "layer": ts.layer,
            "ref": ts.ref_value,
            "alt": ts.alt_value,
            "raw_score": ts.raw_score,
        })
    return rows, dt


def main() -> None:
    print(f"Region: {REGION}")
    print(f"Replacement seq: {len(REPLACEMENT)} bp")
    print()

    print("=== JAX backend ===")
    jax_layers, jax_t = run_one("alphagenome")
    print(f"  swap took {jax_t:.1f}s\n")

    print("=== PT backend ===")
    pt_layers, pt_t = run_one("alphagenome_pt")
    print(f"  swap took {pt_t:.1f}s\n")

    print("# Region-swap effect comparison (Δ log2 / raw_score)\n")
    print("| Track | Layer | JAX score | PT score | abs diff | JAX ref | PT ref | JAX alt | PT alt |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    rows = []
    jax_by_id = {r["track_id"]: r for r in jax_layers}
    pt_by_id = {r["track_id"]: r for r in pt_layers}
    for tid in sorted(jax_by_id.keys() & pt_by_id.keys()):
        rj, rp = jax_by_id[tid], pt_by_id[tid]
        ej = rj["raw_score"] or 0.0
        ep = rp["raw_score"] or 0.0
        d = abs(ej - ep)
        print(f"| `{tid}` | {rj['layer']} | {ej:+.3f} | {ep:+.3f} | {d:.4f} "
              f"| {rj['ref']:.2f} | {rp['ref']:.2f} | {rj['alt']:.2f} | {rp['alt']:.2f} |")
        rows.append({"track": tid, "layer": rj["layer"],
                     "jax_score": ej, "pt_score": ep, "abs_diff": d,
                     "jax_ref": rj["ref"], "pt_ref": rp["ref"],
                     "jax_alt": rj["alt"], "pt_alt": rp["alt"]})

    with open(AUDIT_DIR / "app_swap_compare.json", "w") as f:
        json.dump({
            "region": REGION,
            "replacement_len": len(REPLACEMENT),
            "jax_swap_s": jax_t,
            "pt_swap_s": pt_t,
            "rows": rows,
        }, f, indent=2)


if __name__ == "__main__":
    main()
