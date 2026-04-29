"""Chorus-API level equivalence between alphagenome (JAX) and alphagenome_pt.

Calls oracle._predict(seq, assay_ids) for both backends — this exercises
chorus's own per-head local_index slicing, which is the actual end-user
path. Compares the per-track per-bp values for a representative set of
assay_ids, scoring like the existing variant-effect path would.

Run from chorus base env (uses subprocess to per-oracle env internally):
    mamba run -n chorus python audits/.../run_chorus_api.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import chorus

AUDIT_DIR = Path(__file__).parent
GENOME_FASTA = "/Users/lp698/Projects/chorus.bak-2026-04-28/genomes/hg38.fa"

# 524k window centered at SORT1 TSS — fast on both backends, exercises
# the per-head slicing the same way every chorus user does.
SORT1_TSS = ("chr1", 109_274_968)
WINDOW_BP = 524_288

# Representative assays — one per head, two cell-types per head where possible.
ASSAY_IDS = [
    "DNASE/EFO:0002067 DNase-seq/.",   # K562
    "DNASE/EFO:0001187 DNase-seq/.",   # HepG2
    "DNASE/EFO:0002784 DNase-seq/.",   # GM12878
    "ATAC/EFO:0002067 ATAC-seq/.",     # K562
    "ATAC/EFO:0001187 ATAC-seq/.",     # HepG2
    "CAGE/hCAGE EFO:0002067/+",        # K562 plus strand
    "CAGE/hCAGE EFO:0001187/+",        # HepG2 plus strand
]


def fetch_seq() -> str:
    import pyfaidx
    fa = pyfaidx.Fasta(GENOME_FASTA)
    chrom, tss = SORT1_TSS
    half = WINDOW_BP // 2
    return str(fa[chrom][tss - half:tss + half]).upper().strip("N")[:WINDOW_BP]


def predict_one(oracle_name: str, seq: str) -> tuple[dict[str, np.ndarray], float]:
    o = chorus.create_oracle(oracle_name, use_environment=True)
    o.load_pretrained_model()
    t0 = time.time()
    out = o._predict(seq, assay_ids=ASSAY_IDS)
    dt = time.time() - t0
    arrays: dict[str, np.ndarray] = {}
    for aid in ASSAY_IDS:
        track = out[aid]
        arrays[aid] = np.asarray(track).astype(np.float32)
    return arrays, dt


def main() -> None:
    print(f"Fetching SORT1 ±{WINDOW_BP//2} bp window...", flush=True)
    seq = fetch_seq()
    print(f"  seq length: {len(seq)} bp\n", flush=True)

    print("=== JAX backend (alphagenome) ===", flush=True)
    jax_arrays, jax_t = predict_one("alphagenome", seq)
    print(f"  predict took {jax_t:.1f}s\n", flush=True)

    print("=== PT backend (alphagenome_pt) ===", flush=True)
    pt_arrays, pt_t = predict_one("alphagenome_pt", seq)
    print(f"  predict took {pt_t:.1f}s\n", flush=True)

    print("# Per-assay equivalence (524 kb SORT1, sliced via chorus local_index)\n")
    print("| Assay | JAX shape | PT shape | max abs Δ | mean abs Δ | mean rel Δ | sum jax | sum pt | sum Δ |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")

    rows = []
    for aid in ASSAY_IDS:
        a = jax_arrays[aid].astype(np.float64)
        b = pt_arrays[aid].astype(np.float64)
        if a.shape != b.shape:
            print(f"| `{aid}` | {a.shape} | {b.shape} | shape mismatch | — | — | — | — | — |")
            rows.append({"assay": aid, "shape_mismatch": True,
                         "jax_shape": list(a.shape), "pt_shape": list(b.shape)})
            continue
        diff = np.abs(a - b)
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        mean_mag = float(np.abs(a).mean()) + 1e-9
        mean_rel = mean_abs / mean_mag
        sum_jax = float(a.sum())
        sum_pt = float(b.sum())
        sum_rel_diff = (sum_pt - sum_jax) / (abs(sum_jax) + 1e-9)
        print(f"| `{aid}` | {a.shape} | {b.shape} | {max_abs:.4f} | {mean_abs:.4f} | "
              f"{mean_rel:.4f} | {sum_jax:.2f} | {sum_pt:.2f} | {sum_rel_diff:+.4f} |")
        rows.append({
            "assay": aid, "shape_mismatch": False,
            "max_abs": max_abs, "mean_abs": mean_abs, "mean_rel": mean_rel,
            "sum_jax": sum_jax, "sum_pt": sum_pt, "sum_rel_diff": sum_rel_diff,
        })

    # Save full numpy arrays for offline inspection
    np.savez_compressed(AUDIT_DIR / "chorus_api_jax.npz", **{
        aid.replace("/", "_").replace(":", "_").replace(" ", "_"): v
        for aid, v in jax_arrays.items()
    })
    np.savez_compressed(AUDIT_DIR / "chorus_api_pt.npz", **{
        aid.replace("/", "_").replace(":", "_").replace(" ", "_"): v
        for aid, v in pt_arrays.items()
    })
    with open(AUDIT_DIR / "chorus_api_compare.json", "w") as f:
        json.dump({
            "window_bp": WINDOW_BP,
            "tss": SORT1_TSS,
            "jax_predict_s": jax_t,
            "pt_predict_s": pt_t,
            "rows": rows,
        }, f, indent=2)


if __name__ == "__main__":
    main()
