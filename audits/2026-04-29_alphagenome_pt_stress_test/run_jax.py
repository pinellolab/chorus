"""JAX-side runner for the JAX↔PT equivalence stress test.

Loads the JAX AlphaGenome model once, then predicts at the SORT1 locus
across multiple window sizes × multiple heads. Saves to a single npz.

Run inside chorus-alphagenome env:
    mamba run -n chorus-alphagenome python audits/.../run_jax.py <output.npz>
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np

# JAX gets forced to CPU on macOS — JAX-Metal crashes with default_memory_space
if platform.system() == "Darwin":
    os.environ["JAX_PLATFORMS"] = "cpu"

import pyfaidx  # noqa: E402

GENOME_FASTA = "/Users/lp698/Projects/chorus.bak-2026-04-28/genomes/hg38.fa"
SORT1_TSS = ("chr1", 109_274_968)

# Window sizes to test (need to be powers of 2 in the model's accepted range)
WINDOWS = [(2**17, "128k"), (2**19, "524k"), (2**20, "1m")]

# Heads to extract for the equivalence comparison
HEADS = ["DNASE", "ATAC", "CAGE", "RNA_SEQ"]


def main() -> None:
    out_path = sys.argv[1]
    chrom, tss = SORT1_TSS
    fa = pyfaidx.Fasta(GENOME_FASTA)

    print(f"Loading JAX AlphaGenome model (CPU, {platform.system()})...", flush=True)
    t0 = time.time()
    import jax
    from alphagenome.models.dna_output import OutputType
    from alphagenome_research.model.dna_model import create_from_huggingface

    jax_device = jax.devices("cpu")[0]
    model = create_from_huggingface("all_folds", device=jax_device)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    output_types = [getattr(OutputType, h) for h in HEADS]

    results: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    for win_bp, win_label in WINDOWS:
        half = win_bp // 2
        seq = str(fa[chrom][tss - half:tss + half]).upper().strip("N")
        # Re-trim to power-of-2 if N-stripping shortened it
        valid = [2**p for p in range(15, 21)]
        target = max(v for v in valid if v <= len(seq))
        trim = (len(seq) - target) // 2
        seq = seq[trim:trim + target]
        print(f"window {win_label} ({len(seq)} bp): predicting...", flush=True)
        t0 = time.time()
        out = model.predict_sequence(
            seq,
            requested_outputs=output_types,
            ontology_terms=None,
        )
        dt = time.time() - t0
        timings[win_label] = dt
        print(f"  predict {win_label}: {dt:.2f}s", flush=True)
        for h, ot in zip(HEADS, output_types):
            # Output is an attrs/dataclass with lowercase fields, not a dict
            track_data = getattr(out, h.lower())
            arr = np.asarray(track_data.values)
            results[f"{h.lower()}_{win_label}"] = arr.astype(np.float32)
            print(f"    {h:10s} shape={arr.shape}", flush=True)

    np.savez_compressed(out_path, **results)
    meta = {
        "backend": "jax",
        "device": "cpu",
        "windows": [w[1] for w in WINDOWS],
        "heads": HEADS,
        "timings_s": timings,
        "platform": platform.platform(),
    }
    with open(out_path.replace(".npz", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({"ok": True, "n_arrays": len(results), **meta}))


if __name__ == "__main__":
    main()
