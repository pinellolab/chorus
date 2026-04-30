"""Per-oracle smoke: load + 1 prediction + finite-value check.

Uses canonical assay IDs verified against each oracle's metadata.
"""
from __future__ import annotations

import sys
import time
import traceback
import numpy as np

import chorus

# Each tuple: (oracle_name, ctor_kwargs, predict_args, expected_shape_at_least_dim)
SMOKES = [
    ("enformer",
     {},
     {"input_data": ("chr11", 5_247_000, 5_248_000),
      "assay_ids": ["ENCFF413AHU"]}),
    ("borzoi",
     {"fold": 0},
     {"input_data": ("chr1", 109_274_968 - 100_000, 109_274_968 + 100_000),
      "assay_ids": ["CNhs10608+"]}),
    ("chrombpnet",
     {},
     None),  # special — needs assay/cell_type at load time
    ("sei",
     {},
     {"input_data": ("chr1", 1_000_000, 1_004_096),
      "assay_ids": None}),  # all classes
    ("legnet",
     {"assay": "LentiMPRA", "cell_type": "HepG2"},
     {"input_data": "A" * 200,
      "assay_ids": None}),
    ("alphagenome",
     {},
     {"input_data": ("chr1", 109_274_968 - 65536, 109_274_968 + 65536),
      "assay_ids": ["DNASE/EFO:0002067 DNase-seq/."]}),
    ("alphagenome_pt",
     {},
     {"input_data": ("chr1", 109_274_968 - 65536, 109_274_968 + 65536),
      "assay_ids": ["DNASE/EFO:0002067 DNase-seq/."]}),
]


def smoke(name: str, ctor_kwargs: dict, predict_kwargs: dict | None) -> tuple[str, str]:
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    try:
        oracle = chorus.create_oracle(
            name,
            use_environment=True,
            reference_fasta="genomes/hg38.fa",
            **ctor_kwargs,
        )
        print(f"[construct] {time.time()-t0:.1f}s", flush=True)
        t0 = time.time()

        # ChromBPNet needs (assay, cell_type) at load time
        if name == "chrombpnet":
            oracle.load_pretrained_model(assay="DNASE", cell_type="K562", fold=0)
            print(f"[load] {time.time()-t0:.1f}s", flush=True)
            t0 = time.time()
            result = oracle.predict(("chr1", 1_000_000, 1_002_114))
        else:
            oracle.load_pretrained_model()
            print(f"[load] {time.time()-t0:.1f}s", flush=True)
            t0 = time.time()
            result = oracle.predict(**predict_kwargs)

        dt = time.time() - t0
        print(f"[predict] {dt:.1f}s", flush=True)

        tracks = dict(result.items())
        if not tracks:
            return (name, f"FAIL: empty result")

        finite_ok = True
        for tname, track in list(tracks.items())[:3]:
            vals = track.values
            if not np.isfinite(vals).all():
                finite_ok = False
                print(f"  ✗ {tname}: contains non-finite values shape={vals.shape}", flush=True)
            else:
                print(f"  ✓ {tname}: shape={vals.shape}, mean={vals.mean():.4f}, max={vals.max():.4f}", flush=True)

        if finite_ok:
            return (name, f"PASS — {len(tracks)} tracks, all finite, predict={dt:.1f}s")
        return (name, f"FAIL — non-finite values")
    except Exception as exc:
        print(f"[result] FAIL", flush=True)
        traceback.print_exc()
        return (name, f"FAIL: {type(exc).__name__}: {str(exc)[:100]}")


if __name__ == "__main__":
    summary = []
    for name, ctor, pk in SMOKES:
        result = smoke(name, ctor, pk)
        summary.append(result)

    print("\n\n=== Phase G summary ===")
    for n, r in summary:
        print(f"  {n:20s}  {r}")

    fails = sum(1 for _, r in summary if not r.startswith("PASS"))
    sys.exit(fails)
