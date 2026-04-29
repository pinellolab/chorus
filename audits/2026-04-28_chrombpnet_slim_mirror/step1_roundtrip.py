"""Step 1 round-trip check for the ChromBPNet HF slim mirror.

For each of HepG2 and K562 DNase fold-0 nobias h5:
  1. sha256 source vs slim — must match (already verified outside this script).
  2. tf.keras.models.load_model(source) and load_model(slim) — predictions on
     the GATA1 enhancer at chrX:48,782,929-48,783,129 must be element-wise
     equal (bit-identical bytes guarantee this, but we exercise the load+predict
     pipeline to confirm there's no path-dependent state leaking in).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/lp698/chorus_test/chorus")
STAGING = Path("/tmp/hf-staging")

# 2114 bp window centred on chrX:48,782,929-48,783,129 (200 bp). The plan
# specifies that interval; we centre the ChromBPNet input on its midpoint.
MID = (48_782_929 + 48_783_129) // 2  # 48,783,029
INPUT_LEN = 2114
START = MID - INPUT_LEN // 2     # 48,781,972
END = START + INPUT_LEN           # 48,784,086
CHROM = "chrX"

SOURCES = [
    {
        "label": "DNASE:HepG2",
        "src": REPO / "downloads/chrombpnet/DNASE_HepG2/models/fold_0/model.chrombpnet_nobias.fold_0.ENCSR149XIL.h5",
        "slim": STAGING / "DNASE/HepG2/fold_0/model.chrombpnet_nobias.fold_0.ENCSR149XIL.h5",
    },
    {
        "label": "DNASE:K562",
        "src": REPO / "downloads/chrombpnet/DNASE_K562/models/fold_0/model.chrombpnet_nobias.fold_0.ENCSR000EOT.h5",
        "slim": STAGING / "DNASE/K562/fold_0/model.chrombpnet_nobias.fold_0.ENCSR000EOT.h5",
    },
]


def one_hot(seq: str) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq.upper()):
        if b in mapping:
            arr[i, mapping[b]] = 1.0
    return arr


def main() -> int:
    import pysam
    import tensorflow as tf

    print(f"TF: {tf.__version__}")
    print(f"Devices: {[d.device_type for d in tf.config.list_physical_devices()]}")
    print()

    ref_path = str(REPO / "genomes/hg38.fa")
    ref = pysam.FastaFile(ref_path)
    seq = ref.fetch(CHROM, START, END).upper()
    assert len(seq) == INPUT_LEN, f"expected {INPUT_LEN} bp, got {len(seq)}"
    print(f"Window: {CHROM}:{START}-{END} ({INPUT_LEN} bp), GC = {(seq.count('G')+seq.count('C'))/len(seq):.3f}")
    ohe = one_hot(seq)[None, :, :]   # (1, 2114, 4)

    failures = 0
    for entry in SOURCES:
        print(f"\n── {entry['label']} ──")
        if not entry["src"].exists():
            print(f"  MISSING source: {entry['src']}")
            failures += 1
            continue
        if not entry["slim"].exists():
            print(f"  MISSING slim:   {entry['slim']}")
            failures += 1
            continue
        # Load both
        m_src = tf.keras.models.load_model(str(entry["src"]), compile=False)
        m_slim = tf.keras.models.load_model(str(entry["slim"]), compile=False)
        # Predict (ChromBPNet outputs (profile, counts))
        out_src = m_src(ohe, training=False)
        out_slim = m_slim(ohe, training=False)
        prof_src, count_src = out_src[0].numpy(), out_src[1].numpy()
        prof_slim, count_slim = out_slim[0].numpy(), out_slim[1].numpy()
        # Compare element-wise
        prof_eq = np.array_equal(prof_src, prof_slim)
        count_eq = np.array_equal(count_src, count_slim)
        ok = prof_eq and count_eq
        marker = "✓" if ok else "✗"
        print(f"  {marker} profile bit-equal: {prof_eq}, counts bit-equal: {count_eq}")
        print(f"     profile shape={prof_src.shape}, dtype={prof_src.dtype}, mean={prof_src.mean():.6f}")
        print(f"     counts shape ={count_src.shape}, value ={count_src.tolist()}")
        if not ok:
            print(f"     max abs diff prof  : {np.abs(prof_src-prof_slim).max()}")
            print(f"     max abs diff counts: {np.abs(count_src-count_slim).max()}")
            failures += 1

    if failures:
        print(f"\nFAIL: {failures} round-trip mismatch(es)")
        return 1
    print("\nALL ROUND-TRIPS BIT-IDENTICAL ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
