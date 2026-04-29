#!/usr/bin/env bash
# Re-run section 10 only (Fig 3f triangulation) using the oracle's
# proper _predict API instead of the raw .model.predict — the
# previous reverify failed because in `use_environment=True` mode
# the TF model lives in a subprocess and oracle.model is None.
set -uo pipefail
AUDIT_DIR=/Users/lp698/Projects/chorus.bak-2026-04-28/audits/2026-04-29_v30_slim_mirror_scorched_earth
exec > >(tee "$AUDIT_DIR/reverify2.log") 2>&1

echo "=== 10. Fig 3f triangulation — ChromBPNet HepG2 + K562 + GM12878 DNase ==="
mamba run -n chorus python - <<'PY'
"""Fig 3f triangulation with ChromBPNet via oracle's _predict API."""
import time
import numpy as np
import pyfaidx
from chorus.oracles.chrombpnet import ChromBPNetOracle

REGION_CHR = 'chrX'
REGION_START = 48782929
REGION_END = 48783129
REPLACEMENT = (
    "ATAGAACCTTGGACCTCTCTTTGATCTGTTTGTATTTGTCCTGTTTACGCAATTGTCTTT"
    "GAACTATTTCCACACAGACCTAGGCAGGGGACCCGGCCAATGCATTAATCAGGGGGGAAA"
    "TATTGCTCCATCCAGTTAGTCACTGGACTCTGGACAAATACTTATGCAAGAAGCCACAGA"
    "GTTCACTGTTCTATCCTCAT"
)
INPUT_LEN = 2114
mid = (REGION_START + REGION_END) // 2
half = INPUT_LEN // 2
g = pyfaidx.Fasta('/Users/lp698/Projects/chorus.bak-2026-04-28/genomes/hg38.fa')
ref_seq = str(g[REGION_CHR][mid-half:mid-half+INPUT_LEN]).upper()
swap_off = INPUT_LEN//2 - 100
alt_seq = ref_seq[:swap_off] + REPLACEMENT + ref_seq[swap_off+200:]
assert len(ref_seq) == len(alt_seq) == INPUT_LEN

print(f"region: {REGION_CHR}:{REGION_START}-{REGION_END} (K562 GATA1 enhancer)")
print(f"replacement: 99598_GENERATED_HEPG2 ({len(REPLACEMENT)} bp)")
print(f"window: {INPUT_LEN}bp centered on swap midpoint")
print()

results = {}
for ct in ['HepG2', 'K562', 'GM12878']:
    t0 = time.time()
    oracle = ChromBPNetOracle()
    oracle.load_pretrained_model(assay='DNASE', cell_type=ct)
    print(f"{ct} loaded in {time.time()-t0:.1f}s")
    print(f"  source: {'HF cache' if 'huggingface' in oracle.model_path.lower() else 'local downloads (ENCODE)'}")

    # Use the oracle's _predict API (sequence-based, returns OraclePrediction)
    out_ref = oracle._predict(ref_seq)
    out_alt = oracle._predict(alt_seq)
    # OraclePrediction has .data : (n_tracks, n_bins) — for chrombpnet single-track
    # the per-bp probabilities. Sum to total counts.
    ref_total = float(out_ref.data.sum())
    alt_total = float(out_alt.data.sum())
    log2fc = np.log2(alt_total / ref_total) if ref_total > 0 else float('nan')
    results[ct] = (ref_total, alt_total, log2fc)
    print(f"  ref_signal_sum={ref_total:.2f}  alt_signal_sum={alt_total:.2f}  log2FC={log2fc:+.2f}")
    print()

print("=== summary ===")
print(f"  HepG2 :  ↑  | got log2FC = {results['HepG2'][2]:+.2f}  {'PASS' if results['HepG2'][2] > 0.3 else 'FAIL'}")
print(f"  K562  :  ↓  | got log2FC = {results['K562'][2]:+.2f}  {'PASS' if results['K562'][2] < -0.3 else 'FAIL'}")
print(f"  GM12878: ~0 | got log2FC = {results['GM12878'][2]:+.2f}  {'PASS' if abs(results['GM12878'][2]) < 1.5 else 'FAIL'}")

print()
print("Cross-oracle agreement at chrX:48,782,929-48,783,129 (K562 GATA1 enhancer)")
print("with the same HepG2 DNA-Diffusion replacement (id=99598_GENERATED_HEPG2):")
print(f"  HepG2 DNase   Enformer +4.22  AlphaGenome +8.22  ChromBPNet {results['HepG2'][2]:+.2f}")
print(f"  K562  DNase   Enformer -4.04  AlphaGenome -5.03  ChromBPNet {results['K562'][2]:+.2f}")
print(f"  GM12878 DNase Enformer +0.04  AlphaGenome +0.32  ChromBPNet {results['GM12878'][2]:+.2f}")
PY
echo ""
echo "DONE"
