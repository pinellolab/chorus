#!/usr/bin/env bash
# Re-run sections 6, 9, 10 of the v30 verify matrix with the correct
# class name (`ChromBPNetOracle`, not `ChromBPNet`). Original verify.sh
# had a typo that turned three checks into uncaught ImportErrors.
set -uo pipefail
AUDIT_DIR=/Users/lp698/Projects/chorus.bak-2026-04-28/audits/2026-04-29_v30_slim_mirror_scorched_earth
exec > >(tee "$AUDIT_DIR/reverify.log") 2>&1
sec () { echo ""; echo "=== $* ==="; }

sec "6. Default model_type is chrombpnet_nobias"
mamba run -n chorus python - <<'PY'
import inspect
from chorus.oracles.chrombpnet import ChromBPNetOracle
sig = inspect.signature(ChromBPNetOracle.load_pretrained_model)
default = sig.parameters['model_type'].default
print('  default model_type =', repr(default))
assert default == 'chrombpnet_nobias', f"expected 'chrombpnet_nobias', got {default!r}"
print('PASS  default flipped to chrombpnet_nobias')
PY

sec "9. Round-trip — load K562 DNase ChromBPNet via current code path"
mamba run -n chorus python - <<'PY'
import time, logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
from chorus.oracles.chrombpnet import ChromBPNetOracle
t0 = time.time()
oracle = ChromBPNetOracle()
oracle.load_pretrained_model(assay='DNASE', cell_type='K562')
dt = time.time() - t0
print(f'  load time: {dt:.1f}s')
mp = str(oracle.model_path)
print(f'  model_path: {mp}')
assert 'chrombpnet_nobias' in mp.lower() or 'chrombpnet_wo_bias' in mp.lower(), \
    f"expected nobias variant, got {mp}"
src = 'HF cache' if 'huggingface' in mp.lower() else 'local downloads (ENCODE tarball — F1 fix not active in this env yet)'
print(f'  source: {src}')
print('PASS  K562 DNase nobias loaded')
PY

sec "10. Fig 3f triangulation — ChromBPNet HepG2 + K562 + GM12878 DNase"
mamba run -n chorus python - <<'PY'
"""Fig 3f triangulation with ChromBPNet (the third oracle, paper used).

Replaces the K562 GATA1 enhancer at chrX:48782929-48783129 (hg38)
with the HepG2-optimized DNA-Diffusion seq id=99598_GENERATED_HEPG2
(read directly from Fig 3g) and predicts ChromBPNet log-counts in a
1000bp window centered at the swap, for HepG2 / K562 / GM12878 DNase.

Paper claim: HepG2 ↑, K562 ↓, GM12878 ~unchanged.
"""
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
INPUT_LEN = 2114  # ChromBPNet input
mid = (REGION_START + REGION_END) // 2
half = INPUT_LEN // 2
g = pyfaidx.Fasta('/Users/lp698/Projects/chorus.bak-2026-04-28/genomes/hg38.fa')
ref_seq = str(g[REGION_CHR][mid-half:mid-half+INPUT_LEN]).upper()
swap_off = INPUT_LEN//2 - 100
alt_seq = ref_seq[:swap_off] + REPLACEMENT + ref_seq[swap_off+200:]
assert len(ref_seq) == len(alt_seq) == INPUT_LEN

# One-hot helper
NUC = {'A':0,'C':1,'G':2,'T':3}
def one_hot(s):
    x = np.zeros((1, len(s), 4), dtype=np.float32)
    for i,c in enumerate(s):
        if c in NUC:
            x[0, i, NUC[c]] = 1.0
    return x

print(f"ChromBPNet (chrombpnet_nobias) Fig 3f swap predictions:")
print(f"  region: {REGION_CHR}:{REGION_START}-{REGION_END} (K562 GATA1 enhancer)")
print(f"  replacement: 99598_GENERATED_HEPG2 ({len(REPLACEMENT)} bp)")
print()

results = {}
for ct in ['HepG2', 'K562', 'GM12878']:
    t0 = time.time()
    oracle = ChromBPNetOracle()
    oracle.load_pretrained_model(assay='DNASE', cell_type=ct)
    print(f"  {ct} loaded in {time.time()-t0:.1f}s")
    print(f"    model_path: {oracle.model_path}")
    # Predict
    x_ref = one_hot(ref_seq)
    x_alt = one_hot(alt_seq)
    out_ref = oracle.model.predict(x_ref, verbose=0)
    out_alt = oracle.model.predict(x_alt, verbose=0)
    # ChromBPNet returns (profile_logits, logcounts)
    logcts_ref = float(out_ref[1][0,0])
    logcts_alt = float(out_alt[1][0,0])
    log2fc = (logcts_alt - logcts_ref) / np.log(2)
    ref_cts = float(np.exp(logcts_ref))
    alt_cts = float(np.exp(logcts_alt))
    results[ct] = (ref_cts, alt_cts, log2fc)
    print(f"    ref_counts={ref_cts:.2f}  alt_counts={alt_cts:.2f}  log2FC={log2fc:+.2f}")

print()
print("Expected directions (Fig 3f claim):")
print(f"  HepG2 :  ↑  | got log2FC = {results['HepG2'][2]:+.2f}  {'✓' if results['HepG2'][2] > 0.5 else '✗'}")
print(f"  K562  :  ↓  | got log2FC = {results['K562'][2]:+.2f}  {'✓' if results['K562'][2] < -0.5 else '✗'}")
print(f"  GM12878: ~0 | got log2FC = {results['GM12878'][2]:+.2f}  {'✓' if abs(results['GM12878'][2]) < 1.0 else '✗'}")

print()
print("Comparison vs Enformer / AlphaGenome from earlier in the audit (using same locus + replacement seq):")
print("  HepG2 DNase:  Enformer +4.22 | AlphaGenome +8.22 | ChromBPNet (this run) %+.2f" % results['HepG2'][2])
print("  K562 DNase:   Enformer -4.04 | AlphaGenome -5.03 | ChromBPNet (this run) %+.2f" % results['K562'][2])
print("  GM12878 DNase: Enformer +0.04 | AlphaGenome +0.32 | ChromBPNet (this run) %+.2f" % results['GM12878'][2])
PY

echo ""
echo "DONE"
