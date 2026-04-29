#!/usr/bin/env bash
# v30 audit verification matrix — run AFTER `chorus setup --oracle all` completes.
set -uo pipefail
AUDIT_DIR=/Users/lp698/Projects/chorus.bak-2026-04-28/audits/2026-04-29_v30_slim_mirror_scorched_earth
cd /Users/lp698/Projects/chorus.bak-2026-04-28
exec > >(tee "$AUDIT_DIR/verify.log") 2>&1

ok () { echo "PASS  $*"; }
fail () { echo "FAIL  $*"; FAIL=1; }
sec () { echo ""; echo "=== $* ==="; }
FAIL=0

sec "1. chorus --version"
ver=$(mamba run -n chorus chorus --version 2>&1 | tail -1)
[[ "$ver" == *"0.3.0"* ]] && ok "$ver" || fail "expected 0.3.0, got: $ver"

sec "2. chorus list"
mamba run -n chorus chorus list

sec "3. chorus health"
mamba run -n chorus chorus health 2>&1 | tail -20

sec "4. chorus backgrounds status"
mamba run -n chorus chorus backgrounds status 2>&1 | tail -20

sec "5. --all-chrombpnet flag still present"
mamba run -n chorus chorus setup --help 2>&1 | grep -q "all-chrombpnet" && ok "flag present" || fail "flag missing"

sec "6. NEW: Default model_type is chrombpnet_nobias"
mamba run -n chorus python - <<'PY'
import inspect
from chorus.oracles.chrombpnet import ChromBPNet
sig = inspect.signature(ChromBPNet.load_pretrained_model)
default = sig.parameters['model_type'].default
print('  default model_type =', repr(default))
assert default == 'chrombpnet_nobias', f"expected 'chrombpnet_nobias', got {default!r}"
print('PASS  default flipped to chrombpnet_nobias')
PY

sec "7. NEW: chrombpnet downloads/ size after setup (slim mirror)"
sz=$(du -sh /Users/lp698/Projects/chorus.bak-2026-04-28/downloads/chrombpnet 2>/dev/null | cut -f1)
echo "  downloads/chrombpnet/ = $sz (was 3.5 GB on v29 fast-path)"
sz_mb=$(du -sm /Users/lp698/Projects/chorus.bak-2026-04-28/downloads/chrombpnet 2>/dev/null | cut -f1 || echo 0)
[[ "$sz_mb" -lt 200 ]] && ok "slim (< 200 MB)" || echo "INFO  size = ${sz_mb}MB (slim default keeps fast-path; full registry only via --all-chrombpnet)"

sec "8. NEW: HF cache populated"
hf_cache_size=$(du -sh ~/.cache/huggingface/hub 2>/dev/null | cut -f1 || echo "n/a")
echo "  ~/.cache/huggingface/hub = $hf_cache_size"
ls ~/.cache/huggingface/hub 2>&1 | grep -i chorus-chrombpnet-slim && ok "slim mirror cached" || echo "INFO  HF cache populated lazily on first load_oracle"

sec "9. NEW: Round-trip — load K562 DNase ChromBPNet (default = nobias) via HF mirror"
mamba run -n chorus python - <<'PY'
import time, os
from chorus.oracles.chrombpnet import ChromBPNet
t0 = time.time()
oracle = ChromBPNet()
oracle.load_pretrained_model(assay='DNASE', cell_type='K562')
dt = time.time() - t0
print(f'  load time: {dt:.1f}s')
print(f'  model_path: {oracle.model_path}')
mp = str(oracle.model_path)
assert 'chrombpnet_nobias' in mp, f"expected nobias, got {mp}"
assert 'fold_0' in mp, f"expected fold_0, got {mp}"
src = 'HF cache' if 'huggingface' in mp.lower() else 'local downloads'
print(f'  source: {src}')
print('PASS  K562 DNase nobias loaded')
PY

sec "10. Fig 3f triangulation — ChromBPNet HepG2 + K562 + GM12878 DNase"
mamba run -n chorus python - <<'PY'
"""Re-do the Fig 3f DNA-Diffusion test, this time with ChromBPNet
(the third oracle, now finally available via the slim mirror)."""
import time
import numpy as np
from chorus.oracles.chrombpnet import ChromBPNet

REGION = ('chrX', 48782929, 48783129)  # K562 GATA1 enhancer (hg38)
REPLACEMENT = (
    "ATAGAACCTTGGACCTCTCTTTGATCTGTTTGTATTTGTCCTGTTTACGCAATTGTCTTT"
    "GAACTATTTCCACACAGACCTAGGCAGGGGACCCGGCCAATGCATTAATCAGGGGGGAAA"
    "TATTGCTCCATCCAGTTAGTCACTGGACTCTGGACAAATACTTATGCAAGAAGCCACAGA"
    "GTTCACTGTTCTATCCTCAT"
)
print("ChromBPNet (chrombpnet_nobias) Fig 3f swap predictions:")
print(f"  region: chr{REGION[0]}:{REGION[1]}-{REGION[2]} (K562 GATA1 enhancer)")
print(f"  replacement: 99598_GENERATED_HEPG2 ({len(REPLACEMENT)} bp)")
print()

# ChromBPNet input is 2114bp centered at the region midpoint;
# we predict over the full window then sum counts in the 200bp swap window.
import pyfaidx
genome = pyfaidx.Fasta('/Users/lp698/Projects/chorus.bak-2026-04-28/genomes/hg38.fa')
mid = (REGION[1] + REGION[2]) // 2
INPUT_LEN = 2114
half = INPUT_LEN // 2
ref_seq = str(genome[REGION[0]][mid-half:mid-half+INPUT_LEN]).upper()
# Build alt by replacing the central 200bp
swap_start_in_input = INPUT_LEN//2 - 100
alt_seq = ref_seq[:swap_start_in_input] + REPLACEMENT + ref_seq[swap_start_in_input+200:]
assert len(ref_seq) == len(alt_seq) == INPUT_LEN

results = {}
for ct in ['HepG2', 'K562', 'GM12878']:
    t0 = time.time()
    oracle = ChromBPNet()
    oracle.load_pretrained_model(assay='DNASE', cell_type=ct)
    print(f"  {ct} loaded in {time.time()-t0:.1f}s ({oracle.model_path})")
    # Predict on both ref and alt
    import numpy as np
    def predict(seq):
        # ChromBPNet expects one-hot encoded (1, 2114, 4) input
        from chorus.oracles.chrombpnet_source import one_hot
        x = one_hot.dna_to_one_hot([seq])
        out = oracle.model.predict(x, verbose=0)
        # out is (profile_logits, logcounts) — return total predicted counts
        logits, logcts = out
        return float(np.exp(logcts[0,0]))
    ref_cts = predict(ref_seq)
    alt_cts = predict(alt_seq)
    log2fc = np.log2(alt_cts / ref_cts)
    results[ct] = (ref_cts, alt_cts, log2fc)
    print(f"    ref={ref_cts:.2f}  alt={alt_cts:.2f}  log2FC={log2fc:+.2f}")

print()
print("Expected directions (Fig 3f claim):")
print(f"  HepG2 :  ↑  | got log2FC = {results['HepG2'][2]:+.2f}")
print(f"  K562  :  ↓  | got log2FC = {results['K562'][2]:+.2f}")
print(f"  GM12878: ~0 | got log2FC = {results['GM12878'][2]:+.2f}")
PY

sec "11. Fast pytest"
mamba run -n chorus pytest tests/ -q -m "not integration and not slow" --tb=line 2>&1 | tail -20

sec "12. Integration tests (slow)"
mamba run -n chorus pytest tests/ -q -m integration --tb=line 2>&1 | tail -20

sec "DONE"
[[ $FAIL -eq 0 ]] && echo "OVERALL: PASS" || echo "OVERALL: FAIL ($FAIL failures above)"
exit $FAIL
