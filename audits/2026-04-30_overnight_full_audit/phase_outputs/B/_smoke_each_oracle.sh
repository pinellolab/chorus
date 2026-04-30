#!/usr/bin/env bash
# Per-oracle smoke: construct oracle, load weights, run one prediction,
# verify finite values. Each oracle gets its own log file.
#
# Run from repo root. No assumption on prior caching — uses chorus's
# regular load path so HF mirror fall-back / fall-forward is exercised.

set -u
cd "$(dirname "$0")/../../../.." || exit 1
OUT=audits/2026-04-30_overnight_full_audit/phase_outputs/B
mkdir -p "$OUT"

run_smoke() {
    local oracle="$1"
    local extra_kwargs="$2"
    local predict_call="$3"
    local logfile="$OUT/${oracle}.log"

    echo "=== $oracle ===" | tee -a "$logfile"
    date | tee -a "$logfile"

    mamba run -n chorus python -u <<PY 2>&1 | tee -a "$logfile"
import time, traceback, numpy as np
import chorus

t0 = time.time()
try:
    oracle = chorus.create_oracle(
        '$oracle', use_environment=True, $extra_kwargs
    )
    print(f"[construct] {time.time()-t0:.1f}s")
    t0 = time.time()
    oracle.load_pretrained_model()
    print(f"[load] {time.time()-t0:.1f}s")

    t0 = time.time()
    result = $predict_call
    dt = time.time()-t0
    print(f"[predict] {dt:.1f}s")

    tracks = dict(result.items())
    print(f"[result] {len(tracks)} tracks")
    finite_ok = True
    for name, track in list(tracks.items())[:3]:
        vals = track.values
        if not np.isfinite(vals).all():
            finite_ok = False
            print(f"  ✗ {name}: contains non-finite values (shape={vals.shape})")
        else:
            print(f"  ✓ {name}: shape={vals.shape}, mean={vals.mean():.4f}, max={vals.max():.4f}")

    if finite_ok:
        print(f"[result] PASS — {len(tracks)} tracks, all finite, predict took {dt:.1f}s")
    else:
        print("[result] FAIL — found non-finite values")
except Exception:
    print("[result] FAIL")
    traceback.print_exc()
PY

    echo "" | tee -a "$logfile"
}

# 1. Enformer — TFHub fallback / chorus mirror; small region in K562 DNase
run_smoke "enformer" \
    "reference_fasta='genomes/hg38.fa'" \
    "oracle.predict(('chr11', 5246694, 5247590), assay_ids=['DNASE:K562'])"

# 2. Borzoi — johahi → chorus mirror; SORT1
run_smoke "borzoi" \
    "reference_fasta='genomes/hg38.fa', fold=0" \
    "oracle.predict(('chr1', 109274968 - 100000, 109274968 + 100000), assay_ids=oracle.list_assay_ids()[:3])"

# 3. ChromBPNet — slim mirror; default K562 DNase
run_smoke "chrombpnet" \
    "reference_fasta='genomes/hg38.fa'" \
    "(oracle.load_pretrained_model(assay='DNASE', cell_type='K562', fold=0) or oracle.predict(('chr1', 1000000, 1002114)))"

# 4. Sei — chorus-sei mirror first, Zenodo fallback
run_smoke "sei" \
    "reference_fasta='genomes/hg38.fa'" \
    "oracle.predict(('chr1', 1000000, 1004096), assay_ids=oracle.list_assay_ids()[:3])"

# 5. LegNet — chorus mirror, with assay/cell_type ctor kwargs
run_smoke "legnet" \
    "reference_fasta='genomes/hg38.fa', assay='LentiMPRA', cell_type='HepG2'" \
    "oracle.predict('A' * 200)"

# 6. AlphaGenome JAX
run_smoke "alphagenome" \
    "reference_fasta='genomes/hg38.fa'" \
    "oracle.predict(('chr1', 109_274_968 - 65536, 109_274_968 + 65536), assay_ids=['DNASE/EFO:0002067 DNase-seq/.'])"

# 7. AlphaGenome PT
run_smoke "alphagenome_pt" \
    "reference_fasta='genomes/hg38.fa'" \
    "oracle.predict(('chr1', 109_274_968 - 65536, 109_274_968 + 65536), assay_ids=['DNASE/EFO:0002067 DNase-seq/.'])"

echo "All Phase B smokes done at $(date)" >> "$OUT/SUMMARY.txt"
for f in "$OUT"/*.log; do
    name=$(basename "$f" .log)
    if grep -q "PASS" "$f"; then
        echo "PASS  $name" >> "$OUT/SUMMARY.txt"
    else
        echo "FAIL  $name" >> "$OUT/SUMMARY.txt"
    fi
done
cat "$OUT/SUMMARY.txt"
