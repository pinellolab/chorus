"""Release gate: is a full chorus variant report bit-exact across two processes?

#127 was closed by pinning ``--xla_gpu_deterministic_ops=true`` inside
``load_pretrained_model``, measured 9/9 bit-exact at the raw ``predict_sequence``
level. That is necessary but not sufficient for the shipped claim, because a
report also runs the scorers, the exon/window geometry, the percentile lookups and
the tie-breaking hash on top of it. Any of those could reintroduce variation — the
tie-breaking draw in particular is derived from a hash and would be a *silent*
source if it were keyed on anything process-local.

So this runs the real public entry point twice, in two separate OS processes on
the same GPU, and diffs every numeric leaf. The comparison is bitwise, not
tolerance-based: #127's symptom was 454 differing fields with **36 sign flips**,
and a tolerance would have hidden the sign flips that made 92% of CAGE rows
unrankable.

Two processes rather than two calls in one process, because within one process
AlphaGenome was already bit-exact before the fix — the whole defect was
cross-process, which is the case that matters for a user rerunning a report
tomorrow, or for a builder whose null must match a query's numerator.

Usage:
  python scripts/gate_end_to_end_determinism.py --gpu 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# rs12740374 / SORT1 — the flagship example, and a CAGE-bearing locus, so the
# layer whose noise floor #127 was about is actually exercised.
VARIANT = dict(
    chrom="chr1", position=109_274_968, ref="G", alt="T", gene="SORT1",
    # The shipped HEPG2_TRACKS list from scripts/regenerate_examples.py, including
    # both CAGE strands — CAGE is the layer #127's noise floor made unrankable, so
    # a gate that omitted it would miss the case it exists for.
    assay_ids=[
        "DNASE/EFO:0001187 DNase-seq/.",
        "ATAC/EFO:0001187 ATAC-seq/.",
        "CHIP_TF/EFO:0001187 TF ChIP-seq CEBPB/.",
        "CHIP_HISTONE/EFO:0001187 Histone ChIP-seq H3K27ac/.",
        "CAGE/hCAGE EFO:0001187/+",
        "CAGE/hCAGE EFO:0001187/-",
    ],
)


#: One track per layer, read out of the committed SORT1_enformer example rather than invented, so
#: the gate scores the same kinds of track the example does. Enformer uses ENCODE/FANTOM accessions,
#: not AlphaGenome's ontology strings, so the two arms cannot share a list.
ENFORMER_TRACKS = [
    "ENCFF430NNH",   # tf_binding
    "ENCFF035NGT",   # histone_marks
    "GSM2543098",    # chromatin_accessibility
    "CNhs12328",     # tss_activity
]


def _child(out_path: str, gpu: str, oracle_name: str = "alphagenome") -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    sys.path.insert(0, str(REPO))
    # Exactly the path scripts/regenerate_examples.py takes, so this gates what
    # actually ships rather than a convenience wrapper.
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.normalization import PerTrackNormalizer
    from chorus.analysis.variant_report import build_variant_report
    # reference_fasta is required: predict_variant_effect refuses to guess a
    # genome (#128's strict_ref work made that a hard error rather than a
    # substitution). regenerate_examples.py passes it the same way.
    fasta = str(REPO / "genomes" / "hg38.fa")
    if oracle_name == "enformer":
        # use_environment=True on purpose: that is the path regenerate_examples.py takes for the
        # walkthroughs, and it spawns a fresh process per predict. The AlphaGenome arm below runs
        # in-process, so the two arms cover different execution models rather than duplicating one.
        from chorus.oracles.enformer import EnformerOracle
        oracle = EnformerOracle(use_environment=True, reference_fasta=fasta)
    else:
        from chorus.oracles.alphagenome import AlphaGenomeOracle
        oracle = AlphaGenomeOracle(use_environment=False, reference_fasta=fasta)
    oracle.load_pretrained_model()
    pos = f"{VARIANT['chrom']}:{VARIANT['position']}"
    assay_ids = VARIANT["assay_ids"] if oracle_name == "alphagenome" else ENFORMER_TRACKS
    variant_result = oracle.predict_variant_effect(
        genomic_region=f"{pos}-{VARIANT['position'] + 1}",
        variant_position=pos,
        alleles=[VARIANT["ref"], VARIANT["alt"]],
        assay_ids=assay_ids,
    )
    report = build_variant_report(
        variant_result,
        oracle_name=oracle_name,
        gene_name=VARIANT["gene"],
        normalizer=PerTrackNormalizer(),
        analysis_request=AnalysisRequest(
            user_prompt="determinism gate",
            tool_name="analyze_variant_multilayer",
            oracle_name=oracle_name,
            tracks_requested="determinism gate",
        ),
    )
    Path(out_path).write_text(json.dumps(report.to_dict(), sort_keys=True, default=str))
    return 0


#: Fields that change every run by construction and say nothing about determinism.
_VOLATILE = ("generated_at",)


def _leaves(obj, path="", strings=False):
    """Every leaf, with a stable path, so a diff can name the field.

    ``strings=True`` also yields text leaves. That matters for F8: the failure mode there is a
    *top-N identity* flip — a different track surviving `discover_variant_effects`' hard cutoff —
    which shows up as changed `assay_id` / `description` strings. A numeric-only comparison reports
    it as a pile of moved numbers and hides what actually happened. `generated_at` is excluded
    because it is a wall-clock stamp; set SOURCE_DATE_EPOCH to pin it instead.
    """
    if isinstance(obj, dict):
        for k in sorted(obj):
            if k in _VOLATILE:
                continue
            yield from _leaves(obj[k], f"{path}.{k}", strings=strings)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{path}[{i}]", strings=strings)
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, (int, float)):
        yield path, float(obj)
    elif strings and isinstance(obj, str):
        yield path, obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument(
        "--oracle", default="alphagenome", choices=["alphagenome", "enformer"],
        help="Which oracle to gate. The default AlphaGenome path runs in-process "
             "(use_environment=False); `enformer` exercises the use_environment=True subprocess "
             "path that regenerate_examples.py actually uses, which is where F8 lives.",
    )
    ap.add_argument(
        "--strings", action="store_true",
        help="Compare text leaves too, so a top-N identity flip is reported as such rather than "
             "as numeric drift. `generated_at` is always excluded.",
    )
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return _child(args.child, args.gpu, args.oracle)

    tmp = Path(tempfile.mkdtemp())
    outs = []
    for run in (1, 2):
        out = tmp / f"run{run}.json"
        print(f"[gate] process {run}/2 on GPU {args.gpu} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, "--gpu", args.gpu,
             "--oracle", args.oracle, "--child", str(out)],
            cwd=str(REPO), capture_output=True, text=True,
        )
        if proc.returncode != 0 or not out.exists():
            print(proc.stdout[-3000:]); print(proc.stderr[-3000:])
            raise SystemExit(f"child {run} failed with {proc.returncode}")
        outs.append(json.loads(out.read_text()))

    a = dict(_leaves(outs[0], strings=args.strings))
    b = dict(_leaves(outs[1], strings=args.strings))
    keys = sorted(set(a) | set(b))
    missing = [k for k in keys if k not in a or k not in b]
    differing, sign_flips, worst = [], [], 0.0
    for k in keys:
        if k in missing:
            continue
        x, y = a[k], b[k]
        if x == y:
            continue
        if isinstance(x, str) or isinstance(y, str):
            # A text leaf changed: with --strings this is the top-N identity flip F8 can produce,
            # and it is not a numeric drift, so it must not reach the float arithmetic below.
            differing.append((k, x, y))
            continue
        if math.isnan(x) and math.isnan(y):
            continue
        differing.append((k, x, y))
        if x * y < 0:
            sign_flips.append((k, x, y))
        denom = max(abs(x), abs(y), 1e-12)
        worst = max(worst, abs(x - y) / denom)

    print(f"\n[gate] {len(keys)} numeric fields compared")
    print(f"[gate] structural mismatches : {len(missing)}")
    print(f"[gate] differing values      : {len(differing)}")
    print(f"[gate] SIGN FLIPS            : {len(sign_flips)}")
    print(f"[gate] worst relative delta  : {worst:.3e}")
    for k, x, y in differing[:15]:
        print(f"    {k}: {x!r} != {y!r}")
    if missing[:10]:
        print("  missing:", missing[:10])

    ok = not differing and not missing
    print(f"\n[gate] {'PASS — bit-exact across two processes' if ok else 'FAIL'}")
    print("[gate] #127 baseline for contrast: 454 differing, 36 sign flips, 1.6e-2")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
