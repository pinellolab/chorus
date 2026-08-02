"""Is AlphaGenome bit-exact across identical forward passes, and can flags make it?

#127 measured two identical chorus runs differing on 454 numeric fields with 36
SIGN FLIPS, and for CAGE the run-to-run noise (median 0.0054) EXCEEDS the median
effect being reported (0.0058) — i.e. 92.1% of shipped CAGE rows, the gene-TSS
ones, are currently ranking noise. ChromBPNet is bit-exact by contrast, so
bit-exactness is a demonstrated bar on this box rather than an aspiration.

This probes the model directly instead of going through variant scoring, so the
result cannot be confounded by chorus's own arithmetic: load once, call
``predict_sequence`` repeatedly on the *same* sequence, compare raw output arrays
bitwise.

Established before writing this, so the search space is small:

* AlphaGenome runs **locally** in JAX via ``create_from_huggingface``
  (``chorus/oracles/alphagenome.py:168``) — not a remote service, so the
  execution config is ours to set.
* Its ``ApplyFn`` takes **no** PRNG key; only ``init_fn`` does, at a fixed
  ``jax.random.PRNGKey(0)`` (``dna_model.py:1721``). So there is no dropout and
  no varying seed at inference.
* ``grep`` finds **no XLA determinism flag set anywhere in chorus**.

That leaves XLA autotuning picking different kernels per run, and
non-deterministic GPU reductions. Both are flag-controllable, which is why this
is a configuration spike and not a research project.

Flags must be set before JAX initialises, so the driver runs one process per
configuration:

    python scripts/probe_alphagenome_determinism.py --repeats 3 --tag baseline
    XLA_FLAGS=--xla_gpu_deterministic_ops=true \\
        python scripts/probe_alphagenome_determinism.py --repeats 3 --tag detops

Per-bin arrays are persisted (not derived scores) so that every candidate
statistic's noise stays recomputable offline at zero further GPU — RNA's units
change in the same programme, and any noise constant derived from today's scores
would be stale on arrival.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# SORT1 rs12740374 — the flagship locus, so signal levels are representative
# rather than the near-zero values a random sequence would give.
CHROM, POS = "chr1", 109_274_968
SEQ_LEN = 1_048_576
OUT_DIR = Path("/data/chorus_data/determinism")


def load_sequence() -> str:
    from pyfaidx import Fasta

    fasta = Fasta(str(REPO / "genomes" / "hg38.fa"), as_raw=True, sequence_always_upper=True)
    half = SEQ_LEN // 2
    start = POS - half
    seq = fasta[CHROM][start:start + SEQ_LEN]
    if len(seq) != SEQ_LEN:
        raise SystemExit(f"got {len(seq)} bp, expected {SEQ_LEN}")
    return str(seq)


def collect_arrays(output) -> dict[str, np.ndarray]:
    """Every numeric array in an AlphaGenome Output, keyed by output type."""
    arrays: dict[str, np.ndarray] = {}
    for name in dir(output):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(output, name)
        except Exception:
            continue
        if attr is None or callable(attr):
            continue
        values = getattr(attr, "values", None)
        if isinstance(values, np.ndarray) and values.size:
            arrays[name] = np.asarray(values)
    return arrays


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--keep-bp", type=int, default=20_000,
                    help="central bins persisted per output type (0 = none)")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    xla = os.environ.get("XLA_FLAGS", "")
    print(f"[{args.tag}] XLA_FLAGS={xla or '<unset>'}", flush=True)

    seq = load_sequence()
    print(f"[{args.tag}] sequence {CHROM}:{POS} {len(seq)} bp "
          f"md5={hashlib.md5(seq.encode()).hexdigest()[:12]}", flush=True)

    from chorus.oracles.alphagenome import AlphaGenomeOracle

    # use_environment=False keeps the model in THIS process: the subprocess
    # worker path would hide self._model and add a serialisation boundary that
    # could mask or manufacture differences. We are already inside
    # chorus-alphagenome, so direct loading is the right path.
    t0 = time.time()
    oracle = AlphaGenomeOracle(use_environment=False)
    oracle.load_pretrained_model()
    if oracle._model is None:
        raise SystemExit("model did not load in-process")
    print(f"[{args.tag}] model loaded in {time.time() - t0:.0f}s", flush=True)

    from alphagenome.models.dna_output import OutputType
    from chorus.oracles.alphagenome_source.alphagenome_metadata import (
        SKIPPED_OUTPUT_TYPES,
    )

    requested = [ot for ot in OutputType if ot.name not in SKIPPED_OUTPUT_TYPES]
    print(f"[{args.tag}] requesting {len(requested)} output types", flush=True)

    runs: list[dict[str, np.ndarray]] = []
    times: list[float] = []
    for r in range(args.repeats):
        t = time.time()
        out = oracle._model.predict_sequence(
            seq, requested_outputs=requested, ontology_terms=None
        )
        times.append(time.time() - t)
        runs.append(collect_arrays(out))
        print(f"[{args.tag}] pass {r + 1}/{args.repeats} in {times[-1]:.1f}s", flush=True)

    keys = sorted(set(runs[0]) & set(runs[-1]))
    report: dict[str, object] = {
        "tag": args.tag,
        "xla_flags": xla,
        "repeats": args.repeats,
        "seconds_per_pass": times,
        "locus": f"{CHROM}:{POS}",
        "outputs": {},
    }

    all_exact = True
    print(f"\n[{args.tag}] {'output':22} {'shape':>20} {'bit-exact':>10} "
          f"{'max|diff|':>12} {'sign flips':>11} {'rel':>10}", flush=True)
    for key in keys:
        a = runs[0][key]
        worst_diff, flips, exact = 0.0, 0, True
        for other in runs[1:]:
            b = other[key]
            if b.shape != a.shape:
                exact = False
                continue
            # array_equal allocates nothing; only pay for a diff when it differs.
            # No float64 upcast: these are up to ~1M bins x thousands of tracks,
            # and upcasting two of them at once thrashes for no extra precision.
            if np.array_equal(a, b):
                continue
            exact = False
            step = max(1, a.shape[0] // 64)
            for lo in range(0, a.shape[0], step):
                ca, cb = a[lo:lo + step], b[lo:lo + step]
                worst_diff = max(worst_diff, float(np.abs(ca - cb).max()))
                flips += int(np.count_nonzero((ca > 0) & (cb < 0) | (ca < 0) & (cb > 0)))
        scale = float(np.abs(a).max()) or 1.0
        all_exact &= exact
        report["outputs"][key] = {
            "shape": list(a.shape),
            "bit_exact": exact,
            "max_abs_diff": worst_diff,
            "max_abs_value": scale,
            "relative": worst_diff / scale,
            "sign_flips": flips,
            "n_values": int(a.size),
        }
        print(f"[{args.tag}] {key:22} {str(a.shape):>20} {str(exact):>10} "
              f"{worst_diff:12.3e} {flips:11d} {worst_diff / scale:10.2e}", flush=True)

    report["bit_exact_overall"] = all_exact
    print(f"\n[{args.tag}] BIT-EXACT OVERALL: {all_exact}", flush=True)

    # Persist only the central window, not the full ~1 Mb. Compressing every bin
    # of 9 output types x 3 repeats ran past 2 GB and dominated the runtime, for
    # data no statistic in chorus reads: every center-mask window is <= 2001 bp.
    # RNA's exon mask does span the locus, so widening this is a deliberate
    # decision for the real 50-locus harness, not an oversight here.
    half = args.keep_bp // 2
    slices: dict[str, np.ndarray] = {}
    for r, run in enumerate(runs):
        for k in keys:
            arr = run[k]
            mid = arr.shape[0] // 2
            slices[f"run{r}__{k}"] = arr[max(0, mid - half):mid + half]
    npz = OUT_DIR / f"perbin_{args.tag}.npz"
    np.savez_compressed(str(npz), **slices)
    (OUT_DIR / f"report_{args.tag}.json").write_text(json.dumps(report, indent=2))
    print(f"[{args.tag}] central {args.keep_bp} bins -> {npz} "
          f"({npz.stat().st_size / 1048576:.0f} MB)", flush=True)
    return 0 if all_exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
