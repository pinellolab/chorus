"""Emit the CHANGELOG before/after table from the artefacts, not from notes.

Every number here is read out of the two NPZ files being compared, so the table cannot
drift from what shipped. Re-run it after any oracle finishes rather than editing numbers
by hand.

Three tables, because they answer different questions:

  1. RETENTION -- what was thinned, and what it is now. This is the defect.
  2. CONSEQUENCE -- how the null moved, per layer, as medians of PER-TRACK ratios.
     Ratios of medians flatter the result (measured: 1.37 against a true 1.174 on
     enformer tf_binding), and conflating the two is an error already made once in this
     cycle, so the distinction is enforced here in code.
  3. REFERENCE SET -- which population each oracle drew from, and whether it reproduces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ORACLES = ["alphagenome", "borzoi", "enformer", "chrombpnet",
           "cherimoya", "sei", "legnet", "epinformerseq"]
LAYERS = ["effect", "summary", "perbin"]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def _stat(row: np.ndarray, q):
    a = np.abs(row)
    return float(a.max()) if q is None else float(np.quantile(a, q))


def _median_per_track_ratio(new: np.ndarray, old: np.ndarray, q) -> float:
    """Median of PER-TRACK ratios. Never the ratio of medians."""
    r = []
    for a, b in zip(old, new):
        sa = _stat(a, q)
        if sa > 0:
            r.append(_stat(b, q) / sa)
    return float(np.median(r)) if r else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", default="/data/chorus_data/rebuild_2026-08-06")
    ap.add_argument("--backups", default="/data/chorus_data/pre_unified_rebuild")
    args = ap.parse_args()
    staged, backups = Path(args.staged), Path(args.backups)

    print("### 1. Retention: what was thinned\n")
    print("Two distinct claims, kept apart deliberately. *Was thinned* compares the OLD")
    print("offered count against the old capacity -- an actual defect in shipped data.")
    print("*Would have thinned* is a layer that only exceeds capacity at the NEW position")
    print("count, so the rebuild would have introduced the defect it exists to fix.\n")
    print("| oracle | layer | offered before | offered after | cap | was thinned | "
          "would newly thin | retained after |")
    print("|---|---|---|---|---|---|---|---|")
    CAP = {"alphagenome": 20_000}
    for o in ORACLES:
        new = _load(staged / f"{o}_pertrack.npz")
        old_f = _load(backups / f"{o}_pertrack.npz")
        if new is None:
            continue
        cap = CAP.get(o, 50_000)
        for layer in LAYERS:
            ck = f"{layer}_counts"
            if ck not in new:
                continue
            off_new = int(np.asarray(new[ck]).max())
            off_old = (int(np.asarray(old_f[ck]).max())
                       if old_f is not None and ck in old_f else None)
            was = off_old is not None and off_old > cap
            would = (not was) and off_new > cap
            if not (was or would):
                continue
            ret = new.get(f"{layer}_retained")
            tk = int(new[f"{layer}_tail_k"]) if f"{layer}_tail_k" in new else 0
            after = ("**exact**"
                     if ret is not None and (np.asarray(ret) >= np.asarray(new[ck])).all()
                     else f"{cap:,} + exact top {tk:,}")
            print(f"| {o} | {layer} | {off_old if off_old is None else f'{off_old:,}'} | "
                  f"{off_new:,} | {cap:,} | "
                  f"{f'**{off_old / cap:.2f}x**' if was else '—'} | "
                  f"{f'**{off_new / cap:.2f}x**' if would else '—'} | {after} |")

    print("\n### 2. Consequence: medians of per-track ratios (new/old)\n")
    print("| oracle | layer | n | p50 | p90 | p99 | max | % tracks with a higher ceiling |")
    print("|---|---|---|---|---|---|---|---|")
    for o in ORACLES:
        new, old = _load(staged / f"{o}_pertrack.npz"), _load(backups / f"{o}_pertrack.npz")
        if new is None or old is None:
            continue
        n_ids = [str(x) for x in new["track_ids"]]
        o_ids = [str(x) for x in old["track_ids"]]
        if set(n_ids) != set(o_ids):
            print(f"| {o} | — | — | track set changed; not comparable | | | | |")
            continue
        order = [o_ids.index(i) for i in n_ids]
        for layer in LAYERS:
            k = f"{layer}_cdfs"
            if k not in new or k not in old:
                continue
            nm, om = new[k], old[k][order]
            vals = [_median_per_track_ratio(nm, om, q) for q in (.5, .9, .99, None)]
            higher = float(np.mean([_stat(b, None) > _stat(a, None) * 1.001
                                    for a, b in zip(om, nm)]))
            print(f"| {o} | {layer} | {len(nm)} | " +
                  " | ".join(f"{v:.3f}" for v in vals) + f" | {higher:.0%} |")

    print("\n### 3. The mechanism, checked against every thinned layer\n")
    print("A uniform *m*-of-*N* subsample retains the population maximum with probability")
    print("exactly *m/N*, so **1 - m/N** predicts the share of tracks whose ceiling exact")
    print("retention should raise. If the diagnosis is right, prediction and measurement")
    print("agree across layers spanning a 30-fold range of thinning.\n")
    print("| oracle.layer | thinning | 1 - m/N predicts | measured |")
    print("|---|---|---|---|")
    CAP2 = {"alphagenome": 20_000}
    for o in ORACLES:
        new, old_f = _load(staged / f"{o}_pertrack.npz"), _load(backups / f"{o}_pertrack.npz")
        if new is None or old_f is None:
            continue
        n_ids = [str(x) for x in new["track_ids"]]
        o_ids = [str(x) for x in old_f["track_ids"]]
        if set(n_ids) != set(o_ids):
            continue
        order = [o_ids.index(i) for i in n_ids]
        cap = CAP2.get(o, 50_000)
        for layer in LAYERS:
            ck, dk = f"{layer}_counts", f"{layer}_cdfs"
            if ck not in old_f or dk not in new or dk not in old_f:
                continue
            counts = np.asarray(old_f[ck])
            m = counts > cap
            if not m.any():
                continue
            nm, om = new[dk], old_f[dk][order]
            # compare only the tracks that were thinned
            sel = m[order] if len(m) == len(order) else m
            pk = cap / float(counts[m].max())
            got = float(np.mean([_stat(b, None) > _stat(a, None) * 1.001
                                 for a, b, keep in zip(om, nm, sel) if keep]))
            print(f"| {o}.{layer} | {float(counts[m].max()) / cap:.1f}x | "
                  f"{100 * (1 - pk):.1f}% | **{100 * got:.1f}%** |")

    print("\n### 4. Reference population\n")
    ref = Path("reference_sets/chorus_reference_positions_v1.npz")
    if ref.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "brps", Path("scripts/build_reference_position_sets.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with np.load(ref, allow_pickle=False) as d:
            prov = json.loads(str(d["provenance"][0]))
        print("| set | purpose | size | sha256 (first 16) |")
        print("|---|---|---|---|")
        for name, meta in sorted(prov["sets"].items()):
            size = meta.get("n_snps") or meta.get("n_positions")
            print(f"| `{name}` | {meta['kind']} | {size:,} | `{meta['sha256'][:16]}` |")
        print("\n| oracle | family | reproduces the reference population? |")
        print("|---|---|---|")
        for o in ORACLES:
            if not (staged / f"{o}_pertrack.npz").exists():
                print(f"| {o} | {mod.ORACLE_SNP_SET.get(o, '?')} | not built yet |")
                continue
            ok = mod.verify(ref, o, staged) == 0
            print(f"| {o} | {mod.ORACLE_SNP_SET.get(o, '?')} | "
                  f"{'**yes**' if ok else 'NO'} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
