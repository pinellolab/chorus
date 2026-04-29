"""Compare JAX vs PT npz outputs from run_jax.py / run_pt.py.

Loads both bundles, computes per-key (head × window) deltas, prints a
markdown table, and writes structured JSON for the audit report.

Run from chorus base env:
    mamba run -n chorus python audits/.../compare.py jax_out.npz pt_out.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


# Tolerances mirror the existing test_alphagenome_backends_equivalence.py
ABS_TOL = 0.10
REL_TOL = 0.05


def main() -> None:
    jax_path = Path(sys.argv[1])
    pt_path = Path(sys.argv[2])
    out_dir = jax_path.parent

    jax = np.load(jax_path)
    pt = np.load(pt_path)
    jax_meta = json.load(open(jax_path.with_name(jax_path.stem + "_meta.json")))
    pt_meta = json.load(open(pt_path.with_name(pt_path.stem + "_meta.json")))

    keys_jax = set(jax.files)
    keys_pt = set(pt.files)
    shared = sorted(keys_jax & keys_pt)
    only_jax = sorted(keys_jax - keys_pt)
    only_pt = sorted(keys_pt - keys_jax)

    print()
    print("# AlphaGenome JAX↔PT equivalence (SORT1 chr1:109,274,968)")
    print()
    print(f"- JAX backend: {jax_meta.get('backend')}, device={jax_meta.get('device')}")
    print(f"- PT  backend: {pt_meta.get('backend')}, devices={pt_meta.get('device_per_window')}")
    print(f"- Tolerances: abs ≤ {ABS_TOL}, rel ≤ {REL_TOL*100:.0f}%")
    print()
    if only_jax:
        print(f"  *only in JAX*: {only_jax}")
    if only_pt:
        print(f"  *only in PT*:  {only_pt}")
    print()
    print("| Head × window | shape | max abs Δ | mean abs Δ | mean rel Δ | abs OK | rel OK |")
    print("|---|---|---:|---:|---:|---|---|")

    rows: list[dict] = []
    n_pass = 0
    n_total = 0
    for key in shared:
        a = jax[key].astype(np.float64)
        b = pt[key].astype(np.float64)
        if a.shape != b.shape:
            print(f"| {key} | shape mismatch jax={a.shape} pt={b.shape} | — | — | — | — | — |")
            rows.append({"key": key, "error": "shape_mismatch", "jax_shape": list(a.shape), "pt_shape": list(b.shape)})
            continue
        diff = np.abs(a - b)
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        mean_mag = float(np.abs(a).mean()) + 1e-9
        mean_rel = mean_abs / mean_mag
        abs_ok = max_abs <= ABS_TOL
        rel_ok = mean_rel <= REL_TOL
        n_total += 1
        if abs_ok and rel_ok:
            n_pass += 1
        print(f"| `{key}` | {a.shape} | {max_abs:.4f} | {mean_abs:.4f} | {mean_rel:.4f} | {'✓' if abs_ok else '✗'} | {'✓' if rel_ok else '✗'} |")
        rows.append({
            "key": key,
            "shape": list(a.shape),
            "max_abs": max_abs, "mean_abs": mean_abs, "mean_rel": mean_rel,
            "abs_ok": abs_ok, "rel_ok": rel_ok,
        })

    print()
    print(f"**{n_pass}/{n_total} keys within tolerance.**")
    print()

    print("## Speed comparison")
    print()
    print("| Window | JAX time (s) | PT time (s, device) | speedup |")
    print("|---|---:|---|---:|")
    for win in jax_meta.get("windows", []):
        jt = jax_meta["timings_s"].get(win, "—")
        pt_t = pt_meta["timings_s"].get(win, "—")
        pt_dev = pt_meta.get("device_per_window", {}).get(win, "?")
        try:
            sx = f"{jt/pt_t:.2f}×" if isinstance(jt, (int, float)) and isinstance(pt_t, (int, float)) and pt_t > 0 else "—"
        except Exception:
            sx = "—"
        jt_s = f"{jt:.2f}" if isinstance(jt, (int, float)) else jt
        pt_s = f"{pt_t:.2f} ({pt_dev})" if isinstance(pt_t, (int, float)) else f"— ({pt_dev})"
        print(f"| {win} | {jt_s} | {pt_s} | {sx} |")
    print()

    out = {
        "summary": {
            "n_pass": n_pass, "n_total": n_total,
            "abs_tol": ABS_TOL, "rel_tol": REL_TOL,
            "only_jax": only_jax, "only_pt": only_pt,
        },
        "rows": rows,
        "jax_meta": jax_meta,
        "pt_meta": pt_meta,
    }
    with open(out_dir / "compare_result.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
