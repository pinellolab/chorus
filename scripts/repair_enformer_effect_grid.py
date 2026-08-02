"""Re-grid enformer's padded ``effect_cdfs`` onto the full 10,000-point grid.

THE DEFECT
----------
``enformer_pertrack.npz`` shipped its ``effect_cdfs`` gridded at **9,606** points
— ``max(effect_counts)`` — and then padded to 10,000 by repeating each row's
maximum 395 times. All 5,313 rows, every one with its first maximum at index
9605. Because ``PerTrackNormalizer._get_denominator`` correctly divides by the
stored *width* (#119), the padding meant:

* every enformer effect percentile was compressed by ~0.9606;
* ``(0.9605, 1.0)`` was unreachable — the top 4 % of the scale did not exist;
* a #83 per-track floor at any ``q >= 0.96`` resolves to the null **maximum**,
  so the q sweep is contaminated for all 5,313 tracks until this is fixed.

``ReservoirSampler.to_cdf_matrix`` interpolates short rows onto the full grid and
*cannot* produce that shape, so the shipped file was never reproducible from repo
code. ``build_and_save`` only resampled when ``shape[1] > n_points``, so a narrow
matrix was written verbatim. Both holes are now closed by the write-time guard in
``build_and_save`` and by ``cdf_grid_violations``.

THE REPAIR, AND ITS LIMITS
--------------------------
The original samples are gone, so this recovers the quantile function from the
9,606 stored quantiles and re-expresses it on the 10,000-point grid. No
information is invented and none of the stored quantiles is discarded, but the
result is an *interpolation of a derived artefact*, not a rebuild. The Phase 5
enformer rebuild supersedes it. It is still a strict improvement: it removes a
known multiplicative bias and restores the unreachable top of the scale.

Two conventions matter and are easy to get backwards:

* the stored narrow row means ``narrow[j] = quantile at j / (NARROW - 1)``, and
  ``narrow[-1]`` is the true sample maximum under both ``to_cdf_matrix``
  branches;
* pseudo-samples are a *sample array*, so they span ``linspace(0, 1, n)`` and end
  **at** the maximum, whereas ``to_cdf_matrix``'s interpolation *source* is
  ``arange(n) / n``, which stops short of 1.0. Using ``arange(n)/n`` for the
  pseudo-samples silently discards the top sample — measured, that loses up to
  **74 %** of a row's maximum, precisely the tail this repair exists to restore.

Usage::

    python scripts/repair_enformer_effect_grid.py --dry-run
    python scripts/repair_enformer_effect_grid.py

Republishing to HuggingFace is a separate, authenticated step.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chorus.analysis.background_sampling import cdf_grid_violations  # noqa: E402

WIDE = 10_000
KEY = "effect_cdfs"
COUNT_KEY = "effect_counts"


def regrid_row(row: np.ndarray, n_samples: int, narrow: int) -> np.ndarray:
    """One padded row -> the row ``to_cdf_matrix`` would have produced at WIDE."""
    stored = np.asarray(row[:narrow], dtype=np.float64)
    q_stored = np.linspace(0.0, 1.0, narrow)
    # Back to sample space. linspace, NOT arange(n)/n — see module docstring.
    pseudo = np.interp(np.linspace(0.0, 1.0, n_samples), q_stored, stored)
    # Forward with to_cdf_matrix's own source grid, so the clamp plateau matches.
    return np.interp(np.linspace(0.0, 1.0, WIDE), np.arange(n_samples) / n_samples, pseudo)


def detect_narrow_width(matrix: np.ndarray, counts: np.ndarray) -> int | None:
    """The width the rows were really gridded at, or None if they look healthy."""
    if not cdf_grid_violations(matrix, counts, label=KEY, max_report=1):
        return None
    first_max = {int(np.argmax(matrix[i])) for i in range(matrix.shape[0])
                 if int(counts[i]) > 0}
    if len(first_max) != 1:
        raise SystemExit(
            f"expected one shared first-max index, found {sorted(first_max)[:8]} — "
            "not the known padding shape; investigate before repairing"
        )
    return first_max.pop() + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(
        Path.home() / ".chorus" / "backgrounds" / "enformer_pertrack.npz"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    path = Path(args.npz)
    if not path.exists():
        raise SystemExit(f"not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}

    matrix, counts = arrays[KEY], arrays[COUNT_KEY]
    if matrix.shape[1] != WIDE:
        raise SystemExit(f"{KEY} is {matrix.shape[1]} wide, expected {WIDE}")

    narrow = detect_narrow_width(matrix, counts)
    if narrow is None:
        print(f"{path.name}: {KEY} already satisfies the grid invariant — nothing to do")
        return 0
    print(f"{path.name}: {KEY} gridded at {narrow}, padded to {WIDE} "
          f"({matrix.shape[0]} rows)")
    print(f"  every percentile scaled by ~{narrow / WIDE:.4f}; "
          f"({(narrow - 1) / WIDE:.4f}, 1.0) unreachable")

    repaired = np.empty_like(matrix, dtype=np.float64)
    for i in range(matrix.shape[0]):
        n = int(counts[i])
        repaired[i] = matrix[i] if n <= 0 else regrid_row(matrix[i], n, narrow)
    repaired = repaired.astype(np.float32)

    problems = cdf_grid_violations(repaired, counts, label=KEY)
    if problems:
        raise SystemExit("repair did not satisfy the invariant:\n  "
                         + "\n  ".join(problems))

    live = [i for i in range(matrix.shape[0]) if int(counts[i]) > 0]
    max_err = max(
        abs(float(repaired[i].max()) - float(matrix[i].max()))
        / max(abs(float(matrix[i].max())), 1e-12) for i in live)
    med_shift = float(np.median([
        abs(float(np.median(repaired[i])) - float(np.median(matrix[i][:narrow])))
        for i in live]))
    print(f"  grid invariant: OK | worst relative max error {max_err:.2e} | "
          f"median-quantile shift {med_shift:.2e}")
    print(f"  monotone rows: {sum(bool(np.all(np.diff(repaired[i]) >= -1e-6)) for i in live)}"
          f"/{len(live)}")

    if args.dry_run:
        print("  --dry-run: not written")
        return 0

    if not args.no_backup:
        backup = path.with_suffix(".npz.prepad")
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"  backup -> {backup.name}")

    arrays[KEY] = repaired
    np.savez_compressed(str(path), **arrays)
    print(f"  written -> {path} ({path.stat().st_size / 1048576:.1f} MB)")
    print("  NOTE: republish to HuggingFace separately; local cache only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
