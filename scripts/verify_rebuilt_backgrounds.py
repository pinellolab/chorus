"""Compare rebuilt backgrounds against their backups, and refuse a bad swap.

Run BEFORE replacing any live file. The rebuild changes three things at once -- the
region set (12,000 -> 18,000 positions, DHS measured and excluded), the contig-margin
fix (12% of anchored positions used to be clamped onto ~40 boundary coordinates), and
retention (effect/summary exact, perbin capped with a derived exact tail) -- so "the
numbers moved" is expected and useless as a check. What must hold is the SHAPE of the
move:

* the body must be essentially unchanged. Reservoir sampling was always unbiased there,
  and the region set only grew, so p50/p90 should sit within a few percent. A large body
  move means something other than the intended change happened.
* the ceiling must not fall. More positions from the same populations can only raise
  max(union), and removing thinning can only raise it further.
* no track may be thinned on an exact layer, and a hybrid layer must keep at least
  MIN_EXACT_TAIL_SLOTS exact grid slots.
* the file must still load and resolve through the real query path -- a background no
  query can reach is what Sei shipped for months.

Exits non-zero on any failure, so a driver cannot swap on a bad build. That mattered:
six builders' merge steps used to log an error and exit 0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Below this, a median over per-track maxima is one or two draws of an extreme order
# statistic and cannot support a pass/fail gate. LegNet has 3 tracks, Sei 40.
MIN_TRACKS_FOR_CEILING_GATE = 25

ORACLES = ["alphagenome", "borzoi", "enformer", "chrombpnet",
           "cherimoya", "sei", "legnet", "epinformerseq"]


def _stats(row: np.ndarray) -> dict:
    a = np.abs(row)
    return {"p50": float(np.quantile(a, .5)), "p90": float(np.quantile(a, .9)),
            "p99": float(np.quantile(a, .99)), "max": float(a.max())}


def _median_ratio(new: np.ndarray, old: np.ndarray, key: str) -> float:
    r = []
    for a, b in zip(old, new):
        sa, sb = _stats(a)[key], _stats(b)[key]
        if sa > 0:
            r.append(sb / sa)
    return float(np.median(r)) if r else float("nan")


def pinning_rate(oracle: str, new: dict, old: dict) -> "tuple[int, int, int] | None":
    """How many REAL committed effects pin against each null. The user-facing measure.

    Distributional ratios say the null got wider; this says whether that translates into
    variants that can be ranked. An effect at or beyond the ceiling reads exactly 1.0 and
    carries no ordering information beyond the exceedance ratio, so the fraction of real
    effects in that state is the thing the rebuild exists to reduce.

    Uses the raw scores already committed in ``examples/**/example_output.json``, which
    are real predictions at real variants and need no GPU to re-score.
    """
    import glob
    import json

    rows: list = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("assay_id") and o.get("raw_score") is not None:
                rows.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for f in glob.glob("examples/**/example_output.json", recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        # NB: the key is `oracle`, not `oracle_name` -- getting that wrong silently
        # matches zero rows and reports a clean 0% either way.
        if d.get("oracle") == oracle:
            walk(d)
    if not rows or "effect_cdfs" not in new or not old or "effect_cdfs" not in old:
        return None

    o_ids = {str(t): i for i, t in enumerate(old["track_ids"])}
    n_ids = {str(t): i for i, t in enumerate(new["track_ids"])}
    o_m, n_m = np.abs(old["effect_cdfs"]), np.abs(new["effect_cdfs"])
    n = pin_old = pin_new = 0
    for r in rows:
        i, j = o_ids.get(r["assay_id"]), n_ids.get(r["assay_id"])
        if i is None or j is None:
            continue
        v = abs(r["raw_score"])
        n += 1
        pin_old += int(v >= o_m[i].max())
        pin_new += int(v >= n_m[j].max())
    return (n, pin_old, pin_new) if n else None


def verify(oracle: str, new_path: Path, old_path: Path, *, strict: bool) -> list[str]:
    from chorus.analysis.background_sampling import (
        MIN_EXACT_TAIL_SLOTS,
        thinning_violations,
    )

    problems: list[str] = []
    if not new_path.exists():
        return [f"{oracle}: rebuilt file missing at {new_path}"]
    if not old_path.exists():
        print(f"  {oracle}: no backup to compare against; structural checks only")

    with np.load(new_path, allow_pickle=True) as d:
        new = {k: d[k] for k in d.files}
    old = None
    if old_path.exists():
        with np.load(old_path, allow_pickle=True) as d:
            old = {k: d[k] for k in d.files}

    n_tracks = len(new["track_ids"])
    print(f"\n=== {oracle} ({n_tracks} tracks) ===")

    # --- structural -------------------------------------------------------
    for layer in ("effect", "summary", "perbin"):
        cdf_key, cnt_key = f"{layer}_cdfs", f"{layer}_counts"
        if cdf_key not in new:
            continue
        m, counts = new[cdf_key], new.get(cnt_key)
        zero = int(np.all(m == 0, axis=1).sum())
        if zero and counts is not None:
            live = int((np.asarray(counts) > 0).sum())
            if live == 0:
                problems.append(f"{oracle}.{layer}: EVERY row is all-zero")
            elif zero > 0.5 * len(m):
                problems.append(
                    f"{oracle}.{layer}: {zero}/{len(m)} rows all-zero")
        nonmono = int((~np.all(np.diff(m, axis=1) >= 0, axis=1)).sum())
        if nonmono:
            problems.append(f"{oracle}.{layer}: {nonmono} non-monotone rows")

        # --- retention ----------------------------------------------------
        ret = new.get(f"{layer}_retained")
        if ret is None:
            msg = (f"{oracle}.{layer}: no {layer}_retained recorded -- thinning is "
                   f"not checkable from this artefact")
            if strict:
                problems.append(msg)
            else:
                print("  WARN " + msg)
        else:
            offered = np.asarray(new[cnt_key])
            retained = np.asarray(ret)
            thinned = int((retained < offered).sum())
            tail_k = int(new[f"{layer}_tail_k"]) if f"{layer}_tail_k" in new else 0
            if thinned and not tail_k:
                problems.append(
                    f"{oracle}.{layer}: {thinned}/{n_tracks} tracks thinned with no "
                    f"exact tail -- their ceilings are draws from a subsample")
            elif thinned:
                slots = int(min(tail_k, offered.max()) * m.shape[1] // offered.max())
                # DELEGATE the pass/fail to thinning_violations rather than re-deriving
                # it. This block used to compare `slots >= MIN_EXACT_TAIL_SLOTS` directly
                # while thinning_violations applies a documented 1%-of-floor tolerance for
                # estimation error in n_expected, so the two disagreed on AlphaGenome's
                # perbin: the builder accepted 199 slots and the verifier refused them.
                # Two implementations of one rule is the #144 shape, and it appeared here
                # in code written to catch exactly that class of defect.
                probs = thinning_violations(
                    offered, retained, n_points=m.shape[1], tail_k=tail_k,
                    label=f"{oracle}.{layer}", max_report=2)
                status = "ok" if not probs else "TOO FEW"
                print(f"  {layer}: {thinned}/{n_tracks} thinned, exact tail k={tail_k} "
                      f"-> {slots} exact grid slots ({status}"
                      + (f", intent {MIN_EXACT_TAIL_SLOTS}" if slots < MIN_EXACT_TAIL_SLOTS
                         else "") + ")")
                problems += probs
            else:
                print(f"  {layer}: exact retention, 0 thinned")

    # --- per-row arrays must not be LOST relative to the file being replaced ---
    # layers_per_row went missing from all three rebuilt gene-anchored oracles: the effect
    # interim carried it, build_and_save forwards only its canonical keys, and
    # merge_to_final did not pass it. Every other gate passed. The existing guard test did
    # not fire because it reads the LIVE backgrounds, which still had the field from the
    # previous build -- a guard that cannot see the artefact under test.
    if old is not None:
        n_tracks_old = len(old["track_ids"])
        for key, arr in old.items():
            if key in ("track_ids",) or not hasattr(arr, "shape"):
                continue
            if arr.ndim != 1 or arr.shape[0] != n_tracks_old:
                continue                      # not a per-row array
            if key.endswith(("_counts", "_retained")) or key == "signed_flags":
                continue                      # covered by their own checks
            if key not in new:
                problems.append(
                    f"{oracle}: per-row array {key!r} present in the file being replaced "
                    f"but ABSENT from the rebuild -- downstream code keying on it breaks "
                    f"silently (compose_layers exits, per-layer analysis raises)")

        # And FILE-LEVEL arrays, which the loop above cannot see: it selects on
        # `shape[0] == n_tracks`, and build_config has shape (1,). That hole let all three
        # rebuilt oracles ship with NO provenance at all -- the same "lost a field the old
        # file had" defect as layers_per_row, one shape away from the check written for it.
        for key in old:
            if key in new or key in ("track_ids",):
                continue
            arr = old[key]
            if not hasattr(arr, "shape") or (arr.ndim == 1 and arr.shape[0] == n_tracks_old):
                continue                      # per-row: handled above
            problems.append(
                f"{oracle}: file-level array {key!r} present in the file being replaced "
                f"but ABSENT from the rebuild"
                + (" -- provenance: which regions, which formula, which genome, which "
                   "builder commit. Without it the reference class is unrecoverable from "
                   "the artefact." if key == "build_config" else ""))

    # --- distributional, vs the backup ------------------------------------
    if old is not None and "effect_cdfs" in new and "effect_cdfs" in old:
        o_ids = [str(x) for x in old["track_ids"]]
        n_ids = [str(x) for x in new["track_ids"]]
        if set(o_ids) != set(n_ids):
            missing, added = set(o_ids) - set(n_ids), set(n_ids) - set(o_ids)
            problems.append(
                f"{oracle}: the track SET changed -- {len(missing)} lost "
                f"{sorted(missing)[:3]}, {len(added)} gained {sorted(added)[:3]}")
        else:
            if o_ids != n_ids:
                # Benign for querying: PerTrackNormalizer._resolve_row looks a track up
                # by id, not by row index, so a reordered file answers identically. It
                # is NOT benign for any operation that splices rows BETWEEN files --
                # apply_effect_rebuild.py rightly refuses on a reorder, because
                # carrying a per-row array across would misalign every row.
                # Cherimoya's rebuild emits sorted ids where the shipped file was
                # unsorted; same 1,518 tracks, 1,511 positions moved.
                print(f"  NOTE: track order changed ({sum(a != b for a, b in zip(o_ids, n_ids))}"
                      f" of {n_tracks} positions). Same set, so queries are unaffected, "
                      f"but do NOT splice rows between this file and the old one.")
            # Compare BY ID rather than by position, or a reorder would look like a
            # catastrophic distributional shift.
            order = [o_ids.index(i) for i in n_ids]
            old_aligned = old["effect_cdfs"][order]
            print(f"  {'stat':>6s} {'ratio (new/old)':>16s}")
            for key in ("p50", "p90", "p99", "max"):
                r = _median_ratio(new["effect_cdfs"], old_aligned, key)
                print(f"  {key:>6s} {r:16.3f}")
                if key in ("p50", "p90", "p99") and not (0.5 <= r <= 2.0):
                    problems.append(
                        f"{oracle}: effect {key} moved {r:.2f}x -- the body should be "
                        f"nearly unchanged; a large move means something other than "
                        f"the intended change happened")
                if key == "max" and r < 0.90:
                    # ADVISORY below a track count where the median max is meaningful.
                    #
                    # The ceiling is a single extreme order statistic per track, which
                    # is the instability this whole rebuild exists to reduce -- so
                    # gating on it hard is self-contradictory when there are few tracks
                    # to take a median over. LegNet has 3. Its shipped K562 maximum
                    # (1.2696) recurs EXACTLY in an independent 18,000-position build,
                    # so it is an attainable extreme a build either samples or misses,
                    # not a property that moved.
                    #
                    # And the assumption behind "can only rise" does not hold for this
                    # rebuild: it also REMOVED positions. 12% of anchored positions
                    # used to be clamped onto contig-margin coordinates -- out-of-
                    # distribution windows for a 200 bp promoter model, where erratic
                    # large effects inflate the null's upper body. Dropping them makes
                    # the null more correct and narrower at once.
                    if n_tracks >= MIN_TRACKS_FOR_CEILING_GATE:
                        problems.append(
                            f"{oracle}: effect ceiling fell to {r:.2f}x across "
                            f"{n_tracks} tracks. Adding positions and removing thinning "
                            f"raise it; only dropping positions lowers it, so check "
                            f"what was dropped")
                    else:
                        print(f"  ADVISORY: ceiling {r:.2f}x on only {n_tracks} "
                              f"tracks -- too few for a median max to mean anything; "
                              f"judge this oracle on p50/p90/p99")
    got = pinning_rate(oracle, new, old) if old is not None else None
    if got:
        n, po, pn = got
        print(f"  pinned on {n} committed effects: {po} ({po / n:.1%}) -> "
              f"{pn} ({pn / n:.1%})")
        if pn > po:
            problems.append(
                f"{oracle}: MORE real effects pin than before ({po} -> {pn} of {n}). "
                f"A wider null cannot increase pinning, so the ceiling moved the wrong "
                f"way for the tracks that matter")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", default="/data/chorus_data/rebuild_2026-08-06")
    ap.add_argument("--backups", default="/data/chorus_data/pre_unified_rebuild")
    ap.add_argument("--oracles", nargs="*", default=None)
    ap.add_argument("--strict-retention", action="store_true",
                    help="Treat a missing *_retained array as a failure rather than a "
                         "warning. Off by default so an oracle rebuilt before the "
                         "field existed can still be verified.")
    args = ap.parse_args()

    staged, backups = Path(args.staged), Path(args.backups)
    todo = args.oracles or [o for o in ORACLES
                            if (staged / f"{o}_pertrack.npz").exists()]
    if not todo:
        print(f"no rebuilt backgrounds found in {staged}")
        return 1

    all_problems: list[str] = []
    for o in todo:
        all_problems += verify(o, staged / f"{o}_pertrack.npz",
                               backups / f"{o}_pertrack.npz",
                               strict=args.strict_retention)

    print(f"\n{'=' * 68}")
    if all_problems:
        print(f"REFUSING THE SWAP -- {len(all_problems)} problem(s):")
        for p in all_problems:
            print(f"  - {p}")
        return 1
    print(f"all {len(todo)} rebuilt background(s) pass: {', '.join(todo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
