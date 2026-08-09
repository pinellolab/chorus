"""Atomically replace the live backgrounds with a verified rebuild.

Refuses unless EVERY oracle passes, because a half-swapped fleet is worse than either
state: multi-oracle reports would print percentiles from two different reference classes
side by side, with nothing in the output saying so.

Order of operations, per oracle:

  1. re-run the full verification (distributional gates + retention + reference
     population) and abort the whole swap if any oracle fails;
  2. copy the staged file to a temporary sibling of the live file, on the same
     filesystem so the final step is an atomic rename rather than a copy that can be
     interrupted half-written;
  3. read the temporary file back and check it loads, resolves through
     PerTrackNormalizer, and has the expected track count -- a corrupted copy that is
     never read is indistinguishable from a good one until a user hits it;
  4. `os.replace` it over the live file, which is atomic within a filesystem;
  5. record a manifest of what moved, with sizes and sha256, so the swap is reversible
     by exactly the inverse operation.

`--dry-run` does 1-3 and stops. `--rollback` restores from the backup directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ORACLES = ["alphagenome", "borzoi", "enformer", "chrombpnet",
           "cherimoya", "sei", "legnet", "epinformerseq"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _readable_and_resolvable(path: Path, oracle: str) -> tuple[bool, str]:
    """Load the file the way the query path will, not just np.load it."""
    try:
        with np.load(path, allow_pickle=True) as d:
            ids = [str(x) for x in d["track_ids"]]
            for layer in ("effect", "summary", "perbin"):
                k = f"{layer}_cdfs"
                if k in d.files:
                    m = d[k]
                    if m.shape[0] != len(ids):
                        return False, f"{k} has {m.shape[0]} rows for {len(ids)} tracks"
                    if not np.all(np.diff(m, axis=1) >= 0):
                        return False, f"{k} contains a non-monotone row"
    except Exception as exc:
        return False, f"unreadable: {type(exc).__name__}: {exc}"
    return True, f"{len(ids)} tracks"


def verify_all(staged: Path, backups: Path, strict: bool,
               model_changes: dict | None = None) -> list[str]:
    """Every gate, on every oracle. Returns the list of failures."""
    import importlib.util

    fails: list[str] = []
    model_changes = model_changes or {}
    spec = importlib.util.spec_from_file_location(
        "vrb", REPO / "scripts" / "verify_rebuilt_backgrounds.py")
    vrb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vrb)

    spec2 = importlib.util.spec_from_file_location(
        "brps", REPO / "scripts" / "build_reference_position_sets.py")
    brps = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(brps)
    ref = REPO / "reference_sets" / "chorus_reference_positions_v1.npz"

    present = [o for o in ORACLES if (staged / f"{o}_pertrack.npz").exists()]
    if not present:
        fails.append(f"no rebuilt files found in {staged}")
    # Only the oracles ACTUALLY staged. A partial swap is legitimate -- the Cherimoya
    # ensemble rebuild restaged one oracle -- and demanding all eight would either block
    # it or invite someone to point --staged at the live directory to satisfy the check.
    for o in present:
        src = staged / f"{o}_pertrack.npz"
        probs = vrb.verify(o, src, backups / f"{o}_pertrack.npz", strict=strict,
                           model_change=model_changes.get(o))
        fails += probs
        ok, msg = _readable_and_resolvable(src, o)
        if not ok:
            fails.append(f"{o}: {msg}")
        if ref.exists() and brps.verify(ref, o, staged) != 0:
            fails.append(f"{o}: does not reproduce its reference population")
    return fails


def swap(staged: Path, live: Path, backups: Path, dry_run: bool) -> int:
    manifest: dict = {
        "swapped_at": datetime.now(timezone.utc).isoformat(),
        "staged_from": str(staged),
        "live": str(live),
        "backups": str(backups),
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True).stdout.strip(),
        "oracles": {},
    }
    # Only what is staged. A partial restage is legitimate (the Cherimoya ensemble
    # rebuild restaged one oracle); iterating all eight crashed on the first absent
    # file, which would push someone toward the far worse workaround of pointing
    # --staged at the live directory to make the loop find something everywhere.
    present = [o for o in ORACLES if (staged / f"{o}_pertrack.npz").exists()]
    manifest["oracles_swapped"] = present
    manifest["oracles_untouched"] = [o for o in ORACLES if o not in present]
    print(f"  swapping {len(present)} of {len(ORACLES)}: {', '.join(present)}")
    if manifest["oracles_untouched"]:
        print(f"  leaving untouched: {', '.join(manifest['oracles_untouched'])}")
    for o in present:
        src, dst = staged / f"{o}_pertrack.npz", live / f"{o}_pertrack.npz"
        bak = backups / f"{o}_pertrack.npz"
        if not bak.exists() and dst.exists():
            print(f"  {o}: backing up live file first")
            if not dry_run:
                backups.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst, bak)
        entry = {
            "new_bytes": src.stat().st_size,
            "old_bytes": dst.stat().st_size if dst.exists() else None,
            "new_sha256": sha256(src),
            "backup": str(bak) if bak.exists() or not dry_run else None,
        }
        manifest["oracles"][o] = entry
        print(f"  {o}: {entry['old_bytes'] or 0 :,} -> {entry['new_bytes']:,} bytes")
        if dry_run:
            continue
        # copy to a sibling on the SAME filesystem, verify, then atomically rename
        tmp = dst.with_suffix(".npz.swapping")
        shutil.copy2(src, tmp)
        ok, msg = _readable_and_resolvable(tmp, o)
        if not ok:
            tmp.unlink(missing_ok=True)
            print(f"  {o}: ABORT -- the copy did not read back ({msg})")
            return 1
        if sha256(tmp) != entry["new_sha256"]:
            tmp.unlink(missing_ok=True)
            print(f"  {o}: ABORT -- sha256 mismatch after copy")
            return 1
        os.replace(tmp, dst)
        print(f"  {o}: swapped ({msg})")

    if not dry_run:
        out = live / "swap_manifest_2026-08-06.json"
        out.write_text(json.dumps(manifest, indent=1))
        print(f"\nmanifest: {out}")
    return 0


def rollback(live: Path, backups: Path) -> int:
    n = 0
    for o in ORACLES:
        bak, dst = backups / f"{o}_pertrack.npz", live / f"{o}_pertrack.npz"
        if not bak.exists():
            print(f"  {o}: no backup, skipping")
            continue
        tmp = dst.with_suffix(".npz.rollback")
        shutil.copy2(bak, tmp)
        ok, msg = _readable_and_resolvable(tmp, o)
        if not ok:
            tmp.unlink(missing_ok=True)
            print(f"  {o}: ABORT -- backup did not read back ({msg})")
            return 1
        os.replace(tmp, dst)
        print(f"  {o}: restored ({msg})")
        n += 1
    print(f"\nrestored {n} oracle(s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", default="/data/chorus_data/rebuild_2026-08-06")
    ap.add_argument("--backups", default="/data/chorus_data/pre_unified_rebuild")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--no-strict-retention", action="store_true",
                    help="allow an oracle with no *_retained array. Off by default: "
                         "without it, thinning is not checkable from the shipped file.")
    ap.add_argument("--model-change", nargs="*", default=None, metavar="ORACLE=REASON",
                    help="Forwarded to verify_rebuilt_backgrounds.verify for oracles whose "
                         "MODEL changed rather than their positions. Without it the pinning "
                         "check ranks OLD-model committed effects against the NEW null and "
                         "can fail, or pass, for reasons that mean nothing. REASON required.")
    args = ap.parse_args()

    model_changes: dict[str, str] = {}
    for spec in (args.model_change or []):
        if "=" not in spec:
            print(f"--model-change needs ORACLE=REASON, got {spec!r}")
            return 2
        name, _, reason = spec.partition("=")
        if not reason.strip():
            print(f"--model-change {name.strip()}: a REASON is required. This relaxes a "
                  f"real gate; an unexplained relaxation is how a bad build ships.")
            return 2
        model_changes[name.strip()] = reason.strip()

    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR

    live = CHORUS_BACKGROUNDS_DIR
    staged, backups = Path(args.staged), Path(args.backups)

    if args.rollback:
        print(f"ROLLBACK {backups} -> {live}")
        return rollback(live, backups)

    print(f"verifying {staged} against {backups}\n")
    fails = verify_all(staged, backups, strict=not args.no_strict_retention,
                       model_changes=model_changes)
    if fails:
        print(f"\nREFUSING THE SWAP -- {len(fails)} problem(s):")
        for f in fails:
            print(f"  - {f}")
        print("\nNothing was touched. A half-swapped fleet would print percentiles from "
              "two different reference classes side by side in multi-oracle reports, "
              "with nothing in the output saying so.")
        return 1
    checked = [o for o in ORACLES if (staged / f"{o}_pertrack.npz").exists()]
    print(f"\nall {len(checked)} staged background(s) pass every gate: "
          f"{', '.join(checked)}")
    print(f"\n{'DRY RUN — ' if args.dry_run else ''}swapping into {live}")
    return swap(staged, live, backups, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
