"""Swap a freshly built effect null into a shipped background, and nothing else.

The 2026-08-05 rebuild changed only the EFFECT reference population — which genomic
positions the variant-effect null is drawn from. The baseline/summary and per-bin
passes are untouched, and re-running them would burn GPU to reproduce identical
numbers while risking drift from any unrelated change in the meantime.

Three of the six rebuilt oracles (sei, legnet, epinformerseq) no longer have a
baseline interim on disk at all — theirs were built in an earlier session and only the
merged ``*_pertrack.npz`` survives. So a full ``--part merge`` is not even available
for them, and the surgical swap is the only correct option rather than a shortcut.

What it refuses to do:

* proceed when the interim's ``track_ids`` differ from the shipped file's, in content
  OR order. Row *i* of ``effect_cdfs`` means "the track at index *i* of
  ``track_ids``", so a reordering silently reassigns every null to the wrong track;
* proceed when the new matrix fails ``cdf_grid_violations`` — the guard that caught
  the padded Enformer grid (#143);
* overwrite in place. It writes a sibling, verifies the sibling loads and matches,
  and only then replaces, so an interrupted run cannot leave a half-written 500 MB
  artefact.

Everything not being replaced is copied through verbatim, including ``summary_cdfs``,
``perbin_cdfs``, their counts, ``signed_flags``, and any provenance already present.

Usage:
  python scripts/apply_effect_rebuild.py --oracles enformer borzoi sei legnet
  python scripts/apply_effect_rebuild.py --oracles enformer --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Resolved through the data-dir mechanism, not hardcoded to $HOME. Every
# background-handling script had this literal; CHORUS_BACKGROUNDS_DIR applies
# the legacy ~/.chorus compatibility itself, per kind.
from chorus.core.globals import CHORUS_BACKGROUNDS_DIR
BG = CHORUS_BACKGROUNDS_DIR
BACKUP = Path("/data/chorus_data/pre_effect_rebuild")

# Which region set each oracle's new effect null was drawn from, for provenance.
# Not inferred: written down per oracle, because they genuinely differ and a wrong
# value here is worse than an absent one.
REGION_SET = {
    "alphagenome": "gene-anchored+ccre",
    "enformer": "gene-anchored+ccre",
    "borzoi": "gene-anchored+ccre",
    "sei": "gene-anchored+ccre",
    "epinformerseq": "gene-anchored+ccre",
    "legnet": "promoter-anchored",
}


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()


def apply_one(oracle: str, *, dry_run: bool) -> bool:
    from chorus.analysis.background_sampling import cdf_grid_violations
    from chorus.utils.annotations import (
        DEFAULT_REGION_STRATA, PROMOTER_REGION_STRATA,
    )

    final = BG / f"{oracle}_pertrack.npz"
    interim = BG / f"{oracle}_effect_cdfs_interim.npz"
    if not final.exists():
        print(f"  {oracle}: SKIP, no shipped background"); return False
    if not interim.exists():
        print(f"  {oracle}: SKIP, no effect interim"); return False

    with np.load(final, allow_pickle=True) as f:
        payload = {k: f[k] for k in f.files}
    with np.load(interim, allow_pickle=True) as d:
        new_ids = [str(x) for x in d["track_ids"]]
        new_cdf = d["effect_cdfs"]
        new_cnt = d["effect_counts"]
        new_layers = ([str(x) for x in d["layers_per_row"]]
                      if "layers_per_row" in d.files else None)

    old_ids = [str(x) for x in payload["track_ids"]]
    if new_ids != old_ids:
        only_new = set(new_ids) - set(old_ids)
        only_old = set(old_ids) - set(new_ids)
        same_set = not only_new and not only_old
        print(f"  {oracle}: REFUSED — track_ids differ "
              f"({'same set, DIFFERENT ORDER' if same_set else f'{len(only_new)} new, {len(only_old)} missing'}). "
              f"Row i of effect_cdfs means the track at index i, so this would "
              f"reassign every null to the wrong track.")
        return False

    n = len(old_ids)
    if new_cdf.shape[0] != n:
        print(f"  {oracle}: REFUSED — interim has {new_cdf.shape[0]} rows, "
              f"shipped has {n}")
        return False

    problems = cdf_grid_violations(new_cdf, new_cnt, label=f"{oracle}.effect_cdfs")
    if problems:
        print(f"  {oracle}: REFUSED — grid guard: {problems[0][:150]}")
        return False

    old_cnt = payload.get("effect_counts")
    old_w = payload["effect_cdfs"].shape[1]
    print(f"  {oracle}: {n} tracks | samples/track "
          f"{int(np.max(old_cnt)) if old_cnt is not None else '?'} -> {int(np.max(new_cnt))}"
          f" | width {old_w} -> {new_cdf.shape[1]}")

    strata = (PROMOTER_REGION_STRATA if REGION_SET[oracle] == "promoter-anchored"
              else DEFAULT_REGION_STRATA)
    prov = {}
    if "build_config" in payload:
        try:
            from chorus.analysis.normalization import PerTrackNormalizer
            prov = PerTrackNormalizer._read_build_config(
                payload["build_config"], oracle) or {}
        except Exception:
            prov = {}
    prov.update({
        "schema_version": 3,
        "oracle": oracle,
        "effect_region_set": REGION_SET[oracle],
        "effect_region_strata": dict(strata),
        "effect_samples_per_track": [int(np.min(new_cnt)), int(np.max(new_cnt))],
        "effect_rebuild_commit": _git_sha(),
        # Only the effect rows moved. Saying so explicitly stops a reader assuming
        # the activity percentile was rebuilt too.
        "effect_rebuild_note": (
            "Only effect_cdfs/effect_counts were rebuilt on 2026-08-05. "
            "summary_cdfs, perbin_cdfs and their counts are carried over unchanged "
            "from the previous build."
        ),
    })

    payload["effect_cdfs"] = new_cdf
    payload["effect_counts"] = new_cnt
    if new_layers is not None:
        payload["layers_per_row"] = np.array(new_layers, dtype="U")
    payload["build_config"] = np.array([json.dumps(prov, sort_keys=True, default=str)])

    if dry_run:
        print(f"      (dry run) would write {len(payload)} arrays")
        return True

    BACKUP.mkdir(parents=True, exist_ok=True)
    backup = BACKUP / final.name
    if not backup.exists():
        backup.write_bytes(final.read_bytes())
        print(f"      backed up -> {backup}")

    tmp = final.with_name(final.name.replace(".npz", ".applying.npz"))
    np.savez_compressed(tmp, **payload)
    with np.load(tmp, allow_pickle=True) as chk:
        assert [str(x) for x in chk["track_ids"]] == old_ids
        assert np.array_equal(chk["effect_cdfs"], new_cdf)
        assert np.array_equal(chk["effect_counts"], new_cnt)
        for key in ("summary_cdfs", "perbin_cdfs"):
            if key in payload:
                assert np.array_equal(chk[key], payload[key]), key
    tmp.replace(final)
    print(f"      wrote {final.name} ({final.stat().st_size / 1e6:.1f} MB)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracles", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    unknown = [o for o in args.oracles if o not in REGION_SET]
    if unknown:
        raise SystemExit(f"no region set recorded for {unknown}; add it to REGION_SET "
                         f"rather than letting the provenance be guessed")
    ok = sum(apply_one(o, dry_run=args.dry_run) for o in args.oracles)
    print(f"\napplied {ok}/{len(args.oracles)}")
    return 0 if ok == len(args.oracles) else 1


if __name__ == "__main__":
    raise SystemExit(main())
