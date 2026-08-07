"""Union position shards into one effect interim, and compose per-layer nulls.

Two jobs, both of which have to be exact.

**1. Union the shards.** AlphaGenome, Borzoi and Enformer emit every track from one
forward pass, so a build is split across GPUs by *position*, not by track. Each shard
therefore holds a partial reservoir for EVERY track. Pooling the shards'
10,000-point CDF grids would approximate the unsharded result; instead each shard
writes raw samples and the CDF is built once here, from the union.
``tests/test_position_sharding.py`` pins that this is bit-identical to an unsharded
build for 2, 3, 4 and 8-way splits.

**2. Compose per layer.** Measured over every committed walkthrough row, the *peak*
layers saturate against a gene-anchored null and nothing else does:

    enformer      chromatin_accessibility   50 % of rows above the null max
    alphagenome   histone_marks             30 %
    enformer      tf_binding                25 %
    alphagenome   tss_activity               8 %
    enformer      tss_activity               0 %

because most gene-anchored positions are not inside a peak, and a variant in closed
chromatin cannot move an accessibility or ChIP signal much. A cCRE-anchored null is
1.2-1.6x wider at every fixed quantile (p50 through p99.9), which is the fix. CAGE
needs nothing: the shipped mixture already behaves like a "+/-5 kb of a TSS" null for
it, measured across eight distance scales.

So the shipped file wants **different reference populations for different rows** —
peak-layer rows from the cCRE build, everything else from the gene-anchored build.
That is not a schema change: AlphaGenome already ships three distinct summary
reference classes and two effect classes inside single matrices.

Beware the confound this measurement walked into first: the null *maximum* grows with
sample count, so a small cCRE probe appears to have a NARROWER tail than a large
gene-anchored build even when its distribution is genuinely wider. Compare at fixed
quantiles, or at equal position counts. This script refuses a compose whose two
sources have materially different counts, for exactly that reason.

Usage:
  python scripts/merge_effect_shards.py --oracle enformer --shards 8
  python scripts/merge_effect_shards.py --oracle enformer --compose-layers
"""
from __future__ import annotations

import argparse
import json
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
GENE_ANCHORED_BACKUP = Path("/data/chorus_data/interims_gene_anchored")
# A count ratio outside this band means the two sources are not comparable and the
# composed file would mix a well-estimated tail with a poorly-estimated one.
_MAX_COUNT_RATIO = 1.15


# Overridable so a STAGED rebuild can be merged without reading or writing the live
# directory. Without this the union would have read the 8 stale `.shard*of8.npz` files
# left in the live dir by the 2026-08-05 build and written its output over the live
# interim -- reading the wrong inputs and mutating data that is deliberately untouched
# until the swap.
_DIR: "Path | None" = None


def _effect_path(oracle: str, suffix: str = "") -> Path:
    return (_DIR or BG) / f"{oracle}_effect_cdfs_interim{suffix}.npz"


def union_shards(oracle: str, n_shards: int, n_points: int,
                 exact: bool = True) -> Path:
    from chorus.analysis.background_sampling import (
        DEFAULT_CAPACITY,
        ReservoirSampler,
    )

    parts, ids, flags, layers = [], None, None, None
    for k in range(n_shards):
        p = _effect_path(oracle, f".shard{k}of{n_shards}")
        if not p.exists():
            raise SystemExit(f"missing shard {k}: {p}")
        with np.load(p, allow_pickle=False) as d:
            missing = {"values", "offsets", "counts", "n_tracks"} - set(d.files)
            if missing:
                raise SystemExit(
                    f"{p.name} has no raw samples (missing {sorted(missing)}). It was "
                    f"written by a pre-sharding builder, or without --shard."
                )
            parts.append({k2: d[k2] for k2 in ("values", "offsets", "counts", "n_tracks")})
            shard_ids = [str(x) for x in d["track_ids"]]
            shard_flags = d["signed_flags"] if "signed_flags" in d.files else None
            shard_layers = (d["layers_per_row"] if "layers_per_row" in d.files
                            else None)
        if ids is None:
            ids, flags = shard_ids, shard_flags
            layers = shard_layers
        elif shard_ids != ids:
            raise SystemExit(f"shard {k} track_ids disagree with shard 0")
        print(f"  shard {k}: {int(np.asarray(parts[-1]['counts']).max())} samples/track")

    # capacity=None keeps EVERY value. The previous call passed no capacity at all
    # and silently inherited DEFAULT_CAPACITY=50,000, which thinned every
    # AlphaGenome RNA track's 148,367 samples down to a 50,000 uniform subsample and
    # understated its ceiling by a median 1.33x (up to 8.34x). See
    # ReservoirSampler.from_flat_samples.
    merged = ReservoirSampler.from_flat_samples(
        *parts, capacity=None if exact else DEFAULT_CAPACITY)
    counts = merged.get_counts()
    retained = merged.retained_counts()
    thinned = int((retained < counts).sum())
    print(f"  union: {len(ids)} tracks, counts min={counts.min()} max={counts.max()}, "
          f"{int((counts > 0).sum())} tracks with data")
    print(f"  retention: {'EXACT' if exact else f'capped at {DEFAULT_CAPACITY}'}; "
          f"{thinned} of {len(ids)} tracks thinned "
          f"(retained min={retained.min()} max={retained.max()})")
    if exact and thinned:
        raise SystemExit(
            f"--exact was requested but {thinned} tracks are still thinned; refusing "
            f"to write a background whose ceiling is a subsample"
        )

    matrix = merged.to_cdf_matrix(n_points=n_points)
    from chorus.analysis.background_sampling import (
        cdf_grid_violations,
        thinning_violations,
    )
    problems = cdf_grid_violations(matrix, counts, label=f"{oracle}.effect_cdfs")
    problems += thinning_violations(counts, retained, n_points=n_points,
                                    label=f"{oracle}.effect_cdfs")
    if problems:
        raise SystemExit("refusing to write: " + "\n".join(problems[:3]))

    out = _effect_path(oracle)
    payload = dict(track_ids=np.array(ids, dtype="U"),
                   effect_cdfs=matrix.astype(np.float32),
                   effect_counts=counts,
                   # Retention alongside the offered count, so "was this thinned?" is
                   # answerable from the file. Its absence is exactly why the
                   # AlphaGenome thinning was invisible: only `counts` (offered) was
                   # ever written, and offered == retained is the thing that matters.
                   effect_retained=retained)
    if flags is not None:
        payload["signed_flags"] = flags
    if layers is not None:
        # Carried through from the shards. Dropping it here is exactly what made
        # every rebuilt background ship WITHOUT a per-row layer, while the guard
        # test skipped itself because the field was absent -- a guard that protects
        # nothing. The union is the only place the field can be lost.
        payload["layers_per_row"] = np.asarray(layers)
    np.savez_compressed(out, **payload)
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)"
          + ("" if layers is not None else "  [WARNING: no layers_per_row]"))
    return out


def compose_layers(oracle: str) -> Path:
    """Peak-layer rows from the cCRE interim, all other rows from gene-anchored."""
    from chorus.utils.annotations import CCRE_ANCHORED_LAYERS

    ccre_path = _effect_path(oracle)
    gene_path = GENE_ANCHORED_BACKUP / ccre_path.name
    for p in (ccre_path, gene_path):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    with np.load(ccre_path, allow_pickle=False) as c, np.load(gene_path, allow_pickle=False) as g:
        c_ids = [str(x) for x in c["track_ids"]]
        g_ids = [str(x) for x in g["track_ids"]]
        if c_ids != g_ids:
            raise SystemExit("track_ids differ between the two interims; refusing")
        c_cdf, g_cdf = c["effect_cdfs"], g["effect_cdfs"]
        c_cnt, g_cnt = c["effect_counts"], g["effect_counts"]
        flags = g["signed_flags"] if "signed_flags" in g.files else None
        # The per-row layer the BUILDER wrote, from the same field it used to pick
        # each track's window. Not re-derived here: classify_track_layer needs a
        # track object, and re-deriving a layer from an opaque id (Enformer ships
        # ENCFF accessions, Borzoi FANTOM ones) is how builder and query came to
        # disagree in the first place (#122).
        if "layers_per_row" not in c.files:
            raise SystemExit(
                f"{ccre_path.name} has no layers_per_row. Rebuild it with the current "
                f"builder -- composing without a per-row layer would mean guessing "
                f"which rows are peak layers."
            )
        layers = [str(x) for x in c["layers_per_row"]]

    cmax, gmax = int(np.max(c_cnt)), int(np.max(g_cnt))
    ratio = max(cmax, gmax) / max(min(cmax, gmax), 1)
    if ratio > _MAX_COUNT_RATIO:
        raise SystemExit(
            f"refusing to compose: cCRE build has {cmax} samples/track and the "
            f"gene-anchored build {gmax} ({ratio:.2f}x apart). The null MAXIMUM grows "
            f"with sample count, so mixing rows estimated from different numbers of "
            f"positions would make some layers' upper tails artificially short -- the "
            f"exact confound that made a 400-position cCRE probe look narrower than a "
            f"5,949-position gene-anchored build. Re-run the smaller side at the same "
            f"--n-variants."
        )

    out_cdf, out_cnt = g_cdf.copy(), g_cnt.copy()
    taken = {}
    for i, layer in enumerate(layers):
        if layer in CCRE_ANCHORED_LAYERS:
            out_cdf[i] = c_cdf[i]
            out_cnt[i] = c_cnt[i]
            taken[layer] = taken.get(layer, 0) + 1
    print(f"  composed {sum(taken.values())} of {len(g_ids)} rows from the cCRE build:")
    for layer in sorted(taken):
        print(f"    {layer:26} {taken[layer]:5d} rows")
    print(f"  remaining {len(g_ids) - sum(taken.values())} rows stay gene-anchored")

    payload = dict(track_ids=np.array(g_ids, dtype="U"),
                   effect_cdfs=out_cdf, effect_counts=out_cnt)
    if flags is not None:
        payload["signed_flags"] = flags
    # Record WHICH rows came from where, per row, so the composition is recoverable
    # from the artefact rather than only from this script.
    payload["effect_region_set_per_row"] = np.array(
        ["ccre" if l in CCRE_ANCHORED_LAYERS else "gene-anchored" for l in layers],
        dtype="U")
    payload["layers_per_row"] = np.array(layers, dtype="U")
    out = _effect_path(oracle)
    np.savez_compressed(out, **payload)
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True,
                    choices=["enformer", "borzoi", "alphagenome"])
    ap.add_argument("--shards", type=int, default=None,
                    help="Union this many position shards into one effect interim.")
    ap.add_argument("--compose-layers", action="store_true",
                    help="Take peak-layer rows from the cCRE interim and the rest "
                         "from the backed-up gene-anchored interim.")
    ap.add_argument("--n-points", type=int, default=10_000)
    ap.add_argument("--dir", default=None,
                    help="Directory holding the shards, and where the union is written. "
                         "Defaults to the live CHORUS_BACKGROUNDS_DIR. Point it at a "
                         "staging directory when merging a rebuild that has not been "
                         "swapped in -- otherwise stale shards from an earlier build "
                         "with a different shard count are silently in scope.")
    ap.add_argument("--capped", action="store_true",
                    help="Subsample the union to DEFAULT_CAPACITY instead of keeping "
                         "every value. This reproduces the pre-2026-08-06 behaviour "
                         "that thinned AlphaGenome's RNA ceilings; it exists only so "
                         "the defect can be reproduced in a test.")
    args = ap.parse_args()
    global _DIR
    if args.dir:
        _DIR = Path(args.dir)
        print(f"[shards] reading and writing {_DIR}")

    if args.shards:
        print(f"[{args.oracle}] unioning {args.shards} position shards")
        union_shards(args.oracle, args.shards, args.n_points,
                     exact=not args.capped)
    if args.compose_layers:
        print(f"[{args.oracle}] composing per-layer reference sets")
        compose_layers(args.oracle)
    if not (args.shards or args.compose_layers):
        ap.error("pass --shards N and/or --compose-layers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
