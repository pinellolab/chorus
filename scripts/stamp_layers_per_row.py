#!/usr/bin/env python
"""Add ``layers_per_row`` to the backgrounds that ship without it.

Three oracles carry the array (AlphaGenome 6 distinct layers, Borzoi 5, Enformer 4) because
their builders construct ``track_info`` with a ``'layer'`` key and write
``canonical_layer(t['layer'])`` per row. The other five never had that concept in their
builders at all, so the array was simply absent and ``layers_present`` was ``null``.

For four of the five that costs nothing — they are single-layer, so the array carries no
information. **ChromBPNet is not single-layer**: its 753 rows are ATAC (4), DNASE (5) and
CHIP (744), i.e. accessibility *and* TF binding, and code that keys on ``layers_per_row`` had
to re-derive the layer from the track-id string instead.

The layer is not invented here. Every value is produced by the same
:func:`chorus.analysis.scorers.classify_track_layer` the query path calls, on a shim carrying
the assay_type and assay_id that oracle emits — and then asserted equal to it. A stamped
array that disagreed with what the query path computes would be worse than no array, because
downstream code trusts it.

What has to be supplied per oracle is the **assay_type**, because the track ids do not all
carry it:

    chrombpnet      from the id prefix: ATAC / DNASE / CHIP
                    (CHIP splits TF vs histone via classify_chip_layer on the id)
    cherimoya       from the id prefix: ATAC / DNASE
    epinformerseq   from the id prefix: Enhancer_DNase / Enhancer_H3K27ac / …_DNase
    sei             constant 'sequence-class'   — ids are 'CA#PC1@…@Polycomb-repressed@0'
    legnet          constant 'LentiMPRA'        — ids are bare, e.g. 'K562'

Appends in place. No rebuild, no model, no GPU: the CDF matrices are untouched, so every
percentile this changes is none of them.

    python scripts/stamp_layers_per_row.py --dry-run
    python scripts/stamp_layers_per_row.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chorus.analysis.scorers import (  # noqa: E402
    LAYER_CONFIGS,
    canonical_layer,
    classify_track_layer,
)
from chorus.core.globals import CHORUS_DATA_DIR, resolve_backgrounds_dir  # noqa: E402

#: Oracles whose track ids carry the assay_type as the first colon-delimited field.
_PREFIXED = {"chrombpnet", "cherimoya", "epinformerseq"}

#: Oracles whose ids carry no assay_type, with the one their oracle emits.
_CONSTANT_ASSAY_TYPE = {
    "sei": "sequence-class",
    "legnet": "LentiMPRA",
}

TARGETS = sorted(_PREFIXED | set(_CONSTANT_ASSAY_TYPE))


class _Shim:
    """Minimal stand-in for an OraclePredictionTrack.

    ``classify_track_layer`` reads ``assay_type``, ``assay_id`` and ``metadata`` only, so
    this is enough to get the query path's own answer without loading a model.
    """

    def __init__(self, assay_type: str, assay_id: str):
        self.assay_type = assay_type
        self.assay_id = assay_id
        self.metadata: dict = {}


def assay_type_for(oracle: str, track_id: str) -> str:
    # Sei carries TWO id kinds now that its nulls cover every track it predicts: 21,907 chromatin
    # profiles `TA#{celltype}@{assay}@{id}` and the 40 projected classes `CA#...`. The old constant
    # "sequence-class" was right when only the classes had rows and would now mislabel every profile,
    # sending 21,907 of them to a layer whose scorer config does not apply. Mirrors what the oracle
    # itself assigns in SeiOracle._assemble_prediction (info.assay for targets, "sequence-class" for
    # classes).
    if oracle == "sei":
        if track_id.startswith("TA#"):
            parts = track_id[len("TA#"):].split("@")
            return parts[1] if len(parts) > 1 else ""
        return "sequence-class"
    if oracle in _CONSTANT_ASSAY_TYPE:
        return _CONSTANT_ASSAY_TYPE[oracle]
    return track_id.split(":")[0]


def layers_for(oracle: str, track_ids: list[str]) -> list[str]:
    """Ask the query path, then check the answer is a canonical layer."""
    out = []
    for tid in track_ids:
        assay_type = assay_type_for(oracle, tid)
        layer = classify_track_layer(_Shim(assay_type, tid))
        if layer == "other" or layer not in LAYER_CONFIGS:
            raise SystemExit(
                f"{oracle}: track {tid!r} (assay_type {assay_type!r}) classifies as "
                f"{layer!r}, which is not a LAYER_CONFIGS key. Stamping that would put a "
                f"value in layers_per_row that the query path cannot score against. Fix "
                f"the classification first -- this is the Sei-unreachable-rows failure "
                f"mode, not a stamping problem."
            )
        # canonical_layer is idempotent on a layer name; this catches a synonym slipping in.
        assert canonical_layer(layer) == layer, f"{layer!r} is not canonical"
        out.append(layer)
    return out


#: How build_config is stored in every shipped artefact. AlphaGenome, Borzoi and Enformer all
#: use shape (1,), and readers index it as ``data["build_config"][0]``.
_BUILD_CONFIG_SHAPE = (1,)


def _read_build_config(raw) -> dict:
    """Tolerate either storage shape on the way in."""
    text = raw.item() if getattr(raw, "shape", None) == () else raw[0]
    return json.loads(text) if isinstance(text, str) else text


def _rewrite_build_config(raw, *, layers_present=None, stamped_by=None):
    """Rewrite build_config, preserving the (1,) storage shape.

    The first version of this script wrote ``np.array(json.dumps(...))``, a 0-d array, where
    every shipped artefact stores shape (1,). Nothing in this script noticed, because its own
    reader tolerated both shapes -- but
    ``tests/test_cherimoya_ensemble.py::test_the_shipped_null_records_that_it_was_built_from_the_ensemble``
    indexes ``[0]`` and failed with "too many indices for array: array is 0-dimensional".
    A format change that only breaks *other* readers is exactly what a uniform storage shape
    is for, so this normalises rather than merely tolerating.
    """
    config = _read_build_config(raw)
    if layers_present is not None:
        config["layers_present"] = layers_present
    if stamped_by is not None:
        config["layers_per_row_stamped_by"] = stamped_by
    return np.array([json.dumps(config, sort_keys=True)])


def stamp(path: Path, oracle: str, dry_run: bool) -> bool:
    with np.load(path, allow_pickle=True) as data:
        payload = {k: data[k] for k in data.files}

    if "layers_per_row" in payload:
        raw = payload.get("build_config")
        if raw is not None and getattr(raw, "shape", None) != _BUILD_CONFIG_SHAPE:
            if dry_run:
                print(f"  {oracle:<14} has layers_per_row; build_config shape "
                      f"{raw.shape} needs repair to {_BUILD_CONFIG_SHAPE}")
                return False
            payload["build_config"] = _rewrite_build_config(raw)
            _write(path, payload, oracle)
            print(f"  {oracle:<14} build_config shape repaired to {_BUILD_CONFIG_SHAPE}")
            return True
        print(f"  {oracle:<14} already has layers_per_row -- skipped")
        return False

    track_ids = [str(t) for t in payload["track_ids"]]
    layers = layers_for(oracle, track_ids)
    distinct = sorted(set(layers))
    counts = {layer: layers.count(layer) for layer in distinct}
    print(f"  {oracle:<14} {len(layers):>5} rows -> {counts}")

    if dry_run:
        return False

    payload["layers_per_row"] = np.array(layers, dtype="U")

    # build_config.layers_present was null for exactly these oracles.
    raw = payload.get("build_config")
    if raw is not None:
        payload["build_config"] = _rewrite_build_config(
            raw, layers_present=distinct,
            stamped_by="scripts/stamp_layers_per_row.py",
        )

    _write(path, payload, oracle)
    return True


def _write(path: Path, payload: dict, oracle: str) -> None:
    """Atomic replace with read-back verification."""
    # np.savez_compressed appends ".npz" ONLY if the name does not already end in it.
    # Passing a stem that ends in ".npz" therefore writes that exact path -- which on the
    # first run here meant writing straight over the live artefact instead of a temp file,
    # and then failing the read-back because it looked for "<name>.npz.npz". The file
    # survived intact and correct, but by luck rather than by design. Build the temp name
    # so it cannot collide with the target whichever branch numpy takes.
    written = path.with_name(path.stem + ".stamping.npz")
    assert written != path, "temp path collides with the artefact"
    np.savez_compressed(str(written.with_suffix("")), **payload)
    assert written.exists(), f"savez wrote something other than {written}"

    with np.load(written, allow_pickle=True) as check:
        assert set(check.files) == set(payload), "read-back has different keys"
        assert check["build_config"].shape == _BUILD_CONFIG_SHAPE, (
            f"{oracle}: build_config written with shape {check['build_config'].shape}, "
            f"expected {_BUILD_CONFIG_SHAPE}")
        for key in ("track_ids", "layers_per_row", "effect_cdfs", "summary_cdfs",
                    "perbin_cdfs"):
            if key in payload:
                assert np.array_equal(check[key], payload[key]), f"{key} changed"

    os.replace(written, path)
    print(f"                 written and verified ({path.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="derive and print, write nothing")
    ap.add_argument("--backgrounds-dir", default=None)
    args = ap.parse_args()

    bg = Path(args.backgrounds_dir) if args.backgrounds_dir else Path(
        resolve_backgrounds_dir(Path(CHORUS_DATA_DIR)))
    print(f"backgrounds: {bg}\n")

    changed = []
    for oracle in TARGETS:
        path = bg / f"{oracle}_pertrack.npz"
        if not path.exists():
            print(f"  {oracle:<14} not present -- skipped")
            continue
        if stamp(path, oracle, args.dry_run):
            changed.append(oracle)

    print()
    if args.dry_run:
        print("dry run -- nothing written")
        return 0

    # Final state across every artefact, read back from disk.
    print("final state:")
    missing = []
    for path in sorted(bg.glob("*_pertrack.npz")):
        name = path.stem.replace("_pertrack", "")
        with np.load(path, allow_pickle=True) as data:
            has = "layers_per_row" in data.files
            n = len(sorted(set(str(x) for x in data["layers_per_row"]))) if has else 0
        print(f"  {name:<14} layers_per_row={'yes' if has else 'NO '}  distinct={n}")
        if not has:
            missing.append(name)
    if missing:
        print(f"\nstill missing: {missing}")
        return 1
    print(f"\nstamped {len(changed)}: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
