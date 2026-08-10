"""Every shipped background must say what layer each of its rows is.

``layers_per_row`` shipped on three oracles of eight. AlphaGenome, Borzoi and Enformer had it
because their builders construct ``track_info`` with a ``'layer'`` key; the other five never
had that concept, so the array was absent and ``build_config.layers_present`` was ``null``.

For four of the five it cost nothing — single-layer oracles, where the array carries no
information. **ChromBPNet is not single-layer**: 753 rows over ATAC (4), DNASE (5) and CHIP
(744), i.e. accessibility *and* TF binding. Code keying on the array had to fall back to
re-deriving the layer from the track-id string, and nothing said so.

It stayed invisible because ``test_canonical_layer_vocabulary.py`` validates the array *when
present* and never requires presence — a shape of test worth naming, because it reads like
coverage and is not. This file requires presence, and requires the values to be the ones the
query path itself computes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration  # reads the shipped artefacts

from chorus.analysis.scorers import LAYER_CONFIGS, canonical_layer  # noqa: E402
from chorus.core.globals import CHORUS_DATA_DIR, resolve_backgrounds_dir  # noqa: E402

EXPECTED_ORACLES = {
    "alphagenome", "borzoi", "cherimoya", "chrombpnet",
    "enformer", "epinformerseq", "legnet", "sei",
}


def _backgrounds() -> list[Path]:
    bg = Path(resolve_backgrounds_dir(Path(CHORUS_DATA_DIR)))
    return sorted(bg.glob("*_pertrack.npz"))


def _oracle(path: Path) -> str:
    return path.stem.replace("_pertrack", "")


@pytest.mark.parametrize("path", _backgrounds() or [Path("none")], ids=lambda p: p.name)
def test_the_artefact_declares_a_layer_for_every_row(path: Path):
    if not path.exists():
        pytest.skip("no backgrounds downloaded")
    with np.load(path, allow_pickle=True) as data:
        assert "layers_per_row" in data.files, (
            f"{_oracle(path)} ships no layers_per_row, so a consumer has to re-derive each "
            f"row's layer from its track-id string. That is a silent dependency on id "
            f"formatting, and it is wrong for any oracle spanning more than one layer -- "
            f"ChromBPNet spans two. Run scripts/stamp_layers_per_row.py."
        )
        layers = [str(x) for x in data["layers_per_row"]]
        ids = [str(t) for t in data["track_ids"]]

    assert len(layers) == len(ids), (
        f"{_oracle(path)}: layers_per_row has {len(layers)} entries for {len(ids)} tracks; "
        f"a per-row array that is not per-row is worse than none, because it still looks "
        f"indexable"
    )

    non_canonical = sorted({x for x in layers if x not in LAYER_CONFIGS})
    assert not non_canonical, (
        f"{_oracle(path)}: layers_per_row holds {non_canonical}, which are not "
        f"LAYER_CONFIGS keys, so nothing downstream can score against them"
    )
    assert "other" not in layers, (
        f"{_oracle(path)} has rows stamped 'other'. That is the Sei failure mode -- 40 rows "
        f"shipped that no query could reach -- recorded in the artefact this time rather "
        f"than discovered later."
    )
    assert all(canonical_layer(x) == x for x in set(layers))


@pytest.mark.parametrize("path", _backgrounds() or [Path("none")], ids=lambda p: p.name)
def test_build_config_agrees_with_the_per_row_array(path: Path):
    """``layers_present`` is a summary of ``layers_per_row``; a disagreement means one of
    them was written without the other."""
    if not path.exists():
        pytest.skip("no backgrounds downloaded")
    with np.load(path, allow_pickle=True) as data:
        if "layers_per_row" not in data.files or "build_config" not in data.files:
            pytest.skip("nothing to cross-check")
        layers = sorted({str(x) for x in data["layers_per_row"]})
        raw = data["build_config"]
        text = raw.item() if raw.shape == () else raw[0]
        config = json.loads(text) if isinstance(text, str) else text

    declared = config.get("layers_present")
    if declared is None:
        pytest.fail(
            f"{_oracle(path)} has layers_per_row but build_config.layers_present is null; "
            f"anything reading the config alone concludes the layers are unknown"
        )
    assert sorted(declared) == layers, (
        f"{_oracle(path)}: build_config.layers_present is {sorted(declared)} but "
        f"layers_per_row actually holds {layers}"
    )


def test_the_stamped_layers_are_what_the_query_path_computes():
    """The load-bearing check: the stored value must equal ``classify_track_layer``.

    A stamped array that disagreed with the query path would be worse than no array, since
    downstream code trusts it. Covers the five oracles whose assay_type is derivable from
    the id (or is a per-oracle constant) — the derivation the stamper used.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    stamp = pytest.importorskip(
        "stamp_layers_per_row", reason="scripts/stamp_layers_per_row.py not importable",
    )

    checked = 0
    for path in _backgrounds():
        oracle = _oracle(path)
        if oracle not in stamp.TARGETS:
            continue
        with np.load(path, allow_pickle=True) as data:
            if "layers_per_row" not in data.files:
                pytest.fail(f"{oracle} has no layers_per_row")
            stored = [str(x) for x in data["layers_per_row"]]
            ids = [str(t) for t in data["track_ids"]]
        recomputed = stamp.layers_for(oracle, ids)
        mismatches = [
            (i, ids[i], stored[i], recomputed[i])
            for i in range(len(ids)) if stored[i] != recomputed[i]
        ]
        assert not mismatches, (
            f"{oracle}: {len(mismatches)} rows where the stored layer differs from what "
            f"classify_track_layer computes, e.g. {mismatches[:3]}"
        )
        checked += len(ids)
    assert checked, "no stamped oracle was checked -- has TARGETS changed?"


def test_chrombpnet_is_multi_layer_and_says_so():
    """The specific case this file exists for.

    Kept concrete rather than generic: the gap was invisible precisely because every
    *generic* statement about the five was true.
    """
    path = next((p for p in _backgrounds() if _oracle(p) == "chrombpnet"), None)
    if path is None:
        pytest.skip("chrombpnet background not downloaded")
    with np.load(path, allow_pickle=True) as data:
        layers = [str(x) for x in data["layers_per_row"]]
    counts = {layer: layers.count(layer) for layer in sorted(set(layers))}
    assert counts == {"chromatin_accessibility": 9, "tf_binding": 744}, (
        f"expected ChromBPNet's 753 rows to be 9 accessibility + 744 TF binding, got "
        f"{counts}. If the registry legitimately changed, update this and the protocol; if "
        f"it did not, the stamp is wrong."
    )

@pytest.mark.parametrize("path", _backgrounds() or [Path("none")], ids=lambda p: p.name)
def test_build_config_storage_shape_is_uniform(path: Path):
    """``build_config`` must be shape (1,) in every artefact, because readers index [0].

    Written after breaking exactly this. The first version of
    ``scripts/stamp_layers_per_row.py`` wrote ``np.array(json.dumps(...))`` -- a 0-d array --
    where every shipped artefact stores shape (1,). Nothing in the stamper noticed, because
    its own reader tolerated both shapes; the failure surfaced two steps later as

        tests/test_cherimoya_ensemble.py::test_the_shipped_null_records_that_it_was_built_from_the_ensemble
        IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed

    on five artefacts at once. A format change that only breaks *other* readers is the case
    a uniform-shape assertion exists for, so this asserts the shape rather than tolerating it.
    """
    if not path.exists():
        pytest.skip("no backgrounds downloaded")
    with np.load(path, allow_pickle=True) as data:
        if "build_config" not in data.files:
            pytest.skip(f"{_oracle(path)} ships no build_config")
        shape = data["build_config"].shape
    assert shape == (1,), (
        f"{_oracle(path)}: build_config has shape {shape}, expected (1,). Readers do "
        f"data['build_config'][0]; a 0-d array raises IndexError there."
    )
