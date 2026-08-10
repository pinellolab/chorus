"""Every cell type with a background row must be discoverable through the API.

``LegNetOracle.list_cell_types()`` returned ``[self.cell_type]`` — whichever single
line the instance happened to be constructed with. But LegNet ships weights for three
(``LEGNET_AVAILABLE_CELLTYPES = ["K562", "HepG2", "WTC11"]``), all three are reachable
via ``LegNetOracle(cell_type=...)``, and all three have a row in
``legnet_pertrack.npz``. So two of the three were **undiscoverable**: an agent asking
"what cell types does LegNet cover?" through the MCP ``list_tracks`` path was told
HepG2 and had no way to learn the others existed.

Measured at rs12740374 while checking a blog vignette, the two hidden lines are not
redundant — they carry the signal that makes the locus interesting:

    LentiMPRA:HepG2   +0.3466     (liver, the trait-relevant line)
    LentiMPRA:K562    +0.0141
    LentiMPRA:WTC11   -0.0482

That contrast *is* the liver-specificity evidence. Hiding two thirds of the panel hid
the comparison.

The general invariant, checked for every oracle that can answer it: a track with a
shipped background row should be reachable by name from ``list_cell_types()``. The
reverse direction is deliberately not asserted — an oracle may legitimately advertise
a cell type whose background has not been built yet, and that is a coverage gap rather
than a wrong answer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

BACKGROUNDS = Path.home() / ".chorus" / "backgrounds"


def _shipped_cell_types(oracle: str) -> set[str]:
    """Cell types appearing in an oracle's background track_ids.

    Only used for oracles whose ids are ``ASSAY:CELL`` — the accession-keyed ones
    (Enformer's ENCFF, Borzoi's FANTOM) carry no parseable cell type.
    """
    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR

    path = CHORUS_BACKGROUNDS_DIR / f"{oracle}_pertrack.npz"
    if not path.exists():
        return set()
    with np.load(path, allow_pickle=True) as data:
        ids = [str(x) for x in data["track_ids"]]
    out = set()
    for tid in ids:
        if ":" in tid:
            out.add(tid.rsplit(":", 1)[-1])
        else:
            out.add(tid)
    return out


def test_legnet_lists_every_cell_type_it_has_weights_for():
    from chorus.oracles.legnet import LegNetOracle
    from chorus.oracles.legnet_source.legnet_globals import (
        LEGNET_AVAILABLE_CELLTYPES,
    )

    listed = LegNetOracle().list_cell_types()
    assert set(listed) == set(LEGNET_AVAILABLE_CELLTYPES), (
        f"list_cell_types() returned {listed} but weights exist for "
        f"{LEGNET_AVAILABLE_CELLTYPES}"
    )
    assert len(listed) == 3


def test_legnet_list_does_not_depend_on_the_constructed_cell_type():
    """The exact regression: the answer must not shrink to whatever was loaded."""
    from chorus.oracles.legnet import LegNetOracle

    for cell in ("K562", "HepG2", "WTC11"):
        listed = LegNetOracle(cell_type=cell).list_cell_types()
        assert len(listed) == 3, (
            f"constructed with cell_type={cell!r}, list_cell_types() returned "
            f"{listed} — it is reporting the instance, not the oracle"
        )


def test_every_legnet_background_cell_type_is_reachable_by_name():
    """A shipped background row nobody can select is a row nobody can use."""
    from chorus.oracles.legnet import LegNetOracle

    shipped = _shipped_cell_types("legnet")
    if not shipped:
        pytest.skip("no downloaded background for legnet")
    listed = set(LegNetOracle().list_cell_types())
    unreachable = shipped - listed
    assert not unreachable, (
        f"legnet ships background rows for {sorted(unreachable)} but "
        f"list_cell_types() does not offer them: listed={sorted(listed)}"
    )


def _chrombpnet_bg_cells(prefixes):
    """ChromBPNet ids are ``ATAC:CELL`` / ``DNASE:CELL`` / ``CHIP:CELL:TF``.

    The cell is at index 1, NOT last — an earlier version of this helper used
    ``rsplit(":", 1)[-1]`` and collected TF names (ARNT2, ATF2, BACH1...) as if they
    were cell types, which made the assertion nonsense.
    """
    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR

    path = CHORUS_BACKGROUNDS_DIR / "chrombpnet_pertrack.npz"
    if not path.exists():
        return set()
    with np.load(path, allow_pickle=True) as data:
        ids = [str(x) for x in data["track_ids"]]
    return {t.split(":")[1] for t in ids
            if t.count(":") >= 1 and t.split(":")[0] in prefixes}


def test_chrombpnet_lists_every_accessibility_cell_type_including_h1():
    """H1 had a ``DNASE:H1`` background row but was absent from the API.

    ``list_cell_types()`` was a hardcoded ``["IMR-90", "GM12878", "HepG2", "K562"]``
    beside ``CHROMBPNET_MODELS_DICT``, which has five under DNASE. So H1 could be
    loaded and scored but never discovered — a hardcoded list next to a registry is a
    second source of truth, and that is what it cost.
    """
    from chorus.oracles.chrombpnet import ChromBPNetOracle
    from chorus.oracles.chrombpnet_source.chrombpnet_globals import (
        CHROMBPNET_MODELS_DICT,
    )

    listed = set(ChromBPNetOracle().list_cell_types())
    registry = set(CHROMBPNET_MODELS_DICT.get("ATAC", {}))
    registry |= set(CHROMBPNET_MODELS_DICT.get("DNASE", {}))
    assert listed == registry, (
        f"list_cell_types() returned {sorted(listed)} but the registry the loader "
        f"reads has {sorted(registry)}"
    )
    assert "H1" in listed, "the specific regression: H1 was missing"

    shipped = _chrombpnet_bg_cells({"ATAC", "DNASE"})
    if shipped:
        assert not shipped - listed, (
            f"accessibility background rows exist for {sorted(shipped - listed)} "
            f"but they are not listed"
        )


def test_chrombpnet_chip_cell_types_are_reachable_via_the_assay_argument():
    """172 CHIP lines are excluded from the default on purpose, not lost.

    Returning all 172 by default would bury the answer to "which cell types can I
    profile accessibility in?" (5). They must still be discoverable.
    """
    from chorus.oracles.chrombpnet import ChromBPNetOracle

    o = ChromBPNetOracle()
    chip = set(o.list_cell_types(assay="CHIP"))
    assert len(chip) > 100, f"only {len(chip)} CHIP cell types offered"
    shipped = _chrombpnet_bg_cells({"CHIP"})
    if shipped:
        assert not shipped - chip, (
            f"CHIP background rows exist for {sorted(shipped - chip)[:8]} but they "
            f"are not offered by list_cell_types(assay='CHIP')"
        )


def test_an_unknown_assay_is_refused_rather_than_silently_empty():
    from chorus.core.exceptions import InvalidAssayError
    from chorus.oracles.chrombpnet import ChromBPNetOracle

    with pytest.raises(InvalidAssayError):
        ChromBPNetOracle().list_cell_types(assay="NOT_AN_ASSAY")
