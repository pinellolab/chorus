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
    """A shipped background row nobody can select is a row nobody can use.

    Scoped to LegNet on purpose. ChromBPNet's ``list_cell_types()`` returns 4 while
    its background carries **172** — but the other 168 are CHIP (TF-binding) lines
    rather than accessibility models, so the 4 may be deliberate scoping rather than
    an omission. Its docstring ("Return ChromBPNet's cell types") does not say which,
    and asserting a design decision I have not confirmed is wrong would make this
    test a guess. Recorded as a finding for the maintainer instead; see
    audits/2026-08-05_blog_vignette_recapitulation.md.
    """
    from chorus.oracles.legnet import LegNetOracle

    shipped = _shipped_cell_types("legnet")
    if not shipped:
        pytest.skip("no downloaded background for legnet")
    listed = set(LegNetOracle().list_cell_types())
    oracle = "legnet"
    unreachable = shipped - listed
    assert not unreachable, (
        f"{oracle} ships background rows for {sorted(unreachable)} but "
        f"list_cell_types() does not offer them, so they cannot be selected: "
        f"listed={sorted(listed)}"
    )
