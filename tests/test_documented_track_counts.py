"""A track count in user-facing docs must match what a user can actually query.

The docs advertised AlphaGenome as **5,731 tracks** in ten places across four files.
That is the row count of its metadata table, and 563 of those rows are ``padding``
placeholders: entries that exist only to keep ``local_index`` aligned with the
model's output array. ``iter_tracks()`` skips them, they carry no assay, and the
shipped background has no row for any of them. The number a user can query, and the
number of background rows, is **5,168** — verified both ways: 5,168 metadata tracks
have a background row and 0 do not.

So the docs overstated usable coverage by 563 tracks, in the headline sentence of the
README. Not a large error, but the sort that erodes trust in every other number
beside it, and it survived because nothing compared prose to the artefact.

This test does that comparison. It reads the shipped NPZs and the oracle metadata and
fails on any live doc claiming a per-oracle count that contradicts them.

Dated files under ``audits/`` and historical ``CHANGELOG.md`` entries are excluded on
purpose: they record what was true when written, and rewriting them would destroy the
record rather than fix anything.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
BACKGROUNDS = Path.home() / ".chorus" / "backgrounds"

LIVE_DOCS = [
    "README.md",
    "docs/variant_analysis_framework.md",
    "docs/MCP_WALKTHROUGH.md",
    "docs/API_DOCUMENTATION.md",
]

# Numbers that are legitimately NOT a total track count, so a bare digit match on
# them is a false positive. Each needs a reason to be here.
ALLOWED_IN_CONTEXT = {
    # The explanatory footnote that states the padding split. It must mention 5,731.
    "5,731 rows",
    # Filtered subsets, not totals.
    "266 tracks",
    "1504 tracks",
}


def _shipped_counts() -> dict[str, int]:
    if not BACKGROUNDS.is_dir():
        return {}
    out = {}
    for path in sorted(BACKGROUNDS.glob("*_pertrack.npz")):
        with np.load(path, allow_pickle=True) as data:
            out[path.name.replace("_pertrack.npz", "")] = len(data["track_ids"])
    return out


def test_alphagenome_padding_rows_are_the_whole_discrepancy():
    """Pin the reason the two numbers differ, so the footnote stays true."""
    try:
        from chorus.oracles.alphagenome_source.alphagenome_metadata import get_metadata
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"alphagenome metadata unavailable: {exc}")
    metadata = get_metadata()
    real = list(metadata.iter_tracks())
    total = len(metadata._tracks)
    padding = [t for t in metadata._tracks
               if str(t.get("name", "")).lower() == "padding"]
    assert len(real) + len(padding) == total, "padding is not the only exclusion"
    assert len(real) == 5_168, f"expected 5168 queryable tracks, got {len(real)}"
    assert len(padding) == 563, f"expected 563 padding rows, got {len(padding)}"


def test_every_queryable_alphagenome_track_has_a_background_row():
    """The claim the docs now make: 5,168 tracks, all of them normalisable."""
    counts = _shipped_counts()
    if "alphagenome" not in counts:
        pytest.skip("no downloaded alphagenome background")
    try:
        from chorus.oracles.alphagenome_source.alphagenome_metadata import get_metadata
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"alphagenome metadata unavailable: {exc}")
    with np.load(BACKGROUNDS / "alphagenome_pertrack.npz", allow_pickle=True) as data:
        ids = {str(x) for x in data["track_ids"]}
    missing = [str(t.get("identifier")) for t in get_metadata().iter_tracks()
               if str(t.get("identifier")) not in ids]
    assert not missing, f"{len(missing)} queryable tracks have no background row"
    assert len(ids) == counts["alphagenome"] == 5_168


@pytest.mark.parametrize("rel", LIVE_DOCS)
def test_live_docs_do_not_claim_a_stale_track_count(rel: str):
    counts = _shipped_counts()
    if not counts:
        pytest.skip("no downloaded backgrounds")
    path = REPO / rel
    if not path.exists():
        pytest.skip(f"{rel} absent")
    text = path.read_text()

    problems = []
    for oracle, actual in counts.items():
        for match in re.finditer(
            rf"{oracle}[^\n]{{0,90}}?\b([\d,]{{2,7}})\s*(?:tracks|rows)", text, re.I
        ):
            phrase = match.group(0)
            if any(ok in phrase for ok in ALLOWED_IN_CONTEXT):
                continue
            claimed = int(match.group(1).replace(",", ""))
            if claimed != actual:
                problems.append(
                    f"{rel}: claims {claimed} for {oracle}, shipped is {actual} "
                    f"-- {phrase.strip()!r}"
                )
    assert not problems, "\n".join(problems)
