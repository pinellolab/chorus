"""A ranking table must match its artefact ROW BY ROW, not merely number-by-number.

``test_walkthrough_readmes_match_artefacts`` asks whether each quoted effect size
appears *anywhere* in the walkthrough's artefact, within 5e-3. That is the right
check for prose ("CEBPB opens by +3.316"), and it is far too weak for a summary
table over a large artefact.

Measured on the discovery screen: its artefact holds **1,601 distinct numbers**, and
all three stale claims in the README's top-3 table collided with unrelated ones —

    README +1.914  ->  1.9110411407113541   (a different track entirely)
    README +1.604  ->  1.6040897840641255
    README +1.451  ->  1.4506254344800540

so the guard passed a table that named the wrong three cell types, the wrong track
counts, and concluded "the top hits are prostate (LNCaP) and kidney" where the
artefact says amniotic epithelial cell, MCF 10A and HepG2. It also listed an output
file that does not exist and an HTML glob matching nothing.

The fix for that class is to bind the number to its LABEL. This module does it for
the ranking tables, where the label is a cell type and the artefact records the
pairing explicitly.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DISCOVERY = REPO / "examples" / "walkthroughs" / "discovery"
SUMMARY = DISCOVERY / "SORT1_cell_type_screen" / "discovery_summary.json"

_MINUS = "−"
_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([+\-" + _MINUS + r"]?\d+\.\d+)\s*\|\s*(\d+)\s*\|",
    re.M,
)


def _num(s: str) -> float:
    return float(s.replace(_MINUS, "-"))


@pytest.fixture(scope="module")
def summary():
    if not SUMMARY.exists():
        pytest.skip("discovery_summary.json absent")
    return json.loads(SUMMARY.read_text())


def test_the_ranking_table_names_the_cell_types_the_artefact_ranked(summary):
    readme = DISCOVERY / "README.md"
    if not readme.exists():
        pytest.skip("discovery/README.md absent")
    rows = _ROW.findall(readme.read_text())
    if not rows:
        pytest.skip("no ranking table found")

    claimed = [r[1].strip() for r in rows]
    actual = [r["cell_type"] for r in summary]
    assert claimed == actual, (
        f"the README's ranking names {claimed} but the artefact ranked {actual}. "
        f"A biological conclusion drawn from the wrong cell types is the most "
        f"misleading thing this repo can ship."
    )


def test_each_row_effect_matches_that_cell_types_own_effect(summary):
    """The check the number-anywhere guard cannot make.

    Binding effect to cell type is what stops a stale figure from being excused by a
    coincidental match elsewhere in a 1,601-number artefact.
    """
    readme = DISCOVERY / "README.md"
    if not readme.exists():
        pytest.skip("discovery/README.md absent")
    rows = _ROW.findall(readme.read_text())
    if not rows:
        pytest.skip("no ranking table found")

    by_cell = {r["cell_type"]: r for r in summary}
    for _rank, cell, effect, tracks in rows:
        cell = cell.strip()
        assert cell in by_cell, f"{cell!r} is not in discovery_summary.json"
        want = by_cell[cell]
        assert abs(_num(effect) - want["effect"]) <= 5e-3, (
            f"{cell}: README says effect {effect}, artefact says "
            f"{want['effect']:+.4f}"
        )
        assert int(tracks) == int(want["n_tracks"]), (
            f"{cell}: README says {tracks} tracks, artefact says {want['n_tracks']}"
        )


def test_the_ranking_order_matches_the_metric_the_artefact_used(summary):
    """The table is ranked by ``alt x |effect|``, not by effect.

    Worth pinning because the two orders differ here — HepG2 has the smallest raw
    log2FC of the three and 562 tracks — so a reader who assumes "ranked by effect"
    draws the wrong conclusion about which call is best supported.
    """
    metrics = [r.get("ranking_metric") for r in summary]
    assert len(set(metrics)) == 1, f"mixed ranking metrics: {metrics}"
    assert metrics[0] == "alt_x_abs_effect", (
        f"the artefact now ranks by {metrics[0]!r}; the README explains "
        f"'alt x |effect|' and must be updated together with it"
    )
    scores = [r["ranking_score"] for r in summary]
    assert scores == sorted(scores, reverse=True), (
        f"discovery_summary.json is not in descending ranking_score order ({scores}), "
        f"so the README's rank column cannot be trusted to mirror it"
    )


def test_every_output_file_the_readme_advertises_exists():
    """A README naming a file that was never written sends a user hunting.

    ``cell_type_ranking.json`` was advertised and does not exist anywhere in the
    repo, and the HTML pattern given (``rs12740374_SORT1_*_alphagenome_report.html``)
    matched nothing — the committed names are
    ``chr1_109274968_G_T_SORT1_alphagenome_<cell>_report.html``.
    """
    readme = DISCOVERY / "README.md"
    if not readme.exists():
        pytest.skip("discovery/README.md absent")
    text = readme.read_text()
    screen = DISCOVERY / "SORT1_cell_type_screen"

    # Backticked names that look like a concrete output file, excluding globbed and
    # brace-expanded forms which are patterns rather than names.
    named = {
        m.group(1) for m in re.finditer(r"`([A-Za-z0-9_.<>-]+\.(?:json|md|tsv|html))`", text)
        if "*" not in m.group(1) and "<" not in m.group(1)
    }
    missing = sorted(n for n in named if not (screen / n).exists())
    assert not missing, (
        f"discovery/README.md advertises {missing}, which do not exist in "
        f"{screen.relative_to(REPO)}"
    )
