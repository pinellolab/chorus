"""The Cell Type column must not repeat what the track label already says (audit F7).

ChromBPNet and BPNet track ids carry the biosample, so a row read

    CHIP:CEBPB:IMR-90  |  IMR-90  |  0.0161  |  0.0180  |  +0.157  | ...

saying the cell type twice and spending a column on it. Measured 2026-08-12 across the
committed corpus: **13 of 20 reports**, up to 22 rows in
``rs12740374_SORT1_locus_causal_report.html``. The audit checklist has carried this as a
"known regression" for several passes.

Suppression is deliberately narrow — an exact, case-insensitive match on the **final
colon-delimited component** of the label. Anything looser starts guessing:

* ``DNASE:fibroblast of lung`` next to ``fibroblast of lung`` is also a duplicate, and is
  suppressed, because the tail matches exactly;
* ``CHIP:CEBPB:HepG2_treated`` next to ``HepG2`` is **not** suppressed, because they are not
  the same string and the column may be the only place the plain biosample appears;
* Enformer's opaque ids (``CNhs10608``) keep their cell type, which is the case where the
  column carries all of the meaning.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent

#: The pattern the audit used to count the defect: a label ending in the cell name,
#: immediately followed by a cell that repeats it.
DUPLICATE_ROW = re.compile(
    r'>([A-Z][A-Za-z0-9_]*:[A-Za-z0-9_.\-]+:([A-Za-z0-9_.\-]+))</td>\s*<td[^>]*>\2</td>')


@pytest.mark.parametrize("label,cell,expected", [
    ("CHIP:CEBPB:IMR-90", "IMR-90", "—"),
    ("CHIP:ARID3A:HepG2", "HepG2", "—"),
    ("DNASE:fibroblast of lung", "fibroblast of lung", "—"),
    ("dnase:hepg2", "HepG2", "—"),                      # case-insensitive
    ("DNASE:HepG2", "K562", "K562"),                    # disagreeing: keep both
    ("CNhs10608", "substantia nigra", "substantia nigra"),   # opaque id: keep
    ("CHIP:CEBPB:HepG2_treated", "HepG2", "HepG2"),      # substring only: keep
    ("DNASE:HepG2", "", "—"),                           # nothing to show
])
def test_the_cell_column_is_blanked_only_on_an_exact_tail_match(label, cell, expected):
    from chorus.analysis.variant_report import _cell_type_cell

    ts = SimpleNamespace(description=label, assay_id=label, cell_type=cell)
    assert _cell_type_cell(ts) == expected


def _committed_reports() -> list:
    out = subprocess.check_output(["git", "ls-files", "examples/"], cwd=REPO, text=True)
    return [REPO / p for p in out.split() if p.endswith(".html")]


@pytest.mark.parametrize("report", _committed_reports(), ids=lambda p: p.name)
def test_no_committed_report_repeats_the_cell_type(report: Path):
    """Reads the artefacts, because the fix only counts once they are regenerated."""
    text = report.read_text()
    hits = DUPLICATE_ROW.findall(text)
    assert not hits, (
        f"{report.name}: {len(hits)} rows still print the cell type twice, e.g. "
        f"{hits[0][0]} beside {hits[0][1]}. Regenerate this report "
        f"(see CLAUDE.md's regeneration matrix)."
    )
