"""The README's disk numbers must be internally consistent.

The install prerequisite said **~38 GB free disk** for the better part of a year while a default
Linux + CUDA install actually took ~72 GB — measured with ``du -sh`` on the H100 box during the
v0.7.3 audit. A user who provisioned a 40 GB volume on that advice ran out of space partway
through ``chorus setup``, which is the worst moment to find out.

Why it drifted: each oracle env carries its own copy of the NVIDIA CUDA wheels (2.9 GB of
``nvidia_*`` plus 1.3 GB of ``tensorflow`` in the Enformer env alone). Those come from pip, so
unlike conda packages they are **not** hardlinked between envs — nine envs means nine copies. The
old "~3 GB each" was plausible for a macOS/CPU-only install and was never re-measured after CUDA
wheels became the norm.

This test cannot measure a fresh install from CI, so it guards the thing that *is* checkable and
that actually broke: the table's rows must sum to its own stated total, and the prerequisite at the
top of the README must be at least that total. Someone editing one row and not the total, or
correcting the table without touching the prerequisite, fails here.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"

#: `~53 GB`, `~50 MB`, `**~72 GB**` — the size cell of a bucket row.
_SIZE = re.compile(r"~\s*([\d.]+)\s*(GB|MB)", re.I)

#: Tolerance in GB. The rows are rounded to 0.1 GB, so the sum can legitimately sit a little off
#: the stated total; 1 GB is loose enough for rounding and tight enough to catch a dropped row.
TOLERANCE_GB = 1.0


def _to_gb(value: str, unit: str) -> float:
    return float(value) / 1024 if unit.upper() == "MB" else float(value)


def _disk_table() -> tuple[list[tuple[str, float]], float]:
    """The bucket rows and the stated total, from the Disk usage breakdown table."""
    text = README.read_text()
    start = text.index("#### Disk usage breakdown")
    # The table ends at the blank line after the last `|` row.
    rows, total = [], None
    for line in text[start:].splitlines():
        if not line.startswith("|"):
            if rows:  # past the table
                break
            continue
        if set(line) <= set("|- "):  # the |---|---| separator
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2:
            continue
        label, size = cells
        m = _SIZE.search(size)
        if not m:
            continue  # the `| Bucket | Size |` header
        gb = _to_gb(*m.groups())
        if "total" in label.lower():
            total = gb
        else:
            rows.append((label, gb))
    return rows, total


def test_the_table_has_rows_and_a_total():
    """Fails loudly if the table moved or was reformatted, rather than passing vacuously."""
    rows, total = _disk_table()
    assert len(rows) >= 5, f"only parsed {len(rows)} bucket rows — has the table been reformatted?"
    assert total is not None, "no '**Total default**' row found in the disk usage breakdown"


def test_the_buckets_sum_to_the_stated_total():
    rows, total = _disk_table()
    summed = sum(gb for _, gb in rows)
    detail = "\n  ".join(f"{gb:6.2f} GB  {label[:70]}" for label, gb in rows)
    assert abs(summed - total) <= TOLERANCE_GB, (
        f"the disk table's rows sum to {summed:.2f} GB but it claims {total:.2f} GB:\n"
        f"  {detail}\n"
        f"Update the total when you change a row — a table that does not add up is how the "
        f"'~38 GB' prerequisite survived a real footprint of ~72 GB."
    )


def test_the_prerequisite_covers_the_total():
    """The `- **~N GB free disk**` bullet must not promise less than the table's own total."""
    text = README.read_text()
    m = re.search(r"\*\*~\s*([\d.]+)\s*GB free disk\*\*", text, re.I)
    assert m, "could not find the '**~N GB free disk**' prerequisite bullet in README.md"
    promised = float(m.group(1))
    _, total = _disk_table()
    assert promised >= total, (
        f"the README asks users to free {promised:.0f} GB but its own breakdown totals "
        f"{total:.0f} GB. Provisioning to the smaller number runs out of disk during "
        f"`chorus setup`."
    )
    assert promised - total <= 15, (
        f"the prerequisite ({promised:.0f} GB) is {promised - total:.0f} GB above the breakdown "
        f"({total:.0f} GB) — padding that large stops being a useful number."
    )


@pytest.mark.parametrize("bad_total", ["~38 GB", "~120 GB"])
def test_the_guard_catches_a_total_that_does_not_match_its_rows(bad_total, monkeypatch):
    """Fails-without-fix: the historical wrong number, and an over-correction."""
    text = README.read_text().replace("| **Total default** | **~85 GB** |",
                                      f"| **Total default** | **{bad_total}** |")
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: text)
    with pytest.raises(AssertionError):
        test_the_buckets_sum_to_the_stated_total()
