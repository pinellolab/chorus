"""The README's disk numbers must be internally consistent.

The install prerequisite said **~38 GB free disk** for the better part of a year while a default
Linux + CUDA install actually took ~87 GB — measured with ``du -sh`` on the H100 box during the
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

#: `~53 GB`, `~50 MB`, `**~85 GB**` — the size cell of a bucket row.
_SIZE = re.compile(r"~\s*([\d.]+)\s*(GB|MB)", re.I)

#: Tolerance in GB for the sum-vs-total check. Rows are rounded to 0.1 GB, and there are ~14 of
#: them, so a little slack is legitimate — but keep it *tight*. At the 1.0 GB it started at, an
#: adversarial pass showed that deleting **any** of the five sub-GB rows still passed (Enformer at
#: 0.94 GB moved the sum by only 0.59 GB against 0.35 GB of existing slack), which defeats the point
#: of the guard. 0.4 GB is above the worst-case rounding error of 14 rows (14 × 0.05 = 0.7 GB in
#: theory, ~0.35 GB observed) and below the smallest row that matters.
TOLERANCE_GB = 0.4

#: Below this, a row is small enough that the sum check alone cannot notice it going missing, so the
#: row count is asserted separately.
EXPECTED_ROW_COUNT = 14


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
    """Fails loudly if the table moved or was reformatted, rather than passing vacuously.

    The exact row count is asserted, not just a floor. Five of the fourteen rows are under 1 GB, and
    the sum check cannot see one of those disappear — an adversarial pass proved that deleting the
    Enformer, ChromBPNet, LegNet, EPInformer-seq or Cherimoya-weights row left the total within
    tolerance. So the count is the guard for small rows and the sum is the guard for large ones.
    """
    rows, total = _disk_table()
    assert total is not None, "no '**Total default**' row found in the disk usage breakdown"
    assert len(rows) == EXPECTED_ROW_COUNT, (
        f"the disk table has {len(rows)} bucket rows, expected {EXPECTED_ROW_COUNT}:\n  "
        + "\n  ".join(f"{gb:6.2f} GB  {label[:70]}" for label, gb in rows)
        + f"\nIf a bucket was genuinely added or removed, update EXPECTED_ROW_COUNT. This count "
          f"exists because {sum(1 for _, gb in rows if gb < 1)} rows are under 1 GB and the "
          f"sum check cannot notice one of those going missing."
    )


def test_the_buckets_sum_to_the_stated_total():
    rows, total = _disk_table()
    summed = sum(gb for _, gb in rows)
    detail = "\n  ".join(f"{gb:6.2f} GB  {label[:70]}" for label, gb in rows)
    assert abs(summed - total) <= TOLERANCE_GB, (
        f"the disk table's rows sum to {summed:.2f} GB but it claims {total:.2f} GB:\n"
        f"  {detail}\n"
        f"Update the total when you change a row — a table that does not add up is how the "
        f"'~38 GB' prerequisite survived a real footprint of ~85 GB (now ~87 GB after the 0.7.4 Sei rebuild)."
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
    # The headroom is deliberate and has two named components, so the ceiling allows for both: the
    # mamba package cache (~4 GiB, reclaimable) and the decimal-vs-binary gap, since a volume sold as
    # "N GB" is only N × 1000³/1024³ GiB — at this size roughly 6 GiB less than the label suggests.
    assert promised - total <= 20, (
        f"the prerequisite ({promised:.0f} GB) is {promised - total:.0f} GB above the breakdown "
        f"({total:.0f} GB) — padding that large stops being a useful number."
    )


@pytest.mark.parametrize("row_fragment", [
    "| Enformer weights | ~960 MB |",
    "| ChromBPNet slim HuggingFace mirror",
    "| LegNet weights | ~41 MB |",
    "| EPInformer-seq per-cell weights",
    "| Cherimoya fast-path weights",
])
def test_dropping_a_sub_gb_row_is_caught(row_fragment, monkeypatch):
    """Fails-without-fix for the small rows the sum check alone cannot see.

    At the original 1.0 GB tolerance every one of these deletions passed silently.
    """
    lines = README.read_text().splitlines()
    kept = [ln for ln in lines if row_fragment not in ln]
    assert len(kept) == len(lines) - 1, (
        f"expected to delete exactly one row, deleted {len(lines) - len(kept)} — "
        f"has the table been reworded? fragment: {row_fragment!r}"
    )
    text = "\n".join(kept)
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: text)
    with pytest.raises(AssertionError):
        test_the_table_has_rows_and_a_total()


@pytest.mark.parametrize("bad_total", ["~38 GB", "~120 GB"])
def test_the_guard_catches_a_total_that_does_not_match_its_rows(bad_total, monkeypatch):
    """Fails-without-fix: the historical wrong number, and an over-correction."""
    text = README.read_text().replace("| **Total default** | **~87 GB** |",
                                      f"| **Total default** | **{bad_total}** |")
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: text)
    with pytest.raises(AssertionError):
        test_the_buckets_sum_to_the_stated_total()
