"""Contract checks on the artefacts committed under ``examples/``.

Every defect these catch shipped for weeks or months because nothing compared
one committed file against another, or against the generator that claims to
produce it. There was no validation layer at all: ``scripts/`` holds 8
background builders and 5 regenerators and **0** validators, no test read
``examples/``, and nothing checked a file size or a JSON-vs-TSV disagreement.

All checks here are hermetic in the sense that matters for CI — they read only
files already committed to the repo. No GPU, no network, no model weights, no
reference FASTA.

Concretely, these would have caught:

* the multi-oracle directory shipping ``example_output.json`` with no
  ``example_output.tsv`` beside it, because only one of three regeneration
  scripts ever wrote one;
* ``SORT1_enformer``'s TSV describing entirely different tracks from the JSON
  next to it (``ENCFF571HTM`` at quantile 1.0 vs ``ENCFF430NNH`` at 0.9605);
* eleven examples sitting three months stale with almost every percentile
  pinned to the ceiling by the #119 denominator bug;
* three examples whose generator entries were commented out, so they could not
  be refreshed at all and nothing noticed.
"""

from __future__ import annotations

import ast
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WALKTHROUGHS = REPO_ROOT / "examples" / "walkthroughs"

# Every directory that ships a scored artefact.
EXAMPLE_DIRS = sorted(
    p.parent for p in WALKTHROUGHS.rglob("example_output.json")
)

# ---------------------------------------------------------------------------
# Percentile saturation — the fingerprint of the #119 denominator bug
# ---------------------------------------------------------------------------
# `_get_denominator` used to divide the searchsorted rank by the raw sample
# count instead of the CDF grid width, which pinned most of the range to 1.0.
# Measured fraction of rows at exactly |quantile| == 1.0:
#
#   at 707badb (pre-refresh)   76.1% – 100.0%   for 11 of 13 examples
#   after the refresh           0.0% –  41.0%
#
# 0.60 sits ~20 points above today's worst (region_swap, 40.6%) and ~16 below
# the mildest stale case that mattered. A *median* threshold was rejected:
# SORT1_chrombpnet scores a single track at 0.9995 off a genuine +1.376
# effect, so a median test flags a legitimately strong small example.
_MAX_SATURATED_FRACTION = 0.60


def _quantiles(doc) -> list[float]:
    """Every |quantile_score| in a report, counted once.

    A variant report stores each score object **twice** — once in
    ``alleles.<allele>.all_scores`` and again grouped under
    ``alleles.<allele>.scores_by_layer``. A naive recursive walk therefore
    doubles every value (measured: 104 for a 52-scored-row report), which
    silently breaks any count-based comparison against the TSV.
    ``all_scores`` is the canonical flat list, so read only that.

    Falls back to a de-duplicating recursive walk for the report shapes that
    have no ``all_scores`` (batch scoring, discovery, causal).
    """
    out: list[float] = []
    alleles = doc.get("alleles") if isinstance(doc, dict) else None
    if isinstance(alleles, dict):
        for payload in alleles.values():
            for score in (payload or {}).get("all_scores") or []:
                v = score.get("quantile_score")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append(abs(v))
        if out:
            return out

    seen: set[int] = set()

    def walk(node):
        if isinstance(node, dict):
            if id(node) in seen:
                return
            seen.add(id(node))
            v = node.get("quantile_score")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(abs(v))
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(doc)
    return out


def _ids(d: Path) -> str:
    """Readable parametrise ids — the example path, not d0/d1/d2."""
    return str(d.relative_to(WALKTHROUGHS))


def test_there_are_examples_to_check():
    """Guard the guard: a bad glob silently turns every test below into a no-op."""
    assert len(EXAMPLE_DIRS) >= 13, (
        f"found only {len(EXAMPLE_DIRS)} example dirs under {WALKTHROUGHS} — "
        "if examples/ moved, these tests are checking nothing"
    )


@pytest.mark.parametrize("d", EXAMPLE_DIRS, ids=_ids)
def test_json_has_a_tsv_beside_it(d: Path):
    """A directory that ships one machine-readable form must ship both.

    `scripts/regenerate_multioracle.py` wrote only the JSON, so the
    multi-oracle directory was the sole one of 13 with no TSV — and
    `rerender_examples.py` refreshes a TSV only `if tsv_path.exists()`, so the
    gap could never heal itself.
    """
    assert (d / "example_output.tsv").exists(), (
        f"{d.relative_to(REPO_ROOT)} ships example_output.json but no "
        f"example_output.tsv; whichever generator writes this example must "
        f"write both"
    )


@pytest.mark.parametrize("d", EXAMPLE_DIRS, ids=_ids)
def test_tsv_agrees_with_the_json_beside_it(d: Path):
    """Every value the TSV reports must exist in the JSON beside it.

    A **subset**, not equality: several generators write a deliberate per-layer
    summary rather than a full projection (`_variant_report_tsv_rows`), so
    region_swap ships 4 TSV rows against 32 scored tracks and TERT 18 against
    83. Those are legitimate.

    What is *not* legitimate is the TSV naming values the JSON does not
    contain — which is exactly the drift that shipped: `SORT1_enformer`'s TSV
    described `ENCFF571HTM` at quantile 1.0 while the JSON regenerated beside
    it described `ENCFF430NNH` at 0.9605. A subset check catches that while
    leaving summaries alone.
    """
    tsv_path = d / "example_output.tsv"
    if not tsv_path.exists():
        pytest.skip("no TSV — covered by test_json_has_a_tsv_beside_it")

    with tsv_path.open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows or "quantile_score" not in rows[0]:
        pytest.skip("TSV is not the per-track projection format")

    # The JSON side is every scored document in the directory, not only
    # example_output.json. The multi-oracle example's TSV is a union of its
    # per-oracle `*_variant_report.json` files (the consolidated JSON holds a
    # consensus view instead), so restricting to example_output.json would
    # flag a legitimate projection.
    json_values: list[float] = []
    for jf in [d / "example_output.json", *sorted(d.glob("*_variant_report.json"))]:
        if jf.exists():
            json_values += _quantiles(json.load(open(jf)))
    json_counts = Counter(round(v, 6) for v in json_values)
    tsv_counts = Counter(
        round(abs(float(r["quantile_score"])), 6) for r in rows
        if r.get("quantile_score") not in (None, "")
    )
    if not tsv_counts:
        pytest.skip("TSV reports no percentiles")

    orphans = {
        v: (n, json_counts.get(v, 0))
        for v, n in tsv_counts.items() if n > json_counts.get(v, 0)
    }
    assert not orphans, (
        f"{d.relative_to(REPO_ROOT)}: the TSV reports percentiles the JSON "
        f"beside it does not contain — {dict(list(orphans.items())[:5])} "
        f"(value: tsv_count vs json_count). One was regenerated without the "
        f"other."
    )


@pytest.mark.parametrize("d", EXAMPLE_DIRS, ids=_ids)
def test_percentiles_are_not_saturated(d: Path):
    """Most of a report's percentiles must not sit at the ceiling.

    This is the fingerprint of a stale artefact built before the #119
    denominator fix. See the note on `_MAX_SATURATED_FRACTION`.
    """
    q = _quantiles(json.load(open(d / "example_output.json")))
    if len(q) < 5:
        pytest.skip(f"only {len(q)} scored rows — the fraction is not meaningful")

    saturated = sum(1 for v in q if v >= 1.0) / len(q)
    assert saturated <= _MAX_SATURATED_FRACTION, (
        f"{d.relative_to(REPO_ROOT)}: {saturated:.1%} of {len(q)} percentiles "
        f"are pinned at exactly 1.0 (limit {_MAX_SATURATED_FRACTION:.0%}). "
        f"That is what a pre-#119 artefact looks like — regenerate it."
    )


@pytest.mark.parametrize("d", EXAMPLE_DIRS, ids=_ids)
def test_example_records_when_it_was_generated(d: Path):
    """Without a stamp, staleness cannot be detected at all.

    `discovery/SORT1_cell_type_screen` shipped `generated_at: None`, so it was
    the one example a staleness check could not have evaluated.
    """
    doc = json.load(open(d / "example_output.json"))
    stamp = (doc.get("analysis_request") or {}).get("generated_at")
    assert stamp, (
        f"{d.relative_to(REPO_ROOT)}: analysis_request.generated_at is "
        f"{stamp!r}; every committed example needs a provenance stamp"
    )


# ---------------------------------------------------------------------------
# Reachability — an example nothing can regenerate is an orphan
# ---------------------------------------------------------------------------

_GENERATORS = [
    REPO_ROOT / "scripts" / "regenerate_examples.py",
    REPO_ROOT / "scripts" / "regenerate_remaining_examples.py",
    REPO_ROOT / "scripts" / "regenerate_multioracle.py",
]


def _reachable_dirs() -> set[str]:
    """Directory names any generator can write, read via AST.

    Parsed rather than grepped on purpose: a commented-out registry entry is
    still *text*, so a grep would happily "find" the three examples that had
    been unreachable for three months. It vanishes from the AST.

    `regenerate_examples.py` cannot be imported — it calls `parse_args()` at
    module scope — so this follows the `tests/test_cherimoya.py:581`
    `ast.parse(source)` idiom.
    """
    names: set[str] = set()
    for script in _GENERATORS:
        if not script.exists():
            continue
        tree = ast.parse(script.read_text())
        for node in ast.walk(tree):
            # Any string or f-string mentioning a walkthrough subdirectory.
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.add(node.value)
            elif isinstance(node, ast.JoinedStr):
                names.add("".join(
                    v.value for v in node.values
                    if isinstance(v, ast.Constant) and isinstance(v.value, str)
                ))
    return names


@pytest.mark.parametrize("d", EXAMPLE_DIRS, ids=_ids)
def test_example_is_reachable_from_a_generator(d: Path):
    """Every committed example must be regenerable by some script.

    BCL11A_rs1427407, FTO_rs1421085 and SORT1_rs12740374_with_CEBP had their
    registry entries commented out in an unrelated IGV commit while remaining
    advertised in README.md and examples/walkthroughs/README.md. They were
    frozen for three months with no way to refresh them.
    """
    haystack = _reachable_dirs()
    leaf = d.name
    assert any(leaf in s for s in haystack), (
        f"{d.relative_to(REPO_ROOT)} is an orphan: no generator in "
        f"{[g.name for g in _GENERATORS]} references {leaf!r}. Either wire it "
        f"up or delete the artefacts."
    )


# ---------------------------------------------------------------------------
# Size — an artefact above GitHub's limit cannot be committed at all
# ---------------------------------------------------------------------------

# GitHub rejects any file above 100 MiB outright. The LegNet report reached
# 131 MB and the consolidated multi-oracle report 139 MB, so neither could be
# committed and both had to be left stale. Today's largest tracked file under
# examples/ is ~14.7 MiB, so 20 MiB is a real ceiling with headroom rather
# than a rubber stamp.
_MAX_TRACKED_MIB = 20


def test_no_tracked_example_artefact_is_oversized():
    """A backstop for #129, enforced on what is actually committed.

    `AUDIT_CHECKLIST.md:172` already carried this as a P0 — as a *manual*
    step ("check `find examples -name '*.html' -size +50M` is empty before
    regenerating"), which was duly forgotten and cost a rejected push.
    """
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "examples/"], cwd=REPO_ROOT,
    ).decode().split("\0")

    oversized = []
    for rel in tracked:
        if not rel:
            continue
        p = REPO_ROOT / rel
        if p.is_file():
            mib = p.stat().st_size / (1024 * 1024)
            if mib > _MAX_TRACKED_MIB:
                oversized.append((mib, rel))

    assert not oversized, (
        "tracked artefacts above the "
        f"{_MAX_TRACKED_MIB} MiB ceiling:\n"
        + "\n".join(f"  {m:8.1f} MiB  {r}" for m, r in sorted(oversized, reverse=True))
    )


# ---------------------------------------------------------------------------
# Source invariant — the writer must not drift again
# ---------------------------------------------------------------------------

def test_every_generator_that_writes_json_also_writes_tsv():
    """Source-grep invariant, in the `tests/test_cherimoya.py:609` style.

    The TSV gap was not a one-off: it came from three regeneration paths
    duplicating the write logic, one of which simply omitted the TSV. Assert
    at the source level that no path can write a JSON without a TSV, so the
    drift cannot silently return.
    """
    offenders = []
    for script in _GENERATORS:
        src = script.read_text()
        n_json = src.count("example_output.json")
        n_tsv = src.count("example_output.tsv") + src.count("_write_tsv")
        if n_json and n_tsv == 0:
            offenders.append(f"{script.name}: writes JSON ({n_json}x), no TSV")

    assert not offenders, (
        "generators that write example_output.json but never a TSV:\n  "
        + "\n  ".join(offenders)
    )
