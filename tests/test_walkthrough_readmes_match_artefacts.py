"""A walkthrough README may not quote a number its own artefact disagrees with.

The regeneration scripts rewrite ``example_output.{json,md,tsv}`` and the HTML. They
have never touched ``README.md``, which is hand-written prose — so every correctness
fix that moved a number left the narrative behind. A consistency audit found 13
READMEs carrying 63 numeric claims, and the divergence was not marginal:

* ``variant_analysis/SORT1_chrombpnet/README.md`` describes an **ATAC** run at
  **-0.111** ("moderate closing") and builds a whole section on why AlphaGenome and
  ChromBPNet *disagree*. The committed artefact is **DNASE** at **+1.376** — same
  direction, strong opening. The section explains a contradiction that no longer
  exists.
* ``variant_analysis/SORT1_enformer/README.md`` lists six top tracks, five of which
  appear nowhere in the artefact.
* ``discovery/README.md`` names three top cell types that appear nowhere in the
  committed discovery output.
* Several quote pre-#92 effect sizes that are 7-11x smaller than the current ones.

The numbers moved for good reasons — the #149 denominator fix alone corrected the RNA
numerator by 251-1736x — but a README that states the old ones is simply wrong, and a
reader has no way to tell which of the two to believe.

This test does not check prose. It extracts every signed decimal that looks like an
effect size and requires it to appear somewhere in the sibling artefact. That is
deliberately narrow: it catches the stale-number class without trying to referee
wording, and it fails loudly when a fix moves a number so the README is updated in
the same commit.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

WALKTHROUGHS = Path(__file__).resolve().parent.parent / "examples" / "walkthroughs"

# A signed decimal with at least two fraction digits: +1.376, -0.111, +3.316.
# Two digits minimum keeps out version strings, section numbers and p-values.
#
# The sign class must include U+2212 MINUS SIGN, not just ASCII hyphen-minus. Markdown
# written by a human -- or pasted from a rendered table -- routinely uses the typographic
# minus, and an ASCII-only class silently sees NOTHING: this test reported
# SKIPPED "no effect-size claims" for region_swap/README.md, a file that is *entirely* a
# table of five effect claims, all of them stale, all written with U+2212. A guard that
# skips a file is indistinguishable from a guard that passes it.
_MINUS = "\u2212"
_EFFECT = re.compile(rf"(?<![\w.])([+\-{_MINUS}]\d+\.\d{{2,}})(?![\w])")


def _normalise_sign(claim: str) -> str:
    """U+2212 -> ASCII '-' so float() accepts it."""
    return claim.replace(_MINUS, "-")

# Numbers that appear in prose for reasons other than quoting this artefact.
_ALLOWED = {
    # Thresholds and worked arithmetic, not measurements.
    "+0.00", "-0.00",
}
_TOLERANCE = 5e-3      # READMEs round; the artefact is full precision.


def _artefact_numbers(directory: Path) -> set[float]:
    """Every numeric leaf in this walkthrough's JSON, plus its TSV cells."""
    out: set[float] = set()
    js = directory / "example_output.json"
    if js.exists():
        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    walk(v)
            elif isinstance(o, bool) or o is None:
                return
            elif isinstance(o, (int, float)):
                out.add(float(o))
        walk(json.loads(js.read_text()))
    tsv = directory / "example_output.tsv"
    if tsv.exists():
        for tok in re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", tsv.read_text()):
            try:
                out.add(float(tok))
            except ValueError:
                pass
    return out


def _cases():
    if not WALKTHROUGHS.is_dir():
        return []
    out = []
    for readme in sorted(WALKTHROUGHS.rglob("README.md")):
        # Only leaf walkthroughs have their own artefact; category-level READMEs
        # (variant_analysis/README.md and friends) summarise several and are checked
        # against the union of their children.
        out.append(pytest.param(
            readme, id=str(readme.parent.relative_to(WALKTHROUGHS)) or "root"))
    return out


@pytest.mark.parametrize("readme", _cases())
def test_readme_effect_sizes_appear_in_the_artefacts(readme: Path):
    claims = [c for c in _EFFECT.findall(readme.read_text())
              if _normalise_sign(c)[:5] not in _ALLOWED]
    if not claims:
        pytest.skip("no effect-size claims")

    # A leaf README is checked against its own artefact; a category README against
    # the union over its subdirectories, since it summarises all of them.
    dirs = [readme.parent]
    if not (readme.parent / "example_output.json").exists():
        dirs = [p.parent for p in readme.parent.rglob("example_output.json")]
    if not dirs:
        pytest.skip("no artefact to check against")

    available: set[float] = set()
    for d in dirs:
        available |= _artefact_numbers(d)
    if not available:
        pytest.skip("artefacts carry no numbers")

    stale = []
    for claim in claims:
        want = float(_normalise_sign(claim))
        if not any(abs(want - got) <= _TOLERANCE for got in available):
            stale.append(claim)

    assert not stale, (
        f"{readme.relative_to(WALKTHROUGHS.parent.parent)} quotes "
        f"{len(stale)} effect size(s) absent from its artefact(s): "
        f"{stale[:8]}. The regeneration scripts rewrite the JSON/MD/TSV/HTML but "
        f"never the README, so a correctness fix leaves the prose behind. Update "
        f"the README in the same commit as the number."
    )
