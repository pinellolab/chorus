"""Nothing may hand-roll a centred window. This is what #144 needed and lacked.

#144 — "builder and query compute different statistics against each other" — was
closed on four instances. A consistency audit then found a **fifth**, in
``chorus/analysis/discovery.py``, at two sites:

    ref_track.score_region(chrom, pos - half, pos + half + 1, cfg.aggregation)

That is the exact pre-#144 arithmetic ``scorers.py`` was migrated off. It builds
genomic coordinates and lets ``score_region`` floor/ceil-expand them back to bins,
which yields **4 or 5** bins for ``window_bp=501`` at Enformer's 128 bp resolution
depending on where the variant falls inside its bin — against a null built over 3.
The discovery path reads the same per-track backgrounds as the variant report, so it
was ranking a wider statistic against a narrower reference; and because
``discover_cell_types`` *compares* those effects to order cell types, a span that
varies with sub-bin position can reorder the ranking itself.

The reason it survived the umbrella fix is that the fix was applied where the defect
was known to be, and nothing searched for the shape elsewhere. Four instances were
enumerated by hand; a fifth existed. So this test does the searching: any module that
derives a window from a *layer config* must go through
``PredictionTrack.score_centered_window``, the single shared definition, rather than
constructing coordinates itself.

``score_region`` is not banned outright — it is the right primitive for a genuine
genomic interval, which is what ``core/base.py``'s exon summing and
``core/result.py``'s explicit region scoring both want. What is banned is using it to
approximate a *centred* window, and the tell is arithmetic on a half-width.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

CHORUS = Path(__file__).resolve().parent.parent / "chorus"

# ``pos - half`` / ``pos + half`` in any spacing, which is how every instance of this
# defect has been written.
_HALF_WIDTH = re.compile(r"\bhalf\b\s*[-+]|[-+]\s*\bhalf\b")
_WINDOW_HALF = re.compile(r"\bwindow_bp\b\s*//\s*2|\bcfg\.window_bp\b\s*or\s*\d+\s*\)\s*//\s*2")


def _py_files():
    return sorted(p for p in CHORUS.rglob("*.py") if "_source" not in p.parts)


@pytest.mark.parametrize("path", _py_files(), ids=lambda p: str(p.relative_to(CHORUS)))
def test_no_module_builds_a_centred_window_by_hand(path: Path):
    """A ``score_region`` call on the same line as half-width arithmetic."""
    offenders = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("*"):
            continue
        if "score_region(" not in line:
            continue
        if _HALF_WIDTH.search(line):
            offenders.append(f"{path.relative_to(CHORUS)}:{i}: {stripped[:110]}")
    assert not offenders, (
        "centred window built by hand instead of via score_centered_window "
        "(#144 instance 5):\n" + "\n".join(offenders)
    )


def test_the_shared_helper_is_what_discovery_actually_calls():
    """Positive assertion, so deleting the call cannot pass by absence.

    A test that only forbids the bad pattern goes green if someone removes the
    scoring entirely, which is how a guard ends up protecting nothing.
    """
    src = (CHORUS / "analysis" / "discovery.py").read_text()
    assert src.count("score_centered_window(") >= 4, (
        "discovery.py should call score_centered_window for ref and alt at both "
        "sites (discover_variant_effects and discover_cell_types)"
    )
    # Ignore comment lines: discovery.py quotes the old call in the note explaining
    # why it was replaced, and that explanation is worth keeping. The line-level test
    # above is what actually forbids a live occurrence.
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert "pos - half, pos + half + 1" not in code


def test_rna_is_not_scored_as_a_mean_over_the_whole_prediction():
    """The second discovery defect, pinned.

    ``gene_expression`` has ``window_bp=None`` because its statistic is the mean over
    a gene's merged exon mask. The ``window_bp is None`` branch used to fall through
    to ``np.mean(track.values)`` — the mean over the ENTIRE prediction, 524 kb for
    Borzoi and 1 Mb for AlphaGenome, dominated by intergenic and intronic zeros. That
    number has no relation to the exon-mask null it was then ranked against.

    Discovery has no gene context, so the honest behaviour is to decline the layer
    rather than emit a confidently wrong percentile.
    """
    src = (CHORUS / "analysis" / "discovery.py").read_text()
    assert 'elif layer == "gene_expression"' in src, (
        "discovery must special-case gene_expression before the full-output branch"
    )
    # and the guard must come BEFORE the np.mean fallback, or it is dead code
    idx_guard = src.index('elif layer == "gene_expression"')
    idx_mean = src.index("np.mean(ref_track.values)")
    assert idx_guard < idx_mean


def test_layers_with_no_window_are_enumerated_not_assumed():
    """Which layers legitimately have ``window_bp=None`` — so a new one is noticed.

    Measured: gene_expression, promoter_activity, regulatory_classification,
    enhancer_activity. Only the last three genuinely have no window; RNA's None means
    "use the exon mask", which is a different thing wearing the same value.
    """
    from chorus.analysis.scorers import LAYER_CONFIGS

    none_windowed = {k for k, v in LAYER_CONFIGS.items() if v.window_bp is None}
    assert none_windowed == {
        "gene_expression",
        "promoter_activity",
        "regulatory_classification",
        "enhancer_activity",
    }, (
        f"the set of window-less layers changed to {sorted(none_windowed)}. Each one "
        f"falls through to a mean over the whole prediction in discovery.py — check "
        f"that is right for the new layer before updating this test."
    )
