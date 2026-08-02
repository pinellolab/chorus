"""Builder and query must sum the same bins for the same ``window_bp``.

Instance 2 of #144. Two independent defects, both invisible at ``resolution=1``:

**1. The query's span is not deterministic.** ``scorers.py`` turns a window into
genomic coordinates (``start = pos - w//2``, ``end = pos + w//2 + 1``) and
``core/result.py:159-161`` floor/ceil-expands that interval to bins. The number
of bins therefore depends on where the variant happens to fall *within* its bin:

======================  ========  ==============
oracle / resolution     window    query bins
======================  ========  ==============
enformer, 128 bp             501  **4 or 5**
enformer, 128 bp            2001  **16 or 17**
borzoi, 32 bp                501  **16 or 17**
borzoi, 32 bp               2001  **63 or 64**
======================  ========  ==============

So two variants scored with identical settings are summed over different spans.
No background can match that, because the numerator's own definition moves.

**2. The builders sum a narrower, symmetric span.** All three binned builders use
``hw = window // (2 * resolution)`` then ``[centre - hw, centre + hw + 1)`` —
always odd, always centred, and deterministic:

======================  ========  =============  ============
oracle / resolution     window    builder bins   query bins
======================  ========  =============  ============
enformer, 128 bp             501  3 (384 bp)     4-5 (512-640 bp)
enformer, 128 bp            2001  15 (1920 bp)   16-17
borzoi, 32 bp                501  15 (480 bp)    16-17
chrombpnet, 1 bp             501  501            501  (agree)
chrombpnet, 1 bp            2001  2001           2001 (agree)
======================  ========  =============  ============

At ``resolution=1`` the two conventions coincide for odd windows, which is
exactly why this survived: ChromBPNet, the most audited oracle, cannot show it.

**Which convention wins, and why.** The builders' — for three reasons, not just
because the null is the expensive artefact:

* it is deterministic, so the same window always means the same span;
* it is symmetric about the variant, which is what a "centred window" should mean;
* 501 bp is simply *not representable* at 128 bp resolution (501/128 = 3.9), so
  no convention delivers 501 bp there. Given that, the honest move is to pick one,
  apply it on both sides, and record the effective span in provenance (#124) —
  rather than let the query silently claim a width the null never used.
"""
from __future__ import annotations

import pytest

from chorus.analysis.background_sampling import centered_bin_span

# (oracle, resolution, window_bp, expected bins) — the builders' convention,
# read off scripts/build_backgrounds_{enformer,borzoi,alphagenome}.py.
PRODUCTION_SPANS = [
    ("chrombpnet", 1, 501, 501),
    ("chrombpnet", 1, 2001, 2001),
    ("alphagenome", 1, 501, 501),
    ("alphagenome", 128, 501, 3),
    ("alphagenome", 128, 2001, 15),
    ("borzoi", 32, 501, 15),
    ("borzoi", 32, 2001, 63),
    ("enformer", 128, 501, 3),
    ("enformer", 128, 2001, 15),
]


def _builder_span(window_bp: int, resolution: int, n_bins: int) -> int:
    """The arithmetic every binned builder uses, transcribed."""
    centre = n_bins // 2
    hw = window_bp // (2 * resolution)
    return min(n_bins, centre + hw + 1) - max(0, centre - hw)


def _query_span(window_bp: int, resolution: int, offset: int) -> int:
    """scorers.py -> core/result.py, transcribed, for a variant at sub-bin ``offset``."""
    pos = 4096 * resolution + offset
    half = window_bp // 2
    start, end = pos - half, pos + half + 1
    start_bin = start // resolution
    end_bin = (end + resolution - 1) // resolution
    return end_bin - start_bin


@pytest.mark.parametrize("oracle,res,window,expected", PRODUCTION_SPANS)
def test_shared_helper_matches_the_builders(oracle, res, window, expected):
    """The shared function must reproduce every builder exactly.

    If it does not, adopting it silently moves a shipped background — the failure
    mode that makes this whole class expensive.
    """
    start, end = centered_bin_span(8192, window, res)
    assert end - start == expected == _builder_span(window, res, 8192)


@pytest.mark.parametrize("oracle,res,window,expected", PRODUCTION_SPANS)
def test_shared_helper_is_deterministic_and_centred(oracle, res, window, expected):
    start, end = centered_bin_span(8192, window, res)
    centre = 8192 // 2
    assert start <= centre < end, "span must contain the centre bin"
    # symmetric about the centre bin
    assert centre - start == end - 1 - centre


def test_the_query_span_varies_with_sub_bin_offset():
    """The defect that makes the null unmatchable in principle.

    Not a builder-vs-query disagreement — an internal inconsistency in the query.
    """
    for res, window in ((128, 501), (128, 2001), (32, 501), (32, 2001)):
        spans = {_query_span(window, res, off) for off in range(res)}
        assert len(spans) > 1, (
            f"res={res} w={window}: expected the query span to vary, got {spans}"
        )


def test_resolution_1_is_why_this_hid():
    """ChromBPNet cannot exhibit the bug, so the most audited oracle was clean."""
    for window in (501, 2001):
        spans = {_query_span(window, 1, 0)}
        builder = _builder_span(window, 1, 8192)
        assert spans == {builder} == {window}


@pytest.mark.parametrize("res,window", [(128, 501), (128, 2001), (32, 501)])
def test_shared_helper_is_narrower_than_every_query_span(res, window):
    """Pins the direction: the null summed LESS than the numerator did."""
    start, end = centered_bin_span(8192, window, res)
    query_spans = {_query_span(window, res, off) for off in range(res)}
    assert (end - start) < min(query_spans)


def test_window_wider_than_the_prediction_returns_everything():
    """Matches what each builder did separately, rather than raising."""
    assert centered_bin_span(10, 100_000, 1) == (0, 10)


def test_odd_bin_count_always():
    """A centred window has a centre bin plus equal flanks, so the count is odd."""
    for res in (1, 32, 128):
        for window in (501, 1000, 2001):
            start, end = centered_bin_span(8192, window, res)
            assert (end - start) % 2 == 1


def test_window_none_passes_through():
    """Borzoi's RNA tracks carry ``window=None`` and must get the whole array."""
    assert centered_bin_span(6144, None, 32) == (0, 6144)


# ---------------------------------------------------------------------------
# The builders must delegate, not keep a fourth copy (#144)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", [
    "scripts/build_backgrounds_enformer.py",
    "scripts/build_backgrounds_borzoi.py",
    "scripts/build_backgrounds_alphagenome.py",
])
def test_builders_delegate_to_the_shared_span(builder):
    """Source-text assertion, the ``tests/test_cherimoya.py:609`` pattern.

    Three copies of ``hw = window // (2 * resolution)`` is how the query was
    allowed to disagree with all of them at once.
    """
    from pathlib import Path

    src = Path(builder).read_text()
    assert "centered_bin_span" in src, f"{builder} must import the shared span"
    assert "return centered_bin_span(" in src, f"{builder} must delegate to it"
    assert "hw = track['window'] // (2 *" not in src, \
        f"{builder} still computes its own half-width"


def test_shared_span_reproduces_the_old_builder_arithmetic_exactly():
    """Adopting the shared function must not move a single shipped background.

    This is the gate that makes the migration safe: the arithmetic the builders
    used to carry, transcribed, against the shared function over every production
    shape.
    """
    def old(n_bins, window, res):
        if window is None:
            return 0, n_bins
        centre = n_bins // 2
        hw = window // (2 * res)
        return max(0, centre - hw), min(n_bins, centre + hw + 1)

    for n_bins in (896, 6144, 8192, 1_048_576):
        for res in (1, 32, 128):
            for window in (None, 501, 1000, 2001, 100_000):
                assert centered_bin_span(n_bins, window, res) == old(n_bins, window, res)
