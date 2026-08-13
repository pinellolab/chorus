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

import ast
import subprocess
from pathlib import Path

import pytest

from chorus.analysis.background_sampling import centered_bin_span


def _code_fingerprint(source: str) -> str:
    """A module's AST with every docstring removed, so prose changes are invisible.

    Comments never reach the AST at all, and ``ast.dump`` omits line numbers unless
    asked for them, so reformatting and blank-line churn are invisible too.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:]
    return ast.dump(tree)


REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=True).stdout


def _last_semantic_change(path: str) -> str:
    """ISO commit date of the newest commit that changed *path*'s code.

    Commits that only touched docstrings or comments are skipped. Returns the oldest
    known commit date if every commit in range was prose-only, and "" if the file has
    no history (a shallow clone), which the caller treats as a skip.
    """
    # A shallow clone cannot answer this question at all. `actions/checkout@v4` defaults to
    # `fetch-depth: 1`, so CI has exactly one commit: every path looks like it was *added* there,
    # dated today, and every committed example therefore reads as stale. Detect it and let the
    # caller skip, rather than reporting a staleness that is an artefact of the checkout.
    if (REPO_ROOT / ".git" / "shallow").exists():
        return ""
    shas = _git("log", "--format=%H", "--", path).split()
    if not shas:
        return ""
    for sha in shas:                      # newest first
        try:
            after = _code_fingerprint(_git("show", f"{sha}:{path}"))
            before = _code_fingerprint(_git("show", f"{sha}^:{path}"))
        except (subprocess.CalledProcessError, SyntaxError):
            # No parent (initial commit) or unparseable revision — treat as a real change.
            return _git("log", "-1", "--format=%cI", sha).strip()
        if after != before:
            return _git("log", "-1", "--format=%cI", sha).strip()
    return _git("log", "-1", "--format=%cI", shas[-1]).strip()

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


def _make_track(resolution: int, n_bins: int = 4096, pred_start: int = 1_000_000):
    """A minimal real OraclePredictionTrack with distinguishable per-bin values."""
    import numpy as np

    from chorus.core.interval import GenomeRef, Interval
    from chorus.core.result import OraclePredictionTrack

    values = np.arange(n_bins, dtype=np.float64)
    # ``fasta`` is required by GenomeRef but never read: score_region, pos2bin and
    # score_centered_window touch only chrom/start/end.
    ref = GenomeRef(
        chrom="chr1", start=pred_start, end=pred_start + n_bins * resolution,
        fasta="/nonexistent.fa",
    )
    interval = Interval.make(ref) if hasattr(Interval, "make") else Interval(reference=ref)
    return OraclePredictionTrack(
        source_model="test",
        assay_id="TEST:track",
        assay_type="DNASE",
        cell_type="TEST",
        query_interval=interval,
        prediction_interval=interval,
        input_interval=interval,
        resolution=resolution,
        values=values,
    )


# ---------------------------------------------------------------------------
# The query switch: unchanged at resolution 1, corrected above it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window", [501, 2001])
def test_resolution_1_query_is_bit_identical_before_and_after(window):
    """Why ChromBPNet's committed numbers do not move.

    ``score_region`` and ``score_centered_window`` must agree exactly at 1 bp for
    an odd window, so the switch is a no-op for every ``resolution=1`` oracle.
    """
    track = _make_track(1)
    pos = track.prediction_interval.reference.start + 2048
    half = window // 2
    before = track.score_region("chr1", pos - half, pos + half + 1, "sum")
    after = track.score_centered_window("chr1", pos, window, "sum")
    assert before == after


@pytest.mark.parametrize("res,window", [(128, 501), (128, 2001), (32, 501)])
def test_coarse_resolution_query_changes_and_now_matches_the_builder(res, window):
    """At coarse resolution the old query summed a wider span than the null."""
    track = _make_track(res)
    pos = track.prediction_interval.reference.start + 2048 * res + res // 3
    half = window // 2
    before = track.score_region("chr1", pos - half, pos + half + 1, "sum")
    after = track.score_centered_window("chr1", pos, window, "sum")
    assert before != after

    expected_bins = centered_bin_span(4096, window, res)
    n = expected_bins[1] - expected_bins[0]
    # the new value sums exactly the builder's number of bins
    centre = track.pos2bin("chr1", pos)
    lo = centre - window // (2 * res)
    assert after == float(sum(range(lo, lo + n)))


def test_the_query_is_now_deterministic_across_sub_bin_offsets():
    """The defect this closes: identical settings used to give different spans."""
    res, window = 128, 501
    track = _make_track(res)
    base = track.prediction_interval.reference.start + 2048 * res
    old = {track.score_region("chr1", base + o - window // 2,
                              base + o + window // 2 + 1, "sum")
           for o in range(0, res, 16)}
    new = {track.score_centered_window("chr1", base + o, window, "sum")
           for o in range(0, res, 16)}
    assert len(old) > 1, "old query should vary with sub-bin offset"
    assert len(new) == 1, f"new query must be offset-invariant, got {sorted(new)}"


def test_centred_window_returns_none_outside_the_prediction():
    track = _make_track(128)
    assert track.score_centered_window("chr1", 1, 501, "sum") is None
    assert track.score_centered_window("chr2", 1_000_000 + 500, 501, "sum") is None


def test_centre_bin_override_is_what_the_query_needs():
    """A prediction clamped at a contig edge is not centred on the variant.

    The builders can assume centre == n_bins // 2 because they centre each
    sampled sequence on the variant themselves. The query cannot.
    """
    # centre 500, half 250 -> [250, 751); clamped at the left when centre is 100
    assert centered_bin_span(1000, 501, 1) == (250, 751)
    assert centered_bin_span(1000, 501, 1, centre_bin=100) == (0, 351)


def test_prose_only_edits_do_not_make_the_examples_look_stale():
    """The staleness guard must key off code, not off any commit touching the file.

    #187 changed exactly one line of ``scorers.py``'s module docstring and thereby
    declared all 14 committed examples stale — a false positive that would have cost a
    multi-hour regeneration producing byte-identical numbers. Both directions are
    asserted here: prose churn is invisible, real code is not.
    """
    base = "x = 1\n\n\ndef f():\n    \"\"\"Doc.\"\"\"\n    return x + 1\n"
    prose = "x = 1\n\n\ndef f():\n    \"\"\"Totally different words.\"\"\"\n    # and a comment\n    return x + 1\n"
    code = "x = 1\n\n\ndef f():\n    \"\"\"Doc.\"\"\"\n    return x + 2\n"

    assert _code_fingerprint(base) == _code_fingerprint(prose), \
        "a docstring/comment edit changed the fingerprint — the guard will false-positive"
    assert _code_fingerprint(base) != _code_fingerprint(code), \
        "a real code change did NOT change the fingerprint — the guard is now blind"


@pytest.mark.integration
def test_the_staleness_guard_skips_the_prose_commit_in_real_history():
    """Against this repo's actual history, not a synthetic string.

    ``scorers.py``'s newest commit is #187's docstring fix; the newest *semantic* one is
    #163. If these ever coincide again the guard is back to keying off prose.
    """
    path = "chorus/analysis/scorers.py"
    try:
        semantic = _last_semantic_change(path)
        touched = _git("log", "-1", "--format=%cI", "--", path).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git unavailable")
    if not semantic or not touched:
        pytest.skip("no git history (shallow clone?)")
    assert semantic <= touched
    newest_sha = _git("log", "--format=%H", "--", path).split()[0]
    if _code_fingerprint(_git("show", f"{newest_sha}:{path}")) == \
       _code_fingerprint(_git("show", f"{newest_sha}^:{path}")):
        assert semantic != touched, (
            "the newest commit to scorers.py is prose-only, yet the guard still dated "
            "the module from it — every committed example would read as stale"
        )


@pytest.mark.integration
def test_committed_examples_are_stale_until_the_regen_sweep():
    """Makes the staleness of shipped examples *visible* rather than silent.

    Switching the query to the builders' span moves **258 of 1,090** windowed rows
    across the committed examples (23.7 %):

    ======================================  =============  =======
    example                                 windowed rows  changed
    ======================================  =============  =======
    SORT1_enformer                                    168      168
    SORT1_cell_type_screen                            288       44
    TERT_chr5_1295046                                 114       12
    SORT1_rs12740374_with_CEBP                        130       10
    BCL11A_rs1427407 / FTO_rs1421085                   58       12
    SORT1_rs12740374 / region_swap / others           ...        ...
    SORT1_chrombpnet                                    2        0
    batch_scoring                                      30        0
    ======================================  =============  =======

    ChromBPNet and batch_scoring are untouched because they are ``resolution=1``,
    which is the same reason the bug hid for months.

    Regenerating per number-moving PR costs GPU and gets overwritten by the next
    one, so the plan regenerates **once**, after the rebuilds. Until then the
    committed examples do not match what the code produces — and nothing else in
    the suite notices, which is the gap this test closes. Expected to report
    staleness until that sweep runs; ``integration``-marked so it never blocks CI.
    """
    import glob
    import json
    import os
    import subprocess
    from datetime import datetime, timezone

    # git commit time, NOT mtime: on a fresh clone every file carries checkout
    # time, so an mtime comparison would call every example stale everywhere.
    #
    # And the most recent commit that *touched* scorers.py is the wrong question — a
    # docstring edit cannot move a number. #187 changed one line of this module's
    # docstring (`~/.chorus/backgrounds` -> `<data-dir>/backgrounds`) and that alone
    # declared all 14 committed examples stale, which would have bought a multi-hour
    # regeneration producing byte-identical output. So walk back to the last commit
    # that changed the module's *semantics*.
    try:
        iso = _last_semantic_change("chorus/analysis/scorers.py")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git unavailable")
    if not iso:
        pytest.skip("no git history for scorers.py (shallow clone?)")
    src_mtime = datetime.fromisoformat(iso).astimezone(timezone.utc)

    stale = []
    for path in sorted(glob.glob("examples/**/example_output.json", recursive=True)):
        with open(path) as fh:
            data = json.load(fh)
        stamp = (data.get("analysis_request") or {}).get("generated_at")
        if not stamp:
            stale.append((os.path.basename(os.path.dirname(path)), "no generated_at"))
            continue
        try:
            gen = datetime.strptime(stamp, "%Y-%m-%d %H:%M UTC").replace(
                tzinfo=timezone.utc)
        except ValueError:
            stale.append((os.path.basename(os.path.dirname(path)), f"unparsable {stamp!r}"))
            continue
        if gen < src_mtime:
            stale.append((os.path.basename(os.path.dirname(path)), stamp))

    assert not stale, (
        "examples predate the current scorers.py and need the regen sweep:\n  "
        + "\n  ".join(f"{name}: {why}" for name, why in stale)
    )


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
