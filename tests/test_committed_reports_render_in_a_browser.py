"""Every committed report must actually paint in a browser (#135).

Nothing in this suite had ever opened a report in a browser. The checks were on bytes and
JSON: does the file exist, is it under the size ceiling, do the numbers in it match the
oracle. A report could therefore be regenerated, sized, diffed, committed and shipped while
rendering a blank page, and the only way anyone would find out is by opening it.

That gap is why #139 was filed claiming "the panel never renders" — the reporter measured
``document.querySelectorAll('canvas').length``, got 0, and reasonably concluded nothing was
drawn. IGV renders inside a **shadow root**, so that query returns 0 for a panel that is
painting perfectly. There was no instrument to check against.

**This also replaces size as the thing being policed.** ``_MAX_TRACKED_MIB`` was 20, chosen
as "headroom over today's largest artefact" rather than derived from any failure, and it
blocked a legitimate 25.70 MB report (the CDYL fine-map: 21 lung-fibroblast tracks x 2
alleles, exactly the track list the blog's Analysis B used). The ceiling is now 50 MiB — the
threshold above which GitHub warns, below its hard 100 MiB wall — and the question "is this
report too big" is answered by loading it rather than by weighing it.

Measured across the 18 committed IGV reports, which is what justifies that:

    file size       1.3 MiB .. 14.7 MiB     (11x)
    load to paint   8.8 s   .. 10.8 s       (1.2x)

Size is almost uncorrelated with load time, and the reason is visible in the request log:
every report spends most of those seconds on ~14 **network** round-trips for genome
resources (see the #139 test), not on parsing its own payload. igv.js does not render the
payload either — it indexes inline features into an interval tree, culls to the visible
window, and summarises to one value per screen pixel, so draw cost is bounded by viewport
width. A big report costs history and download, not interactivity.

Marked ``integration``: it needs Chromium and takes about three minutes.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

import browser_harness as bh

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parent.parent

#: Slowest committed report measured 10.8 s (14.7 MiB, 64 canvases). 30 s is roughly 3x
#: that: loose enough to survive a slower machine or a cold network, tight enough that a
#: report which has become genuinely unusable fails rather than merely logging a number.
_LOAD_BUDGET_S = 30.0

#: The smallest report has 6 measurable canvases (ideogram, ruler, one track, axes). Fewer
#: than that means the panel did not get built, which is distinct from being blank.
_MIN_CANVASES = 6


def _committed_reports() -> list:
    out = subprocess.check_output(["git", "ls-files", "examples/"], cwd=REPO, text=True)
    return sorted((REPO / p for p in out.split() if p.endswith(".html")),
                  key=lambda p: -p.stat().st_size)


REPORTS = _committed_reports()


@pytest.fixture(scope="module")
def browser():
    """One Chromium for the whole module -- launching per report triples the runtime."""
    why = bh.unavailable()
    if why:
        pytest.skip(why)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        b = pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"],
                               env=bh.browser_env())
        try:
            yield b
        finally:
            b.close()


_RENDERED: dict = {}


def _render_once(browser, report):
    """Render each report once and share the result across the checks on it.

    Both parametrized tests want the same load; doing it twice doubles a three-minute run
    and, worse, lets the two assertions disagree about the same file.
    """
    if report not in _RENDERED:
        _RENDERED[report] = bh.render(browser, report)
    return _RENDERED[report]


def test_there_are_reports_to_check():
    """A silent zero-parameter parametrize would make this whole file a no-op."""
    assert len(REPORTS) >= 19, f"expected the committed report set, found {len(REPORTS)}"


@pytest.mark.parametrize("report", REPORTS, ids=lambda p: p.name)
def test_every_committed_report_paints_every_track(browser, report):
    """The core check: no canvas that IGV laid out may be left with zero ink.

    Convergence is the assertion, not a stability heuristic. Polling until the painted count
    merely stops changing reported 61/62 and 39/42 on reports that reach 64/64 and 44/44 a
    second later, because two consecutive polls agree while IGV is still laying out
    viewports -- a flaky test that fails on the biggest reports, which are exactly the ones
    #135 is about.
    """
    r = _render_once(browser, report)

    assert not r.page_errors, f"{r.summary()}\nuncaught JS: {r.page_errors[:3]}"
    assert not r.console_errors, f"{r.summary()}\nconsole errors: {r.console_errors[:3]}"

    if not r.is_igv:
        # batch_sort1_locus_scoring.html is a table, checked separately below.
        assert r.text_length > 1000, f"{r.summary()}: no panel and almost no text either"
        return

    assert r.measured >= _MIN_CANVASES, (
        f"{r.summary()}: only {r.measured} canvases were laid out, expected at least "
        f"{_MIN_CANVASES}. The panel did not get built -- distinct from being blank."
    )
    assert not r.blank, (
        f"{r.summary()}\ncanvases with zero ink: {r.blank}\n"
        f"details: {[r.inks[i] for i in r.blank][:5]}\n"
        f"A track IGV failed to draw contains literally no ink. Note the sparsest REAL "
        f"track in the corpus is {r.sparsest} of its pixels (the causal report's point "
        f"tracks), so a fraction threshold would mistake sparse for broken."
    )
    assert r.converged, f"{r.summary()}: painted never reached measured within the timeout"


@pytest.mark.parametrize("report", REPORTS, ids=lambda p: p.name)
def test_no_committed_report_takes_absurdly_long_to_paint(browser, report):
    """"Large is fine if it still loads" is the policy; this is the "still loads" half."""
    r = _render_once(browser, report)
    assert r.seconds < _LOAD_BUDGET_S, (
        f"{r.summary()}: took {r.seconds:.1f}s to paint, over the {_LOAD_BUDGET_S:.0f}s "
        f"budget. The committed corpus measured 8.8-10.8s across 1.3-14.7 MiB when this "
        f"budget was set."
    )


def test_the_table_only_report_renders_its_table(browser):
    """One committed report has no IGV panel at all, and that is not a defect.

    ``batch_sort1_locus_scoring.html`` is a 40 kB table of five variants. Worth pinning
    because the naive version of this check waits 90 s for canvases that were never coming,
    then reports a timeout -- a no-op test that looks like a slow one.
    """
    batch = next((p for p in REPORTS if p.name == "batch_sort1_locus_scoring.html"), None)
    if batch is None:
        pytest.skip("batch report not committed")
    r = _render_once(browser, batch)
    assert not r.is_igv, "the batch report grew a panel; give it the same checks as the rest"
    assert r.text_length > 1000, f"{r.summary()}: the table rendered no text"
    assert r.seconds < 5.0, (
        f"{r.summary()}: a panel-less report should return immediately rather than waiting "
        f"out the canvas timeout"
    )


@pytest.mark.parametrize("label,mutate,expect", [
    # Emptying the inline features leaves IGV with nothing to draw: the axis canvases still
    # paint, the two data canvases do not. This is the failure the check exists for -- a
    # report that is present, well-formed, correctly sized, and shows nothing.
    ("features emptied",
     lambda s: re.subn(r'"features":\s*\[[^\]]*\]', '"features":[]', s)[0],
     "blank"),
    # A bad genome name makes igv.createBrowser throw. The report's own try/catch turns that
    # into console.error rather than an uncaught exception, so page_errors stays empty and
    # only the console channel sees it -- which is why both are checked.
    ("genome broken",
     lambda s: s.replace('"genome":"hg38"', '"genome":"nosuchgenome"', 1),
     "console"),
    # A syntax error never reaches the report's catch block at all.
    ("config truncated",
     lambda s: s.replace('const browser = await igv.createBrowser(',
                         'const browser = await igv.createBrowser(((', 1),
     "pageerror"),
])
def test_the_check_actually_detects_a_broken_panel(browser, tmp_path, label, mutate, expect):
    """Fails-without-fix, for the instrument rather than the code.

    A rendering check that has never been shown to catch a broken render is documentation.
    Each mutation below is caught by a *different* assertion, which is the argument for
    keeping all three channels (zero-ink canvases, console errors, uncaught exceptions)
    rather than picking whichever one seemed sufficient.
    """
    src = next(p for p in REPORTS if p.name == "rs12740374_SORT1_cherimoya_report.html")
    broken = tmp_path / f"{label.replace(' ', '_')}.html"
    broken.write_text(mutate(src.read_text()))

    r = bh.render(browser, broken, timeout_s=20)
    detected = {
        "blank": bool(r.blank),
        "console": bool(r.console_errors),
        "pageerror": bool(r.page_errors),
    }
    assert detected[expect], (
        f"{label}: the {expect} channel did not notice.\n  {r.summary()}\n"
        f"  blank={r.blank[:6]} console={r.console_errors[:1]} page={r.page_errors[:1]}"
    )
    assert not r.converged, f"{label}: a broken report should not converge -- {r.summary()}"


def test_file_size_is_not_what_costs_load_time(browser):
    """The evidence behind raising the ceiling, kept executable rather than in a comment.

    If this ever fails it means the relationship changed -- payload size started dominating
    load time -- and the reasoning that made 50 MiB acceptable needs re-deriving rather than
    the number being nudged.
    """
    igv = [p for p in REPORTS if p.stat().st_size > 1_000_000]
    biggest, smallest = igv[0], igv[-1]
    size_ratio = biggest.stat().st_size / smallest.stat().st_size
    assert size_ratio > 5, "the corpus no longer spans a wide enough size range to compare"

    big = _render_once(browser, biggest)
    small = _render_once(browser, smallest)
    time_ratio = big.seconds / max(small.seconds, 0.1)

    assert time_ratio < size_ratio / 2, (
        f"load time is now tracking file size: {size_ratio:.1f}x the bytes cost "
        f"{time_ratio:.1f}x the time.\n  {big.summary()}\n  {small.summary()}\n"
        f"When the 50 MiB ceiling was set, 11x the bytes cost 1.2x the time, because the "
        f"seconds go on genome-resource round-trips rather than on parsing the payload."
    )
