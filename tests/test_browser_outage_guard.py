"""The outage guard's own logic, checked without a browser.

`test_committed_reports_render_in_a_browser.py` carries `pytestmark = pytest.mark.integration` for
the obvious reason that it drives Chromium over a 19-report corpus. These checks need neither: they
feed a hand-built `RenderResult` to the pure function that decides whether a render is judgeable.

They live in their own module because a module-level mark applies to everything in the file, so
writing them next to the tests they protect would have deselected them from the fast suite — and a
guard that only runs when someone opts into the slow suite is not much of a guard. That function
decides whether CI is allowed to fail, so it should be checked on every push.
"""
from __future__ import annotations

from pathlib import Path

import browser_harness as bh

REPO = Path(__file__).resolve().parent.parent


def _fake(console=(), external=(), page=(), blank=()):
    """A RenderResult standing in for one render, so the skip logic is testable offline."""
    return bh.RenderResult(
        path=REPO / "examples" / "fake_report.html", mib=1.3, canvases=4, measured=0, painted=0,
        blank=list(blank), page_errors=list(page), console_errors=list(console),
        external_urls=list(external), seconds=60.1, is_igv=True, converged=False,
    )


#: Verbatim from the failing run on 2026-08-14, so the guard is pinned to the real signature
#: rather than to my paraphrase of it.
_REAL_OUTAGE = [
    "error: IGV error: Error accessing resource: "
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit Status: 0",
    "error: Failed to load resource: net::ERR_CONNECTION_REFUSED",
]
_UCSC = ["https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit"]


def test_an_unreachable_sequence_host_is_recognised():
    """The exact failure that turned a browser job red while the corpus was fine."""
    why = bh.unreachable_external_host(_fake(console=_REAL_OUTAGE, external=_UCSC))
    assert why and "hgdownload.soe.ucsc.edu" in why, (
        f"the 2026-08-14 outage signature was not recognised, so CI stays hostage to UCSC: {why!r}"
    )


def test_a_broken_report_is_still_a_failure():
    """The case this must never mask: blank canvases with nobody's server to blame."""
    assert bh.unreachable_external_host(_fake(blank=[0, 1], external=_UCSC)) is None, (
        "a report with blank canvases and no console errors was treated as an outage; a real "
        "blank-panel regression would then skip instead of failing, which is the whole risk here"
    )


def test_one_unexplained_error_disqualifies_the_skip():
    """A genuine JS fault alongside an outage must still fail — outages do not grant amnesty."""
    mixed = _REAL_OUTAGE + ["error: TypeError: undefined is not a function"]
    assert bh.unreachable_external_host(_fake(console=mixed, external=_UCSC)) is None, (
        "an unrelated console error was swallowed because an outage signature was also present"
    )


def test_errors_without_any_external_request_are_never_an_outage():
    """No external fetch attempted means no third party to blame."""
    assert bh.unreachable_external_host(_fake(console=_REAL_OUTAGE)) is None


def test_a_clean_render_needs_no_excuse():
    assert bh.unreachable_external_host(_fake()) is None


# ── the three holes an adversarial review found in the first version ─────────────

def test_an_uncaught_js_exception_is_never_excused_by_an_outage():
    """The worst of the three: callers skip BEFORE asserting page_errors.

    A report throwing an uncaught exception was skipped whenever its console errors happened to be
    outage-shaped — so a genuine JS fault in a committed report could hide behind UCSC being down.
    A remote host refusing a connection cannot cause an uncaught exception in page code.
    """
    r = _fake(console=_REAL_OUTAGE, external=_UCSC,
              page=["TypeError: Cannot read properties of undefined (reading 'tracks')"])
    assert bh.unreachable_external_host(r) is None, (
        "a render with uncaught JS was treated as an outage; the paint test skips before it asserts "
        "page_errors, so that fault would never be reported"
    )


def test_a_failing_third_party_url_is_not_an_outage_excuse():
    """The skip once required only that *some* external request had been attempted.

    All 19 reports contact UCSC, so that condition was satisfied essentially always — meaning one
    unrelated refused request disabled every verdict for the report, including blank-canvas.
    """
    r = _fake(
        console=["error: IGV error: Error accessing resource: "
                 "https://cdn.example.com/tracks/mystery.bw Status: 0"],
        external=["https://cdn.example.com/tracks/mystery.bw"] + _UCSC,
    )
    assert bh.unreachable_external_host(r) is None, (
        "a report failing to fetch a non-reference third-party URL was skipped. That is a defect in "
        "the report — it should not ship a dependency on cdn.example.com — and must fail, not skip."
    )


def test_a_typoed_reference_host_committed_in_a_report_still_fails():
    """A wrong host baked into a report is permanent for every reader, not weather."""
    bad = "https://hgdownload.soe.ucsc.edu.example.net/goldenPath/hg38/bigZips/hg38.2bit"
    r = _fake(console=[f"error: IGV error: Error accessing resource: {bad} Status: 0"], external=[bad])
    why = bh.unreachable_external_host(r)
    assert why is None, (
        f"a report citing {bad} was skipped as an outage. A typo'd or hijacked host is a permanent "
        f"defect for every reader and must fail. (got: {why})"
    )


def test_the_real_reference_url_still_skips():
    """The whole point: the genuine UCSC outage must still be tolerated."""
    r = _fake(console=_REAL_OUTAGE, external=_UCSC)
    why = bh.unreachable_external_host(r)
    assert why and "hgdownload.soe.ucsc.edu" in why, (
        f"tightening the guard broke the case it exists for: {why!r}"
    )


def test_the_skip_message_names_only_hosts_that_actually_failed():
    """The old message listed every contacted host and asserted the sequence never loaded.

    With the URL now extracted from the error itself, the message can only name what failed.
    """
    r = _fake(console=_REAL_OUTAGE, external=_UCSC + ["https://fonts.googleapis.com/css"])
    why = bh.unreachable_external_host(r) or ""
    assert "fonts.googleapis.com" not in why, (
        f"the skip message blames a host that did not fail: {why!r}"
    )


def test_igv_reports_a_timeout_in_its_own_words():
    """The exact console pair that failed CI on 2026-08-16, on a change touching no browser code.

    UCSC timed out and igv.js emitted TWO errors: Chromium's "net::ERR_TIMED_OUT" (which the signature
    list matched) and igv.js's own "IGV error: Timed out" (which it did not). Because one unmatched
    error disqualifies the entire skip, the guard from #206/#210 stayed silent and a third-party
    timeout failed the build -- the precise outcome that guard exists to prevent.
    """
    r = _fake(console=["error: IGV error: Timed out",
                       "error: Failed to load resource: net::ERR_TIMED_OUT"],
              external=_UCSC)
    why = bh.unreachable_external_host(r)
    assert why and "hgdownload.soe.ucsc.edu" in why, (
        f"igv.js's own timeout wording is not recognised as an outage, so a UCSC timeout fails the "
        f"build instead of skipping: {why!r}"
    )


def test_the_igv_timeout_signature_did_not_widen_the_guard():
    """Adding a signature is where over-broadness creeps in; the three refusals must still hold."""
    assert bh.unreachable_external_host(_fake(
        console=["error: IGV error: Timed out"], external=_UCSC,
        page=["TypeError: Cannot read properties of undefined"])) is None, "uncaught JS was excused"
    assert bh.unreachable_external_host(_fake(
        console=["error: IGV error: Timed out"],
        external=["https://cdn.example.com/t.bw"])) is None, "a foreign host was excused"
    assert bh.unreachable_external_host(_fake(
        console=["error: something genuinely broke in the report"],
        external=_UCSC)) is None, "an unrelated console error was excused"
