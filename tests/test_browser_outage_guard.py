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
