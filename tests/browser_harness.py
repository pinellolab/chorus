"""Load a chorus HTML report in headless Chromium and measure what actually painted.

Not a test module (no ``test_`` prefix, so pytest does not collect it). Two test files
depend on it, and it exists because **nothing in the suite had ever opened a report in a
browser**: every check was on the file's bytes or its JSON, so a report could be committed,
sized, diffed and shipped while rendering nothing at all.

Three things it took a while to get right, recorded here because each one produces a
failure that reads exactly like "the panel is broken":

1. **Chromium needs shared libraries conda scatters across environments.** ``libgbm`` and
   ``libatk`` live only in ``chorus-browsertest``; ``libXcomposite`` and ``libcups`` only in
   the oracle envs and the base env. With either missing, chrome exits 127 with
   ``error while loading shared libraries`` before a single page loads. So the path is built
   here and passed to the *browser process* via ``launch(env=...)`` rather than being
   exported by whoever runs pytest — a check that depends on the caller remembering an
   environment variable is a check that silently stops working.

2. **IGV renders inside a shadow root.** ``document.querySelectorAll('canvas')`` returns
   **0** for a report that is painting perfectly, because ``querySelectorAll`` does not
   pierce shadow boundaries. Measuring that number and concluding the panel was blank is
   what made #139 originally claim "the panel never renders". :data:`_MEASURE_JS` walks
   shadow roots.

3. **"Loaded" is not "painted".** ``wait_until="load"`` fires while IGV is still building
   tracks asynchronously, and a fixed sleep is either slow or flaky depending on report
   size. Polling until the count merely *stops changing* is not enough either — it reported
   61/62 and 39/42 painted on reports that reach 62/62 and 42/42 given another second,
   because two consecutive polls can agree while IGV is still laying out viewports. So this
   polls to **convergence** (``painted == measured``) and records whether it got there.

Not every committed report has a panel: ``batch_sort1_locus_scoring.html`` is a 40 kB table
of five variants with no IGV at all, and waiting 90 s for canvases that were never coming is
how a no-op test looks like a slow one. :func:`render` detects that and returns immediately.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parent.parent

#: Conda prefixes whose ``lib`` may carry a Chromium dependency, most specific first.
_LIB_CANDIDATES = (
    Path("/home/nvidia/miniforge3/envs/chorus-browsertest/lib"),
    Path(sys.prefix) / "lib",
)

#: Count canvases and how much of each is painted, piercing shadow roots.
#:
#: "Painted" is **any** non-white, non-transparent pixel. A fraction threshold was tried
#: first and 0.05% was too high: the causal report's "Composite Causal Score" and
#: "Max |Effect|" tracks are *sparse point* tracks — roughly 20 marks spread across a 3288 px
#: canvas — and measure 0.000238 and 0.000359, so they were being reported as blank while
#: rendering exactly what they should. Since a track IGV failed to draw contains literally
#: zero ink, "> 0" separates the two cases without a magic number, and the margin is on
#: record: the sparsest real track is ~782 painted pixels out of 197,280.
#:
#: Zero-dimension canvases are skipped rather than counted as failures: IGV allocates some
#: before laying them out.
_MEASURE_JS = """() => {
  const canvases = [];
  const walk = (root) => {
    for (const c of root.querySelectorAll('canvas')) canvases.push(c);
    for (const el of root.querySelectorAll('*')) if (el.shadowRoot) walk(el.shadowRoot);
  };
  walk(document);
  let measured = 0, painted = 0;
  const inks = [], blank = [];
  for (const c of canvases) {
    if (!c.width || !c.height) continue;
    const idx = measured++;
    try {
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      let ink = 0;
      for (let i = 0; i < d.length; i += 4)
        if (d[i+3] > 8 && !(d[i] > 247 && d[i+1] > 247 && d[i+2] > 247)) ink++;
      inks.push({px: ink, frac: +(ink / (c.width * c.height)).toFixed(6),
                 w: c.width, h: c.height});
      if (ink > 0) painted++; else blank.push(idx);
    } catch (e) { inks.push({error: String(e).slice(0, 60)}); }
  }
  return {total: canvases.length, measured, painted, inks, blank};
}"""


@dataclass
class RenderResult:
    """What one report did in a browser."""

    path: Path
    mib: float
    canvases: int             # every <canvas>, including un-laid-out ones
    measured: int             # those with non-zero dimensions
    painted: int              # those with ink
    inks: list = field(default_factory=list)
    #: Indices of measurable canvases with zero ink -- the symptom of a broken panel.
    #: Computed in the browser, not re-derived from :attr:`inks`, so there is exactly one
    #: definition of "blank" rather than two that can drift apart.
    blank: list = field(default_factory=list)
    page_errors: list = field(default_factory=list)
    console_errors: list = field(default_factory=list)
    external_urls: list = field(default_factory=list)
    seconds: float = 0.0
    is_igv: bool = True       # False for the table-only batch report
    converged: bool = False   # reached painted == measured before the timeout
    text_length: int = 0      # rendered body text, the only signal a table report gives

    @property
    def external_hosts(self) -> dict:
        hosts: dict = {}
        for url in self.external_urls:
            hosts.setdefault(urlparse(url).netloc, []).append(url)
        return hosts

    @property
    def sparsest(self) -> float:
        """Smallest non-zero painted fraction, for calibrating what "blank" means."""
        fracs = [i["frac"] for i in self.inks if isinstance(i, dict) and i.get("frac")]
        return min(fracs) if fracs else 0.0

    def summary(self) -> str:
        hosts = ", ".join(f"{h} x{len(u)}" for h, u in self.external_hosts.items())
        what = (f"canvases {self.painted}/{self.measured} painted"
                f"{'' if self.converged else ' (NOT converged)'}"
                if self.is_igv else f"no panel, {self.text_length} chars of text")
        return (f"{self.path.name}  {self.mib:.1f} MiB  {self.seconds:.1f}s  {what}"
                + (f"  external: {hosts}" if hosts else "  external: none"))


#: Console-error signatures that mean "a host did not answer", not "this report is broken".
#: Chromium reports a refused/blocked fetch as ERR_CONNECTION_REFUSED and igv.js surfaces the
#: same event as `Status: 0`, so both spellings of the one condition have to be recognised.
UNREACHABLE_SIGNATURES = (
    "ERR_CONNECTION_REFUSED", "ERR_NAME_NOT_RESOLVED", "ERR_INTERNET_DISCONNECTED",
    "ERR_TIMED_OUT", "ERR_CONNECTION_TIMED_OUT", "ERR_ADDRESS_UNREACHABLE",
    "ERR_CONNECTION_RESET", "ERR_NETWORK_CHANGED", "Status: 0",
)


#: The only host a committed report is *expected* to fetch from: igv.js needs a sequence source and
#: hg38 is ~3 GB, so the reference two-bit is resolved from UCSC at open time. Anything else failing
#: is a defect in the report, not weather — so it must not qualify for a skip.
#:
#: Compared as an **exact netloc**, never as a substring. `"hgdownload.soe.ucsc.edu" in url` also
#: matches `hgdownload.soe.ucsc.edu.example.net`, which would let a typo'd or hijacked domain inherit
#: the trusted host's exemption. That was the first implementation, and this file's own test caught it.
REFERENCE_SEQUENCE_HOSTS = frozenset({"hgdownload.soe.ucsc.edu"})


def _urls_in(message: str) -> list:
    """URLs named inside a console error, e.g. igv.js's "Error accessing resource: <url> Status: 0"."""
    import re

    return re.findall(r"https?://[^\s\"'<>]+", message)


def unreachable_external_host(r) -> "str | None":
    """Why this render cannot be judged, or None if it can.

    Every committed report resolves its reference sequence from `hgdownload.soe.ucsc.edu`: igv.js
    requires a sequence source, hg38 is ~3 GB, and bundling it in a repo is not an option — a
    documented limitation rather than a defect. The consequence was that this file's verdict depended
    on a third party being up: on 2026-08-14 UCSC refused the connection during one PR's run and the
    same corpus that had passed minutes earlier reported `canvases 0/0 painted (NOT converged)` and
    blew the 30 s budget at the 60 s timeout — four failures, none of them about the reports.

    A suite that goes red when someone else's server hiccups trains people to re-run CI until it is
    green, which is how a real blank-panel regression gets waved through. But the first version of
    this function was far too generous, and an adversarial review found three ways a genuine defect
    could take the skip. All three are now closed:

    * **Uncaught JS was swallowed.** The old version looked only at `console_errors`, and the callers
      skip *before* asserting `page_errors` — so a report throwing an uncaught exception was skipped
      whenever the console errors happened to be outage-shaped. An uncaught error is now
      disqualifying on its own: it cannot be caused by a remote host refusing a connection.
    * **Any failing external URL counted.** The old version required only that *some* external
      request had been attempted. Since all 19 reports contact UCSC, that was satisfied
      essentially always, so one unrelated refused request disabled every verdict for that report.
      Now the URLs named in the failures must all be reference-sequence URLs.
    * **A wrong URL committed inside a report looked like weather.** A typo'd or third-party host
      baked into a report is a permanent defect for every reader, and it produced the same console
      error as an outage. Restricting the skip to `REFERENCE_SEQUENCE_HOSTS` means a report citing
      anything else fails instead of being skipped forever.

    One case remains genuinely undecidable and is called out rather than papered over: a *correct*
    reference URL that is permanently dead looks exactly like a transient outage. Nothing observable
    in one render distinguishes them, so it skips. If UCSC ever retires that path, every report will
    skip rather than fail, and this docstring is the note that says to look here.
    """
    if not r.console_errors or not r.external_urls:
        return None

    # An uncaught JS exception is never explained by an unreachable host.
    if getattr(r, "page_errors", None):
        return None

    unexplained = [e for e in r.console_errors
                   if not any(sig in e for sig in UNREACHABLE_SIGNATURES)]
    if unexplained:
        return None

    # Every URL implicated in the failures must be one we expect a report to fetch. When a message
    # names no URL (Chromium's bare "Failed to load resource: net::ERR_*"), fall back to requiring
    # that every external request this report made was a reference-sequence one.
    named = [u for e in r.console_errors for u in _urls_in(e)]
    implicated = named or list(r.external_urls)
    hosts = {urlparse(u).netloc for u in implicated}
    foreign = sorted(h for h in hosts if h not in REFERENCE_SEQUENCE_HOSTS)
    if foreign:
        return None

    where = ", ".join(sorted(hosts)) or "the reference host"
    return (
        f"{r.path.name}: {where} did not answer, so the reference sequence could not load. Not a "
        f"report defect -- igv.js needs a sequence source and hg38 is too large to bundle. "
        f"Errors: {r.console_errors[:2]}"
    )


def unavailable() -> "str | None":
    """Why a browser check cannot run here, or None if it can."""
    try:
        import playwright  # noqa: F401
    except ImportError:
        return "playwright not installed (pip install playwright)"
    cache = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH",
                                Path.home() / ".cache" / "ms-playwright"))
    if not cache.is_dir() or not any(cache.glob("chromium*")):
        return f"no chromium in {cache} (playwright install chromium)"
    if not any(p.is_dir() for p in _LIB_CANDIDATES):
        return f"none of {[str(p) for p in _LIB_CANDIDATES]} exists for Chromium's libraries"
    return None


def browser_env() -> dict:
    """Environment for the Chromium process, with the library path assembled."""
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [str(p) for p in _LIB_CANDIDATES if p.is_dir()]
    if existing:
        parts.append(existing)
    return dict(os.environ, LD_LIBRARY_PATH=":".join(parts))


def render(
    browser,
    path,
    *,
    block_external: bool = False,
    allow_hosts: "tuple | None" = None,
    poll_ms: int = 500,
    timeout_s: float = 60.0,
) -> RenderResult:
    """Open *path*, wait for the panel to converge, and report what painted.

    *block_external* aborts every non-``file:`` request, which is how the offline claim is
    tested: a report that needs the network renders differently, or not at all, with it cut.
    The URLs are still recorded, so a blocked run says what it *would* have fetched.

    *allow_hosts* exempts specific hosts from that block, which is how a *self-hosted*
    resource is tested — an air-gapped site serving its own reference is offline in every
    sense that matters, and blocking localhost too would make that indistinguishable from
    depending on the internet.
    """
    path = Path(path).resolve()
    result = RenderResult(path=path, mib=path.stat().st_size / (1024 * 1024),
                          canvases=0, measured=0, painted=0)
    page = browser.new_page(viewport={"width": 1600, "height": 1200})
    try:
        page.on("pageerror", lambda e: result.page_errors.append(str(e)[:300]))
        page.on("console", lambda m: (
            result.console_errors.append(f"{m.type}: {m.text[:200]}")
            if m.type == "error" else None))

        def _record(route):
            url = route.request.url
            if url.startswith("file:"):
                route.continue_()
                return
            result.external_urls.append(url)
            exempt = allow_hosts and urlparse(url).netloc in allow_hosts
            route.abort() if (block_external and not exempt) else route.continue_()

        page.route("**/*", _record)

        started = time.time()
        page.goto(path.as_uri(), wait_until="load", timeout=int(timeout_s * 1000))
        result.text_length = page.evaluate("document.body.innerText.length")

        # A report with no IGV container has no panel to wait for. Returning here rather
        # than polling for canvases is the difference between 0.3 s and a 90 s timeout.
        result.is_igv = page.evaluate(
            "!!(document.getElementById('igv-div') || document.querySelector('[id*=igv]'))")
        if not result.is_igv:
            result.converged = True
            result.seconds = time.time() - started
            return result

        # Converge, don't merely settle: two consecutive equal polls happen mid-layout.
        m = {"total": 0, "measured": 0, "painted": 0, "inks": [], "blank": []}
        while time.time() - started < timeout_s:
            page.wait_for_timeout(poll_ms)
            m = page.evaluate(_MEASURE_JS)
            if m["measured"] and m["painted"] == m["measured"]:
                result.converged = True
                break
        result.canvases, result.measured = m["total"], m["measured"]
        result.painted, result.inks = m["painted"], m["inks"]
        result.blank = m["blank"]
        result.seconds = time.time() - started
        return result
    finally:
        page.close()
