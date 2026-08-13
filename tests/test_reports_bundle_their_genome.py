"""A report must not resolve its genome through somebody else's web service (#139).

``genome: "hg38"`` reads like a setting. It is a **registry lookup**: igv.js resolves the
string against its hosted catalogue and follows the result, so every shipped report opened
six remote resources across two hosts —

    igv.org/genomes/genomes.json                     the catalogue
    igv.org/genomes/data/hg38/hg38_alias.tab         chromosome aliases
    hgdownload.soe.ucsc.edu .../hg38.chrom.sizes     chromosome lengths
    hgdownload.soe.ucsc.edu .../cytoBandIdeo.txt.gz  the ideogram
    hgdownload.soe.ucsc.edu .../ncbiRefSeq.txt.gz    the gene track
    hgdownload.soe.ucsc.edu .../hg38.2bit            the sequence, ranged

— 14 requests, and the catalogue fetch is **fatal**: with the network cut the panel does not
degrade, it never appears. So the claim that inlining igv.min.js made reports "viewable
offline, on air-gapped hosts" was false in the strongest available sense.

Measured on one report, before and after:

    before   14 requests   2 hosts   9.6 s to paint   offline: dies on genomes.json
    after     9 requests   1 host    2.2 s to paint   offline: dies on hg38.2bit
    after, sequence self-hosted same-origin:
              0 requests   0 hosts   0.8 s to paint   offline: WORKS

The sequence is the one resource that cannot be bundled, and that is igv.js's rule rather
than a choice: omit it and 3.1.1 dies in ``Ec.loadAll`` on ``undefined.startsWith`` while
3.8.5 dies on "url must be either a 'File', 'string', 'function', or 'Promise'". A ``data:``
URI does not substitute either — igv decodes data URIs inline and treats them as a
*non-indexed* FASTA, taking chromosome lengths from the body, so a stub that declares real
lengths in its index renders a perfect ideogram and ruler while every feature track silently
draws nothing (3 of 5 canvases painted, against 5 of 5 with a real reference). hg38 is 3 GB.

What is bundled: the ideogram and the chromosome lengths, both from one vendored 6.1 kB
table, and the gene track, from chorus's own GENCODE annotation scoped to the drawn window.
Cost: +46.7 kB per report, 2.8% of a 1.65 MiB one.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "chorus" / "analysis"
LOCUS = "chr1:108750680-109799256"


# ──────────────────────────────────────────────────────────────────────
# The config
# ──────────────────────────────────────────────────────────────────────

def _config(**kw):
    from chorus.analysis._igv_report import igv_browser_config

    return igv_browser_config(LOCUS, [{"name": "signal", "features": []}], [], **kw)


def test_the_genome_is_described_not_looked_up():
    cfg = _config()
    assert "genome" not in cfg, (
        "the report still asks igv.js to resolve a genome name against its hosted "
        "catalogue; that fetch is fatal with no network"
    )
    assert cfg["loadDefaultGenomes"] is False, (
        "an explicit reference is not enough on its own -- igv.js loads its catalogue "
        "first regardless unless loadDefaultGenomes is false, and that is the fetch "
        "that makes an offline load fail"
    )
    ref = cfg["reference"]
    assert ref["id"] == "hg38"
    for key in ("cytobandURL", "chromSizesURL"):
        assert ref[key].startswith("data:"), f"{key} is still a URL: {ref[key][:80]}"


def test_only_the_sequence_is_remote():
    """One host, one resource, and it is the one igv.js will not let us bundle."""
    cfg = _config()
    remote = [v for v in cfg["reference"].values()
              if isinstance(v, str) and v.startswith(("http://", "https://"))]
    assert len(remote) == 1, f"expected exactly the sequence to be remote, got {remote}"
    assert remote[0].endswith(".2bit")


def test_the_chromosome_lengths_come_from_the_ideogram_table():
    """One asset for two jobs, which is only sound if the lengths are exactly right."""
    from chorus.analysis._igv_report import _cytoband_table

    cyto, sizes = _cytoband_table()
    lengths = dict(
        (ln.split("\t")[0], int(ln.split("\t")[1])) for ln in sizes.splitlines() if ln
    )
    # Chromosome 1's length in GRCh38, independent of any file in this repo.
    assert lengths["chr1"] == 248_956_422
    assert len(lengths) == 25, f"expected the 25 primary chromosomes, got {sorted(lengths)}"

    fai = Path("/data/chorus_data/genomes/hg38.fa.fai")
    if not fai.exists():
        fai = REPO / "genomes" / "hg38.fa.fai"
    if fai.exists():
        from_fai = {ln.split("\t")[0]: int(ln.split("\t")[1])
                    for ln in fai.read_text().splitlines()}
        mismatched = {c: (n, from_fai[c]) for c, n in lengths.items()
                      if c in from_fai and from_fai[c] != n}
        assert not mismatched, (
            f"cytoband-derived lengths disagree with the FASTA index: {mismatched}. "
            f"igv would place features against one and read sequence against the other."
        )


def test_the_gene_track_is_inline_and_scoped_to_the_window():
    """A whole-genome annotation would be enormous; only the drawn interval is visible."""
    cfg = _config()
    genes = [t for t in cfg["tracks"] if t.get("type") == "annotation"]
    if not genes:
        pytest.skip("GENCODE annotation not present on this host")
    track = genes[0]
    assert "url" not in track and "features" in track, "the gene track must be inline"
    chrom, span = LOCUS.split(":")
    start, end = (int(x) for x in span.split("-"))
    outside = [f for f in track["features"]
               if f["chr"] != chrom or f["end"] < start or f["start"] > end]
    assert not outside, f"{len(outside)} gene features fall outside the drawn window"
    assert len(track["features"]) > 5, "suspiciously few genes for a 1 Mb window at SORT1"


def test_a_self_hosted_sequence_is_honoured_and_a_fasta_gets_its_index():
    """The air-gap hook. A FASTA is accepted because every install already has hg38.fa."""
    from chorus.analysis._igv_report import igv_reference_config

    two_bit = igv_reference_config("http://localhost:8000/hg38.2bit")
    assert two_bit["twoBitURL"] == "http://localhost:8000/hg38.2bit"
    assert "fastaURL" not in two_bit

    fasta = igv_reference_config("http://localhost:8000/hg38.fa")
    assert fasta["fastaURL"] == "http://localhost:8000/hg38.fa"
    assert fasta["indexURL"] == "http://localhost:8000/hg38.fa.fai", (
        "igv reads chromosome lengths and byte offsets from the .fai, so naming the "
        "FASTA without its index gives the non-indexed reader and 60 bp chromosomes"
    )

    os.environ["CHORUS_IGV_SEQUENCE_URL"] = "http://example.invalid/hg38.2bit"
    try:
        assert igv_reference_config()["twoBitURL"] == "http://example.invalid/hg38.2bit"
    finally:
        del os.environ["CHORUS_IGV_SEQUENCE_URL"]


def test_all_three_render_paths_share_one_config_builder():
    """The enumeration guard, for the trap this codebase has already fallen into.

    ``_igv_report``, ``multi_oracle_report`` and ``causal`` each built the browser config
    themselves. That is exactly how the display-pooling defect shipped: three copies of the
    same code, one of them fixed. Anything genome-related has to live in one place.
    """
    offenders = []
    for name in ("_igv_report.py", "multi_oracle_report.py", "causal.py"):
        src = (ANALYSIS / name).read_text()
        if re.search(r'"genome":\s*"hg38"', src):
            offenders.append(f"{name} still hardcodes a genome name")
        if "igv.createBrowser" in src and "igv_browser_config" not in src:
            offenders.append(f"{name} builds its own igv config")
    assert not offenders, offenders


# ──────────────────────────────────────────────────────────────────────
# The committed artefacts
# ──────────────────────────────────────────────────────────────────────

def _committed_reports() -> list:
    out = subprocess.check_output(["git", "ls-files", "examples/"], cwd=REPO, text=True)
    return [REPO / p for p in out.split() if p.endswith(".html")]


def _browser_config(text: str) -> dict:
    """The object handed to ``igv.createBrowser``, parsed out of a rendered report.

    Scoped to the config on purpose. Searching the whole file for a host name finds the
    inlined igv.min.js — which mentions ``igv.org`` for its blat service and
    ``hgdownload`` for UCSC hubs, neither of which a report ever calls. A grep over the
    HTML therefore fails on a correct artefact, which is how a test ends up being
    "fixed" by loosening the thing it was meant to check.
    """
    m = re.search(
        r'igv\.createBrowser\(\s*\n\s*document\.getElementById\("[^"]+"\),\s*\n\s*(\{.*\})',
        text)
    assert m, "could not find the igv.createBrowser config in this report"
    return json.loads(m.group(1))


@pytest.mark.parametrize("report", _committed_reports(), ids=lambda p: p.name)
def test_no_committed_report_names_a_hosted_genome_registry(report: Path):
    """Reads the artefact, not the generator: regeneration is what makes the fix real."""
    text = report.read_text()
    if "igv.createBrowser" not in text:
        pytest.skip("no IGV panel (the batch-scoring table)")
    cfg = _browser_config(text)

    assert "genome" not in cfg, (
        f"{report.name} still resolves its genome through igv.org's catalogue -- "
        f"regenerate it (see CLAUDE.md's regeneration matrix)"
    )
    assert cfg.get("loadDefaultGenomes") is False, (
        f"{report.name} does not disable the catalogue fetch"
    )
    remote = [v for v in cfg["reference"].values()
              if isinstance(v, str) and v.startswith(("http://", "https://"))]
    assert len(remote) == 1 and ".2bit" in remote[0], (
        f"{report.name} reference points at {remote}; only the sequence may be remote"
    )
    genes = [t for t in cfg["tracks"] if t.get("type") == "annotation"]
    assert genes and "features" in genes[0], (
        f"{report.name} has no inline gene track; dropping the registry drops UCSC's "
        f"ncbiRefSeq with it, and a locus panel without genes is a real loss"
    )


# ──────────────────────────────────────────────────────────────────────
# In a browser: what a report actually fetches
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def browser():
    import browser_harness as bh

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


@pytest.mark.integration
def test_a_report_fetches_nothing_but_the_sequence(browser):
    """Counted in a browser, because the point is what a *reader* triggers.

    The source check above can only see what the config says. This sees what Chromium
    does with it — which is how the six-resource fan-out was found in the first place,
    and the only way to notice if igv.js starts fetching something new.
    """
    import browser_harness as bh

    report = max((p for p in _committed_reports()
                  if "igv.createBrowser" in p.read_text()),
                 key=lambda p: p.stat().st_size)
    r = bh.render(browser, report)
    hosts = {h: len(u) for h, u in r.external_hosts.items()}

    assert "igv.org" not in hosts, (
        f"{report.name} still calls igv.org's genome registry: {hosts}. That fetch is "
        f"fatal offline and is a dependency on a third party's web service staying up."
    )
    non_sequence = [u for u in r.external_urls if not u.endswith(".2bit")
                    and ".2bit" not in u]
    assert not non_sequence, (
        f"{report.name} fetches more than the reference sequence: {non_sequence[:5]}"
    )
    assert r.converged and not r.blank, f"the panel stopped painting: {r.summary()}"


@pytest.mark.integration
def test_a_self_hosted_sequence_makes_a_report_fully_offline(browser, tmp_path):
    """The headline claim, measured: report + genome on one origin, nothing else reachable.

    Same-origin rather than a separate port on purpose. A page opened from ``file://`` has
    origin ``null``, so it cannot read a served FASTA without CORS headers, and a second
    port needs them too — measured, both fail on
    ``Access to XMLHttpRequest ... has been blocked``. Serving the report beside the genome
    is the recipe that works, so it is the recipe that gets tested.
    """
    import http.server
    import socket
    import threading

    import browser_harness as bh

    genomes = Path("/data/chorus_data/genomes")
    if not (genomes / "hg38.fa").exists():
        genomes = REPO / "genomes"
    if not (genomes / "hg38.fa").exists() or not (genomes / "hg38.fa.fai").exists():
        pytest.skip("no local hg38.fa + .fai to self-host")

    source = next(p for p in _committed_reports() if "igv.createBrowser" in p.read_text())
    (tmp_path / "hg38.fa").symlink_to(genomes / "hg38.fa")
    (tmp_path / "hg38.fa.fai").symlink_to(genomes / "hg38.fa.fai")

    # Rebuild this report's config with a same-origin relative sequence URL.
    from chorus.analysis._igv_report import igv_browser_config

    html = source.read_text()
    m = re.search(r'(\{"(?:genome|locus|reference)":.*?\})\s*\n\s*\);', html, re.S)
    assert m, "could not find the igv config in the report"
    old = json.loads(m.group(1))
    os.environ["CHORUS_IGV_SEQUENCE_URL"] = "hg38.fa"
    try:
        cfg = igv_browser_config(old["locus"], old["tracks"], old.get("roi") or [])
    finally:
        del os.environ["CHORUS_IGV_SEQUENCE_URL"]
    (tmp_path / "report.html").write_text(
        html.replace(m.group(1), json.dumps(cfg, separators=(",", ":"))))

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(tmp_path), **kw)

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        page = browser.new_page(viewport={"width": 1600, "height": 1200})
        blocked, errors = [], []
        host = f"127.0.0.1:{port}"
        page.on("console",
                lambda msg: errors.append(msg.text[:200]) if msg.type == "error" else None)

        def route(r):
            if f"//{host}/" in r.request.url:
                r.continue_()
            else:
                blocked.append(r.request.url)
                r.abort()

        page.route("**/*", route)
        page.goto(f"http://{host}/report.html", wait_until="load", timeout=120_000)
        measured = {"measured": 0, "painted": 0}
        for _ in range(60):
            page.wait_for_timeout(500)
            measured = page.evaluate(bh._MEASURE_JS)
            if measured["measured"] and measured["painted"] == measured["measured"]:
                break
        page.close()
    finally:
        httpd.shutdown()

    assert not blocked, f"something outside the local origin was requested: {blocked[:5]}"
    assert not errors, f"console errors with no internet: {errors[:3]}"
    assert measured["measured"] >= 6 and measured["painted"] == measured["measured"], (
        f"the panel did not fully paint offline: {measured['painted']}/"
        f"{measured['measured']}. This is the claim the whole change rests on."
    )


def test_the_docstring_no_longer_claims_offline_capability():
    """It claimed air-gapped viewability while the genome was a network dependency."""
    src = (ANALYSIS / "_igv_report.py").read_text()
    head = src[:src.index("def _ensure_igv_local")]
    assert "air-gapped" not in head or "does NOT make the report self-contained" in head, (
        "the module comment still promises air-gapped viewability from inlining the JS "
        "alone; the sequence is still fetched unless the reader self-hosts it"
    )
