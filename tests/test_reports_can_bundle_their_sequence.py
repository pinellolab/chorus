"""`CHORUS_IGV_BUNDLE_SEQUENCE=1` must make a report render with no network at all.

Every committed report fetches its reference sequence from `hgdownload.soe.ucsc.edu`, because igv.js
requires a sequence source and hg38 is ~3 GB. Opened offline the panel paints **nothing** — measured
`canvases 0/0 painted` with external requests blocked, meaning igv.js allocated no canvases at all
rather than initialising and drawing empty ones.

The cause is in the bundled igv.js: `fastaURL||t.twobitURL||(i=t.id)`. With no sequence it resolves
`id: "hg38"` against igv.org's hosted registry — the remote catalogue fetch #139 removed by inlining.
So simply omitting the sequence is *worse* than leaving it remote.

The way through is that igv.js takes chromosome lengths from `chromSizesURL`, which reports already
inline, and needs the FASTA only to *initialise*. A tiny placeholder is therefore enough: with one
patched in, the same report went from `0/0 painted` in 45.3 s (timeout) to **100/100 painted in
1.6 s** with zero external requests.

Opt-in, deliberately: bundling the placeholder means the sequence track shows `N` rather than real
bases if a reader zooms in far enough to want them, so the default still fetches the real sequence.
`CHORUS_IGV_SEQUENCE_URL` remains the way to have both.
"""
from __future__ import annotations

import os

import pytest

from chorus.analysis._igv_report import (
    _PLACEHOLDER_BASES,
    _PLACEHOLDER_CONTIG,
    igv_browser_config,
    igv_reference_config,
)


def test_the_default_is_unchanged():
    """Opt-in means opt-in: no bundling unless asked, so committed reports do not shift."""
    ref = igv_reference_config()
    if ref is None:
        pytest.skip("cytoband tables unavailable on this host")
    assert "twoBitURL" in ref and ref["twoBitURL"].startswith("http"), (
        f"the default reference no longer points at a remote sequence: {sorted(ref)}. Bundling is "
        f"opt-in; flipping the default would change every committed report's bytes."
    )
    assert "indexed" not in ref, "the default path should not be declaring an unindexed FASTA"


def test_bundling_replaces_the_remote_sequence_with_an_inline_one():
    ref = igv_reference_config(bundle_sequence=True)
    if ref is None:
        pytest.skip("cytoband tables unavailable on this host")

    assert "twoBitURL" not in ref, "bundling must remove the remote sequence, not add to it"
    assert ref["fastaURL"].startswith("data:"), ref["fastaURL"][:60]
    assert ref["indexed"] is False, (
        "an inlined FASTA must be declared unindexed; igv.js would otherwise look for a .fai it "
        "cannot range-request from a data: URL"
    )
    # lengths must still come from the inlined table, which is what makes a tiny placeholder work
    assert ref["chromSizesURL"].startswith("data:"), (
        "chromSizesURL must stay inlined: igv.js takes chromosome lengths from it rather than from "
        "the FASTA, which is the only reason a placeholder shorter than the locus can work"
    )


def test_the_placeholder_is_unknown_bases_not_fabricated_ones():
    """The one correctness property that matters here.

    An unindexed FASTA positions its bases from offset 0 of the contig, and these reports display loci
    tens of megabases in — so real bases would appear at the wrong coordinates. A reader who zooms in
    must see `N`, not plausible-looking sequence that is silently from the wrong place.
    """
    ref = igv_reference_config(bundle_sequence=True)
    if ref is None:
        pytest.skip("cytoband tables unavailable on this host")

    import base64

    payload = ref["fastaURL"].split(",", 1)[1]
    fasta = base64.b64decode(payload).decode()
    header, *lines = fasta.strip().splitlines()
    assert header == f">{_PLACEHOLDER_CONTIG}", header
    bases = "".join(lines)
    assert set(bases) == {"N"}, (
        f"the bundled placeholder contains bases other than N: {sorted(set(bases))[:8]}. Any real "
        f"base here is displayed at the wrong genomic coordinate, which is worse than showing none."
    )
    assert len(bases) == _PLACEHOLDER_BASES


def test_the_environment_variable_is_honoured():
    saved = os.environ.get("CHORUS_IGV_BUNDLE_SEQUENCE")
    try:
        os.environ["CHORUS_IGV_BUNDLE_SEQUENCE"] = "1"
        ref = igv_reference_config()
        if ref is None:
            pytest.skip("cytoband tables unavailable on this host")
        assert "fastaURL" in ref and ref["fastaURL"].startswith("data:"), (
            "CHORUS_IGV_BUNDLE_SEQUENCE had no effect; the documented offline switch does not work"
        )
    finally:
        if saved is None:
            os.environ.pop("CHORUS_IGV_BUNDLE_SEQUENCE", None)
        else:
            os.environ["CHORUS_IGV_BUNDLE_SEQUENCE"] = saved


def test_an_explicit_sequence_url_still_wins():
    """Someone serving hg38 locally wants the real bases, not the placeholder."""
    ref = igv_reference_config("http://localhost:8000/hg38.fa", bundle_sequence=True)
    if ref is None:
        pytest.skip("cytoband tables unavailable on this host")
    assert ref["fastaURL"] == "http://localhost:8000/hg38.fa", (
        f"an explicit sequence URL was overridden by the placeholder: {ref.get('fastaURL', '')[:60]}. "
        f"Bundling is the fallback for having no sequence, not a preference over having one."
    )
    assert ref.get("indexURL", "").endswith(".fai")


def test_the_browser_config_threads_the_flag_through():
    cfg = igv_browser_config("chr1:109274000-109276000", [], bundle_sequence=True)
    ref = cfg.get("reference")
    if ref is None:
        pytest.skip("cytoband tables unavailable on this host")
    assert ref["fastaURL"].startswith("data:"), sorted(ref)
    assert cfg["loadDefaultGenomes"] is False, (
        "loadDefaultGenomes must stay False: the catalogue fetch is what makes an offline load fatal"
    )
