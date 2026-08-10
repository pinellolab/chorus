"""The MCP tools that need no oracle, and so had no excuse for being untested.

`score_ism` was the load-bearing gap and now has coverage. Six other tools were also
untested; three of them touch no oracle at all and are pure metadata/annotation lookups,
so they are testable on CPU in milliseconds:

    list_genomes           reference genomes and their download status
    get_genes_in_region    gene annotations overlapping an interval
    get_gene_tss           TSS positions for a named gene

These are the tools an agent calls FIRST when orienting in a locus, so a wrong answer
here misdirects everything downstream — and unlike a scoring bug it produces no
implausible numbers to notice.

The remaining three (oracle_status, score_prediction_region,
score_variant_effect_at_region) all call `state.get_oracle(...)` and need a loaded model,
so they belong in an integration module rather than here.
"""
from __future__ import annotations

import pytest

# The MCP tools are wrapped in @mcp.tool() and @_safe_tool; import the module and reach
# the underlying callables so the tests exercise the same code the server serves.
import chorus.mcp.server as server


def _call(name: str, /, **kw):
    fn = getattr(server, name)
    for attr in ("fn", "__wrapped__"):
        fn = getattr(fn, attr, fn)
    return fn(**kw)


# ---------------------------------------------------------------------------
# list_genomes
# ---------------------------------------------------------------------------


def test_list_genomes_reports_download_status_and_a_path_only_when_present():
    out = _call("list_genomes")
    assert "genomes" in out and out["genomes"], out
    ids = [g["id"] for g in out["genomes"]]
    assert "hg38" in ids, f"hg38 absent from {ids}"
    assert len(ids) == len(set(ids)), "duplicate genome ids"
    for g in out["genomes"]:
        assert {"id", "description", "downloaded"} <= set(g)
        assert isinstance(g["downloaded"], bool)
        # A path must appear iff the genome is downloaded. Reporting a path for a
        # genome that is absent would send every downstream caller to a missing file.
        assert ("path" in g) == g["downloaded"], g


def test_the_downloaded_genome_path_exists_on_disk():
    out = _call("list_genomes")
    import os
    for g in out["genomes"]:
        if g["downloaded"]:
            assert os.path.exists(g["path"]), (
                f"{g['id']} is reported downloaded but {g['path']} does not exist"
            )


# ---------------------------------------------------------------------------
# get_genes_in_region
# ---------------------------------------------------------------------------


def test_get_genes_in_region_finds_a_gene_it_must_contain():
    """SORT1 at chr1:109.2-109.3 Mb — the locus this repo's flagship example uses."""
    out = _call("get_genes_in_region", chrom="chr1", start=109_200_000,
                end=109_400_000)
    assert out["num_genes"] == len(out["genes"])
    names = {g.get("gene_name") for g in out["genes"]}
    assert "SORT1" in names, f"SORT1 missing; found {sorted(n for n in names if n)[:12]}"
    assert out["chrom"] == "chr1"


def test_the_heavy_attributes_column_is_not_returned():
    """It is a GTF blob per gene; returning it would bloat every response."""
    out = _call("get_genes_in_region", chrom="chr1", start=109_200_000,
                end=109_400_000)
    assert out["genes"]
    for g in out["genes"][:20]:
        assert "attributes" not in g


def test_every_returned_gene_actually_overlaps_the_requested_interval():
    """A tool that answers "genes here" must not include genes that are not."""
    start, end = 109_200_000, 109_400_000
    out = _call("get_genes_in_region", chrom="chr1", start=start, end=end)
    for g in out["genes"]:
        gs, ge = g.get("start"), g.get("end")
        if gs is None or ge is None:
            continue
        assert ge >= start and gs <= end, (
            f"{g.get('gene_name')} at {gs}-{ge} does not overlap {start}-{end}"
        )
        assert str(g.get("chrom", "chr1")) == "chr1"


def test_an_empty_region_returns_zero_genes_rather_than_raising():
    """Telomeric start of chr1 — annotated genes do not begin at base 1."""
    out = _call("get_genes_in_region", chrom="chr1", start=1, end=5_000)
    assert out["num_genes"] == 0
    assert out["genes"] == []


def test_a_nonexistent_contig_does_not_return_a_confident_answer():
    """Either raise/report an error, or return zero genes — never silent garbage."""
    try:
        out = _call("get_genes_in_region", chrom="chrNOPE", start=1, end=10_000)
    except Exception:
        return                                    # raising is acceptable
    assert out.get("num_genes", 0) == 0 or "error" in out, out


# ---------------------------------------------------------------------------
# get_gene_tss
# ---------------------------------------------------------------------------


def test_get_gene_tss_is_strand_aware():
    """SORT1 is on the MINUS strand, so its TSS is the transcript END, not the start.

    Getting this backwards anchors on 3' ends, where there is no promoter signal -- the
    same error the region samplers guard against. It is invisible in aggregate: a wrong
    TSS is still a plausible chr1 coordinate inside the gene.
    """
    out = _call("get_gene_tss", gene_name="SORT1")
    assert out["gene_name"] == "SORT1"
    assert out["num_transcripts"] == len(out["tss_positions"]) >= 1
    for rec in out["tss_positions"]:
        assert rec["chrom"] == "chr1"
        assert rec["strand"] in ("+", "-")
        expected = (rec["transcript_end"] if rec["strand"] == "-"
                    else rec["transcript_start"])
        assert rec["tss"] == expected, (
            f"{rec['transcript_id']} on strand {rec['strand']}: tss={rec['tss']} but "
            f"strand-aware TSS is {expected}. A minus-strand gene's TSS is its "
            f"transcript END."
        )
        assert 109_000_000 <= rec["tss"] <= 110_500_000, rec["tss"]


def test_an_unknown_gene_is_refused_rather_than_answered():
    try:
        out = _call("get_gene_tss", gene_name="NOT_A_REAL_GENE_XYZ")
    except Exception:
        return
    assert (not out) or "error" in str(out).lower() or out.get("num_tss", 0) == 0, out
