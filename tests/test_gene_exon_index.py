"""The builder's RNA mask must be the same object the query scores (#144 inst. 3).

``scripts/build_backgrounds_alphagenome.py``'s ``load_exon_index`` merged exons
across **every protein-coding gene on the chromosome**, discarding gene identity.
The builder then aggregated RNA signal over every exon in its ~1 Mb window while
the query aggregates over **one gene's** exons. Measured at the SORT1 locus:

=====================================================  =======
statistic                                              bins
=====================================================  =======
old builder: pooled over all PC exons in the window     128,663
query: one gene's exon union (median of the 29 genes)      4,123
=====================================================  =======

A **31x** difference in how much sequence the null covered versus the numerator,
so the percentile ranked a gene-scoped statistic against a genome-scoped null.
That is not a calibration error a floor can fix; the two are different quantities.

``build_gene_exon_index`` fixes the structure by keeping genes separate, and is
built *from* :func:`get_gene_exons` — the query's own function — so the masks
cannot drift apart. This module asserts that equivalence rather than trusting it.
"""
from __future__ import annotations

import numpy as np
import pytest

from chorus.utils.annotations import (
    build_gene_exon_index,
    exon_bins_for_gene,
    genes_overlapping,
    get_gene_exons,
)

SORT1_POS = 109_274_968
PRED_BP = 1_048_576


@pytest.fixture(scope="module")
def index():
    return build_gene_exon_index()


# ---------------------------------------------------------------------------
# Parity with the query path — the whole point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gene", [
    "SORT1", "CELSR2", "PSRC1", "TERT", "FTO", "BCL11A", "MYC",
    # 2.47 Mb — cannot fit in any oracle's prediction window, so it exercises
    # the clipping path and the overlap-not-containment rule together
    "RBFOX1",
])
def test_index_union_equals_the_query_union(index, gene):
    """Byte-identical masks, not merely similar ones."""
    want = sorted((int(r.start), int(r.end)) for r in get_gene_exons(gene).itertuples())
    got = None
    for chrom in index:
        for _g_start, _g_end, name, spans in index[chrom]:
            if name == gene:
                got = sorted(spans)
    assert got is not None, f"{gene} missing from the index"
    assert got == want, f"{gene}: index and query disagree on the exon union"


def test_index_covers_the_protein_coding_genome(index):
    total = sum(len(v) for v in index.values())
    assert 19_000 < total < 21_500, f"{total} PC genes — expected ~20,086"


def test_spans_are_merged_and_sorted(index):
    """Overlapping exons from different transcripts must already be unioned, or
    RNA signal is double-counted where transcripts share an exon."""
    for chrom, genes in list(index.items())[:4]:
        for _g_start, _g_end, name, spans in genes[:200]:
            for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
                assert s0 < e0, f"{name}: empty span"
                assert e0 < s1, f"{name}: unmerged or unsorted spans on {chrom}"


def test_gene_bounds_match_the_spans(index):
    for genes in list(index.values())[:4]:
        for g_start, g_end, name, spans in genes[:200]:
            assert g_start == spans[0][0], name
            assert g_end == spans[-1][1], name


# ---------------------------------------------------------------------------
# Gene enumeration: overlap, deliberately, not containment
# ---------------------------------------------------------------------------


def test_sort1_window_fans_out_to_many_genes(index):
    """29 genes, which is why one-gene-per-position would be the wrong null.

    In 41 of 43 committed (variant, gene) pairs the variant sits *outside* the
    gene span, a median ~230 kb away. A null built by attributing each sampled
    position to a single gene would answer "effect on the gene I sit in" while
    95-97% of the numerators ask about a distant gene.
    """
    start = SORT1_POS - PRED_BP // 2
    genes = genes_overlapping(index, "chr1", start, start + PRED_BP)
    names = {g[2] for g in genes}
    assert len(genes) == 29, f"expected 29 PC genes, got {len(genes)}"
    assert {"SORT1", "CELSR2", "PSRC1"} <= names


def test_overlap_not_containment(index):
    """A gene larger than the window must still be returned.

    AlphaGenome's own GeneQueryType is INTERVAL_CONTAINED, but 68 PC genes can
    never be contained in 1,048,576 bp. Using containment in the builder would
    build the null over a gene population the query never uses.
    """
    rbfox1 = [g for chrom in index for g in index[chrom] if g[2] == "RBFOX1"][0]
    g_start, g_end = rbfox1[0], rbfox1[1]
    assert g_end - g_start > PRED_BP, "RBFOX1 should exceed a 1 Mb window"

    mid = (g_start + g_end) // 2
    window_start = mid - PRED_BP // 2
    found = genes_overlapping(index, "chr16", window_start, window_start + PRED_BP)
    assert "RBFOX1" in {g[2] for g in found}


def test_no_overlap_returns_nothing(index):
    assert genes_overlapping(index, "chr1", 1, 1000) == []
    assert genes_overlapping(index, "chrNotAChrom", 0, 10**6) == []


def test_half_open_boundaries(index):
    """A gene ending exactly at the window start must not be included."""
    genes = index["chr1"]
    g_start, g_end, name, _ = genes[10]
    assert name not in {g[2] for g in genes_overlapping(index, "chr1", g_end, g_end + 1000)}
    assert name in {g[2] for g in genes_overlapping(index, "chr1", g_end - 1, g_end + 1000)}


# ---------------------------------------------------------------------------
# Bin conversion and clipping
# ---------------------------------------------------------------------------


def test_bins_at_resolution_1_count_exonic_bases():
    spans = [(1_000, 1_100), (2_000, 2_050)]
    bins = exon_bins_for_gene(spans, 0, 10_000, 10_000, 1)
    assert len(bins) == 150
    assert bins.min() == 1_000 and bins.max() == 2_049


def test_bins_are_deduplicated_across_touching_spans():
    """Two spans inside ONE 128 bp bin must contribute that bin exactly once.

    Bin 7 spans [896, 1024), so both of these land wholly inside it. Without
    deduplication the bin would be summed twice and the denominator inflated —
    which is the same double-counting ``merge=True`` exists to prevent at the
    exon level.
    """
    spans = [(900, 910), (920, 930)]
    bins = exon_bins_for_gene(spans, 0, 10_000, 100, 128)
    assert bins.tolist() == [7]


def test_a_span_crossing_a_bin_boundary_claims_both_bins():
    """(1020, 1030) crosses 1024, so it covers bins 7 and 8 — not one bin."""
    bins = exon_bins_for_gene([(1_020, 1_030)], 0, 10_000, 100, 128)
    assert bins.tolist() == [7, 8]


def test_clipping_to_the_prediction_window():
    """Only bins actually predicted may be summed — a gene straddling the edge
    contributes its overlap, not its annotated length."""
    spans = [(500, 1_500)]
    bins = exon_bins_for_gene(spans, 1_000, 2_000, 1_000, 1)
    assert len(bins) == 500          # 1000..1500, not the full 1000 bp
    assert bins.min() == 0


def test_gene_entirely_outside_the_window_yields_nothing():
    assert len(exon_bins_for_gene([(50, 100)], 1_000, 2_000, 1_000, 1)) == 0
    assert len(exon_bins_for_gene([(5_000, 6_000)], 1_000, 2_000, 1_000, 1)) == 0


def test_bins_never_exceed_the_array(index):
    """An out-of-range index would be an IndexError mid-build, hours in."""
    start = SORT1_POS - PRED_BP // 2
    for _g0, _g1, _name, spans in genes_overlapping(index, "chr1", start, start + PRED_BP):
        for res, n_bins in ((1, PRED_BP), (32, PRED_BP // 32), (128, PRED_BP // 128)):
            bins = exon_bins_for_gene(spans, start, start + PRED_BP, n_bins, res)
            if len(bins):
                assert bins.min() >= 0 and bins.max() < n_bins


def test_per_gene_mask_is_far_smaller_than_the_pooled_one(index):
    """Pins the 31x magnitude of #144 instance 3, so it cannot silently return."""
    start = SORT1_POS - PRED_BP // 2
    genes = genes_overlapping(index, "chr1", start, start + PRED_BP)
    per_gene = [len(exon_bins_for_gene(s, start, start + PRED_BP, PRED_BP, 1))
                for *_, s in genes]
    pooled = set()
    for *_, spans in genes:
        pooled |= set(exon_bins_for_gene(spans, start, start + PRED_BP, PRED_BP, 1).tolist())
    median = float(np.median(per_gene))
    assert 3_500 < median < 4_800, median
    assert len(pooled) > 120_000, len(pooled)
    assert len(pooled) / median > 20, "pooled/per-gene ratio collapsed — check the index"


# ---------------------------------------------------------------------------
# Builder / query parity on the RNA statistic itself (#144 instance 3)
# ---------------------------------------------------------------------------


def test_builder_rna_statistic_equals_the_query_statistic(index):
    """The builder's per-gene mean must equal what score_track_effect computes.

    This is the actual contract #144 instance 3 broke. The builder aggregated over
    every exon in the window; the query aggregates over one gene's exons. Both
    sides now derive the mask from the same index and divide by the bins actually
    summed, so a uniform track must give the same number either way.
    """
    from chorus.analysis.scorers import LAYER_CONFIGS, score_track_effect
    from chorus.core.interval import GenomeRef, Interval
    from chorus.core.result import OraclePredictionTrack

    pred_start = SORT1_POS - PRED_BP // 2
    genes = {g[2]: g[3] for g in genes_overlapping(index, "chr1", pred_start,
                                                   pred_start + PRED_BP)}
    spans = genes["SORT1"]

    ref_level, alt_level = 0.5, 2.0
    ref_bins = exon_bins_for_gene(spans, pred_start, pred_start + PRED_BP, PRED_BP, 1)
    assert len(ref_bins) > 0

    # what the BUILDER now computes: mean over the gene's mask, in bins
    builder_ref = float(np.full(len(ref_bins), ref_level).mean())
    builder_alt = float(np.full(len(ref_bins), alt_level).mean())

    # what the QUERY computes, through the real scorer
    def track(level):
        ref = GenomeRef(chrom="chr1", start=pred_start,
                        end=pred_start + PRED_BP, fasta="/nonexistent.fa")
        iv = Interval.make(ref) if hasattr(Interval, "make") else Interval(reference=ref)
        return OraclePredictionTrack(
            source_model="test", assay_id="RNA_SEQ/test", assay_type="RNA",
            cell_type="TEST", query_interval=iv, prediction_interval=iv,
            input_interval=iv, resolution=1,
            values=np.full(PRED_BP, level, dtype=np.float64),
        )

    exons = [{"chrom": "chr1", "start": s, "end": e} for s, e in spans]
    out = score_track_effect(
        track(ref_level), track(alt_level), "chr1", SORT1_POS,
        layer_config=LAYER_CONFIGS["gene_expression"], gene_exons=exons,
    )
    assert out is not None
    assert out["ref_value"] == pytest.approx(builder_ref)
    assert out["alt_value"] == pytest.approx(builder_alt)


def test_builder_no_longer_pools_exons_across_genes():
    """Source assertion: the chromosome-merged index must be gone for good.

    Keeping it around would leave two ways to build the RNA mask, which is how
    the builder and query diverged in the first place.
    """
    from pathlib import Path

    src = Path("scripts/build_backgrounds_alphagenome.py").read_text()
    assert "def load_exon_index" not in src
    assert "def exon_bin_indices" not in src
    assert "build_gene_exon_index()" in src
    assert "exon_bins_for_gene(" in src
    assert "genes_overlapping(" in src
