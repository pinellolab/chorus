"""The consensus badge is a direction verdict, and must not be readable as more.

``_consensus_rows`` computes agreement from ``1 if raw_score > 0 else -1`` and
nothing else. That is a defensible thing to measure -- but rendered as a bare
"✅ all ↑" it reads as "the oracles agree", full stop, and a reader has no way to
tell a 1.3x magnitude spread from a 100x one.

It became load-bearing when a third accessibility oracle joined the SORT1
multi-oracle matrix. At rs12740374 that row carries AlphaGenome +1.334,
ChromBPNet +1.376 and Cherimoya +1.793 -- unanimous in direction, while the
extremes differ by 1.37x in linear fold change. Concordant on the finding, not on
the size, and only the first half was being shown.

So every consensus row now also carries a ``spread``, and these tests pin the
three things that matter about it:

  * it is present exactly when there are >= 2 voting oracles,
  * it is in the column's own units (no fold-change conversion, because this
    report deliberately mixes log2FC, lnFC and Δ across layers),
  * it is rendered next to the badge in both markdown and HTML.
"""
from __future__ import annotations

import pytest

from chorus.analysis.multi_oracle_report import MultiOracleReport
from chorus.analysis.variant_report import TrackScore, VariantReport


def _report(oracle: str, layer: str, raw: float, desc: str) -> VariantReport:
    return VariantReport(
        chrom="chr1",
        position=109_274_968,
        ref_allele="G",
        alt_alleles=["T"],
        gene_name="SORT1",
        oracle_name=oracle,
        allele_scores={
            "T": [
                TrackScore(
                    assay_id=desc,
                    assay_type="DNASE",
                    cell_type="HepG2",
                    layer=layer,
                    ref_value=100.0,
                    alt_value=100.0 * (2 ** raw),
                    raw_score=raw,
                    description=desc,
                    quantile_score=0.99,
                )
            ]
        },
    )


def _matrix(*reports: VariantReport):
    return MultiOracleReport.from_reports(reports)._consensus_rows()


def test_spread_is_recorded_when_two_or_more_oracles_vote():
    rows = _matrix(
        _report("chrombpnet", "chromatin_accessibility", 1.3756, "DNASE:HepG2"),
        _report("cherimoya", "chromatin_accessibility", 1.7930, "DNASE:HepG2"),
        _report("alphagenome", "chromatin_accessibility", 1.3345, "DNASE:HepG2"),
    )
    row = next(r for r in rows if r["layer"] == "chromatin_accessibility")
    assert row["agreement"] == "consensus_gain"
    sp = row["spread"]
    assert sp is not None
    assert sp["n_oracles"] == 3
    assert sp["min"] == pytest.approx(1.3345)
    assert sp["max"] == pytest.approx(1.7930)
    assert sp["range"] == pytest.approx(1.7930 - 1.3345)


def test_a_single_voter_has_no_spread():
    """One oracle cannot disagree with itself; a range there would be noise."""
    rows = _matrix(_report("legnet", "promoter_activity", 0.347, "LentiMPRA:HepG2"))
    row = next(r for r in rows if r["layer"] == "promoter_activity")
    assert row["agreement"] == "single_gain"
    assert row["spread"] is None


def test_the_spread_stays_in_the_columns_own_units():
    """No fold-change conversion.

    The report mixes log2FC, lnFC and Δ across layers -- it ships a units glossary
    for exactly that reason -- so converting a spread to a fold ratio would be
    silently wrong for any layer that is not log2FC. The recorded numbers must be
    the raw scores themselves.
    """
    a, b = 0.10, 0.40
    rows = _matrix(
        _report("x", "promoter_activity", a, "LentiMPRA:HepG2"),
        _report("y", "promoter_activity", b, "LentiMPRA:HepG2"),
    )
    sp = next(r for r in rows if r["layer"] == "promoter_activity")["spread"]
    assert sp["min"] == pytest.approx(a)
    assert sp["max"] == pytest.approx(b)
    # NOT 2**(b-a) or exp(b-a) -- the range is a difference of raw scores.
    assert sp["range"] == pytest.approx(b - a)


def test_a_unanimous_but_widely_spread_row_shows_the_spread_in_markdown():
    """The regression this exists to prevent: "all ↑" with the size hidden."""
    m = MultiOracleReport.from_reports([
        _report("chrombpnet", "chromatin_accessibility", 1.3756, "DNASE:HepG2"),
        _report("cherimoya", "chromatin_accessibility", 1.7930, "DNASE:HepG2"),
        _report("alphagenome", "chromatin_accessibility", 1.3345, "DNASE:HepG2"),
    ])
    md = m.to_markdown()
    assert "all ↑" in md
    assert "+1.33" in md and "+1.79" in md, (
        "the consensus cell shows a direction verdict with no magnitude spread, so "
        "a 1.37x disagreement between oracles renders identically to exact agreement"
    )
    assert "Agreement (direction)" in md, "the column header should say what it compares"


def test_the_html_badge_carries_the_spread_and_the_header_says_direction():
    m = MultiOracleReport.from_reports([
        _report("chrombpnet", "chromatin_accessibility", 1.3756, "DNASE:HepG2"),
        _report("cherimoya", "chromatin_accessibility", 1.7930, "DNASE:HepG2"),
    ])
    html = m.to_html()
    assert "✅ all ↑" in html
    assert "2 oracles" in html
    assert "+1.38" in html and "+1.79" in html
    assert "on direction" in html


def test_disagreeing_rows_are_not_given_a_spread_badge():
    """A ⚠ disagree row already tells the reader the important thing.

    Appending a range there would suggest the spread is the story when the sign
    flip is; the badge stays as-is.
    """
    m = MultiOracleReport.from_reports([
        _report("a", "chromatin_accessibility", +1.5, "DNASE:HepG2"),
        _report("b", "chromatin_accessibility", -1.5, "DNASE:HepG2"),
    ])
    rows = m._consensus_rows()
    row = next(r for r in rows if r["layer"] == "chromatin_accessibility")
    assert row["agreement"] == "disagree"
    # spread is still recorded in the machine-readable matrix ...
    assert row["spread"] is not None
    # ... but not rendered onto the badge.
    md = m.to_markdown()
    assert "disagree" in md
    assert "…" not in md.split("disagree")[1].split("|")[0]
