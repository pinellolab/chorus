"""``chorus.analysis.layer_ranking``: per-track percentiles -> a tissue / system rank table.

The numbers that matter are the ones a reader would compare against the published figure, so the
suite has three layers:

* the statistics, pinned to closed forms and to the worked example in
  ``analysis_output/.../AGGREGATION_METHODS.md`` (Fisher and ACAT disagree on purpose there);
* the ranking logic on tiny hand-built tables whose answer is known by construction;
* a golden comparison with the committed SORT1 outputs, skipped when ``analysis_output/`` is absent
  (it is not tracked).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chorus.analysis import layer_ranking as lr

REPO = Path(__file__).resolve().parent.parent
GOLD = REPO / "analysis_output" / "sort1_track_significance" / "layer_rank_table"


# ---------------------------------------------------------------- statistics

def test_p_is_floored_so_a_track_beyond_the_null_has_a_finite_p():
    assert lr.to_p(1.0) == pytest.approx(1.0 / (lr.N_NULL + 1))
    assert lr.to_p(0.0) == 1.0
    assert lr.to_p(0.97) == pytest.approx(0.03)


@pytest.mark.parametrize("pct", [0.0, 0.5, 0.97, 0.9999])
def test_one_track_combines_to_itself(pct):
    p = float(lr.to_p(pct))
    assert np.exp(lr.fisher_logp([pct])) == pytest.approx(p, rel=1e-9)
    assert lr.acat_p([pct]) == pytest.approx(p, rel=1e-6)


def test_fisher_and_acat_reproduce_the_documented_toy_example():
    """AGGREGATION_METHODS.md section 4: tissue A = 2 tracks at the floor, B = 40 tracks at 0.97."""
    a, b = [0.9999] * 2, [0.97] * 40
    assert np.exp(lr.fisher_logp(a)) == pytest.approx(1.9e-07, rel=0.05)
    assert np.exp(lr.fisher_logp(b)) == pytest.approx(4.4e-24, rel=0.05)
    assert lr.acat_p(a) == pytest.approx(1.0e-04, rel=0.05)
    assert lr.acat_p(b) == pytest.approx(3.0e-02, rel=0.05)


def test_acat_is_bonferroni_like_when_one_track_dominates():
    v = [0.9999] + [0.5] * 9
    assert lr.acat_p(v) == pytest.approx(10 * 1e-4, rel=0.05)


def test_fisher_stays_finite_where_scipy_underflows():
    """~120+ tracks at the floor: a plain p is 0 and ``chi2.logsf`` is -inf, so every such group would tie."""
    from scipy.stats import chi2
    v = [1.0] * 200
    x2 = -2.0 * np.log(lr.to_p(v)).sum()
    assert chi2.sf(x2, 400) == 0.0 and chi2.logsf(x2, 400) == -np.inf     # both routes lose it
    assert np.isfinite(lr.fisher_logp(v)) and lr.fisher_logp(v) == pytest.approx(-1203.695, abs=0.01)
    assert lr.fisher_logp([1.0] * 300) < lr.fisher_logp([1.0] * 200)       # still orders groups


def test_fisher_agrees_with_the_chi_square_survival_function_where_that_is_finite():
    from scipy.stats import chi2
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = rng.beta(5, 1, rng.integers(1, 100))
        assert lr.fisher_logp(v) == pytest.approx(chi2.logsf(-2 * np.log(lr.to_p(v)).sum(), 2 * len(v)),
                                                  rel=1e-9, abs=1e-12)


def test_fisher_of_untouched_tracks_is_certainty():
    assert lr.fisher_logp([0.0, 0.0, 0.0]) == 0.0


# ---------------------------------------------------------------- discovery -> table

def _effect(tid, layer, cell, pct, raw, assay="", **kw):
    from chorus.analysis.discovery import TrackEffect
    return TrackEffect(assay_id=tid, layer=layer, cell_type=cell, description=tid, raw_score=raw,
                       abs_score=abs(raw), ref_value=1.0, alt_value=1.0, effect_pctile=pct, assay=assay, **kw)


def test_table_uses_magnitudes_drops_unscored_tracks_and_names_layers():
    t = lr.track_table_from_effects([
        _effect("d1", "chromatin_accessibility", "HepG2", -0.99, -1.2, "DNASE"),
        _effect("c1", "tf_binding", "K562", 0.9, 0.3, "CHIP"),
        _effect("h1", "histone_marks", "K562", 0.8, 0.2, "CHIP"),
        _effect("n1", "chromatin_accessibility", "HepG2", None, 0.2, "DNASE"),
    ], "alphagenome")
    assert sorted(t.molecular_layer) == ["ChIP-TF", "ChIP-histone", "DNase"]
    assert (t.percentile >= 0).all() and "n1" not in set(t.track_id)
    assert t.set_index("track_id").loc["d1", ["model", "percentile", "raw_score"]].tolist() == ["AlphaGenome", 0.99, -1.2]


def test_strands_of_one_bpnet_track_count_once_and_the_stronger_wins():
    t = lr.track_table_from_effects([
        _effect("CHIP:K562:CTCF:+", "tf_binding", "K562", 0.60, 0.1, "CHIP"),
        _effect("CHIP:K562:CTCF:-", "tf_binding", "K562", 0.95, -0.4, "CHIP"),
    ], "chrombpnet")
    assert len(t) == 1
    row = t.iloc[0]
    assert (row.track_id, row.percentile, row.model) == ("CHIP:K562:CTCF", 0.95, "BPNet")


def test_the_oracle_catalogue_overrides_the_parsed_cell_type_and_assay():
    """Discovery parses BPNet cell types wrongly; the catalogue is authoritative."""
    t = lr.track_table_from_effects(
        [_effect("CHIP:HepG2:CEBPB:+", "tf_binding", "WRONG", 0.9, 0.2, assay=None)],
        "chrombpnet", {"CHIP:HepG2:CEBPB": {"cell_type": "HepG2", "assay": "CHIP"}})
    assert t.iloc[0].cell_type_key == "hepg2" and t.iloc[0].molecular_layer == "ChIP-TF"


def test_without_a_catalogue_a_tf_track_still_lands_in_the_chip_tf_column():
    assert lr.display_layer(None, "tf_binding") == "ChIP-TF"
    assert lr.display_layer("", "histone_marks") == "ChIP-histone"
    assert lr.display_layer("DNASE", "chromatin_accessibility") == "DNase"


def test_prediction_matrix_lists_every_track_and_the_fisher_p():
    t = pd.DataFrame({"model": "AlphaGenome", "molecular_layer": "ChIP-histone", "track_id": ["a", "b"],
                      "cell_type_key": "k562", "cell_type": "K562", "percentile": [0.9, 0.99],
                      "raw_score": [0.1, -0.2]})
    m = lr.prediction_matrix(t)
    row = m.iloc[0]
    assert row.alphagenome_chip_histone_all == "0.9900 (b), 0.9000 (a)"
    assert float(row.alphagenome_chip_histone_agg) == pytest.approx(np.exp(lr.fisher_logp([0.9, 0.99])), rel=1e-3)


def test_tables_round_trip_even_when_a_track_id_contains_a_hash(tmp_path):
    """Sei ids are ``TA#...``; ``read_csv(comment='#')`` would cut those rows."""
    t = pd.DataFrame({"model": "Sei", "molecular_layer": "DNase", "track_id": ["TA#1_x", "TA#2_y"],
                      "cell_type_key": ["a", "b"], "cell_type": ["A", "B"], "percentile": [0.9, 0.8],
                      "raw_score": [0.5, -0.5]})
    paths = lr.write_discovery_tables(t, tmp_path, "chr1_1_sei", "chr1:1 A>T")
    back, label = lr.load_track_tables(paths["track_scores_csv"])
    assert label == "chr1:1 A>T"
    assert sorted(back.track_id) == ["TA#1_x", "TA#2_y"]


def test_the_prediction_matrix_is_refused_as_ranking_input(tmp_path):
    t = pd.DataFrame({"model": "AlphaGenome", "molecular_layer": "DNase", "track_id": ["a"],
                      "cell_type_key": ["k"], "cell_type": ["K"], "percentile": [0.9], "raw_score": [0.1]})
    paths = lr.write_discovery_tables(t, tmp_path, "s", "v")
    with pytest.raises(ValueError, match="not a track table"):
        lr.load_track_tables(paths["prediction_matrix_csv"])


# ---------------------------------------------------------------- ranking

def _ann(mapping):
    return pd.DataFrame([{"cell_type_key": k, "tissue_organ": o, "tissue_system": s}
                         for k, (o, s) in mapping.items()])


def _tracks(spec, model="AlphaGenome", layer="DNase"):
    """spec: {cell_type_key: [(percentile, raw_score), ...]}"""
    rows = [{"model": model, "molecular_layer": layer, "track_id": f"{layer}:{c}:{i}", "cell_type_key": c,
             "cell_type": c.upper(), "percentile": p, "raw_score": r}
            for c, tr in spec.items() for i, (p, r) in enumerate(tr)]
    return pd.DataFrame(rows)


@pytest.fixture
def breadth_vs_peak(tmp_path):
    """Liver has ONE track beyond the null; Brain has ten at 0.97. Fisher rewards the latter, ACAT the former."""
    t = pd.concat([
        _tracks({"hepg2": [(1.0, 1.0)], "brain_a": [(0.97, 0.5)] * 10}, layer=layer)
        for layer in ("DNase", "ATAC")])
    ann = _ann({"hepg2": ("Liver", "Digestive"), "brain_a": ("Brain", "Nervous")})
    return t, ann, tmp_path


@pytest.mark.parametrize("method,first", [("fisher", "Brain"), ("acat", "Liver")])
def test_fisher_rewards_breadth_and_acat_rewards_a_single_strong_hit(breadth_vs_peak, method, first):
    t, ann, tmp = breadth_vs_peak
    r = lr.rank_variant_tissues(t, tmp, method=method, annotation=ann, formats=("png",))
    assert r["top"][0]["group"] == first and r["top"][0]["global_rank"] == 1


def test_the_system_level_groups_by_the_other_annotation_column(breadth_vs_peak):
    t, ann, tmp = breadth_vs_peak
    r = lr.rank_variant_tissues(t, tmp, group_level="system", annotation=ann, formats=("png",))
    assert {x["group"] for x in r["top"]} == {"Digestive", "Nervous"}


def test_ties_at_the_floor_are_broken_by_the_larger_raw_effect(tmp_path):
    t = _tracks({"a": [(1.0, 0.2)], "b": [(1.0, 0.9)], "c": [(1.0, 0.5)]})
    ann = _ann({k: (k.upper(), "S") for k in "abc"})
    r = lr.rank_variant_tissues(t, tmp_path, annotation=ann, layers=["DNase"], formats=("png",))
    assert [x["group"] for x in r["top"]] == ["B", "C", "A"]


def test_cell_types_missing_from_the_annotation_are_reported_not_silently_dropped(tmp_path):
    t = _tracks({"known": [(0.99, 1.0)], "mystery": [(0.99, 1.0)] * 3})
    r = lr.rank_variant_tissues(t, tmp_path, annotation=_ann({"known": ("K", "S")}), layers=["DNase"],
                                formats=("png",))
    assert r["unmapped_cell_types"] == ["MYSTERY"] and r["n_unmapped_cell_types"] == 1
    assert any("not in the annotation" in n for n in r["notes"])
    assert (tmp_path / "unmapped_cell_types.csv").exists()
    assert [x["group"] for x in r["top"]] == ["K"]


def test_min_tracks_removes_small_groups_from_a_layer(tmp_path):
    t = _tracks({"solo": [(1.0, 1.0)], "many": [(0.9, 0.5)] * 3})
    ann = _ann({"solo": ("Solo", "S"), "many": ("Many", "S")})
    r = lr.rank_variant_tissues(t, tmp_path, annotation=ann, layers=["DNase"], min_tracks=3, formats=("png",))
    assert [x["group"] for x in r["top"]] == ["Many"]


def test_a_layer_whose_designated_oracle_is_absent_falls_back_and_says_so(tmp_path):
    """The published figure takes DNase from Cherimoya; given only AlphaGenome tracks it must not vanish."""
    t = _tracks({"a": [(0.99, 1.0)]}, model="AlphaGenome", layer="DNase")
    r = lr.rank_variant_tissues(t, tmp_path, annotation=_ann({"a": ("A", "S")}), formats=("png",))
    assert r["layers"]["DNase"]["oracle"] == "AlphaGenome"
    assert any("DNase" in n and "Cherimoya" in n for n in r["notes"])
    assert any("ATAC" in n and "omitted" in n for n in r["notes"])


def test_an_explicit_oracle_the_input_lacks_is_an_error_not_a_fallback(tmp_path):
    t = _tracks({"a": [(0.99, 1.0)]})
    with pytest.raises(ValueError, match="Cherimoya"):
        lr.rank_variant_tissues(t, tmp_path, annotation=_ann({"a": ("A", "S")}),
                                layer_oracles={"DNase": "Cherimoya"})


def test_global_order_needs_fewer_layers_when_fewer_layers_exist(tmp_path):
    """The default needs 4 layers; two layers of data must still produce a Global column."""
    t = pd.concat([_tracks({"a": [(0.99, 1.0)], "b": [(0.9, 0.1)]}, layer=layer) for layer in ("DNase", "ATAC")])
    ann = _ann({"a": ("A", "S"), "b": ("B", "S")})
    r = lr.rank_variant_tissues(t, tmp_path, annotation=ann, formats=("png",))
    assert [x["group"] for x in r["top"]] == ["A", "B"]


@pytest.mark.parametrize("kw,msg", [({"method": "top3"}, "method"), ({"group_level": "cell"}, "group_level")])
def test_bad_arguments_fail_loudly(tmp_path, kw, msg):
    with pytest.raises(ValueError, match=msg):
        lr.rank_variant_tissues(_tracks({"a": [(0.9, 1.0)]}), tmp_path, annotation=_ann({"a": ("A", "S")}), **kw)


def test_outputs_are_written_and_the_figure_has_the_documented_size(tmp_path, breadth_vs_peak):
    from PIL import Image
    t, ann, _ = breadth_vs_peak
    r = lr.rank_variant_tissues(t, tmp_path, annotation=ann, variant_label="X rs1", top_n=15)
    for name in ("layer_rank_table", "layer_rank_over_N_table", "layer_score_table", "layer_score_neglog10p_table",
                 "layer_ntracks_table", "layer_nsig_table", "method_comparison", "run_summary"):
        assert Path(r["files"][name]).stat().st_size > 0, name
    png = next(p for p in r["files"]["figure"] if p.endswith(".png"))
    # 2 groups: width 10 in, height 0.40*2 + 1.95 + 0.95 in, at 300 dpi
    assert Image.open(png).size == (3000, round(300 * (0.40 * 2 + 1.95 + 0.95)))
    assert any(p.endswith(".pdf") for p in r["files"]["figure"])


def test_the_figure_does_not_leak_matplotlib_state(tmp_path, breadth_vs_peak):
    import matplotlib as mpl
    t, ann, _ = breadth_vs_peak
    before = dict(mpl.rcParams)
    lr.rank_variant_tissues(t, tmp_path, annotation=ann, formats=("png",))
    assert mpl.rcParams["font.size"] == before["font.size"] and mpl.rcParams["pdf.fonttype"] == before["pdf.fonttype"]


def test_long_row_labels_are_wrapped_inside_the_left_margin():
    assert lr._wrap2("Placenta / extra-embryonic") == "Placenta / extra-embryonic"
    assert lr._wrap2("Nervous system (other / neural progenitor)") == "Nervous system\n(other / neural progenitor)"


# ---------------------------------------------------------------- golden (committed SORT1 outputs)

needs_gold = pytest.mark.skipif(not (GOLD / "fisher" / "track_long.csv").exists(),
                                reason="analysis_output/ is not tracked; golden comparison runs where it exists")


@needs_gold
@pytest.mark.parametrize("name", ["layer_rank_table", "layer_ntracks_table", "layer_nsig_table", "layer_score_table",
                                  "layer_score_neglog10p_table", "layer_rank_over_N_table", "method_comparison"])
def test_acat_tissue_matches_the_committed_sort1_tables(tmp_path, name):
    lr.rank_variant_tissues(GOLD / "fisher" / "track_long.csv", tmp_path, method="acat", group_level="tissue",
                            annotation=GOLD / "fisher" / "cell_type_annotation.csv", formats=("png",))
    new, old = pd.read_csv(tmp_path / f"{name}.csv"), pd.read_csv(GOLD / "acat_tissue" / f"{name}.csv")
    assert new.shape == old.shape
    new.columns = old.columns            # the index header is now 'group', it was 'cell_type'
    pd.testing.assert_frame_equal(new, old, check_exact=False, rtol=1e-9)


@needs_gold
def test_the_bundled_annotation_is_the_one_the_figure_was_built_with():
    bundled = lr.load_annotation()
    gold, _ = lr._read_commented_csv(GOLD / "fisher" / "cell_type_annotation.csv")
    cols = ["cell_type_key", "tissue_organ", "tissue_system"]
    pd.testing.assert_frame_equal(bundled[cols].reset_index(drop=True), gold[cols].reset_index(drop=True))


def test_the_bundled_annotation_loads_and_has_both_levels():
    a = lr.load_annotation()
    assert a.cell_type_key.is_unique and a.tissue_organ.notna().all() and a.tissue_system.notna().all()
    assert a.tissue_organ.nunique() > a.tissue_system.nunique() > 5


# ---------------------------------------------------------------- RNA-seq: gene x track tables

def _rna_rows(spec):
    """spec: {gene: (distance_bp, [(track_id, cell_type, signed_percentile, raw_score), ...])}"""
    return [dict(gene=g, distance_bp=d, track_id=t, cell_type=c, percentile=p, raw_score=r)
            for g, (d, tr) in spec.items() for t, c, p, r in tr]


@pytest.fixture
def rna_two_genes(tmp_path):
    """Two genes over the same tracks: A is strongly changed, B barely. Percentiles are both ~1 (they saturate)."""
    spec = {"A": (0, [("RNA/x1/+", "HepG2", 1.0, 0.9), ("RNA/x2/+", "K562", 0.99, 0.5)]),
            "B": (5000, [("RNA/x1/+", "HepG2", 0.99, 0.01), ("RNA/x2/+", "K562", 1.0, 0.02)])}
    paths = lr.write_rna_tables(_rna_rows(spec), "alphagenome", tmp_path, "v", "chr1:1 A>T")
    other = tmp_path / "other.csv"
    lr._write_commented(_tracks({"hepg2": [(0.9, 0.1)], "k562": [(0.8, 0.1)]}, layer="DNase"), other,
                        lr._header("chr1:1 A>T", "x"))
    return paths, other, tmp_path


def _stranded_rows():
    """Gene P (+ strand) and gene M (- strand) over a stranded pair of tracks and one unstranded track."""
    tr = [("RNA/x/+", "HepG2"), ("RNA/x/-", "HepG2"), ("RNA/y/.", "K562")]
    spec = {"P": (10, [(t, c, 0.9 if t.endswith("+") else 0.99, 0.5 if t.endswith("+") else 0.9) for t, c in tr]),
            "M": (20, [(t, c, 0.8 if t.endswith("+") else 0.97, 0.1 if t.endswith("+") else 0.4) for t, c in tr])}
    rows = _rna_rows(spec)
    for r in rows:
        r["strand"] = {"P": "+", "M": "-"}[r["gene"]]
    return rows


def test_rna_tables_are_genes_by_tracks_and_keep_strand_specific_tracks():
    pct, raw, tracks = lr.rna_rows_to_tables(_stranded_rows(), "alphagenome")
    assert list(pct.gene) == ["P", "M"]                                    # nearest first
    assert list(tracks.track_id) == ["RNA/x/+", "RNA/x/-", "RNA/y/."]       # nothing collapsed in the table
    assert list(tracks.track_key) == ["RNA/x", "RNA/x", "RNA/y/."] and list(tracks.strand) == ["+", "-", "."]
    assert pct.set_index("gene").loc["P", "strand"] == "+"
    assert raw.set_index("gene").loc["P", "RNA/x/-"] == 0.9


def test_the_gene_summary_counts_only_reads_on_a_matching_strand():
    pct, _, _ = lr.rna_rows_to_tables(_stranded_rows(), "alphagenome")
    g = pct.set_index("gene")
    # P (+) is read on x/+ and the unstranded y, not on x/-; M (-) on x/- and y, not on x/+
    assert g.loc["P", "n_tracks"] == 2 and g.loc["M", "n_tracks"] == 2
    assert g.loc["P", "median_abs_raw"] == pytest.approx(0.7)       # median of x/+ (0.5) and y (0.9); not x/- (0.9) twice
    assert g.loc["M", "median_abs_raw"] == pytest.approx(0.4)


def test_collapse_keeps_the_stronger_strand_whatever_the_gene_strand(tmp_path):
    paths = lr.write_rna_tables(_stranded_rows(), "alphagenome", tmp_path, "s", "v")
    long, _, _ = lr.load_rna_tables(paths["rna_gene_percentiles_csv"], strand_mode="collapse")
    p = long[long.gene == "P"].set_index("track_id")
    assert list(long.track_id.unique()) and set(long.track_id) == {"RNA/x", "RNA/y/."}
    assert p.loc["RNA/x", "percentile"] == 0.99 and p.loc["RNA/x", "track_strand"] == "-"   # P is a + gene: wrong strand won


def test_matched_reads_a_gene_only_on_its_own_strand_or_an_unstranded_track(tmp_path):
    paths = lr.write_rna_tables(_stranded_rows(), "alphagenome", tmp_path, "s", "v")
    long, _, _ = lr.load_rna_tables(paths["rna_gene_percentiles_csv"], strand_mode="matched")
    got = {(r.gene, r.track_id): r.track_strand for r in long.itertuples()}
    assert got == {("P", "RNA/x"): "+", ("P", "RNA/y/."): ".", ("M", "RNA/x"): "-", ("M", "RNA/y/."): "."}
    assert long.set_index(["gene", "track_id"]).loc[("P", "RNA/x"), "raw_score"] == 0.5     # the + track, not the stronger -


def test_a_gene_of_unknown_strand_is_read_on_every_track(tmp_path):
    rows = _stranded_rows()
    for r in rows:
        r["strand"] = "?"
    paths = lr.write_rna_tables(rows, "alphagenome", tmp_path, "s", "v")
    long, _, _ = lr.load_rna_tables(paths["rna_gene_percentiles_csv"], strand_mode="matched")
    # both strands of x are eligible, and they share one track key, so the stronger is kept: x and y per gene
    assert long.groupby("gene").size().tolist() == [2, 2]
    assert long.set_index(["gene", "track_id"]).loc[("P", "RNA/x"), "track_strand"] == "-"


def test_an_unknown_strand_mode_is_an_error(tmp_path):
    paths = lr.write_rna_tables(_stranded_rows(), "alphagenome", tmp_path, "s", "v")
    with pytest.raises(ValueError, match="strand_mode"):
        lr.load_rna_tables(paths["rna_gene_percentiles_csv"], strand_mode="both")


def test_a_noise_floor_percentile_stays_blank_but_the_raw_score_is_kept(tmp_path):
    pct, raw, _ = lr.rna_rows_to_tables(_rna_rows({"A": (0, [("t", "X", None, 0.0004)])}), "alphagenome")
    assert pd.isna(pct.set_index("gene").loc["A", "t"]) and raw.set_index("gene").loc["A", "t"] == 0.0004
    assert pct.set_index("gene").loc["A", "n_tracks"] == 0


def test_rna_tables_round_trip_through_their_files(rna_two_genes):
    paths, _, _ = rna_two_genes
    long, summary, label = lr.load_rna_tables(paths["rna_gene_percentiles_csv"])
    assert label == "chr1:1 A>T" and set(long.gene) == {"A", "B"} and set(long.molecular_layer) == {"RNA-seq"}
    assert (long.percentile >= 0).all()                                   # the ranking works on magnitudes
    assert long.set_index(["gene", "track_id"]).loc[("A", "RNA/x1"), "raw_score"] == 0.9
    assert set(long.model) == {"AlphaGenome"} and set(long.cell_type_key) == {"hepg2", "k562"}
    assert summary.set_index("gene").loc["A", "median_abs_raw"] > summary.set_index("gene").loc["B", "median_abs_raw"]


def test_a_gene_is_never_picked_silently(rna_two_genes):
    paths, other, tmp = rna_two_genes
    r = lr.rank_variant_tissues([other, paths["rna_gene_percentiles_csv"]], tmp / "o", formats=("png",))
    assert "RNA-seq" not in r["layers"] and r["rna_gene"] is None and r["rna_genes_available"] == ["A", "B"]
    note = next(n for n in r["notes"] if n.startswith("RNA-seq"))
    assert "pass rna_gene" in note and note.index("A (") < note.index("B (")   # largest effect listed first


def test_the_chosen_gene_fills_the_rna_column_and_is_named_on_the_figure(rna_two_genes):
    paths, other, tmp = rna_two_genes
    r = lr.rank_variant_tissues([other, paths["rna_gene_percentiles_csv"]], tmp / "o", rna_gene="b", formats=("png",),
                                annotation=_ann({"a": ("A", "S"), "hepg2": ("Liver", "S"), "k562": ("Blood", "S")}))
    assert r["rna_gene"] == "B" and r["layers"]["RNA-seq"]["oracle"] == "AlphaGenome"
    assert not any(n.startswith("RNA-seq: ") and "none was chosen" in n for n in r["notes"])


def test_the_gene_changes_the_ranking(rna_two_genes):
    """Same tracks, same percentiles: only the raw-score tie-break and the gene differ -> A and B must differ."""
    paths, _, tmp = rna_two_genes
    ann = _ann({"hepg2": ("Liver", "S"), "k562": ("Blood", "S")})
    tops = {g: lr.rank_variant_tissues(paths["rna_gene_percentiles_csv"], tmp / g, rna_gene=g, annotation=ann,
                                       layers=["RNA-seq"], formats=("png",), method="acat")["top"]
            for g in "AB"}
    assert [x["group"] for x in tops["A"]] == ["Liver", "Blood"]          # A: HepG2 sits at the floor
    assert [x["group"] for x in tops["B"]] == ["Blood", "Liver"]          # B: K562 does


def test_an_unknown_gene_is_an_error_that_lists_the_real_ones(rna_two_genes):
    paths, other, tmp = rna_two_genes
    with pytest.raises(ValueError, match=r"NOPE.*\['A', 'B'\]"):
        lr.rank_variant_tissues([other, paths["rna_gene_percentiles_csv"]], tmp / "o", rna_gene="NOPE")


def test_a_single_gene_is_used_without_being_named(tmp_path):
    paths = lr.write_rna_tables(_rna_rows({"ONLY": (0, [("t", "HepG2", 0.99, 0.3)])}), "alphagenome", tmp_path, "s", "v")
    r = lr.rank_variant_tissues(paths["rna_gene_percentiles_csv"], tmp_path / "o", formats=("png",),
                                annotation=_ann({"hepg2": ("Liver", "S")}))
    assert r["rna_gene"] == "ONLY" and "RNA-seq" in r["layers"]


def test_the_chosen_gene_replaces_rna_rows_already_in_the_track_table(rna_two_genes):
    paths, _, tmp = rna_two_genes
    old = _tracks({"hepg2": [(0.5, 0.01)]}, layer="RNA-seq")
    lr._write_commented(old, tmp / "old.csv", lr._header("v", "x"))
    r = lr.rank_variant_tissues([tmp / "old.csv", paths["rna_gene_percentiles_csv"]], tmp / "o", rna_gene="A",
                                formats=("png",), annotation=_ann({"hepg2": ("Liver", "S"), "k562": ("Blood", "S")}))
    assert any("replaced" in n for n in r["notes"]) and r["layers"]["RNA-seq"]["n_groups_ranked"] == 2


def test_an_rna_matrix_without_its_companions_says_what_to_pass(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("# v\ngene,distance_bp,n_tracks,n_sig_0.95,median_abs_raw,t\nA,0,1,1,0.1,0.9\n")
    with pytest.raises(ValueError, match="companion"):
        lr.load_rna_tables(p)


@needs_gold
def test_rna_through_the_gene_tables_reproduces_the_committed_sort1_figure(tmp_path):
    """All 29 genes of the SORT1 atlas written as gene x track tables; choosing SORT1 gives the committed ranks."""
    atlas = REPO / "analysis_output" / "sort1_variant_atlas" / "tf_histone_cage_rna_alphagenome.csv"
    if not atlas.exists():
        pytest.skip("SORT1 atlas not present")
    a = pd.read_csv(atlas)
    a = a[a.assay_type == "RNA"]
    rows = [dict(gene=r.region_label.replace(" (exons)", ""), distance_bp=0, track_id=r.assay_id, cell_type=r.cell_type,
                 percentile=None if pd.isna(r.quantile_score) else r.quantile_score, raw_score=r.raw_score)
            for r in a.itertuples()]
    w = lr.write_rna_tables(rows, "alphagenome", tmp_path, "sort1", "SORT1 rs12740374")
    assert len(w["rna_genes"]) == 29
    long, _ = lr.load_track_tables(GOLD / "fisher" / "track_long.csv")
    lr._write_commented(long[long.molecular_layer != "RNA-seq"], tmp_path / "no_rna.csv", lr._header("SORT1 rs12740374", "x"))
    lr.rank_variant_tissues([tmp_path / "no_rna.csv", w["rna_gene_percentiles_csv"]], tmp_path / "r", method="acat",
                            group_level="tissue", rna_gene="SORT1", formats=("png",))
    for name in ("layer_rank_table", "layer_ntracks_table", "layer_nsig_table", "layer_rank_over_N_table"):
        new, old = pd.read_csv(tmp_path / "r" / f"{name}.csv"), pd.read_csv(GOLD / "acat_tissue" / f"{name}.csv")
        new.columns = old.columns
        pd.testing.assert_frame_equal(new, old)
