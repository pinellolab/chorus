"""``plot_variant_gene_response``: the "which cell type responds?" dot plot from the RNA gene tables.

The rule that names a gene is the part worth pinning. The published liver panel's caption says its four genes are
"above the noise floor on every track", but in its own data only PSRC1 is (12/12); CELSR2 and MYBPHL are on 11 and
SORT1 on 8, while all four are boxplot outliers on 11-12 of 12 tracks and no other gene on any. So a gene is named
for being an outlier on most tracks with a median effect above the floor, and the notes say where it dips inside it.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chorus.analysis import gene_response as gr
from chorus.analysis import layer_ranking as lr

REPO = Path(__file__).resolve().parent.parent
LN2 = np.log(2)
FLOOR_LN = 0.0417


def _ann(mapping):
    return pd.DataFrame([{"cell_type_key": k, "tissue_organ": o, "tissue_system": s} for k, (o, s) in mapping.items()])


ANN = _ann({"hepg2": ("Liver", "Digestive"), "liver": ("Liver", "Digestive"), "k562": ("Blood", "Immune"),
            "brain": ("Brain", "Nervous")})


def _rows(effects, tracks, strands=None, n_noise=24, seed=0):
    """effects: {gene: {track_index: log2fc}}; every other (gene, track) is tiny noise. Raw scores are ln fold change."""
    rng = np.random.default_rng(seed)
    genes = list(effects) + [f"N{i}" for i in range(n_noise)]
    rows = []
    for g in genes:
        for i, (tid, cell) in enumerate(tracks):
            l2 = effects.get(g, {}).get(i, rng.normal(0, 0.003))
            rows.append(dict(gene=g, distance_bp=1000 * (genes.index(g) + 1), strand=(strands or {}).get(g, "?"),
                             track_id=tid, cell_type=cell, percentile=1.0 if abs(l2) > 0.05 else 0.5,
                             raw_score=l2 * LN2))
    return rows


TRACKS = [("RNA/a polyA plus RNA-seq/.", "HepG2"), ("RNA/a total RNA-seq/.", "HepG2"),
          ("RNA/b polyA plus RNA-seq/.", "liver"), ("RNA/b total RNA-seq/.", "liver")]


def _write(tmp, rows, model="alphagenome"):
    return lr.write_rna_tables(rows, model, tmp, "v", "chr1:1 G>T")["rna_gene_percentiles_csv"]


@pytest.fixture
def table(tmp_path):
    eff = {"UP": {0: 1.0, 1: 0.8, 2: 0.9, 3: 0.7},                    # outlier and above the floor everywhere
           "DIPS": {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.02},                # outlier on all, inside the floor on one
           "RARE": {0: 0.6}}                                         # outlier on one track of four only
    return _write(tmp_path, _rows(eff, TRACKS)), tmp_path


def run(table_path, out, **kw):
    kw.setdefault("annotation", ANN)
    kw.setdefault("formats", ("png",))
    kw.setdefault("tissue", "Liver")           # required: the test tables' HepG2 + liver tracks are all Liver
    return gr.plot_variant_gene_response(table_path, out, **kw)


# ---------------------------------------------------------------- which genes are named

def test_genes_are_named_for_being_outliers_on_most_tracks(table):
    path, tmp = table
    r = run(path, tmp / "o")
    assert [g["gene"] for g in r["genes"]] == ["UP", "DIPS"]          # largest median |effect| first
    assert "RARE" not in r["named_genes"]                              # an outlier on 1 of 4 tracks is not a responder


def test_a_gene_that_dips_inside_the_floor_is_still_named_and_the_dip_is_reported(table):
    path, tmp = table
    r = run(path, tmp / "o")
    dips = next(g for g in r["genes"] if g["gene"] == "DIPS")
    assert (dips["tracks_as_outlier"], dips["tracks_above_floor"], dips["of_tracks"]) == (4, 3, 4)
    assert any("DIPS is inside the noise floor on 1 of 4 tracks" in n for n in r["notes"])


def test_the_strict_every_track_rule_is_available(table):
    path, tmp = table
    r = run(path, tmp / "o", min_track_fraction=1.0)
    assert r["named_genes"] == ["UP", "DIPS"]                          # both are outliers on every track
    # RARE is an outlier on 1 of 4 tracks (0.25), which clears a 0.1 fraction, but its median effect is inside the
    # floor: being an outlier somewhere is not enough to be named
    r = run(path, tmp / "o2", min_track_fraction=0.1)
    assert "RARE" not in r["named_genes"]
    g = pd.read_csv(r["files"]["by_gene_csv"]).set_index("gene")
    assert g.loc["RARE", "fraction_outlier"] == 0.25 and g.loc["RARE", "median_abs_log2fc"] < r["noise_floor_log2"]


def test_a_gene_below_the_floor_is_not_named_however_consistent_it_is(tmp_path):
    eff = {"QUIET": {i: 0.04 for i in range(4)}}                      # 0.04 log2 < 0.060 floor, an outlier among noise
    r = run(_write(tmp_path, _rows(eff, TRACKS)), tmp_path / "o")
    assert r["named_genes"] == [] and any("no gene is a boxplot outlier" in n for n in r["notes"])


def test_the_floor_is_the_measured_alphagenome_null_in_log2(table):
    path, tmp = table
    assert run(path, tmp / "o")["noise_floor_log2"] == pytest.approx(FLOOR_LN / LN2, abs=1e-5)


def test_an_oracle_without_a_measured_floor_gets_none_unless_one_is_given(tmp_path):
    p = _write(tmp_path, _rows({"UP": {i: 1.0 for i in range(4)}}, TRACKS), model="borzoi")
    r = run(p, tmp_path / "o")
    assert r["noise_floor_log2"] == 0.0 and any("no measured RNA noise floor" in n for n in r["notes"])
    assert run(p, tmp_path / "o2", noise_floor_log2=0.3)["noise_floor_log2"] == 0.3


def test_at_most_max_named_genes_are_named_largest_first(tmp_path):
    eff = {f"G{i}": {t: 1.0 + i for t in range(4)} for i in range(5)}
    r = run(_write(tmp_path, _rows(eff, TRACKS)), tmp_path / "o", max_named=2)
    assert r["named_genes"] == ["G4", "G3"] and any("5 genes qualify" in n for n in r["notes"])


def test_colours_follow_the_order_of_effect_so_the_published_panel_is_reproduced(table):
    path, tmp = table
    assert [g["colour"] for g in run(path, tmp / "o")["genes"]] == gr.GENE_COLOURS[:2]
    assert gr.GENE_COLOURS[:4] == ["#0072B2", "#009E73", "#CC79A7", "#D55E00"]   # PSRC1, CELSR2, MYBPHL, SORT1


# ---------------------------------------------------------------- strands

def test_a_gene_is_read_only_on_the_track_of_its_own_strand(tmp_path):
    """MINUSGENE is changed on the '-' track and untouched on the '+' one; reading it on '+' must not count."""
    tracks = [("RNA/a polyA plus RNA-seq/+", "HepG2"), ("RNA/a polyA plus RNA-seq/-", "HepG2"),
              ("RNA/b polyA plus RNA-seq/+", "liver"), ("RNA/b polyA plus RNA-seq/-", "liver")]
    rows = _rows({"MINUSGENE": {1: 0.9, 3: 0.9}}, tracks, strands={"MINUSGENE": "-"})
    r = run(_write(tmp_path, rows), tmp_path / "o", min_track_fraction=1.0)
    t = pd.read_csv(r["files"]["by_track_csv"])
    m = t[t.gene == "MINUSGENE"]
    assert set(m.track_strand) == {"-"} and len(m) == 2               # two tracks, not four
    assert r["named_genes"] == ["MINUSGENE"] and r["genes"][0]["of_tracks"] == 2


def test_the_two_strands_of_a_track_are_one_track(tmp_path):
    tracks = [("RNA/a polyA plus RNA-seq/+", "HepG2"), ("RNA/a polyA plus RNA-seq/-", "HepG2")]
    rows = _rows({"G": {0: 0.9, 1: 0.9}}, tracks, strands={"G": "+"})
    assert len(run(_write(tmp_path, rows), tmp_path / "o")["tracks"]) == 1


# ---------------------------------------------------------------- selecting tracks

def test_the_tissue_chooses_the_tracks(table):
    path, tmp = table
    ann = _ann({"hepg2": ("Liver", "D"), "liver": ("Blood", "I")})
    r = run(path, tmp / "o", tissue="liver", annotation=ann)           # case-insensitive
    assert r["tissue"] == "Liver" and len(r["tracks"]) == 2 and {t.split()[0] for t in r["tracks"]} == {"RNA/a"}


def test_the_system_level_matches_the_system_column(table):
    path, tmp = table
    r = run(path, tmp / "o", tissue="digestive", group_level="system")
    assert r["tissue"] == "Digestive" and len(r["tracks"]) == 4


def test_an_unknown_tissue_lists_what_exists(table):
    path, tmp = table
    with pytest.raises(ValueError, match=r"'Heart'.*Liver \(4\)"):
        run(path, tmp / "o", tissue="Heart")


def test_the_tissue_is_required_there_is_no_default(table):
    path, tmp = table
    with pytest.raises(TypeError, match="tissue"):
        gr.plot_variant_gene_response(path, tmp / "o", annotation=ANN)
    with pytest.raises(ValueError, match=r"tissue is required.*Liver \(4\)"):
        gr.plot_variant_gene_response(path, tmp / "o", tissue="", annotation=ANN)


def test_a_group_with_too_many_tracks_for_one_panel_is_refused(tmp_path):
    tracks = [(f"RNA/t{i} total RNA-seq/.", "HepG2" if i % 2 else "liver") for i in range(6)]
    p = _write(tmp_path, _rows({"UP": {0: 1.0}}, tracks))
    with pytest.raises(ValueError, match=r"6 RNA-seq tracks.*limit 4"):
        run(p, tmp_path / "o", max_tracks=4)


# ---------------------------------------------------------------- outputs

def test_files_and_the_figure_size(table):
    from PIL import Image
    path, tmp = table
    r = run(path, tmp / "o", formats=("png", "pdf"), variant_label="rs1 G>T")
    for k in ("by_track_csv", "by_gene_csv", "run_summary"):
        assert Path(r["files"][k]).stat().st_size > 0
    png = next(p for p in r["files"]["figure"] if p.endswith(".png"))
    w, h = Image.open(png).size
    assert h == round(1.9 * 300) and w >= round(4.6 * 300)
    json.dumps(r)                                                       # the reply must be JSON-serialisable
    g = pd.read_csv(r["files"]["by_gene_csv"]).set_index("gene")
    assert bool(g.loc["UP", "named"]) and not bool(g.loc["RARE", "named"])


def test_the_figure_does_not_leak_matplotlib_state(table):
    import matplotlib as mpl
    path, tmp = table
    before = (mpl.rcParams["font.size"], mpl.rcParams["savefig.dpi"], mpl.rcParams["pdf.fonttype"])
    run(path, tmp / "o")
    assert (mpl.rcParams["font.size"], mpl.rcParams["savefig.dpi"], mpl.rcParams["pdf.fonttype"]) == before


def test_the_y_label_names_the_alleles_from_the_variant_label(table):
    path, tmp = table
    run(path, tmp / "o", variant_label="rs12740374 (chr1:109,274,968 G>T)")   # drawn without error; label parsed below
    import re
    m = re.search(r"([ACGT]+)>([ACGT]+)", "rs12740374 (chr1:109,274,968 G>T)")
    assert (m.group(2), m.group(1)) == ("T", "G")


def test_cell_types_default_to_the_strongest_responder_first_and_can_be_ordered(table):
    path, tmp = table
    default = run(path, tmp / "o", formats=("png",))
    assert default["named_genes"]
    ordered = run(path, tmp / "o2", cell_type_order=["liver", "HepG2"])    # explicit order is accepted, case-insensitive
    assert ordered["tracks"] == default["tracks"]


def test_an_unknown_group_level_or_fraction_is_an_error(table):
    path, tmp = table
    with pytest.raises(ValueError, match="group_level"):
        run(path, tmp / "o", group_level="cell")
    with pytest.raises(ValueError, match="min_track_fraction"):
        run(path, tmp / "o", min_track_fraction=0)


# ---------------------------------------------------------------- the MCP tool

def import_signature_requires_tissue(tool):
    import inspect
    fn = getattr(tool, "fn", tool)
    return inspect.signature(fn).parameters["tissue"].default is inspect.Parameter.empty


def test_the_tool_is_registered_and_writes_under_the_output_dir(table, monkeypatch):
    import chorus.mcp.server as server
    path, tmp = table
    assert "plot_variant_gene_response" in server.registered_tool_names()
    monkeypatch.setattr(server, "_state", lambda: SimpleNamespace(output_dir=str(tmp)))
    fn = getattr(server.plot_variant_gene_response, "fn", server.plot_variant_gene_response)
    res = fn(rna_gene_percentiles_csv=path, tissue="Liver", annotation_csv=None)
    assert "error" not in res or "annotation" not in res["error"], res
    assert import_signature_requires_tissue(server.plot_variant_gene_response)


def test_the_tool_turns_bad_input_into_a_structured_error(monkeypatch, tmp_path):
    import chorus.mcp.server as server
    monkeypatch.setattr(server, "_state", lambda: SimpleNamespace(output_dir=str(tmp_path)))
    fn = getattr(server.plot_variant_gene_response, "fn", server.plot_variant_gene_response)
    res = fn(rna_gene_percentiles_csv="/nonexistent.csv")
    assert "error" in res and res["tool"] == "plot_variant_gene_response"


# ---------------------------------------------------------------- golden: the published liver panel

needs_gold = pytest.mark.skipif(
    not ((REPO / "fig3" / "data" / "sort1_liver_rnaseq_genes.csv").exists()
         and (REPO / "analysis_output" / "sort1_variant_atlas" / "tf_histone_cage_rna_alphagenome.csv").exists()),
    reason="the SORT1 panel data and atlas are not tracked")


@needs_gold
def test_the_liver_panel_is_reproduced_from_the_discovery_style_tables(tmp_path):
    """Same 12 tracks, same four genes in the same colours, values within 5% of the panel's own scoring script."""
    P = pd.read_csv(REPO / "fig3" / "data" / "sort1_liver_rnaseq_genes.csv")
    P["tid"] = P.track.str.replace(r"/[+-]$", "", regex=True)
    strand = P.drop_duplicates("gene").set_index("gene").strand.to_dict()
    A = pd.read_csv(REPO / "analysis_output" / "sort1_variant_atlas" / "tf_histone_cage_rna_alphagenome.csv")
    A = A[A.assay_type == "RNA"]
    rows = []
    for r in A.itertuples():
        g = r.region_label.replace(" (exons)", "")
        rows.append(dict(gene=g, distance_bp=0, strand=strand.get(g, "?"), track_id=r.assay_id, cell_type=r.cell_type,
                         percentile=None if pd.isna(r.quantile_score) else r.quantile_score, raw_score=r.raw_score))
    path = _write(tmp_path, rows)
    res = gr.plot_variant_gene_response(path, tmp_path / "liver", tissue="Liver", formats=("png",))
    assert set(res["tracks"]) == set(P.tid) and len(res["tracks"]) == 12
    assert [(g["gene"], g["colour"]) for g in res["genes"]] == [
        ("PSRC1", "#0072B2"), ("CELSR2", "#009E73"), ("MYBPHL", "#CC79A7"), ("SORT1", "#D55E00")]
    t = pd.read_csv(res["files"]["by_track_csv"])
    m = P.merge(t, left_on=["gene", "tid"], right_on=["gene", "track_id"], suffixes=("_panel", "_new"))
    assert len(m) == len(P) == 336
    assert (m.log2fc_panel - m.log2fc_new).abs().max() < 0.05
    # the caption's claim is stricter than the data: SORT1 is inside the floor on 4 of the 12 tracks
    assert any("SORT1 is inside the noise floor on 4 of 12 tracks" in n for n in res["notes"])
