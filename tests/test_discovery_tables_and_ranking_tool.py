"""``discover_variant`` writes the tables that ``rank_variant_tissues`` ranks, and the tool is wired up.

The discovery export is opt-in on the Python API (``write_tables=False``) because
``scripts/regenerate_examples.py`` and the walkthrough notebooks point ``output_path`` at committed
example directories and ``json.dumps`` the whole result: an unconditional export would add two files
to every regenerated example, and returning a DataFrame would be stringified into the committed JSON.
The MCP tool turns it on.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from chorus.analysis import discovery
from chorus.analysis.discovery import TrackEffect


def _unwrap(fn):
    for attr in ("fn", "__wrapped__"):
        fn = getattr(fn, attr, fn)
    return fn


def _effects(pctile=0.99):
    def e(tid, layer, cell, assay, raw=0.5, p=pctile):
        return TrackEffect(assay_id=tid, layer=layer, cell_type=cell, description=f"{assay}:{cell}",
                           raw_score=raw, abs_score=abs(raw), ref_value=10.0, alt_value=12.0,
                           effect_pctile=p, assay=assay)
    out = [e("DNASE:HepG2", "chromatin_accessibility", "HepG2", "DNASE"),
           e("CHIP:HepG2:CEBPB:+", "tf_binding", "HepG2", "CHIP", p=0.9),
           e("CHIP:HepG2:CEBPB:-", "tf_binding", "HepG2", "CHIP", p=0.95)]
    if pctile is None:                       # no normalizer: no track has a percentile
        for x in out:
            x.effect_pctile = None
    return out


class _Oracle:
    """Stands in for a loaded multi-track oracle: one prediction call, a catalogue."""

    def __init__(self):
        self.calls = 0

    def describe_tracks(self):
        return [SimpleNamespace(track_id="DNASE:HepG2", cell_type="HepG2", assay="DNASE"),
                SimpleNamespace(track_id="CHIP:HepG2:CEBPB", cell_type="HepG2", assay="CHIP")]

    def predict_variant_effect(self, **kw):
        from chorus.core.result import OraclePrediction
        self.calls += 1
        # empty predictions: the report step finds nothing to filter, so no report is built
        return {"predictions": {"reference": OraclePrediction(), "T": OraclePrediction()},
                "effect_sizes": {}, "variant_info": {"position": "chr1:109274968"}}


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(discovery, "_score_all_tracks", lambda *a, **k: _effects())
    monkeypatch.setattr("chorus.analysis.variant_report.build_variant_report", lambda *a, **k: MagicMock())


def _run(tmp_path, oracle=None, **kw):
    return discovery.discover_variant_effects(
        oracle or _Oracle(), "alphagenome", "chr1:109274968", ["G", "T"], output_path=str(tmp_path), **kw)


def test_tables_are_not_written_unless_asked(patched, tmp_path):
    res = _run(tmp_path)
    assert not list(tmp_path.glob("*.csv")) and "track_scores_csv" not in res
    json.dumps({k: v for k, v in res.items() if k != "report"}, default=str)   # still serialisable


def test_write_tables_writes_both_files_and_returns_their_paths(patched, tmp_path):
    res = _run(tmp_path, write_tables=True, variant_label="SORT1 rs12740374")
    t = pd.read_csv(res["track_scores_csv"], skiprows=5)
    assert res["track_scores_csv"].endswith("chr1_109274968_alphagenome_track_scores.csv")
    assert (res["prediction_matrix_csv"]).endswith("_prediction_matrix.csv")
    assert open(res["track_scores_csv"]).readline().strip() == "# SORT1 rs12740374"
    # the two BPNet-style strands collapse to one track; the stronger one is kept
    assert sorted(t.track_id) == ["CHIP:HepG2:CEBPB", "DNASE:HepG2"]
    assert t.set_index("track_id").loc["CHIP:HepG2:CEBPB", "percentile"] == 0.95
    assert res["tables"] == {"n_tracks_in_table": 2, "n_tracks_without_percentile": 0}


def test_the_default_label_is_the_coordinates_and_alleles(patched, tmp_path):
    res = _run(tmp_path, write_tables=True)
    assert open(res["track_scores_csv"]).readline().strip() == "# chr1:109,274,968 G>T"


def test_write_tables_without_an_output_dir_fails_before_the_model_runs():
    oracle = _Oracle()
    with pytest.raises(ValueError, match="output_path"):
        discovery.discover_variant_effects(oracle, "alphagenome", "chr1:1", ["G", "T"], write_tables=True)
    assert oracle.calls == 0


def test_no_normalizer_means_no_percentiles_and_a_stated_reason_not_empty_files(monkeypatch, tmp_path):
    monkeypatch.setattr(discovery, "_score_all_tracks", lambda *a, **k: _effects(pctile=None))
    monkeypatch.setattr("chorus.analysis.variant_report.build_variant_report", lambda *a, **k: MagicMock())
    res = _run(tmp_path, write_tables=True)
    assert "track_scores_csv" not in res and "no background normalizer" in res["tables"]["skipped"]
    assert not list(tmp_path.glob("*.csv"))


def test_a_failing_export_never_costs_the_caller_the_discovery_result(patched, monkeypatch, tmp_path):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr("chorus.analysis.layer_ranking.write_discovery_tables", boom)
    res = _run(tmp_path, write_tables=True)
    assert "disk full" in res["tables"]["skipped"] and res["cell_type_ranking"]


def test_a_catalogue_failure_falls_back_to_the_parsed_cell_types(patched, tmp_path):
    class NoCatalogue(_Oracle):
        def describe_tracks(self):
            # discovery's own track enumeration is the first call and must work; the export's is the second
            self.n = getattr(self, "n", 0) + 1
            if self.n > 1:
                raise RuntimeError("no catalogue")
            return super().describe_tracks()
    res = _run(tmp_path, oracle=NoCatalogue(), write_tables=True)
    assert res["tables"]["n_tracks_in_table"] == 2


def test_the_exported_table_can_be_ranked_end_to_end(patched, tmp_path):
    from chorus.analysis.layer_ranking import rank_variant_tissues
    res = _run(tmp_path, write_tables=True)
    out = rank_variant_tissues(res["track_scores_csv"], tmp_path / "rank", formats=("png",))
    assert out["variant_label"] == "chr1:109,274,968 G>T"
    assert out["top"][0]["group"] == "Liver"            # HepG2 -> Liver in the bundled annotation
    assert any("RNA-seq" in n and "omitted" in n for n in out["notes"])


# ---------------------------------------------------------------- MCP tools

def test_the_tool_is_registered():
    from chorus.mcp.server import registered_tool_names
    assert "rank_variant_tissues" in registered_tool_names()


def test_discover_variant_turns_the_tables_on_and_labels_them_with_the_rsid(monkeypatch, tmp_path):
    import chorus.mcp.server as server
    monkeypatch.setattr(server, "_state", lambda: SimpleNamespace(
        get_oracle=lambda n: object(), get_normalizer=lambda n: None, output_dir=str(tmp_path)))
    monkeypatch.setattr(server, "_resolve_rsid_to_position", lambda *a, **k: ("chr1:109274968", "G", "T"))
    seen = {}
    monkeypatch.setattr("chorus.analysis.discovery.discover_variant_effects",
                        lambda *a, **k: seen.update(k) or {"layer_rankings": {}, "cell_type_ranking": []})
    _unwrap(server.discover_variant)(oracle_name="alphagenome", position="rs12740374", ref_allele="G",
                                     alt_alleles=["T"])
    assert seen["write_tables"] is True
    assert seen["variant_label"] == "rs12740374 (chr1:109,274,968 G>T)"


def test_rank_variant_tissues_tool_writes_into_the_output_dir_by_default(patched, monkeypatch, tmp_path):
    import chorus.mcp.server as server
    monkeypatch.setattr(server, "_state", lambda: SimpleNamespace(output_dir=str(tmp_path)))
    tables = _run(tmp_path, write_tables=True)["track_scores_csv"]
    res = _unwrap(server.rank_variant_tissues)(track_scores_csv=[tables], method="acat", group_level="system")
    assert "error" not in res, res
    assert (tmp_path / "tissue_rank_acat_system" / "layer_rank_table.png").exists()
    assert res["method"] == "acat" and res["group_level"] == "tissue_system"
    json.dumps(res)                                    # the reply must be JSON-serialisable


def test_rank_variant_tissues_tool_turns_bad_input_into_a_structured_error(monkeypatch, tmp_path):
    import chorus.mcp.server as server
    monkeypatch.setattr(server, "_state", lambda: SimpleNamespace(output_dir=str(tmp_path)))
    res = server.rank_variant_tissues.fn(track_scores_csv=["/nonexistent.csv"]) \
        if hasattr(server.rank_variant_tissues, "fn") else server.rank_variant_tissues(track_scores_csv=["/nonexistent.csv"])
    assert "error" in res and res["tool"] == "rank_variant_tissues"


# ---------------------------------------------------------------- RNA-seq per gene

class _RnaTrack:
    """A prediction track whose exon-mean signal is ``level`` (the real scorer divides sums by bin counts)."""

    def __init__(self, assay_id, cell, level):
        self.assay_id, self.assay_type, self.cell_type = assay_id, "RNA", cell
        self.metadata = {"description": f"RNA:{cell}"}
        self._level = level
        self.prediction_interval = SimpleNamespace(reference=SimpleNamespace(start=0, end=10_000_000))

    def score_region(self, chrom, start, end, agg):
        return self._level(start) * (end - start)

    def region_bin_count(self, chrom, start, end):
        return end - start


class _Pred(dict):
    tracks: dict = {}                              # discovery's report step looks tracks up by id


class _Norm:
    """Legacy-shaped normalizer: percentile = clipped 10x raw effect."""

    def normalize(self, key, v, signed=False):
        return max(-1.0, min(1.0, 10.0 * v))

    def normalize_baseline(self, *a, **k):
        return 0.5


GENE_EXONS = {"NEAR": [(1000, 1100)], "FAR": [(5000, 5100), (5200, 5300)], "NOEXON": []}


@pytest.fixture
def rna_world(monkeypatch):
    """Two RNA tracks over genes NEAR (variant inside), FAR and one gene with no exons."""
    def tracks(levels):
        return _Pred({
            "RNA/t1/+": _RnaTrack("RNA/t1/+", "HepG2", lambda st: levels["t1"][st]),
            "RNA/t2/+": _RnaTrack("RNA/t2/+", "K562", lambda st: levels["t2"][st])})
    ref = tracks({"t1": {1000: 5.0, 5000: 5.0, 5200: 5.0}, "t2": {1000: 5.0, 5000: 5.0, 5200: 5.0}})
    # t1: NEAR strongly up, FAR unchanged. t2: NEAR unchanged, FAR down.
    alt = tracks({"t1": {1000: 20.0, 5000: 5.0, 5200: 5.0}, "t2": {1000: 5.0, 5000: 1.0, 5200: 1.0}})
    result = {"predictions": {"reference": ref, "T": alt}, "variant_info": {"position": "chr1:1050"}}
    monkeypatch.setattr("chorus.analysis.variant_report._find_nearby_genes",
                        lambda chrom, pos, window: ["NEAR", "FAR", "NOEXON"])

    def exons(name):
        return pd.DataFrame([{"chrom": "chr1", "start": a, "end": b} for a, b in GENE_EXONS[name]],
                            columns=["chrom", "start", "end"])
    monkeypatch.setattr("chorus.utils.annotations.get_gene_exons", exons)
    return result


def test_every_nearby_gene_is_scored_against_every_rna_track(rna_world):
    rows = discovery._score_rna_genes(rna_world, "alphagenome", _Norm())
    got = {(r["gene"], r["track_id"]): r for r in rows}
    assert set(got) == {(g, t) for g in ("NEAR", "FAR") for t in ("RNA/t1/+", "RNA/t2/+")}   # no exons -> not scored
    assert got[("NEAR", "RNA/t1/+")]["raw_score"] > 1.0 > 0                  # 5 -> 20: a large log fold change
    assert got[("FAR", "RNA/t2/+")]["raw_score"] < 0                         # 5 -> 1: down
    assert got[("NEAR", "RNA/t1/+")]["percentile"] == 1.0                    # signed percentile from the normalizer
    assert got[("FAR", "RNA/t2/+")]["percentile"] == -1.0
    assert {r["cell_type"] for r in rows} == {"HepG2", "K562"}


def test_an_effect_inside_the_noise_floor_has_no_percentile_exactly_as_in_the_report(rna_world):
    rows = {(r["gene"], r["track_id"]): r for r in discovery._score_rna_genes(rna_world, "alphagenome", _Norm())}
    unchanged = rows[("NEAR", "RNA/t2/+")]                                   # 5.0 -> 5.0
    assert unchanged["raw_score"] == 0.0 and unchanged["percentile"] is None


def test_distance_is_zero_inside_the_gene_and_the_gap_outside_it(rna_world):
    d = {r["gene"]: r["distance_bp"] for r in discovery._score_rna_genes(rna_world, "alphagenome", _Norm())}
    assert d == {"NEAR": 0, "FAR": 5000 - 1050}                              # nearest end of FAR's span (5000..5300)


def test_the_requested_gene_is_scored_even_if_it_is_not_nearby(rna_world, monkeypatch):
    monkeypatch.setattr("chorus.analysis.variant_report._find_nearby_genes", lambda *a: ["FAR"])
    genes = {r["gene"] for r in discovery._score_rna_genes(rna_world, "alphagenome", _Norm(), gene_name="NEAR")}
    assert genes == {"NEAR", "FAR"}


def test_an_oracle_without_rna_tracks_yields_nothing(rna_world):
    for tr in list(rna_world["predictions"]["reference"].values()) + list(rna_world["predictions"]["T"].values()):
        tr.assay_type = "DNASE"
    assert discovery._score_rna_genes(rna_world, "enformer", _Norm()) == []


def _rna_oracle(result):
    class O(_Oracle):
        def predict_variant_effect(self, **kw):
            self.calls += 1
            return result
    return O()


def test_discovery_writes_the_rna_tables_and_keeps_rna_out_of_its_rankings(rna_world, patched, tmp_path):
    res = _run(tmp_path, oracle=_rna_oracle(rna_world), write_tables=True, normalizer=_Norm())
    assert res["tables"]["n_rna_genes"] == 2
    for k in ("rna_gene_percentiles_csv", "rna_gene_raw_scores_csv", "rna_tracks_csv"):
        assert (tmp_path / res[k].split("/")[-1]).exists(), k
    assert {g["gene"] for g in res["rna_genes"]} == {"NEAR", "FAR"}
    assert {"median_abs_raw", "n_sig_0.95", "distance_bp"} <= set(res["rna_genes"][0])
    json.dumps({k: v for k, v in res.items() if k != "report"}, default=str)
    assert not any("RNA" in e["assay_id"] for ly in res["layer_rankings"].values() for e in ly)
    assert "RNA" not in json.dumps(res["cell_type_ranking"])


def test_the_discovery_tables_rank_with_a_gene_chosen_afterwards(rna_world, patched, tmp_path):
    from chorus.analysis.layer_ranking import rank_variant_tissues
    res = _run(tmp_path, oracle=_rna_oracle(rna_world), write_tables=True, normalizer=_Norm())
    ins = [res["track_scores_csv"], res["rna_gene_percentiles_csv"]]
    none = rank_variant_tissues(ins, tmp_path / "a", formats=("png",))
    assert "RNA-seq" not in none["layers"] and set(none["rna_genes_available"]) == {"NEAR", "FAR"}
    near = rank_variant_tissues(ins, tmp_path / "b", rna_gene="NEAR", formats=("png",))
    assert near["rna_gene"] == "NEAR" and "RNA-seq" in near["layers"]


def test_a_failure_while_scoring_rna_does_not_take_the_other_tables_with_it(rna_world, patched, monkeypatch, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("no annotation")
    monkeypatch.setattr(discovery, "_score_rna_genes", boom)
    res = _run(tmp_path, oracle=_rna_oracle(rna_world), write_tables=True, normalizer=_Norm())
    assert "no annotation" in res["tables"]["rna_skipped"]
    assert res["track_scores_csv"] and "rna_gene_percentiles_csv" not in res


def test_per_model_oracles_get_no_rna_tables(tmp_path):
    out = discovery._export_rna_tables("cherimoya", {"predictions": {}}, None, None, str(tmp_path), "s", "v", {})
    assert out == {}
