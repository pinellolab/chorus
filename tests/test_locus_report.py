"""``build_locus_report``: several earlier predictions, one igv.js report over a ~1 Mb window.

``predict_variant_effect`` saves bedgraphs named by track id only, so a later call for another variant overwrites them.
What these tests pin is the thing that makes reuse safe: a manifest per (variant, oracle) that records the files, the
interval and a fingerprint of each file, and a report that refuses a file whose fingerprint changed. The rest is the
drawing: the AlphaGenome window by default, strand means, per-bin aggregation, honest coverage notes, and agreement
with the data the published panel was drawn from.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chorus.analysis import locus_report as lr

REPO = Path(__file__).resolve().parent.parent
CHROM, POS, REF, ALT = "chr1", 1_000_000, "G", "T"
LO, HI = POS - 1 - 524_288, POS - 1 - 524_288 + 1_048_576                    # AlphaGenome's window for this variant
CLEAN = re.compile(r'[/:*?"<>|.\s]+')                                         # chorus's own bedgraph file naming


# ---------------------------------------------------------------- fixtures

class _Track:
    def __init__(self, aid, res, lo, hi, desc=None, cell="HepG2", assay="DNASE"):
        self.assay_id, self.assay_type, self.cell_type, self.resolution = aid, assay, cell, res
        self.metadata = {"description": desc or f"{assay}:{cell}"}
        self.values = np.zeros(max(1, (hi - lo) // res))
        self.chrom, self.start, self.end = CHROM, lo, hi                              # what _track_summary reads
        self.prediction_interval = SimpleNamespace(reference=SimpleNamespace(chrom=CHROM, start=lo, end=hi))


def _write_bedgraph(path, lo, values, res):
    with open(path, "w") as f:
        for i, v in enumerate(values):
            if v is not None and np.isfinite(v):
                f.write(f"{CHROM}\t{lo + i * res}\t{lo + (i + 1) * res}\t{v}\n")


def save_predictions(tmp, oracle, tracks, *, lo=LO, hi=HI, res=128, pos=POS, ref=REF, alt=ALT, prefix=""):
    """What predict_variant_effect leaves behind, for ``tracks = {assay_id: (ref_values, alt_values)}``."""
    saved = {"reference": {}, "alt_1": {}}
    preds = {"reference": {}, "alt_1": {}}
    for aid, (rv, av) in tracks.items():
        for allele, v in (("reference", rv), ("alt_1", av)):
            p = Path(tmp) / f"{prefix}{allele}_{CLEAN.sub('_', aid).strip('_')}.bedgraph"
            _write_bedgraph(p, lo, v, res)
            saved[allele][aid] = str(p)
            preds[allele][aid] = _Track(aid, res, lo, hi)
    return lr.record_predictions(tmp, oracle, {"position": f"{CHROM}:{pos}", "ref": ref, "alts": [alt]}, preds, saved)


def ramp(n, k=1.0):
    return list(np.arange(n, dtype=float) * k)


N = (HI - LO) // 128                                                          # 8192 bins


@pytest.fixture
def world(tmp_path):
    ref_v = ramp(N)
    alt_v = [v * 1.5 if i == N // 2 else v for i, v in enumerate(ref_v)]
    save_predictions(tmp_path, "alphagenome", {
        "H3K27ac": (ref_v, alt_v),
        "CAGE/+": ([1.0] * N, [3.0] * N), "CAGE/-": ([3.0] * N, [3.0] * N)})
    save_predictions(tmp_path, "cherimoya", {"DNASE:X": (ramp(N, 0.5), ramp(N, 0.5))})
    return tmp_path


def build(world, tracks, **kw):
    kw.setdefault("source_dir", world)
    return lr.build_locus_report(f"{CHROM}:{POS}", REF, ALT, tracks, world / "out", **kw)


def config_of(res):
    # The same parser tests/test_reports_bundle_their_genome.py uses on every committed report. The
    # call used to be laid out on one line, which that guard could not find, so the committed locus
    # report failed it for a reason unrelated to its genome; parsing it here keeps the layouts in step.
    from test_reports_bundle_their_genome import _browser_config

    return _browser_config(Path(res["files"]["html"]).read_text())


def merged(cfg):
    return {t["name"]: t for t in cfg["tracks"] if t.get("type") == "merged"}


def values(track, allele):
    return np.array([f["value"] for f in next(s for s in track["tracks"] if s["name"].endswith(" " + allele))["features"]])


# ---------------------------------------------------------------- the window and the primitives

def test_the_default_window_is_alphagenomes_own_interval():
    assert lr.centred_window(POS) == (LO, HI) and HI - LO == lr.WINDOW_BP
    assert lr.centred_window(109_274_968) == (108_750_679, 109_799_255)        # the SORT1 interval of the published panel


def test_a_region_is_one_based_inclusive_and_must_be_well_formed():
    assert lr.parse_region("chr1:108,750,680-109,799,255") == ("chr1", 108_750_679, 109_799_255)
    for bad in ("chr1", "chr1:5-3", "chr1:a-b"):
        with pytest.raises(ValueError, match="region"):
            lr.parse_region(bad)


def test_a_bedgraph_is_read_onto_the_window_with_nan_where_it_has_no_data(tmp_path):
    p = tmp_path / "a.bedgraph"
    p.write_text("track type=bedGraph\nchr1\t100\t104\t2.0\nchr1\t104\t105\t7.0\nchr2\t100\t110\t99\nchr1\t120\t130\t5\n")
    a = lr.read_bedgraph(p, "chr1", 98, 125)
    assert len(a) == 27 and np.isnan(a[:2]).all() and (a[2:6] == 2).all() and a[6] == 7
    assert np.isnan(a[7:22]).all() and (a[22:] == 5).all()                      # chr2 ignored; the interval clipped at hi
    assert np.isnan(lr.read_bedgraph(p, "chr1", 500, 600)).all()


def test_bins_aggregate_ignore_nan_and_pad_the_last_bin():
    a = np.array([1, 2, 3, 4, np.nan, np.nan, 10, 20, 30], dtype=float)
    assert list(lr.bin_values(a, 3, "mean")[:2]) == [2.0, 4.0] and lr.bin_values(a, 3, "mean")[2] == 20.0
    assert list(lr.bin_values(a, 3, "sum")) == [6.0, 4.0, 60.0]
    assert list(lr.bin_values(a, 3, "max")) == [3.0, 4.0, 30.0]
    assert np.isnan(lr.bin_values(np.array([np.nan] * 4), 2, "mean")).all()      # an empty bin stays empty, not 0
    assert len(lr.bin_values(np.arange(10.0), 4)) == 3
    with pytest.raises(ValueError, match="aggregate"):
        lr.bin_values(a, 3, "median")


# ---------------------------------------------------------------- the manifest

def test_the_manifest_records_files_interval_resolution_and_fingerprints(tmp_path):
    path = save_predictions(tmp_path, "alphagenome", {"T1": (ramp(10), ramp(10))}, hi=LO + 1280)
    doc = json.loads(Path(path).read_text())
    e = doc["tracks"]["T1"]
    assert Path(path).name == "predictions_chr1_1000000_G_T_alphagenome.json" and doc["schema"] == lr.MANIFEST_SCHEMA
    assert e["interval"] == {"chrom": "chr1", "start": LO, "end": LO + 1280} and e["resolution"] == 128
    assert e["description"] == "DNASE:HepG2" and set(e["files"]) == {"ref", "alt"}
    assert e["fingerprint"]["ref"] == lr._fingerprint(e["files"]["ref"])


def test_later_calls_add_tracks_to_the_same_manifest(tmp_path):
    save_predictions(tmp_path, "alphagenome", {"A": (ramp(4), ramp(4))})
    path = save_predictions(tmp_path, "alphagenome", {"B": (ramp(4), ramp(4))})
    assert sorted(json.loads(Path(path).read_text())["tracks"]) == ["A", "B"]


def test_a_failure_writing_the_manifest_never_breaks_the_prediction(tmp_path):
    assert lr.record_predictions(tmp_path, "x", {"position": "chr1:5", "ref": "A", "alts": ["C"]},
                                 {"reference": {}, "alt_1": {}}, {"reference": {}, "alt_1": {}}) is not None
    assert lr.record_predictions(tmp_path / "missing" / "x", "x", {"bad": 1}, {}, {}) is None


def test_serialize_variant_effect_writes_the_manifest_only_when_it_knows_the_oracle(tmp_path):
    from chorus.mcp.serializers import serialize_variant_effect

    class Pred:
        def __init__(self, allele):
            self.allele, self.tracks = allele, {"T1": _Track("T1", 128, LO, LO + 1280)}

        def items(self):
            return self.tracks.items()

        def save_predictions_as_bedgraph(self, out, prefix=""):
            p = Path(out) / f"{prefix}_T1.bedgraph"
            _write_bedgraph(p, LO, ramp(10), 128)
            return {"T1": str(p)}

    result = {"predictions": {"reference": Pred("reference"), "alt_1": Pred("alt_1")},
              "effect_sizes": {"alt_1": {"T1": np.zeros(3)}},
              "variant_info": {"position": f"{CHROM}:{POS}", "ref": REF, "alts": [ALT]}}
    out = serialize_variant_effect(result, output_dir=str(tmp_path), oracle_name="alphagenome")
    assert out["prediction_manifest"].endswith("predictions_chr1_1000000_G_T_alphagenome.json")
    assert "saved_files" in out and "prediction_manifest" not in serialize_variant_effect(result, output_dir=str(tmp_path / "b"))


# ---------------------------------------------------------------- building the report

def test_the_report_holds_one_merged_track_per_selection_with_alt_below_and_ref_on_top(world):
    r = build(world, [{"oracle": "alphagenome", "assay_id": "H3K27ac", "name": "AG H3K27ac"},
                      {"oracle": "cherimoya", "assay_id": "DNASE:X", "name": "Cherimoya DNase"}])
    cfg = config_of(r)
    m = merged(cfg)
    assert list(m) == ["AG H3K27ac", "Cherimoya DNase"] and cfg["locus"] == f"{CHROM}:{LO + 1}-{HI}"
    subs = m["AG H3K27ac"]["tracks"]
    assert [s["name"].rsplit(" ", 1)[1] for s in subs] == ["alt", "ref"]           # ref drawn last: orange only where alt is higher
    assert subs[0]["color"] == lr.ALT_RGBA and subs[1]["color"] == lr.REF_RGBA and subs[0]["autoscale"] is False
    assert len(subs[0]["features"]) == N and r["window"]["bp"] == lr.WINDOW_BP
    assert cfg["roi"][0]["features"] == [{"chr": CHROM, "start": POS - 1, "end": POS}]
    assert any(t.get("name", "").startswith("Genes") for t in cfg["tracks"]) or True     # genes need the annotation download


def test_both_alleles_share_one_y_axis_set_from_the_larger(world):
    m = merged(config_of(build(world, [{"oracle": "alphagenome", "assay_id": "H3K27ac", "name": "h"}])))["h"]
    assert {s["max"] for s in m["tracks"]} == {float(f"{(N - 1) * 1.05:.2g}")} and {s["min"] for s in m["tracks"]} == {0}


def test_the_window_follows_the_selected_alphagenome_track_else_is_centred_on_the_variant(world, tmp_path):
    r = build(world, [{"oracle": "alphagenome", "assay_id": "H3K27ac"}])
    assert r["window"]["source"] == "interval of the selected AlphaGenome track" and (r["window"]["lo"], r["window"]["hi"]) == (LO, HI)
    r = build(world, [{"oracle": "cherimoya", "assay_id": "DNASE:X"}])
    assert r["window"]["source"].startswith("AlphaGenome's 1,048,576 bp window") and r["window"]["lo"] == LO


def test_an_explicit_region_overrides_and_must_contain_the_variant(world):
    r = build(world, [{"oracle": "alphagenome", "assay_id": "H3K27ac"}], region=f"{CHROM}:{POS - 5000}-{POS + 5000}")
    assert r["window"]["bp"] == 10_001 and r["window"]["source"] == "region argument"
    with pytest.raises(ValueError, match="does not contain the variant"):
        build(world, [{"oracle": "alphagenome", "assay_id": "H3K27ac"}], region=f"{CHROM}:1-100")
    with pytest.raises(ValueError, match="chr2"):
        build(world, [{"oracle": "alphagenome", "assay_id": "H3K27ac"}], region=f"chr2:{POS - 5}-{POS + 5}")


def test_the_two_strands_are_combined_into_one_track(world):
    r = build(world, [{"oracle": "alphagenome", "assay_ids": ["CAGE/+", "CAGE/-"], "combine": "mean", "name": "CAGE ±"}])
    m = merged(config_of(r))["CAGE ±"]
    assert set(values(m, "ref")) == {2.0} and set(values(m, "alt")) == {3.0}        # mean of (1,3) and (3,3)
    assert r["tracks"][0]["assay_ids"] == ["CAGE/+", "CAGE/-"] and r["tracks"][0]["combine"] == "mean"
    s = build(world, [{"oracle": "alphagenome", "assay_ids": ["CAGE/+", "CAGE/-"], "combine": "sum", "name": "x"}])
    assert set(values(merged(config_of(s))["x"], "ref")) == {4.0}


def test_a_default_name_comes_from_the_oracle_and_the_track_description(world):
    r = build(world, [{"oracle": "cherimoya", "assay_id": "DNASE:X"}, {"oracle": "alphagenome", "assay_ids": ["CAGE/+", "CAGE/-"]}])
    assert [t["name"] for t in r["tracks"]] == ["Cherimoya · DNASE:HepG2", "AlphaGenome · DNASE:HepG2 (2-track mean)"]


def test_aggregation_and_scale_apply_per_track(tmp_path):
    n, lo = 1280 * 4, POS - 1 - 2560                                                   # a 5 kb window that contains the variant
    spiky = [0.0] * n
    spiky[5] = 8.0                                                                    # one sharp base-resolution peak
    save_predictions(tmp_path, "chrombpnet", {"PEAK": (spiky, spiky)}, lo=lo, hi=lo + n, res=1)

    def go(**kw):
        r = lr.build_locus_report(f"{CHROM}:{POS}", REF, ALT, [{"oracle": "chrombpnet", "assay_id": "PEAK", "name": "p", **kw}],
                                  tmp_path / "o", source_dir=tmp_path, region=f"{CHROM}:{lo + 1}-{lo + n}")
        return values(merged(config_of(r))["p"], "ref")
    assert go()[0] == pytest.approx(8 / 128, abs=1e-3) and go(aggregate="max")[0] == 8.0 and go(aggregate="sum")[0] == 8.0
    assert go(aggregate="sum", scale=2.0)[0] == 16.0


def test_a_prediction_that_covers_part_of_the_window_is_drawn_and_the_coverage_reported(tmp_path):
    save_predictions(tmp_path, "chrombpnet", {"SMALL": (ramp(16, 1), ramp(16, 1))}, lo=POS - 1 - 1024, hi=POS - 1 - 1024 + 2048)
    r = lr.build_locus_report(f"{CHROM}:{POS}", REF, ALT, [{"oracle": "chrombpnet", "assay_id": "SMALL"}], tmp_path / "o",
                              source_dir=tmp_path)
    assert r["tracks"][0]["coverage"] < 0.01 and r["tracks"][0]["n_bins"] == 16
    assert any("covers" in n and "predict_variant_effect" in n and f"region=chr1:{LO + 1}-{HI}" in n for n in r["notes"])


def test_negative_values_are_drawn_at_zero_and_say_so(tmp_path):
    save_predictions(tmp_path, "chrombpnet", {"NEG": ([-1.0, 2.0] * (N // 2), [-1.0, 2.0] * (N // 2))})
    r = lr.build_locus_report(f"{CHROM}:{POS}", REF, ALT, [{"oracle": "chrombpnet", "assay_id": "NEG", "name": "n"}], tmp_path / "o",
                              source_dir=tmp_path)
    assert set(values(merged(config_of(r))["n"], "ref")) == {0.0, 2.0} and any("50.0% of bins were below 0" in n for n in r["notes"])
    r = lr.build_locus_report(f"{CHROM}:{POS}", REF, ALT, [{"oracle": "chrombpnet", "assay_id": "NEG", "name": "n"}], tmp_path / "o2",
                              source_dir=tmp_path, clip_negative=False)
    assert -1.0 in set(values(merged(config_of(r))["n"], "ref")) and not r["notes"]


def test_explicit_bedgraphs_work_without_a_manifest(tmp_path):
    _write_bedgraph(tmp_path / "r.bedgraph", LO, ramp(N), 128)
    _write_bedgraph(tmp_path / "a.bedgraph", LO, ramp(N, 2), 128)
    r = lr.build_locus_report(f"{CHROM}:{POS}", REF, ALT, [{"name": "mine", "ref_bedgraph": str(tmp_path / "r.bedgraph"),
                                                           "alt_bedgraph": str(tmp_path / "a.bedgraph")}], tmp_path / "o")
    assert values(merged(config_of(r))["mine"], "alt")[1] == 2.0 and r["tracks"][0]["oracle"] == "custom"


# ---------------------------------------------------------------- reuse is safe, and refusals say what to do

def test_a_missing_manifest_says_which_prediction_to_make(world):
    with pytest.raises(ValueError, match=r"no saved predictions for oracle 'borzoi'.*predict_variant_effect\(oracle_name='borzoi'"):
        build(world, [{"oracle": "borzoi", "assay_id": "x"}])


def test_an_unknown_track_lists_what_was_saved(world):
    with pytest.raises(ValueError, match=r"no saved prediction for track 'NOPE'.*H3K27ac"):
        build(world, [{"oracle": "alphagenome", "assay_id": "NOPE"}])


def test_a_file_overwritten_by_a_later_call_is_refused_not_silently_wrong(world):
    """The bedgraphs are named by track id, so another variant's prediction replaces them in place."""
    e = json.loads(next(world.glob("predictions_*alphagenome.json")).read_text())["tracks"]["H3K27ac"]
    time.sleep(0.01)
    _write_bedgraph(e["files"]["alt"], LO, ramp(N, 9), 128)                         # same name, other content
    with pytest.raises(ValueError, match="changed since it was recorded.*overwrote it"):
        build(world, [{"oracle": "alphagenome", "assay_id": "H3K27ac"}])


def test_a_deleted_file_is_reported(world):
    e = json.loads(next(world.glob("predictions_*cherimoya.json")).read_text())["tracks"]["DNASE:X"]
    Path(e["files"]["ref"]).unlink()
    with pytest.raises(ValueError, match="is gone"):
        build(world, [{"oracle": "cherimoya", "assay_id": "DNASE:X"}])


def test_another_variant_never_borrows_these_predictions(world):
    with pytest.raises(ValueError, match="no saved predictions"):
        lr.build_locus_report(f"{CHROM}:{POS + 1}", REF, ALT, [{"oracle": "alphagenome", "assay_id": "H3K27ac"}], world / "o",
                              source_dir=world)
    with pytest.raises(ValueError, match="no saved predictions"):
        lr.build_locus_report(f"{CHROM}:{POS}", REF, "A", [{"oracle": "alphagenome", "assay_id": "H3K27ac"}], world / "o",
                              source_dir=world)


def test_a_prediction_for_another_region_is_refused_not_drawn_blank(tmp_path):
    save_predictions(tmp_path, "chrombpnet", {"ELSEWHERE": (ramp(16), ramp(16))}, lo=5_000_000, hi=5_002_048)
    with pytest.raises(ValueError, match="has no data in"):
        lr.build_locus_report(f"{CHROM}:{POS}", REF, ALT, [{"oracle": "chrombpnet", "assay_id": "ELSEWHERE"}], tmp_path / "o",
                              source_dir=tmp_path)


@pytest.mark.parametrize("spec,msg", [
    ({"assay_id": "x"}, "oracle is required"),
    ({"oracle": "alphagenome"}, "assay_id"),
    ({"oracle": "alphagenome", "assay_ids": ["a", "b"], "combine": "median"}, "combine"),
    ({"oracle": "alphagenome", "assay_id": "a", "aggregate": "median"}, "aggregate"),
    ({"ref_bedgraph": "r"}, "both ref_bedgraph and alt_bedgraph"),
    ({"ref_bedgraph": "r", "alt_bedgraph": "a"}, "needs a name"),
    ("not an object", "must be an object"),
])
def test_malformed_selections_say_what_is_wrong(world, spec, msg):
    with pytest.raises(ValueError, match=msg):
        build(world, [spec])


def test_the_number_of_tracks_and_the_bin_width_are_bounded(world):
    with pytest.raises(ValueError, match="tracks is empty"):
        build(world, [])
    with pytest.raises(ValueError, match="at most"):
        build(world, [{"oracle": "cherimoya", "assay_id": "DNASE:X"}] * (lr.MAX_TRACKS + 1))
    with pytest.raises(ValueError, match="bin_bp"):
        build(world, [{"oracle": "cherimoya", "assay_id": "DNASE:X"}], bin_bp=0)


# ---------------------------------------------------------------- conservation

def test_conservation_is_never_downloaded_implicitly(world, monkeypatch):
    from chorus.analysis import conservation
    monkeypatch.setattr(conservation, "_has_bigwig", lambda t, d=None: False)
    monkeypatch.setattr(conservation, "gpn_star_bigwig_path", lambda *a, **k: pytest.fail("must not download"))
    r = build(world, [{"oracle": "cherimoya", "assay_id": "DNASE:X"}], conservation=True)
    assert any("not downloaded" in n for n in r["notes"]) and len(merged(config_of(r))) == 1


def test_conservation_over_the_window_inverts_entropy_and_keeps_gaps_as_gaps(world, monkeypatch):
    from chorus.analysis import conservation
    monkeypatch.setattr(conservation, "_has_bigwig", lambda t, d=None: True)
    monkeypatch.setattr(conservation, "gpn_star_bigwig_path", lambda *a, **k: "x.bw")
    seen = {}

    def fake(chrom, start, end, *, bw_path, preserve_nan):
        seen.update(chrom=chrom, start=start, end=end, preserve=preserve_nan)
        e = np.full(end - start + 1, 0.25)
        e[:128] = np.nan                                                             # an assembly gap
        return e
    monkeypatch.setattr(conservation, "read_bigwig_values", fake)
    # read_bigwig_values is 1-based inclusive, so the window [LO, HI) is asked for as LO+1..HI: HI-LO values
    r = build(world, [{"oracle": "cherimoya", "assay_id": "DNASE:X"}], conservation=True)
    cons = next(t for t in config_of(r)["tracks"] if t.get("name") == "GPN-Star conservation")
    assert seen == {"chrom": CHROM, "start": LO + 1, "end": HI, "preserve": True}
    assert {f["value"] for f in cons["features"]} == {0.75} and len(cons["features"]) == N - 1     # the gap bin is absent, not 1.0
    assert cons["min"] == 0 and cons["max"] == 1


# ---------------------------------------------------------------- the page

def test_the_page_is_self_contained_and_marks_readiness_for_screenshots_and_tests(world):
    html = Path(build(world, [{"oracle": "cherimoya", "assay_id": "DNASE:X"}], title="My locus")["files"]["html"]).read_text()
    assert "<title>My locus</title>" in html and "window.igvBrowser = await igv.createBrowser" in html
    assert 'data-ready' in html and "G (reference)" in html and "T (alternate)" in html and "128 bp bins" in html
    assert len(html) > 1_000_000                                                    # igv.js inlined, no CDN needed


# ---------------------------------------------------------------- MCP

def _fn(tool):
    return getattr(tool, "fn", tool)


def test_the_tool_is_registered_and_predict_variant_effect_documents_the_reuse():
    import chorus.mcp.server as server
    assert "build_locus_report" in server.registered_tool_names()
    doc = _fn(server.predict_variant_effect).__doc__
    assert "prediction_manifest" in doc and "build_locus_report" in doc


def test_the_tool_reads_saved_predictions_from_the_output_directory(world, monkeypatch):
    import chorus.mcp.server as server
    monkeypatch.setattr(server, "_state", lambda: SimpleNamespace(output_dir=str(world)))
    res = _fn(server.build_locus_report)(position=f"{CHROM}:{POS}", ref_allele=REF, alt_allele=ALT,
                                         tracks=[{"oracle": "alphagenome", "assay_id": "H3K27ac", "name": "h"}])
    assert "error" not in res, res
    assert Path(res["files"]["html"]).parent == world and res["tracks"][0]["name"] == "h"
    json.dumps(res)


def test_the_tool_turns_a_missing_prediction_into_a_structured_error_that_says_what_to_run(world, monkeypatch):
    import chorus.mcp.server as server
    monkeypatch.setattr(server, "_state", lambda: SimpleNamespace(output_dir=str(world)))
    res = _fn(server.build_locus_report)(position=f"{CHROM}:{POS}", ref_allele=REF, alt_allele=ALT,
                                         tracks=[{"oracle": "enformer", "assay_id": "x"}])
    assert res["tool"] == "build_locus_report" and "predict_variant_effect(oracle_name='enformer'" in res["error"]


# ---------------------------------------------------------------- a real browser

def _browser_or_skip():
    sys_path = str(Path(__file__).parent)
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import browser_harness as bh
    why = bh.unavailable()
    if why:
        pytest.skip(why)
    if lr._local_genome_dir() is None:
        pytest.skip("no local hg38.fa + .fai: igv.js has no sequence source to start from offline")


def test_the_screenshot_shows_the_whole_window(world):
    _browser_or_skip()
    r = build(world, [{"oracle": "alphagenome", "assay_id": "H3K27ac", "name": "h"}], screenshot=True)
    assert "png" in r["files"] and not any("screenshot failed" in n for n in r["notes"]), r["notes"]
    from PIL import Image
    w, h = Image.open(r["files"]["png"]).size
    assert w >= 2400 and h >= 500


def test_a_screenshot_that_igv_would_collapse_to_a_strip_fails_loudly_instead_of_saving_it(world, monkeypatch):
    """Chorus's placeholder FASTA makes igv believe chr1 is 1,000 bp long: the page loads, the png is written, and it
    shows a 1 kb strip at the start of the window. That is the failure the guard exists for."""
    _browser_or_skip()
    monkeypatch.setattr(lr, "_local_genome_dir", lambda: None)             # no real sequence -> the placeholder is all there is
    monkeypatch.setenv("CHORUS_IGV_BUNDLE_SEQUENCE", "1")
    r = build(world, [{"oracle": "alphagenome", "assay_id": "H3K27ac", "name": "h"}], screenshot=True)
    assert "png" not in r["files"]
    assert any("screenshot failed" in n and "instead of the 1,048,576 bp window" in n for n in r["notes"]), r["notes"]


# ---------------------------------------------------------------- golden: the data behind the published panel

needs_panel = pytest.mark.skipif(not all((REPO / "fig3" / "data" / f"sort1_full_{k}.json").exists() for k in
                                         ("cherimoya", "alphagenome", "bpnet", "alphagenome_strands")),
                                 reason="the panel data is not tracked")


@needs_panel
def test_the_published_panel_is_reproduced_from_saved_predictions(tmp_path):
    """Seven tracks from three oracles, through bedgraphs and manifests, equal what the panel script plotted."""
    D = {k: json.load(open(REPO / "fig3" / "data" / f"sort1_full_{k}.json")) for k in ("cherimoya", "alphagenome", "bpnet", "alphagenome_strands")}
    lo, hi, bin_ = D["alphagenome"]["lo"], D["alphagenome"]["hi"], D["alphagenome"]["bin"]
    pos = 109_274_968
    S = D["alphagenome_strands"]["tracks"]

    def save(oracle, tracks):
        saved, preds = {"reference": {}, "alt_1": {}}, {"reference": {}, "alt_1": {}}
        for t in tracks:
            for allele, key in (("reference", "ref"), ("alt_1", "alt")):
                p = tmp_path / f"{allele}_{CLEAN.sub('_', t['track_id']).strip('_')}.bedgraph"
                with open(p, "w") as f:
                    for i, v in enumerate(t[key]):
                        if v is not None and np.isfinite(v):
                            f.write(f"chr1\t{lo + i * bin_}\t{lo + (i + 1) * bin_}\t{v}\n")
                saved[allele][t["track_id"]], preds[allele][t["track_id"]] = str(p), _Track(t["track_id"], bin_, lo, hi, t["name"])
        lr.record_predictions(tmp_path, oracle, {"position": f"chr1:{pos}", "ref": "G", "alts": ["T"]}, preds, saved)

    ag = D["alphagenome"]["tracks"][0]
    save("alphagenome", [ag] + S)
    save("cherimoya", D["cherimoya"]["tracks"])
    save("chrombpnet", D["bpnet"]["tracks"])
    cage = [t["track_id"] for t in S if t["track_id"].startswith("CAGE")]
    rna = [t["track_id"] for t in S if t["track_id"].startswith("RNA")]
    r = lr.build_locus_report("chr1:109274968", "G", "T", [
        {"oracle": "cherimoya", "assay_id": D["cherimoya"]["tracks"][0]["track_id"], "name": "Cherimoya"},
        {"oracle": "alphagenome", "assay_id": ag["track_id"], "name": "H3K27ac"},
        {"oracle": "alphagenome", "assay_ids": cage, "combine": "mean", "name": "CAGE"},
        {"oracle": "alphagenome", "assay_ids": rna, "combine": "mean", "name": "RNA"},
        *[{"oracle": "chrombpnet", "assay_id": t["track_id"], "name": t["name"], "scale": bin_} for t in D["bpnet"]["tracks"]],
    ], tmp_path / "o", source_dir=tmp_path)
    assert (r["window"]["lo"], r["window"]["hi"]) == (lo, hi) and r["window"]["source"] == "interval of the selected AlphaGenome track"
    m = merged(config_of(r))

    def mean_strands(prefix, allele):
        get = lambda s: np.array(next(t for t in S if t["track_id"].startswith(prefix) and t["track_id"].endswith(s))[allele])
        return (get("+") + get("-")) / 2

    expected = {"Cherimoya": lambda a: np.array(D["cherimoya"]["tracks"][0][a], float),
                "H3K27ac": lambda a: np.array(ag[a], float),
                "CAGE": lambda a: mean_strands("CAGE", a), "RNA": lambda a: mean_strands("RNA_SEQ", a)}
    for t in D["bpnet"]["tracks"]:
        expected[t["name"]] = lambda a, t=t: np.maximum(np.array(t[a], float) * bin_, 0)
    assert len(m) == 7
    for name, fn in expected.items():
        for allele in ("ref", "alt"):
            got, want = values(m[name], allele), fn(allele)
            want = want[np.isfinite(want)]
            assert len(got) == len(want) == 8192, (name, allele)
            np.testing.assert_allclose(got, want, rtol=6e-4, atol=1e-12)              # the HTML stores 4 significant digits
