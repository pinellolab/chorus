"""The stamped ``build_config`` has to be read, or it is documentation not an invariant.

``build_and_save`` and ``append_tracks`` learned to write and preserve a file-level
``build_config`` (#124), and the three rebuilt backgrounds carry one. But nothing in
``chorus/`` read it: a grep for ``build_config`` found only the writer, the
append-path preservation, and docstrings. Provenance nobody consults cannot catch
anything.

What it should catch is #122, which is the reason the field exists. AlphaGenome
histone CHIP tracks had their null built over 501 bp while the query summed 2001 bp.
Both artefacts were internally consistent, so neither looked wrong; the defect only
existed *between* them, and it shipped across 1,075 of 5,168 tracks. A stamped build
geometry plus a load-time comparison makes that a detectable disagreement rather
than an invisible one.

The check warns rather than raises, and that is deliberate. Every background built
before the provenance work is unstamped, and an unstamped file makes no claim to
contradict. Refusing to load one would break working installs to enforce a metadata
convention.
"""
from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from chorus.analysis.normalization import PerTrackNormalizer

REPO = Path(__file__).resolve().parent.parent
BACKGROUNDS = Path.home() / ".chorus" / "backgrounds"
REF = REPO / "reference_sets" / "chorus_reference_positions_v1.npz"

# All eight, not the three that were easy. This tuple was ("alphagenome", "borzoi",
# "enformer") until 2026-08-09 -- exactly the three oracles for which the stamped activity
# population happened to be TRUE. The other five carried `regions_genome_dominated` and its
# sha256 because the stamper hardcoded it, and no test looked: chrombpnet, cherimoya and
# epinformerseq had offered their summary reservoir MORE samples per track than that
# 31,500-position set contains, and sei/legnet had drawn a strict subset of it. A guard
# parameterised over the subset that passes is not a guard.
ORACLES = ("alphagenome", "borzoi", "enformer", "chrombpnet", "cherimoya",
           "epinformerseq", "sei", "legnet")

# The activity population each builder samples, as a derivation of the one region set the
# reference artefact carries: strata dropped, strata added. Kept here independently of
# scripts/stamp_provenance_v4.py on purpose -- a test that imports the stamper's own table
# and compares it against the stamp checks nothing.
ACTIVITY_DERIVATION = {
    "alphagenome":   ((), ()),
    "borzoi":        ((), ()),
    "enformer":      ((), ()),
    "sei":           (("gene_body",), ()),
    "legnet":        (("gene_body",), ()),
    "chrombpnet":    (("gene_body",), ("dhs",)),
    "cherimoya":     (("gene_body",), ("dhs",)),
    "epinformerseq": (("gene_body",), ("dhs",)),
}

# Rows a builder offers per activity position per track where that is not 1: ChromBPNet
# scores both strands of its profile head; AlphaGenome and Borzoi emit one RNA summary row
# per gene in the window. See scripts/stamp_provenance_v4.py:FAN_OUT.
SUMMARY_FAN_OUT = {"chrombpnet": 2, "alphagenome": 16, "borzoi": 4}


@lru_cache(maxsize=1)
def _brps():
    """The reference-set generator, for its content-hash convention."""
    spec = importlib.util.spec_from_file_location(
        "brps", REPO / "scripts" / "build_reference_position_sets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def _region_rows():
    with np.load(REF, allow_pickle=False) as d:
        return tuple((str(c), int(p), str(s)) for c, p, s in d["regions_genome_dominated"])


def _require(oracle: str):
    if not (BACKGROUNDS / f"{oracle}_pertrack.npz").exists():
        pytest.skip(f"no downloaded background for {oracle}")
    if not REF.exists():
        pytest.skip("reference sets not generated in this checkout")


def _make_npz(tmp_path: Path, oracle: str, config: dict | None) -> Path:
    """A minimal but genuine one-track background, optionally stamped."""
    row = np.linspace(0.0, 1.0, 1_000)[None, :]
    return Path(PerTrackNormalizer.build_and_save(
        oracle_name=oracle,
        track_ids=["CHIP_HISTONE/EFO:1 Histone ChIP-seq H3K27ac/."],
        effect_cdfs=row,
        effect_counts=[1_000],
        cache_dir=str(tmp_path),
        n_points=1_000,
        provenance=config,
    ))


# ---------------------------------------------------------------------------
# The accessor
# ---------------------------------------------------------------------------


def test_provenance_round_trips_through_build_and_save(tmp_path):
    config = {"schema_version": 2, "genome": "hg38", "histone_window_bp": 2001,
              "other_window_bp": 501, "resolution": 1}
    _make_npz(tmp_path, "provtest", config)
    got = PerTrackNormalizer(cache_dir=str(tmp_path)).provenance("provtest")
    assert got == config


def test_provenance_is_none_for_an_unstamped_background(tmp_path):
    _make_npz(tmp_path, "provtest", None)
    assert PerTrackNormalizer(cache_dir=str(tmp_path)).provenance("provtest") is None


def test_unparseable_provenance_is_treated_as_absent(tmp_path, caplog):
    """A corrupt stamp must not take the whole background down with it."""
    path = _make_npz(tmp_path, "provtest", {"genome": "hg38"})
    with np.load(path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    payload["build_config"] = np.array(["{not json"])
    np.savez_compressed(path, **payload)
    with caplog.at_level("WARNING"):
        assert PerTrackNormalizer(cache_dir=str(tmp_path)).provenance("provtest") is None
    assert "will not parse" in caplog.text


# ---------------------------------------------------------------------------
# The check that makes it load-bearing
# ---------------------------------------------------------------------------


def test_a_stamped_window_mismatch_warns_loudly(tmp_path, caplog):
    """The #122 shape: built at 501 bp, queried at 2001 bp."""
    _make_npz(tmp_path, "provtest", {
        "schema_version": 2, "genome": "hg38",
        "histone_window_bp": 501,       # WRONG on purpose — the #122 defect
        "other_window_bp": 501, "resolution": 1,
    })
    with caplog.at_level("WARNING"):
        PerTrackNormalizer(cache_dir=str(tmp_path)).provenance("provtest")
    assert "was BUILT over 501 bp" in caplog.text
    assert "histone_marks" in caplog.text
    assert "#122" in caplog.text


def test_a_matching_stamp_is_silent(tmp_path, caplog):
    from chorus.analysis.scorers import LAYER_CONFIGS

    _make_npz(tmp_path, "provtest", {
        "schema_version": 2, "genome": "hg38",
        "histone_window_bp": LAYER_CONFIGS["histone_marks"].window_bp,
        "other_window_bp": LAYER_CONFIGS["tf_binding"].window_bp,
        "resolution": 1,
    })
    with caplog.at_level("WARNING"):
        PerTrackNormalizer(cache_dir=str(tmp_path)).provenance("provtest")
    assert "was BUILT over" not in caplog.text


def test_an_unstamped_background_does_not_warn(tmp_path, caplog):
    """Silence is the correct behaviour: it makes no claim to contradict."""
    _make_npz(tmp_path, "provtest", None)
    with caplog.at_level("WARNING"):
        PerTrackNormalizer(cache_dir=str(tmp_path)).provenance("provtest")
    assert "was BUILT over" not in caplog.text


# ---------------------------------------------------------------------------
# The shipped artefacts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("oracle", ORACLES)
def test_the_rebuilt_backgrounds_are_stamped_and_agree_with_the_query(oracle, caplog):
    if not (BACKGROUNDS / f"{oracle}_pertrack.npz").exists():
        pytest.skip(f"no downloaded background for {oracle}")
    from chorus.analysis.scorers import LAYER_CONFIGS

    norm = PerTrackNormalizer(cache_dir=str(BACKGROUNDS))
    with caplog.at_level("WARNING"):
        config = norm.provenance(oracle)
    assert isinstance(config, dict), f"{oracle} carries no build_config"
    assert config["genome"] == "hg38"
    assert config["histone_window_bp"] == LAYER_CONFIGS["histone_marks"].window_bp
    assert config["other_window_bp"] == LAYER_CONFIGS["tf_binding"].window_bp
    assert "was BUILT over" not in caplog.text, "shipped geometry disagrees"

    # The stamp must describe the file it is attached to, not a different build.
    assert config["n_tracks"] == norm.n_tracks(oracle)
    assert config["cdf_points"] == norm._ensure_loaded(oracle)["effect_cdfs"].shape[1]


@pytest.mark.parametrize("oracle", ORACLES)
def test_the_rebuilt_backgrounds_record_the_reference_population_they_used(oracle):
    """Which reference class this background ranks against, checkable from the file.

    Rewritten for schema 4. The previous version asserted
    ``effect_region_set_as_logged`` with hardcoded strata ({tss_near: 1200 ...
    random: 849}, n_positions 6,000) -- the 2026-08-05 build's numbers, scraped from the
    builder's stdout by a regex.

    Both halves were wrong to keep. The numbers pinned ONE build's reference class, so
    they went stale the moment the class changed (n 12,000 -> 18,000, cCRE added, DHS
    tried and rejected). And the field came from log scraping, which is how AlphaGenome
    ended up with a stamped *claim* and a scraped *measurement* contradicting each other,
    and why that scraper crashed on the 2026-08-06 logs when the message changed.

    Schema 4 records the reference sets by CONTENT HASH, so this compares the stamp
    against the artefact rather than against a constant, and stays true across rebuilds.
    """
    _require(oracle)
    config = PerTrackNormalizer(cache_dir=str(BACKGROUNDS)).provenance(oracle)
    rs = (config or {}).get("reference_sets")
    assert rs, f"{oracle}: schema {(config or {}).get('schema_version')} lacks reference_sets"

    with np.load(REF, allow_pickle=False) as d:
        prov = json.loads(str(d["provenance"][0]))

    fam = rs["effect_family"]
    assert f"snps_{fam}" in prov["sets"], (oracle, fam)
    assert rs["effect_sha256"] == prov["sets"][f"snps_{fam}"]["sha256"], (
        f"{oracle}: stamped effect population {rs['effect_sha256'][:12]} != the reference "
        f"artefact's {prov['sets'][f'snps_{fam}']['sha256'][:12]} -- built against a "
        f"different population than the one on disk"
    )
    assert rs["effect_strata"] == prov["sets"][f"snps_{fam}"]["strata_realised"]

    if fam == "gene_anchored":
        st = rs["effect_strata"]
        assert set(st) == {"tss_near", "tss_far", "junction", "gene_body",
                           "random", "ccre"}, st
        assert st["ccre"] == 9_000
        # The uniform stratum is load-bearing: without near-zero mass, small real effects
        # receive artificially LOW percentiles -- the mirror of saturation.
        assert st["random"] >= 1_200, st
        assert "dhs" not in st, "DHS was measured and rejected for this family"


@pytest.mark.parametrize("oracle", ORACLES)
def test_the_stamped_activity_population_is_the_one_the_builder_sampled(oracle):
    """The activity (summary/perbin) population, per oracle, not one claim for all eight.

    Until 2026-08-09 every stamp said ``regions_genome_dominated``, 31,500 positions, with
    that artefact's sha256 -- hardcoded, never measured. It was false for five oracles and
    could not be rebuilt from: sei and legnet never sample the ``gene_body`` stratum, and
    chrombpnet, cherimoya and epinformerseq add 5,000 DHS summits to the other three
    strata instead. The stamper now derives each population from the artefact and this
    asserts the derivation independently.
    """
    _require(oracle)
    drop, add = ACTIVITY_DERIVATION[oracle]
    rs = (PerTrackNormalizer(cache_dir=str(BACKGROUNDS)).provenance(oracle)
          or {}).get("reference_sets") or {}
    assert rs, f"{oracle} carries no reference_sets"

    kept = {s for _c, _p, s in _region_rows()} - set(drop)
    assert set(rs["activity_strata"]) == kept | set(add), (
        f"{oracle}: stamped activity strata {sorted(rs['activity_strata'])} are not what "
        f"its builder samples ({sorted(kept | set(add))})")

    with np.load(REF, allow_pickle=False) as d:
        prov = json.loads(str(d["provenance"][0]))
    genome_dominated = prov["sets"]["regions_genome_dominated"]["sha256"]
    if not drop and not add:
        assert rs["activity_sha256"] == genome_dominated, (
            f"{oracle} samples the whole reference region set but stamps "
            f"{rs['activity_sha256']}")
    else:
        assert rs["activity_sha256"] != genome_dominated, (
            f"{oracle} drops {drop} and adds {add}, so it cannot be ranking against "
            f"regions_genome_dominated -- yet that is the hash it stamps")

    if not add:      # derivable from the shipped artefact alone: check the hash exactly
        expect = _brps()._sha256_of([r for r in _region_rows() if r[2] not in drop])
        assert rs["activity_sha256"] == expect, (
            f"{oracle}: stamped {rs['activity_sha256'][:12]} != {expect[:12]} recomputed "
            f"from the artefact minus {drop}")


@pytest.mark.parametrize("oracle", ORACLES)
def test_the_activity_population_can_hold_the_samples_that_were_drawn_from_it(oracle):
    """The inequality that catches a misdeclared population from the artefact alone.

    A reservoir cannot be offered more samples than the build had positions to offer,
    times the rows the builder emits per position. Nothing here is tautological: the
    counts come from the NPZ's own arrays and the population size from the stamp.
    Against the pre-2026-08-09 stamps this fails on chrombpnet (68,008 > 31,500 x 2),
    cherimoya (34,004 > 31,500) and epinformerseq (34,002 > 31,500).
    """
    _require(oracle)
    norm = PerTrackNormalizer(cache_dir=str(BACKGROUNDS))
    rs = (norm.provenance(oracle) or {}).get("reference_sets") or {}
    assert rs, f"{oracle} carries no reference_sets"
    n_positions = sum(rs["activity_strata"].values())
    data = norm._ensure_loaded(oracle)

    for stat, per_position in (("summary", 1), ("perbin", 32)):
        key = f"{stat}_counts"
        if data.get(key) is None:      # sei and legnet ship no perbin layer at all
            continue
        ceiling = n_positions * per_position * SUMMARY_FAN_OUT.get(oracle, 1)
        got = int(np.asarray(data[key]).max())
        assert got <= ceiling, (
            f"{oracle}: {key}.max()={got:,} cannot come from {n_positions:,} positions "
            f"x {per_position} per position x {SUMMARY_FAN_OUT.get(oracle, 1)} fan-out "
            f"= {ceiling:,}")


def test_every_stamped_oracle_names_a_reference_population():
    """An oracle with no reference_sets must FAIL, not quietly leave the comparison.

    This used to filter with ``if c.get("reference_sets")`` and then compare whatever
    survived, so an unstamped -- or wrongly stamped and later unstamped -- background
    simply disappeared from the check it was supposed to be subject to.
    """
    present = [o for o in ORACLES if (BACKGROUNDS / f"{o}_pertrack.npz").exists()]
    if not present:
        pytest.skip("no downloaded backgrounds")
    nz = PerTrackNormalizer(cache_dir=str(BACKGROUNDS))
    missing = [o for o in present
               if not ((nz.provenance(o) or {}).get("reference_sets") or {}).get(
                   "activity_sha256")]
    assert not missing, f"downloaded but carry no hashed activity population: {missing}"


def test_oracles_sharing_a_builder_family_share_one_activity_population():
    """Comparable percentiles need the SAME population, not a similar one.

    Before the 2026-08 rebuild, effect_cdfs were on a newer null than summary_cdfs for six
    of eight oracles and nothing in any file said so. Note what this does NOT assert: that
    all eight share one activity population. They do not -- three builders sample the full
    region set, two drop the gene-body stratum and three swap it for DHS summits -- and the
    old "all 8 share one hash" assertion held only because the stamper wrote one hash into
    all eight regardless of what they had sampled.
    """
    present = [o for o in ORACLES if (BACKGROUNDS / f"{o}_pertrack.npz").exists()]
    if not present:
        pytest.skip("no downloaded backgrounds")
    nz = PerTrackNormalizer(cache_dir=str(BACKGROUNDS))
    cfgs = {o: (nz.provenance(o) or {})["reference_sets"] for o in present}

    by_family: dict = {}
    for o in present:
        by_family.setdefault(ACTIVITY_DERIVATION[o], []).append(o)
    for derivation, group in by_family.items():
        acts = {o: cfgs[o]["activity_sha256"] for o in group}
        assert len(set(acts.values())) == 1, (
            f"{derivation} builders must share one activity population: {acts}")
    # ...and the families must NOT collapse into each other, or the distinction is fiction.
    across = {cfgs[g[0]]["activity_sha256"] for g in by_family.values()}
    assert len(across) == len(by_family), f"distinct mixtures share a hash: {across}"

    gene = {o: cfgs[o]["effect_sha256"] for o in present
            if cfgs[o]["effect_family"] == "gene_anchored"}
    if len(gene) > 1:
        assert len(set(gene.values())) == 1, f"gene-anchored effect sets differ: {gene}"


# ---------------------------------------------------------------------------
# The stamper's own guard, without needing 2.5 GB of downloaded backgrounds
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stamper():
    spec = importlib.util.spec_from_file_location(
        "stamp_provenance_v4", REPO / "scripts" / "stamp_provenance_v4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_oracle_declares_the_activity_population_it_samples(stamper):
    """A new oracle must say what it sampled, not inherit somebody else's population."""
    assert set(stamper.ACTIVITY_POPULATIONS) == set(stamper.GEOMETRY), (
        "these two tables must cover the same oracles: "
        f"{set(stamper.GEOMETRY) ^ set(stamper.ACTIVITY_POPULATIONS)}")


@pytest.mark.parametrize("oracle,counts,fits", [
    ("cherimoya", 34_004, False),        # the 2026-08-09 defect, exactly
    ("cherimoya", 31_500, True),
    ("chrombpnet", 68_008, False),       # 2 strands, so the ceiling is 63,000
    ("chrombpnet", 63_000, True),
])
def test_the_write_time_check_refuses_a_population_too_small_to_hold_the_samples(
        stamper, oracle, counts, fits):
    payload = {"summary_counts": np.array([counts])}
    if fits:
        stamper.check_counts_fit_the_population(oracle, payload, 31_500)
        return
    with pytest.raises(ValueError, match="exceeds 31,500 activity positions"):
        stamper.check_counts_fit_the_population(oracle, payload, 31_500)


def test_the_write_time_check_is_applied_per_layer(stamper):
    """AlphaGenome's RNA rows fan out per gene; its other layers must stay tight.

    A single oracle-wide fan-out would have let any layer borrow the RNA multiplier and
    hidden a 16x population error in the 4,501 tracks that do not fan out at all.
    """
    payload = {
        "layers_per_row": np.array(["gene_expression", "tf_binding"]),
        "summary_counts": np.array([319_642, 31_005]),
    }
    stamper.check_counts_fit_the_population("alphagenome", payload, 31_500)

    payload["summary_counts"] = np.array([319_642, 40_000])   # tf_binding cannot fan out
    with pytest.raises(ValueError, match="tf_binding"):
        stamper.check_counts_fit_the_population("alphagenome", payload, 31_500)


def test_the_derivation_is_checked_against_the_artefacts_own_hash(stamper, monkeypatch):
    """If the hashing convention moves, every derived population hash is meaningless.

    So the stamper reproduces the FULL region set's recorded sha256 before publishing any
    hash derived from it, and refuses rather than stamping something unverifiable.
    """
    if not REF.exists():
        pytest.skip("reference sets not generated in this checkout")
    brps = _brps()
    with np.load(REF, allow_pickle=False) as d:
        prov = json.loads(str(d["provenance"][0]))
        arrays = {k: d[k] for k in d.files if k != "provenance"}

    got = stamper.activity_population("sei", brps, arrays, prov)
    assert got["activity_sha256"] and got["activity_derivation"]["drop_strata"] == ["gene_body"]

    monkeypatch.setattr(brps, "_sha256_of", lambda rows: "0" * 64)
    with pytest.raises(ValueError, match="hashing convention moved"):
        stamper.activity_population("sei", brps, arrays, prov)
