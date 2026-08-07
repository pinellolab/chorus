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

import json
from pathlib import Path

import numpy as np
import pytest

from chorus.analysis.normalization import PerTrackNormalizer

BACKGROUNDS = Path.home() / ".chorus" / "backgrounds"
REBUILT = ("alphagenome", "borzoi", "enformer")


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


@pytest.mark.parametrize("oracle", REBUILT)
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


@pytest.mark.parametrize("oracle", REBUILT)
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
    import json

    import numpy as np

    if not (BACKGROUNDS / f"{oracle}_pertrack.npz").exists():
        pytest.skip(f"no downloaded background for {oracle}")
    ref = (Path(__file__).resolve().parent.parent / "reference_sets"
           / "chorus_reference_positions_v1.npz")
    if not ref.exists():
        pytest.skip("reference sets not generated in this checkout")

    config = PerTrackNormalizer(cache_dir=str(BACKGROUNDS)).provenance(oracle)
    rs = (config or {}).get("reference_sets")
    assert rs, f"{oracle}: schema {(config or {}).get('schema_version')} lacks reference_sets"

    with np.load(ref, allow_pickle=False) as d:
        prov = json.loads(str(d["provenance"][0]))

    fam = rs["effect_family"]
    assert f"snps_{fam}" in prov["sets"], (oracle, fam)
    assert rs["effect_sha256"] == prov["sets"][f"snps_{fam}"]["sha256"], (
        f"{oracle}: stamped effect population {rs['effect_sha256'][:12]} != the reference "
        f"artefact's {prov['sets'][f'snps_{fam}']['sha256'][:12]} -- built against a "
        f"different population than the one on disk"
    )
    assert rs["effect_strata"] == prov["sets"][f"snps_{fam}"]["strata_realised"]
    assert rs["activity_sha256"] == prov["sets"]["regions_genome_dominated"]["sha256"]

    if fam == "gene_anchored":
        st = rs["effect_strata"]
        assert set(st) == {"tss_near", "tss_far", "junction", "gene_body",
                           "random", "ccre"}, st
        assert st["ccre"] == 9_000
        # The uniform stratum is load-bearing: without near-zero mass, small real effects
        # receive artificially LOW percentiles -- the mirror of saturation.
        assert st["random"] >= 1_200, st
        assert "dhs" not in st, "DHS was measured and rejected for this family"


@pytest.mark.parametrize("oracle", REBUILT)
def test_all_rebuilt_backgrounds_share_one_reference_population(oracle):
    """Comparable percentiles need the SAME population, not a similar one.

    Before the 2026-08 rebuild, effect_cdfs were on a newer null than summary_cdfs for six
    of eight oracles and nothing in any file said so.
    """
    if not (BACKGROUNDS / f"{oracle}_pertrack.npz").exists():
        pytest.skip(f"no downloaded background for {oracle}")
    nz = PerTrackNormalizer(cache_dir=str(BACKGROUNDS))
    cfgs = {}
    for o in REBUILT:
        if (BACKGROUNDS / f"{o}_pertrack.npz").exists():
            c = nz.provenance(o) or {}
            if c.get("reference_sets"):
                cfgs[o] = c["reference_sets"]
    assert cfgs, "no stamped backgrounds"
    acts = {o: r["activity_sha256"] for o, r in cfgs.items()}
    assert len(set(acts.values())) == 1, f"activity populations differ: {acts}"
    gene = {o: r["effect_sha256"] for o, r in cfgs.items()
            if r["effect_family"] == "gene_anchored"}
    if len(gene) > 1:
        assert len(set(gene.values())) == 1, f"gene-anchored effect sets differ: {gene}"
