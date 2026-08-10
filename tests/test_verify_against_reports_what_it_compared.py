"""`--verify-against` must claim only what it measured.

It was a CARDINALITY check that reported itself as a content check. A hand-written NPZ
carrying five fake track ids, all-zero `effect_cdfs`, no `build_config` and
`effect_counts == 17909` was told

    VERIFIED: the reference set reproduces sei's sampled population

and exited 0 -- because the only comparison that ran was `len(snps)` against
`effect_counts.min()`. The one branch that could see composition,
`build_config["effect_region_set_as_logged"]["strata_counts"]`, fires only if that key
exists, and schema 4 replaced `build_config` wholesale, so it exists in NONE of the eight
shipped backgrounds: it is dead code on every file anyone would run this against.

That matters because the result was being quoted as proof of population identity, and a
percentile has no meaning apart from the population it ranks against (§2 of the protocol).
These tests pin the three things that make the output honest:

  * the artefact's content hash is RECOMPUTED from its arrays, so it is a content check
    rather than a comparison between two copies of the same string;
  * a file with no composition evidence gets an explicit "SKIPPED -- counts only" line and
    a closing line that says population identity was not checked;
  * `--strict` refuses such a file.

Note what the stamp comparison can and cannot prove. `stamp_provenance_v4.py` copies
`effect_sha256` OUT of the reference artefact and into every oracle's NPZ post-hoc, so a
match pins the artefact REVISION the file was stamped against -- it is not evidence the
oracle sampled those positions, and nothing in a background records the positions
themselves. The disagreement direction is still worth catching, and is tested below.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
REF = REPO / "reference_sets" / "chorus_reference_positions_v1.npz"

pytestmark = pytest.mark.skipif(not REF.exists(),
                                reason="reference set not generated in this checkout")


@pytest.fixture(scope="module")
def brps():
    spec = importlib.util.spec_from_file_location(
        "brps", REPO / "scripts" / "build_reference_position_sets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ref_prov():
    with np.load(REF, allow_pickle=False) as d:
        return json.loads(str(d["provenance"][0]))


@pytest.fixture(scope="module")
def n_gene_anchored():
    with np.load(REF, allow_pickle=False) as d:
        return len(d["snps_gene_anchored"])


def _fake_background(directory: Path, oracle: str, n_snps: int,
                     build_config: dict | None = None) -> Path:
    """The audit's fabricated file: fake tracks, all-zero CDFs, plausible counts."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        "track_ids": np.array([f"FAKE:{i}" for i in range(5)]),
        "effect_cdfs": np.zeros((5, 10_001), dtype=np.float32),
        "effect_counts": np.array([n_snps] * 5, dtype=np.int64),
    }
    if build_config is not None:
        arrays["build_config"] = np.array([json.dumps(build_config, sort_keys=True)])
    out = directory / f"{oracle}_pertrack.npz"
    np.savez_compressed(out, **arrays)
    return out


def _honest_stamp(ref_prov: dict, family: str = "gene_anchored") -> dict:
    """What stamp_provenance_v4.py writes: the artefact's own hash and strata, copied."""
    meta = ref_prov["sets"][f"snps_{family}"]
    return {"schema_version": 4,
            "reference_sets": {"artefact": REF.name,
                               "effect_family": family,
                               "effect_sha256": meta["sha256"],
                               "effect_strata": meta["strata_realised"]}}


# ---------------------------------------------------------------------------
# The demonstration
# ---------------------------------------------------------------------------


def test_a_fabricated_background_is_not_reported_as_verified(
        brps, tmp_path, caplog, n_gene_anchored):
    """Matching counts is not population identity, and must not be printed as such."""
    _fake_background(tmp_path, "sei", n_gene_anchored)
    with caplog.at_level("INFO"):
        rc = brps.verify(REF, "sei", tmp_path)
    text = caplog.text
    assert rc == 0, "counts do match; the defect was the CLAIM, not the exit code"
    assert "reproduces sei's sampled population" not in text, text
    assert "SKIPPED -- counts only" in text, text
    assert "Population identity was NOT checked" in text, text


def test_strict_refuses_a_background_with_no_composition_evidence(
        brps, tmp_path, n_gene_anchored):
    _fake_background(tmp_path, "sei", n_gene_anchored)
    assert brps.verify(REF, "sei", tmp_path, strict=True) == 1


def test_the_default_stays_permissive_for_a_mid_rebuild_interim(
        brps, tmp_path, n_gene_anchored):
    """An `*_effect_cdfs_interim.npz` is written before any stamping, and checking a
    staged rebuild before it is swapped is the whole point of --backgrounds-dir. So the
    counts-only path must remain usable without --strict; it is the CLAIM that changed."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tmp_path / "sei_effect_cdfs_interim.npz",
                        track_ids=np.array(["FAKE:0"]),
                        effect_counts=np.array([n_gene_anchored], dtype=np.int64))
    assert brps.verify(REF, "sei", tmp_path) == 0
    assert brps.verify(REF, "sei", tmp_path, strict=True) == 1


# ---------------------------------------------------------------------------
# What the stamp comparison does catch
# ---------------------------------------------------------------------------


def test_a_stamped_background_passes_strict_and_says_what_was_compared(
        brps, tmp_path, caplog, ref_prov, n_gene_anchored):
    _fake_background(tmp_path, "sei", n_gene_anchored, _honest_stamp(ref_prov))
    with caplog.at_level("INFO"):
        rc = brps.verify(REF, "sei", tmp_path, strict=True)
    assert rc == 0, caplog.text
    assert "SKIPPED" not in caplog.text, caplog.text
    # and it discloses that the stamp was copied out of this artefact, so a reader cannot
    # mistake the match for evidence about which positions were scored
    assert "copied FROM this artefact" in caplog.text, caplog.text


def test_a_stamp_from_a_different_artefact_revision_fails(
        brps, tmp_path, ref_prov, n_gene_anchored):
    """The one direction the stamp CAN prove: the artefact moved since the stamp.

    Because the artefact's hash is recomputed from its arrays rather than read out of its
    provenance, a background stamped against an older revision of the reference set is
    caught -- its percentiles rank against a population this artefact no longer contains.
    """
    stamp = _honest_stamp(ref_prov)
    stamp["reference_sets"]["effect_sha256"] = "0" * 64
    _fake_background(tmp_path, "sei", n_gene_anchored, stamp)
    assert brps.verify(REF, "sei", tmp_path) == 1        # fails without --strict too


def test_stamped_strata_that_disagree_with_the_artefact_fail(
        brps, tmp_path, ref_prov, n_gene_anchored):
    stamp = _honest_stamp(ref_prov)
    strata = dict(stamp["reference_sets"]["effect_strata"])
    victim = sorted(strata)[0]
    strata[victim] = strata[victim] + 1
    stamp["reference_sets"]["effect_strata"] = strata
    _fake_background(tmp_path, "sei", n_gene_anchored, stamp)
    assert brps.verify(REF, "sei", tmp_path) == 1


def test_a_stamped_family_that_disagrees_with_the_map_fails(
        brps, tmp_path, ref_prov, n_gene_anchored):
    """ORACLE_SNP_SET changing without a re-stamp leaves a file claiming another
    reference class than the one it is now being compared against."""
    stamp = _honest_stamp(ref_prov)
    stamp["reference_sets"]["effect_family"] = "accessibility"
    _fake_background(tmp_path, "sei", n_gene_anchored, stamp)
    assert brps.verify(REF, "sei", tmp_path) == 1


def test_the_build_log_is_still_the_strongest_evidence_when_present(
        brps, tmp_path, caplog, ref_prov, n_gene_anchored):
    """`effect_region_set_as_logged` is written by the BUILDER from what it sampled, so it
    is the only composition evidence here that is independent of this artefact. It exists
    in none of the eight shipped files, so nothing else exercises this branch."""
    cfg = {"schema_version": 3,
           "effect_region_set_as_logged": {
               "strata_counts": ref_prov["sets"]["snps_gene_anchored"]["strata_realised"]}}
    _fake_background(tmp_path, "sei", n_gene_anchored, cfg)
    with caplog.at_level("INFO"):
        assert brps.verify(REF, "sei", tmp_path, strict=True) == 0, caplog.text
    assert "strata match the build log exactly" in caplog.text

    caplog.clear()
    bad = {"schema_version": 3,
           "effect_region_set_as_logged": {"strata_counts": {"random": n_gene_anchored}}}
    _fake_background(tmp_path, "sei", n_gene_anchored, bad)
    assert brps.verify(REF, "sei", tmp_path) == 1


# ---------------------------------------------------------------------------
# The artefact must hash to its own claim
# ---------------------------------------------------------------------------


def test_an_artefact_edited_after_generation_is_refused(
        brps, tmp_path, ref_prov, n_gene_anchored):
    """Otherwise the hash quoted at every oracle describes a file that no longer exists.

    Rewrite the artefact with one alt allele flipped and its provenance untouched -- the
    shape a hand-fixed or re-saved artefact has -- and the comparison must refuse rather
    than compare the stale claim against itself.
    """
    with np.load(REF, allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    s = arrays["snps_gene_anchored"].copy()
    r, a = s["ref"][0], s["alt"][0]
    s["alt"][0] = next(b for b in "ACGT" if b not in (r, a))
    arrays["snps_gene_anchored"] = s
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **arrays)

    _fake_background(tmp_path, "sei", n_gene_anchored, _honest_stamp(ref_prov))
    assert brps.verify(tampered, "sei", tmp_path) == 1
    # and the pristine artefact passes, so the check is not simply always-fail
    assert brps.verify(REF, "sei", tmp_path) == 0


# ---------------------------------------------------------------------------
# The shipped files
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("oracle", ["enformer", "borzoi", "alphagenome", "sei",
                                    "epinformerseq", "legnet", "chrombpnet", "cherimoya"])
def test_every_shipped_background_passes_strict(brps, oracle):
    """All eight carry a schema-4 stamp, so --strict is satisfiable on a real release and
    can be wired into the release gate rather than being a flag nobody can pass."""
    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR

    root = CHORUS_BACKGROUNDS_DIR
    if not (root / f"{oracle}_pertrack.npz").exists():
        pytest.skip(f"no built background for {oracle}")
    assert brps.verify(REF, oracle, root, strict=True) == 0
