"""The reference REGION and SNP sets: versioned populations, not seed-derived each build.

Until 2026-08-07 every build re-derived its positions from a seed at run time.
Reproducible — but only while nothing upstream moves, and the GTF, cCRE BED, DHS index and
FASTA are all inputs that can be updated without anyone noticing that every background's
reference class changed with them. Since **the mixture composition IS the reference-class
definition** (a percentile has no meaning apart from the population it ranks), a silent
change there silently redefines every percentile chorus reports.

So the positions are now an artefact with a content hash, and a build is checkable against
it. That is what makes these *reference* nulls rather than merely reproducible ones.

It found a defect on its first run: epinformerseq had been built on 10,000 positions while
the rest of its family used 18,000 — its builder's `--n-variants` defaults to 10,000 and
the fleet driver never passed it. That background had already passed the distributional
verifier (body ratios ~1.0, ceilings up, retention exact), because nothing else compared
the *population* against what the other oracles used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
REF = REPO / "reference_sets" / "chorus_reference_positions_v1.npz"

pytestmark = pytest.mark.skipif(not REF.exists(),
                                reason="reference set not generated in this checkout")


@pytest.fixture(scope="module")
def ref():
    with np.load(REF, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


@pytest.fixture(scope="module")
def prov(ref):
    return json.loads(str(ref["provenance"][0]))


# ---------------------------------------------------------------------------
# Shape and completeness
# ---------------------------------------------------------------------------


def test_all_three_families_are_present(ref):
    """One per reference class. A missing family means an oracle cannot be checked."""
    for fam in ("gene_anchored", "promoter", "accessibility"):
        assert f"snps_{fam}" in ref, sorted(ref)


def test_every_oracle_maps_to_a_family(ref):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "brps", REPO / "scripts" / "build_reference_position_sets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from chorus.mcp.server import ORACLE_SPECS

    mapped = set(mod.ORACLE_SNP_SET)
    registered = {o for o in ORACLE_SPECS if o != "alphagenome_pt"}
    missing = registered - mapped
    assert not missing, (
        f"{sorted(missing)} have no reference SNP family, so their nulls cannot be "
        f"checked against a known population"
    )
    for oracle, fam in mod.ORACLE_SNP_SET.items():
        assert f"snps_{fam}" in ref, (oracle, fam)


def test_snps_are_well_formed(ref):
    for fam in ("gene_anchored", "promoter", "accessibility"):
        s = ref[f"snps_{fam}"]
        assert len(s) > 5_000, f"{fam}: only {len(s)} SNPs"
        assert set(s["ref"].tolist()) <= set("ACGT"), fam
        assert set(s["alt"].tolist()) <= set("ACGT"), fam
        # An alt equal to the ref is not a variant.
        assert not (s["ref"] == s["alt"]).any(), f"{fam}: ref == alt on some rows"
        assert (s["pos"] > 0).all()
        # Primary contigs only. 81.7% of the promoter set's `random` stratum used to
        # land on unplaced scaffolds and alt haplotypes (2,206 positions across 109
        # contigs) because that sampler filtered on margin alone; alt contigs are
        # redundant copies of primary sequence and scaffolds are largely repetitive, so
        # that stratum was not a uniform genomic background.
        odd = sorted({str(c) for c in s["chrom"]}
                     - ({f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}))
        assert not odd, f"{fam}: non-primary contigs {odd[:6]}"
        # And the chrom field must be wide enough to hold a full name: a U8 dtype
        # truncated chr11_KI270721v1_random to 'chr11_KI', which is lossy AND ambiguous
        # because several scaffolds collapse to the same string.
        assert s["chrom"].dtype.itemsize // 4 >= 32, s["chrom"].dtype


def test_positions_are_distinct_within_a_family(ref):
    """Duplicate positions give identical effects: padded counts, manufactured CDF ties.

    This is the defect that clamping onto contig margins produced (chr16:5,000,000 once
    appeared 64 times), so the reference set is where it must not recur.
    """
    for fam in ("gene_anchored", "promoter"):
        s = ref[f"snps_{fam}"]
        coords = list(zip(s["chrom"].tolist(), s["pos"].tolist()))
        assert len(set(coords)) >= 0.995 * len(coords), (
            f"{fam}: {len(coords) - len(set(coords))} duplicate positions")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_records_what_is_needed_to_reproduce(prov):
    for key in ("schema_version", "generator", "generator_git_sha", "genome",
                "fai_sha256", "n_effect_positions", "seeds", "sets"):
        assert key in prov, sorted(prov)
    assert prov["genome"] == "hg38"
    # Every seed that affects which positions and SNPs come out.
    assert set(prov["seeds"]) >= {"regions", "dhs_pool", "snp_alt"}
    for name, meta in prov["sets"].items():
        assert len(meta["sha256"]) == 64, name
        assert meta["purpose"], name
        assert meta["strata_realised"], name


def test_the_content_hash_is_order_independent_and_actually_covers_the_snps(ref, prov):
    """A hash that does not change when the content does is worse than no hash."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "brps", REPO / "scripts" / "build_reference_position_sets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    s = ref["snps_gene_anchored"]
    rows = list(zip(s["chrom"].tolist(), s["pos"].tolist(), s["ref"].tolist(),
                    s["alt"].tolist(), s["stratum"].tolist()))
    assert mod._sha256_of(rows) == prov["sets"]["snps_gene_anchored"]["sha256"]
    # order-independent
    assert mod._sha256_of(list(reversed(rows))) == mod._sha256_of(rows)
    # and sensitive: flip one alt allele
    tampered = list(rows)
    c, p, r, a, st = tampered[0]
    tampered[0] = (c, p, r, next(b for b in "ACGT" if b not in (r, a)), st)
    assert mod._sha256_of(tampered) != mod._sha256_of(rows)


def test_realised_strata_match_the_requested_proportions(prov):
    from chorus.utils.annotations import DEFAULT_N_EFFECT_POSITIONS

    meta = prov["sets"]["snps_gene_anchored"]
    n = DEFAULT_N_EFFECT_POSITIONS
    for stratum, frac in meta["strata_requested"].items():
        want = round(n * frac)
        got = meta["strata_realised"].get(stratum, 0)
        # Only non-ACGT reference bases are lost, so realised <= requested.
        assert got <= want, f"{stratum}: {got} realised > {want} requested"
        # `random` draws uniformly and so lands in centromeric/telomeric N far more often
        # than an anchored stratum does: measured 93.3% retained against ~100% for the
        # anchored ones. Floor it separately rather than loosening the whole check.
        floor = 0.90 if stratum == "random" else 0.99
        assert got >= floor * want, (
            f"{stratum}: {got} of {want} retained ({got / want:.1%}), floor {floor:.0%}")


def test_the_promoter_family_excludes_dhs(prov):
    """DHS was measured to dilute every quantile of a promoter null; see the protocol."""
    assert "dhs" not in prov["sets"]["snps_promoter"]["strata_realised"]
    assert "dhs" not in prov["sets"]["snps_gene_anchored"]["strata_realised"]
    # but the accessibility family is DHS-anchored BY DESIGN and must keep it
    assert prov["sets"]["snps_accessibility"]["strata_realised"].get("dhs", 0) > 1_000


# ---------------------------------------------------------------------------
# Does a built background actually use it?
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("oracle", ["enformer", "borzoi", "sei", "legnet",
                                    "cherimoya", "epinformerseq"])
def test_a_built_background_reproduces_the_reference_population(oracle):
    """The check that caught epinformerseq at 10,000 positions instead of 18,000.

    Shortfall is allowed but bounded: a window whose N content exceeds max_n_fraction is
    rejected, and window sizes differ by orders of magnitude (Sei ~4 kb, Enformer 393 kb,
    Borzoi 524 kb), so the retained subset is oracle-specific. Measured: Sei 0, Borzoi 1,
    Enformer 2.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "brps", REPO / "scripts" / "build_reference_position_sets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR
    staged = Path("/data/chorus_data/rebuild_2026-08-06")
    # Prefer a staged rebuild, including one that has only reached its effect interim --
    # checking against the LIVE file while a rebuild is staged reports a ~33% shortfall,
    # which is true (the shipped nulls use the older 12,000-position class) and not the
    # question being asked.
    if (staged / f"{oracle}_pertrack.npz").exists() or             (staged / f"{oracle}_effect_cdfs_interim.npz").exists():
        root = staged
    elif (CHORUS_BACKGROUNDS_DIR / f"{oracle}_pertrack.npz").exists():
        root = CHORUS_BACKGROUNDS_DIR
    else:
        pytest.skip(f"no background for {oracle}")
    assert mod.verify(REF, oracle, root) == 0, (
        f"{oracle}'s background does not reproduce its reference population"
    )
