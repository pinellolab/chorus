"""A position labelled ``dhs`` must actually be at a DHS summit.

`sample_gene_anchored_positions` dispatched on the stratum name through a chain of
``elif``s ending in a bare ``else`` that drew a uniformly random position. That
``else`` was simultaneously three things: the handler for the ``random`` stratum, the
fallback when an anchored stratum's source population came back empty, and the
catch-all for any name the chain did not recognise.

So adding ``"dhs": 1/3`` to the strata dict without also adding a branch would have
emitted 6,000 uniformly random positions, tagged them ``"dhs"``, tallied them as DHS
in the build log, and stamped them as DHS in the artefact's provenance. Every
downstream reader -- the CHANGELOG, the README table, anyone re-deriving the reference
class -- would have been told the null contained DHS-anchored variants when it did
not. Nothing would have raised, and the numbers would have looked plausible, because
uniformly random positions produce a perfectly reasonable-looking null. It just would
not have been the null anyone claimed.

The name-level guards below (unknown stratum raises, empty pool raises) make that
specific mistake impossible. This module's real work is the round-trip: take the
positions the sampler actually returned and check each one against the annotation its
label names. That catches the whole class, including a branch that exists but reads
the wrong pool.
"""
from __future__ import annotations

import os
from collections import Counter

import pytest

from chorus.utils.annotations import (
    DEFAULT_N_EFFECT_POSITIONS,
    DEFAULT_REGION_STRATA,
    PROMOTER_REGION_STRATA,
    load_chrom_sizes,
    sample_gene_anchored_positions,
    sample_promoter_anchored_positions,
)

FAI = "genomes/hg38.fa.fai"


@pytest.fixture(scope="module")
def sizes():
    if not os.path.exists(FAI):
        pytest.skip("hg38.fa.fai not available")
    return load_chrom_sizes(FAI)


# ---------------------------------------------------------------------------
# Name-level: the landmine cannot be re-armed
# ---------------------------------------------------------------------------


def test_an_unknown_stratum_raises_instead_of_silently_going_random(sizes):
    with pytest.raises(ValueError, match="unhandled stratum"):
        sample_gene_anchored_positions(
            100, chrom_sizes=sizes, strata={"random": 0.5, "not_a_stratum": 0.5},
        )


def test_an_unknown_stratum_raises_in_the_promoter_sampler_too(sizes):
    """Both samplers had the identical hole; fixing one would have been a half-fix."""
    with pytest.raises(ValueError, match="unhandled stratum"):
        sample_promoter_anchored_positions(
            100, chrom_sizes=sizes, strata={"random": 0.5, "not_a_stratum": 0.5},
        )


@pytest.mark.parametrize("strata_dict,sampler", [
    (DEFAULT_REGION_STRATA, sample_gene_anchored_positions),
    (PROMOTER_REGION_STRATA, sample_promoter_anchored_positions),
])
def test_every_shipped_stratum_name_has_a_handler(strata_dict, sampler, sizes):
    """Each name in the shipped dicts must be individually samplable.

    Requesting one stratum at 1.0 forces its own branch to run. If a name had no
    handler it would either raise (caught here) or quietly return random positions
    tagged with that name (caught by the round-trip tests below).
    """
    for name in strata_dict:
        out = sampler(60, chrom_sizes=sizes, strata={name: 1.0}, seed=3)
        assert {s for _, _, s in out} == {name}, name
        assert len(out) == 60, name


def test_an_empty_source_population_raises_rather_than_substituting(sizes):
    """A missing annotation file must be loud.

    ``margin_bp`` larger than every chromosome empties every anchored pool. Before,
    that fell through to uniform random positions wearing the anchored label.
    """
    with pytest.raises(ValueError):
        sample_gene_anchored_positions(
            50, chrom_sizes=sizes, strata={"ccre": 1.0}, margin_bp=200_000_000,
        )


# ---------------------------------------------------------------------------
# The round trip: positions must match the annotation they claim
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_dhs_positions_are_actually_at_dhs_summits(sizes):
    """The decisive check. Random positions would score ~0% here, not >=99%."""
    import numpy as np

    from chorus.utils.annotations import load_dhs_vocabulary

    out = sample_gene_anchored_positions(
        900, chrom_sizes=sizes, strata={"dhs": 1.0}, seed=11)
    assert len(out) == 900

    df = load_dhs_vocabulary()
    by_chrom = {}
    for chrom, grp in df.groupby("seqname"):
        by_chrom[str(chrom)] = np.sort(grp["summit"].to_numpy())

    tol = 150 + 1
    hits = 0
    for c, p, _ in out:
        summits = by_chrom.get(c)
        if summits is None or not len(summits):
            continue
        j = np.searchsorted(summits, p)
        for k in (j - 1, j):
            if 0 <= k < len(summits) and abs(int(summits[k]) - p) <= tol:
                hits += 1
                break
    frac = hits / len(out)
    assert frac >= 0.99, (
        f"only {frac:.1%} of positions labelled 'dhs' are within {tol} bp of a "
        f"Meuleman summit. Uniformly random positions would score ~0%, which is "
        f"exactly what the old fall-through would have produced."
    )


@pytest.mark.integration
def test_ccre_positions_are_actually_inside_ccres(sizes):
    import numpy as np

    from chorus.utils.annotations import get_screen_ccres

    out = sample_gene_anchored_positions(
        600, chrom_sizes=sizes, strata={"ccre": 1.0}, seed=12)
    ccres = get_screen_ccres()
    starts, ends = {}, {}
    for chrom, grp in ccres.groupby("chrom"):
        order = np.argsort(grp["start"].to_numpy())
        starts[str(chrom)] = grp["start"].to_numpy()[order]
        ends[str(chrom)] = grp["end"].to_numpy()[order]

    hits = 0
    for c, p, _ in out:
        s = starts.get(c)
        if s is None:
            continue
        j = np.searchsorted(s, p, side="right") - 1
        if 0 <= j < len(s) and s[j] <= p <= ends[c][j]:
            hits += 1
    frac = hits / len(out)
    assert frac >= 0.99, f"only {frac:.1%} of 'ccre' positions fall inside a cCRE"


@pytest.mark.integration
def test_tss_near_positions_are_actually_near_a_tss(sizes):
    import numpy as np

    from chorus.utils.annotations import get_annotation_manager

    out = sample_gene_anchored_positions(
        400, chrom_sizes=sizes, strata={"tss_near": 1.0}, seed=13)
    mgr = get_annotation_manager()
    genes = mgr._get_genes_df(mgr.get_annotation_path("gencode_v48_basic"))
    pc = genes[genes["gene_type"] == "protein_coding"]
    tss = {}
    for r in pc.itertuples():
        tss.setdefault(str(r.chrom), []).append(
            int(r.start) if r.strand == "+" else int(r.end))
    tss = {c: np.sort(np.array(v)) for c, v in tss.items()}

    tol = 1_000 + 1
    hits = 0
    for c, p, _ in out:
        arr = tss.get(c)
        if arr is None:
            continue
        j = np.searchsorted(arr, p)
        for k in (j - 1, j):
            if 0 <= k < len(arr) and abs(int(arr[k]) - p) <= tol:
                hits += 1
                break
    frac = hits / len(out)
    assert frac >= 0.99, f"only {frac:.1%} of 'tss_near' positions are within 1 kb"


# ---------------------------------------------------------------------------
# Additivity: adding a component must not move the others
# ---------------------------------------------------------------------------


def test_n_grew_and_the_proportions_held():
    """More positions from the SAME populations is the lever that measured well.

    A DHS third was added here and removed the same day. Three Sei builds, differing
    only as labelled, medians over 40 tracks:

        A = 12,000 no DHS   B = 18,000 +DHS   C = 18,000 no DHS

            p50     p90     p99     p99.9   max
        B/A 0.971   0.937   0.954   0.936   1.000
        C/A 1.035   1.030   1.042   0.992   1.261

    B/A max is exactly 1.000 -- across all 40 tracks not one DHS position beat the best
    cCRE- or gene-anchored position already in the set, so DHS added nothing to the
    ceiling while lowering every quantile. C, the same budget spent on the existing
    populations, raised the ceiling 26%.

    Re-dividing a fixed N would dilute, which is a separate and still-true hazard:
    measured when the cCRE half was first tried that way, TF saturation went 25% ->
    92% because each component got half the draws.
    """
    n = DEFAULT_N_EFFECT_POSITIONS
    assert n == 18_000
    expected = {"tss_near": 1800, "tss_far": 1800, "junction": 2970,
                "gene_body": 1080, "random": 1350, "ccre": 9000}
    for name, want in expected.items():
        assert round(n * DEFAULT_REGION_STRATA[name]) == want, name
    assert sum(expected.values()) == n
    assert "dhs" not in DEFAULT_REGION_STRATA, (
        "DHS was measured to add nothing to the ceiling and to dilute every quantile "
        "on both the gene-anchored and promoter mixtures; see the table above"
    )

    promoter = {"tss_promoter": 7200, "ccre_pls": 5400, "ccre_pels": 2700,
                "random": 2700}
    for name, want in promoter.items():
        assert round(n * PROMOTER_REGION_STRATA[name]) == want, f"promoter {name}"
    assert sum(promoter.values()) == n
    assert "dhs" not in PROMOTER_REGION_STRATA


def test_the_dhs_branch_still_works_even_though_it_is_not_in_the_defaults(sizes):
    """Removed from the mixtures, kept as a capability.

    ChromBPNet's and Cherimoya's nulls have always been DHS-anchored, and the ablation
    that produced the decision needs the branch to exist. Deleting it would make the
    measurement unrepeatable.
    """
    out = sample_gene_anchored_positions(
        120, chrom_sizes=sizes, strata={"dhs": 1.0}, seed=5)
    assert len(out) == 120
    assert {s for _, _, s in out} == {"dhs"}


def test_pool_cursors_are_per_stratum_not_global(sizes):
    """``ccre_pool[len(out) % len(pool)]`` made cCRE draws depend on every other
    stratum's size, so inserting one silently re-drew the cCRE half.

    Verified behaviourally: the positions drawn for a stratum must not change when a
    DIFFERENT stratum's share changes.
    """
    a = sample_gene_anchored_positions(
        400, chrom_sizes=sizes, seed=7,
        strata={"ccre": 0.5, "random": 0.5})
    b = sample_gene_anchored_positions(
        400, chrom_sizes=sizes, seed=7,
        strata={"ccre": 0.5, "tss_near": 0.25, "random": 0.25})
    ccre_a = sorted(p for _, p, s in a if s == "ccre")
    ccre_b = sorted(p for _, p, s in b if s == "ccre")
    assert ccre_a == ccre_b, (
        "the cCRE positions changed when a different stratum's share changed, so the "
        "pool cursor is still global rather than per-stratum"
    )


def test_strata_proportions_are_honoured_with_dhs(sizes):
    out = sample_gene_anchored_positions(1_500, chrom_sizes=sizes, seed=21)
    counts = Counter(s for _, _, s in out)
    for name, frac in DEFAULT_REGION_STRATA.items():
        assert counts[name] == pytest.approx(1_500 * frac, abs=2), name
    assert set(counts) == set(DEFAULT_REGION_STRATA)


# ---------------------------------------------------------------------------
# Clamping: an anchored position must not be silently relocated
# ---------------------------------------------------------------------------


def test_anchored_positions_are_not_clamped_onto_the_contig_margin(sizes):
    """Found by the tss_near round-trip above, and it is the more serious defect.

    ``usable`` selects whole CHROMOSOMES long enough for the margin, but the tss,
    junction and gene-body populations were then filtered only by ``chrom in usable``
    -- never by the margin interval itself. 2,515 of 20,083 protein-coding TSS
    (12.5%) sit within 5 Mb of a contig end, so they passed that test and ``_clamp``
    moved them onto the boundary coordinate, up to 5 Mb from the TSS whose label they
    carried. The cCRE pool had always been filtered properly; these three had not.

    Measured before the fix, over 6,000 positions: 12.1% of ``tss_near``, 12.2% of
    ``junction``, 13.0% of ``tss_far`` and 14.6% of ``gene_body`` landed exactly on a
    boundary; only 5,265 of 6,000 positions were distinct; and chr16:5,000,000 alone
    appeared 64 times.

    The mislabelling is bad. The duplication is worse: identical positions produce
    identical effect values, which inflate the sample count without adding
    information and manufacture tied runs in the CDF -- the same degeneracy
    ``_rank_with_tie_breaking`` exists to compensate for, injected by the sampler.
    """
    margin = 5_000_000
    usable = {c: L for c, L in sizes.items() if L > 2 * margin}
    out = sample_gene_anchored_positions(
        2_000, chrom_sizes=sizes, seed=99,
        strata={"tss_near": 0.25, "tss_far": 0.25,
                "junction": 0.25, "gene_body": 0.25})

    on_boundary = [(c, p, s) for c, p, s in out
                   if p == margin or p == usable.get(c, 1 << 60) - margin]
    frac = len(on_boundary) / len(out)
    assert frac < 0.01, (
        f"{frac:.1%} of anchored positions sit exactly on a contig-end margin "
        f"boundary (was 12-15% before the source populations were margin-filtered). "
        f"Examples: {on_boundary[:4]}"
    )


def test_sampled_positions_are_distinct(sizes):
    """Duplicates are wasted forward passes AND manufactured CDF ties."""
    out = sample_gene_anchored_positions(
        2_000, chrom_sizes=sizes, seed=99,
        strata={"tss_near": 0.25, "tss_far": 0.25,
                "junction": 0.25, "gene_body": 0.25})
    coords = [(c, p) for c, p, _ in out]
    distinct = len(set(coords))
    assert distinct >= 0.995 * len(coords), (
        f"only {distinct} of {len(coords)} positions are distinct; duplicate "
        f"positions give identical effect values, so they pad the sample count and "
        f"create tied runs in the CDF without adding any information"
    )
