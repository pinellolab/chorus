"""The effect null must be drawn where the assay has signal (#83).

Today's effect backgrounds come from ~1,900-2,000 **uniformly random** genomic
positions (``build_backgrounds_alphagenome.py``'s
``random.randint(5_000_000, max_pos)``). For a TSS-peaked assay like CAGE, or an
exon-scoped one like RNA, a random position carries almost no signal — so the null
collapses toward zero and every real effect reads >= 99th percentile. AlphaGenome
RNA's effect null tops out at **0.0417**: anything >= 0.05 saturates at exactly
1.0000, and no floor can fix a saturated column.

Measured, 4,000 positions each, against GENCODE v48:

========================  ==================  ============  ==========================
position set              median dist to TSS  within 1 kb   within 100 bp of a junction
========================  ==================  ============  ==========================
uniform random (today)            102,333 bp         1.4 %                        2.3 %
gene-anchored                       9,430 bp        21.3 %                       37.4 %
========================  ==================  ============  ==========================

15x more TSS-proximal positions and 16x more junction-proximal ones — that is the
mass the CAGE and splice nulls are missing.

Note "does the 1 Mb window contain a gene" is **not** the discriminating metric:
85 % of uniformly random windows already do, because a megabase catches a gene
almost anywhere. What matters is the distance from the **variant** to the anchor,
since CAGE is scored in a 501 bp window centred on it.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from chorus.utils.annotations import (
    DEFAULT_REGION_STRATA,
    load_chrom_sizes,
    sample_gene_anchored_positions,
)

FAI = "genomes/hg38.fa.fai"


@pytest.fixture(scope="module")
def sizes():
    import os
    if not os.path.exists(FAI):
        pytest.skip("hg38.fa.fai not available")
    return load_chrom_sizes(FAI)


@pytest.fixture(scope="module")
def sampled(sizes):
    return sample_gene_anchored_positions(2_000, chrom_sizes=sizes, seed=99)


# ---------------------------------------------------------------------------
# The mixture
# ---------------------------------------------------------------------------


def test_strata_sum_to_one():
    assert sum(DEFAULT_REGION_STRATA.values()) == pytest.approx(1.0)


def test_a_mixture_that_does_not_sum_to_one_is_rejected(sizes):
    with pytest.raises(ValueError, match="sum to 1.0"):
        sample_gene_anchored_positions(
            10, chrom_sizes=sizes, strata={"random": 0.5, "tss_near": 0.2},
        )


def test_the_random_stratum_is_present_and_substantial():
    """Load-bearing, not filler.

    Without a near-zero mass the null loses its lower body and small real effects
    would get artificially LOW percentiles — the exact mirror of today's failure.
    """
    assert DEFAULT_REGION_STRATA["random"] >= 0.10


def test_strata_proportions_are_honoured(sampled):
    counts = Counter(s for _, _, s in sampled)
    for name, frac in DEFAULT_REGION_STRATA.items():
        assert counts[name] == pytest.approx(2_000 * frac, abs=2), name


def test_every_position_is_tagged_with_its_stratum(sampled):
    """The composition has to be recoverable from provenance (#124); a background
    whose reference class cannot be re-derived is one nobody can reproduce."""
    assert {s for _, _, s in sampled} == set(DEFAULT_REGION_STRATA)


def test_custom_strata_are_respected(sizes):
    out = sample_gene_anchored_positions(
        100, chrom_sizes=sizes, strata={"tss_near": 1.0}, seed=1,
    )
    assert {s for _, _, s in out} == {"tss_near"}


# ---------------------------------------------------------------------------
# Safety: nothing may fall off a contig
# ---------------------------------------------------------------------------


def test_positions_leave_room_for_a_1mb_window(sampled, sizes):
    """A 1,048,576 bp prediction must always fit, matching the builders' guard."""
    for chrom, pos, stratum in sampled:
        assert chrom in sizes, chrom
        assert pos >= 5_000_000, f"{chrom}:{pos} ({stratum}) too close to the start"
        assert pos <= sizes[chrom] - 5_000_000, f"{chrom}:{pos} ({stratum}) too close to the end"


def test_spread_across_chromosomes(sampled):
    assert len({c for c, _, _ in sampled}) >= 20


def test_deterministic_for_a_given_seed(sizes):
    a = sample_gene_anchored_positions(200, chrom_sizes=sizes, seed=4)
    b = sample_gene_anchored_positions(200, chrom_sizes=sizes, seed=4)
    assert a == b
    c = sample_gene_anchored_positions(200, chrom_sizes=sizes, seed=5)
    assert a != c


# ---------------------------------------------------------------------------
# The property that actually matters
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def anchors():
    from chorus.utils.annotations import get_annotation_manager

    manager = get_annotation_manager()
    gtf = manager.get_annotation_path("gencode_v48_basic")
    genes = manager._get_genes_df(gtf)
    pc = genes[genes["gene_type"] == "protein_coding"]
    exons = manager._get_exons_df(gtf)
    pc_exons = exons[exons["gene_name"].isin(set(pc["gene_name"]))]

    tss: dict = {}
    for row in pc.itertuples():
        tss.setdefault(row.chrom, []).append(
            int(row.start) if row.strand == "+" else int(row.end))
    junctions: dict = {}
    for row in pc_exons.itertuples():
        junctions.setdefault(row.chrom, []).extend((int(row.start), int(row.end)))
    return ({c: np.array(sorted(v)) for c, v in tss.items()},
            {c: np.array(sorted(v)) for c, v in junctions.items()})


def _nearest(table, chrom, pos):
    arr = table.get(chrom)
    if arr is None or not len(arr):
        return 10 ** 9
    i = np.searchsorted(arr, pos)
    return min(abs(pos - arr[j]) for j in (i - 1, i) if 0 <= j < len(arr))


def test_far_more_tss_proximal_than_uniform_random(sampled, anchors, sizes):
    """The whole point: give CAGE's null some positions where CAGE has signal."""
    import random

    tss, _ = anchors
    got = np.array([_nearest(tss, c, p) for c, p, _ in sampled])

    rng = random.Random(7)
    usable = [k for k in sizes if sizes[k] > 10_000_000]
    uniform = np.array([
        _nearest(tss, c, rng.randint(5_000_000, sizes[c] - 5_000_000))
        for c in (rng.choice(usable) for _ in range(2_000))
    ])

    assert np.median(got) < np.median(uniform) / 5
    assert (got <= 1_000).mean() > 8 * (uniform <= 1_000).mean()


def test_far_more_junction_proximal_than_uniform_random(sampled, anchors, sizes):
    import random

    _, junctions = anchors
    got = np.array([_nearest(junctions, c, p) for c, p, _ in sampled])

    rng = random.Random(11)
    usable = [k for k in sizes if sizes[k] > 10_000_000]
    uniform = np.array([
        _nearest(junctions, c, rng.randint(5_000_000, sizes[c] - 5_000_000))
        for c in (rng.choice(usable) for _ in range(2_000))
    ])
    assert (got <= 100).mean() > 5 * (uniform <= 100).mean()


def test_tss_strata_land_near_a_tss(sampled, anchors):
    """Guards the strand handling. TSS is ``start`` for ``+`` and ``end`` for
    ``-``; getting it backwards anchors CAGE on transcript 3' ends, where there is
    no promoter signal — and the mixture would still look correct by count."""
    tss, _ = anchors
    near = [_nearest(tss, c, p) for c, p, s in sampled if s == "tss_near"]
    far = [_nearest(tss, c, p) for c, p, s in sampled if s == "tss_far"]
    assert np.median(near) <= 1_000
    assert np.median(near) < np.median(far)
    assert np.median(far) <= 10_000


def test_junction_stratum_lands_near_a_junction(sampled, anchors):
    _, junctions = anchors
    d = [_nearest(junctions, c, p) for c, p, s in sampled if s == "junction"]
    assert np.median(d) <= 100


# ---------------------------------------------------------------------------
# The builders must all be on it (source assertions, the #144 pattern)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("oracle", ["alphagenome", "borzoi", "enformer"])
def test_rebuild_set_samples_gene_anchored_positions(oracle):
    """All three oracles being rebuilt, or the fix reaches only one background."""
    from pathlib import Path

    src = Path(f"scripts/build_backgrounds_{oracle}.py").read_text()
    assert "sample_gene_anchored_positions(" in src, f"{oracle} still samples its own way"
    # the old uniform draw must be gone from CODE (comments may reference it)
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "randint(5_000_000" not in code, f"{oracle} still draws uniformly"


@pytest.mark.parametrize("oracle", ["alphagenome", "borzoi"])
def test_rna_oracles_scope_the_mask_per_gene(oracle):
    """Enformer is excluded on purpose: it has no gene_expression layer at all
    (its LAYER_SPEC has no RNA key), so there is no mask to scope."""
    from pathlib import Path

    src = Path(f"scripts/build_backgrounds_{oracle}.py").read_text()
    assert "build_gene_exon_index()" in src
    assert "exon_bins_for_gene(" in src
    assert "genes_overlapping(" in src
    # the chromosome-pooled helpers must be gone, not merely unused
    assert "def load_exon_index" not in src
    assert "def exon_bins_for_window" not in src


def test_enformer_really_has_no_rna_layer():
    """Justifies the exclusion above rather than asserting it by assumption."""
    from pathlib import Path

    src = Path("scripts/build_backgrounds_enformer.py").read_text()
    spec = src[src.index("LAYER_SPEC = {"):]
    spec = spec[:spec.index("}")]
    assert "'RNA'" not in spec and '"RNA"' not in spec
