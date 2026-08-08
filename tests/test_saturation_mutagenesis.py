"""``score_ism`` had zero tests, and a published vignette rests on it.

The MCP tool ``score_ism`` (server.py) delegates to ``saturation_mutagenesis``, which
sweeps every single-base substitution in a window and returns a per-position importance
profile — the thing blog vignette 2 reads as "which bases the oracle actually looks at".
Nothing exercised it.

Tested against a fake oracle rather than a real one, deliberately: the interesting
behaviour is geometry (which positions, which bases, in what order), bookkeeping (the
reference base stays zero, importance is the mean disruption) and failure handling (one
dead site must not lose the sweep). None of that needs a GPU, and on a real oracle all
of it would be hidden behind the numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

from chorus.analysis.saturation import BASES, saturation_mutagenesis


@pytest.fixture
def fasta(tmp_path):
    """A tiny indexed FASTA. chr1 is 60 bp of a known repeating pattern."""
    seq = ("ACGT" * 15)[:60]
    p = tmp_path / "mini.fa"
    p.write_text(f">chr1\n{seq}\n")
    import pyfaidx
    pyfaidx.Fasta(str(p))          # builds the .fai
    return str(p), seq


class FakeOracle:
    """Records every variant it is asked to score and returns a controllable effect.

    ``effect_fn(chrom, pos, ref, alt) -> float`` lets a test make the effect depend on
    position and base, which is what makes the geometry assertions meaningful.
    """

    reference_fasta = None

    def __init__(self, effect_fn=None, fail_at=()):
        self.calls = []
        self.effect_fn = effect_fn or (lambda c, p, r, a: 1.0)
        self.fail_at = set(fail_at)

    def predict_variant_effect(self, region, variant_position, alleles,
                               assay_ids=None, genome=None):
        chrom, pos_s = variant_position.split(":")
        pos = int(pos_s)
        ref, alt = alleles
        self.calls.append((chrom, pos, ref, alt))
        if pos in self.fail_at:
            raise RuntimeError(f"synthetic failure at {pos}")
        return {"_effect": self.effect_fn(chrom, pos, ref, alt)}


@pytest.fixture
def patched_scorer(monkeypatch):
    """Route ``_score_all_tracks`` to the fake's recorded effect."""
    class _TE:
        def __init__(self, v): self.raw_score = v

    import chorus.analysis.discovery as disc
    monkeypatch.setattr(disc, "_score_all_tracks",
                        lambda vr, oracle_name: [_TE(vr["_effect"])])
    return None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_window_is_centred_and_every_position_is_swept(fasta, patched_scorer):
    path, seq = fasta
    oracle = FakeOracle()
    out = saturation_mutagenesis(oracle, "fake", "chr1:30", ["t"],
                                 genome=path, window=9)
    assert out["start"] == 26 and out["end"] == 34
    assert out["positions"] == list(range(26, 35))
    assert out["ref_seq"] == seq[25:34]
    assert len(out["scores"]) == 9
    # 3 substitutions per position, no more and no fewer
    assert len(oracle.calls) == 9 * 3
    assert {c[1] for c in oracle.calls} == set(range(26, 35))


def test_the_reference_base_is_never_substituted_and_stays_zero(fasta, patched_scorer):
    path, seq = fasta
    oracle = FakeOracle()
    out = saturation_mutagenesis(oracle, "fake", "chr1:30", ["t"],
                                 genome=path, window=5)
    scores = np.array(out["scores"])
    for i, ref_b in enumerate(out["ref_seq"]):
        j = BASES.index(ref_b)
        assert scores[i, j] == 0.0, f"position {i}: reference base {ref_b} was scored"
        assert (scores[i, [k for k in range(4) if k != j]] != 0).all()
    # and the oracle was never asked for a ref->ref substitution
    assert all(r != a for _, _, r, a in oracle.calls)


def test_scores_are_indexed_by_the_BASES_order(fasta, patched_scorer):
    """A logo drawn against the wrong base order is silently wrong."""
    path, _ = fasta
    # effect encodes which base was substituted
    oracle = FakeOracle(effect_fn=lambda c, p, r, a: float(BASES.index(a) + 1))
    out = saturation_mutagenesis(oracle, "fake", "chr1:30", ["t"],
                                 genome=path, window=3)
    scores = np.array(out["scores"])
    for i, ref_b in enumerate(out["ref_seq"]):
        for j, b in enumerate(BASES):
            expected = 0.0 if b == ref_b else float(j + 1)
            assert scores[i, j] == expected, (i, b)


# ---------------------------------------------------------------------------
# Importance
# ---------------------------------------------------------------------------


def test_importance_is_the_mean_disruption_and_flips_sign(fasta, patched_scorer):
    """A functional base LOSES signal when mutated, so it must score positive."""
    path, _ = fasta
    losing = FakeOracle(effect_fn=lambda c, p, r, a: -3.0)   # every substitution hurts
    out = saturation_mutagenesis(losing, "fake", "chr1:30", ["t"],
                                 genome=path, window=5)
    assert np.allclose(out["importance"], 3.0), out["importance"]

    gaining = FakeOracle(effect_fn=lambda c, p, r, a: +3.0)
    out = saturation_mutagenesis(gaining, "fake", "chr1:30", ["t"],
                                 genome=path, window=5)
    assert np.allclose(out["importance"], -3.0)


def test_importance_length_matches_the_sequence(fasta, patched_scorer):
    path, _ = fasta
    out = saturation_mutagenesis(FakeOracle(), "fake", "chr1:30", ["t"],
                                 genome=path, window=7)
    assert len(out["importance"]) == len(out["ref_seq"]) == len(out["positions"])


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_one_dead_site_does_not_lose_the_sweep(fasta, patched_scorer):
    """The per-site try/except is deliberate; verify it is also bounded in effect."""
    path, _ = fasta
    oracle = FakeOracle(effect_fn=lambda c, p, r, a: -2.0, fail_at={30})
    out = saturation_mutagenesis(oracle, "fake", "chr1:30", ["t"],
                                 genome=path, window=5)
    scores = np.array(out["scores"])
    idx = out["positions"].index(30)
    # the failed position collapses to all-zero, and so reads as UNIMPORTANT
    assert (scores[idx] == 0).all()
    assert out["importance"][idx] == 0.0
    # every other position is unaffected
    others = [i for i in range(len(scores)) if i != idx]
    assert np.allclose([out["importance"][i] for i in others], 2.0)


def test_a_failed_site_is_indistinguishable_from_an_unimportant_one(fasta,
                                                                    patched_scorer):
    """Documented, not asserted-away: this is a real limitation of the return shape.

    A site whose predictions all raised gets importance 0.0, exactly like a site the
    oracle genuinely does not care about. Nothing in the returned dict marks the
    difference, so a logo drawn from a partially-failed sweep shows confident zeros. The
    warning goes to the log, which a caller rendering a figure will not see.
    """
    path, _ = fasta
    failed = FakeOracle(effect_fn=lambda c, p, r, a: -2.0, fail_at={30})
    flat = FakeOracle(effect_fn=lambda c, p, r, a: 0.0)
    a = saturation_mutagenesis(failed, "fake", "chr1:30", ["t"],
                               genome=path, window=5)
    b = saturation_mutagenesis(flat, "fake", "chr1:30", ["t"],
                               genome=path, window=5)
    i = a["positions"].index(30)
    assert a["importance"][i] == b["importance"][i] == 0.0
    assert not any(k for k in a if "fail" in k.lower() or "drop" in k.lower()), (
        "if a field reporting failed sites has been added, tighten this test to "
        "assert it is populated"
    )


def test_a_non_acgt_reference_base_is_skipped_entirely(tmp_path, patched_scorer):
    import pyfaidx
    p = tmp_path / "n.fa"
    p.write_text(">chr1\n" + ("ACGTN" * 12) + "\n")
    pyfaidx.Fasta(str(p))
    oracle = FakeOracle()
    out = saturation_mutagenesis(oracle, "fake", "chr1:30", ["t"],
                                 genome=str(p), window=11)
    scores = np.array(out["scores"])
    for i, b in enumerate(out["ref_seq"]):
        if b == "N":
            assert (scores[i] == 0).all()
    assert all(r in "ACGT" for _, _, r, _ in oracle.calls)


# ---------------------------------------------------------------------------
# The even-window off-by-one
# ---------------------------------------------------------------------------


def test_an_even_window_returns_one_more_base_than_requested(fasta, patched_scorer):
    """``window`` is documented "odd recommended"; an even value is off by one.

    ``half = window // 2`` then ``start = pos - half``, ``end = pos + half``, so the
    span is always ``2*half + 1`` bases. window=24 returns 25 positions while the
    returned ``window`` field still says 24 — so a caller sizing an array from
    ``window`` and filling it from ``scores`` would be short by one.

    Pinned as current behaviour rather than silently fixed: changing the geometry would
    move every committed ISM artefact, and the docstring already steers callers to odd
    windows. If it is fixed, this test should fail and be updated deliberately.
    """
    path, _ = fasta
    out = saturation_mutagenesis(FakeOracle(), "fake", "chr1:30", ["t"],
                                 genome=path, window=24)
    assert out["window"] == 24
    assert len(out["ref_seq"]) == 25
    assert len(out["positions"]) == 25
    assert len(out["scores"]) == 25


def test_odd_windows_are_self_consistent(fasta, patched_scorer):
    path, _ = fasta
    for w in (3, 5, 9, 25):
        out = saturation_mutagenesis(FakeOracle(), "fake", "chr1:30", ["t"],
                                     genome=path, window=w)
        assert len(out["ref_seq"]) == w == out["window"]
        assert len(out["positions"]) == w
