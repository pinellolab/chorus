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

**UPDATED 2026-08-09.** The failure-handling section below used to pin a defect it
described rather than fixed: every failed substitution was recorded as ``0.0``, so a
sweep in which *nothing* worked returned a fully-populated all-zero motif profile with
no error field. Measured on LegNet with an assay id it does not carry —
``scores [[0,0,0,0],[0,0,0,0],[0,0,0,0]]``, ``importance [-0.0,-0.0,-0.0]`` — while
``predict_variant_effect`` raised ``InvalidAssayError`` on the same arguments. "Not
scored" is now ``None`` (never ``0.0``, which is a real score), the payload carries
``n_attempted``/``n_scored``/``n_failed``/``first_error``, and a sweep that scores
nothing returns an error dict instead of a matrix. A successful sweep is unchanged:
the same LegNet window gives bit-identical importance before and after.
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

    def __init__(self, effect_fn=None, fail_at=(), fail_if=None, empty_at=()):
        self.calls = []
        self.effect_fn = effect_fn or (lambda c, p, r, a: 1.0)
        self.fail_at = set(fail_at)
        # fail_if lets a test kill ONE substitution at a position rather than all
        # three, which is what separates "this cell failed" from "this site failed".
        self.fail_if = fail_if or (lambda c, p, r, a: False)
        self.empty_at = set(empty_at)

    def predict_variant_effect(self, region, variant_position, alleles,
                               assay_ids=None, genome=None):
        chrom, pos_s = variant_position.split(":")
        pos = int(pos_s)
        ref, alt = alleles
        self.calls.append((chrom, pos, ref, alt))
        if pos in self.fail_at or self.fail_if(chrom, pos, ref, alt):
            raise RuntimeError(f"synthetic failure at {pos}")
        if pos in self.empty_at:
            return {"_effect": None}       # patched_scorer turns this into no effects
        return {"_effect": self.effect_fn(chrom, pos, ref, alt)}


@pytest.fixture
def patched_scorer(monkeypatch):
    """Route ``_score_all_tracks`` to the fake's recorded effect."""
    class _TE:
        def __init__(self, v): self.raw_score = v

    import chorus.analysis.discovery as disc
    monkeypatch.setattr(
        disc, "_score_all_tracks",
        lambda vr, oracle_name: [] if vr["_effect"] is None else [_TE(vr["_effect"])])
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
    scores = out["scores"]
    idx = out["positions"].index(30)
    ref_j = BASES.index(out["ref_seq"][idx])
    # the failed substitutions are null -- NOT 0.0, which is a score
    assert [v for j, v in enumerate(scores[idx]) if j != ref_j] == [None, None, None]
    assert scores[idx][ref_j] == 0.0            # the reference base, as everywhere
    assert out["importance"][idx] is None
    # every other position is unaffected
    others = [i for i in range(len(scores)) if i != idx]
    assert np.allclose([out["importance"][i] for i in others], 2.0)
    assert (out["n_attempted"], out["n_scored"], out["n_failed"]) == (15, 12, 3)
    assert "synthetic failure at 30" in out["first_error"]


def test_a_failed_site_is_distinguishable_from_an_unimportant_one(fasta,
                                                                 patched_scorer):
    """The inverse of what this test used to assert, which is the point.

    It previously documented the limitation -- a site whose predictions all raised
    got importance 0.0, exactly like a site the oracle genuinely does not care
    about -- and told the next reader to "tighten this test" if a field reporting
    failed sites was ever added. It has been: failures are null, and the counts
    say how many.
    """
    path, _ = fasta
    failed = FakeOracle(effect_fn=lambda c, p, r, a: -2.0, fail_at={30})
    flat = FakeOracle(effect_fn=lambda c, p, r, a: 0.0)
    a = saturation_mutagenesis(failed, "fake", "chr1:30", ["t"],
                               genome=path, window=5)
    b = saturation_mutagenesis(flat, "fake", "chr1:30", ["t"],
                               genome=path, window=5)
    i = a["positions"].index(30)
    assert a["importance"][i] is None, "a failed site must not report a number"
    assert b["importance"][i] == 0.0, "a genuinely flat site still reports 0.0"
    assert (a["n_failed"], b["n_failed"]) == (3, 0)
    assert a["first_error"] is not None and b["first_error"] is None


def test_a_partly_failed_position_averages_only_what_scored(fasta, patched_scorer):
    """One dead substitution costs precision at that position, not the position.

    Averaging over 3 when only 2 scored would drag the site towards zero, i.e.
    towards "unimportant", which is the same lie the all-zero fill told.
    """
    path, _ = fasta
    # chr1:30 is a C in this fixture, so G is one of its three substitutions
    oracle = FakeOracle(effect_fn=lambda c, p, r, a: -2.0,
                        fail_if=lambda c, p, r, a: p == 30 and a == "G")
    out = saturation_mutagenesis(oracle, "fake", "chr1:30", ["t"],
                                 genome=path, window=5)
    i = out["positions"].index(30)
    assert out["ref_seq"][i] == "C"
    assert out["scores"][i][BASES.index("G")] is None
    assert out["importance"][i] == pytest.approx(2.0)     # mean of the two that scored
    assert (out["n_attempted"], out["n_scored"], out["n_failed"]) == (15, 14, 1)


def test_a_scorer_that_returns_nothing_is_a_failure_not_a_zero(fasta, patched_scorer):
    """``_score_all_tracks`` returning ``[]`` used to be recorded as 0.0 and counted
    as a success, so a track the scorer cannot see looked like a track with no effect.
    """
    path, _ = fasta
    oracle = FakeOracle(effect_fn=lambda c, p, r, a: -2.0, empty_at={30})
    out = saturation_mutagenesis(oracle, "fake", "chr1:30", ["t"],
                                 genome=path, window=5)
    i = out["positions"].index(30)
    assert out["importance"][i] is None
    assert out["n_failed"] == 3
    assert "no track effect returned" in out["first_error"]


def test_a_sweep_that_scores_nothing_returns_an_error_not_a_zero_matrix(fasta,
                                                                       patched_scorer):
    """The headline defect: a bogus track id gave a fully-populated all-zero profile.

    Measured on LegNet before the fix, with an assay id the oracle does not carry:
    ``scores [[0,0,0,0],[0,0,0,0],[0,0,0,0]]``, ``importance [-0.0,-0.0,-0.0]``, no
    error field -- while ``predict_variant_effect`` raised ``InvalidAssayError`` on
    the same arguments. The error dict deliberately carries no ``scores``/
    ``importance`` key, so a caller that ignores ``error`` raises instead of
    plotting zeros.
    """
    path, _ = fasta
    oracle = FakeOracle(fail_at=set(range(20, 40)))
    out = saturation_mutagenesis(oracle, "fake", "chr1:30", ["t"],
                                 genome=path, window=3)
    assert "scores" not in out and "importance" not in out
    assert out["error_type"] == "RuntimeError"      # what the failure actually was
    assert "scored 0 of 9 substitutions" in out["error"]
    assert out["n_attempted"] == 9 and out["n_scored"] == 0 and out["n_failed"] == 9
    assert "synthetic failure" in out["first_error"]
    # the context needed to retry is still there
    assert out["assay_id"] == "t" and out["chrom"] == "chr1" and out["window"] == 3


def test_the_payload_survives_strict_json(fasta, patched_scorer):
    """MCP serialises with ``pydantic_core.to_json``, which writes a bare ``NaN``.

    Every JavaScript client's ``JSON.parse`` rejects that, so "not scored" travels
    as ``null`` and no NaN may reach the payload.
    """
    import json
    import pydantic_core

    path, _ = fasta
    oracle = FakeOracle(effect_fn=lambda c, p, r, a: -2.0, fail_at={30})
    out = saturation_mutagenesis(oracle, "fake", "chr1:30", ["t"],
                                 genome=path, window=5)
    encoded = pydantic_core.to_json(out).decode()
    assert "NaN" not in encoded
    json.loads(encoded, parse_constant=lambda c: (_ for _ in ()).throw(
        ValueError(f"non-JSON constant {c!r} in payload")))


def test_a_non_acgt_reference_base_is_skipped_entirely(tmp_path, patched_scorer):
    import pyfaidx
    p = tmp_path / "n.fa"
    p.write_text(">chr1\n" + ("ACGTN" * 12) + "\n")
    pyfaidx.Fasta(str(p))
    oracle = FakeOracle()
    out = saturation_mutagenesis(oracle, "fake", "chr1:30", ["t"],
                                 genome=str(p), window=11)
    for i, b in enumerate(out["ref_seq"]):
        if b == "N":
            # never scored, so null -- 0.0 here would read as "no effect" too
            assert out["scores"][i] == [None] * 4
            assert out["importance"][i] is None
    assert all(r in "ACGT" for _, _, r, _ in oracle.calls)
    # an N contributes no attempt, so the counts describe real work only
    n_acgt = sum(1 for b in out["ref_seq"] if b in BASES)
    assert out["n_attempted"] == out["n_scored"] == n_acgt * 3


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
