"""Sei's nucleosome-occupancy correction must match upstream, and must actually run.

Two independent defects meant the correction described in `SeiNormalizer`'s own docstring as
"recommended by Sei authors" was not applied at all:

1. **It was a no-op.** `normalize()` computed `sum_alt` from `preds_ref_adjust` instead of
   `preds_alt_adjust`, so `sum_alt == sum_ref`, so both scaling factors were **exactly 1.0**.
2. **It was never called.** `chorus/oracles/sei.py` built a `SeiNormalizer`, required it non-`None`
   in the not-loaded guard, and then never invoked it. Three references in the whole file: assign
   `None`, assign instance, null-check.

The reference implementation is `sc_hnorm_varianteffect` in FunctionLab/sei-framework `utils.py`:
both alleles' histone tracks are scaled so each sums to the pair's mean, then both are projected and
the difference taken. Measured on three real variants, applying it moves **every one of the 40
sequence classes** and changes the top-ranked class for 2 of 3 — for chr11:5247500 the correction is
larger than the signal (median |delta| 0.0181 vs median |effect| 0.0170).

Sei predicts 21,907 chromatin profiles of which **10,064 (46%) are histone tracks**, which is why a
global occupancy shift between two sequences is worth removing before projecting.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def normalizer(tmp_path):
    """A `SeiNormalizer` over a small synthetic histone index set.

    `sei_source/sei.py` imports torch at module scope, which the base `chorus` env does not have, so
    the numerical tests skip here and run in `chorus-sei`. `test_the_sum_is_not_taken_from_the_same
    _array_twice` below is the fast-suite guard for the specific regression and needs no torch.
    """
    pytest.importorskip("torch", reason="sei_source.sei imports torch at module scope")

    from chorus.oracles.sei_source.sei import SeiNormalizer

    inds = tmp_path / "histone_inds.npy"
    np.save(inds, np.arange(6))
    return SeiNormalizer(histone_inds=str(inds))


def _upstream(ref, alt, histone_inds):
    """`sc_hnorm_varianteffect` from FunctionLab/sei-framework utils.py:34, transcribed."""
    ref_adj, alt_adj = ref.copy(), alt.copy()
    s_ref = np.sum(ref[:, histone_inds], axis=1)
    s_alt = np.sum(alt[:, histone_inds], axis=1)
    ref_adj[:, histone_inds] = ref_adj[:, histone_inds] * (
        (s_ref * 0.5 + s_alt * 0.5) / s_ref)[:, None]
    alt_adj[:, histone_inds] = alt_adj[:, histone_inds] * (
        (s_ref * 0.5 + s_alt * 0.5) / s_alt)[:, None]
    return ref_adj, alt_adj


def test_it_matches_upstream_bit_for_bit_for_one_alt(normalizer):
    """The correctness anchor: identical to the published algorithm, not merely close."""
    rng = np.random.default_rng(11)
    ref = rng.random((3, 14)) + 1.0
    alt = rng.random((3, 14)) + 1.0

    got_ref, got_alt = normalizer.normalize(ref, alt)
    want_ref, want_alt = _upstream(ref, alt, normalizer.histone_inds)

    assert np.array_equal(got_ref, want_ref), (
        f"ref differs from upstream by up to {np.abs(got_ref - want_ref).max():.3e}"
    )
    assert np.array_equal(got_alt, want_alt), (
        f"alt differs from upstream by up to {np.abs(got_alt - want_alt).max():.3e}"
    )


def test_the_no_op_bug_cannot_come_back(normalizer):
    """Fails-without-fix. With the old `sum_alt` typo both factors were exactly 1.0.

    Asserted as "the correction actually changes the input", which is the property the typo removed.
    """
    rng = np.random.default_rng(3)
    ref = rng.random((2, 14)) + 1.0
    alt = ref.copy()
    alt[:, normalizer.histone_inds] *= 1.25          # a genuine occupancy difference

    got_ref, got_alt = normalizer.normalize(ref, alt)
    assert not np.allclose(got_ref, ref), (
        "the reference was returned unchanged, so the scaling factor was 1.0 — this is exactly the "
        "`sum_alt` typo, where sum_alt was computed from the ref array"
    )
    assert not np.allclose(got_alt, alt), "the alt was returned unchanged"

    # and the point of the correction: both alleles end on the same histone total
    h = normalizer.histone_inds
    assert np.allclose(np.sum(got_ref[:, h], axis=1), np.sum(got_alt[:, h], axis=1)), (
        "after normalisation the two alleles must carry equal histone totals; that is what removing "
        "the global occupancy shift means"
    )


def test_non_histone_tracks_are_left_alone(normalizer):
    """Only histone tracks are rescaled — 10,064 of Sei's 21,907 in the real model."""
    rng = np.random.default_rng(5)
    ref = rng.random((2, 14)) + 1.0
    alt = rng.random((2, 14)) + 1.0
    got_ref, got_alt = normalizer.normalize(ref, alt)

    other = [i for i in range(14) if i not in set(normalizer.histone_inds.tolist())]
    assert np.array_equal(got_ref[:, other], ref[:, other])
    assert np.array_equal(got_alt[:, other], alt[:, other])


def test_multi_allele_equalises_every_allele(normalizer):
    """chorus reports one prediction per allele, so the correction has to generalise.

    Upstream only ever returns `alt - ref` for a single pair and so never needs a shared reference.
    Equalising every allele to the common mean keeps chorus's schema identical to every other
    oracle's while reducing to upstream exactly when there is one alt (asserted above).
    """
    rng = np.random.default_rng(17)
    alleles = [rng.random((2, 14)) + 1.0 for _ in range(4)]     # ref + 3 alts
    out = normalizer.equalize(alleles)

    h = normalizer.histone_inds
    totals = [np.sum(o[:, h], axis=1) for o in out]
    for t in totals[1:]:
        assert np.allclose(totals[0], t), (
            f"alleles ended with different histone totals: {[float(x[0]) for x in totals]}"
        )


def test_equalize_of_two_is_the_pairwise_path(normalizer):
    """`normalize` is just `equalize` over two arrays; keep them from drifting apart."""
    rng = np.random.default_rng(23)
    ref = rng.random((2, 14)) + 1.0
    alt = rng.random((2, 14)) + 1.0

    a_ref, a_alt = normalizer.normalize(ref, alt)
    b_ref, b_alt = normalizer.equalize([ref, alt])
    assert np.array_equal(a_ref, b_ref) and np.array_equal(a_alt, b_alt)


def test_the_base_class_exposes_a_pairwise_hook():
    """The correction cannot be expressed per-sequence, so the base needs an allele-level seam.

    Asserted here because Sei's fix depends on it, and because the default must remain a plain
    per-allele loop — the other eight oracles must be unaffected.
    """
    import inspect

    from chorus.core.base import OracleBase

    assert hasattr(OracleBase, "_predict_alleles"), (
        "OracleBase lost _predict_alleles; Sei's pairwise nucleosome correction has nowhere to live"
    )
    src = inspect.getsource(OracleBase._predict_alleles)
    assert "self._predict(" in src, (
        "the default _predict_alleles no longer delegates to _predict per allele, which would change "
        "behaviour for the eight oracles that do not override it"
    )


def test_the_sum_is_not_taken_from_the_same_array_twice():
    """The fast-suite guard for the exact regression, readable without importing torch.

    The bug was one token: `sum_alt = np.sum(preds_ref_adjust[...])`. Both sums came from the ref
    array, so both scaling factors were exactly 1.0. Parsing the source catches that in the base env,
    where the numerical tests above cannot run at all.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "chorus" / "oracles" / "sei_source" / "sei.py"
    tree = ast.parse(src.read_text())
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "SeiNormalizer"), None)
    assert cls is not None, "SeiNormalizer has moved; this guard needs to follow it"

    body = ast.get_source_segment(src.read_text(), cls) or ""
    # The corrected implementation sums over a list comprehension of every allele; the buggy one
    # named preds_ref_adjust twice. Either way, the ref array must not be the source of both sums.
    assert body.count("np.sum(preds_ref_adjust") <= 1, (
        "SeiNormalizer computes two histone sums from `preds_ref_adjust`. That was the shipped bug: "
        "sum_alt came from the ref array, so sum_alt == sum_ref, so both scaling factors were "
        "exactly 1.0 and upstream's nucleosome correction became a no-op."
    )
    assert "equalize" in body, (
        "SeiNormalizer no longer exposes `equalize`; the multi-allele path chorus needs is gone"
    )
