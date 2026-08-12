"""The CATv1 5-fold ensemble: where the mean is taken, and that both sides agree.

Two files cite this module as the guard for those facts
(``cherimoya_source/catv1_globals.py`` and ``scripts/build_backgrounds_cherimoya.py``)
and it did not exist. That is worse than having no guard, because the citation
tells the next reader the invariant is already protected.

What actually needs pinning, and why each one bit:

1. **The mean is over expected-counts PROFILES.** Both CATv1 heads enter
   :func:`expected_counts_profile` non-linearly — softmax across positions, and
   ``expm1`` on the count head — so averaging the raw heads across folds computes
   a different quantity, and averaging per-fold log2FCs computes a third. Measured
   at rs12740374/ENCSR149XIL: averaging predictions gives log2FC **1.4576**;
   averaging per-fold effects gives **1.4849**. Only the first is what the model
   card describes ("average the predictions of all five folds").

2. **The builder and the query path must use the same statistic.** The builder
   scored ``model(batch)`` directly, bypassing the oracle's dispatch, so an
   ensemble build would have produced a fold-0 null under an ensemble query path —
   a null and a numerator that are not the same quantity, which makes every
   percentile from it meaningless. ``forward_window_sums`` takes a *list* now.

3. **Dispatch must work with ``use_environment=True``**, which is the user
   default. The first implementation keyed off ``self._models``, which only the
   in-process loader populates, so an ensemble request in env mode silently
   returned fold 0 alone.

The tests that need weights are marked ``integration`` and skip cleanly without
the ``chorus-cherimoya`` env; the arithmetic ones run anywhere.
"""
from __future__ import annotations

import numpy as np
import pytest

from chorus.oracles.cherimoya_source.catv1_globals import (
    CATV1_DEFAULT_FOLD,
    CATV1_ENSEMBLE,
    CATV1_N_FOLDS,
)
from chorus.oracles.cherimoya_source.scoring import (
    expected_counts_profile,
    heads_equivalent_to_profile,
)


# ---------------------------------------------------------------------------
# The default, and the builder/oracle agreement on it
# ---------------------------------------------------------------------------


def test_the_shipped_default_is_fold_0_and_the_ensemble_is_opt_in():
    """Reversed on 2026-08-11, with CATv1's author.

    This used to assert the ensemble was the default. It is fold 0 now, so Cherimoya's
    scores are comparable with ChromBPNet -- which also defaults to fold 0, and whose null is
    built on the same reference sets (both reproduce effect_counts=18672 and
    summary_counts=34004). That comparison is the point of the cross-oracle report, and
    jmschrei's view was that five models complicate and slow most analyses, so fold 0 is the
    right default for an interactive tool.

    The ensemble is still reachable and still has its own null; see
    tests/test_fold_selects_its_own_null.py for the selection and mismatch guards.
    """
    assert CATV1_DEFAULT_FOLD == 0
    assert CATV1_N_FOLDS == 5
    assert CATV1_ENSEMBLE == "ensemble", "the opt-in sentinel must still exist"


def test_the_builder_defaults_to_the_oracle_default_not_a_literal():
    """A literal 0 here is how a null and a query path drift apart.

    The builder's ``--fold`` default must BE ``CATV1_DEFAULT_FOLD``, so changing the
    oracle's default cannot leave the builder behind.
    """
    src = (
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "scripts" / "build_backgrounds_cherimoya.py"
    ).read_text()
    assert "default=CATV1_DEFAULT_FOLD" in src, (
        "build_backgrounds_cherimoya.py's --fold default is not tied to "
        "CATV1_DEFAULT_FOLD, so the null can be built on a different fold than the "
        "query path uses"
    )


def test_forward_window_sums_takes_a_list_of_models():
    """Plural by design: a singular parameter is what let the builder score fold 0."""
    import inspect
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "scripts" / "build_backgrounds_cherimoya.py").read_text()
    sig = src[src.index("def forward_window_sums("):]
    sig = sig[:sig.index(")")]
    assert "models" in sig, (
        f"forward_window_sums signature is {sig!r} -- it must take a LIST, or an "
        f"ensemble build silently scores one fold"
    )


# ---------------------------------------------------------------------------
# Where the mean is taken. Pure arithmetic, no weights needed.
# ---------------------------------------------------------------------------


def _fake_heads(rng, n=3, L=1000):
    logits = rng.normal(size=(n, 1, L)) * 2.0
    log_counts = rng.uniform(4.0, 8.0, size=(n, 1))
    return logits, log_counts


def test_the_profile_inverse_round_trips():
    """``heads_equivalent_to_profile`` is what lets the mean be taken on the output
    while every caller keeps the two-head contract."""
    rng = np.random.default_rng(0)
    for scale in (1.0, 1e-3, 1e3):
        P = np.abs(rng.normal(size=(4, 1000))) ** 3 * scale
        back = expected_counts_profile(*heads_equivalent_to_profile(P))
        assert np.isfinite(back).all()
        assert np.allclose(back, P, rtol=1e-12, atol=0), np.abs(back - P).max()


def test_a_zero_profile_round_trips_to_zero_rather_than_nan():
    """A window with no predicted signal has no shape to recover; log(0) must not
    become nan and poison the average."""
    P = np.zeros((2, 500))
    P[1, 250] = 5.0
    back = expected_counts_profile(*heads_equivalent_to_profile(P))
    assert np.isfinite(back).all()
    assert np.allclose(back, P, atol=1e-12)


def test_averaging_profiles_differs_from_averaging_heads():
    """The distinction the implementation rests on, demonstrated numerically.

    If these were equal, averaging heads would be a legitimate shortcut and the
    inverse helper would be unnecessary. They are not equal, because the count head
    passes through ``expm1``.
    """
    rng = np.random.default_rng(7)
    per_fold = [_fake_heads(rng) for _ in range(CATV1_N_FOLDS)]

    profiles = [expected_counts_profile(lg, lc) for lg, lc in per_fold]
    mean_of_profiles = np.mean(profiles, axis=0)

    mean_logits = np.mean([lg for lg, _ in per_fold], axis=0)
    mean_counts = np.mean([lc for _, lc in per_fold], axis=0)
    profile_of_mean_heads = expected_counts_profile(mean_logits, mean_counts)

    rel = np.abs(mean_of_profiles - profile_of_mean_heads) / np.maximum(mean_of_profiles, 1e-12)
    assert rel.max() > 1e-3, (
        "averaging heads and averaging predictions agree here, which would make the "
        "profile-space mean pointless -- check the fixture, not the claim"
    )


def test_averaging_predictions_differs_from_averaging_per_fold_effects():
    """The third option, also not equivalent. A log ratio is not linear.

    This is the one that produced 1.4576 vs 1.4849 on the real variant.
    """
    rng = np.random.default_rng(11)
    # Per-fold (ref, alt) window sums with realistic spread.
    refs = np.array([603.3, 1201.5, 880.2, 747.0, 482.5])
    alts = np.array([2093.2, 2875.7, 2390.6, 2065.5, 1335.7])

    ens = np.log2(alts.mean() / refs.mean())          # average the predictions
    per_fold_mean = np.log2(alts / refs).mean()       # average the effects
    assert abs(ens - per_fold_mean) > 1e-2, (
        f"the two aggregations agree ({ens:.4f} vs {per_fold_mean:.4f}); the real "
        f"data gives 1.4576 vs 1.4849, so a fixture that agrees is wrong"
    )
    # And the shipped choice is the first one, to 3 dp of the measured value.
    assert abs(ens - 1.4576) < 5e-3, f"expected the model-card aggregation, got {ens:.4f}"


# ---------------------------------------------------------------------------
# The shipped artefact records which checkpoints built it
# ---------------------------------------------------------------------------


def test_the_shipped_null_records_the_fold_it_was_built_from():
    """Provenance must answer "which fold?" -- the folds disagree by 2.02x on the same
    sequence, so a null built on the wrong one produces plausible wrong percentiles.

    The file under the plain name must be the DEFAULT fold's null; the ensemble's lives at
    cherimoya_ensemble_pertrack.npz. The load-time guard enforces the pairing, and this test
    checks the artefact on disk actually satisfies it."""
    import json

    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR

    path = CHORUS_BACKGROUNDS_DIR / "cherimoya_pertrack.npz"
    if not path.exists():
        pytest.skip("no downloaded cherimoya background")
    with np.load(path, allow_pickle=True) as d:
        assert "build_config" in d.files, "cherimoya ships no build_config"
        cfg = json.loads(str(d["build_config"][0]))
    assert cfg.get("fold") == CATV1_DEFAULT_FOLD, (
        f"the shipped cherimoya null records fold={cfg.get('fold')!r}. If it really was "
        f"built from one fold, the oracle default must match it; if it was built from "
        f"five, the stamper dropped the field (it replaces build_config wholesale)."
    )


# ---------------------------------------------------------------------------
# Both execution modes. Needs weights.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_the_ensemble_loads_five_checkpoints_and_dispatches_in_both_modes():
    """``use_environment=True`` is the USER DEFAULT and was the mode that silently
    fell back to fold 0."""
    pytest.importorskip("torch")
    try:
        from chorus.oracles.cherimoya import CherimoyaOracle
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"cherimoya not importable here: {exc}")

    from chorus.core.globals import CHORUS_DATA_DIR
    genome = CHORUS_DATA_DIR / "genomes" / "hg38.fa"
    if not genome.exists():
        pytest.skip("hg38.fa missing")

    for use_env in (False, True):
        oracle = CherimoyaOracle(use_environment=use_env, reference_fasta=str(genome))
        try:
            oracle.load_pretrained_model(encode_id="ENCSR149XIL")
        except Exception as exc:
            pytest.skip(f"cherimoya weights unavailable ({type(exc).__name__})")
        assert len(oracle.model_paths) == CATV1_N_FOLDS, (
            f"use_environment={use_env} loaded {len(oracle.model_paths)} checkpoint(s); "
            f"the ensemble needs {CATV1_N_FOLDS}. Dispatch keys off model_paths, which "
            f"BOTH modes must populate."
        )
        assert oracle.fold == CATV1_ENSEMBLE
