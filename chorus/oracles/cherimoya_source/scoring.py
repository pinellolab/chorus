"""Shared scoring transforms for the Cherimoya oracle.

Both :mod:`chorus.oracles.cherimoya` and
``scripts/build_backgrounds_cherimoya.py`` import from here.  That is
deliberate and load-bearing: a background CDF is only meaningful if the
value it was built from is computed exactly the way ``oracle.predict()``
computes it at query time.  Chorus has already been bitten by the
opposite arrangement — the pre-0.4 ChromBPNet CDFs were built against a
different model variant than predictions used, and had to be rebuilt
(see ``audits/2026-04-29_chrombpnet_cdf_rebuild/``).  Keeping the
transform in one module makes that class of drift a type error rather
than a silent numerical bias.

The one place this deliberately diverges from ChromBPNet's arithmetic is
the count head.  Cherimoya's count head is trained against
``log(count + 1)`` (``cherimoya/losses.py``:
``count_sq_err = (log(y + 1) - y_hat)**2``), so the inverse is
:func:`numpy.expm1`, not ``exp``.  The error from using ``exp`` is exactly
``+1`` count — negligible at a peak (~0.1% at 1,000 counts) but up to
100% at a low-activity site, which is precisely the regime the activity
CDFs are built from.

ChromBPNet uses the *same* ``log(1 + count)`` parameterization
(``chrombpnet/training/data_generators/batchgen_generator.py`` feeds
``np.log(1+batch_cts.sum(-1, keepdims=True))`` as the count target), and
recovered its counts with ``np.exp`` until 2026-07-31 — a latent
``+1``-count bug, not a difference in the models.  It was self-consistent
there (oracle and CDF builder made the same error, so ChromBPNet's
percentiles stayed internally valid), which is why it went unnoticed.
Cherimoya did the correct thing rather than reproducing the bug for
symmetry, on the grounds that matching a bug would only have to be undone
when it was fixed — invalidating our CDFs.  It since was: see
``tests/test_chrombpnet_counts.py`` and the ``chrombpnet_pertrack.npz``
rebuild that followed.
"""

import math
from typing import Tuple

import numpy

from .catv1_globals import (
    CATV1_OUTPUT_LENGTH,
    CATV1_SCORING_WINDOW_BP,
)

# Pseudocount for the log2 fold-change effect formula.  Matches
# LAYER_CONFIGS['chromatin_accessibility'].pseudocount in
# chorus/analysis/scorers.py and PSEUDOCOUNT in the ChromBPNet builder.
PSEUDOCOUNT = 1.0


def expected_counts_profile(
    profile_logits: numpy.ndarray,
    log_counts: numpy.ndarray,
) -> numpy.ndarray:
    """Combine Cherimoya's two output heads into expected counts per bp.

    Mean-centres the profile logits, softmaxes them across positions to
    get a shape distribution, and scales by the predicted total count.

    Args:
        profile_logits: ``(N, 1, L)`` or ``(N, L)`` profile logits.
        log_counts: ``(N, 1)`` or ``(N,)`` predicted ``log(count + 1)``.

    Returns:
        ``(N, L)`` expected counts per base pair.
    """
    logits = numpy.asarray(profile_logits, dtype=numpy.float64)
    if logits.ndim == 3:
        if logits.shape[1] != 1:
            raise ValueError(
                f"CATv1 models have a single output track; got profile "
                f"logits with shape {logits.shape}."
            )
        logits = logits[:, 0, :]
    elif logits.ndim != 2:
        raise ValueError(f"Unsupported profile logits shape {logits.shape}.")

    counts = numpy.asarray(log_counts, dtype=numpy.float64)
    if counts.ndim == 2:
        counts = counts[:, 0]
    elif counts.ndim != 1:
        raise ValueError(f"Unsupported log-count shape {counts.shape}.")

    # Mean-centre before exponentiating for numerical stability; this is
    # a no-op on the resulting softmax.
    centred = logits - logits.mean(axis=1, keepdims=True)
    exp_centred = numpy.exp(centred)
    probs = exp_centred / exp_centred.sum(axis=1, keepdims=True)

    # expm1, NOT exp -- the count head predicts log(count + 1).
    total = numpy.expm1(counts)

    return probs * total[:, None]


def heads_equivalent_to_profile(
    profile: numpy.ndarray,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Invert :func:`expected_counts_profile`: profile -> ``(logits, log_counts)``.

    Why this exists. CATv1 ships five cross-validation folds per experiment, and
    its model card says to "average the predictions of all five folds for a more
    robust estimate". The prediction is the **expected-counts profile**, and
    :func:`expected_counts_profile` is non-linear in both heads (softmax over
    positions, ``expm1`` over the count head), so averaging the raw heads across
    folds is *not* averaging the predictions. The mean has to be taken on the
    output.

    But every caller downstream of ``_forward_windows`` expects the two-head
    shape. Rather than fork the call path, this maps an already-averaged profile
    back onto the unique heads that reproduce it exactly:

        logits     = log(P / T)        so softmax(logits) == P / T
        log_counts = log1p(T)          so expm1(log_counts) == T

    and therefore ``expected_counts_profile(*heads_equivalent_to_profile(P)) == P``
    to floating-point round-trip. The mean-centring inside
    :func:`expected_counts_profile` is a no-op on the softmax, so the additive
    freedom in ``logits`` does not matter.

    Args:
        profile: ``(N, L)`` expected counts per base pair.

    Returns:
        ``(profile_logits, log_counts)`` shaped ``(N, 1, L)`` and ``(N, 1)``,
        matching what a real forward pass returns.
    """
    p = numpy.asarray(profile, dtype=numpy.float64)
    if p.ndim != 2:
        raise ValueError(f"Expected an (N, L) profile, got shape {p.shape}.")

    total = p.sum(axis=1)
    # A window whose expected counts are all zero has no shape to recover. Emit
    # a uniform distribution scaled to zero total, which round-trips to zeros
    # rather than producing nan from log(0)/0.
    safe_total = numpy.where(total > 0, total, 1.0)
    probs = p / safe_total[:, None]
    probs = numpy.where(total[:, None] > 0, probs, 1.0 / p.shape[1])

    with numpy.errstate(divide="ignore"):
        logits = numpy.log(probs)

    # log(0) is -inf, which softmax would map to 0 correctly -- but
    # expected_counts_profile MEAN-CENTRES before exponentiating, so a single
    # absolute floor blows up. With a floor of -745 and a maximally peaked profile
    # (one non-zero bin) the row mean is about -743, centring lifts the peak to
    # +743, and exp overflows to inf: the round-trip returns nan for a profile that
    # is merely sparse. Found by tests/test_cherimoya_ensemble.py, which feeds
    # exactly that shape.
    #
    # So floor RELATIVE to the row's own maximum. exp(-60) is 9e-27, which softmax
    # sends to zero for any realistic profile length, while bounding the centred
    # spread to ~60 and keeping exp safe whatever the sparsity.
    finite_max = numpy.where(numpy.isfinite(logits), logits, -numpy.inf).max(
        axis=1, keepdims=True)
    floor = numpy.where(numpy.isfinite(finite_max), finite_max - 60.0, 0.0)
    logits = numpy.where(numpy.isfinite(logits), logits, numpy.broadcast_to(floor, logits.shape))

    log_counts = numpy.log1p(total)
    return logits[:, None, :], log_counts[:, None]


def score_window_sum(
    profile: numpy.ndarray,
    output_length: int = CATV1_OUTPUT_LENGTH,
    window_bp: int = CATV1_SCORING_WINDOW_BP,
) -> float:
    """Sum a profile over the central ``window_bp`` bases.

    Reproduces ``score_window_sum`` in
    ``scripts/build_backgrounds_chrombpnet.py`` exactly, including its
    centring: the window is centred on ``output_length // 2``, i.e. the
    midpoint of the output window rather than on the variant.  For a
    variant-centred 2114 bp input the substituted base actually lands at
    output index 499, one to the left of the window's centre at 500.
    That off-by-one is inherited on purpose -- matching ChromBPNet's
    background construction is what makes the percentiles comparable, and
    a 1 bp shift inside a 501 bp sum is numerically irrelevant.

    Args:
        profile: ``(L,)`` expected counts per base pair.
        output_length: Model output length in bp.
        window_bp: Width of the central window.

    Returns:
        The window sum.
    """
    centre = output_length // 2
    half = window_bp // 2
    start = max(0, centre - half)
    end = min(output_length, centre + half + 1)
    return float(numpy.sum(profile[start:end]))


def compute_effect(
    ref_value: float,
    alt_value: float,
    pseudocount: float = PSEUDOCOUNT,
) -> float:
    """Signed log2 fold-change of alt versus ref.

    Matches ``compute_effect`` in the ChromBPNet builder and the
    ``log2fc`` branch of ``chorus.analysis.scorers._compute_effect``.
    Callers building the effect CDF take the absolute value, since
    ``chromatin_accessibility`` is an unsigned layer.

    Args:
        ref_value: Reference-allele window sum.
        alt_value: Alternate-allele window sum.
        pseudocount: Added to both sides. Default 1.0.

    Returns:
        ``log2((alt + pc) / (ref + pc))``.
    """
    return math.log2((alt_value + pseudocount) / (ref_value + pseudocount))
