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
