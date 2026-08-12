"""One implementation of the BPNet-family count-head arithmetic (#125).

Every model in the BPNet family — ChromBPNet, BPNet CHIP, CATv1/Cherimoya — emits two
heads: a **profile** of per-position logits and a **count** scalar in log space. Turning
them into a per-position expected-count track is four operations:

    centre the logits -> softmax -> invert the count head -> scale

Four operations, and this project has had them wrong three separate times, each time
because a second copy of the arithmetic disagreed with the first:

* ``exp`` vs ``expm1`` on the count head, across four call sites, fixed one at a time
  because nothing compared them. The error is +1 read, negligible at a peak (~0.1% at
  1,000 counts) and up to **100%** at a quiet site — which is exactly the regime the
  activity CDFs are built from.
* per-strand softmax vs one joint softmax over the flattened both-strand vector. Scaling
  each strand by the full total made the two emitted tracks together claim **2.00x** the
  counts the model predicted, measured on BPNet ``CHIP:K562:REST``. Summing the strands'
  *logits* first — what the background builder did — is a third quantity again,
  corresponding to no observable the oracle emits, and it drifted 0.98-1.30x
  sequence-dependently, so it could not be reconciled by rescaling.
* a count bias hardcoded ``(N, 1)`` where the model declares ``(None, 2)``, which Keras
  silently broadcast, shifting every predicted log-count by a constant 0.5885 — 1.80x too
  low at a peak, 3.04x at a quiet site.

None of those was a crash. Each produced a plausible number, and each shipped.

**THE COUNT HEAD HAS THREE CONVENTIONS AND THEY ARE NOT INTERCHANGEABLE.** This is the
part worth reading before touching anything here:

    n_tracks=1   log1p per track            total = expm1(C)        ChromBPNet ATAC/DNASE,
                                                                    CATv1 (42 + 1,518 tracks)
    n_tracks=2   log1p per track, then      total = exp(C) - 2      BPNet CHIP (744 models)
                 pooled with logsumexp
    log10        a different model family    total = 10 ** C         EPInformer-seq

The first two are what this module handles, and the second is why there is an ``n_tracks``
argument rather than a bare ``expm1``: bpnet-refactor's generator stores a per-track
``log(sum_positions + 1)`` and its count loss pools a task's tracks with
``reduce_logsumexp``, so the trained target is ``C = log(sum_t (1 + c_t)) =
log(n_tracks + c_total)`` and the inverse is ``exp(C) - n_tracks``. For single-track models
that is ``expm1(C)`` exactly; ``expm1`` is kept for those because it is more accurate for
small C. Upstream's own ``bpnet/cli/predict.py`` uses a bare ``exp(C)``, i.e. it is off by
``+n_tracks`` in the other direction — the target construction is the authority, not that
CLI.

**EPInformer-seq is deliberately not routed through here.** It scales by ``10 ** log_count``,
not ``expm1``, because it is a different model trained on a different target. Unifying it
would be the same class of mistake as the three above, in the opposite direction, so
``tests/test_count_head_copies_agree.py`` pins the distinction rather than leaving it to be
noticed.

The torch path in ``scripts/build_backgrounds_cherimoya.py`` also stays where it is, on
purpose: it runs on the accelerator inside the batch loop, and routing it through numpy
would force a device round-trip per batch on a job measured in hours. What makes that safe
is not a comment, it is the test — it asserts the torch expression and this module agree to
float32 precision on the same inputs.
"""
from __future__ import annotations

import numpy as np


def counts_from_log(log_counts, n_tracks: int = 1):
    """Invert the count head: log-space prediction -> expected total counts.

    *n_tracks* is the number of tracks the count loss pooled, **not** a shape: 1 for the
    single-track ATAC/DNASE and CATv1 models, 2 for the two-strand BPNet CHIP models. See
    the module docstring for why that number appears in the inverse at all.
    """
    counts = np.asarray(log_counts)
    if n_tracks == 1:
        # Bit-identical to exp(C) - 1, and more accurate for small C.
        return np.expm1(counts)
    return np.exp(counts) - n_tracks


def _as_float(values):
    """Array of *values*, **preserving** a floating dtype rather than promoting it.

    This is deliberate and it is the difference between a refactor and a silent numerical
    change. The two call sites this function unified disagreed about precision: Cherimoya's
    copy cast to float64 before doing anything, ChromBPNet's did not and therefore ran the
    softmax in whatever TensorFlow handed it (float32). Forcing float64 here would move every
    ChromBPNet number in the last bits; forcing float32 would move every Cherimoya one. So
    the caller keeps its own precision, and Cherimoya's wrapper still casts first, which
    leaves both paths bit-identical to what they computed before #125.

    Whether ChromBPNet *should* run this in float64 is a real question — a softmax over 1,000
    to 2,000 bins accumulates — but it is a numerical decision with its own regeneration, not
    something to slip into a refactor.
    """
    array = np.asarray(values)
    return array if array.dtype.kind == "f" else array.astype(np.float64)


def joint_softmax(logits):
    """Softmax over every axis but the batch, mean-centred first for stability.

    Joint rather than per-strand, which is the distinction that cost 2.00x: the profile
    multinomial is trained over the flattened both-strand vector, so the strands share one
    distribution and together sum to the predicted total. Centring is a no-op on the result
    and only prevents ``exp`` overflowing.
    """
    values = _as_float(logits)
    flat = values.reshape(values.shape[0], -1)
    centred = flat - np.mean(flat, axis=1, keepdims=True)
    exponentiated = np.exp(centred)
    return (exponentiated / np.sum(exponentiated, axis=1, keepdims=True)).reshape(values.shape)


def expected_counts_profile(logits, log_counts, n_tracks: int | None = None):
    """Per-position expected counts from a profile head and a count head.

    ``logits`` is ``(batch, positions)`` or ``(batch, positions, tracks)``; ``log_counts``
    is ``(batch,)`` or ``(batch, 1)``. The returned array has the shape of *logits*.

    *n_tracks* defaults to the trailing dimension of *logits* — the two-strand CHIP models
    pass ``(B, L, 2)`` and need ``exp(C) - 2``, single-track models pass ``(B, L)`` and get
    ``expm1``. Pass it explicitly when the shape does not carry that meaning.
    """
    values = _as_float(logits)
    if n_tracks is None:
        n_tracks = values.shape[2] if values.ndim == 3 else 1

    counts = _as_float(log_counts)
    if counts.ndim == 2:
        counts = counts[:, 0]
    elif counts.ndim != 1:
        raise ValueError(f"Unsupported log-count shape {counts.shape}.")

    totals = counts_from_log(counts, n_tracks)
    probs = joint_softmax(values)
    scale = totals.reshape((-1,) + (1,) * (values.ndim - 1))
    return probs * scale
