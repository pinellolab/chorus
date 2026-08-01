"""One implementation of the background-sampling primitives, shared by every builder.

Each of the eight ``scripts/build_backgrounds_*.py`` scripts carried its own copy
of these — 8 reservoir samplers, 7 ``get_sequence``\\ s, 5 ``compute_effect``\\ s,
6 one-hot encoders. Nothing compared them, so a fix in one never reached the
others. All three ChromBPNet CHIP defects fixed on 2026-07-31 (#113, #120) were
two copies of the same arithmetic disagreeing:

* ``exp`` vs ``expm1`` on the count head — four call sites, fixed one at a time;
* per-strand softmax vs one joint softmax over the flattened both-strand vector;
* a hardcoded ``(N,1)`` count bias against a declared ``(None,2)`` input, which
  Keras silently *broadcasts* rather than rejecting, shifting every predicted
  log-count by a constant 0.5885.

The precedent this follows is ``chorus/oracles/cherimoya_source/scoring.py``,
imported by both the Cherimoya oracle and its builder so that, in its own words,
"that class of drift [is] a type error rather than a silent numerical bias".

WHAT AN AUDIT OF THE COPIES ACTUALLY FOUND
------------------------------------------
Comparing every copy by AST rather than by eye: the vast majority of the
differences are cosmetic — a docstring, a type annotation, ``np`` vs ``numpy``,
``indices`` vs ``idx``. ``to_cdf_matrix`` looked like four distinct
implementations and is **numerically identical** in all four. ``add`` looked like
three and is identical in all three.

Only **four** genuine behavioural divergences exist across ~30 duplicated
definitions, and every one of them is preserved here as a *parameter* rather than
unified away, because each may be deliberate:

===========================  ==============================================
divergence                   preserved as
===========================  ==============================================
AlphaGenome capacity 20,000  ``ReservoirSampler(capacity=...)``, default 50,000
LegNet N-threshold 0.3       ``get_sequence(max_n_fraction=...)``, default 0.5
log2fc / logfc / diff        ``compute_effect(formula=...)``
EPInformer-seq ``(4, L)``    ``one_hot_encode(channels_first=...)``
===========================  ==============================================

Unifying any of those silently would change a shipped background. Whether
LegNet's stricter 0.3 is intentional for a 200 bp window is a question for its
owner, not something to settle by picking the majority value.

See https://github.com/pinellolab/chorus/issues/125.
"""

from __future__ import annotations

import math
import random

import numpy as np

# Reservoir capacity. AlphaGenome uses 20,000; every other builder 50,000.
DEFAULT_CAPACITY = 50_000
# Stored CDF width. Every builder agrees on this, and
# PerTrackNormalizer._get_denominator depends on it — the percentile denominator
# is the grid width, not the sample count (see #119).
DEFAULT_CDF_POINTS = 10_000
# Fraction of a sampled window that may be N before it is rejected. LegNet
# uses 0.3 over its 200 bp window; every other builder 0.5.
DEFAULT_MAX_N_FRACTION = 0.5
# Added to both sides of a fold-change so a zero-signal window is finite.
DEFAULT_PSEUDOCOUNT = 1.0
# Seeded so a rebuild is reproducible. Every builder used 12345 for the
# reservoir; kept rather than randomised.
DEFAULT_SEED = 12345

_BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


class ReservoirSampler:
    """Bounded uniform sample of an unbounded stream, per track.

    Algorithm R: keep the first ``capacity`` values, then replace a uniformly
    chosen slot with decreasing probability. ``counts`` records the *true*
    number of values offered, not the number retained — which is what
    ``*_counts`` in the shipped NPZ means, and why a track can report 18,672
    samples while holding 18,672 or fewer.

    Byte-identical to the implementation in
    ``scripts/build_backgrounds_chrombpnet.py``, which is the most audited copy.
    """

    def __init__(
        self,
        n_tracks: int,
        capacity: int = DEFAULT_CAPACITY,
        seed: int = DEFAULT_SEED,
    ):
        self.n_tracks = n_tracks
        self.capacity = capacity
        self.data: list[list[float]] = [[] for _ in range(n_tracks)]
        self.counts = np.zeros(n_tracks, dtype=np.int64)
        self._rng = random.Random(seed)

    def add(self, track_idx: int, value: float) -> None:
        n = self.counts[track_idx]
        if n < self.capacity:
            self.data[track_idx].append(value)
        else:
            j = self._rng.randint(0, n)
            if j < self.capacity:
                self.data[track_idx][j] = value
        self.counts[track_idx] += 1

    def add_batch(self, track_idx: int, values) -> None:
        """Offer many values to one track.

        Deliberately a plain loop over :meth:`add` rather than a vectorised
        reimplementation, because a different traversal order changes *which*
        samples survive once the reservoir is full — the CDF would move without
        any arithmetic changing.

        AlphaGenome's builder hand-vectorises this into ~37 lines. That version
        has now been **proved equivalent** to this loop
        (``tests/test_background_sampling.py::test_add_batch_matches_where_a_builder_has_one``
        passes for alphagenome over a 400-value stream in 37-value chunks against
        a capacity of 40, so the replacement branch is exercised), which settles
        the open question in #125: it is a safe optimisation, not a divergence.
        It is kept out of here because the speed only matters for AlphaGenome's
        much larger per-variant fan-out, and a 3-line loop is easier to trust.
        """
        for v in values:
            self.add(track_idx, float(v))

    def get_sorted(self, track_idx: int) -> np.ndarray:
        arr = np.array(self.data[track_idx], dtype=np.float64)
        arr.sort()
        return arr

    def to_cdf_matrix(self, n_points: int = DEFAULT_CDF_POINTS) -> np.ndarray:
        """Project every track's reservoir onto a fixed ``n_points`` grid.

        Two regimes, and the second is the one that matters:

        * ``n >= n_points`` — subsample at evenly spaced indices.
        * ``n < n_points``  — **interpolate** onto the full grid. The row is not
          padded, so every stored entry is a real quantile estimate. This is
          exactly why the percentile denominator must be the grid width and not
          the sample count: a short row still spans the whole grid. Dividing by
          the count inflated every AlphaGenome percentile by ~5x (#119).

        A track with no samples stays all-zero, which ``_has_samples`` detects.
        """
        matrix = np.zeros((self.n_tracks, n_points), dtype=np.float64)
        target_q = np.linspace(0, 1, n_points)
        for i in range(self.n_tracks):
            arr = self.get_sorted(i)
            n = len(arr)
            if n == 0:
                continue
            if n >= n_points:
                indices = np.linspace(0, n - 1, n_points, dtype=int)
                matrix[i] = arr[indices]
            else:
                source_q = np.arange(n) / n
                matrix[i] = np.interp(target_q, source_q, arr)
        return matrix

    def get_counts(self) -> np.ndarray:
        return self.counts.copy()

    def total_samples(self) -> int:
        return int(self.counts.sum())

    def tracks_with_data(self) -> int:
        return int((self.counts > 0).sum())


def get_sequence(
    fasta,
    chrom: str,
    center: int,
    length: int,
    max_n_fraction: float = DEFAULT_MAX_N_FRACTION,
) -> str | None:
    """Fetch a ``length``-bp window centred on 1-based ``center``.

    Returns ``None`` when the window runs off the contig or is too repetitive to
    be a useful background sample.

    The 1-based ``center`` lands at index ``length // 2 - 1`` of the returned
    string, which is what every builder's substitution assumes. Only
    EPInformer-seq's builder actually *validates* that
    (``if seq[offset] != ref_base: continue``); the others substitute at that
    fixed offset and trust it. The arithmetic is correct today — for a 2,114 bp
    ChromBPNet window the span is 0-based ``[center-1057, center+1057)``, so a
    1-based ``center`` is index 1056 = 1057-1 — but nothing guards it, which is
    why a windowing change could corrupt a background silently rather than
    raising. Callers substituting a ref allele should assert the base first.
    """
    half = length // 2
    start = center - half
    end = start + length
    if start < 1:
        return None
    try:
        seq = str(fasta[chrom][start - 1:end - 1]).upper()
    except (KeyError, IndexError):
        return None
    if len(seq) != length:
        return None
    if seq.count("N") > length * max_n_fraction:
        return None
    return seq


def one_hot_encode(seq: str, channels_first: bool = False) -> np.ndarray:
    """One-hot encode a DNA string.

    ``(L, 4)`` float32 by default; ``(4, L)`` with ``channels_first=True``,
    which is what EPInformer-seq's builder needs.

    Bases outside ``ACGT`` — including the N runs ``get_sequence`` tolerates up
    to ``max_n_fraction`` — become all-zero columns rather than raising or being
    imputed. Every existing copy behaves this way; it is stated here because it
    is load-bearing and was nowhere documented.
    """
    out = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        idx = _BASE_INDEX.get(base)
        if idx is not None:
            out[i, idx] = 1.0
    return out.T.copy() if channels_first else out


def compute_effect(
    ref_value: float,
    alt_value: float,
    formula: str = "log2fc",
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
) -> float:
    """Signed effect of alt versus ref under one of three conventions.

    * ``log2fc`` — ``log2((alt + pc) / (ref + pc))``. The default, and what
      ``cherimoya_source/scoring.py`` and the ChromBPNet builder both compute.
    * ``logfc``  — the same in natural log.
    * ``diff``   — ``alt - ref``, for layers where a ratio is meaningless.

    Callers building an *unsigned* layer's effect CDF take ``abs()`` of this;
    signed layers (RNA, MPRA, Sei classes) keep the sign. That choice lives in
    the builder, not here, because it is a property of the layer.

    Note the pseudocount interacts with output scale: median ref window-sums
    span 235x across oracles within ``chromatin_accessibility`` alone (enformer
    0.207 to chrombpnet 48.57), so a fixed ``pc=1.0`` is a much larger
    perturbation for a low-scale oracle. That is one of the four causes of
    cross-oracle non-comparability recorded in #83.
    """
    if formula == "log2fc":
        return math.log2((alt_value + pseudocount) / (ref_value + pseudocount))
    if formula == "logfc":
        return math.log((alt_value + pseudocount) / (ref_value + pseudocount))
    if formula == "diff":
        return alt_value - ref_value
    raise ValueError(
        f"Unknown effect formula {formula!r}; expected 'log2fc', 'logfc' or 'diff'."
    )


def get_window_slice(values: np.ndarray, window_bp: int, resolution: int) -> np.ndarray:
    """Centre slice of ``window_bp`` from a prediction array.

    ``values`` is the oracle's output at ``resolution`` bp per bin, so the slice
    is ``window_bp // resolution`` bins wide, centred. A window wider than the
    prediction returns the whole array rather than raising, matching what the
    AlphaGenome, Borzoi and Enformer builders each did separately.
    """
    n_bins = max(1, window_bp // resolution)
    if n_bins >= len(values):
        return values
    mid = len(values) // 2
    half = n_bins // 2
    start = max(0, mid - half)
    return values[start:start + n_bins]


def score_window_sum(
    values: np.ndarray, window_bp: int, resolution: int,
) -> float:
    """Sum of the centre ``window_bp`` — the raw activity a layer is scored on."""
    return float(np.sum(get_window_slice(values, window_bp, resolution)))
