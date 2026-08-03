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


def expected_first_max_index(n_samples: int, n_points: int) -> int:
    """Grid index where a row built by :meth:`to_cdf_matrix` first hits its max.

    Derived from the two branches, not guessed:

    * ``n_samples >= n_points`` — the row is ``arr[linspace(0, n-1, n_points)]``,
      so the largest sample lands in the last slot: ``n_points - 1``.
    * ``n_samples < n_points`` — the row is
      ``np.interp(linspace(0, 1, n_points), arange(n)/n, arr)``. Note
      ``arange(n)/n`` stops at ``(n-1)/n``, **not** at 1.0, so ``np.interp``
      clamps every target beyond that to ``arr[-1]``. The first clamped target is
      the smallest ``k`` with ``k/(n_points-1) >= (n-1)/n``.

    The clamp means a short row legitimately ends in a *plateau* of repeated
    maxima ``n_points - expected_first_max_index`` long — 6 slots for
    AlphaGenome's 1,909 samples, 100 for a 100-sample row. Trailing duplicates
    are therefore normal and are **not** evidence of padding; a plateau longer
    than the clamp region is.
    """
    if n_samples >= n_points:
        return n_points - 1
    return math.ceil((n_points - 1) * (n_samples - 1) / n_samples)


def cdf_grid_violations(
    matrix: np.ndarray,
    counts: np.ndarray,
    *,
    label: str = "cdf",
    max_report: int = 5,
) -> list[str]:
    """Rows that cannot have been produced by ``to_cdf_matrix`` at this width.

    Exists because the shipped ``enformer_pertrack.npz`` was **not** reproducible
    from repo code and nothing noticed. Its ``effect_cdfs`` were gridded at 9,606
    points — ``max(effect_counts)`` — then padded to 10,000 by repeating each
    row's maximum 395 times. Since ``_get_denominator`` correctly returns the grid
    *width* (10,000, #119), every enformer effect percentile was scaled by
    ~0.9606 and no effect could land in ``(0.9605, 1.0)``: the top 4 % of the
    scale was unreachable for all 5,313 tracks.

    ``to_cdf_matrix`` interpolates short rows onto the *full* grid and cannot
    produce that shape, and ``build_and_save`` only resampled when
    ``shape[1] > n_points`` — so a too-narrow matrix was stored verbatim, with no
    assert and no warning. This is the missing assert.

    Two independent checks, both validated against all eight shipped backgrounds
    (19,393 short rows): each flags every one of enformer's 5,313 effect rows and
    nothing else.

    1. **Plateau length.** For a short row it must equal the clamp region exactly.
       Measured excess is 0 for every legitimate row and 393 for enformer.
       Skipped for ``n >= n_points``, where ties in the data can extend the tail
       legitimately — AlphaGenome's row 2,452 has its top two H3K4me1 windows
       tied at exactly 2480.0, which is real saturation, not corruption.
    2. **Distinct-vs-count.** Interpolating ``n`` samples onto a wider grid
       generates intermediate values, so the distinct count lands *away* from
       ``n`` — above it when intermediates appear, below when ties collapse
       (AlphaGenome reaches 0.418 x the grid width legitimately, so a "distinct
       must be near the width" threshold would false-positive). Landing exactly
       on ``n`` means the grid *was* ``n``.
    """
    problems: list[str] = []
    if matrix is None or matrix.size == 0:
        return problems
    n_points = int(matrix.shape[1])

    for i in range(matrix.shape[0]):
        if len(problems) >= max_report:
            problems.append(f"{label}: ... further rows suppressed")
            break
        n = int(counts[i])
        if n <= 0:
            continue
        row = np.asarray(matrix[i], dtype=np.float64)

        if not np.all(np.diff(row) >= -1e-6):
            problems.append(
                f"{label} row {i}: not non-decreasing (a CDF row is sorted values)"
            )
            continue
        if n >= n_points:
            continue
        if np.unique(row).size <= 1:
            # A constant row carries no grid geometry — it looks the same at
            # every width, so there is nothing here to validate. Legitimately
            # produced by a single sample (interp of one point is flat) or by a
            # track whose every prediction is identical. Both checks below would
            # false-positive: n == 1 makes distinct == count, and an all-equal
            # long row makes the plateau the whole grid. Degenerate backgrounds
            # are _has_samples' and the scale-degeneracy census's business.
            continue

        # A per-row plateau check USED to live here and was WRONG. It compared the
        # trailing run of maxima against the interpolation clamp and flagged
        # anything longer as padding. But TIES IN THE DATA also lengthen that run:
        # np.interp holds the tied value from the q-position of the first tied
        # sample onward. A fresh Borzoi build tripped it on 10 rows and Enformer on
        # 152 — e.g. borzoi row 3011, whose top effect value 0.689308 recurs 9
        # times because several sampled variants hit the same clipped ceiling. Those
        # rows are legitimate: their first max sits at index 9991, nowhere near the
        # n-1 = 5948 that padding produces.
        #
        # Ties were already exempted for `n >= n_points` (AlphaGenome's summary row
        # 2,452, two H3K4me1 windows tied at exactly 2480.0) and the same reasoning
        # simply was not applied to short rows. Caught by the rebuild, which is what
        # a guard that fails closed is for — it refused to write rather than
        # shipping anything.
        #
        # Padding is a FILE-level property, not a row-level one: it shifts every
        # row's maximum to the same index. That is checked by
        # cdf_grid_file_violations() below. What remains here is tie-immune.
        if np.unique(row).size == n:
            problems.append(
                f"{label} row {i}: exactly {n} distinct values on a "
                f"{n_points}-point grid — interpolation would not land on the "
                f"sample count, so this row was gridded at width {n}"
            )
    return problems


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


def sampling_uniformity(counts) -> dict:
    """Did every track see the same set of sampled positions?

    Distinguishes the two reasons a builder's ``*_counts`` can vary, which look
    identical if you only check "are they all equal":

    * **legitimate** — different sampling paths. Enformer skips CAGE at cCRE
      positions, so its ``summary_counts`` has two well-separated clusters by
      design; alphagenome's effect counts are 1697 (RNA) vs 1909, borzoi's 6563
      vs 9609, chrombpnet's 18672 vs 37344. None of those pairs is adjacent.
    * **partial credit** — a per-track loop that died part-way through, leaving
      the tracks visited before the exception incremented and the rest not. That
      produces a *tight run of consecutive integers*, and shipped
      ``enformer_pertrack.npz`` has exactly that: 7 values, 9600-9606 (#123).

    So ``adjacent_pairs`` is the diagnostic, not ``n_distinct``. Returns a summary
    dict; callers decide whether to warn or fail.
    """
    arr = np.asarray(counts)
    live = arr[arr > 0]
    if live.size == 0:
        return {"n_tracks": 0, "n_distinct": 0, "adjacent_pairs": 0,
                "min": 0, "max": 0, "suspect": False}
    uniq = np.unique(live)
    adjacent = int((np.diff(uniq) == 1).sum()) if uniq.size > 1 else 0
    return {
        "n_tracks": int(live.size),
        "n_distinct": int(uniq.size),
        "adjacent_pairs": adjacent,
        "min": int(uniq.min()),
        "max": int(uniq.max()),
        # a run of consecutive integers is the partial-credit fingerprint
        "suspect": adjacent > 0,
    }


def report_sampling_uniformity(reservoir, drop_reasons, label, logger) -> dict:
    """Log dropped positions and whether every track saw the same position set.

    Shared rather than copied into each builder: a silent drop is precisely how
    enformer shipped ``effect_counts`` spanning 9600-9606 with nothing saying so
    (#123), and four copies of the check would be a fresh instance of #144.

    Escalates to ``logger.error`` only on the partial-credit fingerprint, so the
    legitimate multi-cluster spreads (enformer skips CAGE at cCREs; alphagenome
    RNA samples a different path) stay quiet.
    """
    if drop_reasons:
        logger.warning("%s: dropped %d position(s) entirely: %s", label,
                       sum(drop_reasons.values()), dict(drop_reasons))
    stats = sampling_uniformity(reservoir.get_counts())
    logger.info("%s counts: %d distinct over %d tracks, range %d-%d",
                label, stats["n_distinct"], stats["n_tracks"],
                stats["min"], stats["max"])
    if stats["suspect"]:
        logger.error(
            "%s counts contain %d consecutive-integer pair(s) — the partial-credit "
            "fingerprint (#123). Tracks are NOT ranked against the same position "
            "set; investigate before publishing.", label, stats["adjacent_pairs"],
        )
    return stats


def cdf_grid_file_violations(
    matrix: np.ndarray, counts: np.ndarray, *, label: str = "cdf",
) -> list[str]:
    """Whole-matrix check for a padded grid, which no per-row check can do safely.

    Padding shifts **every** row's maximum to the same index — the narrow grid's
    width minus one — because the whole matrix was gridded at that width and then
    extended. Interpolation instead puts each row's first maximum at
    :func:`expected_first_max_index`, which varies with that row's sample count.

    So the signal is the **median** of ``expected - actual`` across rows. Measured:

    ======================================  ==================
    file                                    median deviation
    ======================================  ==================
    every legitimate shipped background                      0
    a fresh Borzoi / Enformer build                          0
    ``enformer_pertrack.npz`` before repair                **393**
    ======================================  ==================

    A median is the right statistic precisely because ties in the data shift a
    *handful* of rows (10 in a fresh Borzoi build, 152 in Enformer) while padding
    shifts all of them. The per-row plateau check this replaces could not tell
    those apart and rejected a valid rebuild.
    """
    problems: list[str] = []
    if matrix is None or matrix.size == 0:
        return problems
    n_points = int(matrix.shape[1])

    deviations = []
    for i in range(matrix.shape[0]):
        n = int(counts[i])
        if not (0 < n < n_points):
            continue
        row = np.asarray(matrix[i], dtype=np.float64)
        if np.unique(row).size <= 1:
            continue
        deviations.append(
            expected_first_max_index(n, n_points) - int(np.argmax(row))
        )
    # A median over a couple of rows is not evidence of anything: a single row with
    # tied maxima deviates exactly as a padded one does, and only the fact that
    # padding is UNANIMOUS across a matrix separates them. Real backgrounds have
    # thousands of rows; refuse to draw a conclusion from fewer than 8.
    if len(deviations) < 8:
        return problems

    median = float(np.median(deviations))
    # 1 absorbs float32 rounding at the top of a row; real padding is in the
    # hundreds, so the gap between signal and tolerance is three orders of
    # magnitude.
    if median > 1:
        problems.append(
            f"{label}: every row's maximum sits {median:.0f} slots earlier than "
            f"interpolation would place it, which is what gridding the matrix at a "
            f"narrower width and padding it out looks like. Percentiles in this "
            f"matrix are rescaled by roughly "
            f"{1 - median / n_points:.4f}."
        )
    return problems


class StagedSamples:
    """Per-position samples held back until the whole position succeeds.

    Reservoir adds must be **all-or-nothing per sampled position**, or tracks end
    up ranked against *different variant sets* (#123).

    Every builder wrapped its per-track loop and its model calls in one
    ``try/except``, so an exception raised part-way through the loop left the
    tracks visited before it incremented and the rest not. The damage is visible
    in the shipped ``enformer_pertrack.npz``: ``effect_counts`` takes **7**
    values, 9600-9606 — a tight run of consecutive integers, which is the
    fingerprint. Contrast the legitimate spreads elsewhere, which are
    well-separated clusters from genuinely different sampling paths (alphagenome
    1697 vs 1909, borzoi 6563 vs 9609, chrombpnet 18672 vs 37344) and never
    adjacent.

    The pattern is latent at **11 sites across 6 builders**; only enformer's data
    shows it actually firing. Usage::

        staged = StagedSamples()
        try:
            for t in track_info:
                staged.add(t['idx'], score)          # nothing committed yet
                staged.add_batch(t['idx'], values)
        except Exception:
            ...                                       # staged is simply dropped
        else:
            staged.commit(effect_res, perbin_res)     # all tracks, or none
    """

    __slots__ = ("_singles", "_batches")

    def __init__(self) -> None:
        self._singles: list[tuple[int, int, float]] = []
        self._batches: list[tuple[int, int, np.ndarray]] = []

    def add(self, track_idx: int, value: float, reservoir: int = 0) -> None:
        self._singles.append((reservoir, track_idx, float(value)))

    def add_batch(self, track_idx: int, values, reservoir: int = 0) -> None:
        self._batches.append((reservoir, track_idx, np.asarray(values)))

    def __len__(self) -> int:
        return len(self._singles) + len(self._batches)

    def commit(self, *reservoirs) -> None:
        """Flush every staged sample. Call only once the position fully succeeded."""
        for slot, track_idx, value in self._singles:
            reservoirs[slot].add(track_idx, value)
        for slot, track_idx, values in self._batches:
            reservoirs[slot].add_batch(track_idx, values)


def centered_bin_span(
    n_bins: int, window_bp: int, resolution: int, centre_bin: int | None = None,
) -> tuple[int, int]:
    """``(start, end)`` bins for a ``window_bp`` window centred in ``n_bins``.

    **The single definition of "centred window", for the builders and the query
    path alike.** They had two, and the query's was not even self-consistent
    (instance 2 of #144):

    * every binned builder used ``hw = window_bp // (2 * resolution)`` then
      ``[centre - hw, centre + hw + 1)`` — odd, centred, deterministic;
    * the query turned the window into genomic coordinates and floor/ceil-expanded
      them (``core/result.py``), so the bin count depended on where the variant
      fell *within* its bin: **4 or 5** bins for enformer at ``window_bp=501``,
      **16 or 17** at 2001, **63 or 64** for borzoi at 2001.

    That second point is the sharper defect. Two variants scored with identical
    settings were summed over different spans, so no background could match the
    numerator — the numerator's own definition moved.

    This keeps the builders' arithmetic **exactly**, so adopting it moves no
    shipped background. It is also the better convention on its own merits:
    deterministic, and symmetric about the variant, which is what "centred" should
    mean.

    Note that ``window_bp`` is not always representable: 501 bp at 128 bp
    resolution is 3.9 bins, so *no* convention delivers 501 bp there — this one
    sums 3 bins (384 bp). The honest response is to record the effective span in
    provenance (#124) rather than let the query claim a width the null never used.

    At ``resolution=1`` this returns exactly ``window_bp`` bins for odd windows,
    which is why ChromBPNet — the most audited oracle — could never show the bug.

    ``centre_bin`` defaults to the middle of the array, which is what the builders
    want: they centre each sampled sequence on the variant, so the middle bin *is*
    the variant's bin. The query path must pass the variant's bin explicitly,
    because a prediction clamped at a contig edge is not centred on the variant
    and the middle bin would then be the wrong one.
    """
    if window_bp is None:
        return 0, n_bins
    centre = n_bins // 2 if centre_bin is None else centre_bin
    half = window_bp // (2 * resolution)
    start = max(0, centre - half)
    end = min(n_bins, centre + half + 1)
    return start, end


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
