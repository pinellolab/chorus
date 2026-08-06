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

import logging
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
# How many of a CDF row's TOP grid slots must be exact order statistics rather than
# estimates from a subsample. 200 of 10,000 is the top 2%, which covers p98 upward --
# the region where effect_percentile saturates and where effect_exceedance divides by
# the row's maximum. Below this, the ceiling is a random draw: a uniform m-of-N
# subsample keeps the population maximum with probability only m/N.
MIN_EXACT_TAIL_SLOTS = 200

logger = logging.getLogger(__name__)

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
        tail_k: "int | None" = None,
    ):
        self.n_tracks = n_tracks
        self.capacity = capacity
        self.data: list[list[float]] = [[] for _ in range(n_tracks)]
        self.counts = np.zeros(n_tracks, dtype=np.int64)
        self._rng = random.Random(seed)
        # Optional EXACT extremes, kept alongside the uniform body.
        #
        # Opt-in (``tail_k=None`` by default) so that constructing a sampler the way
        # every builder does today gives byte-identical behaviour. The uniform body is
        # unbiased for the middle of the distribution but its *maximum* is a draw:
        # a uniform m-of-N subsample keeps the population max with probability m/N
        # only, which is how AlphaGenome's RNA ceilings came to be understated by a
        # median 1.33x. Where the union does not fit in memory -- the ``perbin``
        # layer, offering up to 2,176,256 values per track against a 50,000 capacity,
        # a 43.5x thinning -- retaining the exact top and bottom K is what keeps the
        # part of the grid that percentiles actually saturate in truthful.
        #
        # Both ends, not just the top: 12.9% of AlphaGenome's rows and 20.3% of
        # Borzoi's are signed, and for those a strongly repressive effect crosses the
        # LOWER bound. Tracking only the maximum would leave exactly those rows with
        # an estimated floor.
        self.tail_k = int(tail_k) if tail_k else None
        self._top: list[list[float]] = [[] for _ in range(n_tracks)]
        self._bot: list[list[float]] = [[] for _ in range(n_tracks)]

    def add(self, track_idx: int, value: float) -> None:
        n = self.counts[track_idx]
        if n < self.capacity:
            self.data[track_idx].append(value)
        else:
            j = self._rng.randint(0, n)
            if j < self.capacity:
                self.data[track_idx][j] = value
        self.counts[track_idx] += 1
        if self.tail_k is not None:
            self._offer_tail(track_idx, (value,))

    def _offer_tail(self, track_idx: int, values) -> None:
        """Keep the exact largest and smallest ``tail_k`` values for one track.

        Buffered then trimmed rather than heap-maintained. A heapq touched once per
        value would be O(N log K) Python-level operations, and the perbin layer offers
        7.1e9 values across the fleet; trimming a bounded buffer with np.partition is
        O(K) and happens only when the buffer fills, so the amortised cost is one
        append per value.
        """
        k = self.tail_k
        if not k:
            return
        top, bot = self._top[track_idx], self._bot[track_idx]
        top.extend(values)
        bot.extend(values)
        if len(top) > 4 * k:
            arr = np.partition(np.asarray(top, dtype=np.float64), -k)[-k:]
            self._top[track_idx] = arr.tolist()
        if len(bot) > 4 * k:
            arr = np.partition(np.asarray(bot, dtype=np.float64), k - 1)[:k]
            self._bot[track_idx] = arr.tolist()

    def exact_tail(self, track_idx: int) -> "tuple[np.ndarray, np.ndarray]":
        """``(smallest_k_sorted, largest_k_sorted)`` for one track."""
        k = self.tail_k
        if not k:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty
        top = np.sort(np.asarray(self._top[track_idx], dtype=np.float64))[-k:]
        bot = np.sort(np.asarray(self._bot[track_idx], dtype=np.float64))[:k]
        return bot, top

    def add_batch(self, track_idx: int, values) -> None:
        """Offer many values to one track, vectorised where it is safe to be.

        Ported from ``build_backgrounds_alphagenome.py``, which carried the last
        un-migrated copy of this class (#125). That copy had to be migrated, not
        merely kept in sync: adding ``to_flat_samples`` to THIS class while
        AlphaGenome used its own left eight position shards running for fifty GPU
        minutes before every one of them died with ``AttributeError`` at the write
        step. Which is the thesis of #125, demonstrated at my own expense.

        The fast path is only taken while the reservoir still has room, where
        "which samples survive" is not yet a question -- everything offered is kept,
        so extending in bulk is identical to appending one at a time. Once full it
        falls back to per-value Algorithm R, because there the traversal ORDER
        decides which samples survive and a different order moves the CDF without
        any arithmetic changing.

        Equivalence to the plain loop is pinned by
        ``tests/test_background_sampling.py::test_add_batch_matches_where_a_builder_has_one``,
        which exercises the overflow branch (400 values in 37-value chunks against a
        capacity of 40) rather than only the happy path.
        """
        target = self.data[track_idx]
        current = int(self.counts[track_idx])
        cap = self.capacity
        n_new = len(values)
        if n_new == 0:
            return

        if current < cap:
            room = cap - current
            n_append = min(room, n_new)
            if n_append > 0:
                head = values[:n_append]
                if hasattr(head, "tolist"):
                    target.extend(head.tolist())
                else:
                    target.extend(float(v) for v in head)
            for off, v in enumerate(values[n_append:]):
                n_so_far = current + n_append + off
                j = self._rng.randint(0, n_so_far)
                if j < cap:
                    target[j] = float(v)
            self.counts[track_idx] += n_new
        else:
            for off, v in enumerate(values):
                j = self._rng.randint(0, current + off)
                if j < cap:
                    target[j] = float(v)
            self.counts[track_idx] += n_new
        if self.tail_k is not None:
            self._offer_tail(track_idx, values)

    def _add_batch_reference(self, track_idx: int, values) -> None:
        """The plain loop, kept so the equivalence test has something to compare to.

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

    def _row_with_exact_tail(
        self, track_idx: int, body: np.ndarray, offered: int, n_points: int,
    ) -> np.ndarray:
        """One grid row spliced from an exact tail and a uniform body, by population rank.

        Built by mapping each grid slot to the POPULATION rank it represents, rather
        than by concatenating the tail onto the body and re-gridding. Concatenating
        would double-count every value present in both, and would place the tail's
        order statistics at the wrong quantiles -- the top K of N belong at the top
        ``K/N`` of the grid, not at the top ``K/(len(body)+K)``.

        Properties, each pinned by a test:

        * non-decreasing by construction, since both pieces are sorted and the
          exact bounds are the join points;
        * EXACT in the top and bottom ``K * n_points / N`` slots, so the maximum the
          percentile clamps against and the bound ``effect_exceedance`` divides by are
          population values, not draws;
        * identical to the plain gridding in the interior, which is the same uniform
          sample it has always been;
        * never reached when ``offered == len(body)`` -- the caller takes the original
          path then, so an unthinned build is byte-for-byte what it was before.

        The one approximation: ties exactly at a join bound make the count of body
        values below it slightly biased, which shifts the interior estimate within the
        tie region only. ``_rank_with_tie_breaking`` already spreads ranks across ties
        downstream.
        """
        k = int(self.tail_k or 0)
        bot, top = self.exact_tail(track_idx)
        k = min(k, len(bot), len(top))
        if k == 0 or offered < 4 * k:
            # Not enough room for two disjoint exact ends plus an interior; fall back
            # to gridding whatever the body holds rather than inventing structure.
            idx = np.linspace(0, len(body) - 1, n_points, dtype=int)
            return body[idx]

        lo_bound, hi_bound = float(bot[-1]), float(top[0])
        interior = body[(body > lo_bound) & (body < hi_bound)]
        m = len(interior)

        row = np.empty(n_points, dtype=np.float64)
        ranks = np.rint(np.arange(n_points) * (offered - 1) / (n_points - 1)).astype(
            np.int64
        )
        for slot, j in enumerate(ranks):
            if j < k:
                row[slot] = bot[j]
            elif j >= offered - k:
                row[slot] = top[j - (offered - k)]
            elif m:
                # Population ranks k .. offered-k-1 are represented by the interior
                # sample, mapped proportionally.
                frac = (j - k) / max(offered - 2 * k - 1, 1)
                row[slot] = interior[min(int(round(frac * (m - 1))), m - 1)]
            else:
                row[slot] = lo_bound
        # Sorted by construction, but assert cheaply rather than trust the arithmetic:
        # a non-monotone CDF row silently corrupts every searchsorted downstream.
        np.maximum.accumulate(row, out=row)
        return row

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
            offered = int(self.counts[i])
            if self.tail_k and offered > n:
                # Thinned AND an exact tail was kept: splice by POPULATION rank.
                matrix[i] = self._row_with_exact_tail(i, arr, offered, n_points)
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

    # ------------------------------------------------------------------
    # Exact serialisation, for splitting one build across GPUs
    # ------------------------------------------------------------------
    #
    # AlphaGenome, Borzoi and Enformer produce every track from ONE forward pass,
    # so sharding them by *track* (which is how the ChromBPNet builder shards, and
    # which works there because its tracks are separate model files) saves no GPU
    # time at all — each shard would still run every pass. They have to be sharded
    # by *position* instead, and that changes what a merge means: each shard holds a
    # partial reservoir for EVERY track rather than a complete one for some tracks.
    #
    # Pooling the shards' 10,000-point CDF grids would be an approximation. A decent
    # one when the shards are equal-sized, but this codebase has spent enough effort
    # removing approximations that looked exact, so the shards serialise their raw
    # samples and the CDF is built exactly once, from the union.
    #
    # Ragged data, stored flat with offsets rather than as an object array, so the
    # NPZ still loads under allow_pickle=False.

    def to_flat_samples(self) -> dict:
        """Raw retained samples as ``{values, offsets, counts, n_tracks}``.

        ``values[offsets[i]:offsets[i + 1]]`` is track *i*'s reservoir. ``counts``
        is the true number of values ever offered, which is what the shipped
        ``*_counts`` means and is not recoverable from the lengths.
        """
        lengths = np.fromiter((len(d) for d in self.data), dtype=np.int64,
                              count=self.n_tracks)
        offsets = np.zeros(self.n_tracks + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        values = (np.concatenate([np.asarray(d, dtype=np.float64)
                                  for d in self.data if len(d)])
                  if offsets[-1] else np.zeros(0, dtype=np.float64))
        return {
            "values": values,
            "offsets": offsets,
            "counts": self.counts.copy(),
            "n_tracks": np.int64(self.n_tracks),
        }

    @classmethod
    def from_flat_samples(cls, *parts, capacity: "int | None",
                          seed: int = DEFAULT_SEED) -> "ReservoirSampler":
        """Rebuild one sampler from the union of several shards' samples.

        Counts ADD, because each shard offered a disjoint set of positions. The
        retained values concatenate; if that exceeds ``capacity`` the excess is
        subsampled deterministically rather than by re-running Algorithm R, since
        every retained value is already a uniform draw from its own shard and the
        shards are equal-sized by construction.

        ``capacity`` is **keyword-only with no default, deliberately.** This method
        used to default to ``DEFAULT_CAPACITY`` (50,000), and
        ``scripts/merge_effect_shards.py`` called it positionally without one. Every
        AlphaGenome ``gene_expression`` track offers 148,367 effect values, so the
        union was silently thinned to a uniform 50,000-subsample — 2.97x — and the
        shipped grid's *maximum* became the max of that subsample. A uniform
        m-of-N subsample retains the true maximum with probability exactly m/N:
        50,000/148,367 = 0.337, and 33.9% of the 667 RNA rows were measured to have
        kept it. The damage was a median 1.33x understated ceiling, up to 8.34x,
        while p99 stayed correct to 0.6% — which is why no percentile test noticed,
        and why the effect null pinned earlier than it should have.

        Passing ``capacity=None`` means **exact retention**: keep every value, so the
        grid's top slot is the true maximum over all offered positions. Use it
        whenever the union fits in memory, which for every current oracle it does.

        Neither the builders' ``capacity`` argument nor the write-time
        ``cdf_grid_violations`` guard could have prevented this: no *shard* exceeded
        its capacity (each offered ~18.5k), and the guard is fed the *offered* count
        while checking geometry set by the *retained* count, then skips every row
        with ``n >= n_points``. See ``thinning_violations``.
        """
        if not parts:
            raise ValueError("no shards given")
        n_tracks = int(parts[0]["n_tracks"])
        for p in parts[1:]:
            if int(p["n_tracks"]) != n_tracks:
                raise ValueError(
                    f"shard track counts disagree: {n_tracks} vs {int(p['n_tracks'])}"
                )

        totals = np.zeros(n_tracks, dtype=np.int64)
        for p in parts:
            totals += np.asarray(p["counts"]).astype(np.int64)

        if capacity is None:
            # Size the reservoir to the largest stream, so nothing is subsampled AND
            # the returned object stays internally consistent if a caller offers it
            # more values later (``add``/``add_batch`` compare against self.capacity).
            capacity = int(totals.max()) if n_tracks else 0

        merged = cls(n_tracks=n_tracks, capacity=capacity, seed=seed)
        rng = np.random.default_rng(seed)
        for i in range(n_tracks):
            pieces = []
            for p in parts:
                off = np.asarray(p["offsets"])
                lo, hi = int(off[i]), int(off[i + 1])
                if hi > lo:
                    pieces.append(np.asarray(p["values"])[lo:hi])
            if pieces:
                vals = np.concatenate(pieces)
                if len(vals) > capacity:
                    keep = rng.choice(len(vals), capacity, replace=False)
                    keep.sort()
                    vals = vals[keep]
                # .tolist() rather than a float() comprehension: same result, and
                # this path now moves ~150M values for an exact AlphaGenome union.
                merged.data[i] = vals.tolist()
            merged.counts[i] = int(totals[i])
        return merged

    def retained_counts(self) -> np.ndarray:
        """How many values each track actually holds, as opposed to was offered.

        ``counts`` is the offered count. The difference between the two is exactly
        the thinning that went unnoticed for AlphaGenome's RNA tracks, and it is not
        recoverable from a shipped NPZ today because only ``counts`` is written.
        """
        return np.array([len(d) for d in self.data], dtype=np.int64)

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


def derive_tail_k(
    n_expected: int,
    *,
    n_points: int = DEFAULT_CDF_POINTS,
    min_slots: int = MIN_EXACT_TAIL_SLOTS,
) -> int:
    """How large an exact tail a layer needs, derived rather than picked.

    The top ``K`` of ``N`` values occupy the top ``K * n_points / N`` grid slots, so to
    keep ``min_slots`` of them exact you need ``K >= min_slots * N / n_points``. A
    single fixed K cannot work across layers because ``N`` spans two orders of
    magnitude: at ``K = 20,000`` the same choice gives AlphaGenome's perbin 202 exact
    slots but ChromBPNet's only **91** (2,176,256 offered) and Cherimoya's **183**
    (1,088,128) -- silently below the bar on exactly the two oracles with the most
    thinning.

    ``n_expected`` is knowable before the first forward pass: it is
    ``n_positions x fan_out``, where fan_out is 1, the number of bins per position, the
    number of strands, or the per-gene fan-out for RNA. That is what makes this a
    preflight check rather than a post-mortem.
    """
    if n_expected <= 0:
        return 0
    k = -(-int(min_slots) * int(n_expected) // int(n_points))   # ceil division
    return max(k, int(n_points))


def sampler_preflight(
    label: str,
    *,
    n_expected: int,
    capacity: "int | None",
    tail_k: "int | None",
    n_points: int = DEFAULT_CDF_POINTS,
) -> dict:
    """Log, and refuse, a sampling configuration that would thin the tail.

    Called BEFORE any forward pass. The AlphaGenome thinning cost 25 GPU-hours of a
    build whose ceiling was wrong on 667 tracks; every input to this check is known in
    advance, so there is no reason to discover it afterwards.
    """
    exact = capacity is None or n_expected <= capacity
    slots = n_points if exact else int(
        min(tail_k or 0, n_expected) * n_points // max(n_expected, 1)
    )
    info = {
        "label": label, "n_expected": int(n_expected),
        "capacity": None if capacity is None else int(capacity),
        "tail_k": None if not tail_k else int(tail_k),
        "mode": "exact" if exact else "hybrid",
        "exact_top_slots": int(slots),
    }
    logger.info(
        "sampler preflight %s: n_expected=%d capacity=%s tail_k=%s mode=%s "
        "exact_top_slots=%d", label, info["n_expected"], info["capacity"],
        info["tail_k"], info["mode"], info["exact_top_slots"],
    )
    if slots < MIN_EXACT_TAIL_SLOTS:
        raise ValueError(
            f"sampler preflight FAILED for {label}: only {slots} of {n_points} grid "
            f"slots would be exact (need >= {MIN_EXACT_TAIL_SLOTS}). n_expected="
            f"{n_expected}, capacity={capacity}, tail_k={tail_k}. Raise the capacity "
            f"or pass tail_k>={derive_tail_k(n_expected, n_points=n_points)}."
        )
    return info


def sampling_block(*interims, tail_k: "dict | int | None" = None) -> dict:
    """Assemble ``build_and_save(sampling=...)`` from whatever the interims recorded.

    One helper rather than the same seven lines in seven builders. Each interim NPZ may
    carry ``{effect,summary,perbin}_counts`` (offered) and ``*_retained``; a layer is
    included only when both are present, so an interim written before retention was
    tracked degrades to "no claim" instead of a false one.

    Omitting the block entirely is what ``build_and_save`` logs an error about: without
    it there is no thinning check at all, and a guard nobody wired up is how both the
    padded enformer grid and the AlphaGenome thinning reached users.
    """
    block: dict = {}
    for data in interims:
        if data is None:
            continue
        keys = set(getattr(data, "files", None) or data.keys())
        for layer in ("effect", "summary", "perbin"):
            ck, rk = f"{layer}_counts", f"{layer}_retained"
            if ck not in keys or rk not in keys:
                continue
            k = tail_k.get(layer) if isinstance(tail_k, dict) else tail_k
            block[layer] = {
                "offered": np.asarray(data[ck]),
                "retained": np.asarray(data[rk]),
                "tail_k": int(k) if k else None,
            }
    return block


def abort_if_nothing_loads(
    n_attempted: int, n_loaded: int, *, label: str, min_attempts: int = 25,
) -> None:
    """Stop a per-track builder that has failed to load every model so far.

    Cherimoya and ChromBPNet load one model per track inside a
    ``try/except: logger.warning(...); continue`` loop. That tolerance is right -- a
    single missing checkpoint should not lose a whole run -- but with no floor it also
    tolerates loading NOTHING. Run in an env without the ``cherimoya`` package, the
    builder logged "Failed to load <track>" 1,518 times over 75 minutes, then died at
    the provenance step. Had that step not happened to import the missing package, it
    would have written an all-zero background.

    A run that has attempted ``min_attempts`` tracks and loaded none is misconfigured,
    not unlucky: raise while it has cost a minute rather than an hour.
    """
    if n_attempted >= min_attempts and n_loaded == 0:
        raise RuntimeError(
            f"{label}: attempted {n_attempted} tracks and loaded NONE. This is a "
            f"configuration failure, not bad luck -- check the conda env has the "
            f"oracle's package (cherimoya lives only in chorus-cherimoya) and that "
            f"the weights are present. Aborting rather than building an empty null."
        )


def scope_violations(
    n_planned: int, *, label: str, n_shipped: "int | None" = None,
    min_fraction: float = 0.9,
) -> list[str]:
    """Refuse a build that covers far fewer tracks than the background it replaces.

    ``yield_violations`` asks "did the tracks we attempted produce samples?" and cannot
    ask "did we attempt the right tracks?". ChromBPNet's ``--assay`` defaults to
    ATAC_DNASE, so a rebuild launched without it enumerated 9 models, scored all 9
    successfully, and wrote a 1.0 MB background to replace a 753-track 80 MB one --
    dropping every CHIP track, which is the layer the saturation work is about. Every
    guard passed: rc=0, 9 of 9 tracks with data, 100% yield, exact retention, and a
    perbin tail with 400 exact slots. The build was flawless and 1.2% of the job.

    So this is a check on SCOPE, made before the GPU time is spent, against the track
    count of the background actually on disk.
    """
    if not n_shipped or n_planned >= min_fraction * n_shipped:
        return []
    return [
        f"{label}: planning {n_planned} tracks against {n_shipped} in the shipped "
        f"background ({n_planned / n_shipped:.1%}). A build that covers a fraction of "
        f"the tracks still succeeds on every other measure and then replaces the whole "
        f"file -- check the assay/cell selection flags before spending the GPU time."
    ]


def yield_violations(
    counts: np.ndarray,
    *,
    label: str = "cdf",
    min_fraction: float = 0.5,
) -> list[str]:
    """Refuse a build whose positions almost all failed.

    An enformer ablation launched two TensorFlow processes onto the same GPU by
    accident. The second could not allocate a cuBLAS handle, so every single forward
    pass raised ``InternalError: Attempting to perform BLAS operation using
    StreamExecutor without BLAS support`` -- and the builder, whose per-position
    try/except is there so one bad locus cannot lose a whole run, dutifully dropped all
    5,968 of them and wrote a perfectly well-formed interim: 5,313 tracks, every row
    all-zero, every count 0.

    That file would have merged cleanly. ``_has_samples`` would then suppress those
    tracks' percentiles at query time, so the symptom would have been an oracle that
    silently stopped ranking anything -- the same failure shape as Sei's dark 40 rows,
    reached by a different route.

    The per-position tolerance is right; what was missing is a floor on the total. A
    build that retained less than ``min_fraction`` of its tracks' samples is a failed
    build, not a partial one.
    """
    counts = np.asarray(counts)
    if counts.size == 0:
        return [f"{label}: no tracks at all"]
    with_data = int((counts > 0).sum())
    frac = with_data / counts.size
    if frac >= min_fraction:
        return []
    return [
        f"{label}: only {with_data} of {counts.size} tracks have any samples "
        f"({frac:.1%}, floor {min_fraction:.0%}). Every position was probably rejected "
        f"-- check the drop-reason tally in the build log before trusting this file. "
        f"An all-zero interim merges cleanly and then silently disables the oracle."
    ]


def thinning_violations(
    offered: np.ndarray,
    retained: np.ndarray,
    *,
    n_points: int = DEFAULT_CDF_POINTS,
    tail_k: "int | None" = None,
    label: str = "cdf",
    max_report: int = 5,
) -> list[str]:
    """Rows whose TOP grid slots came from a thinned sample rather than the population.

    A separate check from ``cdf_grid_violations``, deliberately, because that function
    is about row *geometry* and is structurally unable to see this. It is handed the
    **offered** count while the geometry it validates is set by the **retained** count,
    and its first act is ``if n >= n_points: continue`` -- offered is always >= 10,000
    for a real build, so every thinned row is skipped. Its docstring promises it
    "refuses to write a CDF matrix that could not have been produced by
    ``to_cdf_matrix``", and that promise is false whenever retained < offered.

    What went wrong without it: ``merge_effect_shards.py`` unioned AlphaGenome's 8
    shards without passing a capacity, inheriting 50,000, against 148,367 offered
    values per ``gene_expression`` track. The body of the null was unaffected --
    reservoir sampling is unbiased -- but the *maximum* became the max of a 33.7%
    subsample, understating the ceiling by a median 1.33x and up to 8.34x. Since the
    ceiling is exactly what ``effect_percentile`` clamps against, the null pinned
    earlier than it should have, and nothing detected it because every calibration
    gate measures the body.

    Passes a row when EITHER:

    * ``retained == offered`` -- nothing was discarded, so every grid slot is a true
      order statistic; or
    * the row keeps an exact top-K tail wide enough to fill at least
      ``MIN_EXACT_TAIL_SLOTS`` grid slots, i.e.
      ``floor(min(tail_k, offered) * n_points / offered) >= MIN_EXACT_TAIL_SLOTS``.

    ``tail_k=None`` means no exact tail is kept, so any thinning is a violation.
    """
    offered = np.asarray(offered, dtype=np.int64)
    retained = np.asarray(retained, dtype=np.int64)
    if offered.shape != retained.shape:
        raise ValueError(
            f"{label}: offered/retained shape mismatch {offered.shape} vs "
            f"{retained.shape}"
        )
    problems: list[str] = []
    for i, (n_off, n_ret) in enumerate(zip(offered, retained)):
        if n_off <= 0 or n_ret >= n_off:
            continue
        if tail_k:
            exact_slots = int(min(tail_k, n_off) * n_points // n_off)
            if exact_slots >= MIN_EXACT_TAIL_SLOTS:
                continue
            problems.append(
                f"{label} row {i}: offered {n_off}, retained {n_ret}, exact top-K "
                f"tail fills only {exact_slots} of {n_points} grid slots "
                f"(need >= {MIN_EXACT_TAIL_SLOTS}) -- the row's maximum is a draw "
                f"from a subsample, not the population maximum"
            )
        else:
            problems.append(
                f"{label} row {i}: offered {n_off} values but retained only {n_ret} "
                f"({n_off / max(n_ret, 1):.2f}x thinned) with no exact tail kept. A "
                f"uniform subsample retains the population maximum with probability "
                f"{n_ret / n_off:.3f}, so this row's ceiling is a random draw."
            )
        if len(problems) >= max_report:
            problems.append(f"{label}: ... and more")
            break
    return problems


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
    warnings: list[str] = []
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
        # THE unambiguous signal, and the only one that raises.
        #
        # A padded row was gridded at width n and then extended, so its maximum
        # first appears at index n-1. Interpolation instead places it at
        # expected_first_max_index(n, n_points) — 9998 for n=5949 — which is
        # nowhere near n-1 unless n is already close to n_points. This is
        # mechanical, not statistical.
        if int(np.argmax(row)) == n - 1 and n < n_points - 2:
            problems.append(
                f"{label} row {i}: maximum first appears at index {n - 1}, which is "
                f"the sample count minus one — the shape of a row gridded at width "
                f"{n} and padded out to {n_points}. Interpolation would place it at "
                f"{expected_first_max_index(n, n_points)}."
            )
            continue

        # ADVISORY ONLY. `distinct == count` was measured as a perfect fingerprint
        # on the shipped files (5,313 of 5,313 padded enformer rows, 0 of 30,000+
        # legitimate ones) and then produced a FALSE POSITIVE on the first fresh
        # AlphaGenome build: row 3966 (CHIP_TF ARID3A) has 913 exact zeros, and the
        # interpolation of the remainder happens to yield exactly 5,949 distinct
        # values. Its first maximum sits at 9998 — precisely where interpolation
        # puts it — so the row is fine.
        #
        # Coincidence is reachable whenever a row has a large block of exact zeros,
        # which the gene-anchored region set makes common. So this warns and does
        # not block: a guard that halts a 10-hour rebuild on a coincidence is worse
        # than no guard.
        if np.unique(row).size == n:
            warnings.append(
                f"{label} row {i}: exactly {n} distinct values on a {n_points}-point "
                f"grid. Usually coincidence (a block of exact zeros); worth a look "
                f"only if many rows report it."
            )
    if warnings:
        logger.warning(
            "%s: %d row(s) have distinct == count. Advisory, not blocking:\n  %s",
            label, len(warnings), "\n  ".join(warnings[:3]),
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


# cdf_grid_file_violations() lived here and was REMOVED. It compared each
# row's first-maximum index against the interpolation clamp and flagged a
# whole-matrix shift as padding. It cried wolf twice: once on a synthetic
# fixture and once on a REALISTIC 120-slot tie at the top of a row, which is
# what a near-degenerate null legitimately looks like (AlphaGenome RNA tops
# out at 0.0417). Padding and a long tie produce the same shape, so the check
# cannot separate them from row geometry alone.
#
# The per-row `distinct == count` test in cdf_grid_violations() does separate
# them, and does it perfectly on real data: 5,313 of 5,313 padded enformer
# rows flagged, 0 of 30,000+ legitimate rows across all eight shipped
# backgrounds AND both fresh rebuilds. Padding preserves distinct values
# (the narrow grid was full-resolution); a tie destroys them. One check that
# is measured to work beats two where the second invents failures.


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
