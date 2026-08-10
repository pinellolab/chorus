"""In-silico saturation mutagenesis (ISM) for Chorus oracles.

ISM probes which bases around a variant a sequence-to-function oracle actually
"reads": every base in a window is mutated to each alternative, the variant
effect on a chosen track is scored, and the per-position disruption is returned
as an importance profile suitable for a motif logo.

The implementation deliberately *reuses* the battle-tested single-variant path
(``oracle.predict_variant_effect`` + :func:`chorus.analysis.discovery._score_all_tracks`)
so an ISM score is exactly a Chorus variant effect, just swept across a window.
Works with any oracle (AlphaGenome, ChromBPNet, LegNet, Borzoi, EPInformer-seq, ...).
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

BASES = ("A", "C", "G", "T")


def _jsonable(arr: np.ndarray) -> list:
    """Array → nested lists with NaN replaced by ``None``.

    NaN marks "not scored" inside the sweep, but this payload crosses MCP, whose
    serializer (``pydantic_core.to_json``) writes a bare ``NaN`` literal that
    strict JSON parsers — including every JavaScript client — refuse. ``None``
    survives the trip and is still distinguishable from a genuine 0.0.
    """
    if arr.ndim == 1:
        return [None if math.isnan(v) else v for v in arr.tolist()]
    return [[None if math.isnan(v) else v for v in row] for row in arr.tolist()]


def saturation_mutagenesis(
    oracle,
    oracle_name: str,
    center: str,
    assay_ids: Sequence[str],
    *,
    genome: str,
    window: int = 25,
) -> dict:
    """Single-base saturation mutagenesis in a window centred on a variant.

    For each position in a ``window``-bp window centred on ``center`` and each
    of the three alternative bases, the variant effect on ``assay_ids[0]`` is
    predicted (via the oracle's own variant path) and recorded as a signed
    log2 fold-change. The per-position **importance** is the mean disruption
    (``-mean`` effect across the three substitutions): a functional site loses
    signal when mutated, so motif positions score high.

    Args:
        oracle: a loaded Chorus oracle.
        oracle_name: oracle key (``'alphagenome'``, ``'chrombpnet'``, ``'legnet'``, ...).
        center: ``'chrom:pos'`` (1-based) — the variant / motif centre.
        assay_ids: track identifier(s) to score; the first is used for the logo.
        genome: path to the reference FASTA (read with pyfaidx).
        window: window size in bp (odd recommended; default 25).

    Returns:
        dict with ``chrom``, ``start``, ``end`` (1-based inclusive), ``ref_seq``,
        ``positions``, ``scores`` (list ``[W][4]`` signed log2FC per base, 0 on
        the reference base and ``None`` where that substitution could not be
        scored), ``importance`` (list ``[W]``, ``None`` where nothing scored),
        ``assay_id``, ``window``, plus the bookkeeping a caller needs to trust
        the profile: ``n_attempted``, ``n_scored``, ``n_failed``, ``first_error``.

        If **nothing** scored, an error dict (``error``, ``error_type``, the
        counts and ``first_error``) is returned *instead* — with no ``scores``
        or ``importance`` key at all, so a caller that ignores ``error`` raises
        rather than plotting a matrix of zeros.
    """
    from .discovery import _score_all_tracks
    import pyfaidx

    chrom, pos_s = center.split(":")
    pos = int(pos_s)
    half = window // 2
    start = pos - half  # 1-based inclusive
    end = pos + half

    fa = pyfaidx.Fasta(genome)
    # pyfaidx slicing is 0-based half-open: [start-1, end)
    ref_seq = str(fa[chrom][start - 1:end]).upper()
    width = len(ref_seq)

    aid = list(assay_ids)[0] if assay_ids else None

    # NaN, not 0.0, is the "never scored" fill. 0.0 is a *result* here — it is the
    # reference base's own cell and it is also a real "this substitution changes
    # nothing" score — so filling failures with 0.0 made a dead sweep read as a
    # flat one: a bogus track id produced scores [[0,0,0,0], ...] and importance
    # [-0.0, -0.0, -0.0] with no error field, while every other entry point raised
    # InvalidAssayError on the same arguments (audit 2026-08-09).
    scores = np.full((width, 4), np.nan, dtype=float)
    is_sub = np.zeros((width, 4), dtype=bool)   # cells we actually attempted
    n_attempted = 0
    n_ok = 0
    first_error: str | None = None
    first_error_type: str | None = None
    for i in range(width):
        p = start + i
        ref_b = ref_seq[i]
        if ref_b not in BASES:
            continue  # row stays NaN: never scored, which is not "no effect"
        scores[i, BASES.index(ref_b)] = 0.0  # reference base: nothing substituted
        for j, b in enumerate(BASES):
            if b == ref_b:
                continue
            is_sub[i, j] = True
            n_attempted += 1
            # Minimal region; base.py auto-widens to the oracle's native window
            region = f"{chrom}:{p}-{p + 1}"
            try:
                vr = oracle.predict_variant_effect(
                    region, f"{chrom}:{p}", [ref_b, b],
                    assay_ids=list(assay_ids), genome=genome,
                )
                effs = _score_all_tracks(vr, oracle_name)
                if not effs:
                    # A scorer that returns no track is a failure to score, not a
                    # zero effect; it used to be recorded as 0.0 and counted as OK.
                    raise RuntimeError(f"no track effect returned for {aid!r}")
                score = float(effs[0].raw_score)
                if math.isnan(score):
                    # NaN is this function's "not scored" marker, so a NaN coming
                    # back from the scorer has to be reported as a failure or the
                    # counts would disagree with the matrix.
                    raise RuntimeError(f"scorer returned NaN for {aid!r}")
                scores[i, j] = score
                n_ok += 1
            except Exception as exc:  # robustness: a single failed site shouldn't kill the sweep
                logger.warning("ISM %s:%d %s>%s failed: %s", chrom, p, ref_b, b, exc)
                if first_error is None:
                    first_error = f"{chrom}:{p} {ref_b}>{b}: {exc}"
                    first_error_type = type(exc).__name__
                # scores[i, j] stays NaN
        if (i + 1) % 5 == 0:
            logger.info("  ISM %s: %d/%d positions scored", oracle_name, i + 1, width)

    n_failed = n_attempted - n_ok
    logger.info("ISM %s complete: %d/%d substitutions scored", oracle_name, n_ok, n_attempted)

    if n_ok == 0:
        # Nothing scored: there is no profile to return, so return the reason
        # instead. `error_type` mirrors the exception the same arguments raise
        # through any other entry point (InvalidAssayError for a bogus track id),
        # so an agent sees the same diagnosis from every tool.
        reason = (f"scored 0 of {n_attempted} substitutions" if n_attempted
                  else f"{ref_seq!r} contains no A/C/G/T reference base to substitute")
        return {
            "error": f"ISM produced no usable profile: {reason}"
                     + (f". First failure — {first_error}" if first_error else ""),
            "error_type": first_error_type or "InvalidRegionError",
            "oracle": oracle_name,
            "assay_id": aid,
            "chrom": chrom,
            "start": start,
            "end": end,
            "window": window,
            "n_attempted": n_attempted,
            "n_scored": 0,
            "n_failed": n_failed,
            "first_error": first_error,
        }

    # Importance is the mean disruption over the substitutions that scored, so one
    # dead substitution costs precision at that position instead of dragging it
    # towards zero. A position where nothing scored is None, not 0.0.
    valid = is_sub & ~np.isnan(scores)
    n_valid = valid.sum(axis=1)
    total = np.where(valid, scores, 0.0).sum(axis=1)
    importance = np.where(n_valid > 0, -total / np.maximum(n_valid, 1), np.nan)

    return {
        "chrom": chrom,
        "start": start,
        "end": end,
        "ref_seq": ref_seq,
        "positions": list(range(start, end + 1)),
        "scores": _jsonable(scores),
        "importance": _jsonable(importance),
        "assay_id": aid,
        "window": window,
        "oracle": oracle_name,
        # Bookkeeping, always present: a partially-failed sweep is still returned
        # (one dead site shouldn't lose 25 positions) and this is how a caller
        # tells it from a complete one without reading the log.
        "n_attempted": n_attempted,
        "n_scored": n_ok,
        "n_failed": n_failed,
        "first_error": first_error,
    }
