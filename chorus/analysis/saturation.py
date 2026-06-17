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
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

BASES = ("A", "C", "G", "T")


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
        the reference base), ``importance`` (list ``[W]``), ``assay_id``, ``window``.
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

    scores = np.zeros((width, 4), dtype=float)
    n_ok = 0
    for i in range(width):
        p = start + i
        ref_b = ref_seq[i]
        if ref_b not in BASES:
            continue
        for j, b in enumerate(BASES):
            if b == ref_b:
                continue  # leave reference base at 0
            # Minimal region; base.py auto-widens to the oracle's native window
            region = f"{chrom}:{p}-{p + 1}"
            try:
                vr = oracle.predict_variant_effect(
                    region, f"{chrom}:{p}", [ref_b, b],
                    assay_ids=list(assay_ids), genome=genome,
                )
                effs = _score_all_tracks(vr, oracle_name)
                scores[i, j] = effs[0].raw_score if effs else 0.0
                n_ok += 1
            except Exception as exc:  # robustness: a single failed site shouldn't kill the sweep
                logger.warning("ISM %s:%d %s>%s failed: %s", chrom, p, ref_b, b, exc)
                scores[i, j] = 0.0
        if (i + 1) % 5 == 0:
            logger.info("  ISM %s: %d/%d positions scored", oracle_name, i + 1, width)

    importance = (-scores.sum(axis=1) / 3.0).tolist()
    logger.info("ISM %s complete: %d/%d substitutions scored", oracle_name, n_ok, width * 3)
    return {
        "chrom": chrom,
        "start": start,
        "end": end,
        "ref_seq": ref_seq,
        "positions": list(range(start, end + 1)),
        "scores": scores.tolist(),
        "importance": importance,
        "assay_id": list(assay_ids)[0] if assay_ids else None,
        "window": window,
        "oracle": oracle_name,
    }
