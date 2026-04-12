"""Legacy in-library API to build per-layer background distributions.

.. note::
    This module produces the **legacy per-layer** backgrounds (one CDF
    per regulatory layer per oracle) used by :class:`QuantileNormalizer`.

    The current production system uses **per-track** CDFs built by the
    standalone scripts in ``scripts/build_backgrounds_<oracle>.py`` and
    consumed via :class:`PerTrackNormalizer` (one CDF per individual
    track, e.g. one per ENCODE assay).  See ``scripts/`` for the
    per-track build pipeline.

This module is kept as a quick convenience API for small-scale tests
and notebooks.  For production use, prefer the standalone build scripts.

Two types of backgrounds:

1. **Variant effect backgrounds** — score common SNPs to establish the
   typical range of variant effects per layer.

2. **Baseline signal backgrounds** — sample wild-type prediction values
   at random positions to contextualise ref_value (is this region active?).

Backgrounds are cached under ``~/.chorus/backgrounds/`` by default
and keyed by ``{oracle}_{layer}.npy``.
"""

import logging
import random
from pathlib import Path

import numpy as np

from .normalization import BackgroundDistribution, QuantileNormalizer
from .scorers import LAYER_CONFIGS, classify_track_layer
from .variant_report import build_variant_report

logger = logging.getLogger(__name__)

# Default cache location
_DEFAULT_CACHE = Path.home() / ".chorus" / "backgrounds"


# ---------------------------------------------------------------------------
# Common SNP sampling
# ---------------------------------------------------------------------------

# Well-characterised common SNPs from 1000 Genomes (MAF > 5%) located
# in functional regions.  Organised by region type so backgrounds are
# enriched for variants that are *plausibly* regulatory.
#
# In a full deployment these would be loaded from a curated BED file;
# the small built-in list here is used when no external file is provided.

_BUILTIN_COMMON_SNPS: list[dict] = [
    # DHS / promoter variants
    {"chrom": "chr1", "pos": 109274968, "ref": "G", "alt": "T", "id": "rs12740374"},
    {"chrom": "chr11", "pos": 5254983, "ref": "G", "alt": "C", "id": "rs_HPFH"},
    {"chrom": "chr5", "pos": 1295046, "ref": "T", "alt": "G", "id": "rs_TERT"},
    {"chrom": "chr9", "pos": 22125503, "ref": "G", "alt": "C", "id": "rs1333049"},
    {"chrom": "chr6", "pos": 161010118, "ref": "T", "alt": "C", "id": "rs9349379"},
    {"chrom": "chr2", "pos": 227896948, "ref": "A", "alt": "G", "id": "rs13382811"},
    {"chrom": "chr12", "pos": 112241766, "ref": "G", "alt": "A", "id": "rs3184504"},
    {"chrom": "chr16", "pos": 53786615, "ref": "C", "alt": "T", "id": "rs12325386"},
    {"chrom": "chr10", "pos": 114758349, "ref": "C", "alt": "T", "id": "rs7903146"},
    {"chrom": "chr1", "pos": 169549811, "ref": "C", "alt": "T", "id": "rs6025"},
]


def get_common_snps(
    bed_path: str | Path | None = None,
    region_type: str = "all",
    n: int = 500,
    seed: int = 42,
) -> list[dict]:
    """Return a list of common SNPs for background scoring.

    Args:
        bed_path: Path to a BED file with columns:
            chrom, start, end, id, ref, alt.  If None, uses a small
            built-in set (only ~10 variants — suitable for tests).
        region_type: Filter by region type (``"dhs"``, ``"promoter"``,
            ``"splice"``, ``"all"``).  Currently only ``"all"`` is
            supported for the built-in set.
        n: Maximum number of SNPs to return.
        seed: Random seed for reproducible sub-sampling.

    Returns:
        List of variant dicts with ``chrom``, ``pos``, ``ref``, ``alt``,
        ``id`` keys.
    """
    if bed_path is not None:
        snps = _load_snps_from_bed(Path(bed_path))
    else:
        snps = list(_BUILTIN_COMMON_SNPS)

    if len(snps) > n:
        rng = random.Random(seed)
        snps = rng.sample(snps, n)

    return snps


def _load_snps_from_bed(path: Path) -> list[dict]:
    """Load SNPs from a 6-column BED file."""
    snps: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            snps.append({
                "chrom": parts[0],
                "pos": int(parts[2]),  # BED end = 1-based position for SNPs
                "ref": parts[4],
                "alt": parts[5],
                "id": parts[3],
            })
    return snps


# ---------------------------------------------------------------------------
# Variant effect backgrounds
# ---------------------------------------------------------------------------

def build_variant_backgrounds(
    oracle,
    oracle_name: str,
    assay_ids: list[str],
    gene_name: str | None = None,
    snps: list[dict] | None = None,
    n_variants: int = 500,
    cache_dir: str | Path | None = None,
) -> QuantileNormalizer:
    """Score common SNPs and build per-layer background distributions.

    Args:
        oracle: A loaded Chorus oracle.
        oracle_name: Oracle identifier (e.g. ``"alphagenome"``).
        assay_ids: Track identifiers to score.
        gene_name: Optional gene name for expression scoring.
        snps: Pre-computed SNP list.  If None, uses ``get_common_snps()``.
        n_variants: Number of SNPs to score (ignored if *snps* given).
        cache_dir: Directory for persisting backgrounds.

    Returns:
        A :class:`QuantileNormalizer` with per-layer backgrounds set.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if snps is None:
        snps = get_common_snps(n=n_variants)

    normalizer = QuantileNormalizer(cache_dir=str(cache_dir))

    # Collect per-layer raw scores
    layer_scores: dict[str, list[float]] = {}

    for i, var in enumerate(snps):
        logger.info(
            "Background scoring %d/%d: %s", i + 1, len(snps), var.get("id", ""),
        )
        try:
            position = f"{var['chrom']}:{var['pos']}"
            region = f"{var['chrom']}:{var['pos']}-{var['pos'] + 1}"

            variant_result = oracle.predict_variant_effect(
                genomic_region=region,
                variant_position=position,
                alleles=[var["ref"], var["alt"]],
                assay_ids=assay_ids,
            )

            report = build_variant_report(
                variant_result,
                oracle_name=oracle_name,
                gene_name=gene_name,
            )

            # Collect scores from first alt allele
            for allele_scores in report.allele_scores.values():
                for ts in allele_scores:
                    if ts.raw_score is not None:
                        layer_scores.setdefault(ts.layer, []).append(ts.raw_score)
                break  # first allele only

        except Exception as exc:
            logger.warning("Background scoring failed for %s: %s", var.get("id"), exc)

    # Build background distributions per layer
    for layer, scores in layer_scores.items():
        if len(scores) < 5:
            logger.warning("Too few scores for %s (%d), skipping", layer, len(scores))
            continue

        cfg = LAYER_CONFIGS.get(layer)
        signed = cfg.signed if cfg else False

        bg = BackgroundDistribution.from_scores(
            np.array(scores, dtype=np.float64), signed=signed,
        )

        key = normalizer.background_key(oracle_name, layer)
        normalizer.set_background(key, bg, persist=True)
        logger.info(
            "Built %s background: %d scores, signed=%s",
            key, len(scores), signed,
        )

    return normalizer


# ---------------------------------------------------------------------------
# Baseline signal backgrounds
# ---------------------------------------------------------------------------

def build_baseline_backgrounds(
    oracle,
    oracle_name: str,
    assay_ids: list[str],
    n_positions: int = 1000,
    chrom: str = "chr1",
    seed: int = 42,
    cache_dir: str | Path | None = None,
) -> dict[str, BackgroundDistribution]:
    """Sample wild-type prediction values to build baseline signal backgrounds.

    Args:
        oracle: A loaded Chorus oracle.
        oracle_name: Oracle identifier.
        assay_ids: Track identifiers.
        n_positions: Number of random positions to sample.
        chrom: Chromosome to sample from.
        seed: Random seed.
        cache_dir: Directory for persisting backgrounds.

    Returns:
        Dict mapping layer names to :class:`BackgroundDistribution`.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    # Sample positions across the chromosome (avoiding telomeres)
    positions = sorted(rng.randint(5_000_000, 200_000_000) for _ in range(n_positions))

    layer_signals: dict[str, list[float]] = {}

    for i, pos in enumerate(positions):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info("Baseline sampling %d/%d", i + 1, len(positions))

        try:
            # Use predict_variant_effect with ref=alt (no change) to get
            # baseline signals.  This works reliably with environment-wrapped
            # oracles where the bare predict() API may not be wired through.
            position_str = f"{chrom}:{pos}"
            region_str = f"{chrom}:{pos}-{pos + 1}"

            variant_result = oracle.predict_variant_effect(
                genomic_region=region_str,
                variant_position=position_str,
                alleles=["A", "A"],  # ref=alt → no change, just baseline
                assay_ids=assay_ids,
            )

            # Extract reference prediction tracks
            ref_pred = variant_result.get("predictions", {}).get("reference")
            if ref_pred is None:
                continue

            tracks = ref_pred.tracks if hasattr(ref_pred, "tracks") else ref_pred
            for aid, track in tracks.items():
                layer = classify_track_layer(track)
                vals = track.values
                center = len(vals) // 2
                half_w = 250  # 501bp window
                w_start = max(0, center - half_w)
                w_end = min(len(vals), center + half_w + 1)
                signal = float(np.sum(vals[w_start:w_end]))
                layer_signals.setdefault(layer, []).append(signal)

        except Exception as exc:
            logger.warning("Baseline sampling failed at %s:%d: %s", chrom, pos, exc)

    # Build and persist distributions
    results: dict[str, BackgroundDistribution] = {}
    for layer, signals in layer_signals.items():
        if len(signals) < 10:
            continue
        bg = BackgroundDistribution(np.array(signals, dtype=np.float64))
        fname = f"{oracle_name}_{layer}_baseline.npy"
        bg.save(str(cache_dir / fname))
        results[layer] = bg
        logger.info("Built baseline %s: %d samples", layer, len(signals))

    return results
