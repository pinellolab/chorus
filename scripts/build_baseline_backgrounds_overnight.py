"""Overnight job: Build baseline signal backgrounds — runs inside chorus-alphagenome env.

Loads AlphaGenome ONCE, then predicts wild-type signal at 500 positions.
Expected runtime: ~500 positions x ~30s/position = ~4 hours on A100.

Run with:
  nohup mamba run -n chorus-alphagenome python scripts/build_baseline_backgrounds_overnight.py &
"""
import logging
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, '/PHShome/lp698/chorus')
os.environ["CHORUS_NO_TIMEOUT"] = "1"

log_dir = "/PHShome/lp698/chorus/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/baseline_bg.log", mode='w'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Load model ONCE ─────────────────────────────────────────────────
import jax
from alphagenome.models.dna_output import OutputType
from alphagenome_research.model.dna_model import create_from_huggingface
from chorus.oracles.alphagenome_source.alphagenome_metadata import (
    get_metadata, SKIPPED_OUTPUT_TYPES,
)
import pysam

logger.info("Loading AlphaGenome model...")
t_load = time.time()
device = jax.devices("gpu")[0]
model = create_from_huggingface("all_folds", device=device)
metadata = get_metadata()
logger.info("Model loaded in %.1f seconds", time.time() - t_load)

ref_path = "/PHShome/lp698/chorus/genomes/hg38.fa"
ref = pysam.FastaFile(ref_path)

# ── Track setup ─────────────────────────────────────────────────────
assay_ids = [
    "DNASE/EFO:0002067 DNase-seq/.",
    "CHIP_TF/EFO:0002067 TF ChIP-seq CTCF/.",
    "CHIP_HISTONE/EFO:0002067 Histone ChIP-seq H3K27ac/.",
    "CAGE/hCAGE EFO:0002067/+",
    "CAGE/hCAGE EFO:0002067/-",
]

needed_output_types = set()
track_infos = {}
for aid in assay_ids:
    idx = metadata.get_track_by_identifier(aid)
    info = metadata.get_track_info(idx)
    needed_output_types.add(info["output_type"])
    track_infos[aid] = (idx, info)

requested_outputs = [
    ot for ot in OutputType
    if ot.name in needed_output_types and ot.name not in SKIPPED_OUTPUT_TYPES
]

INPUT_LENGTH = 1_048_576

LAYER_MAP = {
    "DNASE": "chromatin_accessibility",
    "CHIP_TF": "tf_binding",
    "CHIP_HISTONE": "histone_marks",
    "CAGE": "tss_activity",
}


def get_layer(aid):
    prefix = aid.split("/")[0]
    return LAYER_MAP.get(prefix, "other")


def predict_baseline(chrom, pos):
    """Predict wild-type signal at a single position."""
    half = INPUT_LENGTH // 2
    start = pos - half
    end = pos + half
    chrom_len = ref.get_reference_length(chrom)
    if start < 0 or end > chrom_len:
        return None

    seq = ref.fetch(chrom, start, end).upper()
    if len(seq) != INPUT_LENGTH:
        return None
    if seq.count('N') > INPUT_LENGTH * 0.5:
        return None

    output = model.predict_sequence(
        seq,
        requested_outputs=requested_outputs,
        ontology_terms=None,
    )

    signals = {}
    for aid in assay_ids:
        idx, info = track_infos[aid]
        ot_enum = OutputType[info["output_type"]]
        track_data = output.get(ot_enum)
        if track_data is None:
            continue
        values = np.asarray(track_data.values)
        local_idx = info["local_index"]
        track_vals = values[:, local_idx]

        layer = get_layer(aid)
        n_bins = len(track_vals)
        center = n_bins // 2
        half_w = 250
        w_start = max(0, center - half_w)
        w_end = min(n_bins, center + half_w + 1)
        signal = float(np.sum(track_vals[w_start:w_end]))
        signals.setdefault(layer, []).append(signal)

    return signals


# ── Sample positions across all autosomes ───────────────────────────
random.seed(123)
chroms = [f"chr{i}" for i in range(1, 23)]
positions_per_chrom = 23  # 23 * 22 = 506

all_positions = []
for chrom in chroms:
    chrom_len = ref.get_reference_length(chrom)
    max_pos = min(chrom_len - 10_000_000, 200_000_000)
    if max_pos <= 10_000_000:
        max_pos = chrom_len - 1_000_000
    for _ in range(positions_per_chrom):
        all_positions.append((chrom, random.randint(10_000_000, max_pos)))

logger.info("Sampling %d positions across %d chromosomes", len(all_positions), len(chroms))

# ── Score all positions ─────────────────────────────────────────────
layer_signals = {}
t0 = time.time()

for i, (chrom, pos) in enumerate(all_positions):
    if (i + 1) % 25 == 0 or i == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
        eta = (len(all_positions) - i - 1) / rate if rate > 0 else 0
        logger.info("Baseline %d/%d (%s:%d) — %.1f min elapsed, ETA %.0f min",
                     i + 1, len(all_positions), chrom, pos, elapsed / 60, eta)

    try:
        result = predict_baseline(chrom, pos)
        if result is None:
            logger.debug("Skipped %s:%d (edge or Ns)", chrom, pos)
            continue
        for layer, sigs in result.items():
            layer_signals.setdefault(layer, []).extend(sigs)
    except Exception as exc:
        logger.warning("Failed %s:%d: %s", chrom, pos, str(exc)[:200])

ref.close()

elapsed = time.time() - t0
logger.info("Sampling complete in %.1f hours", elapsed / 3600)

# ── Save backgrounds ────────────────────────────────────────────────
cache_dir = os.path.expanduser("~/.chorus/backgrounds")
os.makedirs(cache_dir, exist_ok=True)

for layer, signals in layer_signals.items():
    if len(signals) < 10:
        logger.warning("Skipping %s: only %d samples", layer, len(signals))
        continue
    arr = np.array(signals, dtype=np.float64)
    arr.sort()
    fname = f"alphagenome_{layer}_baseline.npy"
    np.save(os.path.join(cache_dir, fname), arr)
    logger.info("Built baseline %s: %d samples, median=%.1f, range=[%.1f, %.1f]",
                layer, len(arr), float(np.median(arr)), float(arr.min()), float(arr.max()))

logger.info("=" * 60)
logger.info("BASELINE BACKGROUNDS COMPLETE — %d layers saved to %s", len(layer_signals), cache_dir)
logger.info("Total time: %.1f hours", (time.time() - t0) / 3600)
logger.info("=" * 60)
