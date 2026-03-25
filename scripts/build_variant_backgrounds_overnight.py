"""Overnight job: Build variant effect backgrounds — runs inside chorus-alphagenome env.

Loads AlphaGenome ONCE, then scores all 500 variants in a single process.
Expected runtime: ~500 variants x ~1 min/variant = ~8 hours on A100.

Run with:
  nohup mamba run -n chorus-alphagenome python scripts/build_variant_backgrounds_overnight.py &
"""
import json
import logging
import math
import os
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
        logging.FileHandler(f"{log_dir}/variant_bg.log", mode='w'),
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

# Determine output types needed
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

# ── Scoring helpers ─────────────────────────────────────────────────
INPUT_LENGTH = 1_048_576  # AlphaGenome 1Mbp input

# Layer classification (simplified)
LAYER_MAP = {
    "DNASE": "chromatin_accessibility",
    "CHIP_TF": "tf_binding",
    "CHIP_HISTONE": "histone_marks",
    "CAGE": "tss_activity",
    "RNA_SEQ": "gene_expression",
}

# Window sizes per layer
WINDOW_MAP = {
    "chromatin_accessibility": 501,
    "tf_binding": 501,
    "histone_marks": 2001,
    "tss_activity": 501,
    "gene_expression": None,
}


def get_layer(aid):
    prefix = aid.split("/")[0]
    return LAYER_MAP.get(prefix, "other")


def predict_sequence(seq):
    """Run a single forward pass and return per-assay values."""
    output = model.predict_sequence(
        seq,
        requested_outputs=requested_outputs,
        ontology_terms=None,
    )
    results = {}
    for aid in assay_ids:
        idx, info = track_infos[aid]
        ot_enum = OutputType[info["output_type"]]
        track_data = output.get(ot_enum)
        if track_data is None:
            continue
        values = np.asarray(track_data.values)
        local_idx = info["local_index"]
        results[aid] = values[:, local_idx]
    return results


def score_variant(chrom, pos, ref_allele, alt_allele):
    """Score a single variant: extract sequence, mutate, predict, compute log2FC."""
    # Get reference sequence centered on variant
    center = pos
    half = INPUT_LENGTH // 2
    start = center - half
    end = center + half
    chrom_len = ref.get_reference_length(chrom)
    if start < 0 or end > chrom_len:
        return None  # skip near chromosome edges

    seq_ref = ref.fetch(chrom, start, end).upper()
    if len(seq_ref) != INPUT_LENGTH:
        return None

    # Count Ns — skip if too many
    n_count = seq_ref.count('N')
    if n_count > INPUT_LENGTH * 0.5:
        return None

    # Create alt sequence
    var_offset = pos - start - 1  # 0-based offset into sequence
    seq_alt = seq_ref[:var_offset] + alt_allele + seq_ref[var_offset + 1:]

    # Predict ref and alt
    ref_vals = predict_sequence(seq_ref)
    alt_vals = predict_sequence(seq_alt)

    # Score each track
    scores = {}
    for aid in assay_ids:
        if aid not in ref_vals or aid not in alt_vals:
            continue
        layer = get_layer(aid)
        window = WINDOW_MAP.get(layer, 501)
        if window is None:
            continue  # skip RNA (no windowed scoring)

        rv = ref_vals[aid]
        av = alt_vals[aid]

        _, info = track_infos[aid]
        resolution = info.get("resolution", 1)

        # Find center bin
        n_bins = len(rv)
        center_bin = n_bins // 2
        half_w = window // (2 * resolution)
        w_start = max(0, center_bin - half_w)
        w_end = min(n_bins, center_bin + half_w + 1)

        ref_sum = float(np.sum(rv[w_start:w_end]))
        alt_sum = float(np.sum(av[w_start:w_end]))

        # log2FC with pseudocount
        log2fc = math.log2((alt_sum + 1) / (ref_sum + 1))
        scores[layer] = scores.get(layer, [])
        scores[layer].append(log2fc)

    return scores


# ── Load SNPs ───────────────────────────────────────────────────────
bed_path = "/PHShome/lp698/chorus/chorus/analysis/data/common_snps_500.bed"
snps = []
with open(bed_path) as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split("\t")
        snps.append({
            "chrom": parts[0],
            "pos": int(parts[2]),
            "ref": parts[4],
            "alt": parts[5],
            "id": parts[3],
        })
logger.info("Loaded %d SNPs from %s", len(snps), bed_path)

# ── Score all variants ──────────────────────────────────────────────
layer_scores = {}
t0 = time.time()

for i, snp in enumerate(snps):
    if (i + 1) % 10 == 0 or i == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
        eta = (len(snps) - i - 1) / rate if rate > 0 else 0
        logger.info("Scoring %d/%d (%s) — %.1f min elapsed, ETA %.0f min",
                     i + 1, len(snps), snp["id"], elapsed / 60, eta)

    try:
        result = score_variant(snp["chrom"], snp["pos"], snp["ref"], snp["alt"])
        if result is None:
            logger.warning("Skipped %s (edge position or too many Ns)", snp["id"])
            continue
        for layer, scores in result.items():
            layer_scores.setdefault(layer, []).extend(scores)
    except Exception as exc:
        logger.warning("Failed %s: %s", snp["id"], str(exc)[:200])

ref.close()

elapsed = time.time() - t0
logger.info("Scoring complete in %.1f hours", elapsed / 3600)

# ── Build and save backgrounds ──────────────────────────────────────
cache_dir = os.path.expanduser("~/.chorus/backgrounds")
os.makedirs(cache_dir, exist_ok=True)

# Layer signing (unsigned for most, signed for expression)
SIGNED = {"gene_expression": True, "promoter_activity": True}

for layer, scores in layer_scores.items():
    if len(scores) < 5:
        logger.warning("Too few scores for %s (%d), skipping", layer, len(scores))
        continue

    arr = np.array(scores, dtype=np.float64)
    signed = SIGNED.get(layer, False)

    if not signed:
        arr = np.abs(arr)
    arr.sort()

    fname = f"alphagenome_{layer}.npy"
    np.save(os.path.join(cache_dir, fname), arr)
    logger.info("Built %s: %d scores, range=[%.4f, %.4f], median=%.4f",
                layer, len(arr), arr.min(), arr.max(), float(np.median(arr)))

logger.info("=" * 60)
logger.info("VARIANT BACKGROUNDS COMPLETE — %d layers saved to %s", len(layer_scores), cache_dir)
logger.info("Total time: %.1f hours", (time.time() - t0) / 3600)
logger.info("=" * 60)
