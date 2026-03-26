"""Build variant effect + baseline backgrounds across ALL cell types and tracks.

Loads AlphaGenome once, requests ALL output types per forward pass, and
extracts scores for every track across all 711 cell types. The forward
pass count is identical to K562-only (1000 for variants, 506 for baseline)
since AlphaGenome predicts all tracks of each output type simultaneously.

Expected: ~2.5M variant-effect scores + ~2.5M baseline samples
Runtime: ~30 min variants + ~15 min baselines = ~45 min total on A100.

Run with:
  nohup mamba run -n chorus-alphagenome python scripts/build_all_celltype_backgrounds.py &
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
        logging.FileHandler(f"{log_dir}/all_celltype_bg.log", mode='w'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Load model ──────────────────────────────────────────────────────
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

INPUT_LENGTH = 1_048_576

# ── Build track index ───────────────────────────────────────────────
# Map each track to its layer and output type info
LAYER_MAP = {
    'DNASE': 'chromatin_accessibility',
    'ATAC': 'chromatin_accessibility',
    'CHIP_TF': 'tf_binding',
    'CHIP_HISTONE': 'histone_marks',
    'CAGE': 'tss_activity',
    'PROCAP': 'tss_activity',
    'RNA_SEQ': 'gene_expression',
}

WINDOW_MAP = {
    'chromatin_accessibility': 501,
    'tf_binding': 501,
    'histone_marks': 2001,
    'tss_activity': 501,
    'gene_expression': None,  # skip for now (needs gene exons)
}

# Collect all tracks we want to score, grouped by output type
tracks_by_ot = {}  # output_type_name -> list of (track_idx, local_idx, resolution, layer)
all_output_types = set()

for i in range(len(metadata._tracks)):
    info = metadata.get_track_info(i)
    if info is None:
        continue
    ot_name = info.get('output_type', '')
    layer = LAYER_MAP.get(ot_name)
    if layer is None:
        continue
    if WINDOW_MAP.get(layer) is None:
        continue  # skip gene_expression (no window scoring)
    if ot_name in SKIPPED_OUTPUT_TYPES:
        continue

    all_output_types.add(ot_name)
    tracks_by_ot.setdefault(ot_name, []).append({
        'track_idx': i,
        'local_idx': info['local_index'],
        'resolution': info.get('resolution', 1),
        'layer': layer,
        'cell_type': info.get('cell_type', ''),
        'identifier': metadata._tracks[i].get('identifier', ''),
    })

requested_outputs = [
    ot for ot in OutputType
    if ot.name in all_output_types and ot.name not in SKIPPED_OUTPUT_TYPES
]

total_tracks = sum(len(v) for v in tracks_by_ot.values())
logger.info("Will score %d tracks across %d output types:", total_tracks, len(tracks_by_ot))
for ot_name, tracks in sorted(tracks_by_ot.items()):
    logger.info("  %s: %d tracks", ot_name, len(tracks))


# ── Scoring helpers ─────────────────────────────────────────────────
def predict_all_tracks(seq):
    """Run one forward pass and return raw output dict."""
    return model.predict_sequence(
        seq,
        requested_outputs=requested_outputs,
        ontology_terms=None,
    )


def score_all_tracks_variant(ref_output, alt_output):
    """Score all tracks from ref vs alt outputs. Returns {layer: [scores]}."""
    layer_scores = {}

    for ot_name, track_list in tracks_by_ot.items():
        ot_enum = OutputType[ot_name]
        ref_data = ref_output.get(ot_enum)
        alt_data = alt_output.get(ot_enum)
        if ref_data is None or alt_data is None:
            continue

        ref_vals = np.asarray(ref_data.values)   # shape: (bins, n_tracks)
        alt_vals = np.asarray(alt_data.values)
        n_output_tracks = ref_vals.shape[1]

        for track in track_list:
            local_idx = track['local_idx']
            if local_idx >= n_output_tracks:
                continue  # skip tracks beyond output bounds
            resolution = track['resolution']
            layer = track['layer']
            window = WINDOW_MAP[layer]

            rv = ref_vals[:, local_idx]
            av = alt_vals[:, local_idx]

            n_bins = len(rv)
            center_bin = n_bins // 2
            half_w = window // (2 * resolution)
            w_start = max(0, center_bin - half_w)
            w_end = min(n_bins, center_bin + half_w + 1)

            ref_sum = float(np.sum(rv[w_start:w_end]))
            alt_sum = float(np.sum(av[w_start:w_end]))

            log2fc = math.log2((alt_sum + 1) / (ref_sum + 1))
            layer_scores.setdefault(layer, []).append(log2fc)

    return layer_scores


def extract_baseline_signals(output):
    """Extract baseline signal levels from a single prediction."""
    layer_signals = {}

    for ot_name, track_list in tracks_by_ot.items():
        ot_enum = OutputType[ot_name]
        data = output.get(ot_enum)
        if data is None:
            continue

        vals = np.asarray(data.values)
        n_output_tracks = vals.shape[1]

        for track in track_list:
            if track['local_idx'] >= n_output_tracks:
                continue
            local_idx = track['local_idx']
            layer = track['layer']

            tv = vals[:, local_idx]
            n_bins = len(tv)
            center = n_bins // 2
            half_w = 250
            w_start = max(0, center - half_w)
            w_end = min(n_bins, center + half_w + 1)
            signal = float(np.sum(tv[w_start:w_end]))
            layer_signals.setdefault(layer, []).append(signal)

    return layer_signals


def get_sequence(chrom, pos):
    """Get INPUT_LENGTH sequence centered on pos."""
    half = INPUT_LENGTH // 2
    start = pos - half
    end = pos + half
    chrom_len = ref.get_reference_length(chrom)
    if start < 0 or end > chrom_len:
        return None
    seq = ref.fetch(chrom, start, end).upper()
    if len(seq) != INPUT_LENGTH or seq.count('N') > INPUT_LENGTH * 0.5:
        return None
    return seq


# ── Load SNPs ───────────────────────────────────────────────────────
bed_path = "/PHShome/lp698/chorus/chorus/analysis/data/common_snps_500.bed"
snps = []
with open(bed_path) as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split("\t")
        snps.append({
            "chrom": parts[0], "pos": int(parts[2]),
            "ref": parts[4], "alt": parts[5], "id": parts[3],
        })
logger.info("Loaded %d SNPs", len(snps))

cache_dir = os.path.expanduser("~/.chorus/backgrounds")
os.makedirs(cache_dir, exist_ok=True)

# ── PART 1: Variant effect backgrounds ──────────────────────────────
logger.info("=" * 60)
logger.info("PART 1: Variant effect backgrounds (%d SNPs x %d tracks)",
            len(snps), total_tracks)
logger.info("=" * 60)

variant_layer_scores = {}
t0 = time.time()

for i, snp in enumerate(snps):
    if (i + 1) % 10 == 0 or i == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
        eta = (len(snps) - i - 1) / rate if rate > 0 else 0
        n_scores = sum(len(v) for v in variant_layer_scores.values())
        logger.info("Variant %d/%d (%s) — %.1f min, ETA %.0f min, %d scores so far",
                     i + 1, len(snps), snp["id"], elapsed / 60, eta, n_scores)

    seq_ref = get_sequence(snp["chrom"], snp["pos"])
    if seq_ref is None:
        continue

    var_offset = snp["pos"] - (snp["pos"] - INPUT_LENGTH // 2) - 1
    seq_alt = seq_ref[:var_offset] + snp["alt"] + seq_ref[var_offset + 1:]

    try:
        ref_output = predict_all_tracks(seq_ref)
        alt_output = predict_all_tracks(seq_alt)

        # Log output shapes on first successful variant
        if i == 0 or not variant_layer_scores:
            for ot_name in tracks_by_ot:
                ot_enum = OutputType[ot_name]
                d = ref_output.get(ot_enum)
                if d is not None:
                    v = np.asarray(d.values)
                    n_meta = len(tracks_by_ot[ot_name])
                    n_valid = sum(1 for t in tracks_by_ot[ot_name] if t['local_idx'] < v.shape[1])
                    logger.info("  %s output shape: %s, metadata tracks: %d, valid: %d",
                                ot_name, v.shape, n_meta, n_valid)

        scores = score_all_tracks_variant(ref_output, alt_output)
        for layer, vals in scores.items():
            variant_layer_scores.setdefault(layer, []).extend(vals)
    except Exception as exc:
        logger.warning("Failed %s: %s", snp["id"], str(exc)[:200])

elapsed_v = time.time() - t0
logger.info("Variant scoring complete in %.1f min", elapsed_v / 60)

# Save variant backgrounds
SIGNED = {"gene_expression": True, "promoter_activity": True}
for layer, scores in variant_layer_scores.items():
    arr = np.array(scores, dtype=np.float64)
    signed = SIGNED.get(layer, False)
    if not signed:
        arr = np.abs(arr)
    arr.sort()
    fname = f"alphagenome_{layer}.npy"
    np.save(os.path.join(cache_dir, fname), arr)
    logger.info("  %s: %d scores, range=[%.4f, %.4f], median=%.4f",
                layer, len(arr), arr.min(), arr.max(), float(np.median(arr)))

# ── PART 2: Baseline signal backgrounds ─────────────────────────────
logger.info("=" * 60)
logger.info("PART 2: Baseline signal backgrounds (506 positions x %d tracks)",
            total_tracks)
logger.info("=" * 60)

random.seed(123)
chroms = [f"chr{i}" for i in range(1, 23)]
positions_per_chrom = 23

all_positions = []
for chrom in chroms:
    chrom_len = ref.get_reference_length(chrom)
    max_pos = min(chrom_len - 10_000_000, 200_000_000)
    if max_pos <= 10_000_000:
        max_pos = chrom_len - 1_000_000
    for _ in range(positions_per_chrom):
        all_positions.append((chrom, random.randint(10_000_000, max_pos)))

baseline_layer_signals = {}
t0 = time.time()

for i, (chrom, pos) in enumerate(all_positions):
    if (i + 1) % 25 == 0 or i == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
        eta = (len(all_positions) - i - 1) / rate if rate > 0 else 0
        n_samples = sum(len(v) for v in baseline_layer_signals.values())
        logger.info("Baseline %d/%d (%s:%d) — %.1f min, ETA %.0f min, %d samples",
                     i + 1, len(all_positions), chrom, pos, elapsed / 60, eta, n_samples)

    seq = get_sequence(chrom, pos)
    if seq is None:
        continue

    try:
        output = predict_all_tracks(seq)
        signals = extract_baseline_signals(output)
        for layer, vals in signals.items():
            baseline_layer_signals.setdefault(layer, []).extend(vals)
    except Exception as exc:
        logger.warning("Failed %s:%d: %s", chrom, pos, str(exc)[:200])

elapsed_b = time.time() - t0
logger.info("Baseline sampling complete in %.1f min", elapsed_b / 60)

# Save baseline backgrounds
for layer, signals in baseline_layer_signals.items():
    if len(signals) < 10:
        continue
    arr = np.array(signals, dtype=np.float64)
    arr.sort()
    fname = f"alphagenome_{layer}_baseline.npy"
    np.save(os.path.join(cache_dir, fname), arr)
    logger.info("  %s: %d samples, median=%.1f, range=[%.1f, %.1f]",
                layer, len(arr), float(np.median(arr)), float(arr.min()), float(arr.max()))

ref.close()

logger.info("=" * 60)
logger.info("ALL DONE")
logger.info("  Variant scoring: %.1f min, %d total scores",
            elapsed_v / 60, sum(len(v) for v in variant_layer_scores.values()))
logger.info("  Baseline sampling: %.1f min, %d total samples",
            elapsed_b / 60, sum(len(v) for v in baseline_layer_signals.values()))
logger.info("  Files saved to: %s", cache_dir)
logger.info("=" * 60)

for f in sorted(os.listdir(cache_dir)):
    fpath = os.path.join(cache_dir, f)
    size_kb = os.path.getsize(fpath) / 1024
    logger.info("  %s (%.1f KB)", f, size_kb)
