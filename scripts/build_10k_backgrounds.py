"""Build comprehensive backgrounds: 10K variants + 25K baseline positions (all TSSs + random).

Uses all ~3,763 AlphaGenome tracks across all cell types.
Expected: ~37M variant scores + ~94M baseline samples.
Runtime: ~10 hours variants + ~14 hours baselines on A100.

For parallel execution on 2 GPUs, use the --part flag:
  GPU 0: mamba run -n chorus-alphagenome python scripts/build_10k_backgrounds.py --part variants --gpu 0
  GPU 1: mamba run -n chorus-alphagenome python scripts/build_10k_backgrounds.py --part baselines --gpu 1

Or run everything sequentially on one GPU (no flag needed).
"""
import argparse
import logging
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, '/PHShome/lp698/chorus')
os.environ["CHORUS_NO_TIMEOUT"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--part", choices=["variants", "baselines", "both"], default="both")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--n-variants", type=int, default=10000)
parser.add_argument("--n-random-positions", type=int, default=5000)
parser.add_argument("--max-tss", type=int, default=20000,
                    help="Max protein-coding TSSs to include (0=skip)")
args = parser.parse_args()

log_dir = "/PHShome/lp698/chorus/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/bg_10k_{args.part}_gpu{args.gpu}.log", mode='w'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Load model ──────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import jax
from alphagenome.models.dna_output import OutputType
from alphagenome_research.model.dna_model import create_from_huggingface
from chorus.oracles.alphagenome_source.alphagenome_metadata import (
    get_metadata, SKIPPED_OUTPUT_TYPES,
)
import pysam

logger.info("Loading AlphaGenome on GPU %d...", args.gpu)
t_load = time.time()
device = jax.devices("gpu")[0]
model = create_from_huggingface("all_folds", device=device)
metadata = get_metadata()
logger.info("Model loaded in %.1f seconds", time.time() - t_load)

ref_path = "/PHShome/lp698/chorus/genomes/hg38.fa"
ref = pysam.FastaFile(ref_path)

INPUT_LENGTH = 1_048_576

# ── Build track index ───────────────────────────────────────────────
LAYER_MAP = {
    'DNASE': 'chromatin_accessibility', 'ATAC': 'chromatin_accessibility',
    'CHIP_TF': 'tf_binding', 'CHIP_HISTONE': 'histone_marks',
    'CAGE': 'tss_activity', 'PROCAP': 'tss_activity',
}

WINDOW_MAP = {
    'chromatin_accessibility': 501, 'tf_binding': 501,
    'histone_marks': 2001, 'tss_activity': 501,
}

tracks_by_ot = {}
all_output_types = set()

for i in range(len(metadata._tracks)):
    info = metadata.get_track_info(i)
    if info is None:
        continue
    ot_name = info.get('output_type', '')
    layer = LAYER_MAP.get(ot_name)
    if layer is None or WINDOW_MAP.get(layer) is None:
        continue
    if ot_name in SKIPPED_OUTPUT_TYPES:
        continue
    all_output_types.add(ot_name)
    tracks_by_ot.setdefault(ot_name, []).append({
        'local_idx': info['local_index'],
        'resolution': info.get('resolution', 1),
        'layer': layer,
    })

requested_outputs = [
    ot for ot in OutputType
    if ot.name in all_output_types and ot.name not in SKIPPED_OUTPUT_TYPES
]

total_tracks = sum(len(v) for v in tracks_by_ot.values())
logger.info("Scoring %d tracks across %d output types", total_tracks, len(tracks_by_ot))


# ── Helpers ─────────────────────────────────────────────────────────
def predict_all(seq):
    return model.predict_sequence(seq, requested_outputs=requested_outputs, ontology_terms=None)


def get_sequence(chrom, pos):
    half = INPUT_LENGTH // 2
    start, end = pos - half, pos + half
    chrom_len = ref.get_reference_length(chrom)
    if start < 0 or end > chrom_len:
        return None
    seq = ref.fetch(chrom, start, end).upper()
    if len(seq) != INPUT_LENGTH or seq.count('N') > INPUT_LENGTH * 0.5:
        return None
    return seq


def score_variant(ref_output, alt_output):
    """Score all tracks, return {layer: [log2fc_scores]}."""
    layer_scores = {}
    for ot_name, track_list in tracks_by_ot.items():
        ot_enum = OutputType[ot_name]
        rd, ad = ref_output.get(ot_enum), alt_output.get(ot_enum)
        if rd is None or ad is None:
            continue
        rv, av = np.asarray(rd.values), np.asarray(ad.values)
        n_out = rv.shape[1]
        for t in track_list:
            idx = t['local_idx']
            if idx >= n_out:
                continue
            layer = t['layer']
            window = WINDOW_MAP[layer]
            res = t['resolution']
            n_bins = rv.shape[0]
            c = n_bins // 2
            hw = window // (2 * res)
            ws, we = max(0, c - hw), min(n_bins, c + hw + 1)
            rs = float(np.sum(rv[ws:we, idx]))
            als = float(np.sum(av[ws:we, idx]))
            layer_scores.setdefault(layer, []).append(math.log2((als + 1) / (rs + 1)))
    return layer_scores


def extract_baseline(output):
    """Extract baseline signals from all tracks, return {layer: [signals]}."""
    layer_signals = {}
    for ot_name, track_list in tracks_by_ot.items():
        ot_enum = OutputType[ot_name]
        d = output.get(ot_enum)
        if d is None:
            continue
        vals = np.asarray(d.values)
        n_out = vals.shape[1]
        for t in track_list:
            idx = t['local_idx']
            if idx >= n_out:
                continue
            layer = t['layer']
            n_bins = vals.shape[0]
            c = n_bins // 2
            ws, we = max(0, c - 250), min(n_bins, c + 251)
            signal = float(np.sum(vals[ws:we, idx]))
            layer_signals.setdefault(layer, []).append(signal)
    return layer_signals


cache_dir = os.path.expanduser("~/.chorus/backgrounds")
os.makedirs(cache_dir, exist_ok=True)
SIGNED = {"gene_expression": True, "promoter_activity": True}


# ═══════════════════════════════════════════════════════════════════
# PART 1: VARIANT EFFECT BACKGROUNDS
# ═══════════════════════════════════════════════════════════════════
if args.part in ("variants", "both"):
    logger.info("=" * 60)
    logger.info("VARIANT BACKGROUNDS: %d SNPs x %d tracks", args.n_variants, total_tracks)
    logger.info("=" * 60)

    # Generate random SNPs from reference genome
    random.seed(42)
    chroms = [f"chr{i}" for i in range(1, 23)]
    snps_per_chrom = args.n_variants // len(chroms) + 1
    snps = []

    for chrom in chroms:
        chrom_len = ref.get_reference_length(chrom)
        max_pos = min(chrom_len - 5_000_000, 200_000_000)
        for _ in range(snps_per_chrom):
            if len(snps) >= args.n_variants:
                break
            pos = random.randint(5_000_000, max_pos)
            ref_base = ref.fetch(chrom, pos - 1, pos).upper()
            if ref_base not in "ACGT":
                continue
            alt_base = random.choice([b for b in "ACGT" if b != ref_base])
            snps.append({"chrom": chrom, "pos": pos, "ref": ref_base, "alt": alt_base})

    random.shuffle(snps)
    snps = snps[:args.n_variants]
    logger.info("Generated %d random SNPs", len(snps))

    variant_layer_scores = {}
    t0 = time.time()

    for i, snp in enumerate(snps):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
            eta = (len(snps) - i - 1) / rate if rate > 0 else 0
            n_scores = sum(len(v) for v in variant_layer_scores.values())
            logger.info("Variant %d/%d — %.1f min, ETA %.0f min, %s scores",
                         i + 1, len(snps), elapsed / 60, eta, f"{n_scores:,}")

        seq_ref = get_sequence(snp["chrom"], snp["pos"])
        if seq_ref is None:
            continue
        var_offset = snp["pos"] - (snp["pos"] - INPUT_LENGTH // 2) - 1
        seq_alt = seq_ref[:var_offset] + snp["alt"] + seq_ref[var_offset + 1:]

        try:
            ro, ao = predict_all(seq_ref), predict_all(seq_alt)
            for layer, vals in score_variant(ro, ao).items():
                variant_layer_scores.setdefault(layer, []).extend(vals)
        except Exception as exc:
            logger.warning("Failed variant %d: %s", i, str(exc)[:150])

    elapsed_v = time.time() - t0
    logger.info("Variant scoring complete in %.1f hours (%s total scores)",
                elapsed_v / 3600, f"{sum(len(v) for v in variant_layer_scores.values()):,}")

    for layer, scores in variant_layer_scores.items():
        arr = np.array(scores, dtype=np.float64)
        if not SIGNED.get(layer, False):
            arr = np.abs(arr)
        arr.sort()
        fname = f"alphagenome_{layer}.npy"
        np.save(os.path.join(cache_dir, fname), arr)
        logger.info("  %s: %s scores, median=%.4f, range=[%.4f, %.4f]",
                     layer, f"{len(arr):,}", float(np.median(arr)), arr.min(), arr.max())


# ═══════════════════════════════════════════════════════════════════
# PART 2: BASELINE SIGNAL BACKGROUNDS
# ═══════════════════════════════════════════════════════════════════
if args.part in ("baselines", "both"):
    logger.info("=" * 60)
    logger.info("BASELINE BACKGROUNDS: TSS positions + random regions")
    logger.info("=" * 60)

    # Collect all protein-coding gene TSSs
    all_positions = []

    if args.max_tss > 0:
        from chorus.utils.annotations import get_annotation_manager
        manager = get_annotation_manager()
        gtf_path = manager.get_annotation_path('gencode_v48_basic')
        genes = manager._get_genes_df(gtf_path)
        pc_genes = genes[genes['gene_type'] == 'protein_coding']

        tss_positions = []
        for _, row in pc_genes.iterrows():
            chrom = row['chrom']
            if not chrom.startswith('chr') or chrom in ('chrM', 'chrY'):
                continue
            tss = int(row['start']) if row['strand'] == '+' else int(row['end'])
            tss_positions.append((chrom, tss))

        # Subsample if needed
        random.seed(456)
        if len(tss_positions) > args.max_tss:
            tss_positions = random.sample(tss_positions, args.max_tss)
        all_positions.extend(tss_positions)
        logger.info("Added %d protein-coding TSS positions", len(tss_positions))

    # Add random genomic positions
    random.seed(789)
    chroms = [f"chr{i}" for i in range(1, 23)]
    rand_per_chrom = args.n_random_positions // len(chroms) + 1
    rand_count = 0
    for chrom in chroms:
        chrom_len = ref.get_reference_length(chrom)
        max_pos = min(chrom_len - 10_000_000, 200_000_000)
        for _ in range(rand_per_chrom):
            if rand_count >= args.n_random_positions:
                break
            all_positions.append((chrom, random.randint(10_000_000, max_pos)))
            rand_count += 1
    logger.info("Added %d random genomic positions", rand_count)
    logger.info("Total baseline positions: %d", len(all_positions))

    # Shuffle for even GPU utilization
    random.shuffle(all_positions)

    baseline_layer_signals = {}
    t0 = time.time()

    for i, (chrom, pos) in enumerate(all_positions):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
            eta = (len(all_positions) - i - 1) / rate if rate > 0 else 0
            n_samples = sum(len(v) for v in baseline_layer_signals.values())
            logger.info("Baseline %d/%d (%s:%d) — %.1f min, ETA %.0f min, %s samples",
                         i + 1, len(all_positions), chrom, pos,
                         elapsed / 60, eta, f"{n_samples:,}")

        seq = get_sequence(chrom, pos)
        if seq is None:
            continue

        try:
            output = predict_all(seq)
            for layer, vals in extract_baseline(output).items():
                baseline_layer_signals.setdefault(layer, []).extend(vals)
        except Exception as exc:
            logger.warning("Failed baseline %s:%d: %s", chrom, pos, str(exc)[:150])

    elapsed_b = time.time() - t0
    logger.info("Baseline sampling complete in %.1f hours (%s total samples)",
                elapsed_b / 3600, f"{sum(len(v) for v in baseline_layer_signals.values()):,}")

    for layer, signals in baseline_layer_signals.items():
        if len(signals) < 10:
            continue
        arr = np.array(signals, dtype=np.float64)
        arr.sort()
        fname = f"alphagenome_{layer}_baseline.npy"
        np.save(os.path.join(cache_dir, fname), arr)
        logger.info("  %s: %s samples, median=%.1f, range=[%.1f, %.1f]",
                     layer, f"{len(arr):,}", float(np.median(arr)), arr.min(), arr.max())

ref.close()
logger.info("=" * 60)
logger.info("ALL DONE — files saved to %s", cache_dir)
logger.info("=" * 60)
