"""Build comprehensive backgrounds: 10K variants + 25K baselines, ALL output types.

Covers all 9 AlphaGenome output types including RNA-seq (exon counting)
and splice sites. Uses all ~3,763+ tracks across all cell types.

For parallel execution on 2 GPUs:
  GPU 0: mamba run -n chorus-alphagenome python scripts/build_10k_backgrounds.py --part variants --gpu 0
  GPU 1: mamba run -n chorus-alphagenome python scripts/build_10k_backgrounds.py --part baselines --gpu 1
"""
import argparse
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, '/PHShome/lp698/chorus')
os.environ["CHORUS_NO_TIMEOUT"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--part", choices=["variants", "baselines", "both"], default="both")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--n-variants", type=int, default=10000)
parser.add_argument("--n-random-positions", type=int, default=5000)
parser.add_argument("--max-tss", type=int, default=20000)
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

# ── Load gene annotations for RNA-seq exon scoring ─────────────────
logger.info("Loading gene annotations for RNA-seq scoring...")
from chorus.utils.annotations import get_annotation_manager
ann_manager = get_annotation_manager()
gtf_path = ann_manager.get_annotation_path('gencode_v48_basic')

# Pre-load all protein-coding exons, grouped by gene and chromosome
exon_df = ann_manager._get_exons_df(gtf_path)
# Filter to protein-coding only via gene info
gene_df = ann_manager._get_genes_df(gtf_path)
pc_gene_names = set(gene_df[gene_df['gene_type'] == 'protein_coding']['gene_name'])
pc_exons = exon_df[exon_df['gene_name'].isin(pc_gene_names)].copy()

# Build merged exon intervals per gene (avoid double-counting overlapping exons)
gene_exons = {}  # gene_name -> [(chrom, start, end), ...]
for gene_name, group in pc_exons.groupby('gene_name'):
    intervals = sorted(zip(group['chrom'], group['start'], group['end']))
    # Merge overlapping
    merged = []
    for chrom, s, e in intervals:
        if merged and merged[-1][0] == chrom and s <= merged[-1][2]:
            merged[-1] = (chrom, merged[-1][1], max(e, merged[-1][2]))
        else:
            merged.append((chrom, s, e))
    gene_exons[gene_name] = merged

logger.info("Loaded merged exons for %d protein-coding genes", len(gene_exons))

# Build spatial index: chrom -> sorted list of (start, end, gene_name)
# for fast lookup of genes overlapping a prediction window
gene_intervals_by_chrom = defaultdict(list)
for gene_name, exons in gene_exons.items():
    for chrom, s, e in exons:
        gene_intervals_by_chrom[chrom].append((s, e, gene_name))

for chrom in gene_intervals_by_chrom:
    gene_intervals_by_chrom[chrom].sort()

logger.info("Built spatial index for %d chromosomes", len(gene_intervals_by_chrom))


def get_genes_in_window(chrom, window_start, window_end):
    """Find genes with exons overlapping the window. Returns {gene_name: [(start,end),..]}"""
    genes = defaultdict(list)
    for s, e, gn in gene_intervals_by_chrom.get(chrom, []):
        if s > window_end:
            break
        if e >= window_start and s <= window_end:
            genes[gn].append((max(s, window_start), min(e, window_end)))
    return genes


# ── Build track index ───────────────────────────────────────────────
# Layer mapping: output_type -> (layer_name, window_bp, formula, pseudocount)
LAYER_SPEC = {
    'DNASE':             ('chromatin_accessibility', 501,  'log2fc', 1.0),
    'ATAC':              ('chromatin_accessibility', 501,  'log2fc', 1.0),
    'CHIP_TF':           ('tf_binding',              501,  'log2fc', 1.0),
    'CHIP_HISTONE':      ('histone_marks',           2001, 'log2fc', 1.0),
    'CAGE':              ('tss_activity',             501,  'log2fc', 1.0),
    'PROCAP':            ('tss_activity',             501,  'log2fc', 1.0),
    'RNA_SEQ':           ('gene_expression',          None, 'logfc',  0.001),
    'SPLICE_SITES':      ('splicing',                 501,  'log2fc', 1.0),
    'SPLICE_SITE_USAGE': ('splicing',                 501,  'log2fc', 1.0),
}

# All backgrounds are unsigned (magnitude) — direction is in the raw score
# for interpretation, but the quantile ranks magnitude only.

tracks_by_ot = {}
all_output_types = set()

for i in range(len(metadata._tracks)):
    info = metadata.get_track_info(i)
    if info is None:
        continue
    ot_name = info.get('output_type', '')
    if ot_name not in LAYER_SPEC or ot_name in SKIPPED_OUTPUT_TYPES:
        continue
    all_output_types.add(ot_name)
    tracks_by_ot.setdefault(ot_name, []).append({
        'local_idx': info['local_index'],
        'resolution': info.get('resolution', 1),
        'layer': LAYER_SPEC[ot_name][0],
        'strand': metadata._tracks[i].get('strand', '.'),
    })

requested_outputs = [
    ot for ot in OutputType
    if ot.name in all_output_types and ot.name not in SKIPPED_OUTPUT_TYPES
]

total_tracks = sum(len(v) for v in tracks_by_ot.values())
logger.info("Scoring %d tracks across %d output types:", total_tracks, len(tracks_by_ot))
for ot_name, tl in sorted(tracks_by_ot.items()):
    layer = LAYER_SPEC[ot_name][0]
    logger.info("  %s -> %s: %d tracks", ot_name, layer, len(tl))


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


def compute_effect(ref_val, alt_val, formula, pseudocount):
    """Compute effect size."""
    if formula == 'log2fc':
        return math.log2((alt_val + pseudocount) / (ref_val + pseudocount))
    elif formula == 'logfc':
        return math.log((alt_val + pseudocount) / (ref_val + pseudocount))
    else:  # diff
        return alt_val - ref_val


def score_window_tracks(ref_output, alt_output, chrom, center_pos):
    """Score all window-based tracks. Returns {layer: [scores]}."""
    layer_scores = {}
    pred_start = center_pos - INPUT_LENGTH // 2

    for ot_name, track_list in tracks_by_ot.items():
        layer, window, formula, pc = LAYER_SPEC[ot_name]
        if window is None:
            continue  # RNA-seq handled separately

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
            res = t['resolution']
            n_bins = rv.shape[0]
            center_bin = n_bins // 2
            hw = window // (2 * res)
            ws, we = max(0, center_bin - hw), min(n_bins, center_bin + hw + 1)

            ref_sum = float(np.sum(rv[ws:we, idx]))
            alt_sum = float(np.sum(av[ws:we, idx]))
            score = compute_effect(ref_sum, alt_sum, formula, pc)
            layer_scores.setdefault(layer, []).append(score)

    return layer_scores


def score_rna_tracks(ref_output, alt_output, chrom, center_pos):
    """Score RNA-seq tracks using exon counting. Returns {layer: [scores]}."""
    layer_scores = {}
    pred_start = center_pos - INPUT_LENGTH // 2
    pred_end = center_pos + INPUT_LENGTH // 2

    # Find genes with exons in prediction window
    genes = get_genes_in_window(chrom, pred_start, pred_end)
    if not genes:
        return layer_scores

    for ot_name in ('RNA_SEQ',):
        if ot_name not in tracks_by_ot:
            continue
        ot_enum = OutputType[ot_name]
        rd, ad = ref_output.get(ot_enum), alt_output.get(ot_enum)
        if rd is None or ad is None:
            continue
        rv, av = np.asarray(rd.values), np.asarray(ad.values)
        n_out = rv.shape[1]
        n_bins = rv.shape[0]
        _, _, formula, pc = LAYER_SPEC[ot_name]

        for t in tracks_by_ot[ot_name]:
            idx = t['local_idx']
            if idx >= n_out:
                continue
            res = 1  # RNA_SEQ is 1bp resolution

            # For each gene, sum signal across its exons
            for gene_name, exon_intervals in genes.items():
                ref_total = 0.0
                alt_total = 0.0
                n_exons = 0

                for exon_start, exon_end in exon_intervals:
                    # Convert genomic coords to array indices
                    bin_start = (exon_start - pred_start) // res
                    bin_end = (exon_end - pred_start) // res
                    bin_start = max(0, min(bin_start, n_bins))
                    bin_end = max(0, min(bin_end, n_bins))
                    if bin_start >= bin_end:
                        continue

                    ref_total += float(np.sum(rv[bin_start:bin_end, idx]))
                    alt_total += float(np.sum(av[bin_start:bin_end, idx]))
                    n_exons += 1

                if n_exons > 0:
                    ref_mean = ref_total / n_exons
                    alt_mean = alt_total / n_exons
                    score = compute_effect(ref_mean, alt_mean, formula, pc)
                    layer_scores.setdefault('gene_expression', []).append(score)

    return layer_scores


def extract_all_baselines(output, chrom, center_pos):
    """Extract baseline signals from all tracks including RNA-seq."""
    layer_signals = {}
    pred_start = center_pos - INPUT_LENGTH // 2
    pred_end = center_pos + INPUT_LENGTH // 2

    for ot_name, track_list in tracks_by_ot.items():
        layer, window, _, _ = LAYER_SPEC[ot_name]
        ot_enum = OutputType[ot_name]
        d = output.get(ot_enum)
        if d is None:
            continue
        vals = np.asarray(d.values)
        n_out = vals.shape[1]
        n_bins = vals.shape[0]

        if window is not None:
            # Window-based: extract central window signal
            for t in track_list:
                idx = t['local_idx']
                if idx >= n_out:
                    continue
                res = t['resolution']
                c = n_bins // 2
                hw = window // (2 * res)
                ws, we = max(0, c - hw), min(n_bins, c + hw + 1)
                signal = float(np.sum(vals[ws:we, idx]))
                layer_signals.setdefault(layer, []).append(signal)
        else:
            # RNA-seq: sum across exons of genes in window
            genes = get_genes_in_window(chrom, pred_start, pred_end)
            for t in track_list:
                idx = t['local_idx']
                if idx >= n_out:
                    continue
                for gene_name, exon_intervals in genes.items():
                    total = 0.0
                    n_exons = 0
                    for exon_start, exon_end in exon_intervals:
                        bs = max(0, min((exon_start - pred_start), n_bins))
                        be = max(0, min((exon_end - pred_start), n_bins))
                        if bs < be:
                            total += float(np.sum(vals[bs:be, idx]))
                            n_exons += 1
                    if n_exons > 0:
                        layer_signals.setdefault(layer, []).append(total / n_exons)

    return layer_signals


cache_dir = os.path.expanduser("~/.chorus/backgrounds")
os.makedirs(cache_dir, exist_ok=True)


def save_backgrounds(layer_data, suffix=""):
    """Save sorted background arrays. All unsigned (abs for magnitude ranking)."""
    for layer, scores in layer_data.items():
        if len(scores) < 10:
            logger.warning("Skipping %s: only %d scores", layer, len(scores))
            continue
        arr = np.abs(np.array(scores, dtype=np.float64))
        arr.sort()
        fname = f"alphagenome_{layer}{suffix}.npy"
        np.save(os.path.join(cache_dir, fname), arr)
        logger.info("  %s: %s scores, median=%.4f, range=[%.4f, %.4f]",
                     layer, f"{len(arr):,}", float(np.median(arr)), arr.min(), arr.max())


# ═══════════════════════════════════════════════════════════════════
# PART 1: VARIANT EFFECT BACKGROUNDS
# ═══════════════════════════════════════════════════════════════════
if args.part in ("variants", "both"):
    logger.info("=" * 60)
    logger.info("VARIANT BACKGROUNDS: %d SNPs x %d tracks (all output types)",
                args.n_variants, total_tracks)
    logger.info("=" * 60)

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
            snps.append({"chrom": chrom, "pos": pos, "ref": ref_base,
                         "alt": random.choice([b for b in "ACGT" if b != ref_base])})
    random.shuffle(snps)
    snps = snps[:args.n_variants]
    logger.info("Generated %d random SNPs", len(snps))

    variant_scores = {}
    t0 = time.time()

    for i, snp in enumerate(snps):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
            eta = (len(snps) - i - 1) / rate if rate > 0 else 0
            n = sum(len(v) for v in variant_scores.values())
            logger.info("Variant %d/%d — %.1f min, ETA %.0f min, %s scores",
                        i + 1, len(snps), elapsed / 60, eta, f"{n:,}")

        seq_ref = get_sequence(snp["chrom"], snp["pos"])
        if seq_ref is None:
            continue
        offset = snp["pos"] - (snp["pos"] - INPUT_LENGTH // 2) - 1
        seq_alt = seq_ref[:offset] + snp["alt"] + seq_ref[offset + 1:]

        try:
            ro, ao = predict_all(seq_ref), predict_all(seq_alt)

            # Window-based tracks (chromatin, TF, histone, TSS, splicing)
            for layer, vals in score_window_tracks(ro, ao, snp["chrom"], snp["pos"]).items():
                variant_scores.setdefault(layer, []).extend(vals)

            # RNA-seq exon counting
            for layer, vals in score_rna_tracks(ro, ao, snp["chrom"], snp["pos"]).items():
                variant_scores.setdefault(layer, []).extend(vals)

        except Exception as exc:
            logger.warning("Failed variant %d: %s", i, str(exc)[:150])

    elapsed_v = time.time() - t0
    logger.info("Variant scoring complete in %.1f hours (%s total scores)",
                elapsed_v / 3600, f"{sum(len(v) for v in variant_scores.values()):,}")
    save_backgrounds(variant_scores)


# ═══════════════════════════════════════════════════════════════════
# PART 2: BASELINE SIGNAL BACKGROUNDS
# ═══════════════════════════════════════════════════════════════════
if args.part in ("baselines", "both"):
    logger.info("=" * 60)
    logger.info("BASELINE BACKGROUNDS: TSS + random positions x %d tracks", total_tracks)
    logger.info("=" * 60)

    all_positions = []

    # Protein-coding TSSs
    if args.max_tss > 0:
        tss_positions = []
        pc_genes_df = gene_df[gene_df['gene_type'] == 'protein_coding']
        for _, row in pc_genes_df.iterrows():
            chrom = row['chrom']
            if not chrom.startswith('chr') or chrom in ('chrM', 'chrY'):
                continue
            tss = int(row['start']) if row['strand'] == '+' else int(row['end'])
            tss_positions.append((chrom, tss))
        random.seed(456)
        if len(tss_positions) > args.max_tss:
            tss_positions = random.sample(tss_positions, args.max_tss)
        all_positions.extend(tss_positions)
        logger.info("Added %d protein-coding TSS positions", len(tss_positions))

    # Random genomic positions
    random.seed(789)
    chroms = [f"chr{i}" for i in range(1, 23)]
    rand_per_chrom = args.n_random_positions // len(chroms) + 1
    rand_count = 0
    for chrom in chroms:
        chrom_len = ref.get_reference_length(chrom)
        max_pos = min(chrom_len - 10_000_000, 200_000_000)
        if max_pos <= 10_000_000:
            max_pos = chrom_len - 1_000_000
        for _ in range(rand_per_chrom):
            if rand_count >= args.n_random_positions:
                break
            all_positions.append((chrom, random.randint(10_000_000, max_pos)))
            rand_count += 1
    logger.info("Added %d random positions", rand_count)
    logger.info("Total: %d baseline positions", len(all_positions))
    random.shuffle(all_positions)

    baseline_signals = {}
    t0 = time.time()

    for i, (chrom, pos) in enumerate(all_positions):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
            eta = (len(all_positions) - i - 1) / rate if rate > 0 else 0
            n = sum(len(v) for v in baseline_signals.values())
            logger.info("Baseline %d/%d (%s:%d) — %.1f min, ETA %.0f min, %s samples",
                        i + 1, len(all_positions), chrom, pos,
                        elapsed / 60, eta, f"{n:,}")

        seq = get_sequence(chrom, pos)
        if seq is None:
            continue

        try:
            output = predict_all(seq)
            for layer, vals in extract_all_baselines(output, chrom, pos).items():
                baseline_signals.setdefault(layer, []).extend(vals)
        except Exception as exc:
            logger.warning("Failed %s:%d: %s", chrom, pos, str(exc)[:150])

    elapsed_b = time.time() - t0
    logger.info("Baseline complete in %.1f hours (%s total samples)",
                elapsed_b / 3600, f"{sum(len(v) for v in baseline_signals.values()):,}")
    save_backgrounds(baseline_signals, suffix="_baseline")

ref.close()
logger.info("=" * 60)
logger.info("ALL DONE — files saved to %s", cache_dir)
for f in sorted(os.listdir(cache_dir)):
    sz = os.path.getsize(os.path.join(cache_dir, f)) / 1024
    logger.info("  %s (%.1f KB)", f, sz)
logger.info("=" * 60)
