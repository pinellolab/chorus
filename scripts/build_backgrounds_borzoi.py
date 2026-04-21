"""Build per-track background distributions for Borzoi.

Produces ``borzoi_pertrack.npz`` with three CDF matrices (effect, summary,
perbin) per track for all 7,611 Borzoi tracks: CAGE, RNA, DNASE, ATAC,
CHIP-TF, CHIP-Histone.

RNA-seq tracks use **exon-precise sampling**: only bins overlapping
GENCODE exons of protein-coding genes contribute to RNA perbin and
summary CDFs.

Resolution: 32 bp bins, output: 6,144 bins = 196,608 bp.
Input: 524,288 bp.

Run in chorus-borzoi env:
  mamba run -n chorus-borzoi python scripts/build_backgrounds_borzoi.py --part variants --gpu 0
  mamba run -n chorus-borzoi python scripts/build_backgrounds_borzoi.py --part baselines --gpu 0
  mamba run -n chorus python scripts/build_backgrounds_borzoi.py --part merge
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

import os; REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, REPO_ROOT)
os.environ["CHORUS_NO_TIMEOUT"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--part", choices=["variants", "baselines", "merge", "both", "all"], default="all")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--n-variants", type=int, default=10000)
parser.add_argument("--reservoir-size", type=int, default=50000)
parser.add_argument("--n-cdf-points", type=int, default=10000)
args = parser.parse_args()

log_dir = os.path.join(REPO_ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/bg_borzoi_{args.part}_gpu{args.gpu}.log", mode='w'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
INPUT_LENGTH = 524_288
BIN_SIZE = 32
PERBIN_BINS_PER_POSITION = 32  # random bins to sample per position

# Histone patterns for CHIP classification
HISTONE_PATTERNS = frozenset({
    "H2AFZ", "H2AZ",
    "H3K4ME1", "H3K4ME2", "H3K4ME3",
    "H3K9AC", "H3K9ME1", "H3K9ME2", "H3K9ME3",
    "H3K14AC",
    "H3K27AC", "H3K27ME3",
    "H3K36ME3",
    "H3K79ME2",
    "H4K20ME1",
})

# Layer -> (window_bp, formula, pseudocount, signed)
LAYER_SPEC = {
    'DNASE':     (501,  'log2fc', 1.0, False),
    'ATAC':      (501,  'log2fc', 1.0, False),
    'CAGE':      (501,  'log2fc', 1.0, False),
    'RNA':       (None, 'logfc',  0.001, True),  # exon-based
    'CHIP_TF':   (501,  'log2fc', 1.0, False),
    'CHIP_HIST': (2001, 'log2fc', 1.0, False),
}

cache_dir = os.path.expanduser("~/.chorus/backgrounds")
os.makedirs(cache_dir, exist_ok=True)


# ── Reservoir sampler ─────────────────────────────────────────────

class ReservoirSampler:
    """Fixed-capacity reservoir sampler per track."""

    def __init__(self, n_tracks: int, capacity: int = 50_000):
        self.n_tracks = n_tracks
        self.capacity = capacity
        self.data = [[] for _ in range(n_tracks)]
        self.counts = np.zeros(n_tracks, dtype=np.int64)
        self._rng = random.Random(12345)

    def add(self, track_idx: int, value: float):
        n = self.counts[track_idx]
        if n < self.capacity:
            self.data[track_idx].append(value)
        else:
            j = self._rng.randint(0, n)
            if j < self.capacity:
                self.data[track_idx][j] = value
        self.counts[track_idx] += 1

    def add_batch(self, track_idx: int, values):
        for v in values:
            self.add(track_idx, float(v))

    def get_sorted(self, track_idx: int) -> np.ndarray:
        arr = np.array(self.data[track_idx], dtype=np.float64)
        arr.sort()
        return arr

    def to_cdf_matrix(self, n_points: int = 10_000) -> np.ndarray:
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


# ══════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════

def load_model_and_metadata():
    """Load Borzoi model + metadata + reference. Returns (predict_fn, get_seq, ref, track_info, OUTPUT_BINS, N_TRACKS)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from borzoi_pytorch import Borzoi

    logger.info("Loading Borzoi fold %d on GPU %d...", args.fold, args.gpu)
    t_load = time.time()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Borzoi.from_pretrained(f'johahi/borzoi-replicate-{args.fold}')
    model.to(device)
    model.eval()

    from chorus.oracles.borzoi_source.borzoi_metadata import BorzoiMetadata
    metadata = BorzoiMetadata()

    import pysam
    ref_path = os.path.join(REPO_ROOT, "genomes/hg38.fa")
    ref = pysam.FastaFile(ref_path)

    logger.info("Model loaded in %.1f seconds", time.time() - t_load)

    # Detect output dimensions
    dummy_ohe = np.zeros((INPUT_LENGTH, 4), dtype='bool')
    dummy_ohe[:, 0] = True
    dummy_tensor = torch.from_numpy(dummy_ohe).permute(1, 0).float().unsqueeze(0).to(device)
    with torch.no_grad():
        dummy_out = model(dummy_tensor).squeeze(0)
    OUTPUT_BINS = dummy_out.shape[1]
    N_TRACKS = dummy_out.shape[0]
    del dummy_ohe, dummy_tensor, dummy_out
    torch.cuda.empty_cache()
    logger.info("Output: %d bins x %d tracks (bin_size=%d)", OUTPUT_BINS, N_TRACKS, BIN_SIZE)

    # Build track info
    track_info = []
    for _, row in metadata.tracks_df.iterrows():
        track_idx = int(row['index'])
        identifier = str(row.get('identifier', f'borzoi_{track_idx}'))
        desc = str(row.get('description', ''))
        assay_type = desc.split(':')[0] if ':' in desc else 'unknown'

        if assay_type == 'CHIP':
            upper_desc = desc.upper()
            is_histone = any(p in upper_desc for p in HISTONE_PATTERNS)
            spec_key = 'CHIP_HIST' if is_histone else 'CHIP_TF'
        elif assay_type in LAYER_SPEC:
            spec_key = assay_type
        else:
            continue

        window, formula, pseudocount, signed = LAYER_SPEC[spec_key]

        # Layer name (matches scorers.py)
        if spec_key == 'CHIP_TF':
            layer_name = 'tf_binding'
        elif spec_key == 'CHIP_HIST':
            layer_name = 'histone_marks'
        elif spec_key == 'RNA':
            layer_name = 'gene_expression'
        elif spec_key == 'CAGE':
            layer_name = 'tss_activity'
        else:
            layer_name = 'chromatin_accessibility'

        track_info.append({
            'idx': track_idx,
            'identifier': identifier,
            'assay_type': assay_type,
            'spec_key': spec_key,
            'layer': layer_name,
            'window': window,
            'formula': formula,
            'pseudocount': pseudocount,
            'signed': signed,
        })

    track_info.sort(key=lambda t: t['idx'])

    def dna_1hot(seq):
        seq = seq.upper()
        seq_len = len(seq)
        oh = np.zeros((seq_len, 4), dtype='bool')
        for i in range(seq_len):
            if seq[i] == 'A':   oh[i, 0] = True
            elif seq[i] == 'C': oh[i, 1] = True
            elif seq[i] == 'G': oh[i, 2] = True
            elif seq[i] == 'T': oh[i, 3] = True
        return oh

    def predict_all(seq):
        ohe = dna_1hot(seq)
        ohe_tensor = torch.from_numpy(ohe).permute(1, 0).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(ohe_tensor).squeeze(0)
        # (n_tracks, n_bins) -> (n_bins, n_tracks)
        return pred.permute(1, 0).cpu().numpy()

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

    return predict_all, get_sequence, ref, track_info, OUTPUT_BINS, N_TRACKS


def compute_effect(ref_val, alt_val, formula, pseudocount):
    if formula == 'log2fc':
        return math.log2((alt_val + pseudocount) / (ref_val + pseudocount))
    elif formula == 'logfc':
        return math.log((alt_val + pseudocount) / (ref_val + pseudocount))
    else:
        return alt_val - ref_val


def get_window_slice(track, output_bins):
    """Central scoring window slice for non-RNA tracks."""
    if track['window'] is None:
        return 0, output_bins
    center_bin = output_bins // 2
    hw = track['window'] // (2 * BIN_SIZE)
    ws = max(0, center_bin - hw)
    we = min(output_bins, center_bin + hw + 1)
    return ws, we


# ══════════════════════════════════════════════════════════════════
# Exon-precise RNA helpers
# ══════════════════════════════════════════════════════════════════

def load_exon_index():
    """Load merged protein-coding exons keyed by chromosome.

    Returns dict: chrom -> sorted list of (start, end) intervals.
    """
    from chorus.utils.annotations import get_annotation_manager
    ann_manager = get_annotation_manager()
    gtf_path = ann_manager.get_annotation_path('gencode_v48_basic')

    exon_df = ann_manager._get_exons_df(gtf_path)
    gene_df = ann_manager._get_genes_df(gtf_path)
    pc_gene_names = set(gene_df[gene_df['gene_type'] == 'protein_coding']['gene_name'])
    pc_exons = exon_df[exon_df['gene_name'].isin(pc_gene_names)]

    # Per-chromosome merged exons
    by_chrom = defaultdict(list)
    for chrom, group in pc_exons.groupby('chrom'):
        intervals = sorted(zip(group['start'].tolist(), group['end'].tolist()))
        merged = []
        for s, e in intervals:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
            else:
                merged.append((s, e))
        by_chrom[chrom] = merged

    total_exons = sum(len(v) for v in by_chrom.values())
    logger.info("Loaded %d merged protein-coding exons across %d chromosomes",
                total_exons, len(by_chrom))
    return by_chrom


def exon_bins_for_window(exon_index, chrom, pred_start, pred_end, output_bins):
    """Return bin indices that overlap exons within [pred_start, pred_end).

    Args:
        exon_index: chrom -> sorted [(s, e), ...]
        chrom: chromosome
        pred_start: window start (genomic)
        pred_end: window end (genomic)
        output_bins: total bins in the output window

    Returns:
        Sorted numpy array of bin indices that overlap any exon.
    """
    exons = exon_index.get(chrom, [])
    if not exons:
        return np.array([], dtype=np.int64)
    bin_set = set()
    for es, ee in exons:
        if es >= pred_end:
            break
        if ee <= pred_start:
            continue
        # Overlap
        start = max(es, pred_start)
        end = min(ee, pred_end)
        bs = max(0, (start - pred_start) // BIN_SIZE)
        be = min(output_bins, ((end - pred_start) + BIN_SIZE - 1) // BIN_SIZE)
        for b in range(bs, be):
            bin_set.add(b)
    return np.array(sorted(bin_set), dtype=np.int64)


# ══════════════════════════════════════════════════════════════════
# PART 1: VARIANT EFFECT BACKGROUNDS
# ══════════════════════════════════════════════════════════════════

def build_variant_backgrounds():
    predict_all, get_sequence, ref, track_info, output_bins, n_tracks_total = load_model_and_metadata()
    n_tracks = len(track_info)

    logger.info("=" * 60)
    logger.info("PER-TRACK VARIANT BACKGROUNDS: %d SNPs x %d tracks", args.n_variants, n_tracks)
    logger.info("=" * 60)

    effect_reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size)

    # Load exon index for RNA tracks
    exon_index = load_exon_index()

    # Generate random SNPs
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

    # Use track index in track_info as the row index for the reservoir
    # (track_info is already sorted by 'idx', and we pre-built it above)
    info_idx_by_track_idx = {t['idx']: i for i, t in enumerate(track_info)}

    t0 = time.time()
    for i, snp in enumerate(snps):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
            eta = (len(snps) - i - 1) / rate if rate > 0 else 0
            logger.info("Variant %d/%d — %.1f min, ETA %.0f min, %s effect samples",
                        i + 1, len(snps), elapsed / 60, eta,
                        f"{effect_reservoir.total_samples():,}")

        seq_ref = get_sequence(snp["chrom"], snp["pos"])
        if seq_ref is None:
            continue
        offset = INPUT_LENGTH // 2 - 1
        seq_alt = seq_ref[:offset] + snp["alt"] + seq_ref[offset + 1:]

        try:
            ref_pred = predict_all(seq_ref)
            alt_pred = predict_all(seq_alt)

            pred_start = snp["pos"] - INPUT_LENGTH // 2
            pred_end = snp["pos"] + INPUT_LENGTH // 2

            # Pre-compute exon bins for this position (for RNA tracks)
            rna_exon_bins = exon_bins_for_window(
                exon_index, snp["chrom"], pred_start, pred_end, output_bins,
            )

            for t_i, t in enumerate(track_info):
                idx = t['idx']
                if t['layer'] == 'gene_expression':
                    # RNA-seq: mean over exon bins (genome-wide null)
                    if len(rna_exon_bins) == 0:
                        continue
                    ref_val = float(np.mean(ref_pred[rna_exon_bins, idx]))
                    alt_val = float(np.mean(alt_pred[rna_exon_bins, idx]))
                else:
                    ws, we = get_window_slice(t, output_bins)
                    ref_val = float(np.sum(ref_pred[ws:we, idx]))
                    alt_val = float(np.sum(alt_pred[ws:we, idx]))

                score = compute_effect(ref_val, alt_val, t['formula'], t['pseudocount'])
                if not t['signed']:
                    score = abs(score)
                effect_reservoir.add(t_i, score)
        except Exception as exc:
            logger.warning("Failed variant %d: %s", i, str(exc)[:150])

    elapsed_v = time.time() - t0
    logger.info(
        "Variant scoring complete in %.1f hours: %s total samples across %d tracks",
        elapsed_v / 3600,
        f"{effect_reservoir.total_samples():,}",
        effect_reservoir.tracks_with_data(),
    )

    effect_matrix = effect_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
    track_ids = [t['identifier'] for t in track_info]
    signed_flags = np.array([t['signed'] for t in track_info], dtype=bool)

    interim_path = os.path.join(cache_dir, "borzoi_effect_cdfs_interim.npz")
    np.savez_compressed(
        interim_path,
        track_ids=np.array(track_ids, dtype='U'),
        effect_cdfs=effect_matrix.astype(np.float32),
        effect_counts=effect_reservoir.get_counts(),
        signed_flags=signed_flags,
    )
    logger.info("Saved interim effect CDFs: %s (%.1f MB)",
                interim_path, os.path.getsize(interim_path) / (1024 * 1024))

    ref.close()
    return interim_path


# ══════════════════════════════════════════════════════════════════
# PART 2: BASELINE BACKGROUNDS
# ══════════════════════════════════════════════════════════════════

def build_baseline_backgrounds():
    predict_all, get_sequence, ref, track_info, output_bins, n_tracks_total = load_model_and_metadata()
    n_tracks = len(track_info)

    logger.info("=" * 60)
    logger.info("PER-TRACK BASELINE BACKGROUNDS for Borzoi (%d tracks)", n_tracks)
    logger.info("=" * 60)

    summary_reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size)
    perbin_reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size)

    rng_bins = np.random.RandomState(999)

    # CAGE track set for summary routing
    cage_track_indices = set(
        t['idx'] for t in track_info if t['assay_type'] == 'CAGE'
    )
    rna_track_indices = set(
        t['idx'] for t in track_info if t['layer'] == 'gene_expression'
    )
    logger.info("CAGE tracks: %d, RNA tracks: %d, other: %d",
                len(cage_track_indices), len(rna_track_indices),
                n_tracks - len(cage_track_indices) - len(rna_track_indices))

    # Load exon index for RNA-seq precise sampling
    exon_index = load_exon_index()

    # ── Position set 1: Random genomic ──
    n_random = 15_000
    random.seed(789)
    chroms = [f"chr{i}" for i in range(1, 23)]
    rand_per_chrom = n_random // len(chroms) + 1
    rand_positions = []
    for chrom in chroms:
        chrom_len = ref.get_reference_length(chrom)
        max_pos = min(chrom_len - 10_000_000, 200_000_000)
        if max_pos <= 10_000_000:
            max_pos = chrom_len - 1_000_000
        for _ in range(rand_per_chrom):
            if len(rand_positions) >= n_random:
                break
            rand_positions.append((chrom, random.randint(10_000_000, max_pos)))
    logger.info("Random positions: %d", len(rand_positions))

    # ── Position set 2: SCREEN cCREs (per-category) ──
    from chorus.utils.annotations import sample_ccre_positions
    ccre_positions = sample_ccre_positions(
        n_per_category={
            "PLS": 3000, "dELS": 2500, "pELS": 1500,
            "CA-CTCF": 1500, "CA-TF": 1000, "TF": 500,
            "CA-H3K4me3": 1000, "CA": 500,
        },
        seed=456,
    )
    logger.info("cCRE positions: %d", len(ccre_positions))

    # ── Position set 3: Protein-coding TSSs ──
    from chorus.utils.annotations import get_annotation_manager
    ann_manager = get_annotation_manager()
    gtf_path = ann_manager.get_annotation_path('gencode_v48_basic')
    gene_df = ann_manager._get_genes_df(gtf_path)
    pc_genes = gene_df[gene_df['gene_type'] == 'protein_coding'].copy()
    pc_genes['tss'] = pc_genes.apply(
        lambda r: r['start'] if r['strand'] == '+' else r['end'], axis=1)
    valid_chroms = {f"chr{i}" for i in range(1, 23)}
    pc_genes = pc_genes[pc_genes['chrom'].isin(valid_chroms)]
    tss_dedup = pc_genes.groupby('gene_name').first().reset_index()
    rng_tss = random.Random(111)
    tss_list = list(zip(tss_dedup['chrom'], tss_dedup['tss']))
    if len(tss_list) > 3000:
        tss_list = rng_tss.sample(tss_list, 3000)
    logger.info("TSS positions: %d", len(tss_list))

    # ── Position set 4: Gene body midpoints ──
    long_genes = pc_genes[(pc_genes['end'] - pc_genes['start']) > 10000].copy()
    long_genes['midpoint'] = (long_genes['start'] + long_genes['end']) // 2
    gb_dedup = long_genes.groupby('gene_name').first().reset_index()
    rng_gb = random.Random(222)
    gb_list = list(zip(gb_dedup['chrom'], gb_dedup['midpoint']))
    if len(gb_list) > 2000:
        gb_list = rng_gb.sample(gb_list, 2000)
    logger.info("Gene body midpoints: %d", len(gb_list))

    # Combine
    tagged_positions = []
    for chrom, pos in rand_positions:
        tagged_positions.append((chrom, pos, 'random'))
    for chrom, pos in ccre_positions:
        tagged_positions.append((chrom, pos, 'ccre'))
    for chrom, pos in tss_list:
        tagged_positions.append((chrom, int(pos), 'tss'))
    for chrom, pos in gb_list:
        tagged_positions.append((chrom, int(pos), 'gene_body'))
    random.shuffle(tagged_positions)

    n_by_type = defaultdict(int)
    for _, _, pt in tagged_positions:
        n_by_type[pt] += 1
    logger.info("Total: %d positions — %s", len(tagged_positions),
                ", ".join(f"{k}={v}" for k, v in sorted(n_by_type.items())))

    info_idx_by_track_idx = {t['idx']: i for i, t in enumerate(track_info)}

    t0 = time.time()
    for i, (chrom, pos, pos_type) in enumerate(tagged_positions):
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
            eta = (len(tagged_positions) - i - 1) / rate if rate > 0 else 0
            logger.info(
                "Baseline %d/%d (%s:%d %s) — %.1f min, ETA %.0f min, "
                "%s summary + %s perbin samples",
                i + 1, len(tagged_positions), chrom, pos, pos_type,
                elapsed / 60, eta,
                f"{summary_reservoir.total_samples():,}",
                f"{perbin_reservoir.total_samples():,}",
            )

        seq = get_sequence(chrom, pos)
        if seq is None:
            continue

        try:
            pred = predict_all(seq)

            pred_start = pos - INPUT_LENGTH // 2
            pred_end = pos + INPUT_LENGTH // 2

            # Random bin sample for non-RNA perbin
            bin_sample = rng_bins.choice(output_bins, PERBIN_BINS_PER_POSITION, replace=False)

            # Exon bins for RNA perbin (precise sampling)
            rna_exon_bins = exon_bins_for_window(
                exon_index, chrom, pred_start, pred_end, output_bins,
            )
            # Subsample exon bins to keep similar order of magnitude
            rna_bin_sample = None
            if len(rna_exon_bins) > 0:
                if len(rna_exon_bins) > PERBIN_BINS_PER_POSITION:
                    rna_bin_sample = rng_bins.choice(
                        rna_exon_bins, PERBIN_BINS_PER_POSITION, replace=False,
                    )
                else:
                    rna_bin_sample = rna_exon_bins

            for t_i, t in enumerate(track_info):
                idx = t['idx']
                is_cage = idx in cage_track_indices
                is_rna = idx in rna_track_indices

                # ── Summary CDF ──
                if is_rna:
                    # RNA: mean over all exon bins in window
                    if len(rna_exon_bins) > 0:
                        signal = float(np.mean(pred[rna_exon_bins, idx]))
                        summary_reservoir.add(t_i, signal)
                else:
                    # CAGE skips cCRE positions for summary
                    if not (is_cage and pos_type == 'ccre'):
                        ws, we = get_window_slice(t, output_bins)
                        signal = float(np.sum(pred[ws:we, idx]))
                        summary_reservoir.add(t_i, signal)

                # ── Perbin CDF ──
                if is_rna:
                    # RNA: only exon bins
                    if rna_bin_sample is not None and len(rna_bin_sample) > 0:
                        perbin_reservoir.add_batch(t_i, pred[rna_bin_sample, idx].astype(np.float64))
                else:
                    perbin_reservoir.add_batch(t_i, pred[bin_sample, idx].astype(np.float64))
        except Exception as exc:
            logger.warning("Failed %s:%d: %s", chrom, pos, str(exc)[:150])

    elapsed_b = time.time() - t0
    logger.info(
        "Baseline complete in %.1f hours: %s summary + %s perbin samples",
        elapsed_b / 3600,
        f"{summary_reservoir.total_samples():,}",
        f"{perbin_reservoir.total_samples():,}",
    )

    track_ids = [t['identifier'] for t in track_info]
    summary_matrix = summary_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
    perbin_matrix = perbin_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)

    interim_path = os.path.join(cache_dir, "borzoi_baseline_cdfs_interim.npz")
    np.savez_compressed(
        interim_path,
        track_ids=np.array(track_ids, dtype='U'),
        summary_cdfs=summary_matrix.astype(np.float32),
        summary_counts=summary_reservoir.get_counts(),
        perbin_cdfs=perbin_matrix.astype(np.float32),
        perbin_counts=perbin_reservoir.get_counts(),
    )
    logger.info("Saved interim baseline CDFs: %s (%.1f MB)",
                interim_path, os.path.getsize(interim_path) / (1024 * 1024))

    ref.close()
    return interim_path


# ══════════════════════════════════════════════════════════════════
# PART 3: MERGE
# ══════════════════════════════════════════════════════════════════

def merge_to_final():
    from chorus.analysis.normalization import PerTrackNormalizer

    effect_path = os.path.join(cache_dir, "borzoi_effect_cdfs_interim.npz")
    baseline_path = os.path.join(cache_dir, "borzoi_baseline_cdfs_interim.npz")

    if not os.path.exists(effect_path):
        logger.error("Missing %s — run --part variants first", effect_path)
        return
    if not os.path.exists(baseline_path):
        logger.error("Missing %s — run --part baselines first", baseline_path)
        return

    logger.info("=" * 60)
    logger.info("MERGING interim files into borzoi_pertrack.npz")
    logger.info("=" * 60)

    effect_data = np.load(effect_path, allow_pickle=False)
    baseline_data = np.load(baseline_path, allow_pickle=False)

    effect_ids = list(effect_data["track_ids"].astype(str))
    baseline_ids = list(baseline_data["track_ids"].astype(str))
    assert effect_ids == baseline_ids, "Track ID mismatch!"

    track_ids = effect_ids
    effect_cdfs = effect_data["effect_cdfs"]
    signed_flags = effect_data["signed_flags"]
    effect_counts = effect_data["effect_counts"] if "effect_counts" in effect_data else None
    summary_cdfs = baseline_data["summary_cdfs"]
    summary_counts = baseline_data["summary_counts"] if "summary_counts" in baseline_data else None
    perbin_cdfs = baseline_data["perbin_cdfs"]
    perbin_counts = baseline_data["perbin_counts"] if "perbin_counts" in baseline_data else None

    logger.info("  Tracks: %d", len(track_ids))
    logger.info("  Effect CDFs: %s (%.1f MB)", effect_cdfs.shape, effect_cdfs.nbytes / 1e6)
    logger.info("  Summary CDFs: %s (%.1f MB)", summary_cdfs.shape, summary_cdfs.nbytes / 1e6)
    logger.info("  Perbin CDFs: %s (%.1f MB)", perbin_cdfs.shape, perbin_cdfs.nbytes / 1e6)
    if effect_counts is not None:
        logger.info("  Effect counts: min=%d, median=%d, max=%d",
                     effect_counts.min(), int(np.median(effect_counts)), effect_counts.max())

    path = PerTrackNormalizer.build_and_save(
        oracle_name="borzoi",
        track_ids=track_ids,
        effect_cdfs=effect_cdfs,
        summary_cdfs=summary_cdfs,
        perbin_cdfs=perbin_cdfs,
        signed_flags=signed_flags,
        effect_counts=effect_counts,
        summary_counts=summary_counts,
        perbin_counts=perbin_counts,
        cache_dir=cache_dir,
    )

    norm = PerTrackNormalizer(cache_dir=cache_dir)
    entry = norm._ensure_loaded("borzoi")
    logger.info("Verification: loaded %d tracks, CDFs: %s",
                len(entry["track_ids"]),
                [k for k in ("effect_cdfs", "summary_cdfs", "perbin_cdfs") if entry[k] is not None])

    logger.info("=" * 60)
    logger.info("DONE — final file: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    logger.info("=" * 60)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if args.part == "variants":
    build_variant_backgrounds()
elif args.part == "baselines":
    build_baseline_backgrounds()
elif args.part == "merge":
    merge_to_final()
elif args.part in ("both", "all"):
    build_variant_backgrounds()
    build_baseline_backgrounds()
    merge_to_final()
