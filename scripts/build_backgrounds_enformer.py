"""Build per-track background distributions for Enformer.

Produces a single ``enformer_pertrack.npz`` file containing three CDF
matrices (effect, summary, perbin) with one row per track (5,313 tracks).
Each CDF row is 10,000 sorted values sampled via reservoir sampling.

Enformer specs: 5,313 human tracks, 128 bp bins, 896 output bins = 114,688 bp.
Input: 393,216 bp.

Run in chorus-enformer env:
  mamba run -n chorus-enformer python scripts/build_backgrounds_enformer.py --part variants --gpu 0
  mamba run -n chorus-enformer python scripts/build_backgrounds_enformer.py --part baselines --gpu 1
  mamba run -n chorus-enformer python scripts/build_backgrounds_enformer.py --part merge

The ``--part merge`` step combines variant and baseline reservoirs into
the final ``enformer_pertrack.npz`` file.
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
parser.add_argument("--n-variants", type=int, default=10000)
parser.add_argument("--n-random-positions", type=int, default=5000)
parser.add_argument("--reservoir-size", type=int, default=50000,
                    help="Max samples per track before compaction to 10K CDF")
parser.add_argument("--n-cdf-points", type=int, default=10000,
                    help="Final CDF resolution (points per track)")
args = parser.parse_args()

log_dir = os.path.join(REPO_ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/bg_enformer_{args.part}_gpu{args.gpu}.log", mode='w'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────
INPUT_LENGTH = 393_216
OUTPUT_BINS = 896
BIN_SIZE = 128
PERBIN_BINS_PER_POSITION = 32  # random bins to sample per position for perbin CDF

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
    'DNASE':    (501,  'log2fc', 1.0, False),
    'ATAC':     (501,  'log2fc', 1.0, False),
    'CAGE':     (501,  'log2fc', 1.0, False),
    'CHIP_TF':  (501,  'log2fc', 1.0, False),
    'CHIP_HIST':(2001, 'log2fc', 1.0, False),
}

cache_dir = os.path.expanduser("~/.chorus/backgrounds")
os.makedirs(cache_dir, exist_ok=True)


# ── Reservoir sampler ─────────────────────────────────────────────

class ReservoirSampler:
    """Fixed-capacity reservoir sampler per track.

    Collects up to ``capacity`` values per track using reservoir sampling
    so memory stays bounded regardless of how many positions are scored.
    """

    def __init__(self, n_tracks: int, capacity: int = 50_000):
        self.n_tracks = n_tracks
        self.capacity = capacity
        # Pre-allocate reservoirs for each track
        self.data = [[] for _ in range(n_tracks)]
        self.counts = np.zeros(n_tracks, dtype=np.int64)
        self._rng = random.Random(12345)

    def add(self, track_idx: int, value: float):
        """Add a single value for a track."""
        n = self.counts[track_idx]
        if n < self.capacity:
            self.data[track_idx].append(value)
        else:
            j = self._rng.randint(0, n)
            if j < self.capacity:
                self.data[track_idx][j] = value
        self.counts[track_idx] += 1

    def add_batch(self, track_idx: int, values):
        """Add multiple values for a track."""
        for v in values:
            self.add(track_idx, float(v))

    def get_sorted(self, track_idx: int) -> np.ndarray:
        """Return sorted values for a track."""
        arr = np.array(self.data[track_idx], dtype=np.float64)
        arr.sort()
        return arr

    def to_cdf_matrix(self, n_points: int = 10_000) -> np.ndarray:
        """Build (n_tracks, n_points) CDF matrix from all reservoirs.

        Each row is subsampled to exactly *n_points* evenly-spaced
        percentile positions from the track's sorted scores.

        Tracks with fewer than *n_points* samples are upsampled via
        linear interpolation to fill the full CDF, preserving correct
        percentile resolution (``rank / n_actual_samples``) rather than
        padding with max values that would create a flat tail.

        Tracks with no data get all-zero rows.
        """
        matrix = np.zeros((self.n_tracks, n_points), dtype=np.float64)
        target_quantiles = np.linspace(0, 1, n_points)
        for i in range(self.n_tracks):
            arr = self.get_sorted(i)
            n = len(arr)
            if n == 0:
                continue
            if n >= n_points:
                # Subsample: pick evenly-spaced indices
                indices = np.linspace(0, n - 1, n_points, dtype=int)
                matrix[i] = arr[indices]
            else:
                # Interpolate: map target quantiles to source positions
                # Source quantiles: rank / n for each of the n samples
                source_quantiles = np.arange(n) / n
                matrix[i] = np.interp(target_quantiles, source_quantiles, arr)
        return matrix

    def get_counts(self) -> np.ndarray:
        """Return per-track actual sample counts."""
        return self.counts.copy()

    def total_samples(self) -> int:
        return int(self.counts.sum())

    def tracks_with_data(self) -> int:
        return int((self.counts > 0).sum())


# ══════════════════════════════════════════════════════════════════
# Model loading (only for variants/baselines parts)
# ══════════════════════════════════════════════════════════════════

def load_model_and_metadata():
    """Load Enformer model and track metadata. Returns (predict_fn, track_info)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Pre-load nvidia CUDA libs
    try:
        import nvidia
        nvidia_dir = nvidia.__path__[0]
        for pkg in os.listdir(nvidia_dir):
            lib_dir = os.path.join(nvidia_dir, pkg, 'lib')
            if os.path.isdir(lib_dir):
                for lib in sorted(os.listdir(lib_dir)):
                    if lib.endswith('.so') or '.so.' in lib:
                        try:
                            import ctypes
                            ctypes.CDLL(os.path.join(lib_dir, lib))
                        except OSError:
                            pass
        logger.info("Pre-loaded nvidia CUDA libraries for GPU support")
    except ImportError:
        logger.info("No nvidia pip packages — running on CPU")

    import tensorflow as tf
    import tensorflow_hub as hub

    logger.info("Loading Enformer on GPU %d...", args.gpu)
    t_load = time.time()

    os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"
    enformer = hub.load("https://tfhub.dev/deepmind/enformer/1")
    enformer_model = enformer.model

    from chorus.oracles.enformer_source.enformer_metadata import EnformerMetadata
    metadata = EnformerMetadata()

    import pysam
    ref_path = os.path.join(REPO_ROOT, "genomes/hg38.fa")
    ref = pysam.FastaFile(ref_path)

    logger.info("Model loaded in %.1f seconds", time.time() - t_load)

    # Build track info: list of dicts ordered by track index
    track_info = []
    for _, row in metadata.tracks_df.iterrows():
        track_idx = row['index']
        identifier = row['identifier']
        desc = row['description']
        assay_type = desc.split(':')[0] if ':' in desc else 'unknown'

        # Determine layer spec
        if assay_type == 'CHIP':
            upper_desc = desc.upper()
            is_histone = any(p in upper_desc for p in HISTONE_PATTERNS)
            spec_key = 'CHIP_HIST' if is_histone else 'CHIP_TF'
        elif assay_type in LAYER_SPEC:
            spec_key = assay_type
        else:
            spec_key = None

        if spec_key is not None:
            window, formula, pseudocount, signed = LAYER_SPEC[spec_key]
        else:
            window, formula, pseudocount, signed = (501, 'log2fc', 1.0, False)

        track_info.append({
            'idx': track_idx,
            'identifier': identifier,
            'assay_type': assay_type,
            'spec_key': spec_key,
            'window': window,
            'formula': formula,
            'pseudocount': pseudocount,
            'signed': signed,
        })

    # Sort by index to ensure consistent ordering
    track_info.sort(key=lambda t: t['idx'])
    assert len(track_info) == len(metadata.tracks_df)

    def one_hot_encode(seq):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        one_hot = np.zeros((len(seq), 4), dtype=np.float32)
        for i, base in enumerate(seq):
            if base in mapping:
                one_hot[i, mapping[base]] = 1.0
        return one_hot

    def predict_all(seq):
        one_hot = one_hot_encode(seq)
        one_hot_batch = tf.constant(one_hot[np.newaxis], dtype=tf.float32)
        predictions = enformer_model.predict_on_batch(one_hot_batch)
        return predictions['human'][0].numpy()

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

    return predict_all, get_sequence, ref, track_info


def compute_effect(ref_val, alt_val, formula, pseudocount):
    if formula == 'log2fc':
        return math.log2((alt_val + pseudocount) / (ref_val + pseudocount))
    elif formula == 'logfc':
        return math.log((alt_val + pseudocount) / (ref_val + pseudocount))
    else:
        return alt_val - ref_val


def get_window_slice(track):
    """Return (start_bin, end_bin) for the central window of a track."""
    center_bin = OUTPUT_BINS // 2
    hw = track['window'] // (2 * BIN_SIZE)
    ws = max(0, center_bin - hw)
    we = min(OUTPUT_BINS, center_bin + hw + 1)
    return ws, we


# ══════════════════════════════════════════════════════════════════
# PART 1: VARIANT EFFECT BACKGROUNDS (per-track)
# ══════════════════════════════════════════════════════════════════

def build_variant_backgrounds():
    predict_all, get_sequence, ref, track_info = load_model_and_metadata()
    n_tracks = len(track_info)

    logger.info("=" * 60)
    logger.info("PER-TRACK VARIANT BACKGROUNDS: %d SNPs x %d tracks", args.n_variants, n_tracks)
    logger.info("=" * 60)

    effect_reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size)

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

            for t in track_info:
                idx = t['idx']
                ws, we = get_window_slice(t)
                ref_val = float(np.sum(ref_pred[ws:we, idx]))
                alt_val = float(np.sum(alt_pred[ws:we, idx]))
                score = compute_effect(ref_val, alt_val, t['formula'], t['pseudocount'])
                # Store absolute value for unsigned layers, raw for signed
                if not t['signed']:
                    score = abs(score)
                effect_reservoir.add(idx, score)
        except Exception as exc:
            logger.warning("Failed variant %d: %s", i, str(exc)[:150])

    elapsed_v = time.time() - t0
    logger.info(
        "Variant scoring complete in %.1f hours: %s total samples across %d tracks",
        elapsed_v / 3600,
        f"{effect_reservoir.total_samples():,}",
        effect_reservoir.tracks_with_data(),
    )

    # Save intermediate reservoir
    effect_matrix = effect_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
    track_ids = [t['identifier'] for t in track_info]
    signed_flags = np.array([t['signed'] for t in track_info], dtype=bool)

    interim_path = os.path.join(cache_dir, "enformer_effect_cdfs_interim.npz")
    np.savez_compressed(interim_path,
                        track_ids=np.array(track_ids, dtype='U'),
                        effect_cdfs=effect_matrix.astype(np.float32),
                        effect_counts=effect_reservoir.get_counts(),
                        signed_flags=signed_flags)
    logger.info("Saved interim effect CDFs: %s (%.1f MB)",
                interim_path, os.path.getsize(interim_path) / (1024 * 1024))

    ref.close()
    return interim_path


# ══════════════════════════════════════════════════════════════════
# PART 2: BASELINE SIGNAL BACKGROUNDS (per-track)
# ══════════════════════════════════════════════════════════════════

def build_baseline_backgrounds():
    predict_all, get_sequence, ref, track_info = load_model_and_metadata()
    n_tracks = len(track_info)

    logger.info("=" * 60)
    logger.info("PER-TRACK BASELINE BACKGROUNDS: random + cCREs + TSSs + gene bodies")
    logger.info("  %d tracks, %d bins/position for perbin", n_tracks, PERBIN_BINS_PER_POSITION)
    logger.info("=" * 60)

    summary_reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size)
    perbin_reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size)

    # RNG for random bin sampling (perbin CDF)
    rng_bins = np.random.RandomState(999)

    # Build set of CAGE track indices for summary routing
    cage_track_indices = set()
    for t in track_info:
        if t['assay_type'] == 'CAGE':
            cage_track_indices.add(t['idx'])
    logger.info("CAGE tracks: %d, non-CAGE tracks: %d",
                len(cage_track_indices), n_tracks - len(cage_track_indices))

    # ── Position set 1: Random genomic positions (backbone of the CDF) ──
    # Most of the genome is silent for most tracks.  This must dominate
    # the CDF so that actual peaks stand out as high percentiles.
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

    # ── Position set 2: SCREEN cCRE positions (per-category) ──
    from chorus.utils.annotations import sample_ccre_positions
    ccre_positions = sample_ccre_positions(
        n_per_category={
            "PLS": 3000,       # promoters — CAGE, DNASE, H3K4me3, H3K27ac, TF
            "dELS": 2500,      # distal enhancers — DNASE, H3K27ac, some TF
            "pELS": 1500,      # proximal enhancers — similar
            "CA-CTCF": 1500,   # CTCF sites — CTCF ChIP, some DNASE
            "CA-TF": 1000,     # TF binding sites
            "TF": 500,         # TF-only sites
            "CA-H3K4me3": 1000,# H3K4me3-marked sites
            "CA": 500,         # chromatin-accessible only
        },
        seed=456,
    )
    logger.info("cCRE positions: %d (per-category sampling)", len(ccre_positions))

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

    # ── Position set 4: Gene body midpoints (for H3K36me3, RNA, H3K27me3) ──
    long_genes = pc_genes[(pc_genes['end'] - pc_genes['start']) > 10000].copy()
    long_genes['midpoint'] = (long_genes['start'] + long_genes['end']) // 2
    gb_dedup = long_genes.groupby('gene_name').first().reset_index()
    rng_gb = random.Random(222)
    gb_list = list(zip(gb_dedup['chrom'], gb_dedup['midpoint']))
    if len(gb_list) > 2000:
        gb_list = rng_gb.sample(gb_list, 2000)
    logger.info("Gene body midpoints: %d", len(gb_list))

    # ── Combine all positions ──
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

            # Sample random bins for perbin CDF (same bins for all tracks)
            bin_sample = rng_bins.choice(OUTPUT_BINS, PERBIN_BINS_PER_POSITION, replace=False)

            for t in track_info:
                idx = t['idx']
                is_cage = idx in cage_track_indices
                ws, we = get_window_slice(t)

                # ── Summary CDF: window-sum, with CAGE routing ──
                # CAGE summary only from TSS + random (not cCREs where
                # CAGE signal is biologically irrelevant)
                if not (is_cage and pos_type == 'ccre'):
                    signal = float(np.sum(pred[ws:we, idx]))
                    summary_reservoir.add(idx, signal)

                # ── Perbin CDF: random bins from full output window ──
                # NO track-type routing — every track sees every position.
                # The perbin CDF must represent the full genome-wide
                # per-bin distribution (mostly low/zero with occasional
                # peaks) so the IGV browser shows peaks clearly.
                perbin_reservoir.add_batch(idx, pred[bin_sample, idx].astype(np.float64))
        except Exception as exc:
            logger.warning("Failed %s:%d: %s", chrom, pos, str(exc)[:150])

    elapsed_b = time.time() - t0
    logger.info(
        "Baseline complete in %.1f hours: %s summary + %s perbin samples",
        elapsed_b / 3600,
        f"{summary_reservoir.total_samples():,}",
        f"{perbin_reservoir.total_samples():,}",
    )

    # Save intermediate reservoirs
    track_ids = [t['identifier'] for t in track_info]

    summary_matrix = summary_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
    perbin_matrix = perbin_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)

    interim_path = os.path.join(cache_dir, "enformer_baseline_cdfs_interim.npz")
    np.savez_compressed(interim_path,
                        track_ids=np.array(track_ids, dtype='U'),
                        summary_cdfs=summary_matrix.astype(np.float32),
                        summary_counts=summary_reservoir.get_counts(),
                        perbin_cdfs=perbin_matrix.astype(np.float32),
                        perbin_counts=perbin_reservoir.get_counts())
    logger.info("Saved interim baseline CDFs: %s (%.1f MB)",
                interim_path, os.path.getsize(interim_path) / (1024 * 1024))

    ref.close()
    return interim_path


# ══════════════════════════════════════════════════════════════════
# PART 3: MERGE into final enformer_pertrack.npz
# ══════════════════════════════════════════════════════════════════

def merge_to_final():
    """Merge interim effect + baseline NPZ files into the final pertrack.npz."""
    from chorus.analysis.normalization import PerTrackNormalizer

    effect_path = os.path.join(cache_dir, "enformer_effect_cdfs_interim.npz")
    baseline_path = os.path.join(cache_dir, "enformer_baseline_cdfs_interim.npz")

    if not os.path.exists(effect_path):
        logger.error("Missing %s — run --part variants first", effect_path)
        return
    if not os.path.exists(baseline_path):
        logger.error("Missing %s — run --part baselines first", baseline_path)
        return

    logger.info("=" * 60)
    logger.info("MERGING interim files into enformer_pertrack.npz")
    logger.info("=" * 60)

    effect_data = np.load(effect_path, allow_pickle=False)
    baseline_data = np.load(baseline_path, allow_pickle=False)

    # Verify track IDs match
    effect_ids = list(effect_data["track_ids"].astype(str))
    baseline_ids = list(baseline_data["track_ids"].astype(str))
    assert effect_ids == baseline_ids, "Track ID mismatch between effect and baseline files!"

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
        logger.info("  Effect sample counts: min=%d, median=%d, max=%d",
                     effect_counts.min(), np.median(effect_counts), effect_counts.max())

    path = PerTrackNormalizer.build_and_save(
        oracle_name="enformer",
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

    # Verify the file loads correctly
    norm = PerTrackNormalizer(cache_dir=cache_dir)
    entry = norm._ensure_loaded("enformer")
    logger.info("Verification: loaded %d tracks, CDFs: %s",
                len(entry["track_ids"]),
                [k for k in ("effect_cdfs", "summary_cdfs", "perbin_cdfs") if entry[k] is not None])

    # Log per-assay-type stats
    from chorus.oracles.enformer_source.enformer_metadata import EnformerMetadata
    metadata = EnformerMetadata()
    assay_counts = defaultdict(int)
    for _, row in metadata.tracks_df.iterrows():
        desc = row['description']
        assay_type = desc.split(':')[0] if ':' in desc else 'unknown'
        assay_counts[assay_type] += 1

    logger.info("Track breakdown by assay type:")
    for atype, count in sorted(assay_counts.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d tracks", atype, count)

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
