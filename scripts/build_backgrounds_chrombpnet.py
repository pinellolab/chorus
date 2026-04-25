"""Build per-track background distributions for ChromBPNet.

Each ChromBPNet model is treated as a single track.  Produces
``chrombpnet_pertrack.npz`` with three CDFs (effect, summary, perbin)
per model — typically 24 models = 24 "tracks" (12 ATAC + 12 DNASE).

ChromBPNet output is short (1000 bp), so the perbin CDF captures the
bin-level distribution within the prediction window.

Run in chorus-chrombpnet env:
  mamba run -n chorus-chrombpnet python scripts/build_backgrounds_chrombpnet.py --part variants --gpu 0
  mamba run -n chorus-chrombpnet python scripts/build_backgrounds_chrombpnet.py --part baselines --gpu 0
  mamba run -n chorus python scripts/build_backgrounds_chrombpnet.py --part merge
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
parser.add_argument("--part", choices=["variants", "baselines", "merge", "merge-incremental", "both", "all"], default="all")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--n-variants", type=int, default=10000)
parser.add_argument("--reservoir-size", type=int, default=50000)
parser.add_argument("--n-cdf-points", type=int, default=10000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument(
    "--only-missing",
    action="store_true",
    help="Skip models whose track_id is already present in the existing "
    "chrombpnet_pertrack.npz. Pair with --part merge-incremental to "
    "stitch new rows into the existing NPZ.",
)
args = parser.parse_args()

log_dir = os.path.join(REPO_ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/bg_chrombpnet_{args.part}.log", mode='w'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
INPUT_LENGTH = 2114
OUTPUT_LENGTH = 1000
WINDOW_BP = 501  # central scoring window
PERBIN_BINS_PER_POSITION = 32
FORMULA = 'log2fc'
PSEUDOCOUNT = 1.0

cache_dir = os.path.expanduser("~/.chorus/backgrounds")
os.makedirs(cache_dir, exist_ok=True)


# ── Reservoir sampler (same as Borzoi/Enformer) ──────────────────
class ReservoirSampler:
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


def load_models_and_setup():
    """Load reference, set up GPU, return (oracle, models_to_score, ref)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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
    except ImportError:
        pass

    import tensorflow as tf
    import pysam
    from chorus.oracles.chrombpnet_source.chrombpnet_globals import iter_unique_models
    from chorus.oracles.chrombpnet import ChromBPNetOracle

    ref_path = os.path.join(REPO_ROOT, "genomes/hg38.fa")
    ref = pysam.FastaFile(ref_path)

    # Dedupe by ENCFF: registry has aliases ("limb" + "limb_E12.5"
    # point to the same model). Without dedup we'd compute identical
    # CDFs twice.
    models_to_score = [(assay, ct) for assay, ct, _encff in iter_unique_models()]

    # Optional incremental mode: skip models already present in the NPZ.
    if args.only_missing:
        existing_npz = os.path.join(cache_dir, "chrombpnet_pertrack.npz")
        if os.path.exists(existing_npz):
            existing = set(str(t) for t in np.load(existing_npz, allow_pickle=False)["track_ids"])
            before = len(models_to_score)
            models_to_score = [(a, c) for a, c in models_to_score if f"{a}:{c}" not in existing]
            logger.info(
                "--only-missing: existing NPZ has %d tracks; %d/%d to build (%d skipped).",
                len(existing), len(models_to_score), before, before - len(models_to_score),
            )
        else:
            logger.info("--only-missing: no existing NPZ — building all %d.", len(models_to_score))
    logger.info("Will score %d models (fold %d)", len(models_to_score), args.fold)

    oracle = ChromBPNetOracle(use_environment=False, reference_fasta=ref_path)
    return oracle, models_to_score, ref, tf


# ── Helpers ─────────────────────────────────────────────────────────
def one_hot_encode(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            one_hot[i, mapping[base]] = 1.0
    return one_hot


def get_sequence(ref, chrom, pos):
    half = INPUT_LENGTH // 2
    start, end = pos - half, pos + half
    chrom_len = ref.get_reference_length(chrom)
    if start < 0 or end > chrom_len:
        return None
    seq = ref.fetch(chrom, start, end).upper()
    if len(seq) != INPUT_LENGTH or seq.count('N') > INPUT_LENGTH * 0.5:
        return None
    return seq


def predict_profiles_batch(model, seqs):
    """Run ChromBPNet on a batch of sequences. Returns (B, OUTPUT_LENGTH) array."""
    ohe_batch = np.stack([one_hot_encode(s) for s in seqs])
    predictions = model(ohe_batch, training=False)
    probabilities = predictions[0].numpy()
    counts = predictions[1].numpy()
    norm_prob = probabilities - np.mean(probabilities, axis=1, keepdims=True)
    softmax_probs = np.exp(norm_prob) / np.sum(np.exp(norm_prob), axis=1, keepdims=True)
    profiles = softmax_probs * np.exp(counts[:, 0:1])
    return profiles


def score_window_sum(profile):
    center = OUTPUT_LENGTH // 2
    hw = WINDOW_BP // 2
    ws = max(0, center - hw)
    we = min(OUTPUT_LENGTH, center + hw + 1)
    return float(np.sum(profile[ws:we]))


def compute_effect(ref_val, alt_val):
    return math.log2((alt_val + PSEUDOCOUNT) / (ref_val + PSEUDOCOUNT))


# ══════════════════════════════════════════════════════════════════
# Variant + baseline collection — runs all models for one part
# ══════════════════════════════════════════════════════════════════

def build_all_models(do_variants: bool, do_baselines: bool):
    oracle, models_to_score, ref, tf = load_models_and_setup()
    n_tracks = len(models_to_score)

    track_ids = [f"{a}:{c}" for a, c in models_to_score]
    logger.info("Track IDs: %s", track_ids)

    effect_reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size) if do_variants else None
    summary_reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size) if do_baselines else None
    perbin_reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size) if do_baselines else None

    rng_bins = np.random.RandomState(999)

    # ── SNPs (for variants) ──
    snps = []
    if do_variants:
        random.seed(42)
        chroms = [f"chr{i}" for i in range(1, 23)]
        snps_per_chrom = args.n_variants // len(chroms) + 1
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
        logger.info("Generated %d SNPs", len(snps))

    # ── Baseline positions ──
    baseline_positions = []
    if do_baselines:
        from chorus.utils.annotations import sample_ccre_positions, get_annotation_manager
        ccre_positions = sample_ccre_positions(
            n_per_category={
                "PLS": 3000, "dELS": 2500, "pELS": 1500,
                "CA-CTCF": 1500, "CA-TF": 1000, "TF": 500,
                "CA-H3K4me3": 1000, "CA": 500,
            },
            seed=456,
        )

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

        baseline_positions = []
        for chrom, pos in rand_positions:
            baseline_positions.append((chrom, pos))
        for chrom, pos in ccre_positions:
            baseline_positions.append((chrom, pos))
        for chrom, pos in tss_list:
            baseline_positions.append((chrom, int(pos)))
        random.shuffle(baseline_positions)
        logger.info("Total baseline positions: %d (random=%d, cCRE=%d, TSS=%d)",
                    len(baseline_positions), len(rand_positions), len(ccre_positions), len(tss_list))

    # Iterate over models
    for model_idx, (assay, cell_type) in enumerate(models_to_score):
        logger.info("=" * 60)
        logger.info("Model %d/%d: %s:%s (fold %d)",
                    model_idx + 1, n_tracks, assay, cell_type, args.fold)
        logger.info("=" * 60)

        try:
            oracle.load_pretrained_model(assay=assay, cell_type=cell_type, fold=args.fold)
        except Exception as exc:
            logger.warning("Failed to load %s:%s: %s", assay, cell_type, str(exc)[:200])
            continue

        model = oracle.model

        # ── Variant scoring ──
        if do_variants and snps:
            t0 = time.time()
            ref_seqs, alt_seqs = [], []
            for snp in snps:
                seq_ref = get_sequence(ref, snp["chrom"], snp["pos"])
                if seq_ref is None:
                    continue
                offset = INPUT_LENGTH // 2 - 1
                seq_alt = seq_ref[:offset] + snp["alt"] + seq_ref[offset + 1:]
                ref_seqs.append(seq_ref)
                alt_seqs.append(seq_alt)

            for i in range(0, len(ref_seqs), args.batch_size):
                ref_batch = ref_seqs[i:i + args.batch_size]
                alt_batch = alt_seqs[i:i + args.batch_size]
                try:
                    ref_profiles = predict_profiles_batch(model, ref_batch)
                    alt_profiles = predict_profiles_batch(model, alt_batch)
                    for rp, ap in zip(ref_profiles, alt_profiles):
                        ref_val = score_window_sum(rp)
                        alt_val = score_window_sum(ap)
                        score = abs(compute_effect(ref_val, alt_val))
                        effect_reservoir.add(model_idx, score)
                except Exception as exc:
                    logger.warning("Variant batch failed: %s", str(exc)[:100])

            logger.info("  Variants done in %.1f min, %s effect samples for this model",
                        (time.time() - t0) / 60,
                        f"{int(effect_reservoir.counts[model_idx]):,}")

        # ── Baseline scoring ──
        if do_baselines and baseline_positions:
            t0 = time.time()
            base_seqs = []
            for chrom, pos in baseline_positions:
                seq = get_sequence(ref, chrom, pos)
                if seq is not None:
                    base_seqs.append(seq)

            for i in range(0, len(base_seqs), args.batch_size):
                batch = base_seqs[i:i + args.batch_size]
                try:
                    profiles = predict_profiles_batch(model, batch)
                    for prof in profiles:
                        # Summary: window sum
                        signal = score_window_sum(prof)
                        summary_reservoir.add(model_idx, signal)
                        # Perbin: random bins from full output
                        bin_sample = rng_bins.choice(OUTPUT_LENGTH, PERBIN_BINS_PER_POSITION, replace=False)
                        perbin_reservoir.add_batch(model_idx, prof[bin_sample].astype(np.float64))
                except Exception as exc:
                    logger.warning("Baseline batch failed: %s", str(exc)[:100])

            logger.info("  Baselines done in %.1f min, %s summary + %s perbin samples for this model",
                        (time.time() - t0) / 60,
                        f"{int(summary_reservoir.counts[model_idx]):,}",
                        f"{int(perbin_reservoir.counts[model_idx]):,}")

    ref.close()

    # Save interim files
    signed_flags = np.zeros(n_tracks, dtype=bool)  # all unsigned

    if do_variants:
        effect_matrix = effect_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
        interim_path = os.path.join(cache_dir, "chrombpnet_effect_cdfs_interim.npz")
        np.savez_compressed(
            interim_path,
            track_ids=np.array(track_ids, dtype='U'),
            effect_cdfs=effect_matrix.astype(np.float32),
            effect_counts=effect_reservoir.get_counts(),
            signed_flags=signed_flags,
        )
        logger.info("Saved effect interim: %s", interim_path)

    if do_baselines:
        summary_matrix = summary_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
        perbin_matrix = perbin_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
        interim_path = os.path.join(cache_dir, "chrombpnet_baseline_cdfs_interim.npz")
        np.savez_compressed(
            interim_path,
            track_ids=np.array(track_ids, dtype='U'),
            summary_cdfs=summary_matrix.astype(np.float32),
            summary_counts=summary_reservoir.get_counts(),
            perbin_cdfs=perbin_matrix.astype(np.float32),
            perbin_counts=perbin_reservoir.get_counts(),
        )
        logger.info("Saved baseline interim: %s", interim_path)


def merge_to_final():
    from chorus.analysis.normalization import PerTrackNormalizer

    effect_path = os.path.join(cache_dir, "chrombpnet_effect_cdfs_interim.npz")
    baseline_path = os.path.join(cache_dir, "chrombpnet_baseline_cdfs_interim.npz")

    if not os.path.exists(effect_path) or not os.path.exists(baseline_path):
        logger.error("Missing interim files")
        return

    effect_data = np.load(effect_path, allow_pickle=False)
    baseline_data = np.load(baseline_path, allow_pickle=False)

    effect_ids = list(effect_data["track_ids"].astype(str))
    baseline_ids = list(baseline_data["track_ids"].astype(str))
    assert effect_ids == baseline_ids

    path = PerTrackNormalizer.build_and_save(
        oracle_name="chrombpnet",
        track_ids=effect_ids,
        effect_cdfs=effect_data["effect_cdfs"],
        summary_cdfs=baseline_data["summary_cdfs"],
        perbin_cdfs=baseline_data["perbin_cdfs"],
        signed_flags=effect_data["signed_flags"],
        effect_counts=effect_data["effect_counts"] if "effect_counts" in effect_data else None,
        summary_counts=baseline_data["summary_counts"] if "summary_counts" in baseline_data else None,
        perbin_counts=baseline_data["perbin_counts"] if "perbin_counts" in baseline_data else None,
        cache_dir=cache_dir,
    )
    logger.info("DONE — final file: %s (%.1f MB)", path, path.stat().st_size / 1e6)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def merge_to_final_incremental():
    """Stitch newly-built CDF rows onto the existing chrombpnet_pertrack.npz.

    Loads the current NPZ + the two interim files written by an
    ``--only-missing --part both`` run, concatenates rows preserving
    track-id order (existing first, then new), and writes the merged
    NPZ in place.
    """
    from chorus.analysis.normalization import PerTrackNormalizer

    effect_path = os.path.join(cache_dir, "chrombpnet_effect_cdfs_interim.npz")
    baseline_path = os.path.join(cache_dir, "chrombpnet_baseline_cdfs_interim.npz")
    existing_path = os.path.join(cache_dir, "chrombpnet_pertrack.npz")

    if not os.path.exists(effect_path) or not os.path.exists(baseline_path):
        logger.error("Missing interim files — run with --only-missing --part both first")
        return
    if not os.path.exists(existing_path):
        logger.error("No existing NPZ found at %s — use --part merge instead", existing_path)
        return

    existing = np.load(existing_path, allow_pickle=False)
    effect_data = np.load(effect_path, allow_pickle=False)
    baseline_data = np.load(baseline_path, allow_pickle=False)

    new_ids = list(effect_data["track_ids"].astype(str))
    assert new_ids == list(baseline_data["track_ids"].astype(str)), \
        "interim effect/baseline track_id ordering must agree"

    existing_count = len(existing["track_ids"])  # capture now — `existing`'s file is overwritten by build_and_save
    merged_ids = list(existing["track_ids"].astype(str)) + new_ids

    def stack(name: str):
        return np.concatenate([existing[name], effect_data[name] if "effect" in name else baseline_data[name]])

    merged_effect = np.concatenate([existing["effect_cdfs"], effect_data["effect_cdfs"]])
    merged_summary = np.concatenate([existing["summary_cdfs"], baseline_data["summary_cdfs"]])
    merged_perbin = np.concatenate([existing["perbin_cdfs"], baseline_data["perbin_cdfs"]])
    merged_signed = np.concatenate([existing["signed_flags"], effect_data["signed_flags"]])

    def maybe_concat(name: str, src):
        if name in existing and name in src:
            return np.concatenate([existing[name], src[name]])
        return None

    merged_effect_counts  = maybe_concat("effect_counts",  effect_data)
    merged_summary_counts = maybe_concat("summary_counts", baseline_data)
    merged_perbin_counts  = maybe_concat("perbin_counts",  baseline_data)

    path = PerTrackNormalizer.build_and_save(
        oracle_name="chrombpnet",
        track_ids=merged_ids,
        effect_cdfs=merged_effect,
        summary_cdfs=merged_summary,
        perbin_cdfs=merged_perbin,
        signed_flags=merged_signed,
        effect_counts=merged_effect_counts,
        summary_counts=merged_summary_counts,
        perbin_counts=merged_perbin_counts,
        cache_dir=cache_dir,
    )
    logger.info(
        "DONE — merged NPZ has %d tracks (%d existing + %d new): %s (%.1f MB)",
        len(merged_ids), existing_count, len(new_ids),
        path, path.stat().st_size / 1e6,
    )


if args.part == "variants":
    build_all_models(do_variants=True, do_baselines=False)
elif args.part == "baselines":
    build_all_models(do_variants=False, do_baselines=True)
elif args.part == "merge":
    merge_to_final()
elif args.part == "merge-incremental":
    merge_to_final_incremental()
elif args.part in ("both", "all"):
    build_all_models(do_variants=True, do_baselines=True)
    if args.only_missing:
        merge_to_final_incremental()
    else:
        merge_to_final()
