"""Build per-track background distributions for Sei.

Sei outputs 40 regulatory classes — each treated as a track. Produces
``sei_pertrack.npz`` with effect_cdfs and summary_cdfs per class.
No perbin CDFs since Sei outputs are scalar (one value per window per class).

Input: 4,096 bp window. Fast (~seconds per prediction on GPU).

Run in chorus-sei env:
  mamba run -n chorus-sei python scripts/build_backgrounds_sei.py --part variants --gpu 0
  mamba run -n chorus-sei python scripts/build_backgrounds_sei.py --part baselines --gpu 0
  mamba run -n chorus python scripts/build_backgrounds_sei.py --part merge
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
parser.add_argument("--device", type=str, default=None, help="cpu or cuda:N")
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
        logging.FileHandler(f"{log_dir}/bg_sei_{args.part}.log", mode='w'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

cache_dir = os.path.expanduser("~/.chorus/backgrounds")
os.makedirs(cache_dir, exist_ok=True)

INPUT_LENGTH = 4096


# ── Reservoir sampler ─────────────────────────────────────────────
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


def load_model_and_setup():
    """Load Sei model + reference. Returns (predict_fn, get_seq_fn, ref, class_names)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from chorus.oracles.sei_source.sei import Sei, SeiProjector
    from chorus.oracles.sei_source.sei_globals import SEI_WINDOW, SEI_TARGETS, SEI_CLASSES
    from chorus.oracles.sei_source.annotations import SeiClassesList
    from chorus.core.globals import CHORUS_DOWNLOADS_DIR

    SEI_MODELS_DIR = CHORUS_DOWNLOADS_DIR / "sei"

    device_str = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    logger.info("Loading Sei on %s...", device_str)
    t0 = time.time()

    model = Sei(sequence_length=SEI_WINDOW, n_genomic_features=SEI_TARGETS)
    weights_path = SEI_MODELS_DIR / 'model' / 'sei.pth'
    model_weights = torch.load(weights_path, map_location='cpu', weights_only=True)
    model_weights = {key.replace("module.model.", ""): value for key, value in model_weights.items()}
    model.load_state_dict(model_weights)
    model.eval()
    model.to(device)

    projector = SeiProjector(
        weights=str(SEI_MODELS_DIR / 'model' / 'projvec_targets.npy'),
        n_classes=SEI_CLASSES,
    )
    classes_list = SeiClassesList.load(str(SEI_MODELS_DIR / 'model' / 'seqclass_info.txt'))
    class_names = [str(cl) for cl in classes_list.classes.keys()]
    logger.info("Model loaded in %.1f s, %d classes", time.time() - t0, len(class_names))

    import pysam
    ref = pysam.FastaFile(os.path.join(REPO_ROOT, "genomes/hg38.fa"))

    def one_hot_encode(seq):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        oh = np.zeros((len(seq), 4), dtype=np.float32)
        for i, b in enumerate(seq):
            if b in mapping:
                oh[i, mapping[b]] = 1.0
        return oh

    def predict_classes(seq):
        ohe = one_hot_encode(seq)
        tensor = torch.from_numpy(ohe).permute(1, 0).float().unsqueeze(0).to(device)
        with torch.no_grad():
            target_preds = model(tensor).cpu().numpy()[0]
        class_preds = projector(target_preds[np.newaxis, :])
        return class_preds[0]

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

    return predict_classes, get_sequence, ref, class_names


# ══════════════════════════════════════════════════════════════════
# VARIANT EFFECT BUILD (per-track: each Sei class is a track)
# ══════════════════════════════════════════════════════════════════

def build_variant_backgrounds():
    predict_classes, get_sequence, ref, class_names = load_model_and_setup()
    n_tracks = len(class_names)

    logger.info("=" * 60)
    logger.info("PER-TRACK VARIANT BACKGROUNDS: %d SNPs x %d Sei classes",
                args.n_variants, n_tracks)
    logger.info("=" * 60)

    effect_reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size)

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
    logger.info("Generated %d SNPs", len(snps))

    t0 = time.time()
    for i, snp in enumerate(snps):
        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
            eta = (len(snps) - i - 1) / rate if rate > 0 else 0
            logger.info("Variant %d/%d — %.1f min, ETA %.0f min, %s samples",
                        i + 1, len(snps), elapsed / 60, eta,
                        f"{effect_reservoir.total_samples():,}")

        seq_ref = get_sequence(snp["chrom"], snp["pos"])
        if seq_ref is None:
            continue
        offset = INPUT_LENGTH // 2 - 1
        seq_alt = seq_ref[:offset] + snp["alt"] + seq_ref[offset + 1:]

        try:
            ref_classes = predict_classes(seq_ref)
            alt_classes = predict_classes(seq_alt)
            # Per-class effect (signed)
            for class_idx in range(n_tracks):
                effect = float(alt_classes[class_idx] - ref_classes[class_idx])
                effect_reservoir.add(class_idx, effect)
        except Exception as exc:
            logger.warning("Failed variant %d: %s", i, str(exc)[:150])

    elapsed_v = time.time() - t0
    logger.info("Variants done in %.1f hrs: %s samples", elapsed_v / 3600,
                f"{effect_reservoir.total_samples():,}")

    effect_matrix = effect_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
    signed_flags = np.ones(n_tracks, dtype=bool)  # Sei is signed (diff formula)

    interim_path = os.path.join(cache_dir, "sei_effect_cdfs_interim.npz")
    np.savez_compressed(
        interim_path,
        track_ids=np.array(class_names, dtype='U'),
        effect_cdfs=effect_matrix.astype(np.float32),
        effect_counts=effect_reservoir.get_counts(),
        signed_flags=signed_flags,
    )
    logger.info("Saved effect interim: %s", interim_path)
    ref.close()


# ══════════════════════════════════════════════════════════════════
# BASELINE BUILD (per-track summary CDFs)
# ══════════════════════════════════════════════════════════════════

def build_baseline_backgrounds():
    predict_classes, get_sequence, ref, class_names = load_model_and_setup()
    n_tracks = len(class_names)

    logger.info("=" * 60)
    logger.info("PER-TRACK BASELINE BACKGROUNDS: %d Sei classes", n_tracks)
    logger.info("=" * 60)

    summary_reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size)

    # Position sets (same strategy as Enformer/Borzoi)
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

    from chorus.utils.annotations import sample_ccre_positions, get_annotation_manager
    ccre_positions = sample_ccre_positions(
        n_per_category={
            "PLS": 3000, "dELS": 2500, "pELS": 1500,
            "CA-CTCF": 1500, "CA-TF": 1000, "TF": 500,
            "CA-H3K4me3": 1000, "CA": 500,
        },
        seed=456,
    )
    logger.info("cCRE positions: %d", len(ccre_positions))

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

    all_positions = []
    for chrom, pos in rand_positions:
        all_positions.append((chrom, pos))
    for chrom, pos in ccre_positions:
        all_positions.append((chrom, pos))
    for chrom, pos in tss_list:
        all_positions.append((chrom, int(pos)))
    random.shuffle(all_positions)
    logger.info("Total: %d positions", len(all_positions))

    t0 = time.time()
    for i, (chrom, pos) in enumerate(all_positions):
        if (i + 1) % 500 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
            eta = (len(all_positions) - i - 1) / rate if rate > 0 else 0
            logger.info("Baseline %d/%d (%s:%d) — %.1f min, ETA %.0f min, %s samples",
                        i + 1, len(all_positions), chrom, pos,
                        elapsed / 60, eta, f"{summary_reservoir.total_samples():,}")

        seq = get_sequence(chrom, pos)
        if seq is None:
            continue

        try:
            class_preds = predict_classes(seq)
            for class_idx in range(n_tracks):
                summary_reservoir.add(class_idx, float(class_preds[class_idx]))
        except Exception as exc:
            logger.warning("Failed %s:%d: %s", chrom, pos, str(exc)[:150])

    elapsed_b = time.time() - t0
    logger.info("Baselines done in %.1f hrs: %s samples", elapsed_b / 3600,
                f"{summary_reservoir.total_samples():,}")

    summary_matrix = summary_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
    interim_path = os.path.join(cache_dir, "sei_baseline_cdfs_interim.npz")
    np.savez_compressed(
        interim_path,
        track_ids=np.array(class_names, dtype='U'),
        summary_cdfs=summary_matrix.astype(np.float32),
        summary_counts=summary_reservoir.get_counts(),
    )
    logger.info("Saved baseline interim: %s", interim_path)
    ref.close()


def merge_to_final():
    from chorus.analysis.normalization import PerTrackNormalizer

    effect_path = os.path.join(cache_dir, "sei_effect_cdfs_interim.npz")
    baseline_path = os.path.join(cache_dir, "sei_baseline_cdfs_interim.npz")
    if not os.path.exists(effect_path) or not os.path.exists(baseline_path):
        logger.error("Missing interim files")
        return

    effect_data = np.load(effect_path, allow_pickle=False)
    baseline_data = np.load(baseline_path, allow_pickle=False)

    effect_ids = list(effect_data["track_ids"].astype(str))
    baseline_ids = list(baseline_data["track_ids"].astype(str))
    assert effect_ids == baseline_ids

    path = PerTrackNormalizer.build_and_save(
        oracle_name="sei",
        track_ids=effect_ids,
        effect_cdfs=effect_data["effect_cdfs"],
        summary_cdfs=baseline_data["summary_cdfs"],
        perbin_cdfs=None,  # No perbin for scalar outputs
        signed_flags=effect_data["signed_flags"],
        effect_counts=effect_data["effect_counts"] if "effect_counts" in effect_data else None,
        summary_counts=baseline_data["summary_counts"] if "summary_counts" in baseline_data else None,
        cache_dir=cache_dir,
    )
    logger.info("DONE — final file: %s (%.1f MB)", path, path.stat().st_size / 1e6)


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
