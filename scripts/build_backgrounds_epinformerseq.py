"""Build per-track background distributions for EPInformer-seq.

EPInformer-seq predicts a single combined-activity scalar (geometric mean of
DNase × H3K27ac, in log2(0.1+·) space) for 10 Roadmap cell lines:
    K562, GM12878, HepG2, A549, HeLa, HMEC, HSMM, HUVEC, NHEK, NHLF

Each cell type is one track (assay = ``Enhancer_H3K27ac_DNase``). Produces
``epinformerseq_pertrack.npz`` with effect_cdfs (variant ALT − REF) and
summary_cdfs (baseline at random + cCRE + TSS positions). No perbin CDFs
since the model output is a single scalar per 256-bp window.

Layer = ``promoter_activity``: signed effect (alt − ref).

Run in chorus-epinformerseq env:
  mamba run -n chorus-epinformerseq python scripts/build_backgrounds_epinformerseq.py --part variants
  mamba run -n chorus-epinformerseq python scripts/build_backgrounds_epinformerseq.py --part baselines
  mamba run -n chorus              python scripts/build_backgrounds_epinformerseq.py --part merge
"""
import argparse
import logging
import os
import random
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
os.environ["CHORUS_NO_TIMEOUT"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--part", choices=["variants", "baselines", "merge", "both", "all"], default="all")
parser.add_argument("--device", type=str, default=None)
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
        logging.FileHandler(f"{log_dir}/bg_epinformerseq_{args.part}.log", mode='w'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

cache_dir = os.path.expanduser("~/.chorus/backgrounds")
os.makedirs(cache_dir, exist_ok=True)


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


def load_setup():
    """Load reference + setup. Returns (cell_types, get_sequence, load_cell_type_model, predict_activity_fn, ref, device)."""
    import torch
    import pysam

    from chorus.oracles.epinformerseq_source.epinformerseq_globals import (
        EPINFORMERSEQ_AVAILABLE_CELLTYPES, EPINFORMERSEQ_WINDOW,
    )
    from chorus.oracles.epinformerseq_source.model_usage import load_model, predict_activity
    from chorus.core.globals import CHORUS_DOWNLOADS_DIR

    EPINFORMERSEQ_MODELS_DIR = CHORUS_DOWNLOADS_DIR / "epinformerseq"

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    ref = pysam.FastaFile(os.path.join(REPO_ROOT, "genomes/hg38.fa"))
    cell_types = list(EPINFORMERSEQ_AVAILABLE_CELLTYPES)

    def get_sequence(chrom, pos):
        half = EPINFORMERSEQ_WINDOW // 2
        start, end = pos - half, pos + half
        chrom_len = ref.get_reference_length(chrom)
        if start < 0 or end > chrom_len:
            return None
        seq = ref.fetch(chrom, start, end).upper()
        if len(seq) != EPINFORMERSEQ_WINDOW or seq.count('N') > EPINFORMERSEQ_WINDOW * 0.3:
            return None
        return seq

    def load_cell_type_model(cell_type: str):
        weights_path = EPINFORMERSEQ_MODELS_DIR / cell_type / 'weights.pt'
        if not weights_path.exists():
            logger.warning("Weights missing: %s", weights_path)
            return None
        return load_model(str(weights_path), device=device)

    def predict_one(model, seq: str) -> float:
        # average_reverse=False to match the LegNet build script default;
        # the model was trained with rc-average evaluation, but background
        # CDFs use raw forward-strand for consistency with how ALT/REF
        # comparisons are computed in chorus.
        preds, _ = predict_activity(model, seq=seq, average_reverse=False, device=device)
        return float(preds[0])

    return cell_types, get_sequence, load_cell_type_model, predict_one, ref, device


# ══════════════════════════════════════════════════════════════════
# VARIANT BUILD
# ══════════════════════════════════════════════════════════════════

def build_variant_backgrounds():
    cell_types, get_sequence, load_cell_type_model, predict_activity, ref, device = load_setup()
    n_tracks = len(cell_types)

    logger.info("=" * 60)
    logger.info("PER-TRACK VARIANT BACKGROUNDS: %d SNPs x %d cell types",
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

    import torch

    for ct_i, cell_type in enumerate(cell_types):
        logger.info("Loading EPInformer-seq for %s...", cell_type)
        model = load_cell_type_model(cell_type)
        if model is None:
            logger.warning("Skipping %s — weights not found", cell_type)
            continue

        t0 = time.time()
        for i, snp in enumerate(snps):
            if (i + 1) % 1000 == 0:
                logger.info("  %s variant %d/%d", cell_type, i + 1, len(snps))
            seq_ref = get_sequence(snp["chrom"], snp["pos"])
            if seq_ref is None:
                continue
            offset = (len(seq_ref) // 2) - 1
            seq_alt = seq_ref[:offset] + snp["alt"] + seq_ref[offset + 1:]
            try:
                ref_val = predict_activity(model, seq_ref)
                alt_val = predict_activity(model, seq_alt)
                effect_reservoir.add(ct_i, alt_val - ref_val)  # signed diff
            except Exception:
                pass
        logger.info("  %s variants done in %.1f min, %s samples for this cell type",
                    cell_type, (time.time() - t0) / 60,
                    f"{int(effect_reservoir.counts[ct_i]):,}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    effect_matrix = effect_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
    signed_flags = np.ones(n_tracks, dtype=bool)  # signed (alt − ref)

    interim_path = os.path.join(cache_dir, "epinformerseq_effect_cdfs_interim.npz")
    np.savez_compressed(
        interim_path,
        track_ids=np.array([f"Enhancer_H3K27ac_DNase:{c}" for c in cell_types], dtype='U'),
        effect_cdfs=effect_matrix.astype(np.float32),
        effect_counts=effect_reservoir.get_counts(),
        signed_flags=signed_flags,
    )
    logger.info("Saved effect interim: %s", interim_path)
    ref.close()


# ══════════════════════════════════════════════════════════════════
# BASELINE BUILD
# ══════════════════════════════════════════════════════════════════

def build_baseline_backgrounds():
    cell_types, get_sequence, load_cell_type_model, predict_activity, ref, device = load_setup()
    n_tracks = len(cell_types)

    logger.info("=" * 60)
    logger.info("PER-TRACK BASELINE BACKGROUNDS: %d cell types", n_tracks)
    logger.info("=" * 60)

    summary_reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size)

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

    from chorus.utils.annotations import sample_ccre_positions, get_annotation_manager
    ccre_positions = sample_ccre_positions(
        n_per_category={
            "PLS": 3000, "dELS": 2500, "pELS": 1500,
            "CA-CTCF": 1500, "CA-TF": 1000, "TF": 500,
            "CA-H3K4me3": 1000, "CA": 500,
        },
        seed=456,
    )

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

    all_positions = []
    for chrom, pos in rand_positions:
        all_positions.append((chrom, pos))
    for chrom, pos in ccre_positions:
        all_positions.append((chrom, pos))
    for chrom, pos in tss_list:
        all_positions.append((chrom, int(pos)))
    random.shuffle(all_positions)
    logger.info("Total positions: %d", len(all_positions))

    import torch

    for ct_i, cell_type in enumerate(cell_types):
        logger.info("Loading EPInformer-seq for %s...", cell_type)
        model = load_cell_type_model(cell_type)
        if model is None:
            continue

        t0 = time.time()
        for i, (chrom, pos) in enumerate(all_positions):
            if (i + 1) % 2000 == 0:
                logger.info("  %s baseline %d/%d", cell_type, i + 1, len(all_positions))
            seq = get_sequence(chrom, pos)
            if seq is None:
                continue
            try:
                summary_reservoir.add(ct_i, predict_activity(model, seq))
            except Exception:
                pass
        logger.info("  %s baselines done in %.1f min, %s samples for this cell type",
                    cell_type, (time.time() - t0) / 60,
                    f"{int(summary_reservoir.counts[ct_i]):,}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_matrix = summary_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
    interim_path = os.path.join(cache_dir, "epinformerseq_baseline_cdfs_interim.npz")
    np.savez_compressed(
        interim_path,
        track_ids=np.array([f"Enhancer_H3K27ac_DNase:{c}" for c in cell_types], dtype='U'),
        summary_cdfs=summary_matrix.astype(np.float32),
        summary_counts=summary_reservoir.get_counts(),
    )
    logger.info("Saved baseline interim: %s", interim_path)
    ref.close()


def merge_to_final():
    from chorus.analysis.normalization import PerTrackNormalizer

    effect_path = os.path.join(cache_dir, "epinformerseq_effect_cdfs_interim.npz")
    baseline_path = os.path.join(cache_dir, "epinformerseq_baseline_cdfs_interim.npz")
    if not os.path.exists(effect_path) or not os.path.exists(baseline_path):
        logger.error("Missing interim files (run --part variants and --part baselines first)")
        return

    effect_data = np.load(effect_path, allow_pickle=False)
    baseline_data = np.load(baseline_path, allow_pickle=False)

    effect_ids = list(effect_data["track_ids"].astype(str))
    baseline_ids = list(baseline_data["track_ids"].astype(str))
    assert effect_ids == baseline_ids

    path = PerTrackNormalizer.build_and_save(
        oracle_name="epinformerseq",
        track_ids=effect_ids,
        effect_cdfs=effect_data["effect_cdfs"],
        summary_cdfs=baseline_data["summary_cdfs"],
        perbin_cdfs=None,
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
