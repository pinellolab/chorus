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
parser.add_argument(
    "--part",
    choices=["variants", "baselines", "merge", "merge-incremental", "merge-shards", "both", "all"],
    default="all",
)
parser.add_argument(
    "--assay",
    choices=["ATAC_DNASE", "CHIP", "all"],
    default="ATAC_DNASE",
    help="Which model family to score. ATAC_DNASE = the 42 ChromBPNet "
    "ENCODE models (~22 min/model on Metal). CHIP = the 1259 BPNet "
    "JASPAR models (~3 min/model on Metal — much smaller arch). "
    "all = both, sequentially.",
)
parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    help="Pin TensorFlow to this GPU index. No-op if CUDA_VISIBLE_DEVICES "
    "is already set in the calling shell — the outer env var wins, so "
    "`CUDA_VISIBLE_DEVICES=1 ... --gpu 0` puts the work on physical GPU 1.",
)
parser.add_argument("--fold", type=int, default=0)
parser.add_argument(
    "--model-type",
    choices=["chrombpnet_nobias", "chrombpnet"],
    default="chrombpnet_nobias",
    help="ChromBPNet variant. Default `chrombpnet_nobias` (bias-corrected) "
    "matches the 0.3+ chorus default — the variant the slim HF mirror "
    "ships and the one user-facing predictions go through. The legacy "
    "`chrombpnet` variant (bias-aware) is available for ablation studies "
    "but produces percentiles that don't match what `oracle.predict()` "
    "returns for default loads in 0.3+. The pre-0.3 CDFs on HF were built "
    "against `chrombpnet`; see audits/2026-04-29_chrombpnet_cdf_rebuild/ "
    "for the rebuild against `chrombpnet_nobias`.",
)
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
parser.add_argument(
    "--force",
    action="store_true",
    help="Overwrite an existing interim NPZ even if its track-id set "
    "differs from the current run. Default: refuse to overwrite and "
    "exit with a diff, to protect the documented two-pass flow "
    "(--assay ATAC_DNASE then --assay CHIP) which would otherwise "
    "silently lose the first pass's tracks.",
)
parser.add_argument(
    "--shard",
    type=int,
    default=None,
    help="0-indexed shard for distributed builds. When set with "
    "--shard-of, this process only handles models with idx %% N == "
    "<shard>. Interim files get a `.shard<N>of<M>` suffix; collect "
    "from all shards on one machine and run --part merge-shards.",
)
parser.add_argument(
    "--shard-of",
    type=int,
    default=None,
    help="Total number of shards. Required when --shard is set.",
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


def _track_id_for(spec: dict) -> str:
    """Track-id string used in the NPZ row index.

    Mirrors `chorus/oracles/chrombpnet.py:555` so the NPZ matches what
    predictions actually emit.
    """
    if spec["assay"] == "CHIP":
        return f"CHIP:{spec['cell_type']}:{spec['TF']}"
    return f"{spec['assay']}:{spec['cell_type']}"


def _enumerate_models(assay_choice: str) -> list[dict]:
    """Build the master list of model specs for a given --assay flag."""
    from chorus.oracles.chrombpnet_source.chrombpnet_globals import (
        iter_unique_models, iter_unique_bpnet_models,
    )
    specs: list[dict] = []
    if assay_choice in ("ATAC_DNASE", "all"):
        for assay, ct, _encff in iter_unique_models():
            specs.append({"assay": assay, "cell_type": ct})
    if assay_choice in ("CHIP", "all"):
        for cell_type, tf, _url, _id in iter_unique_bpnet_models():
            specs.append({"assay": "CHIP", "cell_type": cell_type, "TF": tf})
    return specs


def _interim_suffix() -> str:
    """File suffix when sharded: `.shard<N>of<M>`. Empty when not."""
    if args.shard is None or args.shard_of is None:
        return ""
    return f".shard{args.shard}of{args.shard_of}"


def _check_interim_compatibility(interim_path: str, new_track_ids, force: bool, label: str) -> None:
    """Refuse to overwrite an interim NPZ whose track-id set differs from
    the current run, unless --force was passed. Closes #71/#73.

    The documented two-pass flow (``--assay ATAC_DNASE`` then ``--assay
    CHIP``) would otherwise silently overwrite the first pass's interim
    with the second pass's smaller track set, producing a 744-track
    final NPZ where 786 was expected.
    """
    if not os.path.exists(interim_path):
        return
    try:
        existing = list(np.load(interim_path, allow_pickle=False)["track_ids"].astype(str))
    except Exception as exc:
        if force:
            return
        raise SystemExit(
            f"Existing {label} interim at {interim_path} is unreadable "
            f"({exc}); pass --force to overwrite."
        )
    new_set, old_set = set(new_track_ids), set(existing)
    if new_set == old_set:
        return  # same tracks → plain overwrite is harmless
    if force:
        return
    only_existing = sorted(old_set - new_set)
    only_new = sorted(new_set - old_set)
    raise SystemExit(
        f"Refusing to overwrite {label} interim at {interim_path}.\n"
        f"  Existing tracks: {len(existing)} (e.g. {only_existing[:3]})\n"
        f"  New tracks:      {len(new_track_ids)} (e.g. {only_new[:3]})\n"
        f"  Only in existing: {len(only_existing)}; only in new: {len(only_new)}.\n"
        f"This usually means a previous staged run wrote tracks for a "
        f"different --assay group (the documented two-pass flow). Run "
        f"`--part merge` (or `--part merge-incremental`) to consume the "
        f"existing interim first, then re-run, or pass --force to discard "
        f"the existing interim and write only the current run's tracks."
    )


def load_models_and_setup():
    """Load reference, set up GPU, return (oracle, models_to_score, ref)."""
    # Honour pre-set CUDA_VISIBLE_DEVICES from the calling shell so the
    # documented parallel-launch pattern works as cluster-user mental model
    # expects (`CUDA_VISIBLE_DEVICES=N ... --gpu 0` per terminal pins the
    # outer physical GPU, not the inner --gpu arg). Closes #72/#74.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
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
    from chorus.oracles.chrombpnet import ChromBPNetOracle

    ref_path = os.path.join(REPO_ROOT, "genomes/hg38.fa")
    ref = pysam.FastaFile(ref_path)

    models_to_score = _enumerate_models(args.assay)

    # Optional incremental mode: skip models already present in the NPZ.
    if args.only_missing:
        existing_npz = os.path.join(cache_dir, "chrombpnet_pertrack.npz")
        if os.path.exists(existing_npz):
            existing = set(str(t) for t in np.load(existing_npz, allow_pickle=False)["track_ids"])
            before = len(models_to_score)
            models_to_score = [s for s in models_to_score if _track_id_for(s) not in existing]
            logger.info(
                "--only-missing: existing NPZ has %d tracks; %d/%d to build (%d skipped).",
                len(existing), len(models_to_score), before, before - len(models_to_score),
            )
        else:
            logger.info("--only-missing: no existing NPZ — building all %d.", len(models_to_score))

    # Sharding: each process handles only models where idx % shard_of == shard.
    # Stable across processes because _enumerate_models returns a deterministic order.
    if args.shard is not None or args.shard_of is not None:
        if args.shard is None or args.shard_of is None:
            raise SystemExit("--shard and --shard-of must be set together")
        if not (0 <= args.shard < args.shard_of):
            raise SystemExit(f"--shard ({args.shard}) must be in [0, {args.shard_of})")
        before = len(models_to_score)
        models_to_score = [s for i, s in enumerate(models_to_score)
                           if i % args.shard_of == args.shard]
        logger.info(
            "--shard %d/%d: processing %d/%d models on this worker.",
            args.shard, args.shard_of, len(models_to_score), before,
        )

    logger.info("Will score %d models (assay=%s, fold %d)",
                len(models_to_score), args.assay, args.fold)

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
    """Run ChromBPNet or BPNet on a batch of sequences.

    ChromBPNet takes 1 input (sequence) and returns (profile, counts).
    BPNet (CHIP/JASPAR) takes 3 inputs (sequence + zero profile_bias +
    zero counts_bias) and returns the same shape. We auto-detect by
    checking ``len(model.inputs)``.

    Returns (B, OUTPUT_LENGTH) profile array with predicted counts
    folded in (softmax × exp(counts)).
    """
    ohe_batch = np.stack([one_hot_encode(s) for s in seqs])
    if len(model.inputs) == 1:
        predictions = model(ohe_batch, training=False)
    else:
        # BPNet: pad with zero bias inputs that match expected shapes.
        bias_inputs = []
        for inp in model.inputs[1:]:
            shape = [ohe_batch.shape[0]] + [d if d is not None else 1 for d in inp.shape[1:]]
            bias_inputs.append(np.zeros(shape, dtype=np.float32))
        predictions = model([ohe_batch, *bias_inputs], training=False)
    probabilities = predictions[0].numpy()
    counts = predictions[1].numpy()
    # BPNet CHIP models output (B, L, 2) for two strands — sum before scoring
    if probabilities.ndim == 3:
        probabilities = probabilities.sum(axis=-1)   # (B, L, 2) → (B, L)
    if counts.ndim == 2 and counts.shape[1] > 1:
        counts = counts.sum(axis=-1, keepdims=True)  # (B, 2) → (B, 1)
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

    track_ids = [_track_id_for(s) for s in models_to_score]
    if len(track_ids) <= 24:
        logger.info("Track IDs: %s", track_ids)
    else:
        logger.info("Track IDs: %d entries (first 5: %s ... last 5: %s)",
                    len(track_ids), track_ids[:5], track_ids[-5:])

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
    for model_idx, spec in enumerate(models_to_score):
        tid = _track_id_for(spec)
        logger.info("=" * 60)
        logger.info("Model %d/%d: %s (fold %d)", model_idx + 1, n_tracks, tid, args.fold)
        logger.info("=" * 60)

        try:
            # Pass the spec dict as kwargs — chrombpnet.py accepts:
            #   load_pretrained_model(assay='ATAC', cell_type='K562', fold=...)
            #   load_pretrained_model(assay='CHIP', cell_type='K562', TF='REST', fold=...)
            # model_type is pinned to args.model_type (default
            # `chrombpnet_nobias` post-0.3) so the resulting CDF matches
            # what `oracle.predict()` returns for default loads.
            oracle.load_pretrained_model(
                fold=args.fold, model_type=args.model_type, **spec,
            )
        except Exception as exc:
            logger.warning("Failed to load %s: %s", tid, str(exc)[:200])
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

    suffix = _interim_suffix()
    if do_variants:
        effect_matrix = effect_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
        interim_path = os.path.join(cache_dir, f"chrombpnet_effect_cdfs_interim{suffix}.npz")
        _check_interim_compatibility(interim_path, track_ids, args.force, "effect-CDF")
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
        interim_path = os.path.join(cache_dir, f"chrombpnet_baseline_cdfs_interim{suffix}.npz")
        _check_interim_compatibility(interim_path, track_ids, args.force, "baseline-CDF")
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

    Loads the two interim files written by an ``--only-missing --part both``
    run and appends them to the existing NPZ via
    ``PerTrackNormalizer.append_tracks()``, which handles deduplication.
    """
    from chorus.analysis.normalization import PerTrackNormalizer

    effect_path = os.path.join(cache_dir, "chrombpnet_effect_cdfs_interim.npz")
    baseline_path = os.path.join(cache_dir, "chrombpnet_baseline_cdfs_interim.npz")

    if not os.path.exists(effect_path) or not os.path.exists(baseline_path):
        logger.error("Missing interim files — run with --only-missing --part both first")
        return

    effect_data = np.load(effect_path, allow_pickle=False)
    baseline_data = np.load(baseline_path, allow_pickle=False)

    new_ids = list(effect_data["track_ids"].astype(str))
    assert new_ids == list(baseline_data["track_ids"].astype(str)), \
        "interim effect/baseline track_id ordering must agree"

    path, n_added = PerTrackNormalizer.append_tracks(
        oracle_name="chrombpnet",
        new_track_ids=new_ids,
        new_effect_cdfs=effect_data["effect_cdfs"],
        new_summary_cdfs=baseline_data["summary_cdfs"],
        new_perbin_cdfs=baseline_data.get("perbin_cdfs"),
        new_signed_flags=effect_data["signed_flags"],
        new_effect_counts=effect_data.get("effect_counts"),
        new_summary_counts=baseline_data.get("summary_counts"),
        new_perbin_counts=baseline_data.get("perbin_counts"),
        cache_dir=cache_dir,
    )

    total = len(np.load(str(path), allow_pickle=False)["track_ids"])
    logger.info(
        "DONE — merged NPZ has %d tracks (%d new from this run): %s (%.1f MB)",
        total, n_added, path, path.stat().st_size / 1e6,
    )


def merge_shards():
    """Collect interim NPZs from all shards (`.shard<N>of<M>` suffix),
    concatenate by row in shard order, and stitch onto the existing
    NPZ if present (else write a fresh one).

    Run on whichever machine you've ``rsync``'d all the shards onto.
    Expects ``chrombpnet_effect_cdfs_interim.shard*ofM.npz`` and
    ``chrombpnet_baseline_cdfs_interim.shard*ofM.npz`` files at
    ``~/.chorus/backgrounds/``. M is auto-detected from filenames.
    """
    import glob
    import re
    from chorus.analysis.normalization import PerTrackNormalizer

    pattern = os.path.join(cache_dir, "chrombpnet_effect_cdfs_interim.shard*of*.npz")
    effect_files = sorted(glob.glob(pattern))
    if not effect_files:
        logger.error("No shard files found matching %s", pattern)
        return

    # Parse shard indices, verify a contiguous 0..M-1 set.
    shard_re = re.compile(r"shard(\d+)of(\d+)\.npz$")
    shards: dict[int, tuple[str, str]] = {}
    total_shards = None
    for f in effect_files:
        m = shard_re.search(f)
        if not m:
            continue
        idx, total = int(m.group(1)), int(m.group(2))
        if total_shards is None:
            total_shards = total
        elif total_shards != total:
            logger.error("Mismatched --shard-of in shard files: %d vs %d", total_shards, total)
            return
        baseline_f = f.replace("effect", "baseline")
        if not os.path.exists(baseline_f):
            logger.error("Missing baseline shard: %s", baseline_f)
            return
        shards[idx] = (f, baseline_f)

    missing_shards = sorted(set(range(total_shards)) - set(shards))
    if missing_shards:
        logger.error("Missing shards %s of %d total", missing_shards, total_shards)
        return

    logger.info("Merging %d shards.", total_shards)

    all_ids: list[str] = []
    all_effect = []
    all_summary = []
    all_perbin = []
    all_signed = []
    all_effect_counts = []
    all_summary_counts = []
    all_perbin_counts = []

    for i in range(total_shards):
        eff_path, base_path = shards[i]
        eff = np.load(eff_path, allow_pickle=False)
        base = np.load(base_path, allow_pickle=False)
        eff_ids = list(eff["track_ids"].astype(str))
        base_ids = list(base["track_ids"].astype(str))
        assert eff_ids == base_ids, f"shard {i}: effect/baseline track_id ordering must agree"
        all_ids.extend(eff_ids)
        all_effect.append(eff["effect_cdfs"])
        all_summary.append(base["summary_cdfs"])
        all_perbin.append(base["perbin_cdfs"])
        all_signed.append(eff["signed_flags"])
        if "effect_counts" in eff:
            all_effect_counts.append(eff["effect_counts"])
        if "summary_counts" in base:
            all_summary_counts.append(base["summary_counts"])
        if "perbin_counts" in base:
            all_perbin_counts.append(base["perbin_counts"])

    # Concatenate
    new_effect = np.concatenate(all_effect)
    new_summary = np.concatenate(all_summary)
    new_perbin = np.concatenate(all_perbin)
    new_signed = np.concatenate(all_signed)
    new_effect_counts = np.concatenate(all_effect_counts) if all_effect_counts else None
    new_summary_counts = np.concatenate(all_summary_counts) if all_summary_counts else None
    new_perbin_counts = np.concatenate(all_perbin_counts) if all_perbin_counts else None

    # Stitch onto existing NPZ if present (lets a CHIP build extend the
    # ATAC/DNASE 42-track NPZ in place; without an existing NPZ, this
    # writes a CHIP-only file).
    existing_path = os.path.join(cache_dir, "chrombpnet_pertrack.npz")
    if os.path.exists(existing_path):
        existing = np.load(existing_path, allow_pickle=False)
        existing_ids = list(existing["track_ids"].astype(str))
        existing_count = len(existing_ids)
        # De-dup any collisions (e.g. re-running the same shards)
        seen = set(existing_ids)
        merged_ids: list[str] = list(existing_ids)
        keep_mask = np.zeros(len(all_ids), dtype=bool)
        for j, tid in enumerate(all_ids):
            if tid not in seen:
                merged_ids.append(tid)
                keep_mask[j] = True
                seen.add(tid)
        new_effect = new_effect[keep_mask]
        new_summary = new_summary[keep_mask]
        new_perbin = new_perbin[keep_mask]
        new_signed = new_signed[keep_mask]
        if new_effect_counts is not None:
            new_effect_counts = new_effect_counts[keep_mask]
        if new_summary_counts is not None:
            new_summary_counts = new_summary_counts[keep_mask]
        if new_perbin_counts is not None:
            new_perbin_counts = new_perbin_counts[keep_mask]

        merged_effect = np.concatenate([existing["effect_cdfs"], new_effect])
        merged_summary = np.concatenate([existing["summary_cdfs"], new_summary])
        merged_perbin = np.concatenate([existing["perbin_cdfs"], new_perbin])
        merged_signed = np.concatenate([existing["signed_flags"], new_signed])
        merged_effect_counts = (
            np.concatenate([existing["effect_counts"], new_effect_counts])
            if new_effect_counts is not None and "effect_counts" in existing else None
        )
        merged_summary_counts = (
            np.concatenate([existing["summary_counts"], new_summary_counts])
            if new_summary_counts is not None and "summary_counts" in existing else None
        )
        merged_perbin_counts = (
            np.concatenate([existing["perbin_counts"], new_perbin_counts])
            if new_perbin_counts is not None and "perbin_counts" in existing else None
        )
        new_count = sum(keep_mask)
    else:
        merged_ids = all_ids
        merged_effect = new_effect
        merged_summary = new_summary
        merged_perbin = new_perbin
        merged_signed = new_signed
        merged_effect_counts = new_effect_counts
        merged_summary_counts = new_summary_counts
        merged_perbin_counts = new_perbin_counts
        existing_count = 0
        new_count = len(all_ids)

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
        "DONE — merged NPZ has %d tracks (%d existing + %d new from %d shards): %s (%.1f MB)",
        len(merged_ids), existing_count, new_count, total_shards,
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
elif args.part == "merge-shards":
    merge_shards()
elif args.part in ("both", "all"):
    build_all_models(do_variants=True, do_baselines=True)
    if args.shard is not None:
        # Sharded run — DON'T auto-merge. Each shard writes its own
        # interim files; aggregate on one machine via --part merge-shards.
        logger.info("Sharded build complete. Run --part merge-shards on the aggregator.")
    elif args.only_missing:
        merge_to_final_incremental()
    else:
        merge_to_final()
