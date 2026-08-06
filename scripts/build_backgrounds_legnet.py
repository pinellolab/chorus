"""Build per-track background distributions for LegNet.

LegNet predicts MPRA promoter_activity for 3 cell types: K562, HepG2, WTC11.
Each cell type is a track. Produces ``legnet_pertrack.npz`` with effect_cdfs
and summary_cdfs per cell type. No perbin CDFs since LegNet output is scalar.

Input: 200 bp window. Very fast.
promoter_activity layer: diff formula (alt - ref), signed.

Run in chorus-legnet env:
  mamba run -n chorus-legnet python scripts/build_backgrounds_legnet.py --part variants
  mamba run -n chorus-legnet python scripts/build_backgrounds_legnet.py --part baselines
  mamba run -n chorus python scripts/build_backgrounds_legnet.py --part merge
"""
import argparse
import logging
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

import os; REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, REPO_ROOT)

from chorus.utils.annotations import (  # noqa: E402
    load_chrom_sizes,
    sample_promoter_anchored_positions,
)
from chorus.analysis.background_sampling import (  # noqa: E402
    ReservoirSampler,
    sampling_block,
)
os.environ["CHORUS_NO_TIMEOUT"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--part", choices=["variants", "baselines", "merge", "both", "all"], default="all")
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--n-variants", type=int, default=10000)
parser.add_argument("--reservoir-size", type=int, default=50000)
parser.add_argument("--no-dhs", action="store_true",
                    help="Ablation: drop the DHS stratum and rescale the promoter "
                         "strata to their original proportions. Mirrors the existing "
                         "--no-dhs on the Cherimoya builder. Exists so the effect of "
                         "adding DHS to a PROMOTER model's null can be measured "
                         "rather than argued about.")
parser.add_argument("--tail-k", type=int, default=0,
                    help="Keep the exact top/bottom K values per track alongside the "
                         "uniform body. 0 = off, which is correct whenever the offered "
                         "count is under the reservoir capacity (LegNet offers 11,913 "
                         "effect and 29,002 summary values against 50,000).")
parser.add_argument("--n-cdf-points", type=int, default=10000)
args = parser.parse_args()

log_dir = os.path.join(REPO_ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/bg_legnet_{args.part}.log", mode='w'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Honour the data-dir mechanism rather than hardcoding $HOME. All eight
# builders had this literal, so a chorus installed with
# CHORUS_DATA_DIR=/data/... still wrote its backgrounds into the home
# directory the data dir exists to avoid. CHORUS_BACKGROUNDS_DIR applies
# the legacy ~/.chorus compatibility itself, per kind.
from chorus.core.globals import CHORUS_BACKGROUNDS_DIR
cache_dir = os.environ.get("CHORUS_BUILD_CACHE_DIR") or str(CHORUS_BACKGROUNDS_DIR)
os.makedirs(cache_dir, exist_ok=True)


# ── Reservoir sampler ─────────────────────────────────────────────
# ReservoirSampler now comes from chorus.analysis.background_sampling
# (imported above). The local copy was proved byte-identical to the shared one
# before removal, and the behaviour is pinned permanently by the golden values
# in tests/test_background_sampling.py. See #125.


def load_setup():
    """Load reference + setup. Returns (cell_types, get_sequence, predict_activity_factory, ref)."""
    import torch
    import pysam

    from chorus.oracles.legnet_source.legnet_globals import LEGNET_WINDOW, LEGNET_AVAILABLE_CELLTYPES
    from chorus.oracles.legnet_source.model_usage import load_model, predict_bigseq
    from chorus.oracles.legnet_source.agarwal_meta import LEFT_MPRA_FLANK, RIGHT_MPRA_FLANK
    from chorus.core.globals import CHORUS_DOWNLOADS_DIR

    LEGNET_MODELS_DIR = CHORUS_DOWNLOADS_DIR / "legnet"

    device = torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    logger.info("Device: %s", device)

    ref = pysam.FastaFile(os.path.join(REPO_ROOT, "genomes/hg38.fa"))
    cell_types = list(LEGNET_AVAILABLE_CELLTYPES)

    def get_sequence(chrom, pos):
        half = LEGNET_WINDOW // 2
        start, end = pos - half, pos + half
        chrom_len = ref.get_reference_length(chrom)
        if start < 0 or end > chrom_len:
            return None
        seq = ref.fetch(chrom, start, end).upper()
        if len(seq) != LEGNET_WINDOW or seq.count('N') > LEGNET_WINDOW * 0.3:
            return None
        return seq

    def load_cell_type_model(cell_type: str):
        weights_dir = LEGNET_MODELS_DIR / f"LentiMPRA_{cell_type}"
        config_path = weights_dir / 'config.json'
        weights_path = weights_dir / 'example' / 'weights.ckpt'
        if not weights_path.exists():
            return None
        model = load_model(config_path, weights_path)
        model.to(device)
        model.eval()
        return model

    def predict_activity(model, seq):
        preds, _ = predict_bigseq(
            model, seq=seq,
            reverse_aug=False,
            window_size=LEGNET_WINDOW,
            step=LEGNET_WINDOW,
            left_flank=LEFT_MPRA_FLANK,
            right_flank=RIGHT_MPRA_FLANK,
            batch_size=1,
        )
        return float(np.mean(preds))

    return cell_types, get_sequence, load_cell_type_model, predict_activity, ref, device


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
    # LegNet is a promoter MPRA model: 200 bp input, window_bp=None, so the sampled
    # position IS the whole thing being modelled. Anchored on promoters, NOT on the
    # generic cCRE mix (62% of that catalogue is distal enhancer-like) and not on DHS
    # summits, which track accessibility rather than promoter identity. See
    # PROMOTER_REGION_STRATA.
    #
    # This was a uniformly random draw over chr1-chr22. For an assay with localised
    # signal that is the wrong reference population: a random position carries almost
    # no signal, so the pseudocount damps its log-ratio toward zero and the null's
    # body collapses below where real regulatory effects live. Five of the eight
    # oracles already anchored their effect null on peaks; this one did not, even
    # though its own BASELINE pass already used cCREs -- an asymmetry inside a single
    # oracle that is harder to defend than any difference between oracles.
    _sizes = load_chrom_sizes(os.path.join(REPO_ROOT, 'genomes/hg38.fa.fai'))
    _strata = None
    if args.no_dhs:
        # Ablation: drop DHS and restore the original promoter proportions, so the two
        # runs differ ONLY in whether DHS is present.
        from chorus.utils.annotations import PROMOTER_REGION_STRATA
        _strata = {k: v for k, v in PROMOTER_REGION_STRATA.items() if k != "dhs"}
        _tot = sum(_strata.values())
        _strata = {k: v / _tot for k, v in _strata.items()}
        logger.info("ABLATION --no-dhs: strata rescaled to %s", _strata)
    sampled = sample_promoter_anchored_positions(
        args.n_variants, chrom_sizes=_sizes, seed=42, strata=_strata)
    snps = []
    strata_counts = defaultdict(int)
    for chrom, pos, stratum in sampled:
        ref_base = ref.fetch(chrom, pos - 1, pos).upper()
        if ref_base not in "ACGT":
            continue  # N or soft-masked; the tally records the shortfall
        snps.append({"chrom": chrom, "pos": pos, "ref": ref_base,
                     "alt": random.choice([b for b in "ACGT" if b != ref_base]),
                     "stratum": stratum})
        strata_counts[stratum] += 1
    random.shuffle(snps)
    logger.info("Generated %d promoter-anchored SNPs from %d sampled positions: %s",
                len(snps), len(sampled), dict(strata_counts))

    import torch

    for ct_i, cell_type in enumerate(cell_types):
        logger.info("Loading LegNet for %s...", cell_type)
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
        torch.cuda.empty_cache()

    effect_matrix = effect_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
    signed_flags = np.ones(n_tracks, dtype=bool)  # LegNet uses diff, signed

    interim_path = os.path.join(cache_dir, "legnet_effect_cdfs_interim.npz")
    np.savez_compressed(
        interim_path,
        track_ids=np.array(cell_types, dtype='U'),
        effect_cdfs=effect_matrix.astype(np.float32),
        effect_counts=effect_reservoir.get_counts(),
        # Retention beside the offered count, so "was the tail thinned?" is answerable
        # from the artefact. Only the offered count was ever written, which is why the
        # AlphaGenome thinning was invisible for months.
        effect_retained=effect_reservoir.retained_counts(),
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
        logger.info("Loading LegNet for %s...", cell_type)
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
        torch.cuda.empty_cache()

    summary_matrix = summary_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
    interim_path = os.path.join(cache_dir, "legnet_baseline_cdfs_interim.npz")
    np.savez_compressed(
        interim_path,
        track_ids=np.array(cell_types, dtype='U'),
        summary_cdfs=summary_matrix.astype(np.float32),
        summary_counts=summary_reservoir.get_counts(),
        summary_retained=summary_reservoir.retained_counts(),
    )
    logger.info("Saved baseline interim: %s", interim_path)
    ref.close()


def merge_to_final():
    from chorus.analysis.normalization import PerTrackNormalizer

    effect_path = os.path.join(cache_dir, "legnet_effect_cdfs_interim.npz")
    baseline_path = os.path.join(cache_dir, "legnet_baseline_cdfs_interim.npz")
    if not os.path.exists(effect_path) or not os.path.exists(baseline_path):
        logger.error("Missing interim files")
        return

    effect_data = np.load(effect_path, allow_pickle=False)
    baseline_data = np.load(baseline_path, allow_pickle=False)

    effect_ids = list(effect_data["track_ids"].astype(str))
    baseline_ids = list(baseline_data["track_ids"].astype(str))
    assert effect_ids == baseline_ids

    path = PerTrackNormalizer.build_and_save(
        oracle_name="legnet",
        track_ids=effect_ids,
        effect_cdfs=effect_data["effect_cdfs"],
        summary_cdfs=baseline_data["summary_cdfs"],
        perbin_cdfs=None,  # No perbin for scalar outputs
        signed_flags=effect_data["signed_flags"],
        effect_counts=effect_data["effect_counts"] if "effect_counts" in effect_data else None,
        summary_counts=baseline_data["summary_counts"] if "summary_counts" in baseline_data else None,
        cache_dir=cache_dir,
        sampling=sampling_block(effect_data, baseline_data),
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
