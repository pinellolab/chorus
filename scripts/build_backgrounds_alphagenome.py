"""Build per-track background distributions for AlphaGenome.

**RUN INSIDE chorus-alphagenome ENV** (not chorus env). Loads the
AlphaGenome model ONCE and runs all predictions in a single process,
avoiding the slow per-call model load that the env-runner pattern causes.

Each track has its own native resolution (1 bp for ATAC/CAGE/RNA/SPLICE/
PROCAP, 128 bp for CHIP_HISTONE/CHIP_TF). RNA-seq tracks use exon-precise
sampling: only bins overlapping GENCODE protein-coding exons contribute.

Run:
  mamba run -n chorus-alphagenome python scripts/build_backgrounds_alphagenome.py --part variants --gpu 0
  mamba run -n chorus-alphagenome python scripts/build_backgrounds_alphagenome.py --part baselines --gpu 0
  mamba run -n chorus python scripts/build_backgrounds_alphagenome.py --part merge
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
parser.add_argument("--fold", type=str, default="all_folds")
parser.add_argument("--n-variants", type=int, default=2000)
parser.add_argument("--n-random", type=int, default=5000)
parser.add_argument("--n-ccre", type=int, default=4000)
parser.add_argument("--n-tss", type=int, default=1000)
parser.add_argument("--n-gene-body", type=int, default=500)
parser.add_argument("--reservoir-size", type=int, default=20000)
parser.add_argument("--n-cdf-points", type=int, default=10000)
parser.add_argument("--perbin-bins", type=int, default=32)
args = parser.parse_args()

log_dir = os.path.join(REPO_ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/bg_alphagenome_{args.part}.log", mode='w'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

cache_dir = os.path.expanduser("~/.chorus/backgrounds")
os.makedirs(cache_dir, exist_ok=True)

INPUT_LENGTH = 1_048_576  # 1 MB
PERBIN_BINS_PER_POSITION = args.perbin_bins

# Layer mapping (chorus_type -> chorus layer + scoring config)
LAYER_FROM_CHORUS_TYPE = {
    'DNASE':        ('chromatin_accessibility', 501,  'log2fc', 1.0,   False),
    'ATAC':         ('chromatin_accessibility', 501,  'log2fc', 1.0,   False),
    'CAGE':         ('tss_activity',            501,  'log2fc', 1.0,   False),
    'PRO_CAP':      ('tss_activity',            501,  'log2fc', 1.0,   False),
    'RNA':          ('gene_expression',         None, 'logfc',  0.001, True),
    'CHIP':         (None,                      None, 'log2fc', 1.0,   False),
    'SPLICE_SITES': ('splicing',                501,  'log2fc', 1.0,   False),
}

HISTONE_PATTERNS = frozenset({
    "H2AFZ", "H2AZ", "H3K4ME1", "H3K4ME2", "H3K4ME3",
    "H3K9AC", "H3K9ME1", "H3K9ME2", "H3K9ME3", "H3K14AC",
    "H3K27AC", "H3K27ME3", "H3K36ME3", "H3K79ME2", "H4K20ME1",
})


# ── Reservoir sampler ────────────────────────────────────────────
class ReservoirSampler:
    def __init__(self, n_tracks: int, capacity: int = 20_000):
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
        # Vectorized batch insertion: while reservoir has capacity, just extend.
        # Once full, switch to per-value reservoir replacement.
        target = self.data[track_idx]
        current_count = int(self.counts[track_idx])
        cap = self.capacity
        n_new = len(values)

        if current_count < cap:
            # We have room — append as many as fit
            room = cap - current_count
            n_to_append = min(room, n_new)
            if n_to_append > 0:
                # Convert to Python floats once, then extend
                if hasattr(values, 'tolist'):
                    target.extend(values[:n_to_append].tolist())
                else:
                    target.extend(float(v) for v in values[:n_to_append])
            # Handle overflow with reservoir replacement
            if n_to_append < n_new:
                rest = values[n_to_append:]
                rng = self._rng
                for off, v in enumerate(rest):
                    n_so_far = current_count + n_to_append + off
                    j = rng.randint(0, n_so_far)
                    if j < cap:
                        target[j] = float(v)
            self.counts[track_idx] += n_new
        else:
            # Full reservoir — per-value replacement
            rng = self._rng
            for off, v in enumerate(values):
                n_so_far = current_count + off
                j = rng.randint(0, n_so_far)
                if j < cap:
                    target[j] = float(v)
            self.counts[track_idx] += n_new

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


def classify_chip_layer(description: str) -> str:
    upper = description.upper()
    for p in HISTONE_PATTERNS:
        if p in upper:
            return 'histone_marks'
    return 'tf_binding'


def compute_effect(ref_val, alt_val, formula, pseudocount):
    if formula == 'log2fc':
        return math.log2((alt_val + pseudocount) / (ref_val + pseudocount))
    elif formula == 'logfc':
        return math.log((alt_val + pseudocount) / (ref_val + pseudocount))
    else:
        return alt_val - ref_val


# ══════════════════════════════════════════════════════════════════
# Load AlphaGenome model directly (must run in chorus-alphagenome env)
# ══════════════════════════════════════════════════════════════════

def load_model_and_track_info():
    """Load AlphaGenome model directly + build track info list.

    This MUST run inside chorus-alphagenome env (alphagenome packages
    must be importable).
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    logger.info("Importing alphagenome packages...")
    import jax
    from alphagenome.models.dna_output import OutputType
    from alphagenome_research.model.dna_model import create_from_huggingface

    # Pick GPU device
    available_platforms = {d.platform for d in jax.devices()}
    if "gpu" in available_platforms:
        jax_device = jax.devices("gpu")[0]
        logger.info("Using JAX GPU device: %s", jax_device)
    else:
        jax_device = jax.devices("cpu")[0]
        logger.info("Falling back to CPU")

    logger.info("Loading AlphaGenome model (fold=%s)...", args.fold)
    t0 = time.time()
    model = create_from_huggingface(args.fold, device=jax_device)
    logger.info("Model loaded in %.1f s", time.time() - t0)

    from chorus.oracles.alphagenome_source.alphagenome_metadata import (
        get_metadata, SKIPPED_OUTPUT_TYPES,
    )
    metadata = get_metadata()
    all_assay_ids = list(metadata._track_index_map.keys())
    logger.info("Total tracks in metadata: %d", len(all_assay_ids))

    track_info = []
    skipped_padding = 0
    for aid in all_assay_ids:
        idx = metadata.get_track_by_identifier(aid)
        if idx is None:
            continue
        info = metadata.get_track_info(idx)
        if info is None:
            continue
        name = info.get('name', '')
        if name and name.lower() == 'padding':
            skipped_padding += 1
            continue
        if '/Padding/' in aid:
            skipped_padding += 1
            continue
        chorus_type = info.get('chorus_type', '')
        desc = info.get('description', '')
        resolution = info.get('resolution', 1)
        ot_name = info.get('output_type', '')
        local_idx = info.get('local_index', 0)

        spec = LAYER_FROM_CHORUS_TYPE.get(chorus_type)
        if spec is None:
            continue
        if spec[0] is None:
            layer = classify_chip_layer(desc)
            window = 2001 if layer == 'histone_marks' else 501
            formula = 'log2fc'
            pseudocount = 1.0
            signed = False
        else:
            layer, window, formula, pseudocount, signed = spec

        track_info.append({
            'assay_id': aid,
            'chorus_type': chorus_type,
            'output_type': ot_name,
            'local_idx': local_idx,
            'layer': layer,
            'window': window,
            'formula': formula,
            'pseudocount': pseudocount,
            'signed': signed,
            'resolution': resolution,
        })

    logger.info("Track info: %d tracks (skipped %d padding)",
                len(track_info), skipped_padding)
    layer_counts = defaultdict(int)
    for t in track_info:
        layer_counts[t['layer']] += 1
    for layer, n in sorted(layer_counts.items()):
        logger.info("  %s: %d", layer, n)

    return model, track_info, OutputType, SKIPPED_OUTPUT_TYPES


def predict_sequence(model, sequence: str, output_types_needed):
    """Run alphagenome prediction. Returns dict of OutputType -> values array.

    output_types_needed: set of OutputType enum values to request.
    """
    output = model.predict_sequence(
        sequence,
        requested_outputs=list(output_types_needed),
        ontology_terms=None,
    )
    return output


def get_window_slice(track, n_bins):
    if track['window'] is None:
        return 0, n_bins
    res = track['resolution']
    center_bin = n_bins // 2
    hw = track['window'] // (2 * res)
    ws = max(0, center_bin - hw)
    we = min(n_bins, center_bin + hw + 1)
    return ws, we


# ── Exon-precise RNA sampling ──
def load_exon_index():
    from chorus.utils.annotations import get_annotation_manager
    ann_manager = get_annotation_manager()
    gtf_path = ann_manager.get_annotation_path('gencode_v48_basic')
    exon_df = ann_manager._get_exons_df(gtf_path)
    gene_df = ann_manager._get_genes_df(gtf_path)
    pc_gene_names = set(gene_df[gene_df['gene_type'] == 'protein_coding']['gene_name'])
    pc_exons = exon_df[exon_df['gene_name'].isin(pc_gene_names)]
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
    total = sum(len(v) for v in by_chrom.values())
    logger.info("Loaded %d merged exons", total)
    return by_chrom


def exon_bin_indices(exon_index, chrom, pred_start, pred_end, n_bins, resolution):
    exons = exon_index.get(chrom, [])
    if not exons:
        return np.array([], dtype=np.int64)
    bin_set = set()
    for es, ee in exons:
        if es >= pred_end:
            break
        if ee <= pred_start:
            continue
        start = max(es, pred_start)
        end = min(ee, pred_end)
        bs = max(0, (start - pred_start) // resolution)
        be = min(n_bins, ((end - pred_start) + resolution - 1) // resolution)
        for b in range(bs, be):
            bin_set.add(b)
    return np.array(sorted(bin_set), dtype=np.int64)


def get_sequence(ref, chrom, pos, length=INPUT_LENGTH):
    half = length // 2
    start, end = pos - half, pos + half
    chrom_len = ref.get_reference_length(chrom)
    if start < 0 or end > chrom_len:
        return None, None, None
    seq = ref.fetch(chrom, start, end).upper()
    if len(seq) != length or seq.count('N') > length * 0.5:
        return None, None, None
    return seq, start, end


# ══════════════════════════════════════════════════════════════════
# VARIANT EFFECT BUILD
# ══════════════════════════════════════════════════════════════════

def build_variant_backgrounds():
    model, track_info, OutputType, SKIPPED = load_model_and_track_info()
    n_tracks = len(track_info)

    logger.info("=" * 60)
    logger.info("PER-TRACK VARIANT BACKGROUNDS: %d SNPs x %d tracks",
                args.n_variants, n_tracks)
    logger.info("=" * 60)

    effect_reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size)
    exon_index = load_exon_index()

    # Determine which output types we need
    needed_ot_names = set(t['output_type'] for t in track_info)
    output_types_needed = [
        ot for ot in OutputType
        if ot.name in needed_ot_names and ot.name not in SKIPPED
    ]
    logger.info("Output types: %s", [ot.name for ot in output_types_needed])

    # Index tracks by output type for fast lookup
    tracks_by_ot = defaultdict(list)  # ot_name -> [(t_i, t_dict), ...]
    for t_i, t in enumerate(track_info):
        tracks_by_ot[t['output_type']].append((t_i, t))

    import pysam
    ref = pysam.FastaFile(os.path.join(REPO_ROOT, 'genomes/hg38.fa'))

    # Generate SNPs
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
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
            eta = (len(snps) - i - 1) / rate if rate > 0 else 0
            logger.info("Variant %d/%d — %.1f min, ETA %.0f min, %s effect samples",
                        i + 1, len(snps), elapsed / 60, eta,
                        f"{effect_reservoir.total_samples():,}")

        chrom, pos = snp["chrom"], snp["pos"]
        seq_ref, pred_start, pred_end = get_sequence(ref, chrom, pos)
        if seq_ref is None:
            continue
        offset = INPUT_LENGTH // 2 - 1
        seq_alt = seq_ref[:offset] + snp["alt"] + seq_ref[offset + 1:]

        try:
            ref_output = predict_sequence(model, seq_ref, output_types_needed)
            alt_output = predict_sequence(model, seq_alt, output_types_needed)

            for ot_name, track_list in tracks_by_ot.items():
                ot_enum = OutputType[ot_name]
                ref_data = ref_output.get(ot_enum)
                alt_data = alt_output.get(ot_enum)
                if ref_data is None or alt_data is None:
                    continue

                ref_arr = np.asarray(ref_data.values)  # (n_bins, n_tracks_in_ot)
                alt_arr = np.asarray(alt_data.values)
                n_bins = ref_arr.shape[0]

                for t_i, t in track_list:
                    li = t['local_idx']
                    if li >= ref_arr.shape[1]:
                        continue
                    ref_vals = ref_arr[:, li]
                    alt_vals = alt_arr[:, li]
                    res = t['resolution']

                    if t['layer'] == 'gene_expression':
                        eb = exon_bin_indices(exon_index, chrom, pred_start, pred_end, n_bins, res)
                        if len(eb) == 0:
                            continue
                        ref_v = float(np.mean(ref_vals[eb]))
                        alt_v = float(np.mean(alt_vals[eb]))
                    else:
                        ws, we = get_window_slice(t, n_bins)
                        ref_v = float(np.sum(ref_vals[ws:we]))
                        alt_v = float(np.sum(alt_vals[ws:we]))

                    score = compute_effect(ref_v, alt_v, t['formula'], t['pseudocount'])
                    if not t['signed']:
                        score = abs(score)
                    effect_reservoir.add(t_i, score)
        except Exception as exc:
            logger.warning("Failed variant %d: %s", i, str(exc)[:200])

    elapsed_v = time.time() - t0
    logger.info("Variants done in %.1f hrs: %s samples",
                elapsed_v / 3600, f"{effect_reservoir.total_samples():,}")

    track_ids = [t['assay_id'] for t in track_info]
    signed_flags = np.array([t['signed'] for t in track_info], dtype=bool)
    effect_matrix = effect_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)

    interim_path = os.path.join(cache_dir, "alphagenome_effect_cdfs_interim.npz")
    np.savez_compressed(
        interim_path,
        track_ids=np.array(track_ids, dtype='U'),
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
    model, track_info, OutputType, SKIPPED = load_model_and_track_info()
    n_tracks = len(track_info)

    logger.info("=" * 60)
    logger.info("PER-TRACK BASELINE BACKGROUNDS: %d tracks", n_tracks)
    logger.info("=" * 60)

    summary_reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size)
    perbin_reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size)
    rng_bins = np.random.RandomState(999)
    exon_index = load_exon_index()

    needed_ot_names = set(t['output_type'] for t in track_info)
    output_types_needed = [
        ot for ot in OutputType
        if ot.name in needed_ot_names and ot.name not in SKIPPED
    ]
    tracks_by_ot = defaultdict(list)
    for t_i, t in enumerate(track_info):
        tracks_by_ot[t['output_type']].append((t_i, t))

    cage_track_indices = set(
        t_i for t_i, t in enumerate(track_info)
        if t['chorus_type'] in ('CAGE', 'PRO_CAP')
    )
    rna_track_indices = set(
        t_i for t_i, t in enumerate(track_info) if t['layer'] == 'gene_expression'
    )

    import pysam
    ref = pysam.FastaFile(os.path.join(REPO_ROOT, 'genomes/hg38.fa'))

    # Position sets
    random.seed(789)
    chroms = [f"chr{i}" for i in range(1, 23)]
    rand_per_chrom = args.n_random // len(chroms) + 1
    rand_positions = []
    for chrom in chroms:
        chrom_len = ref.get_reference_length(chrom)
        max_pos = min(chrom_len - 10_000_000, 200_000_000)
        if max_pos <= 10_000_000:
            max_pos = chrom_len - 1_000_000
        for _ in range(rand_per_chrom):
            if len(rand_positions) >= args.n_random:
                break
            rand_positions.append((chrom, random.randint(10_000_000, max_pos)))
    logger.info("Random positions: %d", len(rand_positions))

    from chorus.utils.annotations import sample_ccre_positions, get_annotation_manager
    ccre_positions = sample_ccre_positions(
        n_per_category={
            "PLS": args.n_ccre * 26 // 100,
            "dELS": args.n_ccre * 22 // 100,
            "pELS": args.n_ccre * 13 // 100,
            "CA-CTCF": args.n_ccre * 13 // 100,
            "CA-TF": args.n_ccre * 9 // 100,
            "TF": args.n_ccre * 4 // 100,
            "CA-H3K4me3": args.n_ccre * 9 // 100,
            "CA": args.n_ccre * 4 // 100,
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
    if len(tss_list) > args.n_tss:
        tss_list = rng_tss.sample(tss_list, args.n_tss)
    logger.info("TSS positions: %d", len(tss_list))

    long_genes = pc_genes[(pc_genes['end'] - pc_genes['start']) > 10000].copy()
    long_genes['midpoint'] = (long_genes['start'] + long_genes['end']) // 2
    gb_dedup = long_genes.groupby('gene_name').first().reset_index()
    rng_gb = random.Random(222)
    gb_list = list(zip(gb_dedup['chrom'], gb_dedup['midpoint']))
    if len(gb_list) > args.n_gene_body:
        gb_list = rng_gb.sample(gb_list, args.n_gene_body)
    logger.info("Gene body midpoints: %d", len(gb_list))

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
    logger.info("Total positions: %d", len(tagged_positions))

    t0 = time.time()
    for i, (chrom, pos, pos_type) in enumerate(tagged_positions):
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / (elapsed / 60) if elapsed > 0 else 0
            eta = (len(tagged_positions) - i - 1) / rate if rate > 0 else 0
            logger.info("Baseline %d/%d (%s:%d %s) — %.1f min, ETA %.0f min, %s summary + %s perbin",
                        i + 1, len(tagged_positions), chrom, pos, pos_type,
                        elapsed / 60, eta,
                        f"{summary_reservoir.total_samples():,}",
                        f"{perbin_reservoir.total_samples():,}")

        seq, pred_start, pred_end = get_sequence(ref, chrom, pos)
        if seq is None:
            continue

        try:
            output = predict_sequence(model, seq, output_types_needed)

            # Cache per-position auxiliary arrays so we don't recompute
            # them per track (this is the optimization that turns the
            # baseline build from ~36s/position to ~5s/position).
            exon_bins_cache = {}     # resolution -> exon bin indices
            window_slice_cache = {}  # (window_bp, resolution, n_bins) -> (ws, we)
            random_bins_cache = {}   # n_bins -> random bin index array

            for ot_name, track_list in tracks_by_ot.items():
                ot_enum = OutputType[ot_name]
                data = output.get(ot_enum)
                if data is None:
                    continue
                arr = np.asarray(data.values)  # (n_bins, n_tracks_in_ot)
                n_bins, n_local_tracks = arr.shape

                # Pre-compute random bin sample (shared across all tracks
                # at this output_type since they have the same n_bins)
                if n_bins not in random_bins_cache:
                    n_take = min(PERBIN_BINS_PER_POSITION, n_bins)
                    random_bins_cache[n_bins] = rng_bins.choice(
                        n_bins, n_take, replace=False,
                    )
                rand_sample = random_bins_cache[n_bins]

                for t_i, t in track_list:
                    li = t['local_idx']
                    if li >= n_local_tracks:
                        continue
                    vals = arr[:, li]
                    res = t['resolution']
                    is_cage = t_i in cage_track_indices
                    is_rna = t_i in rna_track_indices

                    # Compute exon bins ONCE per resolution (not per track)
                    if is_rna:
                        if res not in exon_bins_cache:
                            exon_bins_cache[res] = exon_bin_indices(
                                exon_index, chrom, pred_start, pred_end, n_bins, res,
                            )
                        eb = exon_bins_cache[res]
                        if len(eb) == 0:
                            continue

                        # Summary: mean over exon bins
                        signal = float(np.mean(vals[eb]))
                        summary_reservoir.add(t_i, signal)

                        # Perbin: sample from exon bins
                        if len(eb) > PERBIN_BINS_PER_POSITION:
                            # Subsample once per resolution
                            cache_key = ('rna_subsample', res)
                            if cache_key not in random_bins_cache:
                                random_bins_cache[cache_key] = rng_bins.choice(
                                    eb, PERBIN_BINS_PER_POSITION, replace=False,
                                )
                            ebs = random_bins_cache[cache_key]
                        else:
                            ebs = eb
                        perbin_reservoir.add_batch(t_i, vals[ebs])
                    else:
                        # Summary: window-sum (skip CAGE at cCREs)
                        if not (is_cage and pos_type == 'ccre'):
                            wkey = (t['window'], res, n_bins)
                            if wkey not in window_slice_cache:
                                window_slice_cache[wkey] = get_window_slice(t, n_bins)
                            ws, we = window_slice_cache[wkey]
                            signal = float(np.sum(vals[ws:we]))
                            summary_reservoir.add(t_i, signal)

                        # Perbin: random bins from full output
                        perbin_reservoir.add_batch(t_i, vals[rand_sample])
        except Exception as exc:
            logger.warning("Failed %s:%d: %s", chrom, pos, str(exc)[:200])

    elapsed_b = time.time() - t0
    logger.info("Baselines done in %.1f hrs", elapsed_b / 3600)

    track_ids = [t['assay_id'] for t in track_info]
    summary_matrix = summary_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
    perbin_matrix = perbin_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)

    interim_path = os.path.join(cache_dir, "alphagenome_baseline_cdfs_interim.npz")
    np.savez_compressed(
        interim_path,
        track_ids=np.array(track_ids, dtype='U'),
        summary_cdfs=summary_matrix.astype(np.float32),
        summary_counts=summary_reservoir.get_counts(),
        perbin_cdfs=perbin_matrix.astype(np.float32),
        perbin_counts=perbin_reservoir.get_counts(),
    )
    logger.info("Saved baseline interim: %s", interim_path)
    ref.close()


def merge_to_final():
    from chorus.analysis.normalization import PerTrackNormalizer

    effect_path = os.path.join(cache_dir, "alphagenome_effect_cdfs_interim.npz")
    baseline_path = os.path.join(cache_dir, "alphagenome_baseline_cdfs_interim.npz")
    if not os.path.exists(effect_path) or not os.path.exists(baseline_path):
        logger.error("Missing interim files")
        return

    effect_data = np.load(effect_path, allow_pickle=False)
    baseline_data = np.load(baseline_path, allow_pickle=False)

    effect_ids = list(effect_data["track_ids"].astype(str))
    baseline_ids = list(baseline_data["track_ids"].astype(str))
    assert effect_ids == baseline_ids

    path = PerTrackNormalizer.build_and_save(
        oracle_name="alphagenome",
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
