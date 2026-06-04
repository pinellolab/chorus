"""Build per-cell background CDFs for EPInformer-seq.

Per-cell = `PerCellProfileNetWide` (2114->1024, no FiLM, no cell embedding) +
frozen per-cell `BiasNet`. One model per cell, DNase-only.

Builds the full ``{oracle}_pertrack.npz`` background bundle for the 11 cells:

* ``effect_cdfs``  — unsigned ``|log2((alt+1)/(ref+1))|`` over ~10 000 random
  SNPs (one ref/alt activity pair per cell). Used for variant-effect
  percentile lookup.
* ``summary_cdfs`` — per-cell activity baseline CDF over ~34 000 positions
  (15 000 random + ~11 500 cCRE + 3 000 TSS + 5 000 DHS-summit). Used for
  locus-level 90th-percentile thresholds in visualization.

Output: ``~/.chorus/backgrounds/epinformerseq_pertrack.npz`` with
``track_ids``, ``summary_cdfs``, ``summary_counts``,
``effect_cdfs``, ``effect_counts``.

Activity derivation: the main (PerCellProfileNetWide, 2114->1024) + bias produce
per-bp profile logits + log10(count). We compute
``signal[i,c] = softmax(main+bias)[i,c] * 10**log_count[c]``, then take
``max_DNase`` over the **central 256 bp** — the same DNase-only slice the oracle
uses at inference (``model_usage.predict_activity``), so percentile lookups are
consistent with the runtime scalar.

Stages — pick with ``--part {variants, baselines, merge, all}``:
  variants : score random SNPs → effect_cdfs interim NPZ
  baselines: score genome positions → summary_cdfs interim NPZ
  merge    : combine interims into the final ``{oracle}_pertrack.npz``
  all      : run all three stages in order (default)

Run in chorus-epinformerseq env:
  mamba run -n chorus-epinformerseq python scripts/build_backgrounds_epinformerseq_v2_percell.py
"""
import argparse
import logging
import os
import random
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# Default to the chorus oracle download cache (populated on first use via the
# HF mirror lucapinello/chorus-epinformerseq-v2). Override with --percell-root
# and --bias-root if you want to point at a local training-output tree.
from chorus.core.globals import CHORUS_DOWNLOADS_DIR  # noqa: E402
_CHORUS_PERCELL_ROOT = str(CHORUS_DOWNLOADS_DIR / "epinformerseq")
V2_PERCELL_DIR_DEFAULT = os.path.join(_CHORUS_PERCELL_ROOT, "per_cell_widewin")
V2_BIAS_DIR_DEFAULT    = os.path.join(_CHORUS_PERCELL_ROOT, "bias")

parser = argparse.ArgumentParser()
parser.add_argument("--percell-root", default=V2_PERCELL_DIR_DEFAULT,
                    help="Dir holding <cell>/main.pt (PerCellProfileNetWide).")
parser.add_argument("--bias-root", default=V2_BIAS_DIR_DEFAULT,
                    help="Dir holding <cell>/bias.pt frozen BiasNets.")
parser.add_argument("--cells", nargs="+",
                    default=["K562", "GM12878", "HepG2", "A549",
                             "H1", "HeLa", "HMEC", "HSMM",
                             "HUVEC", "NHEK", "NHLF"])
parser.add_argument("--dhs-path", default=None)
parser.add_argument("--n-random", type=int, default=15_000)
parser.add_argument("--n-tss", type=int, default=3_000)
parser.add_argument("--n-dhs-peaks", type=int, default=5_000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--reservoir-size", type=int, default=50_000)
parser.add_argument("--n-cdf-points", type=int, default=10_000)
parser.add_argument("--output-name", default="epinformerseq_pertrack.npz",
                    help="Final NPZ name.")
parser.add_argument("--part", choices=["variants", "baselines", "merge", "all"], default="all",
                    help="Which CDF(s) to build. 'all' runs variants + baselines + merge.")
parser.add_argument("--n-variants", type=int, default=10_000,
                    help="Random SNPs for effect_cdfs.")
args = parser.parse_args()

# ---- model constants ------------------------------------------------------
# EPInformer-seq is PerCellProfileNetWide: a 2114-bp input cropped to a central
# 1024-bp profile, 2 channels (ch0 DNase cut-site, ch1 H3K27ac coverage). We
# build one CDF track per (assay, cell): max DNase, max H3K27ac, and the
# composite sqrt(maxD*maxH), all over the central 256 bp.
IN_WINDOW = 2114
ASSAYS = ["Enhancer_DNase", "Enhancer_H3K27ac", "Enhancer_H3K27ac_DNase"]
INTERIM_TAG = ""

log_dir = os.path.join(REPO_ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{log_dir}/bg_epinformerseq_v2_percell.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

cache_dir = os.path.expanduser("~/.chorus/backgrounds")
os.makedirs(cache_dir, exist_ok=True)

V2_WINDOW = 1024
CENTRAL = 256
CENTRAL_START = (V2_WINDOW - CENTRAL) // 2     # 384
CENTRAL_END   = CENTRAL_START + CENTRAL        # 640


class ReservoirSampler:
    def __init__(self, n_tracks, capacity=50_000):
        self.n_tracks = n_tracks
        self.capacity = capacity
        self.data = [[] for _ in range(n_tracks)]
        self.counts = np.zeros(n_tracks, dtype=np.int64)
        self._rng = random.Random(12345)

    def add(self, track_idx, value):
        n = self.counts[track_idx]
        if n < self.capacity:
            self.data[track_idx].append(value)
        else:
            j = self._rng.randint(0, n)
            if j < self.capacity:
                self.data[track_idx][j] = value
        self.counts[track_idx] += 1

    def get_sorted(self, track_idx):
        arr = np.array(self.data[track_idx], dtype=np.float64)
        arr.sort()
        return arr

    def to_cdf_matrix(self, n_points=10_000):
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


def one_hot(seq):
    a = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        if   b == "A": a[0, i] = 1.0
        elif b == "C": a[1, i] = 1.0
        elif b == "G": a[2, i] = 1.0
        elif b == "T": a[3, i] = 1.0
    return a


def sample_positions():
    import pysam
    ref = pysam.FastaFile(os.path.join(REPO_ROOT, "genomes", "hg38.fa"))

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
    logger.info("random positions: %d", len(rand_positions))

    from chorus.utils.annotations import (
        sample_ccre_positions, get_annotation_manager,
    )
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
    gtf_path = ann_manager.get_annotation_path("gencode_v48_basic")
    gene_df = ann_manager._get_genes_df(gtf_path)
    pc_genes = gene_df[gene_df["gene_type"] == "protein_coding"].copy()
    pc_genes["tss"] = pc_genes.apply(
        lambda r: r["start"] if r["strand"] == "+" else r["end"], axis=1)
    valid_chroms = {f"chr{i}" for i in range(1, 23)}
    pc_genes = pc_genes[pc_genes["chrom"].isin(valid_chroms)]
    tss_dedup = pc_genes.groupby("gene_name").first().reset_index()
    rng_tss = random.Random(111)
    tss_list = list(zip(tss_dedup["chrom"], tss_dedup["tss"].astype(int)))
    if len(tss_list) > args.n_tss:
        tss_list = rng_tss.sample(tss_list, args.n_tss)
    logger.info("TSS positions: %d", len(tss_list))

    dhs_positions = []
    if args.n_dhs_peaks > 0:
        from chorus.utils.annotations import sample_dhs_positions
        dhs_path = args.dhs_path or os.path.join(
            REPO_ROOT, "annotations", "dhs_vocabulary_hg38.txt.gz")
        dhs_positions = sample_dhs_positions(
            args.n_dhs_peaks, dhs_path=dhs_path, seed=567,
        )
        logger.info("DHS positions: %d", len(dhs_positions))

    all_positions = []
    for chrom, pos in rand_positions:    all_positions.append((chrom, int(pos)))
    for chrom, pos in ccre_positions:    all_positions.append((chrom, int(pos)))
    for chrom, pos in tss_list:          all_positions.append((chrom, int(pos)))
    for chrom, pos in dhs_positions:     all_positions.append((chrom, int(pos)))
    random.Random(2026).shuffle(all_positions)
    logger.info(
        "TOTAL positions: %d (random=%d, cCRE=%d, TSS=%d, DHS=%d)",
        len(all_positions), len(rand_positions), len(ccre_positions),
        len(tss_list), len(dhs_positions),
    )

    half = IN_WINDOW // 2
    seqs = []
    for chrom, pos in all_positions:
        start, end = pos - half, pos + half
        chrom_len = ref.get_reference_length(chrom)
        if start < 0 or end > chrom_len:
            continue
        seq = ref.fetch(chrom, start, end).upper()
        if len(seq) != IN_WINDOW or seq.count("N") > IN_WINDOW * 0.3:
            continue
        seqs.append(seq)
    ref.close()
    logger.info("Valid sequences after fetch: %d", len(seqs))
    return seqs


def _load_models_and_device():
    """Load 11 per-cell main + bias models onto a device."""
    import torch
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # Use the canonical chorus model classes — same nn.Module, byte-compatible
    # with the per-cell state_dicts on HF (lucapinello/chorus-epinformerseq-v2).
    from chorus.oracles.epinformerseq_source.model import (
        PerCellProfileNetWide, BiasNet,
    )

    cells = args.cells
    main_models, bias_models = [], []
    for c in cells:
        main_ckpt = os.path.join(args.percell_root, c, "main.pt")
        bias_ckpt = os.path.join(args.bias_root, c, "bias.pt")
        if not os.path.exists(main_ckpt) or not os.path.exists(bias_ckpt):
            # Populate the chorus oracle cache via the HF mirror if missing.
            from chorus.oracles import EPInformerSeqOracle
            EPInformerSeqOracle(cell_type=c, use_environment=False).load_pretrained_model()
        m = PerCellProfileNetWide(in_window=IN_WINDOW, out_window=1024)
        m.load_state_dict(torch.load(main_ckpt, map_location="cpu", weights_only=False))
        m.eval().to(device)
        b = BiasNet()
        b.load_state_dict(torch.load(bias_ckpt, map_location="cpu", weights_only=False))
        for p in b.parameters():
            p.requires_grad_(False)
        b.eval().to(device)
        main_models.append(m); bias_models.append(b)
        logger.info("loaded per-cell %s  (main %.1fK + bias %.1fK params)", c,
                    sum(p.numel() for p in m.parameters()) / 1e3,
                    sum(p.numel() for p in b.parameters()) / 1e3)
    return cells, main_models, bias_models, device


def _score_activity_batch(main_models, bias_models, x):
    """Score one (B, 4, IN_WINDOW) batch -> (n_cells, 3, B) for the 3 assays.

    Main sees the full 2114-bp input; the 1024-bp BiasNet sees the central
    1024-bp crop. Per cell we emit three scalars over the central 256 bp:
    [max DNase, max H3K27ac, sqrt(max DNase * max H3K27ac)] (ASSAYS order).
    """
    import torch
    import torch.nn.functional as F
    pad = (IN_WINDOW - 1024) // 2
    x_bias = x if pad == 0 else x[:, :, pad:pad + 1024].contiguous()
    rows = []
    for m, b in zip(main_models, bias_models):
        mp, mc = m(x)                          # (B,2,1024), (B,2)
        bp, _  = b(x_bias)                      # (B,2,1024)
        soft  = F.softmax(mp + bp, dim=-1)
        signal = soft * (10.0 ** mc).unsqueeze(-1)   # (B,2,1024)
        pmax = signal[:, :, CENTRAL_START:CENTRAL_END].max(dim=-1).values  # (B,2)
        d = pmax[:, 0].clamp(min=0)
        h = pmax[:, 1].clamp(min=0)
        rows.append(torch.stack([d, h, torch.sqrt(d * h)], dim=0))  # (3,B)
    return torch.stack(rows, dim=0)            # (n_cells, 3, B)


def build_baseline_backgrounds():
    import torch
    logger.info("=" * 60)
    logger.info("BASELINE (summary) CDFs — random + cCRE + TSS + DHS positions")
    logger.info("=" * 60)
    cells, main_models, bias_models, device = _load_models_and_device()
    seqs = sample_positions()
    n = len(seqs)
    n_assays = len(ASSAYS)
    n_tracks = len(cells) * n_assays
    track_ids = [f"{a}:{c}" for c in cells for a in ASSAYS]   # cell-major, assay-minor
    reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size)

    t0 = time.time()
    with torch.inference_mode():
        for i0 in range(0, n, args.batch_size):
            batch = seqs[i0:i0 + args.batch_size]
            oh = np.stack([one_hot(s) for s in batch], axis=0)
            x = torch.from_numpy(oh).to(device)
            acts = _score_activity_batch(main_models, bias_models, x)  # (n_cells, 3, B)
            acts = acts.float().cpu().numpy()
            for ct_i in range(len(cells)):
                for ai in range(n_assays):
                    tidx = ct_i * n_assays + ai
                    for v in acts[ct_i, ai]:
                        reservoir.add(tidx, float(v))
            if (i0 // args.batch_size) % 50 == 0:
                logger.info("  baseline batch %d / %d  (elapsed %.1fs)",
                            i0 // args.batch_size, (n - 1) // args.batch_size + 1,
                            time.time() - t0)
    logger.info("Baseline inference done in %.1f min", (time.time() - t0) / 60)

    summary_matrix = reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
    interim = os.path.join(cache_dir, f"epinformerseq_{INTERIM_TAG}baseline_cdfs_interim.npz")
    np.savez_compressed(
        interim,
        track_ids=np.array(track_ids, dtype="U"),
        summary_cdfs=summary_matrix.astype(np.float32),
        summary_counts=reservoir.counts.copy(),
    )
    logger.info("Saved baseline interim: %s", interim)
    for i, tid in enumerate(track_ids):
        thr90 = float(np.quantile(summary_matrix[i], 0.90))
        logger.info("  %-28s  90th-pct %.3f  (n=%d)", tid, thr90, int(reservoir.counts[i]))


def build_variant_backgrounds():
    """Score random SNPs to populate effect_cdfs (|log2((alt+1)/(ref+1))|, unsigned)."""
    import pysam
    import torch
    logger.info("=" * 60)
    logger.info("VARIANT (effect) CDFs — %d random SNPs x %d cells",
                args.n_variants, len(args.cells))
    logger.info("=" * 60)
    cells, main_models, bias_models, device = _load_models_and_device()
    n_assays = len(ASSAYS)
    n_tracks = len(cells) * n_assays
    track_ids = [f"{a}:{c}" for c in cells for a in ASSAYS]

    ref = pysam.FastaFile(os.path.join(REPO_ROOT, "genomes", "hg38.fa"))
    rng = random.Random(42)
    chroms = [f"chr{i}" for i in range(1, 23)]
    snps_per_chrom = args.n_variants // len(chroms) + 1
    snps = []
    for chrom in chroms:
        chrom_len = ref.get_reference_length(chrom)
        max_pos = min(chrom_len - 5_000_000, 200_000_000)
        for _ in range(snps_per_chrom):
            if len(snps) >= args.n_variants:
                break
            pos = rng.randint(5_000_000, max_pos)
            ref_base = ref.fetch(chrom, pos - 1, pos).upper()
            if ref_base not in "ACGT":
                continue
            alt_base = rng.choice([b for b in "ACGT" if b != ref_base])
            snps.append((chrom, pos, ref_base, alt_base))
    rng.shuffle(snps)
    snps = snps[: args.n_variants]
    logger.info("Generated %d random SNPs", len(snps))

    # Build ref + alt sequences as numpy arrays once, then batch through.
    half = IN_WINDOW // 2
    ref_seqs, alt_seqs, kept = [], [], 0
    for chrom, pos, ref_base, alt_base in snps:
        start = pos - half
        end   = start + IN_WINDOW
        chrom_len = ref.get_reference_length(chrom)
        if start < 0 or end > chrom_len:
            continue
        seq = ref.fetch(chrom, start, end).upper()
        if len(seq) != IN_WINDOW or seq.count("N") > IN_WINDOW * 0.3:
            continue
        # Build alt by replacing the base at the variant position (1-based -> 0-based offset within window).
        offset = pos - start - 1   # variant base index within [0, IN_WINDOW)
        if seq[offset] != ref_base:
            # ref allele mismatch — skip (probably hit a low-complexity region)
            continue
        alt_seq = seq[:offset] + alt_base + seq[offset + 1:]
        ref_seqs.append(seq)
        alt_seqs.append(alt_seq)
        kept += 1
    ref.close()
    logger.info("Variants after sequence fetch + ref-allele check: %d / %d", kept, len(snps))

    reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size)
    t0 = time.time()
    with torch.inference_mode():
        for i0 in range(0, kept, args.batch_size):
            br = ref_seqs[i0:i0 + args.batch_size]
            ba = alt_seqs[i0:i0 + args.batch_size]
            oh_r = np.stack([one_hot(s) for s in br], axis=0)
            oh_a = np.stack([one_hot(s) for s in ba], axis=0)
            x_ref = torch.from_numpy(oh_r).to(device)
            x_alt = torch.from_numpy(oh_a).to(device)
            ref_act = _score_activity_batch(main_models, bias_models, x_ref)  # (n_cells,3,B)
            alt_act = _score_activity_batch(main_models, bias_models, x_alt)
            # |log2((alt + 1) / (ref + 1))|, unsigned (chromatin convention).
            eff = torch.abs(torch.log2((alt_act + 1.0) / (ref_act + 1.0)))
            eff = eff.float().cpu().numpy()                                   # (n_cells,3,B)
            for ct_i in range(len(cells)):
                for ai in range(n_assays):
                    tidx = ct_i * n_assays + ai
                    for v in eff[ct_i, ai]:
                        reservoir.add(tidx, float(v))
            if (i0 // args.batch_size) % 25 == 0:
                logger.info("  variant batch %d / %d  (elapsed %.1fs)",
                            i0 // args.batch_size, (kept - 1) // args.batch_size + 1,
                            time.time() - t0)
    logger.info("Variant inference done in %.1f min", (time.time() - t0) / 60)

    effect_matrix = reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
    interim = os.path.join(cache_dir, f"epinformerseq_{INTERIM_TAG}effect_cdfs_interim.npz")
    np.savez_compressed(
        interim,
        track_ids=np.array(track_ids, dtype="U"),
        effect_cdfs=effect_matrix.astype(np.float32),
        effect_counts=reservoir.counts.copy(),
        signed_flags=np.zeros(n_tracks, dtype=bool),  # unsigned |log2fc|
    )
    logger.info("Saved effect interim: %s", interim)
    for i, tid in enumerate(track_ids):
        p90 = float(np.quantile(effect_matrix[i], 0.90))
        logger.info("  %-28s  90th-pct |log2fc| %.4f  (n=%d)", tid, p90, int(reservoir.counts[i]))


def merge_to_final():
    """Combine effect + summary interim NPZs into ``epinformerseq_pertrack.npz``."""
    from chorus.analysis.normalization import PerTrackNormalizer
    out_oracle = "epinformerseq"
    effect_path   = os.path.join(cache_dir, f"epinformerseq_{INTERIM_TAG}effect_cdfs_interim.npz")
    baseline_path = os.path.join(cache_dir, f"epinformerseq_{INTERIM_TAG}baseline_cdfs_interim.npz")
    if not os.path.exists(baseline_path):
        logger.error("Missing baseline interim: %s — run --part baselines first.", baseline_path)
        return
    have_effect = os.path.exists(effect_path)
    if not have_effect:
        logger.warning("Missing effect interim: %s — final NPZ will only have summary_cdfs.",
                       effect_path)

    baseline = np.load(baseline_path, allow_pickle=False)
    baseline_ids = list(baseline["track_ids"].astype(str))
    if have_effect:
        effect = np.load(effect_path, allow_pickle=False)
        effect_ids = list(effect["track_ids"].astype(str))
        assert effect_ids == baseline_ids, "track id mismatch between effect + baseline interims"
        signed_flags = effect["signed_flags"]
    else:
        effect = None
        signed_flags = np.zeros(len(baseline_ids), dtype=bool)

    path = PerTrackNormalizer.build_and_save(
        oracle_name=out_oracle,
        track_ids=baseline_ids,
        effect_cdfs=effect["effect_cdfs"] if have_effect else None,
        summary_cdfs=baseline["summary_cdfs"],
        perbin_cdfs=None,                              # scalar-output oracle — perbin not applicable
        signed_flags=signed_flags,
        effect_counts=(effect["effect_counts"] if have_effect else None),
        summary_counts=baseline["summary_counts"],
        cache_dir=cache_dir,
    )
    size_mb = path.stat().st_size / 1e6
    logger.info("DONE — wrote %s  (%.2f MB)", path, size_mb)


if __name__ == "__main__":
    if args.part == "variants":
        build_variant_backgrounds()
    elif args.part == "baselines":
        build_baseline_backgrounds()
    elif args.part == "merge":
        merge_to_final()
    elif args.part == "all":
        build_variant_backgrounds()
        build_baseline_backgrounds()
        merge_to_final()
