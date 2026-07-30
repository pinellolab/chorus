"""Build per-track background CDFs for the Cherimoya / CATv1 oracle.

Each of the 1,518 CATv1 experiments is one track, producing
``cherimoya_pertrack.npz`` with three CDFs per track (effect, summary,
perbin).  CATv1 output is 1,000 bp at base resolution, so the perbin CDF
captures the bin-level distribution inside the prediction window -- the
same situation as ChromBPNet.

Run in the chorus-cherimoya env:

  mamba run -n chorus-cherimoya python scripts/build_backgrounds_cherimoya.py \\
      --part both --device cuda
  mamba run -n chorus python scripts/build_backgrounds_cherimoya.py --part merge

Validate on a handful of tracks before committing to the full atlas:

  ... --part both --limit 9 --chrombpnet-matched   # the 9 shared experiments


**Sampling matches the published ChromBPNet CDFs exactly.**  The default
configuration reproduces the sample counts in
``lucapinello/chorus-backgrounds/chrombpnet_pertrack.npz`` byte for byte:
``effect_counts=18672`` (9,609 random SNPs + 9,063 DHS-proximal) and
``summary_counts=34004`` (15,000 random + 11,500 cCRE + 3,000 TSS, of
which 29,004 are usable, + 5,000 DHS peak summits).  Reproducing those
counts is the check that the shared variant and region sets really are
shared -- an off-by-one in a seed or a bounds test shows up here as a
different count.

``--no-dhs`` drops the DHS components (giving 9,609 / 29,004) and writes
to ``cherimoya_pertrack.no-dhs.npz``.  That configuration is *not*
comparable to the other oracles and exists only for ablation; the build
configuration is stamped into every NPZ (see ``_build_config``) so a CDF
file can always be traced back to how it was made.

**Sequences are encoded once, not per model.**  The ChromBPNet builder
calls ``get_sequence`` and ``one_hot_encode`` inside its per-model loop.
That is affordable for 42 models; at 1,518 it would dominate wall clock
entirely, since a Cherimoya forward pass over the whole sampled set takes
about two seconds.  Here the full one-hot tensor is built once and parked
on the accelerator, so the per-model cost is pure forward pass.
"""

import argparse
import json
import logging
import os
import random
import sys
import time

import numpy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
os.environ["CHORUS_NO_TIMEOUT"] = "1"

from chorus.oracles.cherimoya_source.catv1_globals import (  # noqa: E402
    CATV1_INPUT_LENGTH,
    CATV1_OUTPUT_LENGTH,
    CATV1_SCORING_WINDOW_BP,
    CATV1_TRIMMING,
    catv1_track_id,
)
from chorus.oracles.cherimoya_source.scoring import (  # noqa: E402
    PSEUDOCOUNT,
    compute_effect,
    expected_counts_profile,
    score_window_sum,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--part",
    choices=["variants", "baselines", "both", "all", "merge",
             "merge-incremental", "merge-shards"],
    default="all",
)
parser.add_argument(
    "--device", default="cuda",
    help="Explicit torch device. Pinned rather than auto-detected: the "
         "Triton kernels and the pure-PyTorch CPU fallback disagree by "
         "~1e-2 on the profile logits, so a silent fallback would make "
         "some CDF rows incomparable with the rest.",
)
parser.add_argument(
    "--gpu", type=int, default=None,
    help="Pin CUDA_VISIBLE_DEVICES to this device index. Present because "
         "`chorus backgrounds build --oracle cherimoya --gpu N` passes it "
         "through, and every other build script accepts it -- without it that "
         "CLI path fails with 'unrecognized arguments'. Leave unset when "
         "sharding, where CUDA_VISIBLE_DEVICES is set per worker instead.",
)
parser.add_argument("--fold", type=int, default=0,
                    help="CATv1 fold. 0 matches ChromBPNet's default and CDFs.")
parser.add_argument("--no-dhs", dest="dhs", action="store_false",
                    help="Drop the DHS-proximal SNPs and DHS peak baselines. "
                         "ON by default, because that is what the published "
                         "ChromBPNet CDFs contain; disabling makes the result "
                         "non-comparable to the other oracles. Ablation only.")
parser.set_defaults(dhs=True)
parser.add_argument("--n-variants", type=int, default=10000)
parser.add_argument("--n-dhs-variants", type=int, default=10000)
parser.add_argument("--n-dhs-peaks", type=int, default=5000)
parser.add_argument("--dhs-path", default=None)
parser.add_argument("--reservoir-size", type=int, default=50000)
parser.add_argument("--n-cdf-points", type=int, default=10000)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--reference", default=os.path.join(REPO_ROOT, "genomes/hg38.fa"))
parser.add_argument("--limit", type=int, default=None,
                    help="Score only the first N tracks (validation runs).")
parser.add_argument("--tracks", default=None,
                    help="Comma-separated ENCODE accessions or ASSAY:ENCSR ids.")
parser.add_argument("--chrombpnet-matched", action="store_true",
                    help="Restrict to the 9 experiments ChromBPNet also covers "
                         "(the cross-oracle comparison set).")
parser.add_argument("--only-missing", action="store_true")
parser.add_argument("--shard", type=int, default=None)
parser.add_argument("--shard-of", type=int, default=None)
parser.add_argument("--sequences-on-cpu", action="store_true",
                    help="Keep the encoded sequences in host memory instead of "
                         "device memory. Slower, but needed if the accelerator "
                         "cannot hold ~2 GB of one-hot.")
args = parser.parse_args()

# Must happen before torch is imported anywhere, which is why it sits here
# rather than inside build().
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

log_dir = os.path.join(REPO_ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)
# force=True is required, not cosmetic. Importing chorus pulls in
# chorus/oracles/chrombpnet.py, which calls logging.basicConfig() at module
# scope; that installs a root handler before this line runs, and
# basicConfig is a silent no-op when the root logger already has handlers.
# Without force=True this configuration is discarded, every message goes to
# the inherited stdout handler instead of the log file, and `conda run`
# buffers stdout until exit -- so an 85-minute build produces no visible
# progress at all and no on-disk record.

# The shard tag is part of the log filename, not just the interim NPZ
# names: with `mode="w"` every concurrent shard would otherwise truncate
# and interleave into one file, leaving a multi-GPU build with no usable
# record of which worker did what. (`build_backgrounds_chrombpnet.py` names
# its log by `--part` alone and has the same problem for sharded runs.)
_log_tag = args.part
if args.shard is not None and args.shard_of is not None:
    _log_tag += f".shard{args.shard}of{args.shard_of}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{log_dir}/bg_cherimoya_{_log_tag}.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

CACHE_DIR = os.path.expanduser("~/.chorus/backgrounds")
os.makedirs(CACHE_DIR, exist_ok=True)

PERBIN_BINS_PER_POSITION = 32
NPZ_STEM = "cherimoya" if args.dhs else "cherimoya-no-dhs"


def _build_config() -> dict:
    """Provenance stamped into every NPZ this script writes.

    Tracing whether the published ChromBPNet CDFs included the DHS sets
    required digging through year-old build logs. Recording it in the file
    removes that archaeology for whoever comes next.
    """
    import cherimoya

    return {
        "oracle": "cherimoya",
        "atlas": "CATv1",
        "cherimoya_version": cherimoya.__version__ if hasattr(cherimoya, "__version__") else "0.2.0",
        "fold": args.fold,
        "device": args.device,
        "dhs_enabled": bool(args.dhs),
        "n_variants": args.n_variants,
        "n_dhs_variants": args.n_dhs_variants if args.dhs else 0,
        "n_dhs_peaks": args.n_dhs_peaks if args.dhs else 0,
        "window_bp": CATV1_SCORING_WINDOW_BP,
        "pseudocount": PSEUDOCOUNT,
        "formula": "log2fc",
        "signed": False,
        "input_length": CATV1_INPUT_LENGTH,
        "output_length": CATV1_OUTPUT_LENGTH,
    }


# ── Reservoir sampler (identical to the other builders) ──────────────

class ReservoirSampler:
    def __init__(self, n_tracks: int, capacity: int = 50_000):
        self.n_tracks = n_tracks
        self.capacity = capacity
        self.data = [[] for _ in range(n_tracks)]
        self.counts = numpy.zeros(n_tracks, dtype=numpy.int64)
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

    def get_sorted(self, track_idx: int) -> numpy.ndarray:
        arr = numpy.array(self.data[track_idx], dtype=numpy.float64)
        arr.sort()
        return arr

    def to_cdf_matrix(self, n_points: int = 10_000) -> numpy.ndarray:
        matrix = numpy.zeros((self.n_tracks, n_points), dtype=numpy.float64)
        target_q = numpy.linspace(0, 1, n_points)
        for i in range(self.n_tracks):
            arr = self.get_sorted(i)
            n = len(arr)
            if n == 0:
                continue
            if n >= n_points:
                idx = numpy.linspace(0, n - 1, n_points, dtype=int)
                matrix[i] = arr[idx]
            else:
                matrix[i] = numpy.interp(target_q, numpy.arange(n) / n, arr)
        return matrix

    def get_counts(self) -> numpy.ndarray:
        return self.counts.copy()


# ── Position sampling: byte-for-byte the ChromBPNet procedure ────────

def _random_snps(ref, n: int) -> list:
    random.seed(42)
    chroms = [f"chr{i}" for i in range(1, 23)]
    per_chrom = n // len(chroms) + 1
    snps = []
    for chrom in chroms:
        chrom_len = ref.get_reference_length(chrom)
        max_pos = min(chrom_len - 5_000_000, 200_000_000)
        for _ in range(per_chrom):
            if len(snps) >= n:
                break
            pos = random.randint(5_000_000, max_pos)
            ref_base = ref.fetch(chrom, pos - 1, pos).upper()
            if ref_base not in "ACGT":
                continue
            snps.append({
                "chrom": chrom, "pos": pos, "ref": ref_base,
                "alt": random.choice([b for b in "ACGT" if b != ref_base]),
            })
    random.shuffle(snps)
    return snps[:n]


def _dhs_snps(ref, n: int) -> list:
    from chorus.utils.annotations import sample_dhs_positions

    dhs_path = args.dhs_path or os.path.join(
        REPO_ROOT, "annotations", "dhs_vocabulary_hg38.txt.gz")
    # Deliberately NOT wrapped in a try/except. The ChromBPNet builder
    # warns and continues when the vocabulary is missing, which silently
    # halves the effect CDF and yields a quietly non-comparable NPZ. The
    # file auto-downloads and is checksummed, so a failure here is real.
    summits = sample_dhs_positions(n, dhs_path=dhs_path, seed=43)
    random.seed(44)
    snps = []
    for chrom, summit in summits:
        pos = summit + random.randint(-150, 150)
        chrom_len = ref.get_reference_length(chrom)
        if pos < 5_000_000 or pos > chrom_len - 5_000_000:
            continue
        ref_base = ref.fetch(chrom, pos - 1, pos).upper()
        if ref_base not in "ACGT":
            continue
        snps.append({
            "chrom": chrom, "pos": pos, "ref": ref_base,
            "alt": random.choice([b for b in "ACGT" if b != ref_base]),
        })
    return snps


def _baseline_positions(ref) -> list:
    from chorus.utils.annotations import (
        get_annotation_manager,
        sample_ccre_positions,
        sample_dhs_positions,
    )

    ccre = sample_ccre_positions(
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
    per_chrom = n_random // len(chroms) + 1
    rand_positions = []
    for chrom in chroms:
        chrom_len = ref.get_reference_length(chrom)
        max_pos = min(chrom_len - 10_000_000, 200_000_000)
        if max_pos <= 10_000_000:
            max_pos = chrom_len - 1_000_000
        for _ in range(per_chrom):
            if len(rand_positions) >= n_random:
                break
            rand_positions.append((chrom, random.randint(10_000_000, max_pos)))

    manager = get_annotation_manager()
    gtf = manager.get_annotation_path("gencode_v48_basic")
    genes = manager._get_genes_df(gtf)
    pc = genes[genes["gene_type"] == "protein_coding"].copy()
    pc["tss"] = pc.apply(lambda r: r["start"] if r["strand"] == "+" else r["end"], axis=1)
    pc = pc[pc["chrom"].isin({f"chr{i}" for i in range(1, 23)})]
    tss_dedup = pc.groupby("gene_name").first().reset_index()
    tss_list = list(zip(tss_dedup["chrom"], tss_dedup["tss"]))
    if len(tss_list) > 3000:
        tss_list = random.Random(111).sample(tss_list, 3000)

    positions = list(rand_positions)
    positions += list(ccre)
    positions += [(c, int(p)) for c, p in tss_list]

    n_dhs = 0
    if args.dhs and args.n_dhs_peaks > 0:
        dhs_path = args.dhs_path or os.path.join(
            REPO_ROOT, "annotations", "dhs_vocabulary_hg38.txt.gz")
        dhs = sample_dhs_positions(args.n_dhs_peaks, dhs_path=dhs_path, seed=567)
        positions.extend(dhs)
        n_dhs = len(dhs)

    random.shuffle(positions)
    logger.info(
        "Baseline positions: %d (random=%d, cCRE=%d, TSS=%d, DHS=%d)",
        len(positions), len(rand_positions), len(ccre), len(tss_list), n_dhs,
    )
    return positions


def get_sequence(ref, chrom, pos):
    """Variant-centred window, or None if unusable. Matches ChromBPNet."""
    half = CATV1_INPUT_LENGTH // 2
    start, end = pos - half, pos + half
    if start < 0 or end > ref.get_reference_length(chrom):
        return None
    seq = ref.fetch(chrom, start, end).upper()
    if len(seq) != CATV1_INPUT_LENGTH or seq.count("N") > CATV1_INPUT_LENGTH * 0.5:
        return None
    return seq


# ── Encoding: once, up front ─────────────────────────────────────────

_CODE = numpy.full(256, 255, dtype=numpy.uint8)
for _i, _b in enumerate("ACGT"):
    _CODE[ord(_b)] = _i


def encode_batch(seqs) -> numpy.ndarray:
    """(N, 4, L) float32 one-hot; ambiguous bases stay all-zero."""
    raw = numpy.frombuffer("".join(seqs).encode("ascii"), dtype=numpy.uint8)
    codes = _CODE[raw].reshape(len(seqs), -1)
    out = numpy.zeros((len(seqs), 4, codes.shape[1]), dtype=numpy.float32)
    rows, cols = numpy.nonzero(codes < 4)
    out[rows, codes[rows, cols], cols] = 1.0
    return out


def build_sequence_tensor(seqs, device, torch):
    """One-hot every sequence once and park it where the model will read it."""
    logger.info("Encoding %d sequences (%.1f GB float32)...",
                len(seqs), len(seqs) * 4 * CATV1_INPUT_LENGTH * 4 / 1e9)
    chunks = []
    for i in range(0, len(seqs), 4096):
        chunk = torch.from_numpy(encode_batch(seqs[i:i + 4096]))
        chunks.append(chunk if args.sequences_on_cpu else chunk.to(device))
    return torch.cat(chunks)


# ── Scoring ──────────────────────────────────────────────────────────

def forward_window_sums(model, X, torch, perbin_idx=None):
    """Return (window_sums, perbin_values) for a stack of one-hot inputs.

    The softmax/expm1/window-sum is done on the accelerator for speed, but
    the first batch of every run is cross-checked against
    ``scoring.expected_counts_profile`` + ``scoring.score_window_sum`` --
    the same helpers ``oracle.predict()`` uses -- so the fast path cannot
    drift from the transform the query side applies.
    """
    centre = CATV1_OUTPUT_LENGTH // 2
    half = CATV1_SCORING_WINDOW_BP // 2
    lo, hi = max(0, centre - half), min(CATV1_OUTPUT_LENGTH, centre + half + 1)

    sums, perbins = [], []
    checked = False
    with torch.no_grad():
        for i in range(0, X.shape[0], args.batch_size):
            batch = X[i:i + args.batch_size]
            if args.sequences_on_cpu:
                batch = batch.to(model_device(model, torch))
            logits, log_counts = model(batch)
            logits = logits.float()[:, 0, :]
            probs = torch.softmax(logits - logits.mean(dim=1, keepdim=True), dim=1)
            counts = torch.expm1(log_counts.float()[:, 0])
            profiles = probs * counts[:, None]

            sums.append(profiles[:, lo:hi].sum(dim=1).cpu().numpy())
            if perbin_idx is not None:
                perbins.append(profiles[:, perbin_idx].cpu().numpy())

            if not checked:
                reference = expected_counts_profile(
                    logits.cpu().numpy()[:, None, :], log_counts.float().cpu().numpy())
                mine = profiles.cpu().numpy()
                numpy.testing.assert_allclose(mine, reference, rtol=1e-4, atol=1e-5)
                assert abs(score_window_sum(reference[0]) - float(sums[0][0])) < 1e-3
                checked = True

    return (numpy.concatenate(sums),
            numpy.concatenate(perbins) if perbins else None)


def model_device(model, torch):
    return next(model.parameters()).device


# ── Track enumeration ────────────────────────────────────────────────

CHROMBPNET_MATCHED = [
    "ENCSR868FGK", "ENCSR291GJU", "ENCSR637XSC", "ENCSR200OML",
    "ENCSR149XIL", "ENCSR477RTP", "ENCSR000EMT", "ENCSR000EOT",
    "ENCSR000EMU",
]


def enumerate_tracks() -> list:
    from chorus.oracles.cherimoya_source.catv1_metadata import get_metadata

    df = get_metadata().tracks_df
    specs = [
        {"assay": row["assay"], "encode_id": row["experiment_accession"],
         "track_id": row["track_id"]}
        for _, row in df.sort_values("track_id").iterrows()
    ]

    if args.chrombpnet_matched:
        wanted = set(CHROMBPNET_MATCHED)
        specs = [s for s in specs if s["encode_id"] in wanted]
    if args.tracks:
        wanted = {t.strip() for t in args.tracks.split(",") if t.strip()}
        specs = [s for s in specs
                 if s["encode_id"] in wanted or s["track_id"] in wanted]
    if args.only_missing:
        path = os.path.join(CACHE_DIR, f"{NPZ_STEM}_pertrack.npz")
        if os.path.exists(path):
            have = set(str(t) for t in numpy.load(path, allow_pickle=False)["track_ids"])
            before = len(specs)
            specs = [s for s in specs if s["track_id"] not in have]
            logger.info("--only-missing: %d/%d remain", len(specs), before)
    if args.shard is not None or args.shard_of is not None:
        if args.shard is None or args.shard_of is None:
            raise SystemExit("--shard and --shard-of must be set together")
        specs = [s for i, s in enumerate(specs) if i % args.shard_of == args.shard]
        logger.info("--shard %d/%d: %d tracks", args.shard, args.shard_of, len(specs))
    if args.limit:
        specs = specs[:args.limit]

    # A silently truncated build reads like a complete one; say so.
    logger.info("Will score %d tracks (fold %d, dhs=%s)",
                len(specs), args.fold, args.dhs)
    return specs


def _interim_suffix() -> str:
    if args.shard is None or args.shard_of is None:
        return ""
    return f".shard{args.shard}of{args.shard_of}"


# ── Main build ───────────────────────────────────────────────────────

def build(do_variants: bool, do_baselines: bool):
    import pysam
    import torch

    from chorus.oracles.cherimoya import CherimoyaOracle

    ref = pysam.FastaFile(args.reference)
    specs = enumerate_tracks()
    n_tracks = len(specs)
    if n_tracks == 0:
        logger.warning("No tracks to score; nothing to do.")
        return

    effect_res = ReservoirSampler(n_tracks, args.reservoir_size) if do_variants else None
    summary_res = ReservoirSampler(n_tracks, args.reservoir_size) if do_baselines else None
    perbin_res = ReservoirSampler(n_tracks, args.reservoir_size) if do_baselines else None

    # ── assemble the sequence sets ──
    ref_seqs, alt_seqs = [], []
    if do_variants:
        snps = _random_snps(ref, args.n_variants)
        logger.info("Generated %d random SNPs", len(snps))
        if args.dhs and args.n_dhs_variants > 0:
            dhs = _dhs_snps(ref, args.n_dhs_variants)
            logger.info("Generated %d DHS SNPs", len(dhs))
            snps = snps + dhs
        logger.info("Total SNPs for effect CDF: %d", len(snps))

        offset = CATV1_INPUT_LENGTH // 2 - 1
        for snp in snps:
            seq = get_sequence(ref, snp["chrom"], snp["pos"])
            if seq is None:
                continue
            ref_seqs.append(seq)
            alt_seqs.append(seq[:offset] + snp["alt"] + seq[offset + 1:])
        logger.info("Usable variant pairs: %d", len(ref_seqs))

    base_seqs = []
    if do_baselines:
        for chrom, pos in _baseline_positions(ref):
            seq = get_sequence(ref, chrom, pos)
            if seq is not None:
                base_seqs.append(seq)
        logger.info("Usable baseline positions: %d", len(base_seqs))

    ref.close()

    # Per-bin sample indices are drawn once per baseline position and
    # reused for every model. ChromBPNet re-draws inside its model loop,
    # so its bin choice varies by model; fixing it here makes the perbin
    # CDFs reproducible and lets the whole baseline set be scored in one
    # batched pass. Statistically equivalent -- these are random bins from
    # the same profile either way.
    perbin_idx = None
    if do_baselines and base_seqs:
        rng_bins = numpy.random.RandomState(999)
        perbin_idx = rng_bins.choice(
            CATV1_OUTPUT_LENGTH, PERBIN_BINS_PER_POSITION, replace=False)

    # ── encode once ──
    device = args.device
    all_seqs = ref_seqs + alt_seqs + base_seqs
    X = build_sequence_tensor(all_seqs, device, torch)
    n_var = len(ref_seqs)
    X_ref = X[:n_var]
    X_alt = X[n_var:2 * n_var]
    X_base = X[2 * n_var:]
    logger.info("Encoded tensor resident on %s: %s", X.device, tuple(X.shape))

    oracle = CherimoyaOracle(use_environment=False, device=device)

    track_ids = [s["track_id"] for s in specs]
    loop_start = time.time()
    for idx, spec in enumerate(specs):
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("Track %d/%d: %s (fold %d)", idx + 1, n_tracks,
                    spec["track_id"], args.fold)
        try:
            oracle.load_pretrained_model(
                assay=spec["assay"], encode_id=spec["encode_id"], fold=args.fold)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", spec["track_id"], str(exc)[:200])
            continue

        model = oracle.model
        try:
            if do_variants and n_var:
                ref_sums, _ = forward_window_sums(model, X_ref, torch)
                alt_sums, _ = forward_window_sums(model, X_alt, torch)
                for r, a in zip(ref_sums, alt_sums):
                    effect_res.add(idx, abs(compute_effect(float(r), float(a))))

            if do_baselines and len(base_seqs):
                base_sums, base_bins = forward_window_sums(
                    model, X_base, torch, perbin_idx=perbin_idx)
                for s in base_sums:
                    summary_res.add(idx, float(s))
                perbin_res.add_batch(idx, base_bins.reshape(-1))
        except Exception as exc:
            logger.warning("Scoring failed for %s: %s", spec["track_id"], str(exc)[:200])
            continue

        logger.info("  done in %.1fs (effect=%d summary=%d perbin=%d)",
                    time.time() - t0,
                    int(effect_res.counts[idx]) if do_variants else 0,
                    int(summary_res.counts[idx]) if do_baselines else 0,
                    int(perbin_res.counts[idx]) if do_baselines else 0)

        # Progress + ETA every 50 tracks. A 1,518-track build otherwise
        # gives no way to tell "slow" from "wedged" until it ends.
        if (idx + 1) % 50 == 0 or idx + 1 == n_tracks:
            done = idx + 1
            elapsed = time.time() - loop_start
            rate = elapsed / done
            logger.info(
                "PROGRESS %d/%d (%.1f%%) | %.1f min elapsed | %.2f s/track | "
                "ETA %.1f min", done, n_tracks, 100 * done / n_tracks,
                elapsed / 60, rate, rate * (n_tracks - done) / 60,
            )

    # ── interim files ──
    suffix = _interim_suffix()
    config = json.dumps(_build_config())
    if do_variants:
        path = os.path.join(CACHE_DIR, f"{NPZ_STEM}_effect_cdfs_interim{suffix}.npz")
        numpy.savez_compressed(
            path,
            track_ids=numpy.array(track_ids, dtype="U"),
            effect_cdfs=effect_res.to_cdf_matrix(args.n_cdf_points).astype(numpy.float32),
            effect_counts=effect_res.get_counts(),
            signed_flags=numpy.zeros(n_tracks, dtype=bool),
            build_config=numpy.array([config]),
        )
        logger.info("Saved %s", path)

    if do_baselines:
        path = os.path.join(CACHE_DIR, f"{NPZ_STEM}_baseline_cdfs_interim{suffix}.npz")
        numpy.savez_compressed(
            path,
            track_ids=numpy.array(track_ids, dtype="U"),
            summary_cdfs=summary_res.to_cdf_matrix(args.n_cdf_points).astype(numpy.float32),
            summary_counts=summary_res.get_counts(),
            perbin_cdfs=perbin_res.to_cdf_matrix(args.n_cdf_points).astype(numpy.float32),
            perbin_counts=perbin_res.get_counts(),
            build_config=numpy.array([config]),
        )
        logger.info("Saved %s", path)


# ── Merge ────────────────────────────────────────────────────────────

def _augment(path, config: str):
    """Re-save an NPZ with the build-config provenance attached."""
    data = dict(numpy.load(str(path), allow_pickle=False))
    data["build_config"] = numpy.array([config])
    numpy.savez_compressed(str(path), **data)


def merge(incremental: bool = False):
    from pathlib import Path

    from chorus.analysis.normalization import PerTrackNormalizer

    eff_path = os.path.join(CACHE_DIR, f"{NPZ_STEM}_effect_cdfs_interim.npz")
    base_path = os.path.join(CACHE_DIR, f"{NPZ_STEM}_baseline_cdfs_interim.npz")
    if not (os.path.exists(eff_path) and os.path.exists(base_path)):
        logger.error("Missing interim files -- run --part both first.")
        return

    eff = numpy.load(eff_path, allow_pickle=False)
    base = numpy.load(base_path, allow_pickle=False)
    ids = list(eff["track_ids"].astype(str))
    assert ids == list(base["track_ids"].astype(str)), \
        "interim effect/baseline track ordering must agree"

    # PerTrackNormalizer keys the filename off the oracle name, so the
    # --no-dhs ablation is written under the canonical name and then
    # renamed out of the way. Both configurations can coexist.
    if incremental:
        path, n_added = PerTrackNormalizer.append_tracks(
            oracle_name="cherimoya",
            new_track_ids=ids,
            new_effect_cdfs=eff["effect_cdfs"],
            new_summary_cdfs=base["summary_cdfs"],
            new_perbin_cdfs=base["perbin_cdfs"],
            new_signed_flags=eff["signed_flags"],
            new_effect_counts=eff["effect_counts"],
            new_summary_counts=base["summary_counts"],
            new_perbin_counts=base["perbin_counts"],
            cache_dir=CACHE_DIR,
        )
        logger.info("Appended %d tracks", n_added)
    else:
        path = PerTrackNormalizer.build_and_save(
            oracle_name="cherimoya",
            track_ids=ids,
            effect_cdfs=eff["effect_cdfs"],
            summary_cdfs=base["summary_cdfs"],
            perbin_cdfs=base["perbin_cdfs"],
            signed_flags=eff["signed_flags"],
            effect_counts=eff["effect_counts"],
            summary_counts=base["summary_counts"],
            perbin_counts=base["perbin_counts"],
            cache_dir=CACHE_DIR,
            n_points=args.n_cdf_points,
        )

    path = Path(path)
    if not args.dhs:
        target = path.with_name("cherimoya_pertrack.no-dhs.npz")
        path.replace(target)
        path = target

    _augment(path, json.dumps(_build_config()))
    logger.info("DONE -- %s (%.1f MB, %d tracks)",
                path, path.stat().st_size / 1e6, len(ids))


def merge_shards():
    import glob
    import re
    from pathlib import Path

    from chorus.analysis.normalization import PerTrackNormalizer

    pattern = os.path.join(CACHE_DIR, f"{NPZ_STEM}_effect_cdfs_interim.shard*of*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.error("No shard files matching %s", pattern)
        return

    shard_re = re.compile(r"shard(\d+)of(\d+)\.npz$")
    shards, total = {}, None
    for f in files:
        m = shard_re.search(f)
        if not m:
            continue
        i, tot = int(m.group(1)), int(m.group(2))
        total = total or tot
        b = f.replace("effect", "baseline")
        if not os.path.exists(b):
            logger.error("Missing baseline shard %s", b)
            return
        shards[i] = (f, b)

    missing = sorted(set(range(total)) - set(shards))
    if missing:
        logger.error("Missing shards %s of %d", missing, total)
        return

    ids, eff_c, sum_c, pb_c, sg, ec, sc, pc = [], [], [], [], [], [], [], []
    for i in range(total):
        e = numpy.load(shards[i][0], allow_pickle=False)
        b = numpy.load(shards[i][1], allow_pickle=False)
        ids.extend(list(e["track_ids"].astype(str)))
        eff_c.append(e["effect_cdfs"]); sum_c.append(b["summary_cdfs"])
        pb_c.append(b["perbin_cdfs"]); sg.append(e["signed_flags"])
        ec.append(e["effect_counts"]); sc.append(b["summary_counts"])
        pc.append(b["perbin_counts"])

    path = Path(PerTrackNormalizer.build_and_save(
        oracle_name="cherimoya",
        track_ids=ids,
        effect_cdfs=numpy.concatenate(eff_c),
        summary_cdfs=numpy.concatenate(sum_c),
        perbin_cdfs=numpy.concatenate(pb_c),
        signed_flags=numpy.concatenate(sg),
        effect_counts=numpy.concatenate(ec),
        summary_counts=numpy.concatenate(sc),
        perbin_counts=numpy.concatenate(pc),
        cache_dir=CACHE_DIR,
        n_points=args.n_cdf_points,
    ))
    if not args.dhs:
        target = path.with_name("cherimoya_pertrack.no-dhs.npz")
        path.replace(target)
        path = target
    _augment(path, json.dumps(_build_config()))
    logger.info("DONE -- merged %d shards, %d tracks -> %s", total, len(ids), path)


if args.part == "variants":
    build(True, False)
elif args.part == "baselines":
    build(False, True)
elif args.part == "merge":
    merge()
elif args.part == "merge-incremental":
    merge(incremental=True)
elif args.part == "merge-shards":
    merge_shards()
else:
    build(True, True)
    if args.shard is not None:
        logger.info("Sharded build complete -- run --part merge-shards to aggregate.")
    elif args.only_missing:
        merge(incremental=True)
    else:
        merge()
