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

from chorus.analysis.background_sampling import abort_if_nothing_loads  # noqa: E402
from chorus.analysis.background_sampling import (
    sampling_block,  # noqa: E402
    ReservoirSampler,
    StagedSamples,
    compute_effect as _shared_compute_effect,
    one_hot_encode,
)
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
parser.add_argument("--gpu", type=int, default=0)
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
parser.add_argument("--n-dhs-variants", type=int, default=10000,
    help="DHS-based SNPs (summit ±150 bp) to add to the effect CDF. 0 to disable.")
parser.add_argument("--n-dhs-peaks", type=int, default=5000,
    help="DHS peak summits to add to the baseline (activity) CDF. 0 to disable.")
parser.add_argument("--dhs-path", type=str, default=None,
    help="Path to dhs_vocabulary_hg38.txt.gz. Defaults to annotations/ in repo root.")
parser.add_argument("--reservoir-size", type=int, default=50000)
parser.add_argument("--perbin-tail-k", type=int, default=43526,
                    help="Exact top/bottom K values kept per track for the perbin "
                         "layer, which cannot be retained whole (2,176,256 offered per track). Derived as "
                         "ceil(200 * N_expected / 10000) so at least 200 of the "
                         "10,000 grid slots are true order statistics; a single fixed "
                         "K silently gives ChromBPNet only 91.")
parser.add_argument("--exact-capacity", type=int, default=4000000,
                    help="Reservoir capacity for the effect and summary layers. Large "
                         "enough to retain every offered value, so their ceilings are "
                         "population maxima rather than draws from a subsample.")
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
parser.add_argument(
    "--cells",
    type=str,
    default=None,
    help="Comma-separated cell types to score (e.g. 'K562,GM12878'). "
    "Filters _enumerate_models() down to a subset. Useful for targeted "
    "rebuilds; pair with --part merge-incremental to stitch into the "
    "existing chrombpnet_pertrack.npz.",
)
parser.add_argument(
    "--only-assays",
    type=str,
    default=None,
    help="Comma-separated assays to keep within --assay (e.g. 'DNASE' or "
    "'DNASE,ATAC'). Applied after --cells. Use to scope a build to "
    "DNase-only K562/GM12878, etc.",
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

# Honour the data-dir mechanism rather than hardcoding $HOME. All eight
# builders had this literal, so a chorus installed with
# CHORUS_DATA_DIR=/data/... still wrote its backgrounds into the home
# directory the data dir exists to avoid. CHORUS_BACKGROUNDS_DIR applies
# the legacy ~/.chorus compatibility itself, per kind.
from chorus.core.globals import CHORUS_BACKGROUNDS_DIR
cache_dir = os.environ.get("CHORUS_BUILD_CACHE_DIR") or str(CHORUS_BACKGROUNDS_DIR)
os.makedirs(cache_dir, exist_ok=True)


# ── Reservoir sampler (same as Borzoi/Enformer) ──────────────────
# ReservoirSampler now comes from chorus.analysis.background_sampling
# (imported above) — see #125. The local copy was proved byte-identical
# before removal; the behaviour is pinned permanently by
# tests/test_background_sampling.py's golden values.


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
    if args.cells:
        wanted = {c.strip() for c in args.cells.split(",") if c.strip()}
        before = len(specs)
        specs = [s for s in specs if s["cell_type"] in wanted]
        logger.info("--cells filter: %s -> %d/%d models", sorted(wanted), len(specs), before)
    if args.only_assays:
        wanted_a = {a.strip() for a in args.only_assays.split(",") if a.strip()}
        before = len(specs)
        specs = [s for s in specs if s["assay"] in wanted_a]
        logger.info("--only-assays filter: %s -> %d/%d models", sorted(wanted_a), len(specs), before)
    return specs


def _interim_suffix() -> str:
    """File suffix when sharded: `.shard<N>of<M>`. Empty when not."""
    if args.shard is None or args.shard_of is None:
        return ""
    return f".shard{args.shard}of{args.shard_of}"


def load_models_and_setup():
    """Load reference, set up GPU, return (oracle, models_to_score, ref)."""
    # An explicit CUDA_VISIBLE_DEVICES wins. This used to assign unconditionally, so
    # `CUDA_VISIBLE_DEVICES=1 python build_...py` silently ran on GPU 0 anyway. Two
    # arms of an ablation launched that way both landed on GPU 0; the first grabbed
    # 78 GB, the second could not allocate a cuBLAS handle, and EVERY position was
    # dropped with "Attempting to perform BLAS operation using StreamExecutor without
    # BLAS support". A fleet rebuild sharded across GPUs by env var would have
    # serialised onto one device the same way.
    if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""):
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

    from chorus.analysis.background_sampling import require_reference_assembly

    ref_path = os.path.join(REPO_ROOT, "genomes/hg38.fa")
    # The oracle whose registry had no organism field at all, and so the one that
    # shipped 33 mm10 models scored against hg38 sequence (#121).
    require_reference_assembly(ref_path, ChromBPNetOracle, label="chrombpnet background")
    ref = pysam.FastaFile(ref_path)

    models_to_score = _enumerate_models(args.assay)

    # Scope preflight, before any model is loaded. --assay defaults to ATAC_DNASE, so a
    # rebuild launched without it enumerates 9 of the 753 shipped tracks, scores all 9
    # perfectly, and writes a background that replaces the whole file. Every other guard
    # passes in that case, because nothing else asks whether the right tracks were
    # attempted.
    try:
        import numpy as _np

        from chorus.analysis.background_sampling import scope_violations
        from chorus.core.globals import CHORUS_BACKGROUNDS_DIR as _BG
        _shipped = _BG / "chrombpnet_pertrack.npz"
        _n_shipped = None
        if _shipped.exists():
            with _np.load(_shipped, allow_pickle=True) as _d:
                _n_shipped = len(_d["track_ids"])
        logger.info("scope preflight: --assay=%s enumerates %d models; shipped "
                    "background has %s tracks", args.assay, len(models_to_score),
                    _n_shipped)
        _probs = scope_violations(len(models_to_score),
                                 label=f"chrombpnet(--assay={args.assay})",
                                 n_shipped=_n_shipped)
        if _probs:
            raise SystemExit("refusing to build:\n  " + "\n  ".join(_probs))
    except SystemExit:
        raise
    except Exception as _exc:                      # never let the preflight itself fail
        logger.warning("scope preflight could not run: %s", _exc)

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
def get_sequence(ref, chrom, pos):
    """NOT migrated to the shared helper, deliberately.

    This takes a **pysam** handle (``ref.fetch`` / ``ref.get_reference_length``)
    while ``background_sampling.get_sequence`` takes a pyfaidx-style object and
    slices it. The two also derive their span differently. Unifying them would
    change which positions are accepted as background samples, so it needs its
    own verified step rather than riding along here. See #125.
    """
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
    folded in (softmax × expm1(counts) — the count head predicts
    log(count + 1)).
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
    if counts.ndim == 2 and counts.shape[1] > 1:
        counts = counts.sum(axis=-1, keepdims=True)  # (B, 2) → (B, 1)
    if probabilities.ndim == 2:
        probabilities = probabilities[..., None]      # (B, L) → (B, L, 1)

    return profiles_from_heads(probabilities, counts)


def profiles_from_heads(probabilities, counts):
    """Heads -> expected-count profiles, split out so it can be tested without TF.

    It was inline in ``predict_profiles_batch`` above, which needs a live TensorFlow model,
    so the only way to check it against the oracle was to grep both files for matching
    source text. That is how ``exp`` vs ``expm1`` survived four call sites. Now it is a
    function, and ``tests/test_count_head_copies_agree.py`` feeds it the same inputs as the
    oracle and compares the numbers (#125).

    The arithmetic itself is ``chorus.core.count_head``: a softmax taken JOINTLY over all
    strands so they together sum to the predicted total, scaled by ``exp(C) - n_strands``.
    Both properties are load-bearing and both were once wrong here:

    * this previously summed the two strands' LOGITS before one softmax, which is a
      geometric-mean-like blend corresponding to no observable the oracle emits; its 501 bp
      window sum drifted 0.98-1.30x versus the per-strand values across five test loci, i.e.
      sequence-dependently, so the mismatch could not be undone by rescaling;
    * ``expm1`` on a two-track CHIP target leaves exactly one read of inflation.

    A CDF is only meaningful if it was built from the quantity ``predict()`` returns, which
    is why this shares an implementation with the oracle rather than mirroring it.
    """
    from chorus.core.count_head import expected_counts_profile

    n_strands = probabilities.shape[-1]
    return expected_counts_profile(probabilities, counts, n_tracks=n_strands)


def score_window_sum(profile):
    """NOT migrated to the shared helper, deliberately.

    This window is ``2 * (WINDOW_BP // 2) + 1`` bins — ODD and inclusive of the
    centre — whereas ``background_sampling.score_window_sum`` takes
    ``window_bp // resolution`` bins. For WINDOW_BP=1000 that is 1001 vs 1000,
    so swapping them would shift every ChromBPNet activity value and move the
    shipped summary CDFs. Reconciling the off-by-one is its own change. See #125.
    """
    center = OUTPUT_LENGTH // 2
    hw = WINDOW_BP // 2
    ws = max(0, center - hw)
    we = min(OUTPUT_LENGTH, center + hw + 1)
    return float(np.sum(profile[ws:we]))


def compute_effect(ref_val, alt_val):
    """Thin wrapper so the module keeps its 2-arg call signature."""
    return _shared_compute_effect(ref_val, alt_val, pseudocount=PSEUDOCOUNT)


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

    effect_reservoir = ReservoirSampler(n_tracks, capacity=args.exact_capacity) if do_variants else None
    summary_reservoir = ReservoirSampler(n_tracks, capacity=args.exact_capacity) if do_baselines else None
    perbin_reservoir = ReservoirSampler(n_tracks, capacity=args.reservoir_size, tail_k=args.perbin_tail_k) if do_baselines else None

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
        logger.info("Generated %d random SNPs", len(snps))

    # ── DHS-based SNPs (for effect CDF) ──
    dhs_snps = []
    if do_variants and args.n_dhs_variants > 0:
        try:
            from chorus.utils.annotations import sample_dhs_positions
            dhs_path = args.dhs_path or os.path.join(REPO_ROOT, "annotations",
                                                      "dhs_vocabulary_hg38.txt.gz")
            dhs_summit_positions = sample_dhs_positions(
                args.n_dhs_variants, dhs_path=dhs_path, seed=43,
            )
            random.seed(44)
            for chrom, summit in dhs_summit_positions:
                offset = random.randint(-150, 150)
                pos = summit + offset
                chrom_len = ref.get_reference_length(chrom)
                if pos < 5_000_000 or pos > chrom_len - 5_000_000:
                    continue
                ref_base = ref.fetch(chrom, pos - 1, pos).upper()
                if ref_base not in "ACGT":
                    continue
                dhs_snps.append({"chrom": chrom, "pos": pos, "ref": ref_base,
                                 "alt": random.choice([b for b in "ACGT" if b != ref_base])})
            logger.info("Generated %d DHS SNPs", len(dhs_snps))
        except FileNotFoundError as exc:
            logger.warning("DHS vocabulary not found — skipping DHS variants: %s", exc)

    all_snps = snps + dhs_snps
    if do_variants:
        logger.info("Total SNPs for effect CDF: %d (random=%d, DHS=%d)",
                    len(all_snps), len(snps), len(dhs_snps))

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

        # ── DHS peak positions (for activity CDF) ──
        dhs_baseline = []
        if args.n_dhs_peaks > 0:
            try:
                from chorus.utils.annotations import sample_dhs_positions
                dhs_path = args.dhs_path or os.path.join(REPO_ROOT, "annotations",
                                                          "dhs_vocabulary_hg38.txt.gz")
                dhs_baseline = sample_dhs_positions(
                    args.n_dhs_peaks, dhs_path=dhs_path, seed=567,
                )
                baseline_positions.extend(dhs_baseline)
            except FileNotFoundError as exc:
                logger.warning("DHS vocabulary not found — skipping DHS baselines: %s", exc)

        random.shuffle(baseline_positions)
        logger.info("Total baseline positions: %d (random=%d, cCRE=%d, TSS=%d, DHS=%d)",
                    len(baseline_positions), len(rand_positions), len(ccre_positions),
                    len(tss_list), len(dhs_baseline))

    # Iterate over models
    _n_attempted = _n_loaded = 0
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
            _n_attempted += 1
            abort_if_nothing_loads(_n_attempted, _n_loaded, label="chrombpnet.load")
            continue
        _n_attempted += 1
        _n_loaded += 1

        model = oracle.model

        # ── Variant scoring ──
        if do_variants and all_snps:
            t0 = time.time()
            ref_seqs, alt_seqs = [], []
            for snp in all_snps:
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
                # Whole batch, or none of it (#123). A throw part-way through the
                # strand loop used to commit the variants already scored, so this
                # model's count drifted from its neighbours' by however far the
                # batch got.
                staged = StagedSamples()
                try:
                    ref_profiles = predict_profiles_batch(model, ref_batch)
                    alt_profiles = predict_profiles_batch(model, alt_batch)
                    # profiles are (B, L, n_strands); score each strand and
                    # pool them into this model's single CDF row, because the
                    # normalizer maps both `…:+` and `…:-` onto that one row.
                    for rp, ap in zip(ref_profiles, alt_profiles):
                        for strand in range(rp.shape[-1]):
                            ref_val = score_window_sum(rp[:, strand])
                            alt_val = score_window_sum(ap[:, strand])
                            score = abs(compute_effect(ref_val, alt_val))
                            staged.add(model_idx, score)
                except Exception as exc:
                    logger.warning("Variant batch failed: %s", str(exc)[:100])
                else:
                    staged.commit(effect_reservoir)

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
                # The worse of the two chrombpnet sites: it writes to TWO
                # reservoirs inside one loop, so a throw between the summary add
                # and the perbin add left a summary sample with no matching perbin
                # sample for the same position (#123). Staging makes the pair
                # atomic as well as making the batch atomic.
                #
                # rng_bins is still drawn inside the try, so a failed batch
                # consumes randomness a successful one would not. That is
                # deliberate and unchanged: moving the draw would alter which bins
                # every later position samples, shifting the shipped perbin CDF for
                # a reason unrelated to this fix.
                staged = StagedSamples()
                try:
                    profiles = predict_profiles_batch(model, batch)
                    for prof in profiles:
                        # prof is (L, n_strands). One bin draw per position,
                        # reused across strands, so rng_bins is consumed at
                        # the same rate as for the single-strand models.
                        bin_sample = rng_bins.choice(OUTPUT_LENGTH, PERBIN_BINS_PER_POSITION, replace=False)
                        for strand in range(prof.shape[-1]):
                            p = prof[:, strand]
                            # Summary: window sum
                            signal = score_window_sum(p)
                            staged.add(model_idx, signal, reservoir=0)
                            # Perbin: random bins from full output
                            staged.add_batch(model_idx, p[bin_sample].astype(np.float64),
                                             reservoir=1)
                except Exception as exc:
                    logger.warning("Baseline batch failed: %s", str(exc)[:100])
                else:
                    staged.commit(summary_reservoir, perbin_reservoir)

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
        np.savez_compressed(
            interim_path,
            track_ids=np.array(track_ids, dtype='U'),
            effect_cdfs=effect_matrix.astype(np.float32),
            effect_counts=effect_reservoir.get_counts(),
            effect_retained=effect_reservoir.retained_counts(),
            signed_flags=signed_flags,
        )
        logger.info("Saved effect interim: %s", interim_path)

    if do_baselines:
        summary_matrix = summary_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
        perbin_matrix = perbin_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)
        interim_path = os.path.join(cache_dir, f"chrombpnet_baseline_cdfs_interim{suffix}.npz")
        np.savez_compressed(
            interim_path,
            track_ids=np.array(track_ids, dtype='U'),
            summary_cdfs=summary_matrix.astype(np.float32),
            summary_counts=summary_reservoir.get_counts(),
            summary_retained=summary_reservoir.retained_counts(),
            perbin_cdfs=perbin_matrix.astype(np.float32),
            perbin_counts=perbin_reservoir.get_counts(),
            perbin_retained=perbin_reservoir.retained_counts(),
        )
        logger.info("Saved baseline interim: %s", interim_path)


def merge_to_final():
    from chorus.analysis.normalization import PerTrackNormalizer

    effect_path = os.path.join(cache_dir, "chrombpnet_effect_cdfs_interim.npz")
    baseline_path = os.path.join(cache_dir, "chrombpnet_baseline_cdfs_interim.npz")

    if not os.path.exists(effect_path) or not os.path.exists(baseline_path):
        logger.error("Missing interim files")
        raise SystemExit(1)  # A missing interim is a FAILED merge, not a no-op. Returning here exited 0,
        # so a driver keying off exit codes recorded "rc=0" for a step that wrote
        # nothing -- the same report-success-after-failure shape as the all-zero
        # interim and the guard nobody wired up.

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
        sampling=sampling_block(effect_data, baseline_data, tail_k={"perbin": args.perbin_tail_k}),
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
        raise SystemExit(1)  # A missing interim is a FAILED merge, not a no-op. Returning here exited 0,
        # so a driver keying off exit codes recorded "rc=0" for a step that wrote
        # nothing -- the same report-success-after-failure shape as the all-zero
        # interim and the guard nobody wired up.

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
