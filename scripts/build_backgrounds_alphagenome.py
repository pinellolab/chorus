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


def _shard_suffix():
    """`.shard<K>of<N>` when position-sharded, else empty."""
    if args.shard is None or args.shard_of is None:
        return ""
    return f".shard{args.shard}of{args.shard_of}"

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
parser.add_argument("--effect-regions", choices=["gene-anchored", "ccre"],
                    default="gene-anchored",
                    help="Reference population for the EFFECT null. 'ccre' samples "
                         "inside ENCODE SCREEN cCREs, which is the matched class for "
                         "peak assays (accessibility / histone ChIP / TF ChIP); those "
                         "layers saturate against the gene-anchored mixture because "
                         "most of its positions are not in a peak. Does not affect "
                         "the baseline/summary path, which already uses cCREs.")
parser.add_argument("--shard", type=int, default=None,
                    help="0-indexed POSITION shard. With --shard-of N this process "
                         "scores only positions where i %% N == shard, and writes its "
                         "raw reservoir samples to a .shard<K>of<N> interim. These "
                         "oracles emit every track from one forward pass, so sharding "
                         "by TRACK (as the chrombpnet builder does) would save no GPU "
                         "time -- each shard would still run every pass.")
parser.add_argument("--shard-of", type=int, default=None,
                    help="Total number of position shards. Required with --shard. "
                         "Collect all shards, then run --part merge-shards, which "
                         "unions the raw samples and builds the CDF exactly once "
                         "(pooling the shards' CDF grids would only approximate it).")
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

# HISTONE_PATTERNS used to be duplicated here, a fifth copy of a list that
# already lived in chorus/analysis/scorers.py. It is gone: classify_chip_layer is
# now imported from there, and for AlphaGenome it never needs the patterns at all
# because the CHIP_HISTONE/CHIP_TF identifier prefix is authoritative. See #144.


# ── Reservoir sampler ────────────────────────────────────────────
# ReservoirSampler used to be defined HERE, a local copy of the shared class. It was
# the last un-migrated one of the eight (#125), and keeping it cost real time: adding
# to_flat_samples to the shared class left eight position shards running for fifty
# GPU minutes before every one died with AttributeError at the write step, because
# this file's copy did not have the method. Deleted rather than kept in sync.
#
# The two differences the copy carried are both preserved:
#   * default capacity 20,000 vs the shared 50,000 -- moot here, since all three
#     call sites pass capacity=args.reservoir_size explicitly;
#   * a hand-vectorised add_batch, needed for the baseline pass's per-variant
#     fan-out. That is now the shared implementation, with the plain loop kept as
#     _add_batch_reference so the equivalence test still has something to compare to.
from chorus.analysis.background_sampling import (  # noqa: E402
    ReservoirSampler,
    StagedSamples,
    centered_bin_span,
    report_sampling_uniformity,
)
from chorus.analysis.scorers import canonical_layer, classify_chip_layer  # noqa: E402
from chorus.utils.annotations import (  # noqa: E402
    build_transcript_exon_index,
    exon_bins_for_gene,
    genes_with_tss_in_window,
    load_chrom_sizes,
    sample_ccre_anchored_positions,
    sample_gene_anchored_positions,
)


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
            # `aid`, not `desc`. This one argument IS #122: AlphaGenome's
            # description reads "CHIP:<cell type>" and carries no mark name, so
            # classifying from it made 0 of 2,733 CHIP tracks histone and built
            # every one at 501 bp — while the query path read the identifier and
            # scored 1,075 of them at 2001 bp against that 501 bp null.
            layer = classify_chip_layer(aid, desc)
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
    """Central scoring window slice.

    Delegates to the shared definition (#144, instance 2). Byte-identical to the
    arithmetic that used to live here. Note this oracle spans both regimes: at
    resolution 1 it returns exactly ``window`` bins, while its 128 bp CHIP tracks
    get 3 bins at window=501 and 15 at 2001 — pinned by
    tests/test_window_span_parity.py.
    """
    return centered_bin_span(n_bins, track['window'], track['resolution'])


# ── Exon-precise RNA sampling ──
# load_exon_index()/exon_bin_indices() lived here and merged exons across
# EVERY protein-coding gene on the chromosome, discarding gene identity.
# Replaced by chorus.utils.annotations.build_transcript_exon_index +
# exon_bins_for_gene, which keep genes separate and are built from the
# query's own get_gene_exons() so the masks cannot drift (#144 inst. 3).




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
    # AlphaGenome selects genes by TSS-in-window, then unions the exons of ONLY
    # those transcripts (gene_mask_extractor.py:326, 357-371). Protein-coding only
    # is a DELIBERATE divergence: AlphaGenome applies no gene-type filter, but
    # chorus's query does (variant_report.py:825), and a null over lncRNAs and
    # pseudogenes would be a different population from the numerator. Recorded in
    # provenance rather than left implicit.
    gene_exon_index = build_transcript_exon_index()

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

    # Generate SNPs, anchored on gene structure rather than uniformly at random.
    #
    # This loop used to be `random.randint(5_000_000, max_pos)`, which put the
    # median sampled position 102,333 bp from the nearest TSS and only 1.4 % of
    # them within 1 kb. CAGE is a localised peak at a TSS and RNA is scored over
    # exons, so almost every sample carried no signal for those layers: the null
    # collapsed toward zero and any real effect read >= 99th percentile.
    # AlphaGenome RNA's effect null tops out at 0.0417 — anything >= 0.05
    # saturates at exactly 1.0000, which no floor can fix (#83).
    #
    # The gene-anchored set moves the median to 9,430 bp and puts 21.3 % within
    # 1 kb of a TSS and 37.4 % within 100 bp of a splice junction. 15 % stays
    # uniform on purpose, to keep the null's lower body populated — without it,
    # small real effects would get artificially LOW percentiles.
    random.seed(42)
    # Which reference population the effect null is drawn from. Peak layers
    # (accessibility, histone ChIP, TF ChIP) saturate against the gene-anchored
    # mixture because most of its positions are not inside a peak -- see
    # EFFECT_REGION_SETS in chorus/utils/annotations.py for the measurements.
    _sizes = load_chrom_sizes(os.path.join(REPO_ROOT, 'genomes/hg38.fa.fai'))
    if args.effect_regions == 'ccre':
        sampled = sample_ccre_anchored_positions(
            args.n_variants, chrom_sizes=_sizes, seed=42,
        )
    else:
        sampled = sample_gene_anchored_positions(
            args.n_variants, chrom_sizes=_sizes, seed=42,
        )
    snps = []
    strata_counts = defaultdict(int)
    for chrom, pos, stratum in sampled:
        ref_base = ref.fetch(chrom, pos - 1, pos).upper()
        if ref_base not in "ACGT":
            continue  # N or soft-masked; the stratum tally records the shortfall
        snps.append({
            "chrom": chrom, "pos": pos, "ref": ref_base,
            "alt": random.choice([b for b in "ACGT" if b != ref_base]),
            "stratum": stratum,
        })
        strata_counts[stratum] += 1
    # The region set is named in the message, not hardcoded, because there are now
    # two and this line IS the provenance that scripts/stamp_background_provenance.py
    # reads back. Saying "gene-anchored" while sampling cCREs would stamp a lie into
    # every rebuilt NPZ.
    logger.info("Generated %d %s SNPs from %d sampled positions: %s",
                len(snps), args.effect_regions, len(sampled), dict(strata_counts))

    # Why a position was dropped, reported at the end. A silent drop is how
    # enformer shipped effect_counts spanning 9600-9606 (#123).
    # Position sharding. Applied AFTER the SNP list is built so every shard agrees
    # on the same seeded position set and simply takes a stride through it -- the
    # union over shards is then exactly the unsharded list, which is what makes
    # tests/test_position_sharding.py's equality property hold.
    if args.shard is not None or args.shard_of is not None:
        if args.shard is None or args.shard_of is None:
            raise SystemExit("--shard and --shard-of must be set together")
        if not (0 <= args.shard < args.shard_of):
            raise SystemExit(f"--shard ({args.shard}) must be in [0, {args.shard_of})")
        before = len(snps)
        snps = snps[args.shard::args.shard_of]
        logger.info("--shard %d/%d: scoring %d of %d positions on this worker",
                    args.shard, args.shard_of, len(snps), before)

    drop_reasons = defaultdict(int)
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

        # Genes for the RNA fan-out, resolved ONCE per position rather than per
        # track: the mask depends only on the window, and there are 667 RNA
        # tracks against ~29 genes at a locus like SORT1.
        genes_here = genes_with_tss_in_window(
            gene_exon_index, chrom, pred_start, pred_end)

        # Stage, then commit only if every track scored. This try used to wrap the
        # per-track loop too, so a mid-loop failure credited earlier tracks and not
        # later ones — the defect visible in enformer's 9600-9606 counts (#123).
        staged = StagedSamples()
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
                        # One sample per (GENE, track), matching what the query
                        # emits: variant_report loops over every PC gene near the
                        # variant and reports an RNA row for each. Pooling all
                        # exons in the window instead — which is what this did —
                        # aggregated 128,663 bins against the query's per-gene
                        # median of 4,123, a 31x mismatch (#144 instance 3).
                        #
                        # The mask comes from build_transcript_exon_index(), which is
                        # built from the query's own get_gene_exons(), so the two
                        # cannot drift; parity is asserted in
                        # tests/test_gene_exon_index.py.
                        for _gname, spans in genes_here:
                            eb = exon_bins_for_gene(
                                spans, pred_start, pred_end, n_bins, res,
                            )
                            if len(eb) == 0:
                                continue
                            # mean over the mask, in bins — the same denominator
                            # the query now uses (#149)
                            ref_v = float(np.mean(ref_vals[eb]))
                            alt_v = float(np.mean(alt_vals[eb]))
                            score = compute_effect(
                                ref_v, alt_v, t['formula'], t['pseudocount'],
                            )
                            if not t['signed']:
                                score = abs(score)
                            staged.add(t_i, score)
                        continue

                    ws, we = get_window_slice(t, n_bins)
                    ref_v = float(np.sum(ref_vals[ws:we]))
                    alt_v = float(np.sum(alt_vals[ws:we]))

                    score = compute_effect(ref_v, alt_v, t['formula'], t['pseudocount'])
                    if not t['signed']:
                        score = abs(score)
                    staged.add(t_i, score)
        except Exception as exc:
            drop_reasons[type(exc).__name__] += 1
            logger.warning("Dropped variant %d entirely: %s: %s",
                           i, type(exc).__name__, str(exc)[:200])
        else:
            staged.commit(effect_reservoir)

    report_sampling_uniformity(effect_reservoir, drop_reasons, "effect", logger)
    elapsed_v = time.time() - t0
    logger.info("Variants done in %.1f hrs: %s samples",
                elapsed_v / 3600, f"{effect_reservoir.total_samples():,}")

    track_ids = [t['assay_id'] for t in track_info]
    signed_flags = np.array([t['signed'] for t in track_info], dtype=bool)
    effect_matrix = effect_reservoir.to_cdf_matrix(n_points=args.n_cdf_points)

    # Per-row layer, from the same field the builder uses to choose this
    # track's window. Load-bearing twice over: scripts/merge_effect_shards.py
    # composes peak-layer rows from a cCRE-anchored build and the rest from a
    # gene-anchored one, and #124 asked for a per-row layer so a background's
    # rows can be identified without re-deriving them from opaque ids.
    layers_per_row = np.array(
        [canonical_layer(t['layer']) for t in track_info], dtype='U')
    _suffix = _shard_suffix()
    interim_path = os.path.join(
        cache_dir, f"alphagenome_effect_cdfs_interim{_suffix}.npz")
    if _suffix:
        # A position shard holds a PARTIAL reservoir for every track, so it must
        # ship raw samples: the CDF is built once, from the union, by
        # --part merge-shards. Writing a CDF per shard and pooling the grids would
        # only approximate the unsharded result.
        np.savez_compressed(
            interim_path,
            track_ids=np.array(track_ids, dtype='U'),
            signed_flags=signed_flags,
            layers_per_row=layers_per_row,
            **effect_reservoir.to_flat_samples(),
        )
        logger.info("Saved RAW shard samples: %s (%.1f MB)", interim_path,
                    os.path.getsize(interim_path) / (1024 * 1024))
    else:
        np.savez_compressed(
            interim_path,
            track_ids=np.array(track_ids, dtype='U'),
            effect_cdfs=effect_matrix.astype(np.float32),
            effect_counts=effect_reservoir.get_counts(),
            signed_flags=signed_flags,
            layers_per_row=layers_per_row,
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
    # AlphaGenome selects genes by TSS-in-window, then unions the exons of ONLY
    # those transcripts (gene_mask_extractor.py:326, 357-371). Protein-coding only
    # is a DELIBERATE divergence: AlphaGenome applies no gene-type filter, but
    # chorus's query does (variant_report.py:825), and a null over lncRNAs and
    # pseudogenes would be a different population from the numerator. Recorded in
    # provenance rather than left implicit.
    gene_exon_index = build_transcript_exon_index()

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

    # Why a position was dropped, reported at the end. A silent drop is how
    # enformer shipped effect_counts spanning 9600-9606 (#123).
    drop_reasons = defaultdict(int)
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

        # Genes for the RNA fan-out, resolved once per position (the mask depends
        # only on the window, and there are 667 RNA tracks against ~29 genes).
        genes_here = genes_with_tss_in_window(
            gene_exon_index, chrom, pred_start, pred_end)

        # Staged for the same reason (#123). Slot 0 is summary, slot 1 perbin;
        # both commit or neither does.
        staged = StagedSamples()
        try:
            output = predict_sequence(model, seq, output_types_needed)

            # Cache per-position auxiliary arrays so we don't recompute
            # them per track (this is the optimization that turns the
            # baseline build from ~36s/position to ~5s/position).
            exon_bins_cache = {}     # (resolution, gene) -> exon bin indices
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
                        # One summary sample per (GENE, track), matching the query,
                        # which emits an RNA row per gene near the variant. Pooling
                        # every exon in the window aggregated 128,663 bins against
                        # the query's per-gene median of 4,123 — a 31x mismatch
                        # (#144 instance 3).
                        for gname, spans in genes_here:
                            ck = (res, gname)
                            if ck not in exon_bins_cache:
                                exon_bins_cache[ck] = exon_bins_for_gene(
                                    spans, pred_start, pred_end, n_bins, res,
                                )
                            eb = exon_bins_cache[ck]
                            if len(eb) == 0:
                                continue
                            staged.add(t_i, float(np.mean(vals[eb])), reservoir=0)

                        # Perbin stays POOLED across genes on purpose: it feeds the
                        # IGV browser's per-bin colour scale, which is a
                        # genome-wide "how big is this bin" distribution and has no
                        # gene scoping. Only the summary/effect statistics are
                        # compared against a gene-scoped numerator.
                        pooled_key = ('rna_pooled', res)
                        if pooled_key not in exon_bins_cache:
                            allb = set()
                            for _gname2, spans in genes_here:
                                allb.update(exon_bins_for_gene(
                                    spans, pred_start, pred_end, n_bins, res,
                                ).tolist())
                            exon_bins_cache[pooled_key] = np.array(
                                sorted(allb), dtype=np.int64,
                            )
                        pooled = exon_bins_cache[pooled_key]
                        if len(pooled) == 0:
                            continue
                        if len(pooled) > PERBIN_BINS_PER_POSITION:
                            sk = ('rna_subsample', res)
                            if sk not in random_bins_cache:
                                random_bins_cache[sk] = rng_bins.choice(
                                    pooled, PERBIN_BINS_PER_POSITION, replace=False,
                                )
                            ebs = random_bins_cache[sk]
                        else:
                            ebs = pooled
                        staged.add_batch(t_i, vals[ebs], reservoir=1)
                    else:
                        # Summary: window-sum (skip CAGE at cCREs)
                        if not (is_cage and pos_type == 'ccre'):
                            wkey = (t['window'], res, n_bins)
                            if wkey not in window_slice_cache:
                                window_slice_cache[wkey] = get_window_slice(t, n_bins)
                            ws, we = window_slice_cache[wkey]
                            signal = float(np.sum(vals[ws:we]))
                            staged.add(t_i, signal, reservoir=0)

                        # Perbin: random bins from full output
                        staged.add_batch(t_i, vals[rand_sample], reservoir=1)
        except Exception as exc:
            drop_reasons[type(exc).__name__] += 1
            logger.warning("Dropped %s:%d entirely: %s: %s",
                           chrom, pos, type(exc).__name__, str(exc)[:200])
        else:
            staged.commit(summary_reservoir, perbin_reservoir)

    report_sampling_uniformity(summary_reservoir, drop_reasons, "summary", logger)
    report_sampling_uniformity(perbin_reservoir, drop_reasons, "perbin", logger)
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
