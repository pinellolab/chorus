"""Regenerate the multi-oracle validation example for SORT1 rs12740374.

The report combines three oracles that look at the same variant from very
different angles:

    ChromBPNet      — chromatin accessibility (DNase), HepG2
    Cherimoya       — chromatin accessibility (DNase), HepG2, CATv1/ENCSR149XIL
    LegNet          — promoter activity (LentiMPRA), HepG2
    AlphaGenome     — generalist: ChIP, histone marks, CAGE, etc.

ChromBPNet and Cherimoya answer the *same* question — HepG2 DNase accessibility —
which is the point of including both. Two independently trained models on one
variant is a stronger statement than one model, and because they share a 2,114 bp
input window and base-pair-resolution output, the rows and IGV tracks are directly
comparable rather than merely adjacent.

Each oracle runs inside its own conda env:

    mamba run -n chorus-chrombpnet  python scripts/regenerate_multioracle.py --oracle chrombpnet
    mamba run -n chorus-cherimoya   python scripts/regenerate_multioracle.py --oracle cherimoya
    mamba run -n chorus-legnet      python scripts/regenerate_multioracle.py --oracle legnet
    mamba run -n chorus-alphagenome python scripts/regenerate_multioracle.py --oracle alphagenome

Then the consolidator — which has no GPU requirement and runs in any env:

    mamba run -n chorus python scripts/regenerate_multioracle.py --consolidate

produces a single ``rs12740374_SORT1_multioracle_report.html`` along with
a consolidated ``example_output.md``/``.json``.

The per-oracle JSON files are written to
``examples/walkthroughs/validation/SORT1_rs12740374_multioracle/`` and can
be re-consolidated at any time — e.g. after refreshing a single oracle.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = os.path.join(
    REPO_ROOT,
    "examples/walkthroughs/validation/SORT1_rs12740374_multioracle",
)

# Reference genome (shared by every oracle here).
GENOME_REF = os.path.join(REPO_ROOT, "genomes/hg38.fa")

# ---------------------------------------------------------------------------
# Variant and request description (shared across all oracles)
# ---------------------------------------------------------------------------

VARIANT = {
    "chrom": "chr1",
    "position": 109274968,
    "ref": "G",
    "alt": "T",
    "id": "rs12740374",
    "gene": "SORT1",
}

USER_PROMPT = (
    "Validate rs12740374 (the classic SORT1 LDL-cholesterol causal variant) "
    "by scoring it with three independent deep-learning oracles: ChromBPNet "
    "for chromatin accessibility, LegNet for MPRA promoter activity, and "
    "AlphaGenome as a generalist model covering ChIP, histones and CAGE. "
    "A new user should be able to see at a glance whether the three oracles "
    "agree on direction, and which assay/cell type drove each call."
)

def get_max_output_size():
    """Returns the maximum output size across all possible models."""
    from chorus.oracles.chrombpnet import ChromBPNetOracle
    from chorus.oracles.legnet import LegNetOracle
    from chorus.oracles.alphagenome import AlphaGenomeOracle

    sizes = []
    sizes.append(ChromBPNetOracle().output_size)
    sizes.append(LegNetOracle().sequence_length)    # to match output size
    sizes.append(AlphaGenomeOracle().output_size)

    if not sizes:
        raise RuntimeError("Could not determine any oracle output_size")

    return max(sizes)


# ---------------------------------------------------------------------------
# Per-oracle runners
# ---------------------------------------------------------------------------

def _build_variant_report(oracle, oracle_name: str, assay_ids=None, region=None):
    """Score the SORT1 variant with the given oracle and return the report.

    ``region`` overrides the default ±(max_output_size/2) window. Short-input,
    element-level oracles (e.g. LegNet, 200 bp) must be scored on their own
    native window centred on the variant; a 1 Mb region would tile them into
    tens of thousands of windows and average the single-variant effect away.
    Long-context oracles (ChromBPNet, AlphaGenome) center their scorer on the
    variant and are unaffected by the wide default.
    """
    from chorus.analysis.variant_report import build_variant_report
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.normalization import get_normalizer

    normalizer = None
    try:
        normalizer = get_normalizer(oracle_name=oracle_name)
    except Exception as exc:
        logger.warning("No normalizer for %s: %s — percentile columns absent.",
                       oracle_name, exc)

    logger.info("Predicting variant effect with %s ...", oracle_name)
    # Provide a small genomic region centred on the variant. Most oracles
    # only look ±half-window-size around the variant position; passing a 2bp
    # region here keeps the API contract satisfied without wasting compute.
    if region is not None:
        region_str = region
    else:
        max_window = get_max_output_size()
        half_window = max_window // 2

        start = VARIANT["position"] - half_window
        end = VARIANT["position"] + half_window

        region_str = f"{VARIANT['chrom']}:{start}-{end}"
    position_str = f"{VARIANT['chrom']}:{VARIANT['position']}"
    # position_str = f"{VARIANT['chrom']}:{VARIANT['position']}"
    # region_str = f"{VARIANT['chrom']}:{VARIANT['position']}-{VARIANT['position'] + 1}"
    # Oracles use different attribute names for their single-track id:
    # LegNetOracle exposes ``assay_id`` (e.g. "LentiMPRA:HepG2"); ChromBPNet
    # stores ``assay`` + ``cell_type`` separately. Build the per-oracle
    # default so either shape works.
    if assay_ids is None:
        if hasattr(oracle, "assay_id") and oracle.assay_id:
            assay_ids = [oracle.assay_id]
        elif hasattr(oracle, "assay") and hasattr(oracle, "cell_type"):
            assay_ids = [f"{oracle.assay}:{oracle.cell_type}"]
        else:
            raise RuntimeError(f"Oracle {oracle_name} has no resolvable assay_id")
    result = oracle.predict_variant_effect(
        genomic_region=region_str,
        variant_position=position_str,
        alleles=[VARIANT["ref"], VARIANT["alt"]],
        assay_ids=assay_ids,
        genome=GENOME_REF,
    )
    ar = AnalysisRequest(
        user_prompt=USER_PROMPT,
        tool_name="analyze_variant_multilayer",
        oracle_name=oracle_name,
        normalizer_name="chorus per-track v1" if normalizer else "(none)",
        tracks_requested=(
            f"{len(assay_ids)} tracks" if assay_ids else "all oracle tracks"
        ),
    )
    logger.info("Building variant report ...")
    return build_variant_report(
        result, oracle_name=oracle_name, gene_name=VARIANT["gene"],
        normalizer=normalizer, analysis_request=ar,
    )


def _save_oracle_artefacts(report, oracle_name: str):
    """Persist the per-oracle run to disk in three forms:

    * ``<oracle>_variant_report.json`` — stable, inspectable, no predictions.
    * ``<oracle>_variant_report.pkl``  — full VariantReport including
      the prediction arrays needed to render IGV signal tracks.  Used by
      :func:`consolidate` so the unified multi-oracle IGV has real data.
    * ``rs..._report.html``             — standalone per-oracle report.
    """
    import pickle
    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, f"{oracle_name}_variant_report.json")
    with open(json_path, "w") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=str)
    pkl_path = os.path.join(OUT_DIR, f"{oracle_name}_variant_report.pkl")
    # Pickle keeps ``_predictions`` (numpy arrays) and ``_normalizer`` in
    # place so the consolidator can render IGV without re-running any
    # oracle.  Normalizer is small; predictions dominate the file size.
    try:
        with open(pkl_path, "wb") as fh:
            pickle.dump(report, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        logger.warning("  Could not pickle %s report (%s): IGV may be empty.",
                       oracle_name, exc)
    html_path = os.path.join(
        OUT_DIR, f"{VARIANT['id']}_{VARIANT['gene']}_{oracle_name}_report.html"
    )
    report.to_html(output_path=html_path)
    logger.info("  ✓ wrote %s, %s, and %s",
                os.path.basename(json_path), os.path.basename(pkl_path),
                os.path.basename(html_path))
    return json_path, html_path


def run_chrombpnet():
    from chorus.oracles.chrombpnet import ChromBPNetOracle
    from chorus.core.interval import Interval, GenomeRef
    # use_environment=False so predict_sliding can call self.model directly;
    # we're already inside chorus-chrombpnet env via mamba run.
    oracle = ChromBPNetOracle(use_environment=False, reference_fasta=GENOME_REF)
    oracle.load_pretrained_model(assay="DNASE", cell_type="HepG2", fold=0)
    # Score on ChromBPNet's native 2,114 bp window centred on the variant (the
    # conversational/Python path). The 1 Mb default tiles the 2,114 bp model
    # across ~500 windows and averages the effect down (+1.37 -> +0.68); the
    # wide IGV-display track below is generated separately, so the table value
    # stays the canonical narrow-window number.
    seqlen = int(getattr(oracle, "sequence_length", 2114) or 2114)
    half = seqlen // 2
    # pos-half .. pos+half-1 is `seqlen` bases; pos+half would be seqlen+1.
    # That extra base matters: a query LONGER than sequence_length pushes
    # ChromBPNet down its sliding branch (num_windows=2), which tiles the model
    # twice and leaves values outside the central 1,000 bp populated by a second
    # window rather than untouched. Same arithmetic slip in both runners.
    region = f"{VARIANT['chrom']}:{VARIANT['position'] - half}-{VARIANT['position'] + half - 1}"
    report = _build_variant_report(oracle, oracle_name="chrombpnet", region=region)

    # ChromBPNet's intrinsic prediction window is 2114 bp.  At the
    # multi-oracle locus (1 Mb wide because AlphaGenome's output_size is
    # 1,048,576 bp), that's 0.2 % of the IGV view → invisible at zoom-out.
    # Generate a sliding-window track over a wider region for IGV
    # display only.  Variant scoring (table values, percentiles) is
    # unchanged — those were already computed inside _build_variant_report
    # from the canonical narrow window.  We only swap the IGV-display
    # ``_predictions`` for ref/alt with the wide-locus sliding versions.
    HALF = 524288  # ±~512 kb (covers the 1 Mb AlphaGenome window)
    variant_pos = VARIANT["position"]
    chrom = VARIANT["chrom"]
    wide_start = max(0, variant_pos - HALF)
    wide_end = variant_pos + HALF

    logger.info(
        "Sliding chrombpnet across %s:%d-%d (%.0f kb) for IGV display ...",
        chrom, wide_start, wide_end, (wide_end - wide_start) / 1000,
    )
    ref_iv = Interval.make(GenomeRef(
        chrom=chrom, start=wide_start, end=wide_end, fasta=GENOME_REF,
    ))
    real_pos = (variant_pos - 1) - wide_start  # 0-based variant index
    alt_iv = ref_iv.replace(seq=VARIANT["alt"], start=real_pos, end=real_pos + 1)

    ref_pred = oracle.predict_sliding(ref_iv)
    alt_pred = oracle.predict_sliding(alt_iv)

    if report._predictions is None:
        report._predictions = {}
    report._predictions["reference"] = ref_pred
    alt_key = next(
        (k for k in (report._predictions.keys() if report._predictions else [])
         if k != "reference"),
        "alt_1",
    )
    report._predictions[alt_key] = alt_pred

    return _save_oracle_artefacts(report, "chrombpnet")


def run_cherimoya():
    """Cherimoya (CATv1) on the same HepG2 DNase question ChromBPNet answers.

    Deliberately the same assay and biosample as the ChromBPNet row, because the
    interesting comparison here is not "another accessibility number" but two
    independently trained accessibility models on one variant. They share a
    2,114 bp input window and both emit base-pair-resolution profiles, so the
    rows are directly comparable and the IGV tracks line up.

    ``DNASE:ENCSR149XIL`` is the HepG2 DNase experiment (count_pearson 0.649).
    Cherimoya track ids are ``ASSAY:ENCSR`` rather than ``ASSAY:biosample``,
    because (assay, biosample) is ambiguous for 1,188 of its 1,518 experiments --
    so the accession is what pins the model, and passing ``cell_type="HepG2"``
    alone would not identify one.
    """
    from chorus.oracles.cherimoya import CherimoyaOracle
    from chorus.core.interval import Interval, GenomeRef
    # use_environment=False: we are already inside chorus-cherimoya via mamba
    # run, and predict_sliding needs to reach self.model directly.
    oracle = CherimoyaOracle(use_environment=False, reference_fasta=GENOME_REF)
    oracle.load_pretrained_model(encode_id="ENCSR149XIL", fold=0)

    # Native window centred on the variant, for the same reason as ChromBPNet:
    # tiling a 2,114 bp model across the 1 Mb multi-oracle locus averages the
    # effect down, and the table value should be the canonical narrow-window one.
    seqlen = int(getattr(oracle, "sequence_length", 2114) or 2114)
    half = seqlen // 2
    # pos-half .. pos+half-1 is `seqlen` bases; pos+half would be seqlen+1.
    # That extra base matters: a query LONGER than sequence_length pushes
    # ChromBPNet down its sliding branch (num_windows=2), which tiles the model
    # twice and leaves values outside the central 1,000 bp populated by a second
    # window rather than untouched. Same arithmetic slip in both runners.
    region = f"{VARIANT['chrom']}:{VARIANT['position'] - half}-{VARIANT['position'] + half - 1}"
    report = _build_variant_report(oracle, oracle_name="cherimoya", region=region)

    # Wide sliding track for IGV display only; scoring above is untouched.
    HALF = 524288
    variant_pos = VARIANT["position"]
    chrom = VARIANT["chrom"]
    wide_start = max(0, variant_pos - HALF)
    wide_end = variant_pos + HALF

    logger.info(
        "Sliding cherimoya across %s:%d-%d (%.0f kb) for IGV display ...",
        chrom, wide_start, wide_end, (wide_end - wide_start) / 1000,
    )
    ref_iv = Interval.make(GenomeRef(
        chrom=chrom, start=wide_start, end=wide_end, fasta=GENOME_REF,
    ))
    real_pos = (variant_pos - 1) - wide_start  # 0-based variant index
    alt_iv = ref_iv.replace(seq=VARIANT["alt"], start=real_pos, end=real_pos + 1)

    ref_pred = oracle.predict_sliding(ref_iv)
    alt_pred = oracle.predict_sliding(alt_iv)

    if report._predictions is None:
        report._predictions = {}
    report._predictions["reference"] = ref_pred
    alt_key = next(
        (k for k in (report._predictions.keys() if report._predictions else [])
         if k != "reference"),
        "alt_1",
    )
    report._predictions[alt_key] = alt_pred

    return _save_oracle_artefacts(report, "cherimoya")


def _legnet_sliding_prediction(oracle, chrom, wide_start, wide_end, seq):
    """Tile LegNet's 200 bp window across a wide locus for IGV display.

    LegNet is a 200 bp element-level model, but ``predict_bigseq`` already
    slides that window across an arbitrarily long sequence (one MPRA-activity
    scalar per window).  We tile it non-overlapping (step = window = 200 bp)
    across the full ``seq`` (the ref- or alt-substituted sequence spanning
    ``chrom:wide_start-wide_end``), then expand each window's scalar across
    its 200 bp span to build a 1 bp-resolution ``values`` array covering the
    whole interval.  This mirrors ``ChromBPNet.predict_sliding`` (same
    ``prediction_interval`` == query interval, base-pair resolution, full
    locus coverage), so the LegNet IGV track spans the locus like every other
    oracle instead of a single 200 bp blip.  Used for IGV display only — the
    variant-effect table value is built separately from the native 200 bp
    window in ``_build_variant_report`` and is left untouched.
    """
    import numpy as np
    from chorus.core.interval import Interval, GenomeRef
    from chorus.core.result import OraclePrediction, OraclePredictionTrack
    from chorus.oracles.legnet_source.model_usage import predict_bigseq

    win = int(oracle.sequence_length)  # 200 bp
    step = win                         # non-overlapping tiling
    Q = wide_end - wide_start

    # predict_bigseq's dataset adds the MPRA flanks then pads short windows to
    # window_size; the final (partial) window would otherwise be a different
    # padded length than full windows and break batch collation.  Right-pad the
    # sequence with N up to an exact multiple of win so every window is full.
    pad = (-len(seq)) % win
    seq_padded = seq + ("N" * pad)

    # One scalar per non-overlapping 200 bp window across the full sequence.
    preds, offsets = predict_bigseq(
        oracle._model, seq=seq_padded, step=step, window_size=win,
        reverse_aug=oracle.average_reverse,
        left_flank=oracle.left_flank, right_flank=oracle.right_flank,
        batch_size=max(64, int(oracle.batch_size)),
    )
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)

    # Emit one value per tiled window at ``step`` resolution, NOT one value per
    # base pair.
    #
    # This used to expand each window's scalar across its 200 bp span into a
    # length-Q array and declare ``resolution=1`` ("base-pair resolution, like
    # ChromBPNet"). That was false: LegNet produces one MPRA-activity scalar per
    # non-overlapping 200 bp window, so a 1 bp array repeats every genuine value
    # 200x. Over a 1,048,576 bp locus that is 5,243 real numbers rendered as
    # 1,048,576 IGV features — 99.5% redundant, and it is what made
    # rs12740374_SORT1_legnet_report.html 131 MB and impossible to commit at all
    # (issue #129, above GitHub's hard 100 MiB limit).
    #
    # Reporting the true resolution is lossless — the information was always
    # 200 bp — and it removes the redundancy at the source rather than relying
    # on a downstream feature cap to mop it up. Note the cap in
    # _calculate_track_bin_size worked *because* it ignores this field; a
    # binning rule that trusted ``resolution`` would have read the fabricated 1
    # and reinstated the 131 MB report.
    #
    # Trailing partial window: predict_bigseq was fed a sequence right-padded to
    # a multiple of ``win``, so preds covers ceil(Q/step) windows. Keep only
    # those whose start falls inside the query.
    n_windows = (Q + step - 1) // step
    values = np.asarray(preds[:n_windows], dtype=np.float64)

    query_interval = Interval.make(GenomeRef(
        chrom=chrom, start=wide_start, end=wide_end, fasta=GENOME_REF,
    ))
    track = OraclePredictionTrack.create(
        source_model="legnet",
        assay_id=oracle.assay_id,
        track_id=oracle.assay_id,
        assay_type=oracle.assay,
        cell_type=oracle.cell_type,
        query_interval=query_interval,
        prediction_interval=query_interval,
        input_interval=query_interval,
        resolution=step,                    # one scalar per tiled 200 bp window
        values=values,
        metadata=None,
        preferred_aggregation="mean",
        preferred_interpolation="linear_divided",
        preferred_scoring_strategy="mean",
    )
    final = OraclePrediction()
    final.add(oracle.assay_id, track)
    return final


def run_legnet():
    from chorus.oracles.legnet import LegNetOracle
    from chorus.core.interval import Interval, GenomeRef
    # use_environment=False so _legnet_sliding_prediction can call
    # predict_bigseq on oracle._model directly; we're already inside the
    # chorus-legnet env via mamba run.
    oracle = LegNetOracle(
        cell_type="HepG2", assay="LentiMPRA",
        use_environment=False, reference_fasta=GENOME_REF,
    )
    oracle.load_pretrained_model()
    # LegNet is a 200 bp element-level model. Mirror the conversational/MCP path
    # EXACTLY for the TABLE VALUE (server._auto_region passes a 1 bp region that
    # base.py auto-widens to a single 200 bp window centred on the variant). A
    # wider region tiles the model into several windows and averages the
    # single-variant effect away, so the table value would not match what a user
    # gets by asking conversationally.
    pos = VARIANT["position"]
    region = f"{VARIANT['chrom']}:{pos}-{pos + 1}"
    report = _build_variant_report(oracle, oracle_name="legnet", region=region)

    # IGV-display track: tile LegNet's 200 bp window across a wide locus so the
    # track spans the region (like ChromBPNet) instead of a single 200 bp blip.
    # Variant scoring (table values, percentiles) is unchanged — that was
    # computed above from the native 200 bp window.  We only swap the
    # IGV-display ``_predictions`` for ref/alt with the wide-locus versions.
    HALF = 524288  # ±~512 kb (covers the 1 Mb AlphaGenome window) — matches ChromBPNet
    chrom = VARIANT["chrom"]
    wide_start = max(0, pos - HALF)
    wide_end = pos + HALF

    logger.info(
        "Tiling legnet across %s:%d-%d (%.0f kb) for IGV display ...",
        chrom, wide_start, wide_end, (wide_end - wide_start) / 1000,
    )
    ref_iv = Interval.make(GenomeRef(
        chrom=chrom, start=wide_start, end=wide_end, fasta=GENOME_REF,
    ))
    real_pos = (pos - 1) - wide_start  # 0-based variant index within the interval
    alt_iv = ref_iv.replace(seq=VARIANT["alt"], start=real_pos, end=real_pos + 1)
    ref_seq = ref_iv.sequence
    alt_seq = alt_iv.sequence

    ref_pred = _legnet_sliding_prediction(oracle, chrom, wide_start, wide_end, ref_seq)
    alt_pred = _legnet_sliding_prediction(oracle, chrom, wide_start, wide_end, alt_seq)

    if report._predictions is None:
        report._predictions = {}
    report._predictions["reference"] = ref_pred
    alt_key = next(
        (k for k in (report._predictions.keys() if report._predictions else [])
         if k != "reference"),
        "alt_1",
    )
    report._predictions[alt_key] = alt_pred

    return _save_oracle_artefacts(report, "legnet")


# HepG2 tracks for AlphaGenome — kept small & focused so the consensus
# matrix highlights the multi-layer picture rather than being dominated by
# hundreds of near-zero tracks.
ALPHAGENOME_TRACKS = [
    "DNASE/EFO:0001187 DNase-seq/.",
    # CEBPA/CEBPB ChIP-seq tracks in HepG2 are only available in
    # AlphaGenome as ENCODE's genetically-modified (CRISPR insertion)
    # variants — use those identifiers verbatim.
    "CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA genetically modified (insertion) using CRISPR targeting H. sapiens CEBPA/.",
    "CHIP_HISTONE/EFO:0001187 Histone ChIP-seq H3K27ac/.",
    # CAGE: request a single strand only. Both strands resolve to the same
    # display label ("CAGE:HepG2") in the unified IGV, so listing both produced
    # a duplicate AlphaGenome CAGE track in the multi-oracle browser. We keep the
    # MINUS strand: it is the value cited in the article (+1.52 log2FC; the +
    # strand gives +1.22) and the biologically relevant strand for the
    # SORT1/CELSR2/PSRC1 locus (transcribed on the minus strand).
    "CAGE/hCAGE EFO:0001187/-",
]


def run_alphagenome():
    from chorus.oracles.alphagenome import AlphaGenomeOracle
    # We're already inside the ``chorus-alphagenome`` env (see mamba run
    # invocation above this function's call site), so load the model
    # directly rather than spawning yet another subprocess — the
    # ``use_environment=True`` path was hanging without producing any
    # subprocess output.
    oracle = AlphaGenomeOracle(
        use_environment=False,
        reference_fasta=GENOME_REF,
    )
    oracle.load_pretrained_model()
    report = _build_variant_report(
        oracle, oracle_name="alphagenome", assay_ids=ALPHAGENOME_TRACKS,
    )
    return _save_oracle_artefacts(report, "alphagenome")


# ---------------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------------

def _dedup_duplicate_display_tracks(report) -> None:
    """Drop prediction tracks that collapse to a duplicate IGV display label.

    AlphaGenome's CAGE +/- strands have distinct track_ids but resolve to the
    same unified-IGV label ("CAGE:HepG2"), which would render two identical
    "alphagenome · CAGE:HepG2" bands in the multi-oracle browser.  Keep the
    first track for each display label and drop the rest, in-place, across all
    allele keys.  No-op for a freshly regenerated AlphaGenome artefact (the
    track list now requests a single CAGE strand), but keeps an existing
    two-strand pickle from re-introducing the duplicate band on consolidation.
    """
    from chorus.analysis.variant_report import _track_description

    preds = getattr(report, "_predictions", None)
    if not preds:
        return
    # Decide which track_ids to keep using the first allele's track set; apply
    # the same keep-set to every allele so ref/alt stay aligned.
    first_pred = next(iter(preds.values()))
    seen_labels: set[str] = set()
    keep_ids: list[str] = []
    for aid in first_pred.keys():
        label = _track_description(first_pred[aid]) or aid
        if label in seen_labels:
            logger.info("  dedup: dropping duplicate display track %r (%s)",
                        aid, label)
            continue
        seen_labels.add(label)
        keep_ids.append(aid)
    for allele_key, pred in preds.items():
        preds[allele_key] = pred.subset(keep_ids)


def consolidate():
    """Assemble the multi-oracle HTML from per-oracle artefacts in OUT_DIR.

    Prefers the ``<oracle>_variant_report.pkl`` when present — pickles
    include the prediction arrays the unified IGV needs.  Falls back to
    JSON-only for an oracle whose pickle is missing, which yields an
    IGV panel with the modification marker but no signal tracks for
    that oracle.
    """
    import pickle
    from chorus.analysis import MultiOracleReport
    from chorus.analysis.analysis_request import AnalysisRequest
    # Fix numpy pickle compatibility (old -> new internal paths)
    try:
        import numpy.core.numeric as numeric
        sys.modules["numpy._core.numeric"] = numeric
    except Exception:
        pass

    per_oracle = {}
    reports = []
    ordered_oracles = []
    # Order is the display order in the consensus matrix. chrombpnet and
    # cherimoya sit adjacent on purpose: same assay, same biosample, two
    # independently trained models, so a reader can compare them directly.
    for oracle_name in ("chrombpnet", "cherimoya", "legnet", "alphagenome"):
        pkl = os.path.join(OUT_DIR, f"{oracle_name}_variant_report.pkl")
        jp = os.path.join(OUT_DIR, f"{oracle_name}_variant_report.json")
        if os.path.isfile(pkl):
            with open(pkl, "rb") as fh:
                rep = pickle.load(fh)
            if oracle_name == "alphagenome":
                _dedup_duplicate_display_tracks(rep)
            reports.append(rep)
            ordered_oracles.append(oracle_name)
            logger.info("  loaded %s from pickle (with predictions)", oracle_name)
        elif os.path.isfile(jp):
            from chorus.analysis.variant_report import VariantReport
            with open(jp) as fh:
                data = json.load(fh)
            reports.append(VariantReport.from_dict(data))
            ordered_oracles.append(oracle_name)
            logger.info("  loaded %s from JSON only (no IGV predictions)",
                        oracle_name)
        else:
            logger.warning("Missing per-oracle data for %s — skipped.", oracle_name)
            continue
        html_fname = (
            f"{VARIANT['id']}_{VARIANT['gene']}_{oracle_name}_report.html"
        )
        if os.path.isfile(os.path.join(OUT_DIR, html_fname)):
            per_oracle[oracle_name] = html_fname

    if not reports:
        raise SystemExit(
            "No per-oracle artefacts found. Run --oracle chrombpnet/legnet/"
            "alphagenome first."
        )

    ar = AnalysisRequest(
        user_prompt=USER_PROMPT,
        tool_name="MultiOracleReport",
        oracle_name=", ".join(ordered_oracles),
        normalizer_name="per-oracle chorus per-track v1",
        tracks_requested="assay_ids as listed in each per-oracle request",
    )
    moracle = MultiOracleReport.from_reports(
        reports,
        variant_id=VARIANT["id"],
        analysis_request=ar,
        per_oracle_report_paths=per_oracle,
    )

    html_path = os.path.join(OUT_DIR, f"{VARIANT['id']}_{VARIANT['gene']}_multioracle_report.html")
    moracle.to_html(output_path=html_path)

    md_path = os.path.join(OUT_DIR, "example_output.md")
    with open(md_path, "w") as fh:
        fh.write(moracle.to_markdown())

    json_path = os.path.join(OUT_DIR, "example_output.json")
    with open(json_path, "w") as fh:
        json.dump(moracle.to_dict(), fh, indent=2, default=str)

    # example_output.tsv — every other walkthrough ships one, and this was the
    # only directory of 13 without it, because this script wrote the JSON and
    # md but never a TSV. `rerender_examples.py` refreshes a TSV only when one
    # already exists, so the gap could not heal itself.
    #
    # MultiOracleReport has no to_dataframe() of its own (only to_dict), so
    # project the per-oracle VariantReports that built it and prefix an
    # `oracle` column. Same schema as the single-oracle TSVs plus that column,
    # so the two are directly comparable.
    tsv_path = os.path.join(OUT_DIR, "example_output.tsv")
    try:
        import pandas
        frames = []
        for oracle_name, rep in zip(ordered_oracles, reports):
            df = rep.to_dataframe()
            df.insert(0, "oracle", oracle_name)
            frames.append(df)
        pandas.concat(frames, ignore_index=True).to_csv(
            tsv_path, sep="\t", index=False,
        )
        logger.info("  ✓ wrote %s", os.path.basename(tsv_path))
    except Exception as exc:
        logger.warning("  TSV failed: %s", exc)

    logger.info("  ✓ wrote %s", os.path.basename(html_path))
    logger.info("  ✓ wrote %s", os.path.basename(md_path))
    logger.info("  ✓ wrote %s", os.path.basename(json_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oracle",
        choices=["chrombpnet", "cherimoya", "legnet", "alphagenome"],
        help="Run a single oracle and save its VariantReport JSON to OUT_DIR.",
    )
    parser.add_argument(
        "--consolidate", action="store_true",
        help="Read per-oracle JSONs from OUT_DIR and write the multi-oracle HTML.",
    )
    args = parser.parse_args()

    if args.oracle == "chrombpnet":
        run_chrombpnet()
    elif args.oracle == "cherimoya":
        run_cherimoya()
    elif args.oracle == "legnet":
        run_legnet()
    elif args.oracle == "alphagenome":
        run_alphagenome()

    if args.consolidate:
        consolidate()

    if not (args.oracle or args.consolidate):
        parser.error("pass --oracle <name> and/or --consolidate")


if __name__ == "__main__":
    main()
