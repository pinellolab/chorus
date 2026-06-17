"""Regenerate the multi-oracle validation example for SORT1 rs12740374.

The report combines three oracles that look at the same variant from very
different angles:

    ChromBPNet      — chromatin accessibility (ATAC/DNase), HepG2
    LegNet          — promoter activity (LentiMPRA), HepG2
    AlphaGenome     — generalist: ChIP, histone marks, CAGE, etc.

Each oracle runs inside its own conda env:

    mamba run -n chorus-chrombpnet python scripts/regenerate_multioracle.py --oracle chrombpnet
    mamba run -n chorus-legnet     python scripts/regenerate_multioracle.py --oracle legnet
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
    region = f"{VARIANT['chrom']}:{VARIANT['position'] - half}-{VARIANT['position'] + half}"
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


def run_legnet():
    from chorus.oracles.legnet import LegNetOracle
    oracle = LegNetOracle(cell_type="HepG2", assay="LentiMPRA")
    oracle.load_pretrained_model()
    # LegNet is a 200 bp element-level model. Mirror the conversational/MCP path
    # EXACTLY (server._auto_region passes a 1 bp region that base.py auto-widens to
    # a single 200 bp window centred on the variant). A wider region tiles the
    # model into several windows and averages the single-variant effect away, so
    # the table value would not match what a user gets by asking conversationally.
    pos = VARIANT["position"]
    region = f"{VARIANT['chrom']}:{pos}-{pos + 1}"
    report = _build_variant_report(oracle, oracle_name="legnet", region=region)
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
    "CAGE/hCAGE EFO:0001187/-",
    "CAGE/hCAGE EFO:0001187/+",
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
    for oracle_name in ("chrombpnet", "legnet", "alphagenome"):
        pkl = os.path.join(OUT_DIR, f"{oracle_name}_variant_report.pkl")
        jp = os.path.join(OUT_DIR, f"{oracle_name}_variant_report.json")
        if os.path.isfile(pkl):
            with open(pkl, "rb") as fh:
                reports.append(pickle.load(fh))
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
        choices=["chrombpnet", "legnet", "alphagenome"],
        help="Run a single oracle and save its VariantReport JSON to OUT_DIR.",
    )
    parser.add_argument(
        "--consolidate", action="store_true",
        help="Read per-oracle JSONs from OUT_DIR and write the multi-oracle HTML.",
    )
    args = parser.parse_args()

    if args.oracle == "chrombpnet":
        run_chrombpnet()
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
