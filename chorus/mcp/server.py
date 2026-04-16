"""Chorus MCP Server — FastMCP tool definitions.

Run standalone:
    fastmcp run chorus/mcp/server.py

Or via the console_scripts entry-point:
    chorus-mcp
"""

import logging
import re
import sys
from typing import Optional

from fastmcp import FastMCP

from chorus.mcp.state import OracleStateManager
from chorus.mcp.serializers import (
    serialize_prediction,
    serialize_variant_effect,
    serialize_replacement_or_insertion,
)

logger = logging.getLogger(__name__)

_INSTRUCTIONS = (
    "Unified interface for 6 genomic deep-learning oracles "
    "(Enformer, Borzoi, ChromBPNet, Sei, LegNet, AlphaGenome). "
    "Discover tracks, load models, make predictions, and analyse variant effects."
)

mcp = FastMCP("Chorus Genomics", instructions=_INSTRUCTIONS)

# Oracle specs used by list_oracles — avoids importing heavy oracle modules.
ORACLE_SPECS = {
    "enformer": {
        "description": "Enformer (DeepMind) — predict chromatin & gene expression from DNA sequence",
        "framework": "TensorFlow",
        "input_size_bp": 393_216,
        "output_bins": 896,
        "resolution_bp": 128,
        "assay_types": ["DNASE", "ATAC", "CAGE", "CHIP", "RNA"],
    },
    "borzoi": {
        "description": "Borzoi — high-resolution gene expression & chromatin prediction",
        "framework": "PyTorch",
        "input_size_bp": 524_288,
        "output_bins": 6_144,
        "resolution_bp": 32,
        "assay_types": ["DNASE", "ATAC", "CAGE", "CHIP", "RNA"],
    },
    "chrombpnet": {
        "description": "ChromBPNet — base-resolution TF binding & chromatin accessibility",
        "framework": "TensorFlow",
        "input_size_bp": 2_114,
        "output_bins": 1_000,
        "resolution_bp": 1,
        "assay_types": ["ATAC", "DNASE", "CHIP"],
    },
    "sei": {
        "description": "Sei — sequence-level regulatory element classification",
        "framework": "PyTorch",
        "input_size_bp": 4_096,
        "output_bins": 1,
        "resolution_bp": None,
        "assay_types": ["sequence-class"],
    },
    "legnet": {
        "description": "LegNet — MPRA activity prediction",
        "framework": "PyTorch",
        "input_size_bp": 230,
        "output_bins": 1,
        "resolution_bp": None,
        "assay_types": ["LentiMPRA"],
    },
    "alphagenome": {
        "description": "AlphaGenome (DeepMind) — 1-bp resolution across 5930 tracks",
        "framework": "JAX",
        "input_size_bp": 1_048_576,
        "output_bins": 1_048_576,
        "resolution_bp": 1,
        "assay_types": ["DNASE", "ATAC", "CAGE", "CHIP", "RNA", "SPLICE_SITES", "PRO_CAP"],
    },
}


# ── Region/position parsing helpers ──────────────────────────────────

_REGION_RE = re.compile(r'^(chr[\w]+):(\d+)-(\d+)$')
_POSITION_RE = re.compile(r'^(chr[\w]+):(\d+)$')


def _parse_region(region: str) -> tuple[str, int, int]:
    """Parse 'chrN:start-end' into (chrom, start, end) with validation."""
    m = _REGION_RE.match(region)
    if not m:
        raise ValueError(
            f"Invalid region format: '{region}'. "
            f"Expected 'chrN:start-end' (e.g. 'chr1:1000000-1393216')."
        )
    chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    if start >= end:
        raise ValueError(
            f"Invalid region: start ({start}) must be less than end ({end})."
        )
    return chrom, start, end


def _parse_position(position: str) -> tuple[str, int]:
    """Parse 'chrN:pos' into (chrom, pos) with validation."""
    m = _POSITION_RE.match(position)
    if not m:
        raise ValueError(
            f"Invalid position format: '{position}'. "
            f"Expected 'chrN:position' (e.g. 'chr1:1050000')."
        )
    return m.group(1), int(m.group(2))


def _state() -> OracleStateManager:
    return OracleStateManager()


def _safe_tool(fn):
    """Decorator that converts unhandled exceptions into a structured
    ``{"error": ..., "error_type": ...}`` dict so Claude can recover
    gracefully instead of seeing a raw traceback.

    Wraps the function body only; does not interfere with FastMCP's
    registration (apply *inside* ``@mcp.tool()``).
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.exception("MCP tool %s failed", fn.__name__)
            return {
                "error": str(exc) or type(exc).__name__,
                "error_type": type(exc).__name__,
                "tool": fn.__name__,
            }

    return wrapper


def _auto_region(oracle, position: str) -> str:
    """Compute an input region centered on a variant position.

    Uses a minimal region so the oracle's internal extend() properly sizes both
    the input and prediction intervals.  This avoids a mismatch where
    prediction_interval covers the full input window but values only cover
    the output window (e.g. Enformer: 393 kb input → 114 kb output).
    """
    chrom, pos = _parse_position(position)
    return f"{chrom}:{pos}-{pos + 1}"


# ── Discovery tools ──────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def list_oracles() -> dict:
    """List all 6 genomic oracles with their specs, environment install status, and loaded status.

    No model loading is required — this returns static metadata plus live status.
    """
    state = _state()
    loaded_names = {info["name"] for info in state.list_loaded()}

    # Check environment install status
    env_status: dict[str, bool] = {}
    try:
        from chorus.core.environment import EnvironmentManager
        em = EnvironmentManager()
        for name in ORACLE_SPECS:
            env_status[name] = em.environment_exists(name)
    except Exception:
        pass

    results = []
    for name, spec in ORACLE_SPECS.items():
        results.append({
            "name": name,
            **spec,
            "environment_installed": env_status.get(name, "unknown"),
            "loaded": name in loaded_names,
        })
    return {"oracles": results}


@mcp.tool()
@_safe_tool
def list_tracks(oracle_name: str, query: Optional[str] = None) -> dict:
    """List or search available tracks/assays for an oracle.

    Does not require the oracle to be loaded — uses metadata classes.

    Args:
        oracle_name: Oracle name (enformer, borzoi, chrombpnet, sei, legnet, alphagenome).
        query: Optional search string to filter tracks (e.g. "K562", "DNASE"). Use the returned 'identifier' field as the assay_id for predictions.
    """
    oracle_name = oracle_name.lower()

    # Try Borzoi metadata (richest search)
    if oracle_name == "borzoi":
        from chorus.oracles.borzoi_source.borzoi_metadata import get_metadata
        meta = get_metadata()
        if query:
            df = meta.search_tracks(query)
            results = df.to_dict(orient="records")
            return {"oracle": oracle_name, "query": query, "num_results": len(results), "tracks": results[:200]}
        else:
            return {
                "oracle": oracle_name,
                "assay_types": meta.list_assay_types(),
                "cell_types": meta.list_cell_types(),
                "note": "Use query parameter to search tracks (e.g. query='K562' or query='DNASE:K562'). Use the 'identifier' field as assay_id for predictions.",
            }

    if oracle_name == "enformer":
        from chorus.oracles.enformer_source.enformer_metadata import get_metadata
        meta = get_metadata()
        if query:
            df = meta.search_tracks(query)
            results = df.to_dict(orient="records")
            return {"oracle": oracle_name, "query": query, "num_results": len(results), "tracks": results[:200]}
        return {
            "oracle": oracle_name,
            "assay_types": meta.list_assay_types(),
            "cell_types": meta.list_cell_types(),
            "note": "Use query parameter to search tracks (e.g. query='K562' or query='DNASE:K562'). Use the 'identifier' field as assay_id for predictions.",
        }

    if oracle_name == "chrombpnet":
        from chorus.oracles.chrombpnet_source.metadata import BPNetMetadata
        from chorus.oracles.chrombpnet_source.chrombpnet_globals import CHROMBPNET_MODELS_DICT
        meta = BPNetMetadata()
        atac_cell_types = sorted(CHROMBPNET_MODELS_DICT.get("ATAC", {}).keys())
        dnase_cell_types = sorted(CHROMBPNET_MODELS_DICT.get("DNASE", {}).keys())
        chip_cell_types = meta.list_cell_types()
        chip_tfs = meta.list_TFs()
        if query:
            q = query.upper()
            # Show matching ATAC/DNASE cell types
            matching_atac = [c for c in atac_cell_types if q in c.upper()]
            matching_dnase = [c for c in dnase_cell_types if q in c.upper()]
            # Show matching CHIP TF-cell_type combinations
            matching_chip_cells = [c for c in chip_cell_types if q in c.upper()]
            matching_chip_tfs = [t for t in chip_tfs if q in t.upper()]
            chip_combos = []
            if matching_chip_cells or matching_chip_tfs:
                for ct in (matching_chip_cells or chip_cell_types):
                    for tf in meta.list_TFs_by_cell_type(ct):
                        if not matching_chip_tfs or tf in matching_chip_tfs:
                            chip_combos.append({"cell_type": ct, "TF": tf})
            return {
                "oracle": oracle_name,
                "query": query,
                "ATAC_cell_types": matching_atac,
                "DNASE_cell_types": matching_dnase,
                "CHIP_combinations": chip_combos[:100],
                "note": "Load with: load_oracle('chrombpnet', assay='ATAC', cell_type='K562') or load_oracle('chrombpnet', assay='CHIP', cell_type='K562', TF='GATA1')",
            }
        return {
            "oracle": oracle_name,
            "assay_types": ["ATAC", "DNASE", "CHIP"],
            "ATAC_cell_types": atac_cell_types,
            "DNASE_cell_types": dnase_cell_types,
            "CHIP_cell_types": chip_cell_types,
            "CHIP_TFs": chip_tfs,
            "note": "Load with: load_oracle('chrombpnet', assay='ATAC', cell_type='K562') or load_oracle('chrombpnet', assay='CHIP', cell_type='K562', TF='GATA1')",
        }

    if oracle_name == "alphagenome":
        from chorus.oracles.alphagenome_source.alphagenome_metadata import get_metadata
        meta = get_metadata()
        if query:
            df = meta.search_tracks(query)
            results = df.to_dict(orient="records")
            return {"oracle": oracle_name, "query": query, "num_results": len(results), "tracks": results[:200]}
        return {
            "oracle": oracle_name,
            "assay_types": meta.list_assay_types(),
            "cell_types": meta.list_cell_types(),
            "note": "Use query parameter to search tracks (e.g. query='K562' or query='GATA1'). Use the 'identifier' field as assay_id for predictions.",
        }

    if oracle_name == "sei":
        return {
            "oracle": oracle_name,
            "assay_types": ["sequence-class"],
            "note": "Sei predicts regulatory element classes, not per-assay tracks.",
        }

    if oracle_name == "legnet":
        return {
            "oracle": oracle_name,
            "assay_types": ["LentiMPRA"],
            "cell_types": ["K562", "HepG2", "WTC11"],
            "note": "LegNet predicts lentiMPRA activity. Specify cell_type when loading.",
        }

    valid = ", ".join(ORACLE_SPECS.keys())
    return {"error": f"Unknown oracle: '{oracle_name}'. Valid names: {valid}"}


@mcp.tool()
@_safe_tool
def list_genomes() -> dict:
    """List available reference genomes and their download status."""
    from chorus.utils.genome import GenomeManager

    gm = GenomeManager()
    available = gm.list_available_genomes()
    downloaded = gm.list_downloaded_genomes()

    genomes = []
    for gid, desc in available.items():
        info: dict = {"id": gid, "description": desc, "downloaded": gid in downloaded}
        if gid in downloaded:
            info["path"] = str(gm.get_genome_path(gid))
        genomes.append(info)

    return {"genomes": genomes}


@mcp.tool()
@_safe_tool
def get_genes_in_region(chrom: str, start: int, end: int) -> dict:
    """Get gene annotations in a genomic region.

    Args:
        chrom: Chromosome (e.g. "chr1").
        start: Region start position.
        end: Region end position.
    """
    from chorus.utils.annotations import get_genes_in_region as _get_genes

    df = _get_genes(chrom, start, end)
    # Drop the heavy 'attributes' column for the response
    if "attributes" in df.columns:
        df = df.drop(columns=["attributes"])
    records = df.to_dict(orient="records")
    return {"chrom": chrom, "start": start, "end": end, "num_genes": len(records), "genes": records}


@mcp.tool()
@_safe_tool
def get_gene_tss(gene_name: str) -> dict:
    """Get transcription start site (TSS) positions for a gene.

    Args:
        gene_name: Gene symbol (e.g. "GATA1", "TP53").
    """
    from chorus.utils.annotations import get_gene_tss as _get_tss

    df = _get_tss(gene_name)
    records = df.to_dict(orient="records")
    return {"gene_name": gene_name, "num_transcripts": len(records), "tss_positions": records}


# ── Oracle lifecycle ─────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def load_oracle(
    oracle_name: str,
    device: Optional[str] = None,
    assay: Optional[str] = None,
    cell_type: Optional[str] = None,
    TF: Optional[str] = None,
    fold: Optional[int] = None,
    model_type: Optional[str] = None,
) -> dict:
    """Load a genomic oracle and its pretrained model (cached for reuse).

    This can take 30 seconds to several minutes depending on the model.

    Args:
        oracle_name: Oracle name (enformer, borzoi, chrombpnet, sei, legnet, alphagenome).
        device: Device to use — "cpu", "cuda", "cuda:0", etc. None = auto-detect.
        assay: (ChromBPNet only) Assay type — "ATAC", "DNASE", or "CHIP".
        cell_type: (ChromBPNet/LegNet) Cell type — e.g. "K562", "HepG2".
        TF: (ChromBPNet CHIP only) Transcription factor — e.g. "GATA1", "CTCF".
        fold: (ChromBPNet ATAC/DNASE only) Cross-validation fold 0-4 (default 0).
        model_type: (ChromBPNet only) Model variant — "chrombpnet", "bias_scaled", "chrombpnet_nobias".
    """
    kwargs: dict = {}
    if assay:
        kwargs["assay"] = assay
    if cell_type:
        kwargs["cell_type"] = cell_type
    if TF:
        kwargs["TF"] = TF
    if fold is not None:
        kwargs["fold"] = fold
    if model_type:
        kwargs["model_type"] = model_type
    return _state().load_oracle(oracle_name, device=device, **kwargs)


@mcp.tool()
@_safe_tool
def unload_oracle(oracle_name: str) -> dict:
    """Unload an oracle to free memory.

    Args:
        oracle_name: Oracle name to unload.
    """
    removed = _state().unload_oracle(oracle_name)
    return {"name": oracle_name.lower(), "unloaded": removed}


@mcp.tool()
@_safe_tool
def oracle_status() -> dict:
    """Show which oracles are currently loaded, their device, and load time."""
    return {"loaded_oracles": _state().list_loaded()}


# ── Prediction tools ─────────────────────────────────────────────────

@mcp.tool()
@_safe_tool
def predict(
    oracle_name: str,
    region: str,
    assay_ids: list[str],
) -> dict:
    """Make a wild-type prediction for a genomic region.

    Returns per-track summary stats (mean/max/min/std). For small predictions
    the full values are included inline; for large ones a downsampled preview
    is returned and full data is saved as bedgraph files.

    Args:
        oracle_name: A loaded oracle name.
        region: Genomic region as "chr1:1000000-1393216".
        assay_ids: List of assay identifiers (e.g. ["DNASE:K562", "CAGE:K562"]).
    """
    state = _state()
    oracle = state.get_oracle(oracle_name)

    input_data = _parse_region(region)

    prediction = oracle.predict(input_data, assay_ids)
    normalizer = state.get_normalizer(oracle_name)
    return serialize_prediction(
        prediction, output_dir=state.output_dir, prefix=f"{oracle_name}_wt_",
        normalizer=normalizer, oracle_name=oracle_name,
    )


@mcp.tool()
@_safe_tool
def predict_variant_effect(
    oracle_name: str,
    position: str,
    ref_allele: str,
    alt_alleles: list[str],
    assay_ids: list[str],
    region: Optional[str] = None,
) -> dict:
    """Predict the effect of a genetic variant.

    Compares predictions for reference vs alternate alleles and returns
    per-allele effect sizes with summary statistics.

    Args:
        oracle_name: A loaded oracle name.
        position: Variant position as "chr1:1050000".
        ref_allele: Reference allele (e.g. "A").
        alt_alleles: Alternate alleles (e.g. ["G", "T"]).
        assay_ids: List of assay identifiers.
        region: Genomic region as "chr1:1000000-1393216". If omitted, auto-centered on the variant position using the oracle's input window.
    """
    state = _state()
    oracle = state.get_oracle(oracle_name)

    if region is None:
        region = _auto_region(oracle, position)

    alleles = [ref_allele] + list(alt_alleles)
    result = oracle.predict_variant_effect(
        genomic_region=region,
        variant_position=position,
        alleles=alleles,
        assay_ids=assay_ids,
    )
    normalizer = state.get_normalizer(oracle_name)
    return serialize_variant_effect(
        result, output_dir=state.output_dir,
        normalizer=normalizer, oracle_name=oracle_name,
    )


@mcp.tool()
@_safe_tool
def predict_region_replacement(
    oracle_name: str,
    region: str,
    replacement_sequence: str,
    assay_ids: list[str],
) -> dict:
    """Replace a genomic region with a custom sequence and predict activity.

    Args:
        oracle_name: A loaded oracle name.
        region: Genomic region to replace as "chr1:1000000-1001000".
        replacement_sequence: DNA sequence to insert in place of the region.
        assay_ids: List of assay identifiers.
    """
    state = _state()
    oracle = state.get_oracle(oracle_name)

    result = oracle.predict_region_replacement(
        genomic_region=region,
        seq=replacement_sequence,
        assay_ids=assay_ids,
    )
    normalizer = state.get_normalizer(oracle_name)
    return serialize_replacement_or_insertion(
        result, output_dir=state.output_dir, prefix=f"{oracle_name}_repl_",
        normalizer=normalizer, oracle_name=oracle_name,
    )


@mcp.tool()
@_safe_tool
def predict_region_insertion(
    oracle_name: str,
    position: str,
    sequence: str,
    assay_ids: list[str],
) -> dict:
    """Insert a sequence at a genomic position and predict activity.

    Args:
        oracle_name: A loaded oracle name.
        position: Insertion point as "chr1:1050000".
        sequence: DNA sequence to insert.
        assay_ids: List of assay identifiers.
    """
    state = _state()
    oracle = state.get_oracle(oracle_name)

    result = oracle.predict_region_insertion_at(
        genomic_position=position,
        seq=sequence,
        assay_ids=assay_ids,
    )
    normalizer = state.get_normalizer(oracle_name)
    return serialize_replacement_or_insertion(
        result, output_dir=state.output_dir, prefix=f"{oracle_name}_ins_",
        normalizer=normalizer, oracle_name=oracle_name,
    )


# ── Scoring & gene expression tools ──────────────────────────────────

@mcp.tool()
@_safe_tool
def score_prediction_region(
    oracle_name: str,
    region: str,
    assay_ids: list[str],
    score_region: str,
    scoring_strategy: str = "mean",
) -> dict:
    """Predict for a region and score a sub-region within the output window.

    Useful for quantifying signal at a specific peak, promoter, or element
    rather than the full output window.

    Args:
        oracle_name: A loaded oracle name.
        region: Input region as "chr1:1000000-1393216".
        assay_ids: List of assay identifiers (e.g. ["DNASE:K562"]).
        score_region: Sub-region to score as "chr1:1050000-1051000".
        scoring_strategy: How to summarise bins — mean, max, sum, or median.
    """
    state = _state()
    oracle = state.get_oracle(oracle_name)

    input_data = _parse_region(region)

    prediction = oracle.predict(input_data, assay_ids)

    sc_chrom, sc_start, sc_end = _parse_region(score_region)
    scores = prediction.score_region(
        sc_chrom, sc_start, sc_end, scoring_strategy
    )

    result = {
        "input_region": region,
        "score_region": score_region,
        "scoring_strategy": scoring_strategy,
        "scores": {k: v for k, v in scores.items()},
    }

    # Add activity percentiles when baselines available
    normalizer = state.get_normalizer(oracle_name)
    if normalizer is not None:
        from chorus.analysis.scorers import classify_track_layer
        from chorus.analysis.normalization import PerTrackNormalizer
        percentiles = {}
        for assay_id, score_val in scores.items():
            if score_val is not None:
                track = prediction[assay_id]
                layer = classify_track_layer(track)
                if isinstance(normalizer, PerTrackNormalizer):
                    pctile = normalizer.activity_percentile(oracle_name, assay_id, score_val)
                else:
                    pctile = normalizer.normalize_baseline(oracle_name, layer, score_val)
                if pctile is not None:
                    percentiles[assay_id] = round(pctile, 4)
        if percentiles:
            result["activity_percentiles"] = percentiles

    return result


@mcp.tool()
@_safe_tool
def score_variant_effect_at_region(
    oracle_name: str,
    position: str,
    ref_allele: str,
    alt_alleles: list[str],
    assay_ids: list[str],
    region: Optional[str] = None,
    score_region: Optional[str] = None,
    at_variant: bool = False,
    window_bins: int = 1,
    scoring_strategy: str = "mean",
) -> dict:
    """Predict a variant effect and score it at a specific region or the variant site.

    Two modes:
    - score_region="chr1:X-Y": score ref/alt in that sub-region.
    - at_variant=True: score ref/alt in a window around the variant position.

    Args:
        oracle_name: A loaded oracle name.
        position: Variant position as "chr1:1050000".
        ref_allele: Reference allele.
        alt_alleles: Alternate alleles.
        assay_ids: List of assay identifiers.
        region: Input region as "chr1:1000000-1393216". If omitted, auto-centered on the variant position.
        score_region: Sub-region to score (e.g. "chr1:1050000-1051000").
        at_variant: If true, score around the variant position instead.
        window_bins: Bins on each side when at_variant is true (default 1).
        scoring_strategy: mean, max, sum, median, or abs_max.
    """
    from chorus.core.result import score_variant_effect as _score_ve

    state = _state()
    oracle = state.get_oracle(oracle_name)

    if region is None:
        region = _auto_region(oracle, position)

    alleles = [ref_allele] + list(alt_alleles)
    variant_result = oracle.predict_variant_effect(
        genomic_region=region,
        variant_position=position,
        alleles=alleles,
        assay_ids=assay_ids,
    )

    kwargs: dict = {
        "at_variant": at_variant,
        "window_bins": window_bins,
        "scoring_strategy": scoring_strategy,
    }
    if score_region is not None:
        sc_chrom, sc_start, sc_end = _parse_region(score_region)
        kwargs["chrom"] = sc_chrom
        kwargs["start"] = sc_start
        kwargs["end"] = sc_end

    scores = _score_ve(variant_result, **kwargs)

    result = {
        "variant_info": variant_result["variant_info"],
        "scoring_strategy": scoring_strategy,
        "at_variant": at_variant,
        "scores": scores,
    }

    # Add activity percentiles for reference scores
    normalizer = state.get_normalizer(oracle_name)
    if normalizer is not None:
        from chorus.analysis.scorers import classify_track_layer
        from chorus.analysis.normalization import PerTrackNormalizer
        ref_pred = variant_result["predictions"].get("reference")
        if ref_pred is not None:
            percentiles = {}
            for assay_id in scores:
                allele_scores = scores[assay_id]
                ref_val = allele_scores.get("reference") if isinstance(allele_scores, dict) else None
                if ref_val is not None:
                    track = ref_pred.get(assay_id)
                    if track:
                        layer = classify_track_layer(track)
                        if isinstance(normalizer, PerTrackNormalizer):
                            pctile = normalizer.activity_percentile(oracle_name, assay_id, ref_val)
                        else:
                            pctile = normalizer.normalize_baseline(oracle_name, layer, ref_val)
                        if pctile is not None:
                            percentiles[assay_id] = round(pctile, 4)
            if percentiles:
                result["ref_activity_percentiles"] = percentiles

    return result


@mcp.tool()
@_safe_tool
def predict_variant_effect_on_gene(
    oracle_name: str,
    position: str,
    ref_allele: str,
    alt_alleles: list[str],
    gene_name: str,
    assay_ids: list[str],
    region: Optional[str] = None,
) -> dict:
    """Predict how a variant affects expression of a nearby gene.

    Uses CAGE tracks with TSS-windowed-max and RNA tracks with exon-sum
    quantification, then computes fold change vs reference.

    Args:
        oracle_name: A loaded oracle name.
        position: Variant position as "chr1:1050000".
        ref_allele: Reference allele.
        alt_alleles: Alternate alleles.
        gene_name: Gene symbol (e.g. "MYC", "TP53").
        assay_ids: List of assay identifiers.
        region: Input region as "chr1:1000000-1393216". If omitted, auto-centered on the variant position.
    """
    state = _state()
    oracle = state.get_oracle(oracle_name)

    if region is None:
        region = _auto_region(oracle, position)

    alleles = [ref_allele] + list(alt_alleles)
    variant_result = oracle.predict_variant_effect(
        genomic_region=region,
        variant_position=position,
        alleles=alleles,
        assay_ids=assay_ids,
    )

    result = oracle.analyze_variant_effect_on_gene(variant_result, gene_name)

    # Check if TSS fell outside the prediction window and add a clear warning
    tss_positions = result.get("tss_positions", [])
    ref_expr = result.get("reference_expression", {})
    all_zero = all(
        info.get("n_tss_in_window", 0) == 0
        for info in ref_expr.values()
    ) if ref_expr else True

    if tss_positions and all_zero:
        # Compute distance from variant to nearest TSS
        var_chrom, var_pos = _parse_position(position)
        nearest_tss = min(tss_positions, key=lambda t: abs(t - var_pos))
        distance_kb = abs(nearest_tss - var_pos) / 1000

        # Get output window size
        output_kb = ORACLE_SPECS.get(oracle_name.lower(), {}).get("output_bins", 0) * (
            ORACLE_SPECS.get(oracle_name.lower(), {}).get("resolution_bp", 1) or 1
        ) / 1000

        result["warning"] = (
            f"{gene_name} TSS (nearest: {var_chrom}:{nearest_tss}) is {distance_kb:.0f}kb "
            f"from the variant — outside {oracle_name}'s {output_kb:.0f}kb output window. "
            f"Try: (1) use a larger-window oracle like borzoi (196kb) or alphagenome (1Mb), "
            f"or (2) pass a custom region spanning both variant and TSS."
        )

    # Add baseline activity percentiles for reference expression levels
    normalizer = state.get_normalizer(oracle_name)
    if normalizer is not None and ref_expr:
        from chorus.analysis.normalization import PerTrackNormalizer
        expr_percentiles = {}
        for assay_id, info in ref_expr.items():
            ref_val = info.get("signal")
            if ref_val is not None:
                if isinstance(normalizer, PerTrackNormalizer):
                    pctile = normalizer.activity_percentile(oracle_name, assay_id, ref_val)
                else:
                    pctile = normalizer.normalize_baseline(
                        oracle_name, "tss_activity", ref_val,
                    )
                if pctile is not None:
                    expr_percentiles[assay_id] = round(pctile, 4)
        if expr_percentiles:
            result["ref_expression_percentiles"] = expr_percentiles

    return result


# ── Multi-layer analysis tools ────────────────────────────────────────

@mcp.tool()
@_safe_tool
def analyze_variant_multilayer(
    oracle_name: str,
    position: str,
    ref_allele: str,
    alt_alleles: list[str],
    assay_ids: list[str],
    gene_name: Optional[str] = None,
    region: Optional[str] = None,
    igv_raw: bool = False,
    user_prompt: Optional[str] = None,
) -> dict:
    """Analyze a variant's regulatory impact across all molecular layers.

    Scores each track using modality-specific strategies:
    - Chromatin (DNASE/ATAC): log2 fold-change of sum in 501bp window
    - TF binding (ChIP-TF): log2 fold-change of sum in 501bp window
    - Histone marks (ChIP-Histone): log2 fold-change of sum in 2001bp window
    - TSS activity (CAGE): log2 fold-change of sum in 501bp window
    - Gene expression (RNA): log fold-change of mean over gene exons
    - Promoter activity (MPRA): simple difference

    For non-coding variants, nearby genes are auto-detected within the
    prediction window so that RNA expression effects can be scored even
    without an explicit gene_name.

    Returns a structured report with scores organized by regulatory layer,
    plus a markdown summary for interpretation. Every report carries the
    original user prompt at the top so it stays interpretable months later.

    Args:
        oracle_name: A loaded oracle name.
        position: Variant position as "chr1:1050000".
        ref_allele: Reference allele.
        alt_alleles: Alternate alleles.
        assay_ids: List of assay identifiers covering different layers
                   (e.g. DNASE, CAGE, ChIP tracks for multi-layer coverage).
                   Pass an empty list or None to score all tracks on
                   oracles that support it (AlphaGenome, Enformer, Borzoi).
        gene_name: Gene symbol for RNA expression scoring (e.g. "SORT1").
                   If omitted, the nearest gene is auto-detected.
        region: Input region as "chr1:1000000-1393216". If omitted, auto-centered.
        igv_raw: When True, the IGV browser in the HTML report shows raw
                 signal with autoscale instead of the layer-aware rescaled
                 view. Table scores are unaffected.
        user_prompt: The user's original natural-language question. Claude
                     should forward this verbatim whenever calling from an
                     MCP conversation — it is rendered at the top of the
                     report for traceability.
    """
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.variant_report import build_variant_report

    state = _state()
    oracle = state.get_oracle(oracle_name)

    if region is None:
        region = _auto_region(oracle, position)

    alleles = [ref_allele] + list(alt_alleles)
    variant_result = oracle.predict_variant_effect(
        genomic_region=region,
        variant_position=position,
        alleles=alleles,
        assay_ids=assay_ids,
    )

    analysis_request = AnalysisRequest(
        user_prompt=user_prompt,
        tool_name="analyze_variant_multilayer",
        oracle_name=oracle_name,
        tracks_requested=(
            "all oracle tracks" if not assay_ids else f"{len(assay_ids)} tracks"
        ),
    )

    report = build_variant_report(
        variant_result,
        oracle_name=oracle_name,
        gene_name=gene_name,
        normalizer=state.get_normalizer(oracle_name),
        igv_raw=igv_raw,
        analysis_request=analysis_request,
    )

    result = report.to_dict()
    result["markdown_report"] = report.to_markdown()

    # Save HTML report to output directory
    if state.output_dir:
        try:
            html_path = report.to_html(output_path=state.output_dir)
            result["html_report_path"] = html_path
        except Exception:
            pass  # HTML generation is optional

    return result


@mcp.tool()
@_safe_tool
def discover_variant(
    oracle_name: str,
    position: str,
    ref_allele: str,
    alt_alleles: list[str],
    gene_name: Optional[str] = None,
    top_n: int = 3,
    igv_raw: bool = False,
    user_prompt: Optional[str] = None,
) -> dict:
    """Discover which cell types and regulatory layers are most affected by a variant.

    Predicts variant effect across ALL available tracks (thousands for
    Enformer/Borzoi/AlphaGenome, or iterates all models for ChromBPNet/LegNet),
    ranks by effect magnitude, and returns the top hits with a full report.

    This is the primary tool for variant interpretation — it tells you WHERE
    the variant has impact without requiring you to pre-select tracks.

    Args:
        oracle_name: A loaded oracle name.
        position: Variant position as "chr1:109274968".
        ref_allele: Reference allele (e.g. "G").
        alt_alleles: Alternate alleles (e.g. ["T"]).
        gene_name: Optional gene for expression analysis.
        top_n: Number of top tracks per regulatory layer to show.
        user_prompt: Original user prompt, forwarded into the report header.
    """
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.discovery import discover_variant_effects

    state = _state()
    oracle = state.get_oracle(oracle_name)
    normalizer = state.get_normalizer(oracle_name)

    result = discover_variant_effects(
        oracle,
        oracle_name=oracle_name,
        variant_position=position,
        alleles=[ref_allele] + list(alt_alleles),
        top_n_per_layer=top_n,
        gene_name=gene_name,
        normalizer=normalizer,
        output_path=state.output_dir,
        igv_raw=igv_raw,
    )

    # Serialize: extract report as markdown, remove non-serializable VariantReport
    report = result.pop("report", None)
    if report is not None:
        report.analysis_request = AnalysisRequest(
            user_prompt=user_prompt,
            tool_name="discover_variant",
            oracle_name=oracle_name,
            tracks_requested="all oracle tracks",
        )
        result["markdown_report"] = report.to_markdown()
        if state.output_dir:
            try:
                html_path = report.to_html(output_path=state.output_dir)
                result["html_report_path"] = html_path
            except Exception:
                pass

    return result


@mcp.tool()
@_safe_tool
def discover_variant_cell_types(
    oracle_name: str,
    position: str,
    ref_allele: str,
    alt_alleles: list[str],
    gene_name: Optional[str] = None,
    top_n: int = 5,
    min_effect: float = 0.15,
    user_prompt: Optional[str] = None,
) -> dict:
    """Discovery mode: find which cell types are most affected by a variant.

    Use this when you don't know which cell type is relevant — let the model
    tell you where the variant matters most.

    **Two-stage analysis:**
    1. Screens all DNASE/ATAC tracks (~472 cell types on AlphaGenome, ~638
       on Enformer) to rank cell types by chromatin effect magnitude.
    2. For each top cell type, runs full multi-layer analysis (chromatin,
       TF, histone, CAGE, RNA) limited to that cell type's tracks.

    **Runtime expectations** (AlphaGenome, single A100):
      - Stage 1 screen: ~30–60 s
      - Stage 2 per-cell-type analysis: ~30 s × ``top_n``
      - Typical end-to-end with default ``top_n=5``: 3–4 minutes

    Args:
        oracle_name: A loaded oracle name (ideally AlphaGenome for broadest
            cell-type coverage).
        position: Variant position as "chr1:1050000".
        ref_allele: Reference allele.
        alt_alleles: Alternate alleles.
        gene_name: Optional gene to focus expression analysis on.
        top_n: Number of top cell types to analyze in detail (default 5).
        min_effect: Minimum |log2FC| in DNASE/ATAC to consider a cell type
            hit (default 0.15).
        user_prompt: Original user prompt, forwarded into each sub-report.

    Returns:
        Dict with:
          - ``variant``: position + alleles
          - ``cell_type_ranking``: ordered list of top cell-type hits with
            effect size and best track
          - ``reports``: one full :class:`VariantReport` per top cell type,
            as both ``scores`` (dict) and ``markdown`` (string)
    """
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.discovery import discover_and_report

    state = _state()
    oracle = state.get_oracle(oracle_name)

    alleles = [ref_allele] + list(alt_alleles)
    result = discover_and_report(
        oracle, position, alleles,
        gene_name=gene_name,
        top_n=top_n,
        min_effect=min_effect,
        normalizer=state.get_normalizer(oracle_name),
        oracle_name=oracle_name,
    )

    # Format output
    output = {
        "variant": {"position": position, "ref": ref_allele, "alt": alt_alleles},
        "cell_type_ranking": result["hits"],
        "reports": {},
    }

    for ct_name, report in result.get("reports", {}).items():
        # Attach the user's prompt to each per-cell-type sub-report so the
        # HTML / markdown outputs all carry the original question.
        report.analysis_request = AnalysisRequest(
            user_prompt=user_prompt,
            tool_name="discover_variant_cell_types",
            oracle_name=oracle_name,
            cell_types=[ct_name],
            tracks_requested=f"top tracks for {ct_name}",
        )
        output["reports"][ct_name] = {
            "scores": report.to_dict(),
            "markdown": report.to_markdown(),
        }

    return output


# ── Sequence engineering & batch scoring tools ────────────────────────

@mcp.tool()
@_safe_tool
def analyze_region_swap(
    oracle_name: str,
    region: str,
    replacement_sequence: str,
    assay_ids: list[str],
    gene_name: Optional[str] = None,
    description: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> dict:
    """Replace a genomic region with a custom sequence and score effects across all layers.

    Compares wild-type vs replacement predictions using the same multi-layer
    scoring as variant analysis (chromatin, TF binding, histone, CAGE, RNA).

    Use cases: promoter swaps, enhancer replacements, regulatory element engineering.

    Args:
        oracle_name: A loaded oracle name.
        region: Region to replace as "chr1:1000000-1001000".
        replacement_sequence: DNA sequence to insert in place of the region.
        assay_ids: List of assay identifiers for multi-layer scoring.
        gene_name: Optional gene for expression scoring.
        description: Optional short label for the swap (e.g. "SV40 promoter").
        user_prompt: Original user prompt, rendered at the top of the report.
    """
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.region_swap import analyze_region_swap as _swap

    state = _state()
    oracle = state.get_oracle(oracle_name)

    ar = AnalysisRequest(
        user_prompt=user_prompt,
        tool_name="analyze_region_swap",
        oracle_name=oracle_name,
        tracks_requested=(
            "all oracle tracks" if not assay_ids else f"{len(assay_ids)} tracks"
        ),
        notes=[f"Region swap: {region}" + (f" — {description}" if description else "")],
    )

    report = _swap(
        oracle, region, replacement_sequence, assay_ids,
        gene_name=gene_name,
        normalizer=state.get_normalizer(oracle_name),
        oracle_name=oracle_name,
    )
    report.analysis_request = ar

    result = report.to_dict()
    result["markdown_report"] = report.to_markdown()
    result["analysis_type"] = "region_swap"
    if description:
        result["description"] = description
    if state.output_dir:
        try:
            html_path = report.to_html(output_path=state.output_dir)
            result["html_report_path"] = html_path
        except Exception:
            pass
    return result


@mcp.tool()
@_safe_tool
def simulate_integration(
    oracle_name: str,
    position: str,
    construct_sequence: str,
    assay_ids: list[str],
    gene_name: Optional[str] = None,
    description: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> dict:
    """Simulate inserting a construct at a genomic position and score disruption.

    Compares wild-type vs insertion predictions across all regulatory layers.
    Predicts how a viral vector, transgene cassette, or other construct would
    affect local chromatin, TF binding, and gene expression.

    Args:
        oracle_name: A loaded oracle name.
        position: Insertion point as "chr1:1050000".
        construct_sequence: DNA sequence to insert.
        assay_ids: List of assay identifiers for multi-layer scoring.
        gene_name: Optional gene for expression scoring.
        description: Optional short label (e.g. "AAV-GFP at AAVS1").
        user_prompt: Original user prompt, rendered at the top of the report.
    """
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.integration import simulate_integration as _integrate

    state = _state()
    oracle = state.get_oracle(oracle_name)

    ar = AnalysisRequest(
        user_prompt=user_prompt,
        tool_name="simulate_integration",
        oracle_name=oracle_name,
        tracks_requested=(
            "all oracle tracks" if not assay_ids else f"{len(assay_ids)} tracks"
        ),
        notes=[f"Integration at {position}" + (f" — {description}" if description else "")],
    )

    report = _integrate(
        oracle, position, construct_sequence, assay_ids,
        gene_name=gene_name,
        normalizer=state.get_normalizer(oracle_name),
        oracle_name=oracle_name,
    )
    report.analysis_request = ar

    result = report.to_dict()
    result["markdown_report"] = report.to_markdown()
    result["analysis_type"] = "integration_simulation"
    if description:
        result["description"] = description
    if state.output_dir:
        try:
            html_path = report.to_html(output_path=state.output_dir)
            result["html_report_path"] = html_path
        except Exception:
            pass
    return result


@mcp.tool()
@_safe_tool
def score_variant_batch(
    oracle_name: str,
    variants: list[dict],
    assay_ids: list[str],
    gene_name: Optional[str] = None,
    top_n: int = 20,
    user_prompt: Optional[str] = None,
) -> dict:
    """Score a batch of variants and rank by effect magnitude.

    Processes multiple variants through multi-layer analysis and returns a
    ranked table. Claude can parse VCF content (or any free-text variant
    list) and construct the ``variants`` argument from it.

    **Variant dict schema** — each entry in ``variants`` must be a dict with:

    - ``chrom`` (str): chromosome, e.g. ``"chr1"``
    - ``pos``   (int): 1-based genomic coordinate
    - ``ref``   (str): reference allele, e.g. ``"G"``
    - ``alt``   (str): alternate allele, e.g. ``"T"``
    - ``id``    (str, optional): label, e.g. ``"rs12740374"`` — defaults to
      ``"chrom:pos_ref>alt"`` if omitted

    Example::

        variants = [
            {"chrom": "chr1", "pos": 109274968, "ref": "G", "alt": "T", "id": "rs12740374"},
            {"chrom": "chr1", "pos": 109275684, "ref": "G", "alt": "T", "id": "rs1626484"},
        ]

    Args:
        oracle_name: A loaded oracle name.
        variants: List of variant dicts (schema above).
        assay_ids: Track identifiers to score. Pass an empty list to let
            the oracle score all available tracks (recommended for
            AlphaGenome / Enformer / Borzoi).
        gene_name: Optional gene for expression scoring.
        top_n: Return only the top N variants by effect (default 20).
        user_prompt: Original user prompt, rendered at the top of the report.

    Returns:
        Dict with:
          - ``scores``: list of ranked variant dicts (top_n), each with
            ``variant_id``, ``max_effect``, ``top_layer``, ``top_track``,
            ``per_layer_scores``, and optional ``max_quantile``
          - ``markdown_report``: ready-to-display markdown table
          - ``analysis_request``: metadata (prompt, tool, oracle, normalizer)
    """
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.batch_scoring import score_variant_batch as _batch

    state = _state()
    oracle = state.get_oracle(oracle_name)

    ar = AnalysisRequest(
        user_prompt=user_prompt,
        tool_name="score_variant_batch",
        oracle_name=oracle_name,
        tracks_requested=(
            "all oracle tracks" if not assay_ids else f"{len(assay_ids)} tracks"
        ),
        notes=[f"Scoring {len(variants)} variants"],
    )

    batch_result = _batch(
        oracle, variants, assay_ids,
        gene_name=gene_name,
        normalizer=state.get_normalizer(oracle_name),
        analysis_request=ar,
    )

    # Truncate to top_n
    result = batch_result.to_dict()
    result["scores"] = result["scores"][:top_n]
    result["markdown_report"] = batch_result.to_markdown()
    result["analysis_type"] = "batch_scoring"
    return result


@mcp.tool()
@_safe_tool
def fine_map_causal_variant(
    oracle_name: str,
    lead_variant: str,
    ld_variants: Optional[list[dict]] = None,
    assay_ids: Optional[list[str]] = None,
    gene_name: Optional[str] = None,
    population: str = "CEU",
    r2_threshold: float = 0.8,
    ldlink_token: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> dict:
    """Prioritize causal variants from a GWAS locus using multi-layer regulatory evidence.

    Given a sentinel GWAS variant and its LD proxies, scores each variant
    across all regulatory layers and ranks by a **composite causal score**
    combining four components (each in [0, 1] after min-max normalization):

    1. ``max_effect``    — largest |log2FC| across layers (weight 0.35)
    2. ``n_layers``      — count of layers with effect above threshold (0.25)
    3. ``convergence``   — directional agreement across layers (0.20)
    4. ``ref_activity``  — baseline activity of the variant site (0.20)

    A variant with a *strong effect in many layers, all in the same direction,
    in an already-active region* ends up at the top of the ranking.

    **Two modes:**

    - **Auto-fetch**: provide only ``lead_variant`` + ``ldlink_token`` to
      pull LD proxies from LDlink at the given ``population`` / ``r2_threshold``.
    - **Manual**: provide ``ld_variants`` directly. Each dict needs
      ``chrom``, ``pos``, ``ref``, ``alt``, and optional ``id``, ``r2``.

    **Output (per-variant columns in the ranked table):**

    - ``variant_id`` / rsID (★ marks the sentinel)
    - ``r2`` to the sentinel
    - ``max_effect`` (signed log2FC in the strongest layer)
    - ``n_layers_affected`` — 0 means no layer above threshold
    - ``convergence`` — 1.0 = all effects same sign, 0.0 = split
    - ``composite`` — final ranking score; top row is the most likely
      causal variant

    Use ``result["rankings"][0]["per_layer_scores"]`` to read off *which*
    layers drove the top candidate's score.

    Args:
        oracle_name: A loaded oracle name.
        lead_variant: Sentinel variant as "rs12740374" or "chr1:109274968 G>T".
        ld_variants: Optional list of LD variant dicts (schema above).
            If omitted, auto-fetched from LDlink.
        assay_ids: Track identifiers. Pass an empty list / None to score
            all tracks (recommended for AlphaGenome).
        gene_name: Target gene for expression scoring.
        population: 1000 Genomes population for LD lookup (default CEU).
        r2_threshold: Minimum r² for LD variants (default 0.8).
        ldlink_token: LDlink API token. Register free at
            https://ldlink.nih.gov/?tab=apiaccess
        user_prompt: Original user prompt, rendered at the top of the report.
    """
    from chorus.analysis.causal import prioritize_causal_variants
    from chorus.utils.ld import (
        fetch_ld_variants,
        ld_variants_from_list,
        LDLinkError,
    )

    state = _state()
    oracle = state.get_oracle(oracle_name)

    # Parse lead_variant string
    lead_dict = _parse_lead_variant(lead_variant)

    # Get LD variants
    if ld_variants is not None:
        ld_list = ld_variants_from_list(
            ld_variants,
            sentinel_id=lead_dict.get("id"),
        )
    else:
        try:
            variant_id = lead_dict.get("id", lead_variant.strip())
            ld_list = fetch_ld_variants(
                variant_id,
                population=population,
                r2_threshold=r2_threshold,
                token=ldlink_token,
            )
        except LDLinkError as exc:
            return {"error": str(exc)}

    # When the caller passed an rsID with no coordinates, backfill chrom/pos/
    # ref/alt onto the sentinel from the LDlink-resolved variant list (which
    # always carries them). Without this, prioritize_causal_variants raises
    # KeyError: 'chrom' on lead_dict['chrom'].
    if "chrom" not in lead_dict and ld_list:
        sentinel_entry = next((v for v in ld_list if getattr(v, "is_sentinel", False)), ld_list[0])
        lead_dict.setdefault("chrom", sentinel_entry.chrom)
        lead_dict.setdefault("pos", sentinel_entry.position)
        lead_dict.setdefault("ref", sentinel_entry.ref)
        lead_dict.setdefault("alt", sentinel_entry.alt)

    from chorus.analysis.analysis_request import AnalysisRequest

    ar = AnalysisRequest(
        user_prompt=user_prompt,
        tool_name="fine_map_causal_variant",
        oracle_name=oracle_name,
        tracks_requested=(
            "all oracle tracks" if not assay_ids else f"{len(assay_ids)} tracks"
        ),
        notes=[f"Sentinel {lead_variant}; {len(ld_list)} LD variants (r²≥{r2_threshold})"],
    )

    result = prioritize_causal_variants(
        oracle, lead_dict, ld_list, assay_ids,
        gene_name=gene_name,
        oracle_name=oracle_name,
        normalizer=state.get_normalizer(oracle_name),
        analysis_request=ar,
    )

    output = result.to_dict()
    output["markdown_report"] = result.to_markdown()
    output["analysis_type"] = "causal_prioritization"
    if state.output_dir:
        try:
            html_path = result.to_html(output_path=state.output_dir)
            output["html_report_path"] = html_path
        except Exception:
            pass
    return output


def _parse_lead_variant(text: str) -> dict:
    """Parse lead variant from various formats.

    Accepts:
    - "rs12740374" (rsID only — coordinates must come from LD lookup)
    - "chr1:109274968 G>T"
    - "chr1:109274968 G T"
    """
    text = text.strip()
    parts = text.split()

    if text.startswith("rs"):
        return {"id": text}

    if ":" in parts[0]:
        chrom, pos_str = parts[0].split(":")
        pos = int(pos_str)
        result = {"chrom": chrom, "pos": pos, "id": f"{chrom}:{pos}"}
        if len(parts) >= 2:
            alleles = parts[1] if ">" in parts[1] else " ".join(parts[1:])
            allele_parts = alleles.replace(">", " ").split()
            if len(allele_parts) >= 1:
                result["ref"] = allele_parts[0]
            if len(allele_parts) >= 2:
                result["alt"] = allele_parts[1]
        return result

    return {"id": text}


# ── Prompts ──────────────────────────────────────────────────────────

@mcp.prompt()
def getting_started() -> str:
    """Step-by-step guide for using Chorus genomic oracles."""
    return (
        "You are using Chorus, a unified interface for genomic deep-learning oracles.\n\n"
        "## Getting Started\n\n"
        "1. **Discover oracles**: Call `list_oracles()` to see all 6 available oracles "
        "and which ones have their environments installed.\n\n"
        "2. **Choose an oracle**:\n"
        "   - **AlphaGenome** (recommended): 1Mb window, 5930 tracks, 1bp resolution. Best for variant analysis.\n"
        "   - **Enformer**: 114kb output, 5313 ENCODE tracks. Great general-purpose oracle.\n"
        "   - **Borzoi**: 196kb output at 32bp resolution. Good for distal gene expression.\n"
        "   - **ChromBPNet**: 1bp resolution, 1kb window. Best for motif-level TF binding analysis.\n"
        "   - **Sei**: Regulatory element classification (not per-track signal).\n"
        "   - **LegNet**: MPRA activity prediction for short sequences.\n\n"
        "3. **Find tracks**: Call `list_tracks(oracle_name, query='...')` to search for "
        "relevant assays (e.g. 'DNASE K562', 'CAGE liver', 'GATA1').\n\n"
        "4. **Load an oracle**: Call `load_oracle(oracle_name)`. This takes 30s-5min. "
        "The oracle stays loaded for subsequent calls.\n\n"
        "5. **Analyse a variant** (recommended): Use `analyze_variant_multilayer()` for a "
        "complete multi-layer report with normalization, interpretation labels, and an IGV browser.\n\n"
        "6. **Discover affected cell types**: Use `discover_variant()` to scan ALL tracks and find "
        "which cell types and regulatory layers are most affected — no track pre-selection needed.\n\n"
        "7. **Batch scoring**: Use `score_variant_batch()` to compare multiple variants side-by-side "
        "on specific tracks.\n\n"
        "8. **Fine-mapping**: Use `fine_map_causal_variant()` with an rsID + LDlink token to rank "
        "LD proxies by composite causal evidence.\n\n"
        "9. **Sequence engineering**: Use `analyze_region_swap()` or `simulate_integration()` for "
        "in-silico mutagenesis and transgene insertion analysis.\n\n"
        "## Low-level tools (for custom workflows)\n"
        "- `predict()` — raw oracle prediction on a region\n"
        "- `predict_variant_effect()` — per-track variant effects without normalization\n"
        "- `predict_variant_effect_on_gene()` — fold-change in expression for a specific gene\n"
        "- `predict_region_replacement()` / `predict_region_insertion()` — sequence edits\n\n"
        "## Tips\n"
        "- Positions use `chrN:pos` (e.g. `chr1:109274968`); regions auto-center on the variant\n"
        "- Call `oracle_status()` to see what's currently loaded\n"
        "- Call `unload_oracle(name)` to free memory when done\n"
    )


@mcp.prompt()
def analyze_variant(variant: str, gene: str, cell_type: str = "K562") -> str:
    """Template for a complete variant-to-gene effect analysis.

    Args:
        variant: Variant in 'chrN:pos REF>ALT' format (e.g. 'chr1:109274968 G>T').
        gene: Target gene symbol (e.g. 'SORT1').
        cell_type: Cell type for track selection (e.g. 'K562', 'HepG2').
    """
    return (
        f"Analyse the effect of variant **{variant}** on **{gene}** expression "
        f"in **{cell_type}** cells using the Chorus genomic oracles.\n\n"
        f"## Recommended workflow\n\n"
        f"1. Load AlphaGenome (recommended primary oracle):\n"
        f"   `load_oracle('alphagenome')`\n\n"
        f"2. Search for relevant tracks in {cell_type}:\n"
        f"   `list_tracks('alphagenome', query='{cell_type}')`\n"
        f"   Select DNASE/ATAC (accessibility), H3K27ac (enhancer mark), "
        f"and CAGE (expression) tracks.\n\n"
        f"3. Predict variant effect on gene expression:\n"
        f"   `predict_variant_effect_on_gene(...)` with the variant position, "
        f"alleles, gene name '{gene}', and selected tracks.\n\n"
        f"4. For deeper motif-level analysis at the variant site:\n"
        f"   Load ChromBPNet: `load_oracle('chrombpnet', assay='ATAC', cell_type='{cell_type}')`\n"
        f"   Then `predict_variant_effect(...)` at the variant position.\n\n"
        f"## Interpretation guide\n"
        f"- **Layer 1** (Accessibility): Is the variant in an open chromatin region?\n"
        f"- **Layer 2** (Histone marks): Active enhancer (H3K27ac+) or promoter (H3K4me3+)?\n"
        f"- **Layer 3** (TF binding): Does the variant disrupt or create a TF binding site?\n"
        f"- **Layer 4** (Gene expression): Fold change in CAGE/RNA-seq at {gene} TSS?\n"
        f"- **Layer 5** (Cell-type specificity): Is the effect specific to {cell_type}?\n"
    )


# ── Entry-point ──────────────────────────────────────────────────────

def main():
    """Console-scripts entry-point for ``chorus-mcp``."""
    if "--help" in sys.argv or "-h" in sys.argv:
        print("chorus-mcp — Chorus Genomics MCP Server")
        print()
        print("Starts the Chorus MCP server (Model Context Protocol) for AI")
        print("assistant integration. The server communicates via stdio and")
        print("is normally launched automatically by Claude Code or Claude Desktop.")
        print()
        print("Usage:")
        print("  chorus-mcp          Start the MCP server (stdio transport)")
        print("  chorus-mcp --help   Show this help message")
        print()
        print("Configuration:")
        print("  CHORUS_NO_TIMEOUT=1       Disable prediction timeouts")
        print("  CHORUS_MCP_OUTPUT_DIR=DIR  Set output directory for bedgraph files")
        print()
        print("Tools provided: list_oracles, list_tracks, list_genomes,")
        print("  get_genes_in_region, get_gene_tss, load_oracle, unload_oracle,")
        print("  oracle_status, predict, predict_variant_effect,")
        print("  predict_region_replacement, predict_region_insertion,")
        print("  score_prediction_region, score_variant_effect_at_region,")
        print("  predict_variant_effect_on_gene, analyze_variant_multilayer,")
        print("  discover_variant_cell_types, analyze_region_swap,")
        print("  simulate_integration, score_variant_batch")
        print()
        print("Prompts provided: getting_started, analyze_variant")
        return
    mcp.run()


if __name__ == "__main__":
    main()
