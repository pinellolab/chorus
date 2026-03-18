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
            "cell_types": ["HepG2", "K562"],
            "note": "LegNet predicts lentiMPRA activity. Specify cell_type when loading.",
        }

    return {"error": f"Unknown oracle: {oracle_name}"}


@mcp.tool()
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
def unload_oracle(oracle_name: str) -> dict:
    """Unload an oracle to free memory.

    Args:
        oracle_name: Oracle name to unload.
    """
    removed = _state().unload_oracle(oracle_name)
    return {"name": oracle_name.lower(), "unloaded": removed}


@mcp.tool()
def oracle_status() -> dict:
    """Show which oracles are currently loaded, their device, and load time."""
    return {"loaded_oracles": _state().list_loaded()}


# ── Prediction tools ─────────────────────────────────────────────────

@mcp.tool()
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
    return serialize_prediction(prediction, output_dir=state.output_dir, prefix=f"{oracle_name}_wt_")


@mcp.tool()
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
    return serialize_variant_effect(result, output_dir=state.output_dir)


@mcp.tool()
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
    return serialize_replacement_or_insertion(result, output_dir=state.output_dir, prefix=f"{oracle_name}_repl_")


@mcp.tool()
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
    return serialize_replacement_or_insertion(result, output_dir=state.output_dir, prefix=f"{oracle_name}_ins_")


# ── Scoring & gene expression tools ──────────────────────────────────

@mcp.tool()
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

    return {
        "input_region": region,
        "score_region": score_region,
        "scoring_strategy": scoring_strategy,
        "scores": {k: v for k, v in scores.items()},
    }


@mcp.tool()
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

    return {
        "variant_info": variant_result["variant_info"],
        "scoring_strategy": scoring_strategy,
        "at_variant": at_variant,
        "scores": scores,
    }


@mcp.tool()
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

    return result


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
        "5. **Make predictions**: Use `predict()`, `predict_variant_effect()`, "
        "`predict_region_replacement()`, or `predict_region_insertion()`.\n\n"
        "6. **Analyse gene effects**: Use `predict_variant_effect_on_gene()` to get "
        "fold-change in expression for a specific gene.\n\n"
        "## Tips\n"
        "- Regions use the format `chrN:start-end` (e.g. `chr1:1000000-1393216`)\n"
        "- Positions use `chrN:pos` (e.g. `chr1:1050000`)\n"
        "- The `region` parameter is optional in variant tools - it auto-centers on the variant\n"
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
        print("  predict_variant_effect_on_gene")
        print()
        print("Prompts provided: getting_started, analyze_variant")
        return
    mcp.run()


if __name__ == "__main__":
    main()
