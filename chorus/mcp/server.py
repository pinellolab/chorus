"""Chorus MCP Server — FastMCP tool definitions.

Run standalone:
    fastmcp run chorus/mcp/server.py

Or via the console_scripts entry-point:
    chorus-mcp
"""

import logging
import re
import argparse
import sys
from typing import List, Optional

from fastmcp import FastMCP

from chorus.core.exceptions import InvalidRegionError
from chorus.mcp.state import OracleStateManager
from chorus.mcp.serializers import (
    serialize_prediction,
    serialize_variant_effect,
    serialize_replacement_or_insertion,
)

logger = logging.getLogger(__name__)

_INSTRUCTIONS = (
    "Unified interface for 8 genomic deep-learning oracles "
    "(Enformer, Borzoi, ChromBPNet, Sei, LegNet, Cherimoya/CATv1, EPInformer-seq, "
    "AlphaGenome) — 9 registered names, since AlphaGenome ships two backends. "
    "AlphaGenome "
    "ships with two interchangeable backends — `alphagenome` (JAX, default) "
    "and `alphagenome_pt` (PyTorch, opt-in alternative) — that share the "
    "same model and weights and produce equivalent outputs (1–2 % per-track "
    "fp32 noise). "
    "Discover tracks, load models, make predictions, and analyse variant effects. "
    "Use `recommend_alphagenome_backend` to choose between the JAX and "
    "PyTorch AlphaGenome backends for a given window size. "
    "SCORE EACH VARIANT ONCE: every oracle returns ALL of its tracks (every cell "
    "type, assay, and TF) from a SINGLE forward pass, so one call to "
    "`predict_variant_effect`, `analyze_variant_multilayer`, or `discover_variant` "
    "already covers every track — never loop a scoring tool once per track or per "
    "cell type (that re-runs the model needlessly and can turn minutes into hours). "
    "To score MANY variants (a GWAS credible set, fine-mapping, or a VCF), use the "
    "dedicated multi-variant tools — `fine_map_causal_variant`, `score_variant_batch`, "
    "and `discover_variant_cell_types` — which score each variant exactly once."
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
    "cherimoya": {
        "description": "Cherimoya / CATv1 — base-resolution chromatin accessibility across 1,518 ENCODE experiments",
        "framework": "PyTorch",
        "input_size_bp": 2_114,
        "output_bins": 1_000,
        "resolution_bp": 1,
        "assay_types": ["ATAC", "DNASE"],
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
        "input_size_bp": 200,
        "output_bins": 1,
        "resolution_bp": None,
        "assay_types": ["LentiMPRA"],
    },
    "epinformerseq": {
        "description": (
            "EPInformer-seq — 2114-bp sequence to scalar enhancer activity "
            "(linear max signal over the central 256 bp of the 1024-bp crop), "
            "from a 2-channel per-cell PerCellProfileNetWide (ch0 DNase cut-site, "
            "ch1 H3K27ac coverage) + frozen BiasNet per cell type."
        ),
        "framework": "PyTorch",
        "input_size_bp": 2114,
        "output_bins": 1,
        "resolution_bp": None,
        "assay_types": [
            "Enhancer_DNase",
            "Enhancer_H3K27ac",
            "Enhancer_H3K27ac_DNase",
        ],
    },
    "alphagenome": {
        "description": "AlphaGenome (DeepMind) — 1-bp resolution across 5,731 tracks",
        "framework": "JAX",
        "input_size_bp": 1_048_576,
        "output_bins": 1_048_576,
        "resolution_bp": 1,
        "assay_types": ["DNASE", "ATAC", "CAGE", "CHIP", "RNA", "SPLICE_SITES", "PRO_CAP"],
    },
    "alphagenome_pt": {
        "description": (
            "AlphaGenome (DeepMind) — PyTorch backend. Second of two "
            "interchangeable AlphaGenome oracles (the other is "
            "`alphagenome`, JAX). Same 5,731-track schema, same weights "
            "— `gtca/alphagenome_pytorch` is the official JAX checkpoint "
            "converted to safetensors and produces equivalent outputs "
            "(1–2 % per-track fp32 noise, verified on M3 Ultra + A100). "
            "Differs only in load + forward path. Useful on Apple Silicon "
            "for ≤600 kb windows (5–8× faster than JAX CPU on MPS). On "
            "Linux/CUDA, prefer `alphagenome` (JAX is 1.2–2.8× faster "
            "on A100). See `recommend_alphagenome_backend(window_size_bp)`."
        ),
        "framework": "PyTorch",
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
    """Parse 'chrN:start-end' into (chrom, start, end) with validation.

    Raises ``InvalidRegionError`` (a ``ChorusError`` subclass) so callers
    can handle all Chorus-family errors uniformly. v26 P2 #19.
    """
    m = _REGION_RE.match(region)
    if not m:
        raise InvalidRegionError(
            f"Invalid region format: {region!r}. "
            f"Expected 'chrN:start-end' (e.g. 'chr1:1000000-1393216')."
        )
    chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    if start >= end:
        raise InvalidRegionError(
            f"Invalid region {region!r}: start ({start}) must be less than end ({end})."
        )
    return chrom, start, end


def _parse_position(position: str) -> tuple[str, int]:
    """Parse 'chrN:pos' into (chrom, pos) with validation.

    Raises ``InvalidRegionError`` (a ``ChorusError`` subclass). v26 P2 #19.
    """
    m = _POSITION_RE.match(position)
    if not m:
        raise InvalidRegionError(
            f"Invalid position format: {position!r}. "
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

    Set ``CHORUS_MCP_DEBUG=1`` to include ``"traceback": ...`` in the
    returned dict — useful when debugging a tool call through an MCP
    client that only prints the JSON reply (v26 P1 #18).
    """
    import functools
    import os
    import traceback as _traceback

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.exception("MCP tool %s failed", fn.__name__)
            payload = {
                "error": str(exc) or type(exc).__name__,
                "error_type": type(exc).__name__,
                "tool": fn.__name__,
            }
            if os.environ.get("CHORUS_MCP_DEBUG", "").lower() in ("1", "true", "yes"):
                payload["traceback"] = _traceback.format_exc()
            return payload

    return wrapper


def _describe_tracks_requested(assay_ids, variant_result=None) -> str:
    """Derive a human-readable label for tracks_requested.

    If a single cell type is represented across the returned tracks, include
    it in the label (e.g. "6 HepG2 tracks"). Otherwise fall back to plain
    "N tracks" or "all oracle tracks".
    """
    if not assay_ids:
        return "all oracle tracks"
    cell_types: set[str] = set()
    if variant_result is not None:
        try:
            ref_pred = variant_result.get("predictions", {}).get("reference")
            if ref_pred is not None:
                for tid in assay_ids:
                    track = ref_pred.tracks.get(tid) if hasattr(ref_pred, "tracks") else None
                    ct = getattr(track, "cell_type", None) if track else None
                    if ct:
                        cell_types.add(ct)
        except Exception:
            cell_types = set()
    if len(cell_types) == 1:
        return f"{len(assay_ids)} {cell_types.pop()} tracks"
    return f"{len(assay_ids)} tracks"


def _write_html_report(report, output_dir: str) -> str:
    """Write *report*'s HTML to *output_dir* and return the resulting file path.

    ``report.to_html()`` returns the full HTML string (used elsewhere for
    in-memory rendering), not a path — callers that only need the on-disk
    location must not stash that return value directly into a JSON tool
    result: for reports with large embedded tracks (e.g.
    show_conservation=True) the HTML can run into the tens of MB, which
    breaks the MCP stdio transport if shipped back as a string field.

    Uses ``report.resolve_html_path`` rather than re-deriving the path, so this can
    never disagree with where ``to_html`` actually wrote (e.g. an ``output_dir`` like
    ``/data/run.v2`` that ``Path.suffix`` sees as a file, not a directory).
    """
    report.to_html(output_path=output_dir)
    return str(report.resolve_html_path(output_dir))


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
def recommend_alphagenome_backend(window_size_bp: int) -> dict:
    """Suggest which AlphaGenome backend to use for a given query window size.

    Two AlphaGenome backends ship with chorus: the JAX reference (`alphagenome`,
    default) and the upstream PyTorch port (`alphagenome_pt`, opt-in). Their
    public API is interchangeable — same 5,731-track schema, same predict
    surface — but speed profiles differ by platform and window size:

    - macOS + MPS, ≤600 kb: PyTorch backend wins (5–8× over JAX CPU).
    - macOS + MPS, >600 kb: JAX wins (post a GPU on-die cache cliff at ~768→896 kb).
    - Linux + CUDA, any window: JAX wins (1.2–2.8× over PyTorch on A100).
    - No GPU: JAX wins.

    Args:
        window_size_bp: Centred input window in base pairs (e.g. 524288 for
            a 512 kb query, 1048576 for a 1 MB query).

    Returns:
        Dict with `oracle` (string), `device` (string), `reason` (string),
        `confidence` ("high"/"medium"), and a short `benchmarks` table.
        Suggestion-only — no auto-routing happens.
    """
    from chorus import recommend_alphagenome_backend as _recommend
    return _recommend(window_size_bp)


@mcp.tool()
@_safe_tool
def list_oracles() -> dict:
    """List all genomic oracles (8, plus an alternative PyTorch backend for AlphaGenome = 9 names) with their specs, environment install status, and loaded status.

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


_TRACK_RESULT_CAP = 200


def _track_page(oracle_name: str, query: str, results: list) -> dict:
    """Cap a track search at ``_TRACK_RESULT_CAP`` rows and *say so in the payload*.

    All four search branches used to return ``{"num_results": len(results),
    "tracks": results[:200]}``. ``num_results`` did carry the true count, but a
    caller that read ``tracks`` — which is the field named after the thing it
    wants — saw 200 of AlphaGenome's 1,504 RNA tracks with nothing anywhere in
    the response indicating that 1,304 were dropped. For an MCP tool the caller
    is usually a model, and "the list I was handed is the list that exists" is
    the natural reading.

    So the cap is now explicit: ``truncated`` and ``showing`` are always
    present, and when rows were dropped ``note`` says how to reach them. Same
    class of defect as the reservoir thinning this release fixes — a silent
    subsample presented as the whole population — so it gets the same
    treatment: make the loss visible at the point of loss.
    """
    shown = results[:_TRACK_RESULT_CAP]
    truncated = len(results) > len(shown)
    out = {
        "oracle": oracle_name,
        "query": query,
        "num_results": len(results),
        "showing": len(shown),
        "truncated": truncated,
        "tracks": shown,
    }
    if truncated:
        out["note"] = (
            f"{len(results)} tracks matched; showing the first {len(shown)}. "
            "Narrow the query (e.g. add an assay or cell type) to see the rest — "
            "'num_results' is the full match count."
        )
    return out


@mcp.tool()
@_safe_tool
def list_tracks(oracle_name: str, query: Optional[str] = None) -> dict:
    """List or search available tracks/assays for an oracle.

    Does not require the oracle to be loaded — uses metadata classes.

    Args:
        oracle_name: Oracle name (enformer, borzoi, chrombpnet, cherimoya, sei, legnet, epinformerseq, alphagenome).
        query: Optional search string to filter tracks (e.g. "K562", "DNASE"). Use the returned 'identifier' field as the assay_id for predictions.
    """
    oracle_name = oracle_name.lower()

    # Validate oracle name up front — v26 P2 #20: previous behaviour
    # dropped through to the fall-through error dict, which was easy to
    # miss. Surface the mismatch explicitly with the valid names.
    if oracle_name not in ORACLE_SPECS:
        valid = ", ".join(sorted(ORACLE_SPECS.keys()))
        logger.warning(
            "list_tracks called with unknown oracle %r (valid: %s)",
            oracle_name, valid,
        )
        return {
            "error": f"Unknown oracle: {oracle_name!r}. Valid names: {valid}",
            "oracle": oracle_name,
        }

    # Try Borzoi metadata (richest search)
    if oracle_name == "borzoi":
        from chorus.oracles.borzoi_source.borzoi_metadata import get_metadata
        meta = get_metadata()
        if query:
            df = meta.search_tracks(query)
            results = df.to_dict(orient="records")
            return _track_page(oracle_name, query, results)
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
            return _track_page(oracle_name, query, results)
        return {
            "oracle": oracle_name,
            "assay_types": meta.list_assay_types(),
            "cell_types": meta.list_cell_types(),
            "note": "Use query parameter to search tracks (e.g. query='K562' or query='DNASE:K562'). Use the 'identifier' field as assay_id for predictions.",
        }

    if oracle_name == "cherimoya":
        # 1,518 tracks — search-first, like borzoi/enformer above, rather
        # than enumerating. Track ids are ASSAY:ENCSR (the ENCODE
        # experiment accession), because (assay, biosample) is ambiguous
        # for 1,188 of the 1,518 experiments.
        from chorus.oracles.cherimoya_source.catv1_metadata import get_metadata
        meta = get_metadata()
        if query:
            df = meta.search_tracks(query)
            columns = [
                "track_id", "experiment_accession", "assay", "biosample",
                "biosample_classification", "biosample_summary",
                "profile_pearson", "count_pearson",
            ]
            results = df[columns].to_dict(orient="records")
            return _track_page(oracle_name, query, results)
        return {
            "oracle": oracle_name,
            "assay_types": meta.list_assay_types(),
            "cell_types": meta.list_cell_types(),
            "num_tracks": len(meta.tracks_df),
            "track_summary": meta.get_track_summary(),
            "note": (
                "1,518 tracks — use the query parameter to search by "
                "biosample, assay, or ENCODE accession (e.g. query='K562', "
                "query='T cell', query='ENCSR000EOT'). Use the returned "
                "'track_id' (ASSAY:ENCSR) for predictions, or pass "
                "assay= and cell_type= to load_oracle to get that "
                "biosample's default experiment."
            ),
        }

    if oracle_name == "chrombpnet":
        # This branch read BPNetMetadata directly and reported CHIP_cell_types x CHIP_TFs as
        # 172 x 240 -- implying 41,280 available models when only **744** exist. Same trap as
        # describe_tracks hit (1,268 vs 753), an order of magnitude worse: it invites Claude to request
        # a cell/TF pair with no model behind it. Derived from describe_tracks(), which enumerates
        # through the background builder's own iter_unique_* helpers, so catalogue and reality agree.
        #
        # The payload SHAPE is unchanged -- the ATAC/DNASE/CHIP split is more useful to a caller than a
        # flat list -- but every list is now real, and `num_tracks` plus search are additions.
        from chorus import create_oracle

        records = create_oracle("chrombpnet").describe_tracks()
        if query:
            hits = [r for r in records if r.matches(query)]
            return _track_page(oracle_name, query, [r.as_dict() for r in hits])

        by_assay: dict = {}
        for r in records:
            by_assay.setdefault(r.assay, []).append(r)
        chip = by_assay.get("CHIP", [])
        return {
            "oracle": oracle_name,
            "num_tracks": len(records),
            "assay_types": sorted(by_assay),
            "ATAC_cell_types": sorted({r.cell_type for r in by_assay.get("ATAC", [])}),
            "DNASE_cell_types": sorted({r.cell_type for r in by_assay.get("DNASE", [])}),
            "CHIP_cell_types": sorted({r.cell_type for r in chip}),
            "CHIP_TFs": sorted({r.extra.get("tf") for r in chip if r.extra.get("tf")}),
            "num_chip_models": len(chip),
            "note": (
                f"{len(records)} models exist: {len(records) - len(chip)} accessibility plus "
                f"{len(chip)} CHIP. CHIP is NOT the full cross product of CHIP_cell_types x CHIP_TFs "
                f"-- only the {len(chip)} listed pairs have models. Pass a query to search them, or "
                f"call oracle.describe_tracks() for the exact list."
            ),
        }

    if oracle_name in ("alphagenome", "alphagenome_pt"):
        # Both backends share the same 5,731-track metadata cache; only
        # the load + forward path differs. list_tracks routes through
        # the same metadata module either way.
        from chorus.oracles.alphagenome_source.alphagenome_metadata import get_metadata
        meta = get_metadata()
        if query:
            df = meta.search_tracks(query)
            results = df.to_dict(orient="records")
            return _track_page(oracle_name, query, results)
        return {
            "oracle": oracle_name,
            "assay_types": meta.list_assay_types(),
            "cell_types": meta.list_cell_types(),
            "note": "Use query parameter to search tracks (e.g. query='K562' or query='GATA1'). Use the 'identifier' field as assay_id for predictions.",
        }

    if oracle_name == "sei":
        # Was a hardcoded ["sequence-class"] that never consulted the oracle -- and by 2026-08-16 it
        # was simply false: Sei predicts 21,907 chromatin profiles across 1,176 assay types as well as
        # the 40 projected classes, and since the background rebuild every one of them has a
        # percentile. Derived from describe_tracks() so it cannot go stale again. Payload keys are
        # unchanged; `num_tracks` and the searchable path are additions.
        from chorus import create_oracle

        records = create_oracle("sei").describe_tracks(query=query)
        if query:
            return _track_page(oracle_name, query, [r.as_dict() for r in records])
        assays = sorted({r.assay for r in records if r.assay})
        return {
            "oracle": oracle_name,
            "num_tracks": len(records),
            "assay_types": assays,
            "cell_types": sorted({r.cell_type for r in records if r.cell_type})[:200],
            "note": (
                "Sei predicts 21,907 chromatin profiles (TA# ids, per assay x cell type) plus 40 "
                "projected regulatory sequence classes (CA# ids). Both accept percentiles. Pass a "
                "query to search; cell_types is truncated to 200."
            ),
        }

    if oracle_name == "legnet":
        # The cell list was a third hardcoded copy of LEGNET_AVAILABLE_CELLTYPES (the others being the
        # constant itself and the background builder). Derived now, so there is one source.
        from chorus import create_oracle

        records = create_oracle("legnet").describe_tracks()
        return {
            "oracle": oracle_name,
            "num_tracks": len(records),
            "assay_types": sorted({r.assay for r in records if r.assay}),
            "cell_types": sorted({r.cell_type for r in records if r.cell_type}),
            "note": "LegNet predicts lentiMPRA activity. Specify cell_type when loading.",
        }

    if oracle_name == "epinformerseq":
        # The assay list here was a third hardcoded copy (the constant, the background builder, and
        # this). Derived now; the explanatory note is kept verbatim because it carries real modelling
        # detail a caller cannot infer from the ids.
        from chorus import create_oracle

        records = create_oracle("epinformerseq").describe_tracks()
        if query:
            hits = [r for r in records if r.matches(query)]
            return _track_page(oracle_name, query, [r.as_dict() for r in hits])
        return {
            "oracle": oracle_name,
            "num_tracks": len(records),
            "assay_types": sorted({r.assay for r in records if r.assay}),
            "cell_types": sorted({r.cell_type for r in records if r.cell_type}),
            "note": (
                "EPInformer-seq returns a single scalar per 2114-bp window from a "
                "2-channel per-cell PerCellProfileNetWide (ch0 DNase cut-site, ch1 "
                "H3K27ac coverage) + frozen BiasNet. Assays: 'Enhancer_DNase' (default), "
                "'Enhancer_H3K27ac', 'Enhancer_H3K27ac_DNase' (composite sqrt(D*H)), each "
                "the max over the central 256 bp. Switch cells with load_pretrained_model(cell_type=...)."
            ),
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


def _annotation_page(results: list) -> dict:
    """Cap an annotation listing at ``_TRACK_RESULT_CAP`` rows, same explicit
    truncation convention as :func:`_track_page` (see its docstring for why),
    with field names that fit annotations rather than oracle tracks.
    """
    shown = results[:_TRACK_RESULT_CAP]
    truncated = len(results) > len(shown)
    out = {
        "num_annotations": len(results),
        "showing": len(shown),
        "truncated": truncated,
        "annotations": shown,
    }
    if truncated:
        out["note"] = (
            f"{len(results)} annotations matched; showing the first {len(shown)}. "
            "'num_annotations' is the full count."
        )
    return out


@mcp.tool()
@_safe_tool
def list_annotations() -> dict:
    """List every known annotation: conservation tracks (GPN-Star vertebrate model,
    PhyloP 100-way, PhastCons 100-way), GENCODE gene annotations, and any custom
    annotation registered via `chorus annotation add`.

    Each row reports its reference genome build, download status, and (once
    downloaded) local path. Does not download anything.
    """
    from chorus.utils.annotation_store import get_annotation_store

    entries = get_annotation_store().list_annotations()
    rows = [e.as_dict() for e in entries]
    return _annotation_page(rows)


@mcp.tool()
@_safe_tool
def describe_annotation(annotation_id: str) -> dict:
    """Full metadata for one annotation, including a physically-verified genome
    build for downloaded bigwig-format tracks.

    Args:
        annotation_id: Id from `list_annotations` (e.g. "gpn_star", "gencode_v48_basic").

    Raises if a downloaded bigwig's actual chromosome-1 length doesn't match its
    declared genome build — a mismatch would otherwise return a plausible-looking
    score about the wrong piece of DNA.
    """
    from chorus.utils.annotation_store import get_annotation_store

    entry = get_annotation_store().describe_annotation(annotation_id)
    return entry.as_dict()


@mcp.tool()
@_safe_tool
def download_annotation(annotation_id: str) -> dict:
    """Download (or confirm already-cached) an annotation by id.

    WARNING: some conservation tracks are 7-44 GB; this can be a long-running,
    large download. No-ops if already downloaded.

    Args:
        annotation_id: Id from `list_annotations`.
    """
    from chorus.utils.annotation_store import get_annotation_store

    path = get_annotation_store().download_annotation(annotation_id)
    return {"annotation_id": annotation_id, "path": str(path)}


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
    encode_id: Optional[str] = None,
) -> dict:
    """Load a genomic oracle and its pretrained model (cached for reuse).

    This can take 30 seconds to several minutes depending on the model.

    Args:
        oracle_name: Oracle name (enformer, borzoi, chrombpnet, cherimoya, sei, legnet, epinformerseq, alphagenome).
        device: Device to use — "cpu", "cuda", "cuda:0", etc. None = auto-detect.
        assay: (ChromBPNet/Cherimoya) Assay type — "ATAC", "DNASE", or (ChromBPNet only) "CHIP".
        cell_type: (ChromBPNet/Cherimoya/LegNet) Cell type — e.g. "K562", "HepG2".
        TF: (ChromBPNet CHIP only) Transcription factor — e.g. "GATA1", "CTCF".
        fold: (ChromBPNet/Cherimoya) Cross-validation fold 0-4 (default 0).
        model_type: (ChromBPNet only) Model variant — "chrombpnet", "bias_scaled", "chrombpnet_nobias".
        encode_id: (Cherimoya only) ENCODE experiment accession, e.g. "ENCSR000EOT".
            Pins one specific CATv1 experiment. Use this with the
            'experiment_accession' returned by list_tracks: (assay, cell_type)
            is ambiguous for most biosamples — K562 alone has 4 ATAC
            experiments — and resolves to a committed default, so passing
            cell_type only is not enough to reach a specific track.
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
    if encode_id:
        kwargs["encode_id"] = encode_id
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

    One forward pass scores the variant across every track in ``assay_ids``
    (or ALL of the oracle's tracks when ``assay_ids`` is omitted), so request
    the full set in a single call — never loop this tool once per track or per
    cell type. To score many variants, use ``score_variant_batch`` or
    ``fine_map_causal_variant`` (each scores a variant exactly once).

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

    # A null score must explain itself. This tool exists to "score a sub-region within
    # the output window", and it returned a well-formed response with `scores:
    # {track: None}` for every sub-region of a LegNet prediction -- no error, no note.
    # An agent reads a populated `scores` key as success and proceeds on nothing.
    #
    # The cause is a geometry inconsistency, not a coding slip: LegNet declares
    # resolution=50 over a 200 bp interval (implying 4 bins) while `values` holds a
    # single scalar. region_bin_span computes bins 1..3 for a 100 bp sub-region, clamps
    # end_bin to len(values)=1, and returns None. Only the full window scores, because
    # only it maps to bin 0. Naming the arithmetic is what makes the answer actionable.
    notes = {}
    for assay_id, val in scores.items():
        if val is not None:
            continue
        try:
            track = prediction[assay_id]
            iv = track.prediction_interval.reference
            n_vals = len(track.values)
            implied = max((iv.end - iv.start) // max(track.resolution, 1), 1)
            if sc_chrom != iv.chrom or sc_end <= iv.start or sc_start >= iv.end:
                notes[assay_id] = (
                    f"score_region {score_region} does not overlap the prediction "
                    f"window {iv.chrom}:{iv.start}-{iv.end}")
            elif n_vals != implied:
                notes[assay_id] = (
                    f"no score: this track carries {n_vals} value(s) but its declared "
                    f"resolution ({track.resolution} bp over "
                    f"{iv.end - iv.start} bp) implies {implied} bins, so a sub-region "
                    f"maps outside the values array. Score the full window "
                    f"{iv.chrom}:{iv.start}-{iv.end} instead.")
            else:
                notes[assay_id] = (
                    f"no score: score_region spans fewer than one {track.resolution} bp "
                    f"bin. Widen it to at least {track.resolution} bp.")
        except Exception:                                    # never let a note raise
            notes[assay_id] = "no score, and the reason could not be determined"
    if notes:
        result["score_notes"] = notes

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

    # Same rule as score_prediction_region: an all-null score must say why. This tool
    # returned {"ref_score": None, "alt_score": None, "effect": None} for LegNet in BOTH
    # modes -- at_variant and score_region -- with no error field, which reads as success.
    #
    # The note must also stay silent when there IS a score, and for one release it did
    # not: `_score_ve` returns {allele: {assay_id: {ref,alt,effect}}} (result.py) and the
    # lookup below was `scores_obj.get(aid)` at the TOP level, i.e. against allele names.
    # It never hit, `flat` was always empty, and the guard fell through -- so every call
    # carried "no score: the scored slice spans fewer than one 128 bp bin" next to
    # Enformer's ref 2.0686 / alt 2.4550 / effect 0.3864 (audit 2026-08-09). Iterate the
    # allele dicts, and derive the assay ids from the payload rather than from the
    # request, because an oracle may rename them (ChromBPNet answers a requested "ATAC"
    # as "ATAC:K562", see base.py).
    def _scored_assay_ids(scores_obj) -> list:
        aids: dict = {}
        if isinstance(scores_obj, dict):
            for per_assay in scores_obj.values():
                if isinstance(per_assay, dict):
                    aids.update({aid: None for aid in per_assay})
        return list(aids)

    def _null_note(scores_obj) -> dict:
        notes: dict = {}
        try:
            ref_pred = variant_result.get("predictions", {}).get("reference")
            var_chrom, var_pos = _parse_position(position)
            for aid in _scored_assay_ids(scores_obj):
                flat = []
                for per_assay in scores_obj.values():
                    vals = per_assay.get(aid) if isinstance(per_assay, dict) else None
                    if isinstance(vals, dict):
                        flat.extend(vals.values())
                    elif vals is not None:
                        flat.append(vals)
                if flat and not all(x is None for x in flat):
                    continue
                if ref_pred is None or aid not in ref_pred:
                    continue
                tr = ref_pred[aid]
                iv = tr.prediction_interval.reference
                n_vals, res = len(tr.values), max(tr.resolution, 1)
                implied = max((iv.end - iv.start) // res, 1)
                if score_region is not None and (
                        sc_chrom != iv.chrom or sc_end <= iv.start or sc_start >= iv.end):
                    # The branch score_prediction_region already had: a null that is
                    # only null because the caller asked about somewhere else.
                    notes[aid] = (
                        f"score_region {score_region} does not overlap the prediction "
                        f"window {iv.chrom}:{iv.start}-{iv.end}")
                elif n_vals != implied:
                    # Ordered BEFORE the at_variant branch below, because a track
                    # whose declared resolution overstates its sampling also makes
                    # pos2bin return None (it bounds-checks the derived index, see
                    # result.py), for a variant that is plainly inside the window.
                    # Blaming that null on the variant's position names the wrong
                    # cause and reads as a contradiction: measured on a 1-value
                    # track declared at 128 bp over 640 bp, the note said
                    # "chr1:1000300 maps outside chr1:1000000-1000640".
                    notes[aid] = (
                        f"no score: this track carries {n_vals} value(s) but its "
                        f"declared resolution ({res} bp over {iv.end - iv.start} bp) "
                        f"implies {implied} bins, so the scored slice falls outside the "
                        f"values array. This oracle has no positional resolution to "
                        f"slice; score the whole window instead.")
                elif at_variant and tr.pos2bin(var_chrom, var_pos) is None:
                    notes[aid] = (
                        f"no score: variant position {position} maps outside this "
                        f"track's prediction window {iv.chrom}:{iv.start}-{iv.end}")
                else:
                    notes[aid] = (
                        f"no score: the scored slice spans fewer than one {res} bp bin")
        except Exception:
            pass
        return notes

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
            # Same {allele: {assay_id: ...}} shape as above, and the same slip: this
            # iterated allele names and asked each per-assay dict for a "reference"
            # key, so ref_val was always None and `ref_activity_percentiles` never
            # appeared. The ref score is identical across alleles -- take the first
            # allele that carries the assay.
            for assay_id in _scored_assay_ids(scores):
                ref_val = None
                for per_assay in scores.values():
                    vals = per_assay.get(assay_id) if isinstance(per_assay, dict) else None
                    if isinstance(vals, dict) and vals.get("ref_score") is not None:
                        ref_val = vals["ref_score"]
                        break
                if ref_val is not None:
                    # OraclePrediction has no .get(); membership + [] is the accessor.
                    track = ref_pred[assay_id] if assay_id in ref_pred else None
                    if track is not None:
                        layer = classify_track_layer(track)
                        if isinstance(normalizer, PerTrackNormalizer):
                            pctile = normalizer.activity_percentile(oracle_name, assay_id, ref_val)
                        else:
                            pctile = normalizer.normalize_baseline(oracle_name, layer, ref_val)
                        if pctile is not None:
                            percentiles[assay_id] = round(pctile, 4)
            if percentiles:
                result["ref_activity_percentiles"] = percentiles

    _notes = _null_note(scores)
    if _notes:
        result["score_notes"] = _notes

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
    show_conservation: bool = False,
    ldlink_token: Optional[str] = None,
    genome_build: str = "grch38",
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
        position: Variant position as ``"chr1:1050000"`` OR an rsID
                  like ``"rs12740374"`` (requires LDlink token; resolves
                  to coords via the LDproxy API).
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
        show_conservation: When True, adds conservation tracks to the IGV
                 browser: two GPN-Star tracks (coverage + sequence logo,
                 vertebrate-alignment model — GPN-Star also ships
                 mammalian and primate variants, not used here) showing a
                 fixed clip(1 - entropy, 0, 1) conservation score (most
                 conserved = highest value/tallest per-base letters when
                 zoomed in below 2bp/pixel), and raw PhyloP 100-way /
                 PhastCons 100-way coverage tracks from UCSC (same
                 100-way vertebrate alignment) — all capped to a bounded
                 window around the variant. **hg38 only** — a non-hg38 report
                 raises. Downloads bigwigs on first use: ~25 GB for the three
                 coverage sources, ~45 GB more of per-allele LLR files for the
                 sequence-logo track, so budget ~70 GB.
        ldlink_token: LDlink API token (only used when ``position`` is an
                  rsID). Register free at https://ldlink.nih.gov/?tab=apiaccess
                  or set ``LDLINK_TOKEN``.
        genome_build: Reference build for the rsID lookup
                  (``"grch38"`` / ``"hg38"`` default; or ``"grch37"`` / ``"hg19"``).
                  Only used when ``position`` is an rsID.
        user_prompt: The user's original natural-language question. Claude
                     should forward this verbatim whenever calling from an
                     MCP conversation — it is rendered at the top of the
                     report for traceability.
    """
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.variant_report import build_variant_report

    state = _state()
    oracle = state.get_oracle(oracle_name)

    # Resolve an rsID to chr:pos via LDlink before downstream parsing.
    if position.strip().startswith("rs"):
        position, _ref_from_ld, _alt_from_ld = _resolve_rsid_to_position(
            position.strip(), ldlink_token=ldlink_token,
            genome_build=genome_build,
        )

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
        tracks_requested=_describe_tracks_requested(assay_ids, variant_result),
    )

    report = build_variant_report(
        variant_result,
        oracle_name=oracle_name,
        gene_name=gene_name,
        normalizer=state.get_normalizer(oracle_name),
        igv_raw=igv_raw,
        show_conservation=show_conservation,
        analysis_request=analysis_request,
    )

    result = report.to_dict()
    result["markdown_report"] = report.to_markdown()

    # Save HTML report to output directory
    if state.output_dir:
        try:
            result["html_report_path"] = _write_html_report(report, state.output_dir)
        except Exception:
            pass  # HTML generation is optional

    return result


@mcp.tool()
@_safe_tool
def score_ism(
    oracle_name: str,
    center: str,
    assay_ids: list[str],
    window: int = 25,
    genome: Optional[str] = None,
) -> dict:
    """In-silico saturation mutagenesis (ISM) around a variant.

    Sweeps every single-base substitution in a ``window``-bp window centred on
    ``center`` and scores the variant effect on ``assay_ids[0]``, returning a
    per-position importance profile (the reference base's disruption) suitable
    for a motif logo. Reveals which bases the oracle actually reads — e.g. a
    disrupted TF motif. Works with any loaded oracle (AlphaGenome, ChromBPNet,
    LegNet, Borzoi, EPInformer-seq, ...).

    Args:
        oracle_name: A loaded oracle name.
        center: Variant / motif centre as ``"chrom:pos"`` (1-based).
        assay_ids: Track id(s) to score; the first drives the importance profile.
        window: Window size in bp (default 25).
        genome: Reference FASTA path; defaults to the oracle's ``reference_fasta``.

    Returns:
        Dict with ``ref_seq``, ``positions``, ``scores`` ([W][4] signed log2FC),
        ``importance`` ([W]), ``assay_id``, ``window`` — render as a motif logo —
        plus ``n_attempted``/``n_scored``/``n_failed``/``first_error``. A cell that
        could not be scored is ``null``, never 0.0 (0.0 means "no effect"), so
        check ``n_failed`` before reading a flat position as uninformative. If
        nothing scored at all, an ``error``/``error_type`` dict is returned with
        no ``scores`` or ``importance`` key.
    """
    from chorus.analysis.saturation import saturation_mutagenesis
    state = _state()
    oracle = state.get_oracle(oracle_name)
    g = genome or getattr(oracle, "reference_fasta", None)
    return saturation_mutagenesis(
        oracle, oracle_name, center, assay_ids, genome=g, window=window,
    )


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
    ranking_metric: str = "alt_x_abs_effect",
    min_ref_value: float = 0.0,
    ldlink_token: Optional[str] = None,
    genome_build: str = "grch38",
) -> dict:
    """Discover which cell types and regulatory layers are most affected by a variant.

    Predicts variant effect across ALL available tracks (thousands for
    Enformer/Borzoi/AlphaGenome, or iterates all models for ChromBPNet/LegNet),
    ranks by ``ranking_metric``, and returns the top hits with a full
    report.

    This is the primary tool for variant interpretation — it tells you WHERE
    the variant has impact without requiring you to pre-select tracks.

    Args:
        oracle_name: A loaded oracle name.
        position: Variant position as ``"chr1:109274968"`` OR an rsID
            like ``"rs12740374"`` (requires LDlink token; resolves
            to coords via the LDproxy API).
        ref_allele: Reference allele (e.g. "G").
        alt_alleles: Alternate alleles (e.g. ["T"]).
        gene_name: Optional gene for expression analysis.
        top_n: Number of top tracks per regulatory layer to show.
        user_prompt: Original user prompt, forwarded into the report header.
        ranking_metric: How to rank tracks and cell types. Default
            ``"alt_x_abs_effect"`` (``alt_value × |log2FC|``) avoids
            inflating closed-baseline tracks where a small absolute
            change yields a large fold-change. Use ``"abs_effect"`` for
            the historical raw |log2FC| ranking.
        min_ref_value: Threshold used by ``"abs_effect_min_ref"``.
        ldlink_token: LDlink API token (only used when ``position`` is
            an rsID). Register free at https://ldlink.nih.gov/?tab=apiaccess.
        genome_build: Reference build for the rsID lookup
            (``"grch38"`` / ``"hg38"`` default; or ``"grch37"`` / ``"hg19"``).
    """
    from chorus.analysis.analysis_request import AnalysisRequest
    from chorus.analysis.discovery import discover_variant_effects

    state = _state()
    oracle = state.get_oracle(oracle_name)
    normalizer = state.get_normalizer(oracle_name)

    if position.strip().startswith("rs"):
        position, _ref_from_ld, _alt_from_ld = _resolve_rsid_to_position(
            position.strip(), ldlink_token=ldlink_token,
            genome_build=genome_build,
        )

    # Built up front and passed into discover_variant_effects so the report renders
    # with the user prompt already on it -- setting report.analysis_request after the
    # call, then writing the HTML a second time, rendered (and wrote, when
    # show_conservation makes that expensive) the same report twice for no
    # difference in the end result.
    ar = AnalysisRequest(
        user_prompt=user_prompt,
        tool_name="discover_variant",
        oracle_name=oracle_name,
        tracks_requested="all oracle tracks",
    )

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
        analysis_request=ar,
        ranking_metric=ranking_metric,
        min_ref_value=min_ref_value,
    )

    # Serialize: extract report as markdown, remove non-serializable VariantReport
    report = result.pop("report", None)
    if report is not None:
        result["markdown_report"] = report.to_markdown()
        if state.output_dir:
            try:
                result["html_report_path"] = str(report.resolve_html_path(state.output_dir))
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
    ranking_metric: str = "alt_x_abs_effect",
    min_ref_value: float = 0.0,
    ldlink_token: Optional[str] = None,
    genome_build: str = "grch38",
) -> dict:
    """Discovery mode: find which cell types are most affected by a variant.

    Use this when you don't know which cell type is relevant — let the model
    tell you where the variant matters most.

    **Two-stage analysis:**
    1. Screens all DNASE/ATAC tracks (~472 cell types on AlphaGenome, ~638
       on Enformer) to rank cell types by ``ranking_metric``.
    2. For each top cell type, runs full multi-layer analysis (chromatin,
       TF, histone, CAGE, RNA) limited to that cell type's tracks.

    **Runtime expectations** (AlphaGenome, single A100):
      - Stage 1 screen: ~30–60 s
      - Stage 2 per-cell-type analysis: ~30 s × ``top_n``
      - Typical end-to-end with default ``top_n=5``: 3–4 minutes

    Args:
        oracle_name: A loaded oracle name (ideally AlphaGenome for broadest
            cell-type coverage).
        position: Variant position as ``"chr1:1050000"`` OR an rsID
            like ``"rs12740374"`` (requires LDlink token; resolves to
            coords via the LDproxy API).
        ref_allele: Reference allele.
        alt_alleles: Alternate alleles.
        gene_name: Optional gene to focus expression analysis on.
        top_n: Number of top cell types to analyze in detail (default 5).
        min_effect: Minimum |log2FC| in DNASE/ATAC to consider a cell type
            hit (default 0.15).
        user_prompt: Original user prompt, forwarded into each sub-report.
        ranking_metric: How to rank cell types. Default
            ``"alt_x_abs_effect"`` (recommended) ranks by
            ``alt_signal × |log2FC|``, which surfaces the cell type where
            the post-variant element is most active. Use ``"abs_effect"``
            to reproduce the historical |log2FC|-only ranking (which has a
            bias toward closed-baseline cell types — see
            :func:`chorus.analysis.discovery.discover_cell_types`).
            Use ``"abs_effect_min_ref"`` with a non-zero ``min_ref_value``
            to rank by |log2FC| while excluding closed baselines.
        min_ref_value: Threshold for ``"abs_effect_min_ref"`` (ignored
            otherwise). Default 0 = no filter.
        ldlink_token: LDlink API token (only used when ``position`` is
            an rsID).
        genome_build: Reference build for the rsID lookup
            (``"grch38"`` / ``"hg38"`` default; or ``"grch37"`` / ``"hg19"``).

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

    if position.strip().startswith("rs"):
        position, _ref_from_ld, _alt_from_ld = _resolve_rsid_to_position(
            position.strip(), ldlink_token=ldlink_token,
            genome_build=genome_build,
        )

    alleles = [ref_allele] + list(alt_alleles)
    result = discover_and_report(
        oracle, position, alleles,
        gene_name=gene_name,
        top_n=top_n,
        min_effect=min_effect,
        normalizer=state.get_normalizer(oracle_name),
        oracle_name=oracle_name,
        ranking_metric=ranking_metric,
        min_ref_value=min_ref_value,
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
            _describe_tracks_requested(assay_ids)
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
            result["html_report_path"] = _write_html_report(report, state.output_dir)
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
            _describe_tracks_requested(assay_ids)
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
            result["html_report_path"] = _write_html_report(report, state.output_dir)
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
            _describe_tracks_requested(assay_ids)
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
    ldlink_timeout: float = 30.0,
    genome_build: str = "grch38",
    snvs_only: bool = False,
    user_prompt: Optional[str] = None,
    report_top_n: int = 3,
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
        ldlink_timeout: HTTP timeout in seconds for the LDlink request
            (default 30). Increase for slow networks or large LD blocks.
        genome_build: Reference build for the LDlink LD lookup. Accepts
            ``"grch38"`` / ``"hg38"`` (default) or ``"grch37"`` / ``"hg19"``.
        snvs_only: When True, restrict scoring to single-nucleotide
            substitutions (drops insertions, deletions, MNVs, and
            complex multi-base proxies). Default False — indels and
            multi-allelic LDlink rows are scored by default as of
            v0.5.5. Set True to reproduce pre-v0.5.5 behavior.
        user_prompt: Original user prompt, rendered at the top of the report.
        report_top_n: Number of top-ranked variants to render full IGV signal
            tracks for in the HTML report (default 3). Every variant is
            scored with the same per-track logic; this only controls how many
            of the top hits get the (more expensive) interactive browser
            tracks. Set to 0 to skip the IGV signal tracks entirely.
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
                timeout=ldlink_timeout,
                genome_build=genome_build,
                snvs_only=snvs_only,
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
            _describe_tracks_requested(assay_ids)
        ),
        notes=[f"Sentinel {lead_variant}; {len(ld_list)} LD variants (r²≥{r2_threshold})"],
    )

    result = prioritize_causal_variants(
        oracle, lead_dict, ld_list, assay_ids,
        gene_name=gene_name,
        oracle_name=oracle_name,
        normalizer=state.get_normalizer(oracle_name),
        analysis_request=ar,
        snvs_only=snvs_only,
        report_top_n=report_top_n,
    )

    output = result.to_dict()
    output["markdown_report"] = result.to_markdown()
    output["analysis_type"] = "causal_prioritization"
    if state.output_dir:
        try:
            output["html_report_path"] = _write_html_report(result, state.output_dir)
        except Exception:
            pass
    return output


def _resolve_rsid_to_position(
    rsid: str,
    ldlink_token: Optional[str] = None,
    timeout: float = 30.0,
    genome_build: str = "grch38",
) -> tuple[str, str, str]:
    """Resolve an rsID to ``(position_str, ref_allele, alt_allele)`` via LDlink.

    Used by ``analyze_variant_multilayer`` / ``discover_variant`` /
    ``discover_variant_cell_types`` so that callers can pass an rsID
    where those tools previously required ``chr:pos``. Hits the LDlink
    LDproxy API with ``r2_threshold=1.1`` so only the sentinel line is
    returned (cheap one-record lookup, not a full LD proxy fetch).

    Raises ``ValueError`` on missing alleles or LDlink failure with a
    fix-it pointer.
    """
    from chorus.utils.ld import fetch_ld_variants, LDLinkError
    try:
        ld = fetch_ld_variants(
            rsid,
            r2_threshold=1.1,  # keep only the sentinel row (always returned)
            token=ldlink_token,
            timeout=timeout,
            genome_build=genome_build,
        )
    except LDLinkError as exc:
        raise ValueError(
            f"Could not resolve {rsid!r} via LDlink: {exc}. "
            f"Pass coordinates directly as 'chr1:109274968' if LDlink is "
            f"unavailable, or set LDLINK_TOKEN."
        ) from exc
    if not ld:
        raise ValueError(
            f"LDlink returned no record for {rsid!r}. Check that the rsID "
            f"is valid and present in genome_build={genome_build!r}."
        )
    v = ld[0]  # sentinel
    if not v.ref or not v.alt:
        raise ValueError(
            f"LDlink returned {rsid!r} at {v.chrom}:{v.pos} but no alleles. "
            f"Pass coordinates + alleles directly."
        )
    return f"{v.chrom}:{v.pos}", v.ref, v.alt


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
        "1. **Discover oracles**: Call `list_oracles()` to see all 7 available oracles "
        "and which ones have their environments installed.\n\n"
        "2. **Choose an oracle**:\n"
        "   - **AlphaGenome** (recommended): 1Mb window, 5,731 tracks, 1bp resolution. Best for variant analysis.\n"
        "   - **Enformer**: 114kb output, 5,313 ENCODE tracks. Great general-purpose oracle.\n"
        "   - **Borzoi**: 196kb output at 32bp resolution, 7,611 tracks. Good for distal gene expression.\n"
        "   - **ChromBPNet**: 1bp resolution, 1kb window. Best for motif-level TF binding analysis.\n"
        "   - **Sei**: Regulatory element classification (not per-track signal).\n"
        "   - **LegNet**: MPRA activity prediction for short sequences.\n"
        "   - **EPInformer-seq**: scalar enhancer activity from 2,114bp sequence (per-cell DNase + H3K27ac).\n\n"
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

def registered_tool_names() -> List[str]:
    """The tool names FastMCP actually registered, read from the server object.

    Asking `mcp` rather than maintaining a list. The hand-written list this replaces had drifted
    twice: a v27 audit found it missing `discover_variant` and `fine_map_causal_variant`, and by
    v0.7.3 it still said "Tools provided (22)" while 24 were registered (`recommend_alphagenome_backend`
    and `score_ism` were absent). A count that is derived cannot disagree with the code.
    """
    import asyncio

    try:
        tools = asyncio.run(mcp.list_tools())
    except Exception:                      # no sync accessor in FastMCP 3.x; never break --help
        return []
    return sorted(getattr(t, "name", str(t)) for t in tools)


def main(argv: Optional[List[str]] = None):
    """Console-scripts entry-point for ``chorus-mcp``."""
    names = registered_tool_names()
    tool_lines = "\n".join(f"  {n}" for n in names) or "  (none discovered)"

    parser = argparse.ArgumentParser(
        prog="chorus-mcp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Chorus Genomics MCP Server (Model Context Protocol).\n\n"
            "Communicates over stdio and is normally launched for you by Claude Code or Claude\n"
            "Desktop rather than run by hand. It takes no options: transport, host and port come\n"
            "from FastMCP's own settings (see below)."
        ),
        epilog=(
            f"Tools registered ({len(names)}):\n{tool_lines}\n\n"
            "Prompts registered: getting_started, analyze_variant\n\n"
            "Environment:\n"
            "  CHORUS_NO_TIMEOUT=1        Disable prediction AND model-load timeouts\n"
            "                             (read in chorus/core/base.py, not in this module)\n"
            "  CHORUS_MCP_OUTPUT_DIR=DIR  Where bedgraphs and reports are written. Defaults to\n"
            "                             ./chorus_mcp_output relative to the *client's* working\n"
            "                             directory, captured when the server starts\n"
            "  CHORUS_MCP_DEBUG=1         Include a traceback in error payloads\n"
            "  FASTMCP_TRANSPORT=...      FastMCP reads its own FASTMCP_*-prefixed settings (and a\n"
            "                             .env file), so this can switch the transport away from\n"
            "                             stdio; FASTMCP_HOST/FASTMCP_PORT default to\n"
            "                             127.0.0.1:8000 and apply only to a network transport\n"
        ),
    )
    # No flags on purpose — but parse anyway, so `chorus-mcp --port 9000` fails loudly instead of
    # being swallowed by a `"--help" in sys.argv` check and silently starting a stdio server.
    parser.parse_args(argv)
    mcp.run()


if __name__ == "__main__":
    main()
