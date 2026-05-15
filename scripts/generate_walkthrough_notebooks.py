"""Generate one reproduction Jupyter notebook per walkthrough.

Run inside the chorus base env (``mamba run -n chorus``); the
generated notebooks themselves can be executed in the same base env
because every oracle call uses ``use_environment=True`` and the
subprocess delegation handles the per-oracle env automatically.

Usage::

    mamba run -n chorus python scripts/generate_walkthrough_notebooks.py

Each walkthrough directory ends up with a ``notebook.ipynb`` whose
top-to-bottom execution reproduces the same artifacts the matching
MCP tool would (``example_output.md`` / ``.json`` / ``.tsv`` and the
walkthrough's HTML report).

Notebook contract (see plan v0.5.6):

* imports cell (no later imports)
* assume chorus is installed
* one logical step per cell with a leading ``#`` comment
* all function arguments explicit; no implicit defaults
* dedicated save cell(s) at the bottom
* top-to-bottom run reproduces the MCP output
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Walkthrough specs
#
# One dict per walkthrough. ``mcp_tool`` selects which template-builder
# function runs; ``oracle`` selects the env. Every other field is forwarded
# verbatim into the generated notebook so the notebook's arguments are a
# 1:1 image of the MCP arguments documented in the walkthrough's README.
# ---------------------------------------------------------------------------

# AlphaGenome HepG2 track IDs reused by several walkthroughs.
_AG_HEPG2 = [
    "DNASE/EFO:0001187 DNase-seq/.",
    "ATAC/EFO:0001187 ATAC-seq/.",
    "CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA genetically modified (insertion) using CRISPR targeting H. sapiens CEBPA/.",
    "CHIP_TF/EFO:0001187 TF ChIP-seq CEBPB/.",
    "CHIP_HISTONE/EFO:0001187 Histone ChIP-seq H3K27ac/.",
    "CAGE/hCAGE EFO:0001187/+",
    "CAGE/hCAGE EFO:0001187/-",
]

# K562 track IDs (sequence-engineering walkthroughs).
_AG_K562_SWAP = [
    "DNASE/EFO:0002067 DNase-seq/.",
    "CHIP_HISTONE/EFO:0002067 Histone ChIP-seq H3K27ac/.",
    "CHIP_HISTONE/EFO:0002067 Histone ChIP-seq H3K4me3/.",
    "CAGE/hCAGE EFO:0002067/+",
]
_AG_K562_INTEG = [
    "DNASE/EFO:0002067 DNase-seq/.",
    "CHIP_HISTONE/EFO:0002067 Histone ChIP-seq H3K27ac/.",
    "CAGE/hCAGE EFO:0002067/+",
]

# BCL11A walkthrough uses erythroid K562 tracks.
_AG_BCL11A = [
    "DNASE/EFO:0002067 DNase-seq/.",
    "CHIP_TF/EFO:0002067 TF ChIP-seq GATA1/.",
    "CHIP_TF/EFO:0002067 TF ChIP-seq TAL1/.",
    "CHIP_HISTONE/EFO:0002067 Histone ChIP-seq H3K27ac/.",
    "CAGE/hCAGE EFO:0002067/+",
    "CAGE/hCAGE EFO:0002067/-",
]

# Pre-fetched SORT1 LD proxies (chorus 0.5.4 LDlink fetch, CEU r²≥0.85).
# Inlined here so the causal-prioritization notebook runs offline.
_SORT1_LD = [
    {"id": "rs12740374", "chrom": "chr1", "pos": 109274968, "ref": "G", "alt": "T", "r2": 1.0},
    {"id": "rs4970836",  "chrom": "chr1", "pos": 109279175, "ref": "G", "alt": "A", "r2": 0.907},
    {"id": "rs1624712",  "chrom": "chr1", "pos": 109275908, "ref": "C", "alt": "T", "r2": 1.0},
    {"id": "rs660240",   "chrom": "chr1", "pos": 109275216, "ref": "T", "alt": "C", "r2": 0.9509},
    {"id": "rs142678968","chrom": "chr1", "pos": 109275536, "ref": "C", "alt": "T", "r2": 0.9509},
    {"id": "rs1626484",  "chrom": "chr1", "pos": 109275684, "ref": "G", "alt": "T", "r2": 1.0},
    {"id": "rs7528419",  "chrom": "chr1", "pos": 109274570, "ref": "A", "alt": "G", "r2": 1.0},
    {"id": "rs56960352", "chrom": "chr1", "pos": 109278685, "ref": "G", "alt": "T", "r2": 0.907},
    {"id": "rs1277930",  "chrom": "chr1", "pos": 109279521, "ref": "G", "alt": "A", "r2": 0.907},
    {"id": "rs599839",   "chrom": "chr1", "pos": 109279544, "ref": "G", "alt": "A", "r2": 0.907},
    {"id": "rs602633",   "chrom": "chr1", "pos": 109278889, "ref": "T", "alt": "G", "r2": 0.8582},
]

_SORT1_CAUSAL_TRACKS = [
    "DNASE/EFO:0001187 DNase-seq/.",
    "CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA genetically modified (insertion) using CRISPR targeting H. sapiens CEBPA/.",
    "CHIP_TF/EFO:0001187 TF ChIP-seq CEBPB/.",
    "CHIP_HISTONE/EFO:0001187 Histone ChIP-seq H3K27ac/.",
    "CAGE/hCAGE EFO:0001187/+",
    "CAGE/hCAGE EFO:0001187/-",
]

# Batch-scoring walkthrough variants (5 SORT1-locus SNPs).
_BATCH_VARIANTS = [
    {"chrom": "chr1", "pos": 109274968, "ref": "G", "alt": "T", "id": "rs12740374"},
    {"chrom": "chr1", "pos": 109275684, "ref": "G", "alt": "T", "id": "rs1626484"},
    {"chrom": "chr1", "pos": 109275216, "ref": "T", "alt": "C", "id": "rs660240"},
    {"chrom": "chr1", "pos": 109279175, "ref": "G", "alt": "A", "id": "rs4970836"},
    {"chrom": "chr1", "pos": 109274570, "ref": "A", "alt": "G", "id": "rs7528419"},
]

# Region-swap walkthrough — 630 bp GFP-like construct (from
# regenerate_remaining_examples.py:275-286, verbatim).
_REGION_SWAP_REPLACEMENT = (
    "GCCACCATGGCCACCATGGCCACCATGGCCACCATGGCCACCATGGCCACCATGGCCACCATG"
    "CGAATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGAC"
    "GGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGC"
    "AAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTG"
    "ACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGAC"
    "TTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGAC"
    "GGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAG"
    "CTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTAC"
    "AACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAG"
    "ATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCC"
)

# Integration-simulation walkthrough — 378 bp CMV-promoter construct.
_INTEGRATION_CONSTRUCT = (
    "TAGTTATTAATAGTAATCAATTACGGGGTCATTAGTTCATAGCCCATATATGGAGTTCCGCGT"
    "TACATAACTTACGGTAAATGGCCCGCCTGGCTGACCGCCCAACGACCCCCGCCCATTGACGTC"
    "AATAATGACGTATGTTCCCATAGTAACGCCAATAGGGACTTTCCATTGACGTCAATGGGTGGA"
    "GTATTTACGGTAAACTGCCCACTTGGCAGTACATCAAGTGTATCATATGCCAAGTACGCCCCC"
    "TATTGACGTCAATGACGGTAAATGGCCCGCCTGGCATTATGCCCAGTACATGACCTTATGGGA"
    "CTTTCCTACTTGGCAGTACATCTACGTATTAGTCATCGCTATTACCATGGTGATGCGGTTTTG"
)


WALKTHROUGHS: list[dict[str, Any]] = [
    # ── variant_analysis ────────────────────────────────────────────────
    {
        "name": "SORT1 rs12740374 — AlphaGenome multi-layer",
        "dir": "examples/walkthroughs/variant_analysis/SORT1_rs12740374",
        "mcp_tool": "analyze_variant_multilayer",
        "oracle": "alphagenome",
        "position": "chr1:109274968", "ref": "G", "alt": "T",
        "gene": "SORT1",
        "assay_ids": _AG_HEPG2,
        "html": "rs12740374_SORT1_alphagenome_report.html",
    },
    {
        "name": "SORT1 rs12740374 — ChromBPNet DNASE HepG2 (1bp resolution)",
        "dir": "examples/walkthroughs/variant_analysis/SORT1_chrombpnet",
        "mcp_tool": "analyze_variant_multilayer",
        "oracle": "chrombpnet",
        "position": "chr1:109274968", "ref": "G", "alt": "T",
        "gene": "SORT1",
        "chrombpnet_assay": "DNASE",
        "chrombpnet_cell_type": "HepG2",
        "chrombpnet_fold": 0,
        "html": "rs12740374_SORT1_chrombpnet_report.html",
    },
    {
        "name": "SORT1 rs12740374 — Enformer discovery (top tracks per layer)",
        "dir": "examples/walkthroughs/variant_analysis/SORT1_enformer",
        "mcp_tool": "discover_variant",
        "oracle": "enformer",
        "position": "chr1:109274968", "ref": "G", "alt": "T",
        "gene": "SORT1",
        "top_n_per_layer": 12,
        "html": "rs12740374_SORT1_enformer_report.html",
    },
    {
        "name": "BCL11A rs1427407 — K562 erythroid",
        "dir": "examples/walkthroughs/variant_analysis/BCL11A_rs1427407",
        "mcp_tool": "analyze_variant_multilayer",
        "oracle": "alphagenome",
        "position": "chr2:60490908", "ref": "G", "alt": "T",
        "gene": "BCL11A",
        "assay_ids": _AG_BCL11A,
        "html": "rs1427407_BCL11A_alphagenome_report.html",
    },
    {
        "name": "FTO rs1421085 — HepG2 negative control",
        "dir": "examples/walkthroughs/variant_analysis/FTO_rs1421085",
        "mcp_tool": "analyze_variant_multilayer",
        "oracle": "alphagenome",
        "position": "chr16:53767042", "ref": "T", "alt": "C",
        "gene": "FTO",
        "assay_ids": _AG_HEPG2,
        "html": "rs1421085_FTO_alphagenome_report.html",
    },
    # ── validation ──────────────────────────────────────────────────────
    {
        "name": "SORT1 rs12740374 — validation with CEBPA/CEBPB co-occupancy",
        "dir": "examples/walkthroughs/validation/SORT1_rs12740374_with_CEBP",
        "mcp_tool": "analyze_variant_multilayer",
        "oracle": "alphagenome",
        "position": "chr1:109274968", "ref": "G", "alt": "T",
        "gene": "SORT1",
        "assay_ids": _AG_HEPG2,
        "html": "rs12740374_SORT1_CEBP_validation_report.html",
    },
    {
        "name": "SORT1 rs12740374 — multi-oracle consensus (ChromBPNet + LegNet + AlphaGenome)",
        "dir": "examples/walkthroughs/validation/SORT1_rs12740374_multioracle",
        "mcp_tool": "multioracle",
        "oracle": "alphagenome",  # also runs chrombpnet + legnet in subprocess envs
        "position": "chr1:109274968", "ref": "G", "alt": "T",
        "gene": "SORT1",
        "ag_assay_ids": _AG_HEPG2,
        "chrombpnet_assay": "DNASE",
        "chrombpnet_cell_type": "HepG2",
        "legnet_cell_type": "HepG2",
        "legnet_assay": "LentiMPRA",
        "consolidated_html": "rs12740374_SORT1_multioracle_report.html",
    },
    {
        "name": "TERT promoter — chr5:1295046 discovery",
        "dir": "examples/walkthroughs/validation/TERT_chr5_1295046",
        "mcp_tool": "discover_variant",
        "oracle": "alphagenome",
        "position": "chr5:1295046", "ref": "T", "alt": "G",
        "gene": "TERT",
        "top_n_per_layer": 3,
        "html": "chr5_1295046_T_G_TERT_alphagenome_report.html",
    },
    # ── discovery ───────────────────────────────────────────────────────
    {
        "name": "SORT1 rs12740374 — cell-type screen",
        "dir": "examples/walkthroughs/discovery/SORT1_cell_type_screen",
        "mcp_tool": "discover_variant_cell_types",
        "oracle": "alphagenome",
        "position": "chr1:109274968", "ref": "G", "alt": "T",
        "gene": "SORT1",
        "top_n": 5,
        "min_effect": 0.15,
    },
    # ── causal_prioritization ───────────────────────────────────────────
    {
        "name": "SORT1 locus — fine-map 11 LD proxies",
        "dir": "examples/walkthroughs/causal_prioritization/SORT1_locus",
        "mcp_tool": "fine_map_causal_variant",
        "oracle": "alphagenome",
        "lead_variant_id": "rs12740374",
        "ld_variants": _SORT1_LD,
        "assay_ids": _SORT1_CAUSAL_TRACKS,
        "gene": "SORT1",
        "r2_threshold": 0.85,
        "html": "rs12740374_SORT1_locus_causal_report.html",
    },
    # ── sequence_engineering ────────────────────────────────────────────
    {
        "name": "SORT1 enhancer — 630 bp GFP/reporter region swap (K562)",
        "dir": "examples/walkthroughs/sequence_engineering/region_swap",
        "mcp_tool": "analyze_region_swap",
        "oracle": "alphagenome",
        "region": "chr1:109274500-109275500",
        "replacement_sequence": _REGION_SWAP_REPLACEMENT,
        "assay_ids": _AG_K562_SWAP,
        "gene": "SORT1",
        "html": "region_swap_SORT1_K562_report.html",
    },
    {
        "name": "AAVS1 (PPP1R12C) — 378 bp CMV cassette integration (K562)",
        "dir": "examples/walkthroughs/sequence_engineering/integration_simulation",
        "mcp_tool": "simulate_integration",
        "oracle": "alphagenome",
        "position": "chr19:55115000",
        "construct_sequence": _INTEGRATION_CONSTRUCT,
        "assay_ids": _AG_K562_INTEG,
        "gene": "PPP1R12C",
        "html": "integration_CMV_PPP1R12C_report.html",
    },
    # ── batch_scoring ───────────────────────────────────────────────────
    {
        "name": "SORT1 locus — batch-score 5 variants",
        "dir": "examples/walkthroughs/batch_scoring",
        "mcp_tool": "score_variant_batch",
        "oracle": "alphagenome",
        "variants": _BATCH_VARIANTS,
        "assay_ids": _SORT1_CAUSAL_TRACKS,
        "gene": "SORT1",
        "top_n": 20,
        "html": "batch_sort1_locus_scoring.html",
    },
]


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

def _md(text: str) -> dict:
    return nbf.v4.new_markdown_cell(text)


def _code(text: str) -> dict:
    return nbf.v4.new_code_cell(text)


def _repr_list(items: list[Any]) -> str:
    """Pretty-print a list (one item per line, 4-space indent)."""
    if not items:
        return "[]"
    body = ",\n    ".join(repr(x) for x in items)
    return f"[\n    {body},\n]"


def _repr_dict_list(dicts: list[dict[str, Any]]) -> str:
    if not dicts:
        return "[]"
    body = ",\n    ".join(repr(d) for d in dicts)
    return f"[\n    {body},\n]"


def _wrapped_dna(seq: str, width: int = 64) -> str:
    """Wrap a long DNA string into ``"<chunk>"`` lines for readability."""
    chunks = [seq[i:i + width] for i in range(0, len(seq), width)]
    return "(\n        " + "\n        ".join(f'"{c}"' for c in chunks) + "\n    )"


def _imports_cell(extra: list[str] | None = None) -> dict:
    base = [
        "# All imports the notebook needs — top-level, no later imports.",
        "import json",
        "from pathlib import Path",
        "",
        "import chorus",
        "from chorus.analysis.normalization import get_normalizer",
    ]
    if extra:
        base.extend(extra)
    return _code("\n".join(base))


def _save_artifacts_cell(*, report_var: str, dir_var: str, html_filename: str) -> dict:
    """Cell that saves markdown / JSON / TSV / HTML the same way regenerate_*.py does."""
    return _code(
        "# Save the same artifacts the MCP tool would produce:\n"
        "#   - example_output.md  (markdown report)\n"
        "#   - example_output.json (structured scores)\n"
        "#   - example_output.tsv (track-level table)\n"
        "#   - {html} (interactive IGV report)\n".format(html=html_filename)
        + f"{dir_var}.joinpath(\"example_output.md\").write_text({report_var}.to_markdown())\n"
        f"{dir_var}.joinpath(\"example_output.json\").write_text(\n"
        f"    json.dumps({report_var}.to_dict(), indent=2, default=str)\n"
        f")\n"
        f"try:\n"
        f"    {report_var}.to_dataframe().to_csv(\n"
        f"        {dir_var} / \"example_output.tsv\", sep=\"\\t\", index=False,\n"
        f"    )\n"
        f"except Exception as exc:\n"
        f"    print(f\"TSV write skipped: {{exc}}\")\n"
        f"\n"
        f"_html_path = {report_var}.to_html(output_path=str({dir_var} / \"{html_filename}\"))\n"
        f"print(f\"Wrote artifacts to {{ {dir_var} }}\")\n"
    )


# ---------------------------------------------------------------------------
# Cell templates — one per MCP-tool flow
# ---------------------------------------------------------------------------

def _cells_analyze_variant_multilayer(spec: dict) -> list[dict]:
    """analyze_variant_multilayer (AlphaGenome multi-track flow)."""
    cells: list[dict] = []
    cells.append(_md(
        f"# {spec['name']}\n\n"
        f"Reproduces the MCP call ``analyze_variant_multilayer(...)``\n"
        f"for variant **{spec['position']} {spec['ref']}>{spec['alt']}**\n"
        f"on the **{spec['oracle']}** oracle. Top-to-bottom execution\n"
        f"writes the same `example_output.md/json/tsv` and HTML report\n"
        f"the MCP tool would.\n\n"
        f"_Requires the `chorus` package installed and the `chorus-{spec['oracle']}` env\n"
        f"created (`chorus setup --oracle {spec['oracle']}`)._"
    ))
    cells.append(_imports_cell(extra=[
        "from chorus.analysis.variant_report import build_variant_report",
    ]))

    cells.append(_code(
        "# Locate the walkthrough directory so artifacts land alongside the notebook.\n"
        "WALKTHROUGH_DIR = Path.cwd()\n"
        "print(f\"Writing artifacts to: {WALKTHROUGH_DIR}\")"
    ))

    cells.append(_code(
        "# Instantiate the oracle and load weights. ``use_environment=True``\n"
        "# delegates the heavy model load to the matching mamba env\n"
        f"# (``chorus-{spec['oracle']}``) so the notebook itself runs in base chorus.\n"
        f"oracle = chorus.create_oracle(\n"
        f"    oracle_name={spec['oracle']!r},\n"
        f"    use_environment=True,\n"
        f")\n"
        f"oracle.load_pretrained_model()"
    ))

    cells.append(_code(
        "# Define every input explicitly — no implicit defaults.\n"
        f"oracle_name = {spec['oracle']!r}\n"
        f"position = {spec['position']!r}\n"
        f"ref_allele = {spec['ref']!r}\n"
        f"alt_alleles = [{spec['alt']!r}]\n"
        f"gene_name = {spec['gene']!r}\n"
        f"assay_ids = {_repr_list(spec['assay_ids'])}"
    ))

    cells.append(_code(
        "# Auto-pick a prediction region of the oracle's natural width centred\n"
        "# on the variant. Matches what the MCP tool does internally.\n"
        "_chrom, _pos = position.split(\":\")\n"
        "_pos = int(_pos)\n"
        "_half = oracle.output_size // 2\n"
        "region = f\"{_chrom}:{max(1, _pos - _half)}-{_pos + _half}\""
    ))

    cells.append(_code(
        "# Run the oracle on ref + alt allele(s). Returns a dict with\n"
        "# 'predictions', 'effect_sizes', and 'variant_info'.\n"
        "variant_result = oracle.predict_variant_effect(\n"
        "    genomic_region=region,\n"
        "    variant_position=position,\n"
        "    alleles=[ref_allele] + alt_alleles,\n"
        "    assay_ids=assay_ids,\n"
        ")"
    ))

    cells.append(_code(
        "# Build the multi-layer report — same function the MCP tool calls.\n"
        "# get_normalizer() auto-loads the per-track CDF for percentile scoring.\n"
        "normalizer = get_normalizer(oracle_name=oracle_name)\n"
        "report = build_variant_report(\n"
        "    variant_result=variant_result,\n"
        "    oracle_name=oracle_name,\n"
        "    gene_name=gene_name,\n"
        "    normalizer=normalizer,\n"
        "    igv_raw=False,\n"
        ")"
    ))

    cells.append(_save_artifacts_cell(
        report_var="report",
        dir_var="WALKTHROUGH_DIR",
        html_filename=spec["html"],
    ))

    cells.append(_md(
        "## What this notebook produced\n\n"
        f"- `example_output.md` — markdown report (top tracks per layer)\n"
        f"- `example_output.json` — structured per-layer / per-track scores\n"
        f"- `example_output.tsv` — track-level table (one row per track)\n"
        f"- `{spec['html']}` — interactive IGV browser with ref / alt overlays\n"
    ))
    return cells


def _cells_chrombpnet_variant(spec: dict) -> list[dict]:
    """analyze_variant_multilayer flow tuned for ChromBPNet (assay + cell_type loader)."""
    cells: list[dict] = []
    cells.append(_md(
        f"# {spec['name']}\n\n"
        f"Reproduces the MCP `analyze_variant_multilayer` call for the\n"
        f"ChromBPNet oracle, which loads one model per (assay, cell_type)\n"
        f"pair rather than taking a generic track list.\n"
    ))
    cells.append(_imports_cell(extra=[
        "from chorus.analysis.variant_report import build_variant_report",
        "from chorus.analysis.analysis_request import AnalysisRequest",
    ]))
    cells.append(_code(
        "WALKTHROUGH_DIR = Path.cwd()\n"
        "print(f\"Writing artifacts to: {WALKTHROUGH_DIR}\")"
    ))
    cells.append(_code(
        "# Create the ChromBPNet oracle and load a specific assay+cell_type model.\n"
        "oracle = chorus.create_oracle(\n"
        f"    oracle_name={spec['oracle']!r},\n"
        "    use_environment=True,\n"
        ")\n"
        "oracle.load_pretrained_model(\n"
        f"    assay={spec['chrombpnet_assay']!r},\n"
        f"    cell_type={spec['chrombpnet_cell_type']!r},\n"
        f"    fold={spec['chrombpnet_fold']!r},\n"
        ")"
    ))
    cells.append(_code(
        "# Inputs.\n"
        f"oracle_name = {spec['oracle']!r}\n"
        f"position = {spec['position']!r}\n"
        f"ref_allele = {spec['ref']!r}\n"
        f"alt_alleles = [{spec['alt']!r}]\n"
        f"gene_name = {spec['gene']!r}"
    ))
    cells.append(_code(
        "# ChromBPNet works on a narrow ±100 bp prediction interval around the\n"
        "# variant (the model's input width is 2114 bp; we keep the user-facing\n"
        "# region small and let the oracle extend internally).\n"
        "_chrom, _pos = position.split(\":\")\n"
        "_pos = int(_pos)\n"
        "region = f\"{_chrom}:{_pos - 100}-{_pos + 100}\""
    ))
    cells.append(_code(
        "# Score the variant. assay_ids=[] tells the loaded ChromBPNet model to\n"
        "# return its single track.\n"
        "variant_result = oracle.predict_variant_effect(\n"
        "    genomic_region=region,\n"
        "    variant_position=position,\n"
        "    alleles=[ref_allele] + alt_alleles,\n"
        "    assay_ids=[],\n"
        ")"
    ))
    cells.append(_code(
        "# Build the report with the ChromBPNet per-track normalizer.\n"
        "normalizer = get_normalizer(oracle_name=oracle_name)\n"
        "analysis_request = AnalysisRequest(\n"
        f"    user_prompt=f\"Score {{position}} {{ref_allele}}>{{alt_alleles[0]}} \"\n"
        f"                f\"using ChromBPNet {spec['chrombpnet_assay']!r} model in \"\n"
        f"                f\"{spec['chrombpnet_cell_type']!r}.\",\n"
        "    tool_name=\"analyze_variant_multilayer\",\n"
        f"    oracle_name={spec['oracle']!r},\n"
        f"    tracks_requested=\"{spec['chrombpnet_assay']}:{spec['chrombpnet_cell_type']}\",\n"
        ")\n"
        "report = build_variant_report(\n"
        "    variant_result=variant_result,\n"
        "    oracle_name=oracle_name,\n"
        "    gene_name=gene_name,\n"
        "    normalizer=normalizer,\n"
        "    igv_raw=False,\n"
        "    analysis_request=analysis_request,\n"
        ")"
    ))
    cells.append(_save_artifacts_cell(
        report_var="report",
        dir_var="WALKTHROUGH_DIR",
        html_filename=spec["html"],
    ))
    cells.append(_md(
        "## What this notebook produced\n\n"
        f"- `example_output.md` — markdown report (1bp-resolution ChromBPNet profile)\n"
        f"- `example_output.json` — structured per-bin scores\n"
        f"- `example_output.tsv` — track table\n"
        f"- `{spec['html']}` — IGV browser with 1bp ref / alt profile overlay\n"
    ))
    return cells


def _cells_discover_variant(spec: dict) -> list[dict]:
    """discover_variant — score ALL tracks, return top hits per layer."""
    cells: list[dict] = []
    cells.append(_md(
        f"# {spec['name']}\n\n"
        f"Reproduces the MCP `discover_variant` call. Scores every track\n"
        f"on the **{spec['oracle']}** oracle for the variant\n"
        f"**{spec['position']} {spec['ref']}>{spec['alt']}** and returns the\n"
        f"top tracks per regulatory layer.\n"
    ))
    cells.append(_imports_cell(extra=[
        "from chorus.analysis.discovery import discover_variant_effects",
    ]))
    cells.append(_code(
        "WALKTHROUGH_DIR = Path.cwd()\n"
        "print(f\"Writing artifacts to: {WALKTHROUGH_DIR}\")"
    ))
    cells.append(_code(
        "# Instantiate + load the oracle in its subprocess env.\n"
        "oracle = chorus.create_oracle(\n"
        f"    oracle_name={spec['oracle']!r},\n"
        "    use_environment=True,\n"
        ")\n"
        "oracle.load_pretrained_model()"
    ))
    cells.append(_code(
        "# Inputs.\n"
        f"oracle_name = {spec['oracle']!r}\n"
        f"variant_position = {spec['position']!r}\n"
        f"alleles = [{spec['ref']!r}, {spec['alt']!r}]\n"
        f"gene_name = {spec['gene']!r}\n"
        f"top_n_per_layer = {spec['top_n_per_layer']!r}\n"
        f"top_n_cell_types = 10\n"
        f"ranking_metric = \"alt_x_abs_effect\"\n"
        f"min_ref_value = 0.0"
    ))
    cells.append(_code(
        "# Discovery scoring across ALL tracks (no assay_ids whitelist).\n"
        "# Saves the IGV report directly into the walkthrough dir.\n"
        "normalizer = get_normalizer(oracle_name=oracle_name)\n"
        "discovery_result = discover_variant_effects(\n"
        "    oracle=oracle,\n"
        "    oracle_name=oracle_name,\n"
        "    variant_position=variant_position,\n"
        "    alleles=alleles,\n"
        "    top_n_per_layer=top_n_per_layer,\n"
        "    top_n_cell_types=top_n_cell_types,\n"
        "    gene_name=gene_name,\n"
        "    normalizer=normalizer,\n"
        "    output_path=str(WALKTHROUGH_DIR),\n"
        f"    output_filename={spec['html']!r},\n"
        "    igv_raw=False,\n"
        "    analysis_request=None,\n"
        "    ranking_metric=ranking_metric,\n"
        "    min_ref_value=min_ref_value,\n"
        ")"
    ))
    cells.append(_code(
        "# Save markdown summary + structured JSON. discover_variant_effects already\n"
        "# wrote the HTML in the previous cell.\n"
        "report = discovery_result.pop(\"report\", None)\n"
        "if report is not None:\n"
        "    WALKTHROUGH_DIR.joinpath(\"example_output.md\").write_text(report.to_markdown())\n"
        "WALKTHROUGH_DIR.joinpath(\"example_output.json\").write_text(\n"
        "    json.dumps(discovery_result, indent=2, default=str)\n"
        ")\n"
        "try:\n"
        "    if report is not None:\n"
        "        report.to_dataframe().to_csv(\n"
        "            WALKTHROUGH_DIR / \"example_output.tsv\", sep=\"\\t\", index=False,\n"
        "        )\n"
        "except Exception as exc:\n"
        "    print(f\"TSV write skipped: {exc}\")\n"
        "print(f\"Wrote artifacts to {WALKTHROUGH_DIR}\")"
    ))
    cells.append(_md(
        "## What this notebook produced\n\n"
        f"- `example_output.md`, `.json`, `.tsv` — top tracks per layer + cell-type ranking\n"
        f"- `{spec['html']}` — interactive IGV report for the selected top tracks\n"
    ))
    return cells


def _cells_discover_variant_cell_types(spec: dict) -> list[dict]:
    cells: list[dict] = []
    cells.append(_md(
        f"# {spec['name']}\n\n"
        f"Reproduces the MCP `discover_variant_cell_types` call. Screens all\n"
        f"DNASE/ATAC tracks on **{spec['oracle']}**, ranks cell types by\n"
        f"variant effect magnitude (weighted by alt activity), then runs a\n"
        f"full multi-layer report per top cell type.\n"
    ))
    cells.append(_imports_cell(extra=[
        "from chorus.analysis.discovery import discover_and_report",
    ]))
    cells.append(_code(
        "WALKTHROUGH_DIR = Path.cwd()\n"
        "print(f\"Writing artifacts to: {WALKTHROUGH_DIR}\")"
    ))
    cells.append(_code(
        "oracle = chorus.create_oracle(\n"
        f"    oracle_name={spec['oracle']!r},\n"
        "    use_environment=True,\n"
        ")\n"
        "oracle.load_pretrained_model()"
    ))
    cells.append(_code(
        "# Inputs.\n"
        f"oracle_name = {spec['oracle']!r}\n"
        f"variant_position = {spec['position']!r}\n"
        f"alleles = [{spec['ref']!r}, {spec['alt']!r}]\n"
        f"gene_name = {spec['gene']!r}\n"
        f"top_n = {spec['top_n']!r}\n"
        f"min_effect = {spec['min_effect']!r}\n"
        f"ranking_metric = \"alt_x_abs_effect\"\n"
        f"min_ref_value = 0.0"
    ))
    cells.append(_code(
        "# Two-stage discovery: screen → full report per top cell type.\n"
        "# Writes one HTML per top cell type into the walkthrough dir.\n"
        "normalizer = get_normalizer(oracle_name=oracle_name)\n"
        "result = discover_and_report(\n"
        "    oracle=oracle,\n"
        "    variant_position=variant_position,\n"
        "    alleles=alleles,\n"
        "    gene_name=gene_name,\n"
        "    top_n=top_n,\n"
        "    min_effect=min_effect,\n"
        "    output_path=str(WALKTHROUGH_DIR),\n"
        "    normalizer=normalizer,\n"
        "    igv_raw=False,\n"
        "    oracle_name=oracle_name,\n"
        "    user_prompt=None,\n"
        "    tool_name=\"discover_variant_cell_types\",\n"
        "    ranking_metric=ranking_metric,\n"
        "    min_ref_value=min_ref_value,\n"
        ")"
    ))
    cells.append(_code(
        "# Save the screening summary (cell-type ranking + per-cell-type report dicts).\n"
        "serializable = {\n"
        "    \"hits\": result[\"hits\"],\n"
        "    \"reports\": {\n"
        "        ct: rep.to_dict() for ct, rep in result.get(\"reports\", {}).items()\n"
        "    },\n"
        "}\n"
        "WALKTHROUGH_DIR.joinpath(\"discovery_summary.json\").write_text(\n"
        "    json.dumps(serializable, indent=2, default=str)\n"
        ")\n"
        "WALKTHROUGH_DIR.joinpath(\"example_output.json\").write_text(\n"
        "    json.dumps(serializable, indent=2, default=str)\n"
        ")\n"
        "# Markdown summary lists the top hits.\n"
        "_md_lines = [\n"
        "    f\"# Cell-type screen — {gene_name} {variant_position}\",\n"
        "    \"\",\n"
        "    f\"Top {top_n} cell types ranked by {ranking_metric}:\",\n"
        "    \"\",\n"
        "    \"| Cell type | Effect | |Effect| | Best track | n tracks |\",\n"
        "    \"|---|---|---|---|---|\",\n"
        "]\n"
        "for h in result[\"hits\"]:\n"
        "    _md_lines.append(\n"
        "        f\"| {h['cell_type']} | {h['effect']:+.3f} | {h['abs_effect']:.3f} | {h['best_track']} | {h['n_tracks']} |\"\n"
        "    )\n"
        "WALKTHROUGH_DIR.joinpath(\"example_output.md\").write_text(\"\\n\".join(_md_lines))\n"
        "print(f\"Wrote artifacts to {WALKTHROUGH_DIR}\")"
    ))
    cells.append(_md(
        "## What this notebook produced\n\n"
        "- `discovery_summary.json` / `example_output.json` — cell-type ranking + per-cell-type report dicts\n"
        "- `example_output.md` — markdown table of top cell types\n"
        "- One HTML per top cell type (e.g. `chr*_*_<celltype>_report.html`)\n"
    ))
    return cells


def _cells_fine_map_causal_variant(spec: dict) -> list[dict]:
    cells: list[dict] = []
    cells.append(_md(
        f"# {spec['name']}\n\n"
        f"Reproduces the MCP `fine_map_causal_variant` call for the lead\n"
        f"variant **{spec['lead_variant_id']}** at the SORT1 locus.\n\n"
        f"The 11 LD proxies (CEU, r²≥{spec['r2_threshold']}) are inlined as a\n"
        f"Python list so the notebook runs offline. Pass them through\n"
        f"`prioritize_causal_variants` to get the composite ranking.\n\n"
        f"_To re-fetch from LDlink instead, uncomment the `fetch_ld_variants`\n"
        f"cell at the bottom._"
    ))
    cells.append(_imports_cell(extra=[
        "from chorus.utils.ld import ld_variants_from_list",
        "from chorus.analysis.causal import prioritize_causal_variants",
    ]))
    cells.append(_code(
        "WALKTHROUGH_DIR = Path.cwd()\n"
        "print(f\"Writing artifacts to: {WALKTHROUGH_DIR}\")"
    ))
    cells.append(_code(
        "oracle = chorus.create_oracle(\n"
        f"    oracle_name={spec['oracle']!r},\n"
        "    use_environment=True,\n"
        ")\n"
        "oracle.load_pretrained_model()"
    ))
    cells.append(_code(
        "# Inputs — inline LD proxies (no LDlink network dep).\n"
        f"oracle_name = {spec['oracle']!r}\n"
        f"lead_variant_id = {spec['lead_variant_id']!r}\n"
        f"gene_name = {spec['gene']!r}\n"
        f"r2_threshold = {spec['r2_threshold']!r}\n"
        f"ld_variants_raw = {_repr_dict_list(spec['ld_variants'])}\n"
        f"assay_ids = {_repr_list(spec['assay_ids'])}"
    ))
    cells.append(_code(
        "# Convert the inline dicts to LDVariant objects, marking the lead as sentinel.\n"
        "ld_variants = ld_variants_from_list(\n"
        "    variants=ld_variants_raw,\n"
        "    sentinel_id=lead_variant_id,\n"
        ")\n"
        f"lead_dict = next(v for v in ld_variants_raw if v[\"id\"] == lead_variant_id)\n"
        f"print(f\"Sentinel: {{lead_dict['id']}} at {{lead_dict['chrom']}}:{{lead_dict['pos']}} {{lead_dict['ref']}}>{{lead_dict['alt']}}\")\n"
        f"print(f\"Proxies: {{len(ld_variants) - 1}}\")"
    ))
    cells.append(_code(
        "# Score + rank each proxy by the composite causal score\n"
        "# (max_effect + n_layers + convergence + ref_activity).\n"
        "normalizer = get_normalizer(oracle_name=oracle_name)\n"
        "result = prioritize_causal_variants(\n"
        "    oracle=oracle,\n"
        "    lead_variant=lead_dict,\n"
        "    ld_variants=ld_variants,\n"
        "    assay_ids=assay_ids,\n"
        "    gene_name=gene_name,\n"
        "    oracle_name=oracle_name,\n"
        "    weights=None,\n"
        "    normalizer=normalizer,\n"
        "    analysis_request=None,\n"
        "    snvs_only=False,\n"
        ")"
    ))
    cells.append(_save_artifacts_cell(
        report_var="result",
        dir_var="WALKTHROUGH_DIR",
        html_filename=spec["html"],
    ))
    cells.append(_code(
        "# Optional: refresh LD proxies from LDlink (requires a token).\n"
        "# import os\n"
        "# from chorus.utils.ld import fetch_ld_variants\n"
        "# ld_variants_live = fetch_ld_variants(\n"
        "#     variant_id=lead_variant_id,\n"
        "#     population=\"CEU\",\n"
        "#     r2_threshold=r2_threshold,\n"
        "#     token=os.environ.get(\"LDLINK_TOKEN\"),\n"
        "#     timeout=30.0,\n"
        "#     genome_build=\"grch38\",\n"
        "#     snvs_only=False,\n"
        "# )"
    ))
    cells.append(_md(
        "## What this notebook produced\n\n"
        f"- `example_output.md` — ranked table of LD proxies by composite causal score\n"
        f"- `example_output.json` — per-variant per-layer scores\n"
        f"- `example_output.tsv` — flat table view\n"
        f"- `{spec['html']}` — interactive IGV report\n"
    ))
    return cells


def _cells_score_variant_batch(spec: dict) -> list[dict]:
    cells: list[dict] = []
    cells.append(_md(
        f"# {spec['name']}\n\n"
        f"Reproduces the MCP `score_variant_batch` call. Scores N variants\n"
        f"across the supplied AlphaGenome tracks and ranks them by max\n"
        f"effect magnitude across layers.\n"
    ))
    cells.append(_imports_cell(extra=[
        "from chorus.analysis.batch_scoring import score_variant_batch",
        "from chorus.analysis.analysis_request import AnalysisRequest",
    ]))
    cells.append(_code(
        "WALKTHROUGH_DIR = Path.cwd()\n"
        "print(f\"Writing artifacts to: {WALKTHROUGH_DIR}\")"
    ))
    cells.append(_code(
        "oracle = chorus.create_oracle(\n"
        f"    oracle_name={spec['oracle']!r},\n"
        "    use_environment=True,\n"
        ")\n"
        "oracle.load_pretrained_model()"
    ))
    cells.append(_code(
        "# Inputs.\n"
        f"oracle_name = {spec['oracle']!r}\n"
        f"gene_name = {spec['gene']!r}\n"
        f"top_n = {spec['top_n']!r}\n"
        f"variants = {_repr_dict_list(spec['variants'])}\n"
        f"assay_ids = {_repr_list(spec['assay_ids'])}"
    ))
    cells.append(_code(
        "# Run batch scoring. Each variant is scored multi-layer with the same\n"
        "# scorer the single-variant API uses; the batch wrapper ranks results.\n"
        "normalizer = get_normalizer(oracle_name=oracle_name)\n"
        "analysis_request = AnalysisRequest(\n"
        "    user_prompt=f\"Batch-score {len(variants)} SORT1-locus variants.\",\n"
        "    tool_name=\"score_variant_batch\",\n"
        "    oracle_name=oracle_name,\n"
        "    tracks_requested=f\"{len(assay_ids)} AlphaGenome tracks\",\n"
        ")\n"
        "batch_result = score_variant_batch(\n"
        "    oracle=oracle,\n"
        "    variants=variants,\n"
        "    assay_ids=assay_ids,\n"
        "    gene_name=gene_name,\n"
        "    normalizer=normalizer,\n"
        "    analysis_request=analysis_request,\n"
        "    oracle_name=oracle_name,\n"
        ")"
    ))
    cells.append(_save_artifacts_cell(
        report_var="batch_result",
        dir_var="WALKTHROUGH_DIR",
        html_filename=spec["html"],
    ))
    cells.append(_md(
        "## What this notebook produced\n\n"
        f"- `example_output.md` — ranked variant table (top {spec['top_n']})\n"
        f"- `example_output.json` — per-variant per-layer scores\n"
        f"- `example_output.tsv` — flat table\n"
        f"- `{spec['html']}` — batch HTML report\n"
    ))
    return cells


def _cells_analyze_region_swap(spec: dict) -> list[dict]:
    cells: list[dict] = []
    cells.append(_md(
        f"# {spec['name']}\n\n"
        f"Reproduces the MCP `analyze_region_swap` call. Replaces the\n"
        f"genomic region **{spec['region']}** with a custom DNA sequence\n"
        f"and scores the same regulatory layers the variant API uses.\n"
    ))
    cells.append(_imports_cell(extra=[
        "from chorus.analysis.region_swap import analyze_region_swap",
        "from chorus.analysis.analysis_request import AnalysisRequest",
    ]))
    cells.append(_code(
        "WALKTHROUGH_DIR = Path.cwd()\n"
        "print(f\"Writing artifacts to: {WALKTHROUGH_DIR}\")"
    ))
    cells.append(_code(
        "oracle = chorus.create_oracle(\n"
        f"    oracle_name={spec['oracle']!r},\n"
        "    use_environment=True,\n"
        ")\n"
        "oracle.load_pretrained_model()"
    ))
    cells.append(_code(
        "# Inputs.\n"
        f"oracle_name = {spec['oracle']!r}\n"
        f"region = {spec['region']!r}\n"
        f"gene_name = {spec['gene']!r}\n"
        f"assay_ids = {_repr_list(spec['assay_ids'])}\n"
        f"replacement_sequence = {_wrapped_dna(spec['replacement_sequence'])}\n"
        "print(f\"Replacement length: {len(replacement_sequence)} bp\")"
    ))
    cells.append(_code(
        "# Score the swap. Returns a VariantReport with ref (WT) vs alt\n"
        "# (post-replacement) predictions in each layer.\n"
        "normalizer = get_normalizer(oracle_name=oracle_name)\n"
        "analysis_request = AnalysisRequest(\n"
        "    user_prompt=(\n"
        "        f\"Replace {region} with a {len(replacement_sequence)} bp \"\n"
        "        f\"construct and predict effects on K562 tracks.\"\n"
        "    ),\n"
        "    tool_name=\"analyze_region_swap\",\n"
        "    oracle_name=oracle_name,\n"
        "    tracks_requested=f\"{len(assay_ids)} K562 tracks\",\n"
        ")\n"
        "report = analyze_region_swap(\n"
        "    oracle=oracle,\n"
        "    region=region,\n"
        "    replacement_sequence=replacement_sequence,\n"
        "    assay_ids=assay_ids,\n"
        "    gene_name=gene_name,\n"
        "    normalizer=normalizer,\n"
        "    oracle_name=oracle_name,\n"
        ")\n"
        "report.analysis_request = analysis_request"
    ))
    cells.append(_save_artifacts_cell(
        report_var="report",
        dir_var="WALKTHROUGH_DIR",
        html_filename=spec["html"],
    ))
    cells.append(_md(
        "## What this notebook produced\n\n"
        f"- `example_output.md/.json/.tsv` — WT vs replacement per-layer scores\n"
        f"- `{spec['html']}` — IGV browser with WT/replacement track overlay\n"
    ))
    return cells


def _cells_simulate_integration(spec: dict) -> list[dict]:
    cells: list[dict] = []
    cells.append(_md(
        f"# {spec['name']}\n\n"
        f"Reproduces the MCP `simulate_integration` call. Inserts a\n"
        f"construct at **{spec['position']}** (no genomic bases removed)\n"
        f"and scores regulatory disruption at the surrounding locus.\n"
    ))
    cells.append(_imports_cell(extra=[
        "from chorus.analysis.integration import simulate_integration",
        "from chorus.analysis.analysis_request import AnalysisRequest",
    ]))
    cells.append(_code(
        "WALKTHROUGH_DIR = Path.cwd()\n"
        "print(f\"Writing artifacts to: {WALKTHROUGH_DIR}\")"
    ))
    cells.append(_code(
        "oracle = chorus.create_oracle(\n"
        f"    oracle_name={spec['oracle']!r},\n"
        "    use_environment=True,\n"
        ")\n"
        "oracle.load_pretrained_model()"
    ))
    cells.append(_code(
        "# Inputs.\n"
        f"oracle_name = {spec['oracle']!r}\n"
        f"position = {spec['position']!r}\n"
        f"gene_name = {spec['gene']!r}\n"
        f"assay_ids = {_repr_list(spec['assay_ids'])}\n"
        f"construct_sequence = {_wrapped_dna(spec['construct_sequence'])}\n"
        "print(f\"Construct length: {len(construct_sequence)} bp\")"
    ))
    cells.append(_code(
        "# Score the integration. Same scorer as analyze_region_swap but\n"
        "# semantically an insertion (genomic bases retained).\n"
        "normalizer = get_normalizer(oracle_name=oracle_name)\n"
        "analysis_request = AnalysisRequest(\n"
        "    user_prompt=(\n"
        "        f\"Insert a {len(construct_sequence)} bp construct at {position} \"\n"
        "        \"and predict local chromatin disruption.\"\n"
        "    ),\n"
        "    tool_name=\"simulate_integration\",\n"
        "    oracle_name=oracle_name,\n"
        "    tracks_requested=f\"{len(assay_ids)} K562 tracks\",\n"
        ")\n"
        "report = simulate_integration(\n"
        "    oracle=oracle,\n"
        "    position=position,\n"
        "    construct_sequence=construct_sequence,\n"
        "    assay_ids=assay_ids,\n"
        "    gene_name=gene_name,\n"
        "    normalizer=normalizer,\n"
        "    oracle_name=oracle_name,\n"
        ")\n"
        "report.analysis_request = analysis_request"
    ))
    cells.append(_save_artifacts_cell(
        report_var="report",
        dir_var="WALKTHROUGH_DIR",
        html_filename=spec["html"],
    ))
    cells.append(_md(
        "## What this notebook produced\n\n"
        f"- `example_output.md/.json/.tsv` — WT vs integration per-layer scores\n"
        f"- `{spec['html']}` — IGV browser overlay\n"
    ))
    return cells


def _cells_multioracle(spec: dict) -> list[dict]:
    cells: list[dict] = []
    cells.append(_md(
        f"# {spec['name']}\n\n"
        f"Reproduces the multi-oracle consensus walkthrough: scores\n"
        f"**{spec['position']} {spec['ref']}>{spec['alt']}** with three\n"
        f"oracles in turn — ChromBPNet (DNase HepG2 / 1bp resolution), LegNet\n"
        f"(LentiMPRA / promoter activity), AlphaGenome (multi-layer at HepG2) —\n"
        f"and writes per-oracle JSON + a consolidated HTML report.\n\n"
        f"All three oracles run in their own mamba envs via\n"
        f"`use_environment=True`; the notebook stays in base chorus.\n"
    ))
    cells.append(_imports_cell(extra=[
        "from chorus.analysis.variant_report import build_variant_report",
        "from chorus.analysis.analysis_request import AnalysisRequest",
    ]))
    cells.append(_code(
        "WALKTHROUGH_DIR = Path.cwd()\n"
        "print(f\"Writing artifacts to: {WALKTHROUGH_DIR}\")"
    ))
    cells.append(_code(
        "# Shared inputs across all three oracles.\n"
        f"position = {spec['position']!r}\n"
        f"ref_allele = {spec['ref']!r}\n"
        f"alt_alleles = [{spec['alt']!r}]\n"
        f"gene_name = {spec['gene']!r}\n"
        "_chrom, _pos = position.split(\":\")\n"
        "_pos = int(_pos)"
    ))
    # ── ChromBPNet block
    cells.append(_md("## 1. ChromBPNet — DNase HepG2 (1bp resolution)"))
    cells.append(_code(
        "# ChromBPNet: load one model per (assay, cell_type, fold) and score.\n"
        "chrombpnet_oracle = chorus.create_oracle(\n"
        "    oracle_name=\"chrombpnet\",\n"
        "    use_environment=True,\n"
        ")\n"
        "chrombpnet_oracle.load_pretrained_model(\n"
        f"    assay={spec['chrombpnet_assay']!r},\n"
        f"    cell_type={spec['chrombpnet_cell_type']!r},\n"
        "    fold=0,\n"
        ")\n"
        "chrombpnet_region = f\"{_chrom}:{_pos - 100}-{_pos + 100}\"\n"
        "chrombpnet_result = chrombpnet_oracle.predict_variant_effect(\n"
        "    genomic_region=chrombpnet_region,\n"
        "    variant_position=position,\n"
        "    alleles=[ref_allele] + alt_alleles,\n"
        "    assay_ids=[],\n"
        ")\n"
        "chrombpnet_report = build_variant_report(\n"
        "    variant_result=chrombpnet_result,\n"
        "    oracle_name=\"chrombpnet\",\n"
        "    gene_name=gene_name,\n"
        "    normalizer=get_normalizer(oracle_name=\"chrombpnet\"),\n"
        "    igv_raw=False,\n"
        ")\n"
        "WALKTHROUGH_DIR.joinpath(\"chrombpnet_variant_report.json\").write_text(\n"
        "    json.dumps(chrombpnet_report.to_dict(), indent=2, default=str)\n"
        ")\n"
        "chrombpnet_report.to_html(\n"
        "    output_path=str(WALKTHROUGH_DIR / \"rs12740374_SORT1_chrombpnet_report.html\")\n"
        ")"
    ))
    # ── LegNet block
    cells.append(_md("## 2. LegNet — LentiMPRA promoter activity (HepG2)"))
    cells.append(_code(
        "# LegNet: cell-type + assay are constructor args (one model per cell_type).\n"
        "from chorus.oracles.legnet import LegNetOracle\n"
        "legnet_oracle = LegNetOracle(\n"
        f"    cell_type={spec['legnet_cell_type']!r},\n"
        f"    assay={spec['legnet_assay']!r},\n"
        "    use_environment=True,\n"
        ")\n"
        "legnet_oracle.load_pretrained_model()\n"
        "legnet_half = legnet_oracle.sequence_length // 2\n"
        "legnet_region = f\"{_chrom}:{max(1, _pos - legnet_half)}-{_pos + legnet_half}\"\n"
        "legnet_result = legnet_oracle.predict_variant_effect(\n"
        "    genomic_region=legnet_region,\n"
        "    variant_position=position,\n"
        "    alleles=[ref_allele] + alt_alleles,\n"
        "    assay_ids=[legnet_oracle.assay_id],\n"
        ")\n"
        "legnet_report = build_variant_report(\n"
        "    variant_result=legnet_result,\n"
        "    oracle_name=\"legnet\",\n"
        "    gene_name=gene_name,\n"
        "    normalizer=get_normalizer(oracle_name=\"legnet\"),\n"
        "    igv_raw=False,\n"
        ")\n"
        "WALKTHROUGH_DIR.joinpath(\"legnet_variant_report.json\").write_text(\n"
        "    json.dumps(legnet_report.to_dict(), indent=2, default=str)\n"
        ")\n"
        "legnet_report.to_html(\n"
        "    output_path=str(WALKTHROUGH_DIR / \"rs12740374_SORT1_legnet_report.html\")\n"
        ")"
    ))
    # ── AlphaGenome block
    cells.append(_md("## 3. AlphaGenome — full multi-layer (HepG2)"))
    cells.append(_code(
        f"ag_assay_ids = {_repr_list(spec['ag_assay_ids'])}\n"
        "ag_oracle = chorus.create_oracle(\n"
        "    oracle_name=\"alphagenome\",\n"
        "    use_environment=True,\n"
        ")\n"
        "ag_oracle.load_pretrained_model()\n"
        "ag_half = ag_oracle.output_size // 2\n"
        "ag_region = f\"{_chrom}:{max(1, _pos - ag_half)}-{_pos + ag_half}\"\n"
        "ag_result = ag_oracle.predict_variant_effect(\n"
        "    genomic_region=ag_region,\n"
        "    variant_position=position,\n"
        "    alleles=[ref_allele] + alt_alleles,\n"
        "    assay_ids=ag_assay_ids,\n"
        ")\n"
        "ag_report = build_variant_report(\n"
        "    variant_result=ag_result,\n"
        "    oracle_name=\"alphagenome\",\n"
        "    gene_name=gene_name,\n"
        "    normalizer=get_normalizer(oracle_name=\"alphagenome\"),\n"
        "    igv_raw=False,\n"
        ")\n"
        "WALKTHROUGH_DIR.joinpath(\"alphagenome_variant_report.json\").write_text(\n"
        "    json.dumps(ag_report.to_dict(), indent=2, default=str)\n"
        ")\n"
        "ag_report.to_html(\n"
        "    output_path=str(WALKTHROUGH_DIR / \"rs12740374_SORT1_alphagenome_report.html\")\n"
        ")"
    ))
    # ── Consolidation
    cells.append(_md("## 4. Consolidate into a single multi-oracle report"))
    cells.append(_code(
        "# The unified HTML overlays the three oracles' tracks in one IGV browser\n"
        "# with a consensus matrix at the top. scripts/regenerate_multioracle.py\n"
        "# `--consolidate` does this from the per-oracle JSONs that the cells\n"
        "# above just wrote. Re-use the same code path.\n"
        "from chorus.analysis.multi_oracle_report import build_multi_oracle_report\n"
        "oracle_reports = {\n"
        "    \"chrombpnet\": chrombpnet_report,\n"
        "    \"legnet\": legnet_report,\n"
        "    \"alphagenome\": ag_report,\n"
        "}\n"
        "consolidated = build_multi_oracle_report(\n"
        "    reports=oracle_reports,\n"
        "    gene_name=gene_name,\n"
        ")\n"
        f"WALKTHROUGH_DIR.joinpath(\"example_output.md\").write_text(consolidated.to_markdown())\n"
        f"WALKTHROUGH_DIR.joinpath(\"example_output.json\").write_text(\n"
        f"    json.dumps(consolidated.to_dict(), indent=2, default=str)\n"
        f")\n"
        f"consolidated.to_html(\n"
        f"    output_path=str(WALKTHROUGH_DIR / {spec['consolidated_html']!r})\n"
        f")\n"
        "print(f\"Wrote consolidated report to {WALKTHROUGH_DIR}\")"
    ))
    cells.append(_md(
        "## What this notebook produced\n\n"
        "- `chrombpnet_variant_report.json`, `legnet_variant_report.json`,\n"
        "  `alphagenome_variant_report.json` — per-oracle structured reports\n"
        "- `rs12740374_SORT1_{oracle}_report.html` — per-oracle IGV HTML (×3)\n"
        f"- `example_output.md/.json` + `{spec['consolidated_html']}` — consolidated multi-oracle view\n"
    ))
    return cells


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------

_BUILDERS = {
    "analyze_variant_multilayer": _cells_analyze_variant_multilayer,
    "discover_variant": _cells_discover_variant,
    "discover_variant_cell_types": _cells_discover_variant_cell_types,
    "fine_map_causal_variant": _cells_fine_map_causal_variant,
    "score_variant_batch": _cells_score_variant_batch,
    "analyze_region_swap": _cells_analyze_region_swap,
    "simulate_integration": _cells_simulate_integration,
    "multioracle": _cells_multioracle,
}


def _build_notebook(spec: dict) -> Any:
    """Construct an `nbformat` notebook for a single walkthrough spec."""
    tool = spec["mcp_tool"]
    if tool == "analyze_variant_multilayer" and spec["oracle"] == "chrombpnet":
        cells = _cells_chrombpnet_variant(spec)
    elif tool in _BUILDERS:
        cells = _BUILDERS[tool](spec)
    else:
        raise ValueError(f"Unknown mcp_tool: {tool!r} for {spec['name']}")

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (chorus)",
            "language": "python",
            "name": "chorus",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10",
        },
    }
    return nb


def main() -> int:
    n_written = 0
    for spec in WALKTHROUGHS:
        out_dir = REPO_ROOT / spec["dir"]
        if not out_dir.is_dir():
            print(f"SKIP (dir missing): {spec['dir']}", file=sys.stderr)
            continue
        nb = _build_notebook(spec)
        out_path = out_dir / "notebook.ipynb"
        with out_path.open("w") as fh:
            nbf.write(nb, fh)
        n_written += 1
        print(f"  ✓ {spec['dir']}/notebook.ipynb  ({len(nb.cells)} cells)")
    print(f"\nWrote {n_written}/{len(WALKTHROUGHS)} notebooks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
