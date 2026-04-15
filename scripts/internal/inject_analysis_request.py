"""Retrofit an ``Analysis Request`` block into existing example outputs.

For every example in ``examples/applications/``, this script:

1. Reads ``example_output.json`` (or ``discovery_summary.json``).
2. Builds an :class:`AnalysisRequest` with a synthesized natural-language
   prompt appropriate for that example (no brackets — these read like real
   user questions).
3. Prepends the rendered markdown block to ``example_output.md``.
4. Injects the HTML fragment into every ``*.html`` file just after the
   first ``<h1>...</h1>`` tag.
5. Writes the ``analysis_request`` dict back into the JSON.

This lets us add the "what was asked" context to *all existing outputs*
without rerunning the oracle — critical because regenerating AlphaGenome
reports takes hours per example.

Usage::

    python scripts/inject_analysis_request.py               # all examples
    python scripts/inject_analysis_request.py --dry-run     # preview only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/PHShome/lp698/chorus")

from chorus.analysis.analysis_request import AnalysisRequest  # noqa: E402

BASE = Path("/PHShome/lp698/chorus/examples/applications")


# ── Example metadata --------------------------------------------------------
#
# Each entry captures the natural-language question that motivated the demo
# and the tool / context that produced the outputs. Keep prompts short and
# conversational — they're meant to *demonstrate* how a user would ask.

EXAMPLES: dict[str, dict] = {
    # ── variant_analysis ────────────────────────────────────────────────
    "variant_analysis/SORT1_rs12740374": {
        "user_prompt": (
            "Load AlphaGenome and analyze rs12740374 (chr1:109274968 G>T) "
            "in HepG2 liver cells. The gene is SORT1 — I want to understand "
            "whether this variant changes chromatin accessibility, CEBP binding, "
            "H3K27ac, and SORT1 expression."
        ),
        "tool_name": "analyze_variant_multilayer",
        "oracle_name": "alphagenome",
        "tracks_requested": "all oracle tracks (discovery mode)",
        "notes": [
            "Ref/alt notation: dbSNP lists rs12740374 as C>T on the plus strand; "
            "chr1:109274968 G>T here is equivalent (hg38 + strand).",
            "The report shows the top tracks per regulatory layer. CEBPA/CEBPB "
            "tracks exist in AlphaGenome's catalog but did not rank in the top "
            "of liver ChIP-TF for this variant; inspect the full JSON or call "
            "`analyze_variant_multilayer` with an explicit CEBPA/CEBPB assay_id "
            "to score them directly.",
        ],
    },
    "variant_analysis/BCL11A_rs1427407": {
        "user_prompt": (
            "I'm studying fetal hemoglobin reactivation for sickle-cell therapy. "
            "Can you analyze rs1427407 (chr2:60490908 G>T) in erythroid cells? "
            "The target gene is BCL11A."
        ),
        "tool_name": "analyze_variant_multilayer",
        "oracle_name": "alphagenome",
        "tracks_requested": "all oracle tracks (discovery mode)",
        "notes": [
            "Published mechanism: Bauer et al. Science 2013 identified this "
            "variant as disrupting a GATA1/TAL1 binding motif in the +58 "
            "erythroid-specific enhancer of BCL11A, reducing BCL11A expression "
            "and enabling HbF reactivation. TAL1 loss is visible in the ChIP-TF "
            "ranking here (negative effect). GATA1 itself may not rank in the "
            "top tracks because AlphaGenome's GATA1 ChIP coverage is limited "
            "to specific cell lines; score GATA1 tracks explicitly for a "
            "stronger signal.",
        ],
    },
    "variant_analysis/FTO_rs1421085": {
        "user_prompt": (
            "Analyze the obesity-associated FTO variant rs1421085 "
            "(chr16:53767042 T>C) in adipocyte tracks. IRX3/IRX5 are the "
            "proposed target genes — show me the chromatin and TF binding effects."
        ),
        "tool_name": "analyze_variant_multilayer",
        "oracle_name": "alphagenome",
        "tracks_requested": "all oracle tracks (discovery mode)",
        "notes": [
            "Published mechanism: Claussnitzer et al. NEJM 2015 showed that "
            "the risk allele (C) disrupts an ARID5B repressor motif, "
            "de-repressing IRX3 and IRX5 in adipocyte progenitors. ARID5B "
            "ChIP data is not well represented in AlphaGenome's training set, "
            "so the direct ARID5B binding-loss signal may be absent from the "
            "top tracks. IRX3/IRX5 expression effects are in the gene "
            "expression layer — check the RNA rows for those genes directly.",
        ],
    },
    "variant_analysis/SORT1_enformer": {
        "user_prompt": (
            "Run discovery mode on rs12740374 (chr1:109274968 G>T) using "
            "Enformer — I want to compare its predictions to AlphaGenome "
            "on the same variant. Gene is SORT1."
        ),
        "tool_name": "discover_variant",
        "oracle_name": "enformer",
        "tracks_requested": "all Enformer tracks",
    },
    "variant_analysis/SORT1_chrombpnet": {
        "user_prompt": (
            "Load ChromBPNet with the HepG2 ATAC model and score rs12740374 "
            "(chr1:109274968 G>T) — I want base-resolution chromatin effects "
            "around the CEBP motif."
        ),
        "tool_name": "analyze_variant_multilayer",
        "oracle_name": "chrombpnet",
        "tracks_requested": "ATAC:HepG2",
    },
    # ── validation ────────────────────────────────────────────────
    "validation/SORT1_rs12740374_with_CEBP": {
        "user_prompt": (
            "Replicate the SORT1 rs12740374 result from the AlphaGenome Nature "
            "paper (Avsec et al. 2026). I want to see the HepG2 DNASE, CEBPA/CEBPB "
            "ChIP, H3K27ac, and SORT1 RNA tracks to verify the paper's findings."
        ),
        "tool_name": "analyze_variant_multilayer",
        "oracle_name": "alphagenome",
        "tracks_requested": "HepG2 DNASE / CEBP ChIP / H3K27ac / CAGE / RNA",
    },
    "validation/TERT_chr5_1295046": {
        "user_prompt": (
            "Validate the TERT chr5:1295046 T>G finding from the AlphaGenome "
            "paper in melanocytes. Gene is TERT — I want to see the "
            "chromatin and ETS TF binding effects."
        ),
        "tool_name": "analyze_variant_multilayer",
        "oracle_name": "alphagenome",
        "tracks_requested": "all oracle tracks (discovery mode)",
    },
    # ── discovery ────────────────────────────────────────────────
    "discovery/SORT1_cell_type_screen": {
        "user_prompt": (
            "Screen all cell types to find where rs12740374 (chr1:109274968 G>T) "
            "has the strongest regulatory impact. I don't have a specific tissue "
            "in mind — let the model tell me where this variant matters most. "
            "Gene is SORT1."
        ),
        "tool_name": "discover_variant_cell_types",
        "oracle_name": "alphagenome",
        "tracks_requested": "all DNASE/ATAC tracks (~472 cell types)",
    },
    # ── causal_prioritization ────────────────────────────────────────────────
    "causal_prioritization/SORT1_locus": {
        "user_prompt": (
            "I'm fine-mapping the SORT1 LDL cholesterol GWAS locus. The sentinel "
            "is rs12740374 and I have 11 LD variants (r²≥0.85). Score each one "
            "across all regulatory layers in HepG2 and rank by causal evidence. "
            "Gene is SORT1."
        ),
        "tool_name": "fine_map_causal_variant",
        "oracle_name": "alphagenome",
        "tracks_requested": "all oracle tracks (discovery mode)",
    },
    # ── batch_scoring ────────────────────────────────────────────────
    "batch_scoring": {
        "user_prompt": (
            "I have 5 candidate SNPs from the SORT1 GWAS locus. Score all of "
            "them in HepG2 with AlphaGenome and rank by regulatory effect size. "
            "Target gene SORT1."
        ),
        "tool_name": "score_variant_batch",
        "oracle_name": "alphagenome",
        "tracks_requested": "all oracle tracks (discovery mode)",
    },
    # ── sequence_engineering ────────────────────────────────────────────────
    "sequence_engineering/region_swap": {
        "user_prompt": (
            "Can you simulate replacing the chr1:109274500-109275500 region "
            "with a strong K562 promoter sequence? I want to see how the "
            "swap would change DNASE, H3K27ac, H3K4me3, and CAGE around the "
            "SORT1 enhancer."
        ),
        "tool_name": "analyze_region_swap",
        "oracle_name": "alphagenome",
        "tracks_requested": "DNASE:K562, H3K27ac:K562, H3K4me3:K562, CAGE:K562",
    },
    "sequence_engineering/integration_simulation": {
        "user_prompt": (
            "Simulate inserting a CMV promoter construct at chr19:55115000 "
            "(PPP1R12C locus) and predict the local disruption in K562 — "
            "I want to see chromatin, H3K27ac, and CAGE effects from the insertion."
        ),
        "tool_name": "simulate_integration",
        "oracle_name": "alphagenome",
        "tracks_requested": "DNASE:K562, H3K27ac:K562, CAGE:K562",
    },
}


HEADER_PATTERNS_TO_STRIP = [
    re.compile(r"^## Analysis Request.*?(?=\n## |\n---|$)", re.DOTALL),
]

# Any previous interpretation block is stripped before we re-append,
# so the script remains idempotent.
INTERPRETATION_PATTERN = re.compile(
    r"\n+## Interpretation.*?(?=\Z)",
    re.DOTALL,
)


# ── Interpretation text per example ──────────────────────────────────────────
#
# Each block is a 3-part structure:
#   1. "What the oracle sees" — a plain-English reading of what the numbers
#      in the report actually say (top effect, direction, which layers).
#   2. "How this fits the published biology" — honest comparison with the
#      known mechanism; flags where the oracle agrees or disagrees and why.
#   3. "Suggested next steps" — concrete follow-up actions a user could
#      take, framed around the Chorus tools available.
#
# Tone: modest, honest, cites caveats. We never overclaim the model.

INTERPRETATIONS: dict[str, str] = {
    "variant_analysis/SORT1_rs12740374": """\
**What the oracle sees.** The alt allele (T) produces a very strong
*opening* signal in chromatin accessibility (top DNASE effects of +1.9
log2FC in liver-adjacent cell types), a strong gain in liver TF binding
(RXRA, SP1, HNF4A), a strong gain in active-promoter histone marks
(H3K27ac, H3K4me3), and a strong increase in CAGE at the SORT1 /
PSRC1 / CELSR2 promoter. All four layers converge on the same
direction, which is the signature of a single regulatory disruption
acting on multiple readouts.

**How this fits the published biology.** Musunuru et al. (Nature 2010)
showed that the minor allele T creates a C/EBP binding site in a liver
enhancer and *increases* SORT1 expression in hepatocytes, lowering plasma
LDL. The direction and multi-layer convergence Chorus reports match the
paper. CEBPA/CEBPB are not in the top 3 ChIP-TF tracks here — AlphaGenome
ranked RXRA/SP1/HNF4A higher in the all-tracks discovery mode. This is a
ranking limitation, not a disagreement: liver CEBP tracks exist in the
oracle catalog and can be scored explicitly.

**Suggested next steps.**
- Re-run with explicit CEBPA / CEBPB / C/EBP tracks in HepG2 to confirm
  the specific TF mechanism (`analyze_variant_multilayer` with
  `assay_ids=["CHIP_TF/... CEBPA ...", "CHIP_TF/... CEBPB ..."]`).
- Compare to a ChromBPNet run anchored on HepG2 ATAC + CEBPA ChIP for
  base-resolution motif effects.
- If you're fine-mapping the LDL-C GWAS locus, use the worked
  [causal_prioritization/SORT1_locus](../../causal_prioritization/SORT1_locus/)
  example as a template.
""",

    "variant_analysis/BCL11A_rs1427407": """\
**What the oracle sees.** Effect magnitudes are modest here: the top
TF-binding rows show TAL1 and CBFA2T3 losses in K562 and erythroid
progenitors in the 0.1–0.15 log2FC range, and the chromatin-accessibility
change is smaller still. No layer shows a "very strong" effect, and the
summary is deliberately labelled *strong binding loss* / *moderate
opening*, not catastrophic disruption.

**How this fits the published biology.** Bauer et al. (Science 2013)
traced rs1427407 to a GATA1/TAL1 motif in the +58 erythroid enhancer of
BCL11A; disrupting that motif reduces BCL11A expression and allows HbF
reactivation in adult red cells. Chorus reproduces the direction of the
TAL1 binding loss (good) but under-reports the magnitude and does not
surface GATA1 in the top rows. AlphaGenome's GATA1 ChIP coverage is
limited to a handful of cell lines, so the weak ranking here is an oracle
limitation rather than evidence the variant is inert.

**Suggested next steps.**
- Call `analyze_variant_multilayer` with explicit GATA1 ChIP tracks
  (K562, HUDEP-2-like) to test whether the mechanism is detectable when
  the right assays are forced into the top-N.
- Score the same variant with ChromBPNet loaded for an erythroid ATAC
  model for base-resolution motif disruption.
- Validate the predicted effect in a reporter assay or HUDEP-2 CRISPRi
  screen if you're prioritising it for a wet-lab follow-up.
""",

    "variant_analysis/FTO_rs1421085": """\
**What the oracle sees.** Effects are modest (top raw scores in the 0.1
to 0.3 log2FC range) and span multiple TF, chromatin, and histone tracks
in neuroblastoma and several other lineages. IRX3 / IRX5 expression
changes appear in the RNA layer but in the smallest-effect category.
None of the layers reach "very strong" by our magnitude-gated labels.

**How this fits the published biology.** Claussnitzer et al.
(NEJM 2015) showed that the risk allele (C) disrupts an ARID5B repressor
motif in adipocyte progenitors, de-repressing IRX3 and IRX5 and driving
a thermogenesis-to-lipid-storage switch. Chorus agrees with the modesty
of the effect but does not recover the ARID5B mechanism — ARID5B ChIP is
not well represented in AlphaGenome's training corpus, and adipocyte
progenitor tracks are sparse. This is a case where the oracle's cell
type coverage lags what the paper used.

**Suggested next steps.**
- Use `discover_variant_cell_types` to confirm whether any of the
  available adipose / preadipocyte tracks in AlphaGenome show a stronger
  signal than the current default top hits.
- Score the variant with a LegNet MPRA model or a ChromBPNet adipocyte
  ATAC model for a second opinion.
- If IRX3 / IRX5 expression is the readout you care about, query
  `analyze_variant_multilayer` with `gene_name="IRX3"` (and separately
  "IRX5") so the RNA layer is scored at those specific TSSs.
""",

    "variant_analysis/SORT1_enformer": """\
**What the oracle sees.** Enformer shows the same signal Musunuru reported
and AlphaGenome reproduces in the main SORT1 example: a strong DNASE
opening around the variant, gain of liver ChIP-TF signal, and a CAGE
increase at the SORT1 promoter. Effect magnitudes are in the "strong"
(0.3–0.7) to "very strong" (>0.7) range depending on the track.

**How this fits the published biology.** The direction of effect matches
Musunuru et al. 2010 and is consistent with the AlphaGenome result on the
same variant. Enformer has narrower input context (114 kb output window)
than AlphaGenome (1 Mb), which is why the top tracks differ slightly
between the two oracles — Enformer is more conservative about distal TSS
scoring.

**Suggested next steps.**
- Compare side-by-side with the AlphaGenome SORT1 example. Agreement
  between two independent architectures is a stronger signal than either
  alone.
- If your target gene is far from the variant (>100 kb), prefer Borzoi
  or AlphaGenome — Enformer's 114 kb output window will miss it.
""",

    "variant_analysis/SORT1_chrombpnet": """\
**What the oracle sees.** ChromBPNet scores the variant on one specific
assay/cell-type combination (HepG2 ATAC). The effect is modest and
layer-limited to chromatin accessibility because ChromBPNet is a
single-assay base-resolution model.

**How this fits the published biology.** A modest HepG2 chromatin
effect is consistent with the classic C/EBP site creation story:
ChromBPNet captures the motif disruption at base resolution but only
reports one layer. Treat this as a complement — not a replacement — to
the multi-layer AlphaGenome result on the same variant.

**Suggested next steps.**
- Load `chrombpnet` with `assay="CHIP"`, `cell_type="HepG2"`,
  `TF="CEBPA"` for a direct motif-creation readout.
- Use this model for high-throughput VCF triage, then escalate the top
  hits to AlphaGenome for multi-layer analysis.
""",

    "validation/SORT1_rs12740374_with_CEBP": """\
**What the oracle sees.** This validation example forces the CEBP /
HepG2 tracks the Musunuru paper used. The direction (chromatin opening,
TF binding gain, H3K27ac gain) reproduces the main SORT1 result. Effects
are in the strong range.

**How this fits the published biology.** Good agreement with the paper
in direction across all forced layers. The forced-track approach is the
recommended pattern when you already know which assays matter and want
a cleaner, biology-driven readout than full discovery mode.

**Suggested next steps.**
- This is the pattern to copy for other published variants where you
  have a priori knowledge of the relevant TFs or cell types.
- Compare the forced-track output to the discovery-mode output in the
  sibling `variant_analysis/SORT1_rs12740374` folder to see how the two
  approaches surface different signals.
""",

    "validation/TERT_chr5_1295046": """\
**What the oracle sees.** This distal TERT variant (chr5:1295046 T>G)
shows a clean *gain* signal
across all four multi-layer layers: modest DNASE opening, strong TF
binding gain in the ChIP-TF rows, strong histone mark gain, and a
moderate CAGE increase. Effects are in the 0.2–0.5 log2FC range.

**How this fits the published biology.** The validation set from the
AlphaGenome Nature paper (Avsec et al. 2026) includes this variant as a
positive control for distal ETS-family binding-site creation, and our
Chorus output matches the paper's direction of effect.

**Suggested next steps.**
- Use this example as a reference for "what a well-behaved distal
  regulatory variant looks like in a Chorus report" when training
  yourself or colleagues on how to read the tables.
- When in doubt about a TERT variant, score multiple nearby positions
  and compare directions — consistent gain or loss across neighbouring
  sites is more trustworthy than a single prediction.
""",

    "discovery/SORT1_cell_type_screen": """\
**What the oracle sees.** Screening all ~472 cell types for rs12740374
returns LNCaP (prostate, +1.9 log2FC), epithelial cell of proximal
tubule (+1.6), and renal cortical epithelial (+1.5) as the top hits for
chromatin accessibility. Notably, liver / HepG2 is NOT in the top 3 even
though the published mechanism is a liver CEBP site.

**How this fits the published biology.** The Musunuru paper localises
the effect to hepatocytes. That LNCaP and kidney cell types out-rank
HepG2 here is a reminder that AlphaGenome's cell-type-level rankings
depend on each cell type's *available* DNASE track quality; a single
strong LNCaP DNASE signal can outrank liver when liver tracks are
sparser or noisier. The variant DOES open chromatin in every top cell
type — the direction is correct, but the absolute ranking should be
taken with a grain of salt.

**Suggested next steps.**
- Use this screen as a *filter*, not an answer. Any variant that reaches
  |log2FC| > 0.3 in a discovery screen is worth a full multi-layer
  analysis in each of the top hit cell types.
- For the specific case of SORT1, the biological answer is known to be
  hepatocyte-driven — treat the LNCaP result as a model quirk and
  prioritise the hepatocyte multi-layer report.
""",

    "causal_prioritization/SORT1_locus": """\
**What the oracle sees.** Across 11 LD variants (r²≥0.85) the sentinel
rs12740374 wins decisively: composite score 0.954, maximum effect +1.9
log2FC, effects in **5 of 5** scored regulatory layers, directional
convergence 1.00 (every layer agrees on direction). The next-best
variant, rs660240, scores 0.545 with a +0.21 log2FC effect in only 3
layers. All other variants are well below 0.5.

**How this fits the published biology.** This is exactly the pattern
published fine-mapping studies found for the SORT1 locus: rs12740374 is
the single causal variant despite being in perfect LD with several
other SNPs. Chorus's composite causal score recovers it by combining
effect magnitude with multi-layer convergence, which is the same
intuition wet-lab functional assays encode.

**Suggested next steps.**
- This is the template for any GWAS fine-mapping task. Replace the
  lead variant with your own and rerun — if one variant in your
  credible set has composite >> 0.7 and multi-layer convergence = 1.0,
  that's your prime functional candidate.
- Inspect `result.scores[0].per_layer_scores` in the JSON for the
  mechanism: here the top layer is `chromatin_accessibility`, which
  matches the published C/EBP-mediated enhancer opening.
- For truly ambiguous loci, run the same fine-mapping with two oracles
  (AlphaGenome + Borzoi) and intersect the top candidate — agreement
  between independent architectures strengthens the conclusion.
""",

    "batch_scoring": """\
**What the oracle sees.** Out of 5 SORT1-locus SNPs, rs12740374 is the
clear top hit with a +1.9 log2FC chromatin effect, 10x larger than the
next-strongest variant. The other 4 variants have |max_effect| in the
0.2 range and would not reach any meaningful regulatory threshold on
their own.

**How this fits the published biology.** Consistent with the
fine-mapping result: one functional variant among several in LD. Batch
scoring is the fastest way to triage a credible set or a candidate list
from a VCF — the top variant is the one to escalate to full multi-layer
analysis.

**Suggested next steps.**
- For any list of >5 variants, run this batch-scoring step first, sort
  by `max_effect`, and only run the full `analyze_variant_multilayer`
  on the top 1–3 hits. This saves 5–10x compute.
- The `Top Track` column uses AlphaGenome's native assay IDs (e.g.
  `DNASE/EFO:0005726 DNase-seq/.`). Look up the cell type via
  `list_tracks(oracle_name="alphagenome", filter="EFO:0005726")` or in
  the HTML report column.
""",

    "sequence_engineering/region_swap": """\
**What the oracle sees.** Replacing the SORT1 enhancer region with the
prepared K562 promoter sequence produces strong chromatin closing and
H3K27ac loss at the swap site and a very strong CAGE decrease at the
SORT1 TSS. The predicted effect is a *loss* of regulatory activity.

**How this fits expectations.** The swap removes a well-characterised
liver enhancer and replaces it with an out-of-context promoter
sequence. In the K562 (erythroid leukaemia) context used by these
assays, the inserted sequence is not a functional K562 promoter,
so the loss-of-function prediction is biologically consistent with
"removed the real signal and replaced it with a weaker one".

**Suggested next steps.**
- This is a *loss-of-function* demonstration, not a gain-of-function
  one. For a gain demo, swap in a known strong K562 promoter (e.g. the
  MYB intron-1 enhancer) into a normally silent locus.
- The same workflow (`analyze_region_swap`) can be used for enhancer
  grafting experiments, reporter construct design, and enhancer
  deletion simulation.
""",

    "sequence_engineering/integration_simulation": """\
**What the oracle sees.** Inserting a CMV promoter construct at
chr19:55115000 produces localised chromatin and CAGE increases at the
insertion site and modest collateral effects on the surrounding
regulatory landscape. Effects are tightly localised — the oracle does
not predict widespread disruption of neighbouring genes.

**How this fits expectations.** Integration of a strong constitutive
promoter into a permissive locus is predicted to create a local
transcriptionally active island without major effects on the
surrounding genes — consistent with what is known about AAVS1-like safe
harbour integration.

**Suggested next steps.**
- Use this example as a template for predicting safe-harbour integration
  effects before committing to a gene-therapy construct.
- Compare predicted versus wet-lab signals from the specific integration
  site, if available. The oracle should agree with the published
  promoter-insertion-at-safe-harbour datasets.
""",
}


def _file_mtime(path: Path) -> str:
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def _build_request(entry: dict, mtime_str: str) -> AnalysisRequest:
    return AnalysisRequest(
        user_prompt=entry["user_prompt"],
        tool_name=entry.get("tool_name"),
        oracle_name=entry.get("oracle_name"),
        normalizer_name="per-track background CDFs",
        tracks_requested=entry.get("tracks_requested"),
        cell_types=list(entry.get("cell_types") or []),
        notes=list(entry.get("notes") or []),
        generated_at=mtime_str,
    )


def _prepend_markdown(md_path: Path, request_block: str) -> None:
    text = md_path.read_text()
    # Strip any previous Analysis Request block to stay idempotent
    for pat in HEADER_PATTERNS_TO_STRIP:
        text = pat.sub("", text).lstrip()
    md_path.write_text(f"{request_block}\n{text}")


def _append_interpretation_markdown(md_path: Path, interpretation: str) -> None:
    """Append (or replace) the ``## Interpretation`` block at the end of MD."""
    text = md_path.read_text()
    text = INTERPRETATION_PATTERN.sub("", text).rstrip()
    body = (
        "\n\n---\n\n## Interpretation\n\n"
        + interpretation.rstrip()
        + "\n"
    )
    md_path.write_text(text + body)


def _append_interpretation_html(html_path: Path, interpretation_md: str) -> None:
    """Insert an ``Interpretation`` block into the HTML body (before </body>).

    Converts the markdown interpretation to minimal HTML (paragraphs and
    bulleted lists) without pulling in a full markdown renderer.
    """
    import html as _html

    text = html_path.read_text()
    # Remove any previously injected Interpretation section
    text = re.sub(
        r'<section class="interpretation".*?</section>',
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )

    # Cheap markdown → HTML: split on blank lines into blocks, render <ul> for
    # contiguous "- " lines, <p> otherwise. Bold survives via <strong>.
    blocks: list[str] = []
    for chunk in re.split(r"\n\s*\n", interpretation_md.strip()):
        lines = chunk.splitlines()
        if all(ln.lstrip().startswith("- ") for ln in lines if ln.strip()):
            items = "".join(
                f"<li>{_bold_escape(ln.lstrip()[2:])}</li>"
                for ln in lines if ln.strip()
            )
            blocks.append(f"<ul>{items}</ul>")
        else:
            blocks.append(f"<p>{_bold_escape(' '.join(ln.strip() for ln in lines))}</p>")

    body = (
        '<section class="interpretation" '
        'style="margin:24px 0;padding:16px 20px;background:#f6f8fa;'
        'border-left:4px solid #1f883d;border-radius:4px;">'
        '<h3 style="margin:0 0 8px 0;">Interpretation</h3>'
        + "".join(blocks)
        + "</section>"
    )

    # Use rfind to insert before the LAST </body> — not the first, which
    # may be inside minified JS (DOMPurify, IGV.js contain literal HTML
    # tag strings that would match a naive str.replace).
    idx = text.rfind("</body>")
    if idx >= 0:
        text = text[:idx] + body + "\n" + text[idx:]
    else:
        text = text + body
    html_path.write_text(text)


def _bold_escape(s: str) -> str:
    """Escape HTML entities but keep `**bold**` → <strong>."""
    import html as _html

    # Escape everything first
    safe = _html.escape(s)
    # Re-introduce bold
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    # Keep literal `code`
    safe = re.sub(r"`([^`]+)`", r"<code>\1</code>", safe)
    return safe


def _inject_html(html_path: Path, fragment: str) -> None:
    text = html_path.read_text()
    # Strip previous injected section
    text = re.sub(
        r'<section class="analysis-request".*?</section>',
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )
    # Insert after first </h1>
    m = re.search(r"</h1>", text)
    if not m:
        return
    idx = m.end()
    new_text = text[:idx] + "\n" + fragment + "\n" + text[idx:]
    html_path.write_text(new_text)


def _augment_json(json_path: Path, request: AnalysisRequest) -> None:
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return
    data["analysis_request"] = request.to_dict()
    json_path.write_text(json.dumps(data, indent=2, default=str))


def process_example(rel_path: str, entry: dict, dry_run: bool) -> None:
    example_dir = BASE / rel_path
    if not example_dir.exists():
        print(f"  skip (missing dir): {rel_path}")
        return

    md_path = example_dir / "example_output.md"
    json_path = example_dir / "example_output.json"
    html_paths = sorted(example_dir.glob("*.html"))

    # Use the markdown file's mtime as "generated" timestamp (falls back to
    # json, then now) so the displayed time reflects when the numbers were
    # actually produced.
    for src in (md_path, json_path):
        if src.exists():
            mtime_str = _file_mtime(src)
            break
    else:
        mtime_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    request = _build_request(entry, mtime_str)
    md_block = request.to_markdown()
    html_fragment = request.to_html_fragment()

    actions: list[str] = []
    if md_path.exists():
        actions.append(f"prepend MD {md_path.name}")
    if json_path.exists():
        actions.append(f"augment JSON {json_path.name}")
    for hp in html_paths:
        actions.append(f"inject HTML {hp.name}")

    print(f"  {rel_path}: {', '.join(actions) if actions else '(no files)'}")
    if dry_run:
        return

    if md_path.exists():
        _prepend_markdown(md_path, md_block)
    if json_path.exists():
        _augment_json(json_path, request)
    for hp in html_paths:
        _inject_html(hp, html_fragment)

    # Append "Interpretation" section if we have one for this example
    interpretation = INTERPRETATIONS.get(rel_path)
    if interpretation:
        if md_path.exists():
            _append_interpretation_markdown(md_path, interpretation)
        for hp in html_paths:
            _append_interpretation_html(hp, interpretation)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        help="Only process the example at this relative path (e.g. variant_analysis/SORT1_rs12740374)",
    )
    args = parser.parse_args()

    targets = [args.only] if args.only else list(EXAMPLES)
    print(f"Processing {len(targets)} example(s){' (dry-run)' if args.dry_run else ''}")
    for rel in targets:
        entry = EXAMPLES.get(rel)
        if entry is None:
            print(f"  unknown example: {rel}")
            continue
        process_example(rel, entry, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
