"""Shared glossary and formula-label helpers for all Chorus report renderers.

Every report (variant_report, causal, region_swap, integration, discovery,
batch_scoring) uses these helpers so that:

  * Effect-size formulas are labelled identically across reports.
  * A single "How to read this report" block is rendered with consistent
    wording and consistent CSS.
  * New users can understand what ``+0.3`` means without reading the code.

This module must not import from other ``chorus.analysis`` modules at import
time — ``LAYER_CONFIGS`` is resolved lazily inside :func:`render_how_to_read`
to avoid circular imports.
"""
from __future__ import annotations

from typing import Iterable


# ---------------------------------------------------------------------------
# Formula labels and human-readable meanings
# ---------------------------------------------------------------------------

_FORMULA_LABELS: dict[str, str] = {
    "log2fc": "log2FC",
    "logfc": "lnFC",
    "diff": "Δ (alt−ref)",
}


def formula_label(formula: str | None) -> str:
    """Short label for a layer's effect formula.

    Examples
    --------
    >>> formula_label("log2fc")
    'log2FC'
    >>> formula_label("logfc")
    'lnFC'
    >>> formula_label("diff")
    'Δ (alt−ref)'
    >>> formula_label("unknown")
    'unknown'
    >>> formula_label(None)
    '—'
    """
    if not formula:
        return "—"
    return _FORMULA_LABELS.get(formula, formula)


def formula_meaning(formula: str | None, pseudocount: float = 1.0) -> str:
    """Single-sentence HTML explanation of a layer's effect formula.

    The string contains inline ``<code>`` spans and may be embedded directly
    in a ``<dd>`` or ``<li>``.
    """
    eps = f"{pseudocount:g}" if pseudocount else "0"
    if formula == "log2fc":
        return (f"<code>log2((alt+{eps})/(ref+{eps}))</code>. "
                "+0.3 ≈ alt signal 1.23× ref; +1.0 ≈ 2× ref; "
                "negative values = loss of signal.")
    if formula == "logfc":
        return (f"<code>ln((alt+{eps})/(ref+{eps}))</code>. "
                "Natural-log fold change — the RNA-seq convention.")
    if formula == "diff":
        return ("<code>alt − ref</code>. Raw difference in predicted "
                "value (not a ratio). Useful when ref is near zero.")
    return formula or "—"


def formula_chip(formula: str | None) -> str:
    """Return a small inline ``<span class='formula-chip'>...</span>``.

    Callers are expected to include :data:`HOW_TO_READ_CSS` in their report
    ``<style>`` block so the chip gets rendered with a monospace background.
    """
    return f'<span class="formula-chip">{formula_label(formula)}</span>'


# ---------------------------------------------------------------------------
# CSS fragment (shared across every report)
# ---------------------------------------------------------------------------

HOW_TO_READ_CSS: str = """
.how-to-read { margin: 1rem 0; padding: 0.7rem 1rem; background: #fffbea;
               border-left: 3px solid #f0ad4e; border-radius: 4px;
               font-size: 0.9rem; color: #343a40; }
.how-to-read b { color: #212529; }
.how-to-read dl { margin: 0.4rem 0 0; display: grid;
                  grid-template-columns: max-content 1fr; gap: 0.25rem 0.8rem; }
.how-to-read dt { font-weight: 600; color: #343a40; }
.how-to-read dd { margin: 0; color: #495057; }
.how-to-read ul { margin: 0.2rem 0 0.2rem 1.1rem; padding: 0; }
.how-to-read code { background: #f1f3f5; padding: 0 0.25rem; border-radius: 3px;
                    font-size: 0.85em; font-family: ui-monospace, SFMono-Regular,
                    Menlo, monospace; }
.formula-chip { display: inline-block; background: #e9ecef; color: #495057;
                padding: 0 0.35rem; border-radius: 3px; font-size: 0.75rem;
                font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                margin-left: 0.2rem; vertical-align: middle; }
"""


# ---------------------------------------------------------------------------
# The glossary block
# ---------------------------------------------------------------------------

def render_how_to_read(
    *,
    layers_present: Iterable[str] | None = None,
    include_composite: bool = False,
    weights: object | None = None,
    include_percentile: bool = False,
    include_activity: bool = False,
    lead_sentence: str | None = None,
) -> str:
    """Return an HTML ``<section class="how-to-read">`` block.

    Parameters
    ----------
    layers_present:
        Iterable of ``LAYER_CONFIGS`` keys actually shown in this report.
        Each layer will appear as a bullet describing its effect formula.
        Pass ``None`` (the default) to list no per-layer formulas.
    include_composite:
        If True, emit the composite-score definition (used by the causal
        prioritization report). ``weights`` — when provided — is a
        :class:`~chorus.analysis.causal.CausalWeights` used to mention the
        ``layer_effect_threshold``.
    include_percentile:
        If True, emit the definition of the "Effect %ile" column.
    include_activity:
        If True, emit the definition of the "Activity %ile" column.
    lead_sentence:
        Override the default lead. Pass a short prompt like "Variants are
        ranked by...". The default is deliberately report-agnostic.
    """
    # Local import to break the import cycle with scorers -> (nothing) and to
    # keep this module free of analysis-package side effects.
    from .scorers import LAYER_CONFIGS

    parts: list[str] = []
    parts.append('<section class="how-to-read">')
    lead = lead_sentence or (
        "<b>How to read this report.</b> "
        "Every numeric effect below is produced by a deep-learning model. "
        "The scale depends on the regulatory layer — see the formulas below "
        "so you can interpret <code>+0.3</code> correctly."
    )
    parts.append(f"<p style='margin:0'>{lead}</p>")
    parts.append("<dl>")

    if include_composite:
        threshold = getattr(weights, "layer_effect_threshold", 0.1)
        parts.append(
            "<dt>Composite</dt><dd>Weighted combination of Max Effect, Layers, "
            "Convergence and Reference Activity, min–max scaled to 0–1 across "
            "the variants in this run. Higher = more multi-layer causal evidence.</dd>"
        )
        parts.append(
            "<dt>Max Effect</dt><dd>Signed effect of the single strongest track "
            "for a variant (across all layers). See layer row for the formula.</dd>"
        )
        parts.append(
            f"<dt>Layers</dt><dd>Number of regulatory layers with "
            f"|effect| ≥ {threshold:g}.</dd>"
        )
        parts.append(
            "<dt>Convergence</dt><dd>Sign agreement among affected layers "
            "(1.0 = all push the same direction; 0.0 = perfectly split).</dd>"
        )

    if include_percentile:
        parts.append(
            "<dt>Effect %ile</dt><dd>How unusual is this variant's effect? "
            "Ranked against ~10,000 random SNPs scored genome-wide. "
            "<code>95%</code> means the effect is larger than 95% of random "
            "variants. Range [0,1] for unsigned layers (chromatin, TF, histone, "
            "TSS, splicing); [−1,1] for signed layers (gene expression, MPRA).</dd>"
        )

    if include_activity:
        parts.append(
            "<dt>Activity %ile</dt><dd>How active is this region genome-wide? "
            "Reference signal ranked against ~26,000 ENCODE SCREEN cCREs and "
            "random regions. <code>0.95</code> = predicted signal exceeds 95% of "
            "genome-wide positions — a highly active regulatory element.</dd>"
        )

    # Per-layer formula bullets
    layers = list(layers_present) if layers_present is not None else []
    # Dedup, preserve order
    seen: set[str] = set()
    unique_layers = []
    for layer in layers:
        if layer not in seen:
            seen.add(layer)
            unique_layers.append(layer)

    layer_entries = []
    for layer in unique_layers:
        cfg = LAYER_CONFIGS.get(layer)
        if cfg is None:
            continue
        label = formula_label(cfg.formula)
        meaning = formula_meaning(cfg.formula, cfg.pseudocount)
        layer_entries.append(
            f"<li><b>{cfg.description}</b> "
            f"<span class='formula-chip'>{label}</span> — {meaning}</li>"
        )

    if layer_entries:
        parts.append(
            "<dt>Per-layer effect</dt>"
            "<dd>Unit depends on the layer:<ul>"
            + "".join(layer_entries)
            + "</ul></dd>"
        )

    parts.append("</dl>")
    parts.append("</section>")
    return "".join(parts)
