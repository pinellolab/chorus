"""Two panels a notebook invites the reader to compare must share one y-axis.

CoolBox gives a track whatever axis its source hands it, and the two ways the
notebooks build a panel hand it different ones:

* ``BedGraph("...bedgraph")`` reads the raw predicted values off disk and lets
  CoolBox autoscale to the data (plus 5% headroom);
* ``track.get_coolbox_representation()`` CDF-rescales through
  :func:`chorus.analysis._igv_report.rescale_for_display` and pins the axis to
  ``MinValue(0)`` / ``MaxValue(3.0)``, where 1.0 is that track's genome-wide p99.

``advanced_multi_oracle_analysis.ipynb`` built its two "spot the difference"
figures out of one of each. Re-rendering the committed cells with the committed
bedgraphs, the panels a reader is told to compare were:

    figure 6   Enformer Original      y = -1.1229 .. 23.5815   (raw, autoscaled)
               Enformer Replacement   y =  0.0    ..  3.0      (CDF-rescaled)
               ChromBPNet Original    y = -7.4203 .. 155.8253  (raw, autoscaled)
               ChromBPNet Replacement y =  0.0    ..  3.0      (CDF-rescaled)
    figure 9   same four axes, K562 (raw) against HepG2 (CDF-rescaled)

The Enformer pair in figure 6 is very nearly the same data -- raw max 22.458
(original) against 22.079 (replacement) -- so essentially all of the difference
the markdown pointed at was the 8x change of units, and the notebook then drew a
conclusion about model sensitivity from it.

These checks are hermetic: they read the committed .ipynb file and parse the cell
sources. Nothing is executed, no oracle, no GPU, no reference FASTA.

Scope is this one notebook because that is the file the fix owns. No other
notebook under ``examples/notebooks/`` mixes the two regimes today (measured
2026-08-09 with the same AST scan), so widening ``ADVANCED`` to a glob is safe
whenever the sibling notebooks are covered by their own checks -- the min-max
labelling in ``single_oracle_quickstart.ipynb`` and the Sei panel scales in
``comprehensive_oracle_showcase.ipynb`` have their own test files.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "examples" / "notebooks"
ADVANCED = NOTEBOOK_DIR / "advanced_multi_oracle_analysis.ipynb"

# The scale annotation every panel of a cross-condition comparison must carry, so a
# reader can see the two panels are on one ruler and what 1.0 on it means.
_AXIS_MARKER = "p99"


def _code_cells(path: Path) -> list[tuple[int, str]]:
    nb = json.loads(path.read_text())
    return [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code"]


def _parse(source: str) -> ast.Module | None:
    """Parse a cell, or None if it is not plain Python (IPython magics)."""
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _panels(tree: ast.Module) -> tuple[list[ast.Call], list[ast.Call]]:
    """Split the cell's track panels into (CDF-rescaled, autoscaled) calls."""
    cdf, autoscaled = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "get_coolbox_representation":
            opted_out = any(
                kw.arg == "normalize" and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
                for kw in node.keywords
            )
            (autoscaled if opted_out else cdf).append(node)
        elif isinstance(func, ast.Name) and func.id == "BedGraph":
            autoscaled.append(node)
    return cdf, autoscaled


def _string_consts(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` bindings, for resolving f-string titles."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value.value
    return out


def _static_str(node: ast.AST, consts: dict[str, str]) -> str | None:
    """Best-effort static value of a title argument (literal or f-string)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for piece in node.values:
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                parts.append(piece.value)
            elif isinstance(piece, ast.FormattedValue) and isinstance(piece.value, ast.Name):
                parts.append(consts.get(piece.value.id, ""))
            else:
                return None
        return "".join(parts)
    return None


def test_no_cell_mixes_autoscaled_and_cdf_rescaled_panels():
    """One CoolBox frame may not stack a raw autoscaled panel next to a 0-3 one."""
    notebook = ADVANCED
    offenders = []
    for idx, source in _code_cells(notebook):
        tree = _parse(source)
        if tree is None:
            continue
        cdf, autoscaled = _panels(tree)
        if cdf and autoscaled:
            offenders.append(
                f"cell {idx}: {len(cdf)} CDF-rescaled panel(s) (0-3) beside "
                f"{len(autoscaled)} autoscaled panel(s)"
            )
    assert not offenders, (
        f"{notebook.name} draws panels on two different y-axes in one frame, so the "
        "differences it shows are partly a change of units:\n  " + "\n  ".join(offenders)
    )


# The two figures the reader is explicitly told to compare across a condition.
# Anchors are substrings unique to those cells: the other K562-vs-HepG2 figures in
# the notebook (the quantile-normalization section) all read from
# ``joined_prediction`` and carry no such anchor.
COMPARISON_FIGURES = {
    "figure 6 - original vs replacement": 'replacement_results["raw_predictions"]',
    "figure 9 - K562 vs HepG2": "enformer_HepG2[",
}


@pytest.mark.parametrize("label,anchor", sorted(COMPARISON_FIGURES.items()))
def test_comparison_figures_state_the_shared_axis(label, anchor):
    """Every panel of those figures is CDF-rescaled *and* says so in its title."""
    matches = []
    for idx, source in _code_cells(ADVANCED):
        if anchor not in source:
            continue
        tree = _parse(source)
        if tree is None:
            continue
        cdf, autoscaled = _panels(tree)
        if len(cdf) + len(autoscaled) >= 2:
            matches.append((idx, tree, cdf, autoscaled))

    assert len(matches) == 1, f"{label}: expected one figure cell, found {[m[0] for m in matches]}"
    idx, tree, cdf, autoscaled = matches[0]
    assert not autoscaled, f"{label} (cell {idx}): {len(autoscaled)} panel(s) still autoscaled"
    assert len(cdf) == 4, f"{label} (cell {idx}): expected 4 panels, found {len(cdf)}"

    consts = _string_consts(tree)
    for call in cdf:
        title = next((kw.value for kw in call.keywords if kw.arg == "title"), None)
        assert title is not None, f"{label} (cell {idx}): a panel has no explicit title"
        text = _static_str(title, consts)
        assert text and _AXIS_MARKER in text, (
            f"{label} (cell {idx}): panel title {text!r} does not state the display "
            f"scale, so a reader cannot tell the panels share one axis"
        )


def test_replacement_summary_reports_raw_signal():
    """A "Max signal" readout may not come from the min-max normalized scores.

    ``predict_region_replacement`` returns ``normalized_scores =
    predictions.normalize()``, i.e. per-track min-max, so its max is 1.0000 by
    construction and its mean is a unitless fraction -- printed under a bare
    "Max signal" label that reads like a predicted value.
    """
    offenders = []
    for idx, source in _code_cells(ADVANCED):
        tree = _parse(source)
        if tree is None or "Max signal" not in source:
            continue
        # Look at the code, not the prose: a cell may well *mention*
        # normalized_scores in a comment explaining why it does not read from it.
        keys = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        if "normalized_scores" in keys:
            offenders.append(idx)
    assert not offenders, (
        f"cells {offenders} print 'Max signal' from normalized_scores (min-max, so "
        "always 1.0000); read it from raw_predictions or label it as normalized"
    )
