"""A committed notebook figure may not be an empty pair of axes.

Nothing in the suite has ever looked at a notebook's *pixels*.
``grep -rl "image/png" tests/`` returned zero files, and the three tests that do
police notebook figures --- ``test_notebook_panels_share_one_axis.py``,
``test_notebook_signal_stats_are_raw.py`` and
``test_notebook_sei_figure_scales.py`` --- reason about cell *source* and about
CoolBox track properties. All 13 of their assertions pass against a notebook
whose figure renders as a flat line on an empty axis, because axis limits,
titles and normalization flags can all be perfectly correct while the trace
drawn between them is invisible. Two shipped figures were exactly that:

    single_oracle_quickstart  cell 47  two 0-3 panels, tallest trace 0.32% of the axis
    comprehensive_oracle_showcase cell 16  one 0-3 panel, tallest trace 7.80%

This file closes that hole: it decodes every ``image/png`` output committed under
``examples/notebooks/`` and asks whether a reader would see anything.

The ``single_oracle_quickstart`` figure was re-rendered while this file was being
written and now passes on merit (its two panels peak at 56.8% and 98.1% of the
0-3 axis), so only ``comprehensive_oracle_showcase`` cell 16 is still blank. The
0.32% render is kept in the table above because it is half the measurement the
threshold below is derived from, and because a revert would put it back.

Why an ink threshold alone is not a test
----------------------------------------
The obvious check --- count non-white pixels, require a minimum fraction --- is
not merely weak here, it is inverted. Measured ink fractions over the whole
canvas:

    blank      single_oracle_quickstart cell 47      0.01066   (colored only 0.00236)
    blank-ish  comprehensive_oracle_showcase cell 16 0.00490   (colored only 0.00006)
    sparse but entirely fine, advanced_multi_oracle_analysis
                                                     0.00191, 0.00381, 0.00189, 0.00384
    healthy    cherimoya_quickstart                  0.0812, 0.0828
               klf1_validated_enhancer_profiles      0.054 .. 0.208
               epinformerseq_testing                 0.0907 .. 0.4363 (plus 0.0407,
                                                     a two-point dot plot)

The sparse-but-fine figures are narrow spiky ChIP/DNase footprints: a few
hundred base pairs of real signal inside a 100 kb window. They carry **less**
ink (0.0019-0.0038) than the two blank ones (0.0049-0.0107), so *any* ink
threshold that fails the blanks also fails four correct figures, and any
threshold that spares those four waves the blanks through. Do not "simplify"
this file back into an ink threshold; the numbers above are why it cannot work.
``MIN_INK_FRACTION`` below is kept only as a floor on a canvas with nothing on
it at all --- no axes, no ticks --- and is deliberately set below the sparsest
legitimate figure, so it can never be the check that catches a blank panel.

The signal that does separate them
----------------------------------
Trace peak height as a fraction of its own axis. The same figures:

    blank        0.32%, 1.29%, 7.80%
    sparse/fine  14.05% .. 24.59%

A footprint that occupies 300 bp of a 100 kb window still *rises*; a diluted or
mis-scaled trace does not. ``MIN_PEAK_FRACTION`` is the geometric midpoint of
that measured gap, ``sqrt(0.0780 * 0.1405) = 0.1047`` --- 1.34x above the
tallest blank and 1.34x below the shortest good one. It is not a round number on
purpose: rounding it to 10% or 15% would shrink one of those margins for no
reason anyone could reconstruct later.

Re-measured over all 41 committed images with the implementation in this file
(which finds the axis from the rendered y-spine rather than from the plotting
call, so it does not reproduce the numbers above exactly): the tallest blank
figure is 6.99% (comprehensive_oracle_showcase cell 16) and the shortest
legitimate one is 24.52% (advanced_multi_oracle_analysis cell 122, the sparse
REST ChIP pair), so 10.47% sits 1.5x above the blank and 2.3x below the sparsest
good figure. Every other judgeable figure measures 49.8% or more.

How the trace is isolated, and what that costs
----------------------------------------------
CoolBox draws the signal in the oracle's colour and every piece of furniture ---
spines, ticks, tick labels, track titles, the dashed region and variant markers
--- in black or grey. So chromatic pixels (``max(R,G,B) - min(R,G,B)`` above
``CHROMA_TOLERANCE``) are the data, with no geometry heuristics needed, and in
particular the full-height dashed markers cannot be mistaken for a tall trace.
Two honest consequences:

* Anything else drawn in colour inside a panel (a legend swatch, coloured text)
  is counted as trace and can only make a peak look *taller*. Every failure
  mode of this masking is therefore permissive: it lets a bad figure through, it
  does not invent a failure.
* A figure whose trace is achromatic cannot be judged at all and is skipped, not
  failed. There is exactly one today, 40 of the 41 committed images being
  judgeable: ``epinformerseq_testing`` cell 25 fills with ``color='#999999'`` and
  has 0 chromatic pixels. ``test_most_committed_figures_are_judgeable`` keeps
  that escape hatch from quietly swallowing the suite.

This is a heuristic on rendered pixels, not a proof about the data. It answers
one question --- "would a reader see a trace here?" --- and the failure message
prints the notebook, the cell index, the panel row bands and every measured
number so a human can open the figure and disagree.

The still-blank figure is marked ``xfail(strict=True)`` **by the sha256 of its
exact PNG bytes**, not by cell index: it is being fixed in parallel with this
file, cell indices shift when a markdown note is inserted above them, and the
committed output stays stale until the re-execution pass described under
"Regeneration" in ``CLAUDE.md``. Keying on the bytes means the moment the figure
is re-rendered the xfail stops applying and the new render has to pass on merit
--- including if it is re-rendered still blank, which is the case an index-keyed
xfail would have hidden. ``strict=True`` means that loosening
``MIN_PEAK_FRACTION`` far enough to let today's blank through turns into an XPASS
failure rather than a silently weaker test. So this file is informative either
way: green today, and green after the fix only because the figure really changed.

Hermetic: reads the committed ``.ipynb`` files, decodes their committed images.
Nothing is executed, no oracle, no GPU, no reference FASTA.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO / "examples" / "notebooks"

# --- what counts as a pixel ------------------------------------------------
# Ink: any channel this far below 255 after compositing onto white. 12/255 is
# below matplotlib's lightest alpha-blended fill and above PNG quantisation, and
# reproduces the whole-canvas measurements quoted in the module docstring (the
# 0.32% single_oracle_quickstart render: 0.01058 here against 0.01066 measured).
WHITE_TOLERANCE = 12
# Trace: chromatic pixels. 25/255 keeps CoolBox's alpha-blended light-blue fill
# and drops every grey the renderer uses for furniture (a grey has spread 0).
CHROMA_TOLERANCE = 25

# --- how the axis is located ----------------------------------------------
# A y-spine is a near-vertical run of non-white pixels near the left edge. The
# brightness bound is loose (235) because matplotlib's default spine here is
# light grey: at <200 the four panels of advanced_multi_oracle_analysis cell 72
# are invisible and the figure would be skipped rather than checked.
SPINE_MAX_BRIGHTNESS = 235
# Measured spine columns run 2%-33% of the width -- 28 px of 1468 for a CoolBox
# frame, 340 px of 1034 for epinformerseq cell 32, whose y tick labels are full
# "oracle :: track" strings. 40% covers those; the tolerance rule below is what
# keeps a wide search from picking up something that is not a spine.
SPINE_SEARCH_WIDTH = 0.40
# A spine spans one panel. The shortest measured is the 8-panel
# klf1_validated_enhancer_profiles cell 21, at 110 px of 1325 (8.3% of the
# canvas); 5% admits that and still rejects dashed markers and text, whose runs
# are ~12 px.
SPINE_MIN_RUN = 0.05
# Dash gaps in the region/variant markers measure 5-7 px, so joining runs across
# gaps of 3 keeps a dashed line from impersonating a spine.
SPINE_MAX_GAP = 3
# A saturated trace can fill its panel top to bottom and so tie the spine on
# total run length. The spine is the leftmost column that ties, so candidates are
# taken leftmost-first among those within 10% of the best total. On the committed
# figures that moves five picks off a trace column and onto the real axis
# (comprehensive_oracle_showcase cell 13: x=182 -> x=33; advanced_multi_oracle_
# analysis cell 26: x=153 -> x=28), shifts the rest by a pixel of antialiasing,
# and changes no verdict.
SPINE_TOTAL_TOLERANCE = 0.90

# --- the two thresholds ---------------------------------------------------
# Geometric midpoint of the measured gap between blank (7.80%) and legitimately
# sparse (14.05%) trace peaks; see the module docstring.
MIN_PEAK_FRACTION = 0.1047
# Half the sparsest legitimate figure measured (0.00189). This is a floor on an
# empty canvas only -- it sits *below* the blank figures (0.0049-0.0107) by
# design, because ink does not separate them from the sparse-but-correct ones.
MIN_INK_FRACTION = 0.00094
# 40 of 41 committed images are judgeable today (98%). A floor of 0.8 leaves room
# for a new achromatic figure without letting the skip path grow quietly.
MIN_JUDGEABLE_FRACTION = 0.8

# sha256 of the exact committed PNG bytes of a figure already known to be blank.
# Cleared by the local-effect fix for that notebook plus the figure re-execution
# pass ("Regeneration" in CLAUDE.md): re-rendering changes the bytes, which
# removes the xfail and requires the new figure to pass on merit.
#
# The companion single_oracle_quickstart render
# (5c862aebcd2afd6f27bbc89f992af45d740912e16d6117a52044d3accc8d0730, tallest
# trace 0.32%) was listed here and has been dropped: that notebook's cell 47 was
# re-rendered on 2026-08-10 and its panels now peak at 56.8% and 98.1%, so it is
# checked like any other figure. Add an entry only for a figure that is blank
# *now* -- a stale hash matches nothing and reads like coverage that is not there.
KNOWN_BLANK_RENDERS: dict[str, str] = {
    # Empty, and that is the intended end state: both figures this file was written
    # for now pass on merit. Kept as the documented mechanism for the next one.
    #
    # The comprehensive_oracle_showcase ChromBPNet panel was the last holdout at
    # 6.72% of its axis. Restricting the plot to the span the prediction covers
    # raised its coloured ink 33x but did not lift the trace, because the cause was
    # not the rendering: the notebook profiles the erythroid GATA1 locus while that
    # panel loaded ChromBPNet's HepG2 (hepatocyte) model, so the model was correctly
    # predicting a closed promoter. Measured over GATA1_REGION: HepG2 raw max 1.126
    # -> display 0.236 (7.9% of the axis), K562 raw max 6.077 -> display 0.930
    # (31.0%). Switched to K562, which is what every other panel in that notebook
    # already used.
}


@dataclass(frozen=True)
class CommittedImage:
    """One ``image/png`` output as it sits in a committed notebook."""

    notebook: Path
    cell: int
    output: int
    png: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.png).hexdigest()

    @property
    def where(self) -> str:
        return f"{self.notebook.name} cell {self.cell} output {self.output}"


@dataclass(frozen=True)
class Panel:
    """One y-axis found in a figure, and the trace drawn against it."""

    top: int
    bottom: int
    colored_pixels: int
    peak_fraction: float | None      # None when no trace pixel was found

    def describe(self) -> str:
        peak = "n/a" if self.peak_fraction is None else f"{self.peak_fraction:.2%}"
        return (f"rows {self.top}-{self.bottom} peak {peak} "
                f"({self.colored_pixels} coloured px)")


@dataclass(frozen=True)
class Figure:
    """Everything measured off one rendered figure."""

    width: int
    height: int
    ink_fraction: float
    colored_fraction: float
    spine_column: int | None
    panels: tuple[Panel, ...]

    @property
    def peaks(self) -> list[float]:
        return [p.peak_fraction for p in self.panels if p.peak_fraction is not None]

    @property
    def tallest_peak(self) -> float | None:
        """Tallest trace in the figure, as a fraction of its own axis.

        The maximum over panels, not the minimum: on a shared axis a genuinely
        small track *should* draw a short trace (that is what
        ``test_notebook_sei_figure_scales.py`` measures panel by panel), so
        requiring every panel to rise would fail correct figures. A figure in
        which no panel rises at all is the blank case.
        """
        return max(self.peaks) if self.peaks else None

    @property
    def unjudgeable(self) -> str | None:
        """Why this figure cannot be judged, or None if it can."""
        if not self.panels:
            return "no y-axis spine found near the left edge"
        if not self.peaks:
            return ("no chromatic pixels inside any panel -- the trace is drawn "
                    "in grey or black, which this check cannot separate from the "
                    "axes and labels")
        return None

    def describe(self) -> str:
        head = (f"{self.width}x{self.height}px, ink {self.ink_fraction:.3%} of the "
                f"canvas, coloured {self.colored_fraction:.3%}")
        if self.spine_column is None:
            return head + "; no axis spine found"
        peak = self.tallest_peak
        peak_txt = "none measurable" if peak is None else f"{peak:.2%} of its axis"
        return (f"{head}; spine at x={self.spine_column}; tallest trace peak "
                f"{peak_txt}; panels: "
                + "; ".join(p.describe() for p in self.panels))


def _committed_images() -> list[CommittedImage]:
    """Every ``image/png`` output in every notebook, in file and cell order."""
    found: list[CommittedImage] = []
    for notebook in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        nb = json.loads(notebook.read_text())
        for cell_index, cell in enumerate(nb.get("cells", [])):
            for output_index, output in enumerate(cell.get("outputs") or []):
                payload = (output.get("data") or {}).get("image/png")
                if payload is None:
                    continue
                if isinstance(payload, list):        # nbformat may split base64
                    payload = "".join(payload)
                try:
                    png = base64.b64decode(payload, validate=False)
                except (binascii.Error, ValueError) as exc:   # pragma: no cover
                    pytest.fail(f"{notebook.name} cell {cell_index} output "
                                f"{output_index}: undecodable image/png ({exc})")
                found.append(CommittedImage(notebook, cell_index, output_index, png))
    return found


IMAGES = _committed_images()


def _params():
    """One test per committed image, with any known-blank render xfailed."""
    out = []
    for image in IMAGES:
        marks = []
        reason = KNOWN_BLANK_RENDERS.get(image.sha256)
        if reason is not None:
            marks.append(pytest.mark.xfail(
                strict=True,
                reason=f"known blank render, fix pending re-execution: {reason}",
            ))
        out.append(pytest.param(image, marks=marks,
                                id=f"{image.notebook.stem}-cell{image.cell}"
                                   f"-out{image.output}"))
    return out


def _as_rgb(png: bytes) -> np.ndarray:
    """Decode to an ``(H, W, 3)`` int16 array composited onto white.

    Notebook figures are saved RGBA with a transparent background, so anything
    that skips the composite measures alpha, not ink.

    Imported here rather than at module scope so a machine without Pillow skips
    these checks instead of failing collection. Written out rather than via
    ``pytest.importorskip`` on purpose: since pytest 9.1 that helper only skips on
    ``ModuleNotFoundError``, so a Pillow that is *installed but broken* (a missing
    libjpeg, say) would surface here as 41 red tests about the notebooks.
    """
    try:
        from PIL import Image
    except ImportError as exc:                    # pragma: no cover - env-specific
        pytest.skip(f"Pillow is needed to decode the committed figures: {exc}")
    im = Image.open(io.BytesIO(png))
    if im.mode != "RGB":
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
    return np.asarray(im.convert("RGB")).astype(np.int16)


def _runs(rows: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Group sorted row indices into ``(first, last)`` runs, joining small gaps."""
    if rows.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(rows) > max_gap)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [rows.size - 1]))
    return [(int(rows[s]), int(rows[e])) for s, e in zip(starts, ends)]


def _find_panels(furniture: np.ndarray) -> tuple[int | None, list[tuple[int, int]]]:
    """Locate the y-spine column and the row band of each panel it serves.

    The spine is the column near the left edge carrying the most panel-height
    vertical run, resolved leftmost-first among near-ties so that a trace which
    fills its panel cannot stand in for the axis.
    """
    height, width = furniture.shape
    min_run = SPINE_MIN_RUN * height
    candidates = []
    for x in range(max(1, int(width * SPINE_SEARCH_WIDTH))):
        runs = [r for r in _runs(np.flatnonzero(furniture[:, x]), SPINE_MAX_GAP)
                if (r[1] - r[0]) >= min_run]
        if runs:
            candidates.append((x, sum(b - a for a, b in runs), runs))
    if not candidates:
        return None, []
    tallest = max(total for _, total, _ in candidates)
    # Always yields: the candidate that set ``tallest`` satisfies the bound.
    x, _, runs = next(c for c in candidates
                      if c[1] >= SPINE_TOTAL_TOLERANCE * tallest)
    return x, runs


def measure(png: bytes) -> Figure:
    """Measure ink, panels and per-panel trace peak height of one figure."""
    rgb = _as_rgb(png)
    height, width, _ = rgb.shape
    ink = (255 - rgb).max(axis=2) > WHITE_TOLERANCE
    colored = (rgb.max(axis=2) - rgb.min(axis=2)) > CHROMA_TOLERANCE
    furniture = rgb.max(axis=2) < SPINE_MAX_BRIGHTNESS

    spine, bands = _find_panels(furniture)
    panels = []
    for top, bottom in bands:
        # Everything right of the spine: the track title and the y tick labels
        # sit outside the axes but are achromatic, so they contribute nothing.
        band = colored[top:bottom + 1, (spine or 0) + 2:]
        rows = np.flatnonzero(band.any(axis=1))
        span = bottom - top
        peak = None
        if rows.size and span > 0:
            # Tallest column of trace, measured up from the bottom of the axis.
            peak = float((span - rows.min()) / span)
        panels.append(Panel(top, bottom, int(band.sum()), peak))
    return Figure(width, height, float(ink.mean()), float(colored.mean()),
                  spine, tuple(panels))


_CACHE: dict[str, Figure] = {}


def measured(image: CommittedImage) -> Figure:
    """``measure`` with memoisation, so both tests share one decode per image."""
    if image.sha256 not in _CACHE:
        _CACHE[image.sha256] = measure(image.png)
    return _CACHE[image.sha256]


@pytest.mark.parametrize("image", _params())
def test_committed_figure_shows_a_trace(image: CommittedImage):
    """A figure must put ink on the canvas and rise off its own baseline."""
    figure = measured(image)

    assert figure.ink_fraction >= MIN_INK_FRACTION, (
        f"{image.where} is a blank canvas: ink {figure.ink_fraction:.5f} of the "
        f"image, below {MIN_INK_FRACTION} (half the sparsest legitimate figure "
        f"measured). Not even axes rendered. {figure.describe()}"
    )

    why = figure.unjudgeable
    if why is not None:
        pytest.skip(f"{image.where}: {why}. {figure.describe()}")

    peak = figure.tallest_peak
    assert peak >= MIN_PEAK_FRACTION, (
        f"{image.where} looks blank to a reader: its tallest trace reaches "
        f"{peak:.2%} of its axis, under the {MIN_PEAK_FRACTION:.2%} floor "
        f"(geometric midpoint of the measured 7.80% blank / 14.05% "
        f"legitimately-sparse gap). Note the ink fraction does NOT catch this -- "
        f"blank figures measure 0.0049-0.0107 and correct sparse ones "
        f"0.0019-0.0038. Open the figure and look: {figure.describe()}"
    )


def test_most_committed_figures_are_judgeable():
    """The skip paths above must stay exceptions, not become the rule.

    ``test_committed_figure_shows_a_trace`` skips a figure whose axis it cannot
    find or whose trace is not drawn in colour. Both are real limits of reading
    pixels, and both are silent: without this test a change of notebook style ---
    a greyscale palette, a renderer that stops drawing spines --- would turn the
    check above into 41 green skips.
    """
    assert IMAGES, f"no image/png outputs found under {NOTEBOOK_DIR}; " \
                   "the notebooks ship executed outputs, so this means the " \
                   "collector or the notebooks are broken"

    unjudgeable = {img.where: (measured(img).unjudgeable, measured(img).describe())
                   for img in IMAGES if measured(img).unjudgeable is not None}
    judgeable = len(IMAGES) - len(unjudgeable)
    detail = "\n  ".join(f"{where}: {why}" for where, (why, _) in unjudgeable.items())
    assert judgeable >= MIN_JUDGEABLE_FRACTION * len(IMAGES), (
        f"only {judgeable}/{len(IMAGES)} committed figures can be checked for "
        f"blankness (floor {MIN_JUDGEABLE_FRACTION:.0%}). Unjudgeable:\n  {detail}"
    )
