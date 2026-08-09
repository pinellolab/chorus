"""Figure 4 of ``comprehensive_oracle_showcase.ipynb`` must not mix normalized
and unnormalized panels.

The shipped figure ranked its own four Sei tracks upside down. Cell 19 prints
maxima 2.7644 / 2.1710 (sequence classes) and 0.2457 / 0.3531 (K562 H3K4me3),
and cell 20 drew all four with the renderer's default ``normalize=True``:

* ``sei_pertrack.npz`` holds CDF rows for the 40 sequence classes **only**, and
  those rows are flagged signed, so ``CA#E1`` was divided by ``p99(|cdf|)``
  = 22.097 and pinned to the fixed ±3.0 display axis: 2.7644 -> 0.1251, which
  is 2.1% of the panel's height. ``CA#E2`` -> 0.0924 = 1.5%.
* the two ``TA#`` histone targets have no CDF row (``_match_track_id`` returns
  ``None``; the run logged four "not found" WARNINGs), so
  ``rescale_for_display`` returned ``rescaled=False`` and the renderer fell
  back to per-panel autoscale, 0-0.2457 and 0-0.3531 — both **100%** of their
  panels.

So the two larger tracks rendered as flat lines and the two smaller ones as
full-height peaks. The measurements above were reproduced outside the notebook
against the real ``sei_pertrack.npz``; after the fix the same four maxima give
90.2% / 70.8% (class group, shared axis -0.30 to 2.76) and 69.6% / 100.0%
(target group, shared axis 0.00 to 0.35).

These checks are hermetic: no oracle, no model weights, no reference FASTA.
They build ``OraclePredictionTrack`` objects carrying the four maxima, run the
notebook cell's own drawing code on them, and read back what CoolBox was asked
to draw. Nothing is plotted — the y-limits and the plotted values are exactly
the track properties and the bedgraph the renderer writes.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "examples" / "notebooks" / "comprehensive_oracle_showcase.ipynb"

# The cell is located by its header comment rather than by index so inserting a
# cell above it does not silently retarget these tests at the wrong figure.
CELL_TAG = "# Visualize Sei predictions"

GATA1_CHROM, GATA1_START, GATA1_END = "chrX", 48777634, 48790694
PLOT_RANGE = "chrX:48726820-48841508"
RESOLUTION, N_BINS = 128, 103

# (assay_id, assay_type, group, raw max, raw min). The maxima are what cell 19
# printed on the 2026-08-08 run; the minima are plausible (Sei class scores go
# slightly negative, per-target probabilities do not) and only set the
# autoscaled lower bound.
TRACK_SPECS = [
    ("CA#E1@Stem cell@Enhancer@5", "Stem cell", "class", 2.7644, -0.30),
    ("CA#E2@Multi-tissue@Enhancer@6", "Multi-tissue", "class", 2.1710, -0.21),
    ("TA#K562_Erythroblast_Bone_Marrow@H3K4me3@ID:2034", "H3K4me3", "target", 0.2457, 0.0),
    ("TA#K562_Erythroblast_Bone_Marrow@H3K4me3@ID:2035", "H3K4me3", "target", 0.3531, 0.0),
]


def _cell_source() -> str:
    nb = json.loads(NOTEBOOK.read_text())
    hits = [
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and "".join(c["source"]).startswith(CELL_TAG)
    ]
    assert len(hits) == 1, f"expected exactly one {CELL_TAG!r} cell, found {len(hits)}"
    return hits[0]


def _make_track(assay_id, assay_type, vmax, vmin):
    from chorus.core.interval import GenomeRef, Interval
    from chorus.core.result import OraclePredictionTrack

    # The FASTA is never opened: nothing here touches ``.sequence``, and the
    # renderer only reads chrom/start/end off the intervals.
    fasta = "/nonexistent/hg38.fa"
    pred = Interval.make(GenomeRef(GATA1_CHROM, GATA1_START,
                                   GATA1_START + RESOLUTION * N_BINS, fasta,
                                   hash_sequence=False))
    query = Interval.make(GenomeRef(GATA1_CHROM, GATA1_START, GATA1_END, fasta,
                                    hash_sequence=False))
    x = np.arange(N_BINS)
    shape = np.exp(-((x - 34) ** 2) / 6.0) + 0.8 * np.exp(-((x - 62) ** 2) / 9.0)
    values = vmin + (vmax - vmin) * (shape / shape.max())
    values[int(np.argmax(values))] = vmax          # pin the printed maximum exactly
    return OraclePredictionTrack(
        source_model="sei", assay_id=assay_id, assay_type=assay_type,
        cell_type="K562", query_interval=query, prediction_interval=pred,
        input_interval=pred, resolution=RESOLUTION, values=values,
    )


def _panels(source: str):
    """Run the cell's drawing code and return one record per signal panel.

    Each record is ``(spec, raw_values, plotted_values, ymin, ymax, title)``,
    read straight off the CoolBox track the cell built: ``plotted_values`` is
    the bedgraph the renderer wrote, and ``ymin``/``ymax`` are the
    ``MinValue``/``MaxValue`` properties that become the panel's y-limits.
    """
    pytest.importorskip("coolbox")
    if shutil.which("bgzip") is None:          # coolbox indexes the bedgraph on construction
        pytest.skip("bgzip not on PATH")
    import coolbox.api as cb

    tracks = [_make_track(tid, atype, vmax, vmin)
              for tid, atype, _, vmax, vmin in TRACK_SPECS]
    namespace = {k: getattr(cb, k) for k in dir(cb) if not k.startswith("_")}
    namespace.update(
        np=np,
        sei_results={t.assay_id: t for t in tracks},
        # cell 18 selects 13 classes and 92 targets; cell 19 takes [:2] of each.
        sei_classes=[t[0] for t in TRACK_SPECS[:2]] + [f"CA#PAD{i}" for i in range(11)],
        sei_targets=[t[0] for t in TRACK_SPECS[2:]] + [f"TA#PAD{i}" for i in range(90)],
        PLOT_RANGE=PLOT_RANGE,
    )
    # Assembling the frame is what these tests inspect; rendering it needs a
    # figure and adds nothing, so the plot call is dropped.
    exec(compile(source.replace("frame.plot(PLOT_RANGE)", "pass"),
                 "<cell>", "exec"), namespace)

    signal = [t for t in namespace["frame"].tracks.values()
              if type(t).__name__ == "BedGraph"]
    assert len(signal) == len(TRACK_SPECS), f"{len(signal)} signal panels drawn"
    # Each track writes its bedgraph into its own storage dir, so the file path
    # identifies which prediction a panel came from without assuming the cell
    # kept them in prediction order.
    by_file = {str(t._storage / t.COOLBOX_FILE_NAME): (spec, t)
               for spec, t in zip(TRACK_SPECS, tracks)}
    out = []
    for cbt in signal:
        spec, track = by_file.pop(cbt.properties["file"])
        plotted = np.loadtxt(cbt.properties["file"], usecols=3)
        out.append((spec, track.values, plotted,
                    cbt.properties["min_value"], cbt.properties["max_value"],
                    cbt.properties.get("title", "")))
    assert not by_file, f"tracks never drawn: {[s[0] for s, _ in by_file.values()]}"
    return out


def _ink_fraction(peak, ymin, ymax):
    """Fraction of the panel's height covered by the peak.

    CoolBox's ``fill`` style fills between 0 and the value, so what the reader
    sees is the distance from the zero line, over the axis span.
    """
    base = 0.0 if ymin <= 0.0 <= ymax else ymin
    return (peak - base) / (ymax - ymin)


def test_no_panel_is_cdf_rescaled_while_another_is_not():
    """The defect in one assertion: some panels rescaled, others not.

    Whether the figure normalizes is a choice; doing it to half the panels is
    not, because the axes then mean different things with nothing on screen
    saying so. Before the fix this failed with the two ``CA#`` panels rescaled
    (values divided by p99(|cdf|) ~ 22, axis ±3.0) and the two ``TA#`` panels raw.
    """
    rescaled = {
        spec[0]: not np.allclose(plotted, raw, rtol=1e-9, atol=1e-12)
        for spec, raw, plotted, _, _, _ in _panels(_cell_source())
    }
    assert len(set(rescaled.values())) == 1, (
        f"figure mixes rescaled and raw panels: {rescaled}"
    )


def test_panels_that_share_a_unit_share_one_axis():
    """Sequence-class scores and per-profile signal are different quantities.

    Panels of the same kind must share y-limits (otherwise per-panel autoscale
    makes every peak full height, which is how the ``TA#`` pair came to look
    identical at 0.2457 and 0.3531); panels of different kinds must not be
    assumed comparable, so they are only required to be internally consistent.
    """
    by_group: dict[str, set] = {}
    for spec, _, _, ymin, ymax, _ in _panels(_cell_source()):
        by_group.setdefault(spec[2], set()).add((ymin, ymax))
    for group, limits in by_group.items():
        assert len(limits) == 1, f"{group} panels drawn on different axes: {limits}"


def test_panel_heights_preserve_the_raw_ranking():
    """A taller bar must mean a larger prediction, within a unit group.

    Before the fix the ``TA#`` pair both filled 100% of their panels (0.2457 on
    a 0-0.2457 axis, 0.3531 on a 0-0.3531 axis), so the smaller track looked
    equal to the larger one; and the smallest value in the whole figure (0.2457)
    drew a full-height peak while the largest (2.7644) drew 2.1%.
    """
    panels = _panels(_cell_source())
    heights = {}
    for spec, _, plotted, ymin, ymax, _ in panels:
        heights[spec[0]] = (spec[2], spec[3], _ink_fraction(plotted.max(), ymin, ymax))

    for group in {g for g, _, _ in heights.values()}:
        members = sorted(((raw, h, tid) for tid, (g, raw, h) in heights.items()
                          if g == group), reverse=True)
        for (raw_hi, h_hi, tid_hi), (raw_lo, h_lo, tid_lo) in zip(members, members[1:]):
            assert h_hi > h_lo, (
                f"{tid_hi} (max {raw_hi}) draws {h_hi:.1%} of its panel but "
                f"{tid_lo} (max {raw_lo}) draws {h_lo:.1%}"
            )

    # A shared axis has to be shared with every member of the group, not just
    # with the track that happened to set it: anything above ymax is clipped
    # flat at the top, which would hide a peak just as effectively.
    for spec, _, plotted, ymin, ymax, _ in panels:
        assert ymin <= plotted.min() and plotted.max() <= ymax, (
            f"{spec[0]} data [{plotted.min():.4f}, {plotted.max():.4f}] is "
            f"clipped by its axis [{ymin:.4f}, {ymax:.4f}]"
        )


def test_each_panel_states_the_scale_it_is_drawn_on():
    """A log line is not enough — the scale has to be on the panel.

    The mixed state was only visible as four ``normalization`` WARNINGs in the
    cell output, which no reader of the rendered figure connects to a flat
    green track. Each title now carries its axis and its normalization state.
    """
    for spec, _, plotted, ymin, ymax, title in _panels(_cell_source()):
        assert spec[0] in title, f"panel title lost its track id: {title!r}"
        assert "CDF-rescaled" in title, (
            f"{spec[0]} panel does not say whether it is CDF-rescaled: {title!r}"
        )
        for bound in (ymin, ymax):
            assert f"{bound:.2f}" in title, (
                f"{spec[0]} panel title omits its axis bound {bound:.2f}: {title!r}"
            )


def test_the_cell_opts_out_of_normalization_explicitly():
    """Every panel in the cell passes the same ``normalize=``.

    The default is ``True``, so an added panel that just omits the argument
    would silently reintroduce the mixed figure for any track Sei's CDF file
    happens to cover. Read statically so it holds for panels these tests do
    not build.
    """
    calls = [
        node for node in ast.walk(ast.parse(_cell_source()))
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "get_coolbox_representation"
    ]
    assert calls, "no get_coolbox_representation call found in the Sei cell"
    seen = set()
    for call in calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        assert "normalize" in kwargs, (
            "get_coolbox_representation call relies on the default normalize=True"
        )
        assert isinstance(kwargs["normalize"], ast.Constant)
        seen.add(kwargs["normalize"].value)
    assert seen == {False}, f"cell mixes normalize values: {seen}"
