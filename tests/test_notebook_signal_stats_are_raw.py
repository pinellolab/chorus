"""``single_oracle_quickstart.ipynb`` must not label min-max values as signal.

The quickstart told the reader that a 200 bp insert destroyed GATA1 regulatory
signal across 100 kb. It did not. Three cells read
``results['normalized_scores']`` — which is per-track **min-max** (
``OraclePredictionTrack.normalize`` -> ``chorus.core.result.minmax``) — and
presented it as raw signal:

* cells 28 and 36 printed "Mean signal / Max signal" from it, so every track
  reported ``Max signal: 1.0000``: true by construction, and read against the
  wild-type maxima that cell 17 prints from raw values (22.4586 and 120.7775)
  it looks like a 95%+ collapse. Reconstructed from the raw bedgraph the same
  run wrote, cell 28's real numbers are mean 0.5196 / max 22.0786
  (ENCFF413AHU) and mean 0.7238 / max 158.6910 (CNhs11250) — i.e. the
  replacement barely moved the region;
* cell 37 fed those [0, 1] values into ``get_coolbox_representation()``, which
  CDF-rescales for display and therefore reads a min-max 1.0 as a *raw* 1.0.
  Measured against enformer's shipped per-track CDFs that is 0.7506 display
  units for ENCFF413AHU (p90 floor 0.3801, p99 peak 1.2059) and 0.9851 for
  CNhs11250, on a 0-3 axis where 1.0 is the genome-wide p99 — so 29 bins that
  should clip at the 3.0 cap render at a quarter height and the panel goes
  flat. Re-rendering the *same* array raw versus min-max reproduces the
  committed figure 3 and the committed figure 4 respectively, which is what
  established that figure 4 showed a normalization artefact and not biology.

Both checks below are hermetic: the notebook scan reads committed JSON, and the
display test builds its own one-row CDF. No GPU, no weights, no backgrounds.

Scope is this one notebook because that is the file the fix owns.
``advanced_multi_oracle_analysis.ipynb`` (cell 49, printed stats) and
``comprehensive_oracle_showcase.ipynb`` (cell 33, CoolBox panel) still read
``normalized_scores`` the same way; widen ``NOTEBOOKS`` once those are fixed.
"""

from __future__ import annotations

import io
import json
import re
import tokenize
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [REPO_ROOT / "examples" / "notebooks" / "single_oracle_quickstart.ipynb"]

# Labels that promise the reader a signal level, not a rescaled one.
_SIGNAL_LABEL = re.compile(
    r"Mean signal|Max signal|Overall mean|Signal around|Peak near", re.I
)

# The real thresholds for ENCFF413AHU in ``enformer_pertrack.npz``, so the
# display test asserts the measured collapse rather than an invented one.
_ENCFF413AHU_P90 = 0.3801
_ENCFF413AHU_P99 = 1.2059


def _code(source: str) -> str:
    """Cell source with comments stripped.

    The fixed cells *mention* ``normalized_scores`` in a comment explaining why
    they do not use it, so a plain substring scan would flag the fix itself.
    """
    try:
        toks = tokenize.generate_tokens(io.StringIO(source).readline)
        return tokenize.untokenize(
            (t.type, t.string) for t in toks if t.type != tokenize.COMMENT
        )
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Cells with IPython magics do not tokenize; line-based fallback.
        return "\n".join(line.split("#", 1)[0] for line in source.splitlines())


def _code_cells(path: Path):
    nb = json.loads(path.read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            yield i, _code("".join(cell["source"]))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_printed_signal_statistics_come_from_raw_predictions(path):
    """A cell that prints "Max signal" from min-max always prints 1.0000."""
    offenders = [
        i for i, code in _code_cells(path)
        if _SIGNAL_LABEL.search(code) and "normalized_scores" in code
    ]
    assert not offenders, (
        f"{path.name} cells {offenders} print a signal statistic from "
        "normalized_scores (per-track min-max: the max is 1.0 by construction). "
        "Read ['raw_predictions'] instead, so the numbers are comparable to the "
        "wild-type cell."
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_coolbox_panels_are_not_fed_already_normalized_values(path):
    """``get_coolbox_representation`` CDF-rescales — it wants raw input."""
    offenders = [
        i for i, code in _code_cells(path)
        if "get_coolbox_representation" in code and "normalized_scores" in code
    ]
    assert not offenders, (
        f"{path.name} cells {offenders} plot normalized_scores. "
        "get_coolbox_representation() rescales against the genome-wide CDF, so "
        "min-max input is normalized twice and real peaks flatten into noise."
    )


def test_minmax_pins_the_max_at_one_whatever_the_signal():
    """Why "Max signal: 1.0000" carried no information about the edit."""
    from chorus.core.result import minmax

    for peak in (0.01, 1.0, 22.4586, 158.6910):
        values = np.array([0.0, 0.05, 0.4, peak])
        assert minmax(values).max() == pytest.approx(1.0)


def _one_track_normalizer(tmp_path, p90: float, p99: float):
    """A per-track CDF whose p90 and p99 are exactly ``p90`` and ``p99``.

    Piecewise-linear and strictly increasing so the row is a legal CDF (
    ``build_and_save`` rejects rows padded with a repeated tail) and so the
    indices ``perbin_floor_rescale_batch`` reads — ``cdf[int(0.90 * n)]`` and
    ``cdf[int(0.99 * n)]`` — land exactly on the requested thresholds.
    """
    from chorus.analysis.normalization import PerTrackNormalizer

    n = 10_000
    i90, i99 = int(0.90 * n), int(0.99 * n)
    cdf = np.concatenate([
        np.linspace(0.0, p90, i90, endpoint=False),
        np.linspace(p90, p99, i99 - i90, endpoint=False),
        np.linspace(p99, p99 * 30, n - i99),
    ])
    PerTrackNormalizer.build_and_save(
        oracle_name="minmaxtest",
        track_ids=["ENCFF413AHU"],
        perbin_cdfs=cdf[None, :],
        perbin_counts=[n],
        cache_dir=str(tmp_path),
        n_points=n,
    )
    return PerTrackNormalizer(cache_dir=str(tmp_path))


def test_display_rescale_flattens_minmax_input(tmp_path):
    """The figure-4 mechanism, measured.

    A raw DNase peak of 22 is 26x the track's genome-wide p99 and clips at the
    3.0 display cap. Min-max first maps that peak to 1.0, which the same
    rescaler reads as a raw 1.0 — below the p99 threshold — so the tallest bin
    in the panel ends up at 0.75 of 3.0, a quarter of the axis.
    """
    from chorus.analysis._igv_report import rescale_for_display
    from chorus.core.result import minmax

    normalizer = _one_track_normalizer(tmp_path, _ENCFF413AHU_P90, _ENCFF413AHU_P99)
    raw = np.array([0.02, 0.05, 0.3, 1.4, 22.0786])

    from_raw, cfg = rescale_for_display(
        raw, "chromatin_accessibility", normalizer=normalizer,
        oracle_name="minmaxtest", assay_id="ENCFF413AHU",
    )
    from_minmax, _ = rescale_for_display(
        minmax(raw), "chromatin_accessibility", normalizer=normalizer,
        oracle_name="minmaxtest", assay_id="ENCFF413AHU",
    )

    assert cfg["rescaled"] and (cfg["ymin"], cfg["ymax"]) == (0.0, 3.0)
    assert from_raw.max() == pytest.approx(3.0)          # clipped at the cap
    expected = (1.0 - _ENCFF413AHU_P90) / (_ENCFF413AHU_P99 - _ENCFF413AHU_P90)
    assert from_minmax.max() == pytest.approx(expected, abs=1e-3)  # 0.7506
    assert from_raw.max() / from_minmax.max() > 3.9
