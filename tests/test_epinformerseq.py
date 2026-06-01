"""Tests for the EPInformer-seq per-cell oracle.

Per-cell architecture: one ``PerCellProfileNetWide`` checkpoint per cell (2114-bp
input -> central 1024-bp crop, no FiLM, no cell embedding) + frozen per-cell
``BiasNet``. 11 Roadmap cells, trained on 5' DNase cut-sites (DNase-only).

Tests are split into:

- **Static checks** (no torch / no weights): registry, globals, layer config,
  CDF schema, weights probe, import sanity.
- **Functional checks** (require torch + chorus-epinformerseq env): load
  per-cell model, run a forward pass on a random 2114-bp sequence. Skipped
  when prerequisites missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

# Make chorus importable regardless of cwd.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chorus.oracles import EPInformerSeqOracle, get_oracle
from chorus.oracles.epinformerseq_source.globals import (
    EPINFORMERSEQ_AVAILABLE_ASSAYS,
    EPINFORMERSEQ_AVAILABLE_CELLTYPES,
    EPINFORMERSEQ_DEFAULT_ASSAY,
    EPINFORMERSEQ_DEFAULT_STEP,
    EPINFORMERSEQ_WINDOW,
    EPINFORMERSEQ_WIDE_WINDOW,
)


EXPECTED_CELLS = {
    "K562", "GM12878", "HepG2", "A549",
    "HeLa", "HMEC", "HSMM", "HUVEC", "NHEK", "NHLF", "H1",
}


def _weights_available():
    from chorus.core.globals import CHORUS_DOWNLOADS_DIR
    return (CHORUS_DOWNLOADS_DIR / "epinformerseq" / "per_cell_widewin" / "K562" / "main.pt").exists()


def _torch_available():
    """True iff torch can be imported (chorus-epinformerseq env has it; base does not)."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------

class TestEPInformerSeqRegistry:
    """Factory + import surface."""

    def test_get_oracle_lookup(self):
        assert get_oracle("epinformerseq") is EPInformerSeqOracle
        # Case-insensitive
        assert get_oracle("EPINFORMERSEQ") is EPInformerSeqOracle

    def test_no_stale_cellcondprofilenet_export(self):
        """The joint v2 model was retired; nothing should re-export it.

        ``chorus/oracles/epinformerseq_source/model.py`` does ``import torch``
        at module scope, so this runtime check needs torch. We import-skip
        when torch is missing (e.g. base CI env) — the companion
        ``test_no_stale_cellcondprofilenet_source`` does a static source grep
        with no torch dependency, so the regression is still caught there.
        In the chorus-epinformerseq env both tests run and fail loudly on
        any reappearance of the retired symbols.
        """
        pytest.importorskip("torch")
        from chorus.oracles.epinformerseq_source import model as model_mod
        assert not hasattr(model_mod, "CellCondProfileNet"), (
            "CellCondProfileNet should be retired in favor of PerCellProfileNet"
        )
        assert not hasattr(model_mod, "FiLM1d"), \
            "FiLM1d belongs to the retired joint model"
        # Single architecture: PerCellProfileNetWide (the 1024-bp PerCellProfileNet
        # was retired when EPInformer-seq consolidated to the 2114-bp cut-site model).
        assert hasattr(model_mod, "PerCellProfileNetWide")
        assert not hasattr(model_mod, "PerCellProfileNet"), \
            "the 1024-bp PerCellProfileNet was retired in favor of PerCellProfileNetWide"
        assert hasattr(model_mod, "BiasNet")

    def test_no_stale_cellcondprofilenet_source(self):
        """Static check (no torch needed): grep model.py for the retired class name."""
        src = (Path(__file__).resolve().parent.parent
               / "chorus" / "oracles" / "epinformerseq_source" / "model.py").read_text()
        # The class definition (not docstring/comment) must be absent.
        assert "class CellCondProfileNet" not in src, \
            "CellCondProfileNet class still defined in model.py"
        assert "class FiLM1d" not in src, "FiLM1d class still defined in model.py"
        assert "class PerCellProfileNetWide" in src, \
            "PerCellProfileNetWide class missing from model.py"
        assert "class PerCellProfileNet(" not in src, \
            "the 1024-bp PerCellProfileNet class should be retired from model.py"

    def test_globals_no_cell2idx(self):
        """Per-cell architecture removes the cell→index map."""
        from chorus.oracles.epinformerseq_source import globals as g
        assert not hasattr(g, "EPINFORMERSEQ_CELL2IDX")

    def test_discovery_import_path(self):
        """chorus/analysis/discovery imports from .globals (not .epinformerseq_globals)."""
        from chorus.analysis import discovery  # noqa
        # If this import succeeded, the deferred import in list_discovery_specs
        # would also succeed when triggered. Smoke test it explicitly:
        from chorus.oracles.epinformerseq_source.globals import (
            EPINFORMERSEQ_AVAILABLE_CELLTYPES as cells,
        )
        assert set(cells) == EXPECTED_CELLS

    def test_mcp_import_path_source(self):
        """chorus/mcp/server.py imports from the correct globals module
        (static grep — skips importing the module itself, which needs fastmcp)."""
        src = (Path(__file__).resolve().parent.parent
               / "chorus" / "mcp" / "server.py").read_text()
        assert "epinformerseq_source.globals" in src, \
            "chorus/mcp/server.py must import from epinformerseq_source.globals"
        assert "epinformerseq_globals" not in src, \
            "chorus/mcp/server.py still references the renamed epinformerseq_globals module"


class TestEPInformerSeqGlobals:
    """Constants advertised by the oracle."""

    def test_eleven_cells(self):
        assert set(EPINFORMERSEQ_AVAILABLE_CELLTYPES) == EXPECTED_CELLS
        assert len(EPINFORMERSEQ_AVAILABLE_CELLTYPES) == 11

    def test_assays(self):
        # 2-channel model: DNase + H3K27ac + composite. DNase is the default.
        assert set(EPINFORMERSEQ_AVAILABLE_ASSAYS) == {
            "Enhancer_DNase", "Enhancer_H3K27ac", "Enhancer_H3K27ac_DNase",
        }
        assert EPINFORMERSEQ_DEFAULT_ASSAY == "Enhancer_DNase"

    def test_window_and_step(self):
        assert EPINFORMERSEQ_WINDOW == 1024          # profile/aggregation crop
        assert EPINFORMERSEQ_WIDE_WINDOW == 2114     # model input
        assert EPINFORMERSEQ_DEFAULT_STEP == 64


class TestEPInformerSeqOracleInit:
    """Oracle class can be constructed without loading weights."""

    def test_default_init(self):
        oracle = EPInformerSeqOracle(use_environment=False)
        assert oracle.cell_type == "K562"
        assert oracle.assay == "Enhancer_DNase"
        assert oracle.sequence_length == 2114
        assert oracle.bin_size == 64
        assert oracle.loaded is False
        assert oracle._main_model is None
        assert oracle._bias_model is None

    def test_init_with_cell_type(self):
        for c in ["K562", "GM12878", "HepG2", "A549", "H1", "HSMM", "NHLF"]:
            oracle = EPInformerSeqOracle(cell_type=c, use_environment=False)
            assert oracle.cell_type == c

    def test_list_methods(self):
        oracle = EPInformerSeqOracle(use_environment=False)
        assays = oracle.list_assay_types()
        assert "Enhancer_DNase" in assays
        cells = oracle.list_cell_types()
        assert "K562" in cells  # default cell

    def test_main_weights_path_per_cell(self, tmp_path):
        """The oracle resolves main.pt under per_cell_widewin/<cell>/."""
        # Build a fake model_dir with the expected layout
        for cell in ["K562", "GM12878"]:
            (tmp_path / "per_cell_widewin" / cell).mkdir(parents=True)
            (tmp_path / "bias" / cell).mkdir(parents=True)
            (tmp_path / "per_cell_widewin" / cell / "main.pt").touch()
            (tmp_path / "bias" / cell / "bias.pt").touch()
        oracle = EPInformerSeqOracle(
            cell_type="GM12878", use_environment=False, model_dir=str(tmp_path)
        )
        assert oracle.get_main_weights_path() == tmp_path / "per_cell_widewin" / "GM12878" / "main.pt"
        assert oracle.get_bias_weights_path() == tmp_path / "bias" / "GM12878" / "bias.pt"


class TestEPInformerSeqWeightsProbe:
    """Weights probe matches the new per-cell layout."""

    def test_probe_function(self):
        from chorus.core.weights_probe import _probe_epinformerseq
        ok, missing = _probe_epinformerseq()
        # We don't assert ok=True (CI may not have weights), but if FALSE,
        # the missing paths must reference per_cell + bias, not the old K562/weights.pt.
        if not ok:
            joined = " ".join(missing)
            assert "per_cell_widewin" in joined, f"probe still uses legacy path: {missing}"
            assert "bias" in joined,     f"probe missing bias path: {missing}"

    def test_probe_registered(self):
        from chorus.core.weights_probe import _ARTIFACT_PROBES
        assert "epinformerseq" in _ARTIFACT_PROBES


# ---------------------------------------------------------------------------
# CDF / background checks
# ---------------------------------------------------------------------------

class TestEPInformerSeqCDF:
    """Background CDF artifact, if present, has the right shape."""

    @pytest.fixture
    def cdf_path(self):
        p = Path.home() / ".chorus" / "backgrounds" / "epinformerseq_pertrack.npz"
        if not p.exists():
            pytest.skip(f"CDF not built yet: {p}")
        return p

    def test_cdf_track_ids(self, cdf_path):
        d = np.load(cdf_path, allow_pickle=False)
        try:
            tracks = [str(t) for t in d["track_ids"]]
        finally:
            d.close()
        # Three assays per cell (DNase, H3K27ac, composite) => 33 tracks.
        assays = ["Enhancer_DNase", "Enhancer_H3K27ac", "Enhancer_H3K27ac_DNase"]
        expected = {f"{a}:{c}" for c in EXPECTED_CELLS for a in assays}
        assert set(tracks) == expected, \
            f"unexpected tracks: extra={set(tracks)-expected}, missing={expected-set(tracks)}"

    def test_cdf_summary_shape(self, cdf_path):
        d = np.load(cdf_path, allow_pickle=False)
        try:
            assert "summary_cdfs" in d.files
            assert d["summary_cdfs"].shape == (33, 10_000), d["summary_cdfs"].shape
            assert d["summary_counts"].shape == (33,)
            assert (d["summary_counts"] > 1000).all(), \
                f"baseline sample counts suspiciously low: {d['summary_counts']}"
        finally:
            d.close()

    def test_cdf_monotonic(self, cdf_path):
        """Each row of summary_cdfs is sorted (it's a CDF)."""
        d = np.load(cdf_path, allow_pickle=False)
        try:
            arr = d["summary_cdfs"]
        finally:
            d.close()
        diffs = np.diff(arr, axis=1)
        assert (diffs >= -1e-6).all(), "CDF rows must be monotonically non-decreasing"


# ---------------------------------------------------------------------------
# Functional checks (skipped unless prerequisites available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_weights_available() and _torch_available()),
    reason="per-cell weights and torch (chorus-epinformerseq env) required",
)
class TestEPInformerSeqLoad:
    """Load + forward pass on a random 2114-bp sequence (one cell).

    Runs in chorus-epinformerseq env (has torch) when per-cell weights are
    cached. Skipped cleanly in the base chorus env (no torch) or when
    weights have not been fetched. Static checks above (registry, globals,
    source greps) always run, so any regression to the retired joint model
    is still caught in CI without torch.
    """

    def test_load_k562_and_predict_random(self):
        oracle = EPInformerSeqOracle(use_environment=False)
        oracle.load_pretrained_model()
        assert oracle.loaded
        # Per-cell architecture has no cell_id in forward
        assert type(oracle._main_model).__name__ == "PerCellProfileNetWide"
        assert type(oracle._bias_model).__name__ == "BiasNet"

        import random
        random.seed(42)
        seq = "".join(random.choices("ACGT", k=EPINFORMERSEQ_WIDE_WINDOW))
        result = oracle.predict(seq, ["Enhancer_DNase:K562"])
        track = next(iter(result.values()))
        val = float(track.values[0])
        assert val > 0, f"K562 random-sequence activity should be > 0: {val}"
        assert np.isfinite(val), f"non-finite activity: {val}"

    def test_switch_cell_type_reloads(self):
        oracle = EPInformerSeqOracle(use_environment=False)
        oracle.load_pretrained_model()
        prev_main = oracle._main_model
        # Switch
        oracle.load_pretrained_model(cell_type="GM12878")
        assert oracle.cell_type == "GM12878"
        # main_model object should change (per-cell ckpt swap)
        assert oracle._main_model is not prev_main
