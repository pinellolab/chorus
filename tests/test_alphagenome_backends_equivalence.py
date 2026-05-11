"""Numerical equivalence between AlphaGenome JAX and PyTorch backends.

Gated with ``@pytest.mark.integration`` — runs both backends inside their
respective conda envs via chorus's normal Oracle API, predicts a 1 MB
SORT1 window with each, and asserts per-track agreement.

Run from the repo root::

    pytest tests/test_alphagenome_backends_equivalence.py -m integration -v

Skips automatically if either env or the hg38 reference is missing.

History
-------
The original test (pre-#63) compared JAX and PyTorch outputs at the
*raw model head*. JAX's DNase head exposes 305 tracks and the PyTorch
port's exposes 384 — they're different upstream filtering choices,
not different weights. The shape-parity assertion was bound to fail
on any platform; it just didn't surface until the Linux/CUDA spot-check
on `audit/linux-cuda-pr62`.

The fix is to compare at the chorus-API layer instead. ``oracle.predict()``
goes through ``_local_index`` slicing in
``chorus/oracles/alphagenome.py:_predict``, which selects the user-
requested tracks by identifier from the shared 5,731-track metadata
cache (`alphagenome_tracks.json`) — so post-slicing arrays are
shape-compatible across backends regardless of raw-head shape. Closes
#63 (raw-head shape mismatch) and #65 (eliminates the bare-`mamba`
subprocess calls that broke when this test was invoked under
``mamba run -n chorus pytest``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# 1 MB SORT1 window centered at rs12740374's TSS (chr1:109,274,968).
# Same window as the v30 macOS audit and the audit/linux-cuda-pr62 audit
# so the equivalence numbers stay comparable.
SORT1_WINDOW = (
    "chr1",
    109_274_968 - 524_288,
    109_274_968 + 524_288,
)

GENOME_FASTA = Path(__file__).parent.parent / "genomes" / "hg38.fa"

# Three canonical AlphaGenome assay identifiers (format
# "{output_type}/{name}/{strand}", verified via
# chorus.oracles.alphagenome_source.alphagenome_metadata.get_metadata().search_tracks).
# Same K562 / HepG2 / hepatocyte set used in the v30 macOS Tier-1 audit.
ASSAY_IDS = [
    "DNASE/EFO:0002067 DNase-seq/.",   # K562 DNase
    "DNASE/EFO:0001187 DNase-seq/.",   # HepG2 DNase
    "DNASE/CL:0000182 DNase-seq/.",    # hepatocyte DNase
]


@pytest.mark.integration
def test_jax_pt_chorus_api_equivalence_at_sort1():
    """JAX and PyTorch AlphaGenome backends produce equivalent chorus-API outputs.

    Compares the ``oracle.predict()`` output array per assay — chorus's
    ``local_index`` slicing maps the user-requested identifier to the
    same logical track on both backends, so post-slicing arrays are
    shape-compatible regardless of raw head shape.

    Tolerances (post-Issue #81 investigation, 2026-05-11):
    - ``Pearson correlation > 0.99`` — confirms the two backends produce
      the same signal shape per track.
    - ``max(|pt - jax|) / peak_magnitude < 0.02`` (2% relative at peak).
      An absolute ``max(|d|) < 0.1`` was infeasible at signal magnitudes
      ~55 because JAX outputs are bf16-quantized
      (``jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')``
      in ``alphagenome_research/model/dna_model.py``). At magnitude 55,
      adjacent bf16 values are 0.25 apart, so the bf16 quantization alone
      contributes ~0.5% of peak. The remaining ~0.5–1% is per-position
      fp32 numerical kernel drift between torch's conv path and JAX's
      XLA conv path. Mean abs diff observed on Linux/CUDA: 0.0007 on
      1 M positions × 3 tracks (correlation 1.0000 on every track).
    - ``mean(|pt - jax|) / mean(|jax|) < 5%`` — protects against any
      systematic per-track scale mismatch.

    The earlier absolute ``< 0.1`` bound failed not because of model
    drift but because the SORT1 peak signal grew between PR #62 (when
    the bound was set, peak ~10 → bf16 grid ~0.04) and v0.5.1 (peak ~77
    → bf16 grid ~0.25). See
    ``audits/2026-05-10_v0.5.0_scorched_earth_linux_cuda/issue_81_root_cause.md``
    for the full diagnostic.
    """
    if not GENOME_FASTA.exists():
        pytest.skip(
            "hg38.fa not present — run `chorus genome download hg38` first"
        )

    # Skip cleanly when either env is missing — uses chorus's own
    # EnvironmentManager so we resolve `mamba`/`conda` the same way the
    # framework does (handles `mamba run` invocation, MAMBA_EXE, etc.).
    from chorus.core.environment import EnvironmentManager

    mgr = EnvironmentManager()
    for env in ("alphagenome", "alphagenome_pt"):
        if not mgr.environment_exists(env):
            pytest.skip(f"chorus-{env} env missing")

    import chorus

    preds = {}
    for name in ("alphagenome", "alphagenome_pt"):
        oracle = chorus.create_oracle(
            name,
            use_environment=True,
            reference_fasta=str(GENOME_FASTA),
        )
        oracle.load_pretrained_model()
        preds[name] = oracle.predict(SORT1_WINDOW, ASSAY_IDS)

    for aid in ASSAY_IDS:
        jax_track = preds["alphagenome"][aid]
        pt_track = preds["alphagenome_pt"][aid]
        a = jax_track.values
        b = pt_track.values

        assert a.shape == b.shape, (
            f"{aid}: chorus-API per-track shape mismatch "
            f"(jax={a.shape}, pt={b.shape}). The local_index slicing was "
            f"supposed to align both backends — investigate metadata."
        )
        assert a.ndim == 1, (
            f"{aid}: per-track values should be 1-D after slicing, got {a.shape}"
        )

        max_abs = float(np.abs(a - b).max())
        peak_mag = max(float(np.abs(a).max()), float(np.abs(b).max()))
        max_rel_at_peak = max_abs / (peak_mag + 1e-9)
        mean_rel = float(np.abs(a - b).mean() / (np.abs(a).mean() + 1e-9))
        if a.std() > 0 and b.std() > 0:
            corr = float(np.corrcoef(a, b)[0, 1])
        else:
            corr = 1.0  # both arrays constant — degenerate case, treat as equal

        assert corr > 0.99, (
            f"{aid}: per-track Pearson correlation {corr:.4f} ≤ 0.99 — "
            f"backends produce different signals (not just numerical drift)."
        )
        assert max_rel_at_peak < 0.02, (
            f"{aid}: max abs diff {max_abs:.4f} is {max_rel_at_peak:.2%} of "
            f"peak magnitude {peak_mag:.4f} (limit 2%). At this peak, bf16 "
            f"output quantization in JAX gives ~{peak_mag/256:.4f} intrinsic "
            f"noise, so >2% relative indicates real model divergence."
        )
        assert mean_rel < 0.05, (
            f"{aid}: mean rel diff {mean_rel:.4f} exceeds 5% — systematic "
            f"per-track scale mismatch between JAX and PyTorch backends."
        )
