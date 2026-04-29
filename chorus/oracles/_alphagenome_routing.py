"""Backend-routing helper for AlphaGenome oracles.

Two AlphaGenome backends ship with chorus: the JAX reference
implementation (`alphagenome`, default) and the upstream PyTorch port
(`alphagenome_pt`). Their public API is interchangeable; their speed
profiles are not. This helper lets users (or `chorus.create_oracle`)
ask "which backend should I use for this query?" and get a concrete
recommendation grounded in the audit numbers from
``audits/2026-04-29_alphagenome_pytorch_spike/``.

Heuristics derive from a single-machine benchmark (Apple M3 Ultra,
96 GB unified, macOS arm64). Linux/CUDA recommendations are
qualitative — defer to user's actual hardware for confirmation.
"""

from __future__ import annotations

import platform as _platform
from typing import Optional


# Empirical cliff observed on M3 Ultra MPS at the 768→896 kb boundary.
# Pick a conservative threshold below the cliff so cumulative session
# pressure (which can pull the cliff earlier) doesn't bite users.
_MAC_MPS_SAFE_WINDOW_BP = 600_000


def _detect_capabilities() -> dict:
    """Probe what's actually available on this host.

    Tries torch first (authoritative when installed); falls back to a
    platform/architecture heuristic when torch isn't available — chorus
    base env doesn't ship torch, but the routing helper still needs to
    give a useful answer there.
    """
    system = _platform.system()
    machine = _platform.machine().lower()
    has_cuda = False
    has_mps = False
    torch_seen = False
    try:
        import torch  # noqa: WPS433 — best-effort probe
        torch_seen = True
        has_cuda = bool(torch.cuda.is_available())
        has_mps = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception:
        pass

    # Fallback heuristic: macOS on arm64 (Apple Silicon) implies MPS-capable.
    # macOS x86_64 (Intel Mac) does NOT have MPS.
    if not torch_seen and system == "Darwin" and machine in {"arm64", "aarch64"}:
        has_mps = True

    # Fallback for Linux: assume CUDA absent unless torch confirmed it.
    # (Probing nvidia-smi here would be expensive; users on CUDA boxes
    # generally have torch installed in the right env to confirm.)
    return {"system": system, "has_cuda": has_cuda, "has_mps": has_mps}


def recommend_alphagenome_backend(
    window_size_bp: int,
    *,
    system: Optional[str] = None,
    has_cuda: Optional[bool] = None,
    has_mps: Optional[bool] = None,
) -> dict:
    """Suggest which AlphaGenome backend to use for a given query.

    Parameters
    ----------
    window_size_bp:
        Centred input window in base pairs (e.g. ``524288`` for a 512 kb
        query, ``1_048_576`` for a 1 MB query).
    system, has_cuda, has_mps:
        Override host detection — intended for testing or for asking
        "what would you recommend if I had X?".

    Returns
    -------
    dict with keys:
        ``oracle``: ``'alphagenome'`` (JAX) or ``'alphagenome_pt'`` (PT)
        ``device``: suggested device string (or ``None`` for default)
        ``reason``: one-sentence English explanation
        ``confidence``: ``'high'`` (benchmarked) or ``'medium'`` (extrapolated)
        ``benchmarks``: short table of supporting numbers (subset of audit)

    Notes
    -----
    Numbers come from
    ``audits/2026-04-29_alphagenome_pytorch_spike/diagnostic_mps_pressure.md``.
    The Apple-Silicon recommendation derives from an empirical cliff in
    GPU on-die cache locality at ~768→896 kb on M3 Ultra; we use a
    600 kb safety margin. Linux/CUDA is qualitative pending the user's
    own benchmark.
    """
    if system is None or has_cuda is None or has_mps is None:
        caps = _detect_capabilities()
        system = caps["system"] if system is None else system
        has_cuda = caps["has_cuda"] if has_cuda is None else has_cuda
        has_mps = caps["has_mps"] if has_mps is None else has_mps

    if window_size_bp <= 0:
        raise ValueError(f"window_size_bp must be positive, got {window_size_bp}")

    benchmarks_mac = {
        "JAX CPU @128kb": "5.09 s",
        "JAX CPU @524kb": "21.96 s",
        "JAX CPU @1MB":   "50.29 s",
        "PT MPS @128kb":  "0.61 s",
        "PT MPS @524kb":  "3.77 s",
        "PT MPS @896kb":  "84.87 s (cliff)",
        "PT MPS @1MB":    "174.48 s (post-cliff)",
    }

    # Linux + CUDA: PyTorch CUDA is the standard fastest path. We lack
    # benchmarks for this machine but the recommendation is well
    # supported by the upstream port's design.
    if system == "Linux" and has_cuda:
        return {
            "oracle": "alphagenome_pt",
            "device": "cuda",
            "reason": (
                "Linux/CUDA: the PyTorch port is expected to be the fastest "
                "path at every window size. Benchmark on your hardware to "
                "confirm — chorus has not run a CUDA pin."
            ),
            "confidence": "medium",
            "benchmarks": {},
        }

    # macOS + MPS: PyTorch MPS wins by 5–8× for windows that fit in
    # GPU on-die cache (~600 kb safety margin under the empirical cliff
    # at 768→896 kb on M3 Ultra).
    if system == "Darwin" and has_mps:
        if window_size_bp <= _MAC_MPS_SAFE_WINDOW_BP:
            return {
                "oracle": "alphagenome_pt",
                "device": "mps",
                "reason": (
                    f"macOS + MPS, window {window_size_bp:,} bp ≤ "
                    f"{_MAC_MPS_SAFE_WINDOW_BP:,} bp safe-zone: the PyTorch "
                    "port on MPS is 5–8× faster than JAX CPU for queries "
                    "that fit in GPU on-die cache."
                ),
                "confidence": "high",
                "benchmarks": benchmarks_mac,
            }
        return {
            "oracle": "alphagenome",
            "device": "cpu",
            "reason": (
                f"macOS + MPS, window {window_size_bp:,} bp > "
                f"{_MAC_MPS_SAFE_WINDOW_BP:,} bp safe-zone: past the GPU "
                "cache cliff at ~768–896 kb, MPS regresses to 3–4× slower "
                "than JAX CPU. Stay on JAX."
            ),
            "confidence": "high",
            "benchmarks": benchmarks_mac,
        }

    # macOS without MPS, or Linux without CUDA, or Windows: stay on
    # JAX CPU. PyTorch CPU is consistently slower than JAX CPU for this
    # model in our benchmarks (likely XLA matmul kernels beating eager
    # PyTorch on Mac CPU; assume similar on Linux).
    return {
        "oracle": "alphagenome",
        "device": "cpu",
        "reason": (
            f"{system} without GPU acceleration: JAX CPU is ~30–65 % "
            "faster than PyTorch CPU for this model on our benchmark "
            "machine. Use the JAX backend."
        ),
        "confidence": "medium",
        "benchmarks": benchmarks_mac if system == "Darwin" else {},
    }


def format_recommendation(rec: dict) -> str:
    """Pretty-print a recommendation as a 2-3 line block for CLI / repl."""
    lines = [
        f"  Recommended oracle: {rec['oracle']!r}  (device={rec['device']!r}, "
        f"confidence={rec['confidence']})",
        f"  Reason: {rec['reason']}",
    ]
    if rec.get("benchmarks"):
        lines.append("  Supporting benchmarks (M3 Ultra):")
        for k, v in rec["benchmarks"].items():
            lines.append(f"    {k:<20s} {v}")
    return "\n".join(lines)
