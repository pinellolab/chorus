"""Lightweight on-disk probe for oracle weights.

Runs in the base ``chorus`` environment (no heavy deps). Answers: *is
``<oracle>`` ready to predict without a surprise network fetch?* Used by
``chorus health`` to distinguish "not installed" from "unhealthy", and
by ``chorus setup`` to skip work that is already done.

A setup-complete marker file ``downloads/<oracle>/.chorus_setup_v1`` is
written by ``chorus setup <oracle>`` once env + weights (+ backgrounds,
+ auth for gated oracles) are in place. The marker is the primary
signal; we additionally verify the expected weight artifacts for
oracles where chorus controls the cache path, and re-check HF auth for
AlphaGenome on every probe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

from .globals import CHORUS_DOWNLOADS_DIR

SETUP_MARKER_NAME = ".chorus_setup_v1"


def setup_marker_path(oracle: str) -> Path:
    """Path to the setup-complete marker for ``oracle``."""
    return CHORUS_DOWNLOADS_DIR / oracle.lower() / SETUP_MARKER_NAME


def _probe_sei() -> Tuple[bool, List[str]]:
    d = CHORUS_DOWNLOADS_DIR / "sei" / "model"
    required = [
        d / "sei.pth",
        d / "projvec_targets.npy",
        d / "histone_inds.npy",
        d / "target.names",
        d / "seqclass_info.txt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    return (not missing, missing)


def _probe_legnet() -> Tuple[bool, List[str]]:
    # Matches LegNetOracle.__init__ defaults: assay=LentiMPRA, cell_type=HepG2.
    default = CHORUS_DOWNLOADS_DIR / "legnet" / "LentiMPRA_HepG2"
    if not default.exists() or not any(default.iterdir()):
        return (False, [str(default)])
    return (True, [])


def _probe_chrombpnet() -> Tuple[bool, List[str]]:
    # ChromBPNet has no constructor-level default; `chorus setup chrombpnet`
    # pre-downloads the canonical DNASE / K562 model.
    default = CHORUS_DOWNLOADS_DIR / "chrombpnet" / "DNASE_K562"
    if not default.exists() or not any(default.iterdir()):
        return (False, [str(default)])
    return (True, [])


def _probe_alphagenome() -> Tuple[bool, List[str]]:
    # Gated HF model. The marker alone isn't enough — a token can be
    # revoked after setup — so we re-check auth on every probe.
    try:
        from huggingface_hub import whoami  # type: ignore
    except Exception as exc:  # pragma: no cover - dep always present
        return (False, [f"huggingface_hub import failed: {exc}"])
    try:
        whoami()
        return (True, [])
    except Exception as exc:
        return (
            False,
            [
                "HuggingFace auth failed "
                f"({type(exc).__name__}: {exc}). Set HF_TOKEN or run "
                "'huggingface-cli login'."
            ],
        )


def _probe_library_cached() -> Tuple[bool, List[str]]:
    # Enformer (TF Hub hashed cache) and Borzoi (HF Hub cache) are
    # library-managed. We can't cheaply verify their caches from the
    # base env, so we lean on the setup marker as proof that `chorus
    # setup` completed a successful first-load. The marker is checked
    # in the outer dispatch.
    return (True, [])


_ARTIFACT_PROBES: Dict[str, Callable[[], Tuple[bool, List[str]]]] = {
    "sei": _probe_sei,
    "legnet": _probe_legnet,
    "chrombpnet": _probe_chrombpnet,
    "alphagenome": _probe_alphagenome,
    "enformer": _probe_library_cached,
    "borzoi": _probe_library_cached,
}


def probe_weights(oracle: str) -> Tuple[bool, List[str]]:
    """Return ``(installed, reasons)``.

    ``installed`` is True when ``chorus setup <oracle>`` has completed
    *and* the expected artifacts are still in place (for oracles with
    cheap artifact probes) *and* (for AlphaGenome) HF auth still works.
    Otherwise ``reasons`` lists what's missing.
    """
    name = oracle.lower()

    reasons: List[str] = []
    if not setup_marker_path(name).exists():
        reasons.append(
            f"setup marker missing: {setup_marker_path(name)}"
        )

    probe = _ARTIFACT_PROBES.get(name)
    if probe is None:
        # Unknown oracle — don't block; just trust the marker.
        return (not reasons, reasons)

    ok, artifact_reasons = probe()
    if not ok:
        reasons.extend(artifact_reasons)

    return (not reasons, reasons)


def write_setup_marker(oracle: str) -> Path:
    """Write the setup-complete marker for ``oracle`` and return its path.

    Called at the end of a successful ``chorus setup <oracle>`` run,
    after env build + weight pre-download (+ auth validation for
    AlphaGenome) have all completed. Parent directory is created if
    missing.
    """
    path = setup_marker_path(oracle)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path
