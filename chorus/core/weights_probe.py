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


def _hf_cache_dir() -> Path:
    """Resolve the HuggingFace hub cache directory.

    Prefer the canonical ``huggingface_hub.constants.HF_HUB_CACHE`` (which
    honours ``HF_HOME`` / ``HF_HUB_CACHE`` env vars); fall back to the
    documented default ``~/.cache/huggingface/hub`` if the import fails
    (chorus base env always has huggingface_hub, so the fallback is
    defensive).
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def _probe_chrombpnet() -> Tuple[bool, List[str]]:
    """Accept either the 0.3.0+ slim HF mirror or the legacy ENCODE-tarball
    layout as proof of install.

    Default in 0.3.0+: weights stream from
    ``lucapinello/chorus-chrombpnet-slim`` and live in the HF hub cache.
    The local ``downloads/chrombpnet/DNASE_K562`` directory is only
    populated for users who explicitly request ``model_type='chrombpnet'``
    (bias-aware) or fold ≠ 0, both of which fall back to ENCODE tarballs.
    `chorus setup chrombpnet` succeeds on the slim path, so we must
    accept either cache as installed.
    """
    # 0.3.0+ default: slim HF mirror.
    slim_snapshots = (
        _hf_cache_dir() / "models--lucapinello--chorus-chrombpnet-slim" / "snapshots"
    )
    if slim_snapshots.exists():
        for snap in slim_snapshots.iterdir():
            if (snap / "manifest.json").exists():
                return (True, [])
    # Legacy ENCODE-tarball cache.
    legacy = CHORUS_DOWNLOADS_DIR / "chrombpnet" / "DNASE_K562"
    if legacy.exists() and any(legacy.iterdir()):
        return (True, [])
    return (
        False,
        [
            f"neither HF slim mirror cache ({slim_snapshots.parent}) "
            f"nor legacy {legacy} is populated"
        ],
    )


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


def _probe_alphagenome_pt() -> Tuple[bool, List[str]]:
    """Check the HF cache for the upstream PyTorch port's safetensors.

    Weights live at ``models--gtca--alphagenome_pytorch/snapshots/<rev>/
    model_all_folds.safetensors`` after ``chorus setup --oracle
    alphagenome_pt`` (or the first ``load_pretrained_model()`` call).
    """
    pt_snapshots = (
        _hf_cache_dir() / "models--gtca--alphagenome_pytorch" / "snapshots"
    )
    if pt_snapshots.exists():
        for snap in pt_snapshots.iterdir():
            if any(p.suffix == ".safetensors" for p in snap.iterdir()):
                return (True, [])
    return (
        False,
        [f"HF cache not populated: {pt_snapshots.parent}"],
    )


_ARTIFACT_PROBES: Dict[str, Callable[[], Tuple[bool, List[str]]]] = {
    "sei": _probe_sei,
    "legnet": _probe_legnet,
    "chrombpnet": _probe_chrombpnet,
    "alphagenome": _probe_alphagenome,
    "alphagenome_pt": _probe_alphagenome_pt,
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
