"""Post-env-build pre-download for `chorus setup <oracle>`.

After ``EnvironmentManager.create_environment`` builds the conda env,
this module drives the "wait once, then ready" part: pull the oracle's
default weights, background CDFs from HuggingFace, and (once per
session) the hg38 reference. On success the caller writes the
setup-complete marker via ``chorus.core.weights_probe.write_setup_marker``.

Design notes
------------
* **Weights** are fetched by spawning a one-shot script in the oracle's
  conda env that does ``chorus.create_oracle(name, use_environment=False)``
  + ``.load_pretrained_model(<default-args>)``. This reuses each
  oracle's existing lazy-download path — no parallel downloader per
  oracle.
* **Backgrounds** live in the ``lucapinello/chorus-backgrounds`` HF
  dataset; ``download_backgrounds(oracle)`` in ``chorus.analysis.normalization``
  is already idempotent and skips files that are already cached.
* **Genome** is handled by ``GenomeManager.download_genome("hg38")`` —
  also idempotent.
* All three steps are skippable via flags, but the default is "do all"
  so ``chorus setup`` genuinely leaves the oracle ready to predict.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default kwargs for prefetching a canonical model config. These are
# split into constructor kwargs vs load_pretrained_model kwargs per
# oracle because they disagree on which layer takes which arg:
#   - LegNet: assay/cell_type live on __init__; load_pretrained_model
#     takes no required args.
#   - ChromBPNet: assay/cell_type/fold are load_pretrained_model args.
#   - Everything else (enformer, borzoi, sei, alphagenome): no required
#     config for prefetch — bare load_pretrained_model().
# Passing ChromBPNet-style kwargs to LegNet.load_pretrained_model breaks
# `chorus setup --oracle legnet` with a TypeError (v22 audit finding).
_DEFAULT_CTOR_KWARGS: Dict[str, Dict[str, object]] = {
    "legnet": {"assay": "LentiMPRA", "cell_type": "HepG2"},
}
_DEFAULT_LOAD_KWARGS: Dict[str, Dict[str, object]] = {
    "chrombpnet": {"assay": "DNASE", "cell_type": "K562", "fold": 0},
}


def _weight_prefetch_script(oracle: str) -> str:
    """Render a throwaway script that loads the oracle once.

    Runs inside the oracle's conda env via ``EnvironmentRunner``. The
    first ``load_pretrained_model`` call triggers any lazy downloads
    (ENCODE tarballs, Zenodo tarballs, TF Hub, HF Hub) into their
    respective caches.
    """
    ctor_kwargs = _DEFAULT_CTOR_KWARGS.get(oracle.lower(), {})
    load_kwargs = _DEFAULT_LOAD_KWARGS.get(oracle.lower(), {})
    ctor_repr = "".join(f", {k}={v!r}" for k, v in ctor_kwargs.items())
    load_repr = ", ".join(f"{k}={v!r}" for k, v in load_kwargs.items())
    # ``use_environment=False`` makes the oracle load directly in the
    # current subprocess (we are already inside the oracle's conda env
    # via ``run_script_in_environment``); chorus.__init__.create_oracle
    # propagates this flag into the instance so no nested subprocess is
    # spawned.
    return f"""
import json, sys
import chorus
try:
    oracle = chorus.create_oracle({oracle!r}, use_environment=False{ctor_repr})
    oracle.load_pretrained_model({load_repr})
    print(json.dumps({{'success': True}}))
except Exception as exc:
    import traceback
    print(json.dumps({{'success': False, 'error': f'{{type(exc).__name__}}: {{exc}}', 'traceback': traceback.format_exc()}}))
"""


def prefetch_weights(oracle: str, runner, timeout: Optional[int] = None) -> Tuple[bool, Optional[str]]:
    """Trigger the oracle's lazy-download path by loading the default model.

    Returns ``(success, error_message)``.
    """
    import json

    logger.info("Pre-downloading %s weights ...", oracle)
    # No hard cap: large Zenodo archives (3 GB+) can't finish under the
    # default 120 s health timeout. Let the network take what it needs.
    # Callers can pass an explicit timeout for tests.
    script = _weight_prefetch_script(oracle)
    result = runner.run_script_in_environment(oracle, script, timeout=timeout)

    if result.returncode != 0:
        return (False, f"Subprocess exited {result.returncode}: {result.stderr[-500:]}.")

    try:
        payload = json.loads(result.stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return (False, f"Unparseable output: {result.stdout[-500:]}.")

    if payload.get("success"):
        logger.info("✓ %s weights ready", oracle)
        return (True, None)
    return (False, f"{payload.get('error', 'unknown load failure')}.")


def prefetch_backgrounds(oracle: str) -> Tuple[bool, Optional[str]]:
    """Pull background CDFs for ``oracle`` from HuggingFace. Idempotent."""
    try:
        from ..analysis.normalization import download_backgrounds
    except Exception as exc:
        return (False, f"Import failed: {exc}.")

    try:
        n = download_backgrounds(oracle)
    except Exception as exc:
        return (False, f"Download failed: {exc}.")

    if n == 0:
        logger.info("Backgrounds for %s already cached (or unavailable)", oracle)
    else:
        logger.info("✓ pulled %d background file(s) for %s", n, oracle)
    return (True, None)


def prefetch_genome(genome_id: str = "hg38") -> Tuple[bool, Optional[str]]:
    """Download the reference genome if it isn't already present. Idempotent."""
    try:
        from ..utils.genome import GenomeManager
    except Exception as exc:
        return (False, f"Import failed: {exc}.")

    mgr = GenomeManager()
    if mgr.is_genome_downloaded(genome_id):
        logger.info("Reference genome %s already present", genome_id)
        return (True, None)

    logger.info("Pre-downloading reference genome %s ...", genome_id)
    try:
        ok = mgr.download_genome(genome_id)
    except Exception as exc:
        return (False, f"Download failed: {exc}.")

    if not ok:
        return (False, f"{genome_id} download returned False. Retry with `chorus genome download {genome_id}`.")
    logger.info("✓ %s ready", genome_id)
    return (True, None)


def prefetch_for_oracle(
    oracle: str,
    runner,
    *,
    skip_weights: bool = False,
    skip_backgrounds: bool = False,
    skip_genome: bool = False,
    weights_timeout: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    """Run all applicable prefetch steps for ``oracle``.

    Returns ``(all_succeeded, error_messages)``. An empty error list on
    success lets the caller safely write the setup-complete marker.
    """
    errors: List[str] = []

    if not skip_weights:
        ok, err = prefetch_weights(oracle, runner, timeout=weights_timeout)
        if not ok:
            errors.append(f"weights: {err}")

    if not skip_backgrounds:
        ok, err = prefetch_backgrounds(oracle)
        if not ok:
            errors.append(f"backgrounds: {err}")

    if not skip_genome:
        ok, err = prefetch_genome("hg38")
        if not ok:
            errors.append(f"genome: {err}")

    return (not errors, errors)
