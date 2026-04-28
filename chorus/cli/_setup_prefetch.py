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
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Default kwargs for prefetching a canonical model config. These are
# split into constructor kwargs vs load_pretrained_model kwargs per
# oracle because they disagree on which layer takes which arg:
#   - LegNet: assay/cell_type live on __init__; load_pretrained_model
#     takes no required args.
#   - ChromBPNet: assay/cell_type/fold are load_pretrained_model args.
#     Pre-fetch BOTH K562 + HepG2 because the shipped notebooks use
#     both (advanced_multi_oracle_analysis.ipynb and
#     comprehensive_oracle_showcase.ipynb hit HepG2 mid-run; v27 P1
#     finding). Each entry in the list triggers one load → one ENCODE
#     tarball download.
#   - Everything else (enformer, borzoi, sei, alphagenome): no required
#     config for prefetch — bare load_pretrained_model().
# Passing ChromBPNet-style kwargs to LegNet.load_pretrained_model breaks
# `chorus setup --oracle legnet` with a TypeError (v22 audit finding).
_DEFAULT_CTOR_KWARGS: Dict[str, Dict[str, object]] = {
    "legnet": {"assay": "LentiMPRA", "cell_type": "HepG2"},
}
# A LIST of dicts means "load each in sequence" (e.g. for pre-caching
# multiple ChromBPNet (assay, cell_type) tarballs). A single dict means
# "load once with these kwargs". Empty / missing means "bare call".
# ChromBPNet uses a special prefetch script (see _chrombpnet_prefetch_script)
# that iterates ALL registered models, so it is NOT listed here.
_DEFAULT_LOAD_KWARGS: Dict[str, Union[Dict[str, object], List[Dict[str, object]]]] = {
    # ChromBPNet fast-path default: pre-cache K562 + HepG2 DNase only
    # (~1.4 GB total). The shipped quickstart + advanced + comprehensive
    # notebooks all use these two. Pre-caching every published model is
    # opt-in via `chorus setup --oracle chrombpnet --all-chrombpnet`.
    "chrombpnet": [
        {"assay": "DNASE", "cell_type": "K562",  "fold": 0},
        {"assay": "DNASE", "cell_type": "HepG2", "fold": 0},
    ],
}


def _chrombpnet_prefetch_script() -> str:
    """Prefetch ALL ChromBPNet + BPNet model weights.

    Iterates every registered ATAC/DNASE model (42 unique ENCFFs) and
    every CHIP/BPNet model (744 TF x cell_type pairs), downloading each
    on first access.  This makes discovery mode work without mid-analysis
    download delays.
    """
    return """
import json, sys, os, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("prefetch")

import chorus
from chorus.oracles.chrombpnet_source.chrombpnet_globals import (
    iter_unique_models, iter_unique_bpnet_models,
)

oracle = chorus.create_oracle("chrombpnet", use_environment=False)

# --- ATAC / DNASE models (42 unique) ---
atac_dnase = list(iter_unique_models())
logger.info("Prefetching %d ChromBPNet ATAC/DNASE models ...", len(atac_dnase))
n_ok, n_fail = 0, 0
for i, (assay, cell_type, encff) in enumerate(atac_dnase, 1):
    try:
        oracle.load_pretrained_model(assay=assay, cell_type=cell_type, fold=0)
        n_ok += 1
        logger.info("  [%d/%d] %s:%s OK", i, len(atac_dnase), assay, cell_type)
    except Exception as exc:
        n_fail += 1
        logger.warning("  [%d/%d] %s:%s FAILED: %s", i, len(atac_dnase), assay, cell_type, exc)
logger.info("ATAC/DNASE: %d OK, %d failed", n_ok, n_fail)

# --- CHIP / BPNet models (744 unique) ---
chip_models = list(iter_unique_bpnet_models())
logger.info("Prefetching %d BPNet/CHIP models ...", len(chip_models))
n_ok_chip, n_fail_chip = 0, 0
for i, (cell_type, tf, model_url, identifier) in enumerate(chip_models, 1):
    try:
        oracle.load_pretrained_model(assay="CHIP", cell_type=cell_type, TF=tf, fold=0)
        n_ok_chip += 1
        if i % 50 == 0 or i == len(chip_models):
            logger.info("  [%d/%d] CHIP models downloaded (%d OK so far)", i, len(chip_models), n_ok_chip)
    except Exception as exc:
        n_fail_chip += 1
        if n_fail_chip <= 5:
            logger.warning("  CHIP:%s:%s FAILED: %s", cell_type, tf, exc)

logger.info("CHIP: %d OK, %d failed", n_ok_chip, n_fail_chip)
total_fail = n_fail + n_fail_chip
print(json.dumps({"success": total_fail == 0, "atac_dnase": n_ok, "chip": n_ok_chip, "failed": total_fail}))
"""


def _weight_prefetch_script(oracle: str) -> str:
    """Render a throwaway script that loads the oracle once (or N times).

    Runs inside the oracle's conda env via ``EnvironmentRunner``. The
    first ``load_pretrained_model`` call triggers any lazy downloads
    (ENCODE tarballs, Zenodo tarballs, TF Hub, HF Hub) into their
    respective caches. When ``_DEFAULT_LOAD_KWARGS[oracle]`` is a list
    we loop over entries, each one tripping one cache fetch, so a
    single ``chorus setup`` run pre-caches every (assay, cell_type)
    pair the shipped notebooks need.
    """
    ctor_kwargs = _DEFAULT_CTOR_KWARGS.get(oracle.lower(), {})
    load_kwargs = _DEFAULT_LOAD_KWARGS.get(oracle.lower(), {})
    ctor_repr = "".join(f", {k}={v!r}" for k, v in ctor_kwargs.items())

    # Normalize to a list of load-kwarg dicts. Empty dict → single bare
    # call; single dict → single call with those kwargs; list of dicts
    # → one call per dict.
    if isinstance(load_kwargs, list):
        load_calls = load_kwargs
    else:
        load_calls = [load_kwargs]
    load_reprs = [
        ", ".join(f"{k}={v!r}" for k, v in entry.items())
        for entry in load_calls
    ]
    load_lines = "\n    ".join(
        f"oracle.load_pretrained_model({rep})" for rep in load_reprs
    )

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
    {load_lines}
    print(json.dumps({{'success': True}}))
except Exception as exc:
    import traceback
    print(json.dumps({{'success': False, 'error': f'{{type(exc).__name__}}: {{exc}}', 'traceback': traceback.format_exc()}}))
"""


def prefetch_weights(
    oracle: str,
    runner,
    timeout: Optional[int] = None,
    *,
    full_chrombpnet: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Trigger the oracle's lazy-download path by loading the default model.

    For ChromBPNet, the default fast path pre-caches only K562 + HepG2
    DNase (~1.4 GB; matches the shipped notebooks). Pass
    ``full_chrombpnet=True`` to instead pre-cache all 786 ChromBPNet +
    BPNet models (~76 GB on disk after extraction; 42 × ~720 MB
    tarballs + 5-fold extracted weights, plus 744 tiny BPNet h5 files;
    3-4 hours of network downloads). All other
    oracles ignore this flag.

    Returns ``(success, error_message)``.
    """
    import json

    logger.info("Pre-downloading %s weights ...", oracle)
    # No hard cap: large Zenodo archives (3 GB+) can't finish under the
    # default 120 s health timeout. Let the network take what it needs.
    # Callers can pass an explicit timeout for tests.
    if oracle.lower() == "chrombpnet" and full_chrombpnet:
        script = _chrombpnet_prefetch_script()
    else:
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
    full_chrombpnet: bool = False,
) -> Tuple[bool, List[str]]:
    """Run all applicable prefetch steps for ``oracle``.

    Returns ``(all_succeeded, error_messages)``. An empty error list on
    success lets the caller safely write the setup-complete marker.

    ``full_chrombpnet=True`` switches the chrombpnet weight prefetch from
    the fast 2-model default (~1.4 GB) to the full 786-model catalogue
    (~76 GB on disk). Has no effect on other oracles.
    """
    errors: List[str] = []

    if not skip_weights:
        ok, err = prefetch_weights(
            oracle, runner, timeout=weights_timeout,
            full_chrombpnet=full_chrombpnet,
        )
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
