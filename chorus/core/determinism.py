"""Make JAX/XLA reproducible across processes.

WHY THIS EXISTS
---------------
#127 recorded two identical chorus runs differing on 454 numeric fields with 36
sign flips, and for CAGE the run-to-run noise (median 0.0054) *exceeded* the
median effect being reported (0.0058) — so 92.1 % of shipped CAGE rows, the
gene-TSS ones, were ranking noise rather than signal.

Measured on this box (8xH100, AlphaGenome via ``create_from_huggingface``,
identical 1,048,576 bp sequence at SORT1, all 9 output types):

===================================  ==================  ==================
configuration                        cross-process       steady-state pass
===================================  ==================  ==================
no flags                             **not reproducible**, 0.6 s
                                     ~1e-2 relative
``--xla_gpu_deterministic_ops=true`` 9/9 bit-exact       1.2 s
``--xla_gpu_autotune_level=0``       9/9 bit-exact       108 s
===================================  ==================  ==================

Three findings behind those numbers:

* **Within one process AlphaGenome is already bit-exact** — three repeated
  ``predict_sequence`` calls gave ``0.000e+00`` on every output type with no
  flags at all. So the model is not the problem and there is no inference RNG
  (``ApplyFn`` takes no PRNG key; only ``init_fn`` does, at a fixed
  ``PRNGKey(0)``).
* **The divergence is per-process, not per-device.** Two processes on the *same*
  GPU differ exactly as much as two on different GPUs (0/9 bit-exact, worst
  relative 1.6e-2), which points at compilation rather than hardware.
* The absolute differences land on bfloat16 quantisation steps — 0.5, 3.0, 128.0,
  0.125, 7.8125e-3 (= 1/128) — and the relative errors, 4e-3 to 1.6e-2, are 1-3
  ULP of bfloat16's ``2**-8``. Different accumulation orders, amplified by
  reduced output precision.

``--xla_gpu_autotune_level=0`` also fixes it, by making kernel choice
independent of benchmark timings, but at **180x** the steady-state cost. It is
deliberately *not* set: it is the expensive way to buy something
``deterministic_ops`` already gives.

WHY EXACTNESS RATHER THAN "CLOSE ENOUGH"
----------------------------------------
A ~1 % relative wobble sounds harmless and is not: it is the same size as the
CAGE effects chorus prints, which is why those rows could not be ranked at all.
The cost of removing it is ~0.6 s per forward pass — about 3.75 extra GPU-hours
spread across a full AlphaGenome rebuild.

A background and the queries scored against it MUST run under the same setting.
Mixing them computes the null and the numerator differently, which is a fresh
instance of the statistic-mismatch class in #144. The chosen flags are recorded
in the NPZ provenance for that reason.
"""
from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

# The one flag we set, and the reason it is this one and not the other.
DETERMINISTIC_XLA_FLAGS = "--xla_gpu_deterministic_ops=true"

# Escape hatch. Set CHORUS_NONDETERMINISTIC=1 to skip pinning — for measuring
# the noise itself (scripts/probe_alphagenome_determinism.py) or for a
# throughput experiment where reproducibility is explicitly not wanted.
_OPT_OUT = "CHORUS_NONDETERMINISTIC"


def pin_deterministic_xla_flags(*, force: bool = False) -> str:
    """Add the determinism flag to ``XLA_FLAGS``. Call *before* importing jax.

    Idempotent, and preserves any flags the caller already set. Returns the
    resulting ``XLA_FLAGS``.

    Warns rather than raises when ``jax`` is already imported: by then XLA has
    read its configuration and the flag will not take effect, but failing hard
    would break callers who legitimately loaded jax first.
    """
    if os.environ.get(_OPT_OUT) == "1" and not force:
        logger.warning(
            "%s=1: leaving XLA_FLAGS alone. Predictions will NOT be reproducible "
            "across processes (~1e-2 relative, #127).", _OPT_OUT,
        )
        return os.environ.get("XLA_FLAGS", "")

    existing = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_deterministic_ops" not in existing:
        os.environ["XLA_FLAGS"] = f"{existing} {DETERMINISTIC_XLA_FLAGS}".strip()

    if "jax" in sys.modules:
        logger.warning(
            "jax is already imported, so XLA_FLAGS=%r may not take effect; "
            "predictions could vary across processes (#127). Call "
            "pin_deterministic_xla_flags() before importing jax.",
            os.environ["XLA_FLAGS"],
        )
    return os.environ["XLA_FLAGS"]


def determinism_provenance() -> dict[str, object]:
    """What to stamp into a background so its queries can match it (#124)."""
    return {
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "deterministic": "xla_gpu_deterministic_ops" in os.environ.get("XLA_FLAGS", ""),
    }
