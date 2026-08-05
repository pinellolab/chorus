"""Persist Triton's autotune results across processes.

Cherimoya's inference kernels are wrapped in ``triton.autotune``, which
benchmarks its candidate configs the first time a kernel is launched for a
given shape.  That is the right call for training, where a process compiles
once and then runs for hours.  It is badly mismatched to Chorus, where
``run_code_in_environment`` spawns a *fresh subprocess per call*: the
benchmark is re-run every time, to serve a single forward pass.

Measured on an H200 with CATv1 at its native 2114 bp geometry, the first
forward costs ~7.6 s against ~0.96 ms steady-state — so a one-shot
prediction spends about 99.99% of its GPU time choosing how to run the
kernel rather than running it.

Triton 3.6+ can cache the *selected config* to disk, keyed on the same
tuple ``autotune`` keys on (shape, dtypes, and the config list itself), so
a later process reuses the winner instead of re-benchmarking.  It is off by
default and has no backing environment variable, so it has to be set from
Python before the decorators are evaluated at ``import cherimoya``.

This is a cache, not a guess: the config used is the one ``autotune``
picked for that exact key, and a shape Chorus has not seen falls back to
benchmarking.  Predictions are bit-identical either way.

The knob is global to the Triton runtime rather than per-kernel, so it also
covers Cherimoya's gradient-path kernels and any other autotuned kernel in
the process.
"""

import logging

logger = logging.getLogger(__name__)


def enable_autotune_cache() -> bool:
    """Turn on Triton's on-disk autotune cache for this process.

    Must be called *before* ``cherimoya`` is imported — ``triton.autotune``
    reads the knob in ``Autotuner.__init__``, which runs at decoration
    time, so setting it afterwards has no effect.

    Returns:
        ``True`` if the knob was set, ``False`` when it is unavailable —
        Triton is not installed (CPU-only and macOS installs), or the
        version predates the knob.  Callers should ignore the result; it
        exists for tests and logging.  A ``False`` return is a missed
        optimization, never a functional difference.
    """
    try:
        from triton import knobs
    except ImportError:
        return False

    autotuning = getattr(knobs, "autotuning", None)
    if autotuning is None or not hasattr(autotuning, "cache"):
        logger.debug(
            "Triton is installed but exposes no autotuning.cache knob; "
            "per-call autotune benchmarking will not be cached."
        )
        return False

    autotuning.cache = True
    return True
