"""Pin the one flag that makes AlphaGenome reproducible across processes.

#127: two identical chorus runs differed on 454 numeric fields with 36 sign
flips, and for CAGE the run-to-run noise (0.0054) *exceeded* the median effect
reported (0.0058) — so 92.1% of shipped CAGE rows were ranking noise.

Measured on this box (AlphaGenome, identical 1,048,576 bp sequence at SORT1, all
9 output types, 8xH100):

* within one process, no flags -> already bit-exact, ``0.000e+00``. The model is
  not the problem, and there is no inference RNG.
* two processes, same GPU, no flags -> **0/9 bit-exact**, worst relative 1.6e-2.
  Two processes on *different* GPUs differ by the same amount, so the cause is
  per-process compilation, not the device.
* two processes, ``--xla_gpu_deterministic_ops=true`` -> **9/9 bit-exact**,
  steady-state pass 0.6s -> 1.2s.
* two processes, ``--xla_gpu_autotune_level=0`` -> 9/9 bit-exact but 108s per
  pass, **180x** slower. Deliberately not used.

These tests are cheap and hermetic: they assert the flag is pinned and that the
subprocess env carries it. The actual bit-exactness check needs a GPU and lives
in ``scripts/probe_alphagenome_determinism.py``.
"""
from __future__ import annotations

import os
import sys
from unittest import mock

import pytest

from chorus.core.determinism import (
    DETERMINISTIC_XLA_FLAGS,
    determinism_provenance,
    pin_deterministic_xla_flags,
)


def test_the_pinned_flag_is_deterministic_ops_not_autotune():
    """Guard the choice, because the alternative costs 180x for the same result."""
    assert DETERMINISTIC_XLA_FLAGS == "--xla_gpu_deterministic_ops=true"
    assert "autotune" not in DETERMINISTIC_XLA_FLAGS


def test_pin_sets_the_flag_when_unset():
    with mock.patch.dict(os.environ, {}, clear=True):
        result = pin_deterministic_xla_flags()
        assert result == DETERMINISTIC_XLA_FLAGS
        assert os.environ["XLA_FLAGS"] == DETERMINISTIC_XLA_FLAGS


def test_pin_preserves_flags_the_caller_already_set():
    with mock.patch.dict(os.environ, {"XLA_FLAGS": "--xla_dump_to=/tmp/x"}, clear=True):
        result = pin_deterministic_xla_flags()
        assert "--xla_dump_to=/tmp/x" in result
        assert DETERMINISTIC_XLA_FLAGS in result


def test_pin_is_idempotent():
    with mock.patch.dict(os.environ, {}, clear=True):
        pin_deterministic_xla_flags()
        once = os.environ["XLA_FLAGS"]
        pin_deterministic_xla_flags()
        assert os.environ["XLA_FLAGS"] == once
        assert once.count("xla_gpu_deterministic_ops") == 1


def test_opt_out_is_respected_and_loud():
    with mock.patch.dict(os.environ, {"CHORUS_NONDETERMINISTIC": "1"}, clear=True):
        assert pin_deterministic_xla_flags() == ""
        assert "XLA_FLAGS" not in os.environ


def test_opt_out_can_be_overridden_by_force():
    with mock.patch.dict(os.environ, {"CHORUS_NONDETERMINISTIC": "1"}, clear=True):
        assert DETERMINISTIC_XLA_FLAGS in pin_deterministic_xla_flags(force=True)


def test_warns_if_jax_already_imported(caplog):
    """XLA has read its config by then, so the flag is a no-op — say so."""
    with mock.patch.dict(os.environ, {}, clear=True), \
            mock.patch.dict(sys.modules, {"jax": mock.MagicMock()}):
        with caplog.at_level("WARNING"):
            pin_deterministic_xla_flags()
        assert "already imported" in caplog.text


def test_provenance_reports_what_is_actually_set():
    with mock.patch.dict(os.environ, {}, clear=True):
        pin_deterministic_xla_flags()
        prov = determinism_provenance()
        assert prov["deterministic"] is True
        assert "xla_gpu_deterministic_ops" in str(prov["xla_flags"])
    with mock.patch.dict(os.environ, {"XLA_FLAGS": "--xla_dump_to=/tmp/x"}, clear=True):
        assert determinism_provenance()["deterministic"] is False


def test_subprocess_env_carries_the_flag():
    """The subprocess path is the DEFAULT (``use_environment=True``).

    A child process cannot be fixed after it has imported jax, so the flag must
    be in the env handed to it. Injected before ``_prepare_env``'s early return
    so an oracle with no registered environment still gets it.
    """
    from chorus.core.environment.runner import EnvironmentRunner

    runner = EnvironmentRunner.__new__(EnvironmentRunner)
    runner.env_manager = mock.MagicMock()
    runner.env_manager.get_environment_info.return_value = None  # early-return path

    with mock.patch.dict(os.environ, {}, clear=True):
        env = runner._prepare_env("alphagenome")
    assert "xla_gpu_deterministic_ops" in env["XLA_FLAGS"]


def test_subprocess_env_respects_opt_out():
    from chorus.core.environment.runner import EnvironmentRunner

    runner = EnvironmentRunner.__new__(EnvironmentRunner)
    runner.env_manager = mock.MagicMock()
    runner.env_manager.get_environment_info.return_value = None

    with mock.patch.dict(os.environ, {"CHORUS_NONDETERMINISTIC": "1"}, clear=True):
        env = runner._prepare_env("alphagenome")
    assert "XLA_FLAGS" not in env


@pytest.mark.parametrize("source", ["chorus/oracles/alphagenome.py"])
def test_oracle_pins_before_importing_jax(source):
    """Order matters: after ``import jax`` the flag does nothing.

    A source-order assertion rather than a behavioural one because the direct
    load path needs a GPU and gated weights. Same enforcement style as
    ``tests/test_cherimoya.py:609``.
    """
    from pathlib import Path

    text = Path(source).read_text()
    pin = text.index("pin_deterministic_xla_flags()")
    jax_import = text.index("            import jax")
    assert pin < jax_import, "must pin XLA_FLAGS before importing jax"
