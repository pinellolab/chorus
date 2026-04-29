"""Pin the env-setup gating policy in `OracleBase._setup_environment`.

Issue #64 — chorus's env validator at `chorus/core/environment/manager.py`
has a 60 s probe timeout. On cold-NFS lab boxes a `python -c "import jax"`
probe regularly times out even though the env is healthy. The base class
used to silently downgrade `use_environment=True → False` on every
validation failure (including timeouts), then `oracle.load_pretrained_model()`
would crash with `ModuleNotFoundError` because `_load_direct` tried to
import the framework dep in the chorus base env.

The fix distinguishes the two failure modes:

1. **Timeout-only failures**: log a warning, keep `use_environment=True`.
   The actual subprocess invocation has its own per-call timeout and
   will surface a real error if the env is genuinely broken.
2. **Genuine missing-dep failures**: keep the downgrade + raise
   `EnvironmentNotReadyError` from the next user-facing call (preserves
   the v26 P1 #11 invariant — users who passed `use_environment=True`
   never silently fall back to the wrong env).

These tests pin both branches without building a real conda env or
needing a GPU. They run in the default fast suite.
"""
from __future__ import annotations

import pytest

from chorus.core.exceptions import EnvironmentNotReadyError
from chorus.oracles.alphagenome import AlphaGenomeOracle


def _patch_validation(monkeypatch, issues):
    """Stub `EnvironmentManager.environment_exists` (returns True so we
    reach the validation step) and `validate_environment` (returns the
    given issues so we exercise the failure branch). Return value lets
    the construction call reach `_setup_environment`'s validation block."""
    from chorus.core.environment import manager as _mgr

    monkeypatch.setattr(
        _mgr.EnvironmentManager, "environment_exists", lambda self, oracle: True
    )
    monkeypatch.setattr(
        _mgr.EnvironmentManager,
        "validate_environment",
        lambda self, oracle: (False, issues),
    )


def test_validation_timeout_does_not_downgrade(monkeypatch):
    """Timeout-only failures must NOT flip use_environment=True → False
    and must NOT set _env_setup_error. Cold-NFS / slow-import probes are
    not a useful signal of env brokenness."""
    _patch_validation(monkeypatch, ["Timeout while checking dependency jax"])

    oracle = AlphaGenomeOracle(use_environment=True)

    assert oracle.use_environment is True, (
        "Validation timeout should leave use_environment=True so the "
        "user's explicit request is honored. Genuine missing-dep failures "
        "are a separate branch (see test_validation_missing_dep_raises)."
    )
    assert getattr(oracle, "_env_setup_error", None) is None, (
        "_env_setup_error should be None on timeout — only genuine "
        "failures populate it."
    )


def test_validation_missing_dep_raises_on_load(monkeypatch):
    """Genuine missing-dep failures must downgrade use_environment AND
    cause `load_pretrained_model()` to raise `EnvironmentNotReadyError`.
    This pins both the downgrade behavior (v26 P1 #11) and the gap fix
    (each oracle's load_pretrained_model now calls _check_env_ready).
    """
    _patch_validation(
        monkeypatch, ["Missing dependency jax: not importable"]
    )

    oracle = AlphaGenomeOracle(use_environment=True)
    assert oracle.use_environment is False, (
        "Genuine missing-dep failure must downgrade use_environment so "
        "the next user-facing call surfaces the failure clearly."
    )
    assert oracle._env_setup_error is not None
    assert "Missing dependency" in oracle._env_setup_error

    with pytest.raises(EnvironmentNotReadyError):
        oracle.load_pretrained_model()


def test_validation_mixed_issues_treated_as_genuine_failure(monkeypatch):
    """If issues include both a timeout AND a real missing-dep, treat as
    genuine failure (the missing-dep is the real signal). Pin the
    'all-timeouts' check is conjunctive."""
    _patch_validation(
        monkeypatch,
        [
            "Timeout while checking dependency jax",
            "Missing dependency tensorflow: not importable",
        ],
    )

    oracle = AlphaGenomeOracle(use_environment=True)
    assert oracle.use_environment is False
    assert oracle._env_setup_error is not None


def test_use_environment_false_skips_validation_entirely(monkeypatch):
    """When the user passes use_environment=False at construction, the
    validation path should not run — there's nothing to validate. Pin
    that the v26 P1 #11 raise gate (`_user_asked_for_env`) doesn't fire
    in the test/library-internal path."""
    # Make validate_environment explode if it's called — it shouldn't be.
    from chorus.core.environment import manager as _mgr

    def _boom(self, oracle):  # pragma: no cover — should not execute
        raise AssertionError("validate_environment should not be called when use_environment=False")

    monkeypatch.setattr(_mgr.EnvironmentManager, "validate_environment", _boom)

    oracle = AlphaGenomeOracle(use_environment=False)
    assert oracle.use_environment is False
    # _check_env_ready should not raise even with err set, because
    # _user_asked_for_env is False.
    oracle._env_setup_error = "synthetic"
    oracle._check_env_ready()  # must not raise
