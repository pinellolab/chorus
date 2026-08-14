"""A failed package listing must not be reported as "this environment does not exist".

`get_environment_info()` looks the environment up, then runs a second subprocess purely to fill a
`packages` field. That field has exactly one consumer — a count printed by `chorus list --verbose`
— but its failure used to take down the whole function: `CalledProcessError` fell through to a
handler that returned `None`, and all eleven callers read `None` as "no such environment".

The visible symptom was `get_python_executable()` reporting **"Python executable not found in
environment"** for an environment whose interpreter was sitting right where it belonged. That
message asserts a specific, wrong cause, and it cost three diagnostic attempts pointed inside the
environment instead of at the lookup.

How it was found: a test handed an MCP child the wrong `MAMBA_ROOT_PREFIX`. `mamba env list --json`
still located the env — so `environment_exists()` correctly returned `True` — while
`mamba list -n <name>`, which resolves the *name* against the root, exited 1. Two functions
disagreeing about existence, with only one of them right.

`environment_exists()` was **not** at fault and is unchanged. The fixes are to resolve packages by
`-p <path>` (root-independent, and the path was already in hand) and to keep the failure non-fatal.
"""
from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from chorus.core.environment import EnvironmentManager


@pytest.fixture
def manager():
    return EnvironmentManager()


def _fake_run(env_list_payload, list_behaviour):
    """A `subprocess.run` that answers `env list` from a payload and delegates `list`."""
    def run(cmd, *a, **kw):
        if "env" in cmd and "list" in cmd:
            return subprocess.CompletedProcess(cmd, 0, json.dumps(env_list_payload), "")
        if "list" in cmd:
            return list_behaviour(cmd)
        return subprocess.CompletedProcess(cmd, 0, "{}", "")
    return run


def test_a_package_list_failure_still_yields_the_path(manager, monkeypatch):
    """The regression. Before the fix this returned None and callers concluded "no env"."""
    env_path = "/opt/envs/chorus-enformer"
    payload = {"envs": [env_path]}

    def failing_list(cmd):
        raise subprocess.CalledProcessError(1, cmd, "", "libmamba could not resolve the prefix")

    monkeypatch.setattr(manager, "_env_cache", {})
    with patch("subprocess.run", side_effect=_fake_run(payload, failing_list)):
        info = manager.get_environment_info("enformer")

    assert info is not None, (
        "a failed package listing made get_environment_info return None, which every caller "
        "reads as 'the environment does not exist'"
    )
    assert info["path"] == env_path
    assert info["packages"] == [], "packages should degrade to empty, not vanish with the env"
    assert info["exists"] is True


def test_the_python_executable_survives_it_too(manager, monkeypatch, tmp_path):
    """`get_python_executable` only needs the path; a package hiccup must not hide it."""
    env_dir = tmp_path / "chorus-enformer"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()

    def failing_list(cmd):
        raise subprocess.CalledProcessError(1, cmd, "", "boom")

    monkeypatch.setattr(manager, "_env_cache", {})
    with patch("subprocess.run", side_effect=_fake_run({"envs": [str(env_dir)]}, failing_list)):
        exe = manager.get_python_executable("enformer")

    assert exe == str(env_dir / "bin" / "python")


def test_packages_are_resolved_by_path_not_by_name(manager, monkeypatch):
    """`-n` resolves against the mamba root; `-p` does not. That is the root-sensitivity fix."""
    env_path = "/opt/envs/chorus-enformer"
    seen = {}

    def capture_list(cmd):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "[]", "")

    monkeypatch.setattr(manager, "_env_cache", {})
    with patch("subprocess.run", side_effect=_fake_run({"envs": [env_path]}, capture_list)):
        manager.get_environment_info("enformer")

    assert "-p" in seen["cmd"] and env_path in seen["cmd"], seen.get("cmd")
    assert "-n" not in seen["cmd"], (
        "package resolution went back to `-n <name>`, which fails whenever MAMBA_ROOT_PREFIX "
        "does not contain the env — the exact configuration that produced the misleading error"
    )


def test_a_genuinely_absent_env_still_returns_none(manager, monkeypatch):
    """The other half: don't paper over a real absence."""
    monkeypatch.setattr(manager, "_env_cache", {})
    with patch("subprocess.run", side_effect=_fake_run({"envs": ["/opt/envs/something-else"]},
                                                       lambda cmd: subprocess.CompletedProcess(
                                                           cmd, 0, "[]", ""))):
        assert manager.get_environment_info("enformer") is None


def test_the_unresolvable_interpreter_message_names_the_root(manager, monkeypatch):
    """The message must not assert an interpreter-less env when the lookup is the likelier cause."""
    monkeypatch.setattr(manager, "environment_exists", lambda oracle: True)
    monkeypatch.setattr(manager, "get_python_executable", lambda oracle: None)

    ok, issues = manager.validate_environment("enformer")

    assert not ok
    joined = " ".join(issues)
    assert "MAMBA_ROOT_PREFIX" in joined, (
        f"the message should name the wrong-root possibility; got: {issues}"
    )
    assert "chorus-enformer" in joined, "it should name the env it could not resolve"


def test_environment_exists_is_unchanged_and_still_authoritative(manager):
    """Guards against 'fixing' the function that was right.

    `environment_exists` uses `env list`, which is root-independent for discovery. It was correct
    throughout; the disagreement was entirely on the other side.
    """
    import inspect

    src = inspect.getsource(manager.environment_exists)
    assert "env" in src and "list" in src
    assert "'-n'" not in src and '"-n"' not in src, (
        "environment_exists picked up a name-based resolution, which would make it "
        "root-sensitive like the bug it helped expose"
    )
