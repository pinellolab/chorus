"""`chorus cleanup --hf-cache` must remove the weights and never the credentials.

Nothing covered `chorus/cli/_cleanup.py` before this file — `grep -rn cleanup_resources tests/`
returned nothing — and it is a command whose whole job is `shutil.rmtree`, with no confirmation
prompt. So the tests here are mostly about what it must *not* delete.

The danger is specific. `chorus/core/globals.py`'s `describe_layout()["hf_cache"]` returns
`os.environ["HF_HOME"]` when that is set, and an HF *home* contains the `token` and `stored_tokens`
files written by `huggingface-cli login`. A cleanup keyed on that value would delete the user's
credential and any non-chorus model cache — and the README recipe added in #187 did exactly that
before this change replaced it. Resolution therefore goes through
`huggingface_hub.constants.HF_HUB_CACHE`, and the parent is never a target.

The other half is that `--all` must *not* quietly acquire this behaviour: on the upgrade path
keeping the cache is the point, and a shared `HF_HOME` makes deleting it actively destructive.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _targets(monkeypatch, hf_home: Path | None = None, hub: Path | None = None):
    """Re-resolve `_hf_cache_targets` under a given HF env."""
    import chorus.cli._cleanup as cleanup

    for var in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(var, raising=False)
    if hf_home is not None:
        monkeypatch.setenv("HF_HOME", str(hf_home))
    if hub is not None:
        monkeypatch.setenv("HF_HUB_CACHE", str(hub))

    import huggingface_hub.constants as hf_constants
    importlib.reload(hf_constants)
    try:
        return cleanup._hf_cache_targets()
    finally:
        importlib.reload(hf_constants)


def test_the_credential_parent_is_never_a_target(tmp_path, monkeypatch):
    """The case that made the old README recipe dangerous."""
    home = tmp_path / "hf_home"
    (home / "hub").mkdir(parents=True)
    (home / "token").write_text("hf_notarealtoken")

    targets, parent = _targets(monkeypatch, hf_home=home)

    assert parent == home
    assert home not in targets, "the HF home itself was targeted — that holds the login token"
    assert not any(t == home / "token" for t in targets)
    assert all(home in t.parents or t == home / "hub" for t in targets), targets


def test_xet_is_only_taken_when_the_cache_is_inside_the_chorus_data_dir(tmp_path, monkeypatch):
    """A shared HF_HOME keeps everything except the hub cache."""
    home = tmp_path / "shared_hf"
    (home / "hub").mkdir(parents=True)
    (home / "xet").mkdir()

    targets, _ = _targets(monkeypatch, hf_home=home)

    assert home / "hub" in targets
    assert home / "xet" not in targets, (
        "xet was removed from a cache outside the chorus data dir — that directory may belong to "
        "another project"
    )


def test_all_does_not_include_the_hf_cache():
    """`--all` must keep meaning 'what chorus put in its own data dir'."""
    import inspect

    import chorus.cli._cleanup as cleanup

    src = inspect.getsource(cleanup.cleanup_resources)
    body = src[src.index("if do_all:"):src.index("if not any(")]
    assert "do_hf_cache = True" not in body, (
        "`--all` now sets do_hf_cache. That makes it destructive for a shared HF_HOME and "
        "expensive on the upgrade path; the flag is opt-in on purpose."
    )


def test_all_says_out_loud_that_it_left_the_cache(capsys, tmp_path, monkeypatch):
    """Silence would read as 'there was nothing to remove'."""
    import chorus.cli._cleanup as cleanup

    hub = tmp_path / "hf" / "hub"
    hub.mkdir(parents=True)
    monkeypatch.setattr(cleanup, "_hf_cache_targets", lambda: ([hub], hub.parent))
    monkeypatch.setattr(cleanup, "_ALL_ORACLES", [])

    class _Args:
        dry_run = True
        oracle = None
        backgrounds = False
        genomes = False
        hf_cache = False
        all = True

    cleanup.cleanup_resources(_Args())
    out = capsys.readouterr().out
    assert "NOT removed" in out and str(hub) in out, out


def test_dry_run_deletes_nothing(tmp_path, monkeypatch):
    import chorus.cli._cleanup as cleanup

    hub = tmp_path / "hf" / "hub"
    hub.mkdir(parents=True)
    (hub / "blob").write_text("weights")
    token = hub.parent / "token"
    token.write_text("hf_notarealtoken")
    monkeypatch.setattr(cleanup, "_hf_cache_targets", lambda: ([hub], hub.parent))
    monkeypatch.setattr(cleanup, "_ALL_ORACLES", [])

    class _Args:
        dry_run = True
        oracle = None
        backgrounds = False
        genomes = False
        hf_cache = True
        all = False

    assert cleanup.cleanup_resources(_Args()) == 0
    assert hub.exists() and (hub / "blob").exists(), "--dry-run deleted something"
    assert token.exists()


def test_hf_cache_removes_the_hub_and_keeps_the_token(tmp_path, monkeypatch):
    """The real deletion path, on a fake tree."""
    import chorus.cli._cleanup as cleanup

    hub = tmp_path / "hf" / "hub"
    hub.mkdir(parents=True)
    (hub / "blob").write_text("weights")
    token = hub.parent / "token"
    token.write_text("hf_notarealtoken")
    monkeypatch.setattr(cleanup, "_hf_cache_targets", lambda: ([hub], hub.parent))
    monkeypatch.setattr(cleanup, "_ALL_ORACLES", [])

    class _Args:
        dry_run = False
        oracle = None
        backgrounds = False
        genomes = False
        hf_cache = True
        all = False

    assert cleanup.cleanup_resources(_Args()) == 0
    assert not hub.exists(), "the hub cache was not removed"
    assert token.exists(), "the login token was deleted — this is the failure mode that matters"


def test_no_flags_lists_hf_cache_as_an_option(capsys):
    import chorus.cli._cleanup as cleanup

    class _Args:
        dry_run = False
        oracle = None
        backgrounds = False
        genomes = False
        hf_cache = False
        all = False

    assert cleanup.cleanup_resources(_Args()) == 1
    out = capsys.readouterr().out
    assert "--hf-cache" in out
    assert "except --hf-cache" in out, "the help must say --all excludes it"
