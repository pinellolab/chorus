"""Downloaded data must not default into ``$HOME``.

Everything large used to: 7.8 GB of per-track backgrounds under
``~/.chorus/backgrounds/`` and — because nothing set ``HF_HOME`` — another 12 GB of
model weights under ``~/.cache/huggingface/``. On a shared machine, or any box with a
home quota, that fills the wrong filesystem. It filled the wrong one here.

The default is now the installation directory, with three ways to redirect it, in
precedence order:

1. ``CHORUS_DATA_DIR`` in the environment
2. ``<install>/chorus_data_dir.txt``, written by ``chorus setup --data-dir`` or
   ``chorus config data-dir --set``
3. the installation directory, or ``~/.chorus`` when the install tree is not writable

Two things deliberately do NOT follow it, and the tests pin both:

* **credentials**, because the point of a data directory is that it can be shared
  between users and a group-readable install tree is the wrong home for a personal
  API token;
* **conda environments**, because a shared data directory must not imply shared
  conda prefixes.

The module is import-time: ``HF_HOME`` has to be set before ``huggingface_hub``
imports its constants, so the resolution cannot be lazy. That makes these tests
subprocess-based — re-importing a module whose side effects already ran proves
nothing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _layout(env_extra: dict | None = None, cwd: Path | None = None) -> dict:
    """Resolve the layout in a FRESH interpreter and return describe_layout()."""
    env = dict(os.environ)
    env.pop("CHORUS_DATA_DIR", None)
    for var in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        env.pop(var, None)
    env.update(env_extra or {})
    code = textwrap.dedent(
        """
        import json, os, sys
        sys.path.insert(0, %r)
        from chorus.core.globals import describe_layout
        out = describe_layout()
        out["_hf_home_env"] = os.environ.get("HF_HOME", "")
        out["_hf_hub_cache_env"] = os.environ.get("HF_HUB_CACHE", "")
        from huggingface_hub import get_token
        out["_token_found"] = bool(get_token())
        print("@@@" + json.dumps(out))
        """
    ) % str(REPO)
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          cwd=str(cwd or REPO), capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = [l for l in proc.stdout.splitlines() if l.startswith("@@@")][-1]
    return json.loads(line[3:])


# ---------------------------------------------------------------------------
# The default
# ---------------------------------------------------------------------------


def test_default_is_the_installation_directory_not_home():
    layout = _layout()
    assert layout["data_dir"] == str(REPO), (
        f"default data dir is {layout['data_dir']}, expected the installation "
        f"directory {REPO}"
    )
    home = str(Path.home())
    for key in ("downloads", "genomes", "annotations", "hf_cache"):
        assert not layout[key].startswith(home + "/."), (
            f"{key} still defaults into a dotdir under $HOME: {layout[key]}"
        )


def test_hf_cache_is_redirected_out_of_home():
    """The 12 GB that nothing used to control."""
    layout = _layout()
    assert layout["_hf_hub_cache_env"], (
        "HF_HUB_CACHE was not set, so model weights land in ~/.cache/huggingface/hub"
    )
    assert layout["_hf_hub_cache_env"].startswith(layout["hf_cache"])
    assert ".cache/huggingface" not in layout["_hf_hub_cache_env"]


def test_redirecting_the_cache_does_not_orphan_the_login_token():
    """The regression: chorus set HF_HOME, which also relocated the credential.

    ``HF_HOME`` is the parent of both the blob store (``hub/``) and the token written
    by ``huggingface-cli login``. Setting it moved both, so on a machine that had
    already logged in the token stayed at ``~/.cache/huggingface/token`` where
    ``huggingface_hub`` no longer looked. Every gated model -- i.e. AlphaGenome --
    then failed with "requires HuggingFace authentication ... run
    'huggingface-cli login'", which the user had already done. Found while running a
    measurement that needed AlphaGenome, not by a test, which is why this exists.

    Only ``HF_HUB_CACHE`` may be set. Skipped where no token is configured, since
    then there is nothing to orphan.
    """
    # Baseline must be a process that never imports chorus -- passing HF_HOME=""
    # does NOT work, because huggingface_hub reads the empty string as a real path
    # and finds no token, which made an earlier version of this test skip itself and
    # assert nothing.
    env = dict(os.environ)
    for var in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        env.pop(var, None)
    probe = subprocess.run(
        [sys.executable, "-c",
         "from huggingface_hub import get_token; print(bool(get_token()))"],
        env=env, capture_output=True, text=True, timeout=300)
    if probe.stdout.strip() != "True":
        pytest.skip("no huggingface token configured on this machine")
    layout = _layout()
    assert layout["_token_found"], (
        "a token was discoverable with a default HF_HOME but not after chorus "
        "redirected the cache -- the credential has been orphaned"
    )
    assert not layout["_hf_home_env"], (
        f"chorus set HF_HOME={layout['_hf_home_env']!r}; it must set HF_HUB_CACHE "
        f"only, so that huggingface-cli login credentials stay discoverable"
    )


def test_a_user_set_hf_home_is_left_alone():
    """Someone who already pointed the HF cache somewhere meant it."""
    layout = _layout({"HF_HOME": "/tmp/my-own-hf"})
    assert layout["_hf_home_env"] == "/tmp/my-own-hf"


# ---------------------------------------------------------------------------
# The three ways to redirect
# ---------------------------------------------------------------------------


def test_env_var_redirects_everything(tmp_path):
    layout = _layout({"CHORUS_DATA_DIR": str(tmp_path)})
    assert layout["data_dir"] == str(tmp_path.resolve())
    for key in ("backgrounds", "downloads", "genomes", "annotations", "hf_cache"):
        assert layout[key].startswith(str(tmp_path.resolve())), key
    assert "environment variable" in layout["source"]


def test_env_var_expands_a_tilde(tmp_path):
    layout = _layout({"CHORUS_DATA_DIR": "~/chorus-data-tilde-test"})
    assert layout["data_dir"] == str((Path.home() / "chorus-data-tilde-test").resolve())
    assert "~" not in layout["data_dir"]


def test_marker_file_redirects_and_env_var_beats_it(tmp_path, monkeypatch):
    """Precedence: env var > marker file > default."""
    from chorus.core.globals import DATA_DIR_MARKER

    marker = REPO / DATA_DIR_MARKER
    existed = marker.exists()
    original = marker.read_text() if existed else None
    marked = tmp_path / "from-marker"
    other = tmp_path / "from-env"
    try:
        marker.write_text(f"{marked}\n")
        assert _layout()["data_dir"] == str(marked.resolve())
        assert "chorus_data_dir.txt" in _layout()["source"]
        # env var must win
        assert _layout({"CHORUS_DATA_DIR": str(other)})["data_dir"] == str(other.resolve())
    finally:
        if existed:
            marker.write_text(original)
        else:
            marker.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The two regressions this module actually had
# ---------------------------------------------------------------------------


def test_legacy_compat_does_not_relocate_the_other_kinds(tmp_path):
    """The regression: a whole-directory legacy fallback dragged three kinds out.

    Backgrounds are the only kind that ever lived under ``~/.chorus``. Annotations,
    downloads and genomes were always in the installation tree. An earlier version
    resolved the ENTIRE data dir to ``~/.chorus`` whenever a legacy background
    existed, which moved those three out of the install dir on any machine that had
    ever downloaded a background — relocating data that was never misplaced.
    """
    layout = _layout()
    home_dotchorus = str((Path.home() / ".chorus").resolve())
    for key in ("downloads", "genomes", "annotations"):
        assert not layout[key].startswith(home_dotchorus), (
            f"{key} resolved to {layout[key]}; legacy compat must apply to "
            f"backgrounds ONLY"
        )


def test_every_backgrounds_default_goes_through_the_global(tmp_path):
    """The other regression: a scripted edit replaced only the FIRST of 8 sites.

    ``normalization.py`` had eight ``Path.home() / ".chorus" / "backgrounds"``
    defaults. A ``str.replace(..., 1)`` fixed one and left seven, so most entry
    points still defaulted into ``$HOME`` while the headline one did not — the worst
    kind of half-fix, because the obvious check passes.
    """
    import re

    offenders = []
    for path in (REPO / "chorus").rglob("*.py"):
        if "_source" in path.parts:
            continue
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            if re.search(r'Path\.home\(\)\s*/\s*"\.chorus"\s*/\s*"backgrounds"', line):
                offenders.append(f"{path.relative_to(REPO)}:{i}")
    assert not offenders, (
        "backgrounds path hardcoded to $HOME instead of CHORUS_BACKGROUNDS_DIR at: "
        + ", ".join(offenders)
    )


# ---------------------------------------------------------------------------
# What must NOT move
# ---------------------------------------------------------------------------


def test_credentials_stay_in_home_even_when_data_moves(tmp_path):
    """A shared data directory is the wrong place for a personal API token."""
    layout = _layout({"CHORUS_DATA_DIR": str(tmp_path)})
    assert layout["config_secrets"].startswith(str(Path.home())), layout["config_secrets"]
    assert not layout["config_secrets"].startswith(str(tmp_path))


def test_conda_environments_stay_with_the_installation(tmp_path):
    """A shared data dir must not imply shared conda prefixes."""
    layout = _layout({"CHORUS_DATA_DIR": str(tmp_path)})
    assert layout["environments"] == str(REPO / "environments")


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argv,needle", [
    (["setup", "--help"], "--data-dir"),
    (["config", "data-dir", "--help"], "--migrate"),
])
def test_cli_exposes_the_switch(argv, needle):
    proc = subprocess.run([sys.executable, "-m", "chorus.cli.main", *argv],
                          cwd=str(REPO), capture_output=True, text=True, timeout=300)
    assert needle in proc.stdout, proc.stdout[-1500:] + proc.stderr[-500:]


def test_config_data_dir_reports_where_things_actually_are():
    proc = subprocess.run([sys.executable, "-m", "chorus.cli.main",
                           "config", "data-dir"],
                          cwd=str(REPO), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-1500:]
    for needle in ("data_dir", "chosen via", "backgrounds", "hf_cache",
                   "CHORUS_DATA_DIR", "credentials"):
        assert needle in proc.stdout, f"{needle!r} missing from the report"
