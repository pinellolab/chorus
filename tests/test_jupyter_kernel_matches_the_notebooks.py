"""The kernel `chorus setup` registers must be the one the shipped notebooks ask for.

15 of the 18 committed notebooks declare ``kernelspec.name == "chorus"``, and until now nothing
created that kernel. On a fresh install every one of them failed:
``jupyter nbconvert --execute`` raises ``NoSuchKernel: No such kernel named chorus``, and JupyterLab
silently prompts for a kernel instead — which reads as a broken notebook rather than a missed step.
The step existed only as item 4 of ``examples/notebooks/README.md``, a file you reach *after* deciding
to open a notebook.

`chorus setup` now registers it. These tests pin the two halves that can drift apart: the name and
display name in :mod:`chorus.cli._jupyter` against the ``.ipynb`` files themselves, and the fact that
a failure to register is a warning rather than a setup failure.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _declared_kernels() -> Counter:
    """(name, display_name) counts across every committed notebook."""
    found: Counter = Counter()
    for pattern in ("examples/notebooks/*.ipynb", "examples/walkthroughs/*/*/notebook.ipynb"):
        for path in sorted(REPO.glob(pattern)):
            try:
                spec = json.loads(path.read_text()).get("metadata", {}).get("kernelspec", {})
            except (ValueError, OSError):
                continue
            found[(spec.get("name"), spec.get("display_name"))] += 1
    return found


def test_the_registered_kernel_is_the_one_the_notebooks_declare():
    """The whole point. If these drift, every notebook declaring `chorus` breaks again."""
    from chorus.cli._jupyter import KERNEL_DISPLAY_NAME, KERNEL_NAME

    declared = _declared_kernels()
    assert declared, "no committed notebooks found — has examples/ moved?"

    wanted = [(n, d) for (n, d) in declared if n == KERNEL_NAME]
    assert wanted, (
        f"no committed notebook declares kernel {KERNEL_NAME!r}; the registered kernel would be "
        f"useless. Declared: {dict(declared)}"
    )
    for _, display in wanted:
        assert display == KERNEL_DISPLAY_NAME, (
            f"notebooks declaring kernel {KERNEL_NAME!r} expect display name {display!r} but "
            f"_jupyter.py registers {KERNEL_DISPLAY_NAME!r}. JupyterLab shows the display name, so a "
            f"mismatch makes the right kernel look like the wrong one."
        )


def test_most_notebooks_depend_on_this_kernel_existing():
    """Documents the stake, and fails if the corpus quietly stops needing it.

    If someone rewrites the notebooks to use `python3`, registering a kernel becomes dead code and
    this test says so rather than leaving it in place forever.
    """
    from chorus.cli._jupyter import KERNEL_NAME

    declared = _declared_kernels()
    total = sum(declared.values())
    needing = sum(n for (name, _), n in declared.items() if name == KERNEL_NAME)
    assert needing >= total // 2, (
        f"only {needing} of {total} notebooks declare kernel {KERNEL_NAME!r}. If the corpus no longer "
        f"depends on it, the registration step in `chorus setup` is dead code."
    )


def test_setup_exposes_an_opt_out():
    """Writing into a user's Jupyter config should be refusable without editing anything.

    Asserted through the real CLI rather than by importing a parser builder: `main.py` constructs its
    parser inline, so `--help` is the only honest source for what the command accepts.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "chorus.cli.main", "setup", "--help"],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"`chorus setup --help` exited {proc.returncode}: {proc.stderr[-400:]}"
    assert "--no-jupyter-kernel" in proc.stdout, (
        "`chorus setup --help` does not offer --no-jupyter-kernel. Registering a kernel writes into "
        "the user's Jupyter config, which should be refusable without editing code."
    )


def test_registration_failure_never_fails_setup():
    """A convenience step must not turn a completed multi-GB install into a non-zero exit."""
    import chorus.cli._jupyter as j

    calls = []
    original = j.register_kernel
    try:
        j.register_kernel = lambda: (False, "simulated: ipykernel missing")
        j.logger.warning = lambda *a, **k: calls.append(a)  # type: ignore[assignment]
        # must return None and must not raise
        assert j.register_kernel_and_report() is None
    finally:
        j.register_kernel = original

    assert calls, "a failed registration logged nothing; the user would not know to run it by hand"


def test_the_manual_command_names_this_interpreter_and_the_right_kernel():
    """The fallback we print has to actually work when pasted."""
    import sys

    from chorus.cli._jupyter import KERNEL_DISPLAY_NAME, KERNEL_NAME, manual_command

    cmd = manual_command()
    assert sys.executable in cmd, (
        f"the printed command uses a bare `python` rather than {sys.executable}; pasted into another "
        f"shell it would register the wrong environment"
    )
    assert f"--name {KERNEL_NAME}" in cmd and KERNEL_DISPLAY_NAME in cmd, cmd
    assert "--user" in cmd, (
        "the command must use --user, not --sys-prefix: a kernel registered into this prefix is "
        "invisible to a JupyterLab started from anywhere else"
    )


@pytest.mark.integration
def test_the_kernel_resolves_the_way_nbconvert_resolves_it():
    """`get_kernel_spec` is the call that raised NoSuchKernel. Integration-marked: needs it installed."""
    jc = pytest.importorskip("jupyter_client.kernelspec")

    from chorus.cli._jupyter import KERNEL_DISPLAY_NAME, KERNEL_NAME

    try:
        spec = jc.get_kernel_spec(KERNEL_NAME)
    except jc.NoSuchKernel:
        pytest.skip(f"kernel {KERNEL_NAME!r} not registered on this host (run `chorus setup`)")

    assert spec.display_name == KERNEL_DISPLAY_NAME
    assert spec.argv and spec.argv[0].endswith("python"), (
        f"kernel {KERNEL_NAME!r} does not launch a python interpreter: argv={spec.argv}"
    )
