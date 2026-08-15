"""Register the chorus environment as a Jupyter kernel.

15 of the 18 shipped notebooks declare ``kernelspec.name == "chorus"``. Nothing created that kernel,
so on a fresh install every one of them failed to run: ``jupyter nbconvert --execute`` raises
``jupyter_client.kernelspec.NoSuchKernel: No such kernel named chorus``, and JupyterLab silently
prompts for a kernel instead — which reads as "the notebook is broken" rather than "a step was
missed". The step existed only as the fourth item in ``examples/notebooks/README.md``, a file a user
reaches *after* deciding to open a notebook.

So ``chorus setup`` does it, and ``--no-jupyter-kernel`` opts out.

Two deliberate choices:

* **Failure never aborts setup.** Registering a kernel is a convenience on top of an install that took
  tens of minutes and tens of GB; if ``ipykernel`` is missing or the Jupyter config directory is not
  writable, that is worth a warning and the exact command to run later, not a non-zero exit.
* **``--user``, never ``--sys-prefix``.** The kernel must be visible to a JupyterLab started from any
  environment, which is how people actually work — ``--sys-prefix`` registers it only for a Jupyter
  running inside this same prefix, so the notebooks would still fail from a system Jupyter.
"""
from __future__ import annotations

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

#: What the shipped notebooks ask for. Changing either value orphans every committed notebook, so
#: `tests/test_jupyter_kernel_matches_the_notebooks.py` pins both to the `.ipynb` files themselves.
KERNEL_NAME = "chorus"
KERNEL_DISPLAY_NAME = "Python 3 (chorus)"

#: The command a user can run by hand if this step is skipped or fails. Kept as one string because it
#: is printed for copy-paste, and formatted with the *running* interpreter so it points at the
#: environment chorus is actually installed in rather than whatever `python` resolves to later.
MANUAL_COMMAND = (
    '{python} -m ipykernel install --user --name {name} --display-name "{display}"'
)


def manual_command() -> str:
    """The copy-pasteable equivalent of what this module does."""
    return MANUAL_COMMAND.format(
        python=sys.executable, name=KERNEL_NAME, display=KERNEL_DISPLAY_NAME
    )


def register_kernel() -> tuple[bool, str]:
    """Register this interpreter as the ``chorus`` Jupyter kernel.

    Returns ``(ok, detail)``. ``ok=False`` is never fatal — see the module docstring.
    """
    try:
        import ipykernel  # noqa: F401
    except ImportError:
        return False, (
            "ipykernel is not installed in this environment, so the notebooks' `chorus` kernel was "
            "not registered. Install it and re-run, or do it by hand:\n    " + manual_command()
        )

    cmd = [
        sys.executable, "-m", "ipykernel", "install",
        "--user",
        "--name", KERNEL_NAME,
        "--display-name", KERNEL_DISPLAY_NAME,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"could not run ipykernel install ({exc}). By hand:\n    {manual_command()}"

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit {proc.returncode}"
        return False, f"ipykernel install failed: {tail}\n    By hand: {manual_command()}"

    # ipykernel prints the destination to stderr; surface it so the user knows what was written.
    where = (proc.stdout or proc.stderr or "").strip().splitlines()
    return True, (where[-1] if where else f"registered kernel {KERNEL_NAME!r}")


def register_kernel_and_report(skip: bool = False) -> None:
    """Register the kernel as part of a setup run, logging rather than raising.

    Never returns a status: no outcome here should change ``chorus setup``'s exit code.
    """
    if skip:
        logger.info(
            "Skipping Jupyter kernel registration (--no-jupyter-kernel). The shipped notebooks "
            "declare kernel %r; without it they will not run. To do it later:\n    %s",
            KERNEL_NAME, manual_command(),
        )
        return

    ok, detail = register_kernel()
    if ok:
        logger.info("✓ Jupyter kernel %r registered (%s)", KERNEL_NAME, detail)
    else:
        logger.warning("Jupyter kernel not registered — %s", detail)
