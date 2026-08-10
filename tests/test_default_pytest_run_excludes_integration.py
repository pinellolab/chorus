"""`pytest` must mean the same thing for a contributor, for CI, and in the checklist.

It did not. `pytest.ini` set no ``addopts``, so a bare ``pytest tests/`` collected the
integration tests -- which spawn oracle subprocesses and download weights from
HuggingFace/ENCODE, and therefore fail on any machine without the per-oracle conda
environments. CI was green only because it passed two extra flags:

    pytest tests/ --ignore=tests/test_smoke_predict.py -m "not integration" -q

So "the tests pass" had two meanings, and the difference was invisible unless you read
the workflow file. That is the same class of defect as the rest of this release: a
report describing something other than what it did.

Both halves are now fixed at the source rather than at the call site:

  * ``pytest.ini`` sets ``addopts = -m "not integration"``, so the default excludes them.
  * ``tests/test_smoke_predict.py`` is marked ``integration`` (it always was in
    substance) and its fixtures call ``_require_oracle``, so a machine without the envs
    gets skips instead of a wall of fixture ERRORs.

This file is the drift guard: if the ``addopts`` line goes away, if the smoke tests lose
their marker or a fixture loses its guard, or if CI widens its selection, these fail.

One loose end, recorded rather than hidden: the workflow still passes
``-m "not integration"`` and ``--ignore=tests/test_smoke_predict.py``. Both are now
redundant -- the marker filter is in pytest.ini and the smoke tests skip cleanly -- but
deleting them requires pushing ``.github/workflows/tests.yml``, which needs the GitHub
``workflow`` OAuth scope that this checkout's credentials lack. Redundancy is harmless
here; the test below fails only if CI ever *widens* the selection instead.
"""
from __future__ import annotations

import configparser
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PYTEST_INI = REPO / "pytest.ini"
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"


def _addopts() -> str:
    parser = configparser.ConfigParser()
    parser.read(PYTEST_INI)
    assert parser.has_section("pytest"), f"{PYTEST_INI} has no [pytest] section"
    return parser.get("pytest", "addopts", fallback="")


def test_the_default_run_deselects_integration():
    addopts = _addopts()
    assert addopts, (
        "pytest.ini declares no addopts, so a bare `pytest tests/` collects the "
        "integration tests and fails on any machine without the per-oracle conda "
        "environments -- while CI stays green by passing its own marker filter. That "
        "gap is what this file exists to prevent."
    )
    normalised = addopts.replace('"', "").replace("'", "")
    assert "-m not integration" in " ".join(normalised.split()), (
        f"pytest.ini addopts is {addopts!r}, which does not deselect the integration "
        f"marker. Keep `-m \"not integration\"` there, or update CI and "
        f"audits/AUDIT_CHECKLIST.md together and rewrite this test."
    )


def test_integration_is_a_declared_marker():
    """A filter on an undeclared marker is a typo waiting to happen."""
    body = PYTEST_INI.read_text()
    assert re.search(r"^\s*integration:", body, re.MULTILINE), (
        "the `integration` marker is filtered by addopts but not declared under "
        "[pytest] markers, so --strict-markers would reject it"
    )


def test_ci_does_not_widen_the_selection():
    """CI must not run the integration tests, whichever way it says so.

    This deliberately checks equivalence rather than exact flags. The workflow still
    carries a redundant ``-m "not integration"`` and an ``--ignore`` of the smoke file;
    both are now unnecessary -- the exclusion lives in pytest.ini and the smoke tests are
    marked and guarded -- but removing them needs a push with the GitHub ``workflow``
    OAuth scope, which the tooling here does not have. Redundant is harmless; what would
    be harmful is CI *widening* the selection back to something the documented command
    does not run, so that is what fails here.
    """
    if not WORKFLOW.exists():
        pytest.skip("no CI workflow in this checkout")
    body = WORKFLOW.read_text()
    run_lines = [
        line.strip() for line in body.splitlines()
        if "pytest" in line and not line.strip().startswith("#")
    ]
    widened = [
        line for line in run_lines
        if re.search(r'-m\s+(["\']?)integration\1', line)
        or re.search(r'-m\s+(""|\'\')', line)
    ]
    assert not widened, (
        f"the CI workflow selects integration tests: {widened}. CI runs on a 14 GB "
        f"runner with no oracle environments; those tests need ~10 GB of models and "
        f"spawn subprocesses. Keep CI on the default selection."
    )


def test_the_smoke_tests_are_marked_integration():
    """They spawn subprocesses and download weights; that is the marker's definition."""
    src = (REPO / "tests" / "test_smoke_predict.py").read_text()
    assert "pytestmark = pytest.mark.integration" in src, (
        "tests/test_smoke_predict.py is not marked integration, so it runs in the fast "
        "suite and errors on a machine without the oracle environments"
    )


def test_the_smoke_fixtures_skip_rather_than_error():
    """Every oracle fixture must guard its prerequisites.

    An ERROR reads as "chorus is broken"; a SKIP reads as "run chorus setup". The
    difference matters most to whoever is looking at CI for the first time.
    """
    src = (REPO / "tests" / "test_smoke_predict.py").read_text()
    fixtures = re.findall(r"\ndef (\w+_oracle)\(\):\n(.*?)(?=\n@|\ndef |\Z)", src, re.S)
    assert fixtures, "no *_oracle fixtures found -- has the file been restructured?"
    unguarded = [name for name, body in fixtures if "_require_oracle(" not in body]
    assert not unguarded, (
        f"these fixtures call into an oracle without a prerequisite check, so they "
        f"raise instead of skipping when the env is missing: {unguarded}"
    )
