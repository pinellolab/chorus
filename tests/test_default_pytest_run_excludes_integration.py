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

CI now runs the bare command too -- no marker filter, no ``--ignore`` -- so the three
statements of what "the tests" means (pytest.ini, the workflow, the checklist) cannot
disagree. The tests below fail if a filter or an ignore comes back, or if CI widens its
selection.
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


def test_ci_runs_the_same_command_a_contributor_runs():
    """No marker filter and no --ignore in the workflow.

    Neither is wrong in itself; both are wrong because they can drift from pytest.ini
    silently, which is exactly what happened -- CI green, documented command red, and the
    difference visible only to someone reading the workflow file. Also fails if CI
    *widens* the selection to something the documented command does not run.
    """
    if not WORKFLOW.exists():
        pytest.skip("no CI workflow in this checkout")
    body = WORKFLOW.read_text()
    # Lines that INVOKE pytest, not ones that merely mention it: `pip install pytest` and
    # `pytest --version` are not test runs and were tripping this check.
    run_lines = [
        line.strip() for line in body.splitlines()
        if re.match(r"pytest\s+(?!--version)", line.strip())
    ]

    # The property is about the run that stands in for "the tests pass": a whole-suite
    # invocation must not carry its own selection, or CI and a bare `pytest` can disagree
    # silently -- which is what happened, CI green and the documented command red.
    #
    # A job that names ONE test file and marks itself integration is a different claim and
    # is allowed: the browser-smoke job renders a reduced report set on every PR because the
    # full 19 belong to the release host, and running none of it in CI is how a blank-panel
    # regression reaches main between audits (2026-08-12 audit; see checklist section 7).
    # Narrow, deliberate, and recorded here as that message asked.
    whole_suite = [l for l in run_lines if re.search(r"pytest\s+(tests/\s|tests/\s*$|tests/ )", l + " ")]
    filtered = [l for l in whole_suite if "-m " in l or "--ignore" in l]
    assert not filtered, (
        f"a whole-suite CI run passes its own marker filter or --ignore: {filtered}. The "
        f"exclusion lives in pytest.ini so that CI, the docs and a bare `pytest` cannot "
        f"disagree; if CI genuinely needs a different selection, say so in the checklist "
        f"and here."
    )
    assert whole_suite, (
        "no whole-suite `pytest tests/` invocation left in the workflow -- if the fast job "
        "was renamed or removed, this guard is now checking nothing"
    )

    # And a single-file job must still declare which marker it is opting into, so nobody
    # can widen it back to the whole suite by deleting a path.
    for line in run_lines:
        if line in whole_suite:
            continue
        assert "-m " in line and ".py" in line, (
            f"CI runs pytest without either a whole-suite path or an explicit marker+file: "
            f"{line!r}"
        )

    ignored = [
        line.strip() for line in body.splitlines()
        if "--ignore" in line and not line.strip().startswith("#")
    ]
    assert not ignored, (
        f"CI ignores test files: {ignored}. --ignore was needed only because the smoke "
        f"fixtures raised instead of skipping without the oracle envs; they are marked and "
        f"guarded now, so an --ignore here hides whatever actually broke."
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


#: A fresh clone with no genomes, backgrounds or oracle envs collected 125 on CI on 2026-08-14.
#: The floor sits below that: it exists to catch "the marker selected nothing", not to track the
#: exact number, which is not a property of the repo at all.
_INTEGRATION_FLOOR = 100


def test_the_integration_marker_still_selects_a_real_suite():
    """`pytest -m integration` must select a suite, and pytest.ini must not claim an exact size.

    The usage comment used to name a figure, and it drifted four times: 66, 153, 154, 157. The first
    version of this guard pinned that figure by collection -- and failed on CI, which collects **125**
    where this host collects **162**. That is not drift; the count is genuinely environment-dependent,
    because integration tests parametrize over locally-present artefacts (genomes, backgrounds,
    per-oracle envs). An exact number was never a fact about the repository, so no guard could have
    made it one, and the honest fix was to stop asserting it.

    What remains worth checking is what a contributor actually uses the figure for: confirming their
    run selected the suite rather than silently collecting nothing -- the failure mode that shipped in
    CONTRIBUTING's browser recipe, which collected 0 because `addopts` deselected everything it named.
    """
    import subprocess
    import sys

    text = PYTEST_INI.read_text()
    assert "pytest -m integration" in text, "pytest.ini no longer documents the integration opt-in"
    assert re.search(r"fresh clone|provisioned host|magnitude, not an identity", text), (
        "pytest.ini states an integration-test count without saying it varies by environment. CI "
        "collects 125 where a provisioned host collects 162; a bare number reads as exact, goes "
        "stale, and has already done so four times."
    )

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(REPO / "tests"), "-m", "integration",
         "--collect-only", "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=REPO, timeout=900,
    )
    got = re.search(r"(\d+)(?:/\d+)? tests collected", proc.stdout)
    assert got, f"could not read a collection count from pytest output:\n{proc.stdout[-1500:]}"
    actual = int(got.group(1))

    assert actual >= _INTEGRATION_FLOOR, (
        f"`pytest -m integration` collected only {actual} tests, below the floor of "
        f"{_INTEGRATION_FLOOR}. Either the marker stopped being applied or a collection error is "
        f"swallowing most of the suite -- a run that quietly selects almost nothing reads as a pass."
    )
