"""``--model-change`` relaxes a real gate, so it must be hard to misuse.

``verify_rebuilt_backgrounds.py`` refuses a swap when the effect ceiling falls,
and the reasoning is sound *for the rebuild it was written for*: more positions
from the same populations can only raise ``max(union)``, and removing thinning
raises it further, so a drop means positions went missing.

That premise does not survive a **model** change. Averaging CATv1's five
cross-validation folds reduces the variance of every statistic including the
maximum, so the ensemble's tails are legitimately narrower than fold 0's. Gating
on "the ceiling must not fall" would refuse a strictly better estimator.

The wrong fix is to delete the gate. The right one is a second mode that says out
loud which premise it is operating under, and these tests pin the three
properties that keep it from becoming a bypass:

  * a reason is mandatory -- an unexplained relaxation is how a bad build ships;
  * the relaxed gate still has a floor, because variance reduction does not halve
    a ceiling;
  * the pinning comparison is SKIPPED rather than reported, because under a model
    change the committed raw scores come from the old model and ranking them
    against the new null is apples to oranges.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_rebuilt_backgrounds.py"


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *argv],
        capture_output=True, text=True, timeout=300,
    )


def test_the_script_exists_and_advertises_the_flag():
    r = _run("--help")
    assert r.returncode == 0
    assert "--model-change" in r.stdout
    assert "REASON" in r.stdout


def test_an_empty_reason_is_refused():
    """The flag turns a hard gate into a soft one; it must be justified."""
    r = _run("--model-change", "cherimoya=")
    assert r.returncode == 2, r.stdout + r.stderr
    out = r.stdout + r.stderr
    assert "REASON is required" in out
    assert "unexplained relaxation" in out, (
        "the refusal should say WHY a reason is demanded, not just that one is"
    )


def test_a_malformed_spec_is_refused():
    r = _run("--model-change", "cherimoya")
    assert r.returncode == 2
    assert "ORACLE=REASON" in r.stdout + r.stderr


def test_a_reason_with_an_equals_sign_in_it_survives():
    """``partition`` not ``split``, so prose containing '=' is not truncated."""
    src = SCRIPT.read_text()
    assert '.partition("=")' in src, (
        "use partition so a reason like 'ensemble=mean of 5 folds' keeps its tail"
    )
    assert '.split("=")' not in src


def test_the_ceiling_floor_exists_and_is_not_permissive():
    """A relaxed gate is still a gate."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("vrb", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    floor = mod.MODEL_CHANGE_MIN_CEILING_RATIO
    assert 0.0 < floor < 1.0
    assert floor >= 0.25, (
        f"a floor of {floor} would accept a ceiling collapse as 'variance reduction'"
    )


def test_pinning_is_skipped_not_reported_under_a_model_change():
    """Reporting it would be worse than omitting it.

    ``pinning_rate`` ranks the raw scores committed in
    ``examples/**/example_output.json`` against the new null. Under a model change
    those scores came from the OLD model, so the numerator moved too and the
    comparison flatters or damns arbitrarily. It only becomes meaningful once the
    artefacts have been regenerated through the new model.
    """
    src = SCRIPT.read_text()
    # The skip must be reached from the model_change branch, and must explain itself.
    assert "pinning: not compared" in src
    assert "Re-measure after regenerating artefacts" in src
    # ... and pinning_rate must NOT be called on that branch.
    idx = src.index('print(f"  pinning: not compared')
    branch = src[max(0, idx - 900):idx]
    assert "if model_change:" in branch, "the skip is not guarded by model_change"
    after = src[idx:idx + 400]
    assert "got = None" in after, "the skip must leave `got` unset, not fall through"


def test_the_module_docstring_states_which_premise_each_mode_uses():
    """Someone reading only the docstring must not conclude the gate is absolute."""
    src = SCRIPT.read_text()
    doc = src[:src.index('"""', 3)]
    assert "does not hold for a" in doc and "MODEL" in doc
    assert "silent bypass" in doc
