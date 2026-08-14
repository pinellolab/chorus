"""CONTRIBUTING's "which script, which oracle, which env" table must match the scripts.

Adding a worked example is the smallest useful contribution to chorus, and for a long time it was
documented nowhere a contributor would look: the script/oracle/env matrix lived in `CLAUDE.md` (for
Claude sessions) and in an audit report, while `CONTRIBUTING.md` opened by saying it was a guide to
implementing an oracle. Someone who wanted to add a variant walkthrough had to reverse-engineer four
regeneration scripts to find which one owned it.

The matrix is now in CONTRIBUTING as well, which makes it a drift risk in **two** places:
`regenerate_multioracle.py` accepts four oracles and pointedly **not** `enformer`,
`regenerate_examples.py` accepts a different three plus `all`, and only two of the three take
`--gpu`. Every one of those asymmetries has already cost a debugging session, and a table that
silently goes stale costs the next one too.

`CLAUDE.md` keeps its inline copy on purpose — it is auto-loaded into every session, so a pointer
would cost an extra read on exactly the work where getting this wrong is the documented failure mode.
The price of that choice is two copies, so both are pinned to the same `argparse` definitions rather
than to each other: pinning them to each other would let them agree and both be wrong. Verified that
breaking *either* copy fails only that copy's parametrization.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CONTRIBUTING = REPO / "CONTRIBUTING.md"
SCRIPTS = REPO / "scripts"

#: Every document carrying a copy of the regeneration matrix, with the heading that starts its
#: section. There are deliberately two: `CLAUDE.md` is auto-loaded into every Claude Code session, so
#: replacing its inline copy with a pointer would cost an extra file read on work where getting the
#: matrix wrong is the documented failure mode. But a second copy is a second thing that can drift —
#: which is the exact problem moving the matrix into CONTRIBUTING was meant to end — so both are
#: pinned to the same `argparse` definitions rather than to each other.
MATRIX_DOCS = [
    ("CONTRIBUTING.md", "## Contributing an example or walkthrough"),
    ("CLAUDE.md", "**Which script, which oracle, which env.**"),
]


def _arg_choices(script: Path, flag: str) -> list[str] | None:
    """The `choices=[...]` of `flag` in the script's argparse setup, or None if the flag is absent."""
    tree = ast.parse(script.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if getattr(fn, "attr", None) != "add_argument":
            continue
        if not any(isinstance(a, ast.Constant) and a.value == flag for a in node.args):
            continue
        for kw in node.keywords:
            if kw.arg == "choices" and isinstance(kw.value, (ast.List, ast.Tuple)):
                return [e.value for e in kw.value.elts if isinstance(e, ast.Constant)]
        return []  # flag exists but takes no choices
    return None


def _table(doc: str = "CONTRIBUTING.md", start: str = "## Contributing an example or walkthrough") -> str:
    """The section of `doc` that carries the regeneration matrix, plus the prose after it."""
    text = (REPO / doc).read_text()
    i = text.index(start)
    nxt = text.find("\n## ", i + len(start))
    return text[i:nxt if nxt != -1 else len(text)]


@pytest.mark.parametrize("doc,start", MATRIX_DOCS, ids=[d for d, _ in MATRIX_DOCS])
@pytest.mark.parametrize("script,flag", [
    ("regenerate_examples.py", "--oracle"),
    ("regenerate_multioracle.py", "--oracle"),
    ("regenerate_remaining_examples.py", "--only"),
])
def test_every_documented_choice_is_a_real_choice(script: str, flag: str, doc: str, start: str):
    """No value in either copy of the table that argparse would reject."""
    choices = _arg_choices(SCRIPTS / script, flag)
    assert choices is not None, f"{script} no longer defines {flag}; the {doc} table is stale"
    table = _table(doc, start)
    row = next((ln for ln in table.splitlines() if script in ln and ln.startswith("|")), None)
    assert row is not None, f"the {doc} matrix has no table row for {script}"

    # anything rendered as `code` in that row which looks like an oracle name must be valid —
    # except names the row explicitly calls out as *unsupported* ("no `enformer`"), which are
    # covered by test_the_enformer_exception_is_still_true instead
    import re
    positive = re.sub(r"no\s+`[a-z_]+`", "", row)
    cited = {m for m in re.findall(r"`([a-z_]+)`", positive)}
    known_oracles = {"alphagenome", "enformer", "chrombpnet", "cherimoya", "legnet", "sei",
                     "borzoi", "epinformerseq", "alphagenome_pt"}
    for name in cited & known_oracles:
        assert name in choices, (
            f"{doc}'s row for {script} cites `{name}`, but its {flag} choices are {choices}. "
            f"argparse would reject it — an invalid --oracle is an error that scrolls past in a "
            f"tailed log, which is exactly how a regeneration silently does nothing."
        )


@pytest.mark.parametrize("doc,start", MATRIX_DOCS, ids=[d for d, _ in MATRIX_DOCS])
def test_the_enformer_exception_is_still_true(doc: str, start: str):
    """`regenerate_multioracle.py` deliberately has no `enformer`, and the docs say so.

    This is the asymmetry most likely to be "helpfully" corrected by someone who assumes the table
    is a typo. If enformer is ever added, the sentence has to go.
    """
    choices = _arg_choices(SCRIPTS / "regenerate_multioracle.py", "--oracle") or []
    table = _table(doc, start)
    stated = "no `enformer`" in table or "no enformer" in table

    if "enformer" in choices:
        assert not stated, (
            f"regenerate_multioracle.py now accepts --oracle enformer, so {doc}'s "
            f"'no enformer' note is wrong and must be removed"
        )
    else:
        assert stated, (
            f"regenerate_multioracle.py rejects --oracle enformer and {doc} no longer says so; "
            f"a contributor will try it and read the argparse error as a broken script"
        )


@pytest.mark.parametrize("doc,start", MATRIX_DOCS, ids=[d for d, _ in MATRIX_DOCS])
def test_the_gpu_flag_asymmetry_is_documented_correctly(doc: str, start: str):
    """Two of the three regeneration scripts take `--gpu`; the multioracle one does not."""
    has_gpu = {
        s: _arg_choices(SCRIPTS / s, "--gpu") is not None
        for s in ("regenerate_examples.py", "regenerate_multioracle.py",
                  "regenerate_remaining_examples.py")
    }
    table = _table(doc, start)
    if not has_gpu["regenerate_multioracle.py"]:
        assert "no `--gpu` flag" in table, (
            f"regenerate_multioracle.py still has no --gpu, but {doc} does not warn about it; "
            f"passing one is an argparse error. (--gpu presence: {has_gpu})"
        )
    else:
        pytest.fail(
            f"regenerate_multioracle.py gained a --gpu flag — remove the warning from {doc}"
        )


def test_the_notebook_generator_is_still_codegen_only():
    """Documented as taking no arguments and needing no GPU."""
    script = SCRIPTS / "generate_walkthrough_notebooks.py"
    tree = ast.parse(script.read_text())
    adds = [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "add_argument"]
    assert not adds, (
        f"generate_walkthrough_notebooks.py now takes {len(adds)} argument(s); CONTRIBUTING "
        f"describes it as codegen-only with no flags"
    )


def test_the_guard_files_named_in_contributing_exist():
    """The section tells a contributor which tests to run. All of them must be real files."""
    import re

    table = _table()
    named = re.findall(r"tests/(test_[a-z0-9_]+)\.py", table)
    assert named, "the example section no longer names any guard tests to run"
    missing = [n for n in named if not (REPO / "tests" / f"{n}.py").is_file()]
    assert not missing, (
        f"CONTRIBUTING tells contributors to run tests that do not exist: {missing}. A documented "
        f"command that errors is worse than no command."
    )


def test_the_routing_table_covers_the_non_oracle_paths():
    """The doc used to open by saying it was about implementing an oracle, and nothing else.

    A contributor with an example to add read sentence one and concluded they were in the wrong
    place. The routing table exists so that does not happen.
    """
    head = CONTRIBUTING.read_text()[:2000].lower()
    for want, why in (
        ("example", "an example contributor must see themselves in the first screen"),
        ("bug fix", "small fixes are wanted and were never mentioned"),
        ("oracle", "the oracle path is still the main one and must stay findable"),
    ):
        assert want in head, f"CONTRIBUTING's opening does not mention {want!r} — {why}"
