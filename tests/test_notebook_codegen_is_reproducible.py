"""Re-running the walkthrough notebook codegen must produce no diff.

`nbformat.v4.new_code_cell` mints a **random** `id` per call, so every run of
`scripts/generate_walkthrough_notebooks.py` rewrote all 13 committed notebooks with fresh ids and no
other change — measured 121 insertions and 121 deletions, of which **zero** lines were anything but
`"id"`.

The cost was not the churn, it was that "have the committed notebooks drifted from their generator?"
became unanswerable. `git status` was dirty after every run whether anything had changed or not, so
the meaningful check and the meaningless one looked identical — and a real drift would have hidden in
noise nobody reads.

Cell ids are now derived from `(index, cell_type, source)`, so they are stable across runs and change
exactly when a cell's content does. This test is the thing that keeps that true: it regenerates into
a temporary tree and compares against what is committed.

Integration-marked because it executes the generator over every walkthrough spec.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "generate_walkthrough_notebooks.py"


def test_cell_ids_are_derived_from_content_not_random():
    """The unit-level property, cheap enough for the fast suite."""
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import generate_walkthrough_notebooks as gen
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"generator not importable: {exc}")

    assert hasattr(gen, "_stabilise_cell_ids"), (
        "the generator no longer stabilises cell ids; nbformat will mint random ones and every "
        "regeneration will rewrite all 13 notebooks with no real change"
    )

    import nbformat as nbf

    def build():
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell("# same"), nbf.v4.new_code_cell("print(1)"),
                    nbf.v4.new_code_cell("print(1)")]  # deliberately duplicated content
        gen._stabilise_cell_ids(nb)
        return [c["id"] for c in nb.cells]

    first, second = build(), build()
    assert first == second, f"ids are not reproducible across builds: {first} vs {second}"
    assert len(set(first)) == len(first), (
        f"two cells received the same id ({first}); nbformat requires them unique within a notebook, "
        f"which is why the index is part of the hash"
    )
    for cid in first:
        assert 1 <= len(cid) <= 64 and cid.replace("-", "").replace("_", "").isalnum(), cid


def test_changing_a_cell_changes_only_that_cells_id():
    """The property that makes a diff readable: ids track content."""
    sys.path.insert(0, str(REPO / "scripts"))
    import generate_walkthrough_notebooks as gen
    import nbformat as nbf

    def ids(second_source: str):
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_code_cell("print(1)"), nbf.v4.new_code_cell(second_source),
                    nbf.v4.new_code_cell("print(3)")]
        gen._stabilise_cell_ids(nb)
        return [c["id"] for c in nb.cells]

    before, after = ids("print(2)"), ids("print(22)")
    assert before[0] == after[0] and before[2] == after[2], (
        f"editing one cell changed its neighbours' ids, so a one-cell change still produces a "
        f"whole-file diff: {before} -> {after}"
    )
    assert before[1] != after[1], "the edited cell kept its id, so content drift is invisible"


@pytest.mark.integration
def test_regenerating_leaves_the_committed_notebooks_unchanged():
    """The real guarantee: run the generator, expect no diff.

    Skips rather than fails on a dirty tree — a developer mid-edit should not see this as breakage.
    """
    dirty = subprocess.run(["git", "status", "--porcelain", "examples/"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        pytest.skip(f"examples/ has uncommitted changes; cannot compare:\n{dirty[:300]}")

    proc = subprocess.run([sys.executable, str(SCRIPT)], cwd=REPO,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"codegen failed: {proc.stderr[-500:]}"

    after = subprocess.run(["git", "status", "--porcelain", "examples/"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    if after:
        # leave the tree as we found it before reporting
        subprocess.run(["git", "checkout", "--", "examples/"], cwd=REPO, capture_output=True)
    assert not after, (
        "regenerating the walkthrough notebooks produced a diff, so the committed ones are stale or "
        "the codegen is not reproducible:\n" + after[:800]
    )


@pytest.mark.integration
def test_every_committed_notebook_has_stable_looking_ids():
    """A committed notebook carrying nbformat's random uuid4 ids predates the fix or bypassed it."""
    import re

    suspicious = []
    for path in sorted(REPO.glob("examples/walkthroughs/*/*/notebook.ipynb")):
        try:
            nb = json.loads(path.read_text())
        except ValueError:
            continue
        for cell in nb.get("cells", []):
            cid = cell.get("id", "")
            # nbformat's default is a uuid4 prefix: 8 hex chars. Ours is 12 hex chars.
            if cid and not re.fullmatch(r"[0-9a-f]{12}", cid):
                suspicious.append(f"{path.relative_to(REPO)}: id={cid!r}")
                break
    assert not suspicious, (
        "these committed notebooks have ids that do not look content-derived, so regenerating them "
        "will churn:\n  " + "\n  ".join(suspicious[:10])
    )
