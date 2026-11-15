"""The AlphaGenome env must install alphagenome and alphagenome-research at fixed commits.

Both are installed from git, and until 2026-11-15 neither carried a ref, so a fresh
``chorus-alphagenome`` build took whatever each repo's HEAD was that day. That broke on
2026-11-11, when alphagenome 2.0.0 removed the legacy ``alphagenome.models.*`` import paths
chorus (and the research build the backgrounds were made with) import from: a fresh env
loaded the model and then failed at ``predict()`` with
``No module named 'alphagenome.models.dna_output'`` (PR #249).

The two packages also have to move together. research <= 1e55dcf imports the legacy paths;
research HEAD imports ``alphagenome.models.v1``, which only exists from alphagenome 0.10 —
while still reporting ``__version__ = '0.3.0'``, so a version number cannot express the pin.
A full commit sha can.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENV_FILE = REPO / "environments" / "chorus-alphagenome.yml"

_GIT_DEP = re.compile(
    r"^\s*-\s*(?P<name>alphagenome(?:-research)?)\s*@\s*git\+(?P<url>\S+?)(?:@(?P<ref>\S+))?\s*$"
)


def _git_deps() -> dict[str, str | None]:
    deps: dict[str, str | None] = {}
    for line in ENV_FILE.read_text().splitlines():
        m = _GIT_DEP.match(line)
        if m:
            deps[m.group("name")] = m.group("ref")
    return deps


def test_both_alphagenome_git_deps_are_present():
    assert set(_git_deps()) == {"alphagenome", "alphagenome-research"}, (
        f"expected alphagenome and alphagenome-research as git deps in {ENV_FILE.name}; "
        f"found {sorted(_git_deps())}. If one moved to PyPI, update this test to check its "
        f"version pin instead."
    )


def test_each_alphagenome_git_dep_is_pinned_to_a_full_commit_sha():
    for name, ref in _git_deps().items():
        assert ref is not None, (
            f"{name} is installed from git without a ref, so every fresh build takes that "
            f"repo's HEAD. alphagenome 2.0.0 broke chorus exactly this way (PR #249)."
        )
        assert re.fullmatch(r"[0-9a-f]{40}", ref), (
            f"{name} is pinned to {ref!r}. Use a full 40-character commit sha: a branch moves, "
            f"and alphagenome-research's tag/version string does not identify its import layout."
        )
