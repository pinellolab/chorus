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


# ── every git dep in every env file, and the PyTorch backend's alphagenome ────────────────────
#
# The tests above cover one file. The PyTorch backend (environments/chorus-alphagenome_pt.yml)
# had the same defect -- alphagenome-pytorch from git with no ref -- and nothing looked at it.
# Generalised so a git dep added to any env file later is held to the same rule.

_ANY_GIT_DEP = re.compile(r"^\s*-\s*(?P<spec>[A-Za-z0-9_.\-\[\]]+)\s*@\s*git\+(?P<url>\S+)\s*$")
ENV_FILES = sorted([REPO / "environment.yml", *(REPO / "environments").glob("*.yml")])


def _all_git_deps():
    for env_file in ENV_FILES:
        for line in env_file.read_text().splitlines():
            m = _ANY_GIT_DEP.match(line)
            if m:
                url, _, ref = m.group("url").rpartition("@") if "@" in m.group("url") else (m.group("url"), "", "")
                yield env_file.name, m.group("spec"), (ref or None)


def test_the_env_scan_sees_the_known_git_deps():
    found = {(f, spec) for f, spec, _ in _all_git_deps()}
    for expected in [("chorus-alphagenome.yml", "alphagenome"),
                     ("chorus-alphagenome.yml", "alphagenome-research"),
                     ("chorus-alphagenome_pt.yml", "alphagenome-pytorch[scoring]"),
                     ("environment.yml", "coolbox")]:
        assert expected in found, f"the env-file scan no longer sees {expected}; fix the parser"


def test_every_git_dep_in_every_env_file_is_pinned_to_a_full_commit_sha():
    unpinned = [f"{f}: {spec} @ {ref or '(no ref)'}" for f, spec, ref in _all_git_deps()
                if not (ref and re.fullmatch(r"[0-9a-f]{40}", ref))]
    assert not unpinned, (
        "git deps without a full commit sha make every fresh build take that repo's HEAD:\n"
        + "\n".join(unpinned)
    )


def test_the_pytorch_backend_pins_alphagenome_below_2():
    """At the pinned alphagenome-pytorch commit the [scoring] extra leaves alphagenome
    unbounded, so without this line pip resolves alphagenome 2.0.0."""
    text = (REPO / "environments" / "chorus-alphagenome_pt.yml").read_text()
    m = re.search(r"^\s*-\s*alphagenome\s*(==|<|<=|~=)\s*([0-9][^\s#]*)", text, re.M)
    assert m, "chorus-alphagenome_pt.yml must pin or upper-bound alphagenome"
    major = int(m.group(2).split(".")[0])
    assert major < 2 or m.group(1) == "<", f"alphagenome is pinned to {m.group(2)}, which is >= 2"
