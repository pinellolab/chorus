"""A non-editable install must ship every data file the code reads.

``setup.py``'s ``package_data`` was hand-maintained and had fallen six files behind
the tree. Each is read unconditionally, with no fallback:

    oracles/cherimoya_source/CATv1-metadata.tsv           catv1_metadata.py
    oracles/cherimoya_source/CATv1-performance-fold0.tsv  catv1_metadata.py
    oracles/chrombpnet_source/chrombpnet_JASPAR_metadata.tsv   chrombpnet.py
    oracles/chrombpnet_source/templates/input_data.json   bpnet.py
    oracles/enformer_source/enformer_human_targets.txt    enformer_metadata.py
    oracles/sei_source/target.names                       weights_probe.py

So ``pip install .`` produced a package that could not load Enformer, Cherimoya or
ChromBPNet track metadata. The README documents the editable install, which is why
this only bit users who deviated — but ``setup.py`` declares a ``console_scripts``
entry point, so a normal install is a supported thing to attempt.

The test deliberately enumerates candidates **from the tree** rather than asserting
a fixed list. A test that hard-codes the six would pass forever while a seventh file
was added, which is exactly how the original list drifted.
"""
from __future__ import annotations

import ast
import fnmatch
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "chorus"
SETUP = REPO / "setup.py"

# Extensions that are data rather than code.
DATA_SUFFIXES = {".tsv", ".txt", ".json", ".bed", ".js", ".names", ".csv", ".yml", ".yaml"}


def _package_data_patterns() -> list[str]:
    """The globs setup.py declares for the ``chorus`` package."""
    tree = ast.parse(SETUP.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "package_data":
                continue
            data = ast.literal_eval(kw.value)
            return list(data.get("chorus", []))
    pytest.fail("could not find package_data in setup.py")


def _data_files_in_tree() -> list[Path]:
    return sorted(
        p for p in PKG.rglob("*")
        if p.is_file()
        and p.suffix in DATA_SUFFIXES
        and "__pycache__" not in p.parts
    )


def _referenced_by_code(path: Path, all_source: str) -> bool:
    """Is this file named anywhere in the package's Python source?

    Matching on basename is deliberately generous: a file referenced by an f-string or
    joined path still shows its name. False positives here only make the test
    stricter, which is the safe direction.
    """
    return path.name in all_source


@pytest.fixture(scope="module")
def all_source() -> str:
    return "\n".join(
        p.read_text(errors="replace") for p in PKG.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def test_every_data_file_the_code_reads_is_declared(all_source: str):
    patterns = _package_data_patterns()
    missing = []
    for path in _data_files_in_tree():
        rel = str(path.relative_to(PKG))
        if not _referenced_by_code(path, all_source):
            continue
        if not any(fnmatch.fnmatch(rel, pat) for pat in patterns):
            missing.append(rel)
    assert not missing, (
        f"{len(missing)} data file(s) are read by shipped code but absent from "
        f"setup.py's package_data, so `pip install .` omits them: {missing}. "
        f"Add a glob covering each."
    )


def test_no_declared_pattern_has_gone_dead():
    """A pattern matching nothing is either a typo or a leftover.

    Not a hard failure on its own, but it means the declaration and the tree have
    drifted, which is the condition that produced the six missing files.
    """
    dead = []
    for pat in _package_data_patterns():
        if not list(PKG.glob(pat)):
            dead.append(pat)
    assert not dead, (
        f"package_data patterns match nothing in the tree: {dead}. Either the files "
        f"moved or the pattern is a typo; a dead pattern hides a real gap."
    )


def test_every_environment_yml_is_shipped():
    """The hand-written list omitted three envs, including cherimoya's."""
    declared = SETUP.read_text()
    on_disk = sorted(p.name for p in (REPO / "environments").glob("*.yml"))
    assert on_disk, "no environment ymls found"
    # Either every name appears literally, or the list is built by glob.
    if 'Path("environments").glob' in declared:
        return
    missing = [n for n in on_disk if n not in declared]
    assert not missing, (
        f"setup.py's data_files omits {missing}, so `chorus setup --oracle <name>` "
        f"has no yml to read from a non-editable install"
    )


def test_the_installed_environments_path_mismatch_is_recorded_not_silent():
    """``data_files`` puts the ymls in ``<prefix>/chorus_environments/`` while
    ``CHORUS_ENVIRONMENTS_DIR`` resolves inside the repo, so nothing reads the
    installed copy.

    That is a real inconsistency, and shipping the files is still the right call —
    but it should be written down where the next person looks, rather than being
    rediscovered. This test fails if the comment explaining it disappears.
    """
    src = SETUP.read_text()
    assert "CHORUS_ENVIRONMENTS_DIR" in src, (
        "setup.py ships environment ymls to a path nothing reads; keep the comment "
        "explaining that, or fix the mismatch and remove this test"
    )
