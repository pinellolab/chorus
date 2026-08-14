"""Every registered oracle must appear in the registries that actually affect behaviour.

Adding an oracle means hand-editing about ten places. There is no single registry, nothing is
auto-discovered beyond the `chorus-*.yml` filename glob, and a missing entry usually fails *quietly*
— which is the property this file exists to remove.

The concrete case that prompted it: `chorus health` reported **"✓ alphagenome_pt: Healthy"** on a
box where the dependency probe could not have checked anything, because
`EnvironmentRunner`'s `dependencies` dict had no `alphagenome_pt` key and
`dependencies.get(oracle, [])` turns a missing key into "nothing to check". `cherimoya` was missing
from the same dict, and `alphagenome_pt` from `EnvironmentManager.oracle_deps` — two near-duplicate
dicts that had drifted to three different contents.

**Only live registries are checked here.** `ORACLE_CLASS_MAP` is deliberately absent: its sole
consumer computed `class_name` and never used it, so demanding completeness of it would encode an
invariant nothing relies on. It was deleted rather than filled in. If you add a registry, add it
here only if a missing entry changes what chorus *does*.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from chorus.oracles import ORACLES

REPO = Path(__file__).resolve().parent.parent
ORACLE_NAMES = sorted(ORACLES)


def _dict_keys_at(path: str, name: str) -> set[str]:
    """String keys of a module-level (or nested) dict literal called *name*.

    AST-parsed rather than regexed, so a name appearing in a comment or docstring cannot satisfy
    the check — the same reason the report stamper moved off regex.
    """
    tree = ast.parse((REPO / path).read_text())
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        if not isinstance(target, ast.Name) or target.id != name:
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        return {k.value for k in node.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    raise AssertionError(f"no dict literal named {name!r} found in {path} — has it been renamed?")


def _dict_keys_in_generated_script(path: str, script_var: str, dict_name: str) -> set[str]:
    """Same, for a dict written inside a generated-script f-string.

    `EnvironmentRunner` builds its dependency probe as `deps_script = f\"\"\"...\"\"\"` and runs it in
    the oracle's env, so the dict is real code that never executes in *this* process and has no
    AST dict literal here. The f-string's `{{` is already unescaped to `{` by the time it reaches
    `ast.JoinedStr`, so concatenating the literal segments yields parseable Python; interpolations
    are replaced with a placeholder because their values are irrelevant to the key set.
    """
    tree = ast.parse((REPO / path).read_text())
    for node in ast.walk(tree):
        target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None
        if not isinstance(target, ast.Name) or target.id != script_var:
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            body = node.value.value
        elif isinstance(node.value, ast.JoinedStr):
            body = "".join(p.value if isinstance(p, ast.Constant) else "PLACEHOLDER"
                           for p in node.value.values)
        else:
            continue
        inner = ast.parse(body)
        for sub in ast.walk(inner):
            t = sub.targets[0] if isinstance(sub, ast.Assign) and len(sub.targets) == 1 else None
            if isinstance(t, ast.Name) and t.id == dict_name and isinstance(sub.value, ast.Dict):
                return {k.value for k in sub.value.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    raise AssertionError(
        f"no dict named {dict_name!r} inside the {script_var!r} script in {path}")


#: (label, path, dict name, why a missing entry matters)
DICT_REGISTRIES = [
    ("mcp ORACLE_SPECS", "chorus/mcp/server.py", "ORACLE_SPECS",
     "the oracle is invisible to every MCP client"),
    ("weights_probe _ARTIFACT_PROBES", "chorus/core/weights_probe.py", "_ARTIFACT_PROBES",
     "`chorus health` cannot tell 'not installed' from 'unhealthy'"),
    ("manager oracle_deps", "chorus/core/environment/manager.py", "oracle_deps",
     "`validate_environment` checks no dependency and passes"),
]

#: The runner's probe is generated code, so it needs the script-aware reader.
RUNNER_DEPS = ("chorus/core/environment/runner.py", "deps_script", "dependencies")


def _runner_dependencies() -> set[str]:
    return _dict_keys_in_generated_script(*RUNNER_DEPS)


@pytest.mark.parametrize("label,path,dict_name,consequence", DICT_REGISTRIES,
                         ids=[r[0] for r in DICT_REGISTRIES])
def test_dict_registry_covers_every_oracle(label, path, dict_name, consequence):
    missing = sorted(set(ORACLE_NAMES) - _dict_keys_at(path, dict_name))
    assert not missing, (
        f"{label} ({path}) has no entry for {missing}.\n"
        f"Consequence: {consequence}.\n"
        f"Registered oracles: {ORACLE_NAMES}"
    )


def test_create_oracle_handles_every_oracle():
    """`create_oracle` needs a branch per oracle — there is no dispatch table."""
    src = (REPO / "chorus" / "__init__.py").read_text()
    missing = [o for o in ORACLE_NAMES if f"'{o}'" not in src and f'"{o}"' not in src]
    assert not missing, (
        f"chorus/__init__.py's create_oracle has no branch for {missing}; "
        f"create_oracle('{missing[0] if missing else 'x'}') would raise ValueError."
    )


def test_the_valid_names_error_string_lists_every_oracle():
    """The typo message must not silently omit an oracle.

    Separate from the branch check because the string is maintained by hand next to the branches,
    so it is exactly the kind of thing that gets forgotten.
    """
    src = (REPO / "chorus" / "__init__.py").read_text()
    m = re.search(r"valid\s*=\s*\(([^)]*)\)", src, re.S)
    assert m, "could not find the `valid = (...)` names string in create_oracle"
    listed = {n.strip() for n in re.findall(r"[a-z_]+", m.group(1))}
    missing = sorted(set(ORACLE_NAMES) - listed)
    assert not missing, (
        f"create_oracle's valid-names message omits {missing}, so a user who mistypes an oracle "
        f"name is told those do not exist."
    )


def test_the_cleanup_oracle_list_covers_every_oracle():
    """`chorus cleanup --oracle all` iterates a hardcoded list, so an omission is a silent skip.

    Missed by the first version of this file, which is the same class of gap it was written to close:
    the list is a plain module-level list rather than a dict, so the dict-based checks above walked
    straight past it. An oracle absent here survives `cleanup --oracle all` with no error and no
    mention — the user believes they removed everything.
    """
    tree = ast.parse((REPO / "chorus" / "cli" / "_cleanup.py").read_text())
    listed = None
    for node in ast.walk(tree):
        target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None
        if isinstance(target, ast.Name) and target.id == "_ALL_ORACLES" \
                and isinstance(node.value, (ast.List, ast.Tuple)):
            listed = {e.value for e in node.value.elts
                      if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    assert listed is not None, "no _ALL_ORACLES list literal in chorus/cli/_cleanup.py"
    missing = sorted(set(ORACLE_NAMES) - listed)
    assert not missing, (
        f"chorus/cli/_cleanup.py's _ALL_ORACLES omits {missing}, so `chorus cleanup --oracle all` "
        f"silently leaves their env and weights on disk."
    )


def test_every_oracle_has_an_environment_file():
    """The `chorus-*.yml` filename is what `list_available_oracles` globs."""
    missing = [o for o in ORACLE_NAMES
               if not (REPO / "environments" / f"chorus-{o}.yml").is_file()]
    assert not missing, f"no environments/chorus-<name>.yml for {missing}"


def test_the_runner_dependency_probe_covers_every_oracle():
    """`chorus health`'s probe. A missing key means `dependencies.get(oracle, [])` checks nothing.

    Measured symptom before the fix: `chorus health --oracle alphagenome_pt` printed
    "✓ alphagenome_pt: Healthy" while checking zero dependencies.
    """
    missing = sorted(set(ORACLE_NAMES) - _runner_dependencies())
    assert not missing, (
        f"the runner's dependency probe has no entry for {missing}, so `chorus health` reports "
        f"Healthy for them without importing anything."
    )


def test_the_two_dependency_probes_agree():
    """They are near-duplicates; when they disagree, one of the two CLI paths is lying.

    `runner.dependencies` backs `chorus health`; `manager.oracle_deps` backs
    `validate_environment`. Nothing keeps them in step, and they had drifted to different contents.
    """
    runner = _runner_dependencies()
    manager = _dict_keys_at("chorus/core/environment/manager.py", "oracle_deps")
    assert runner == manager, (
        "the two dependency probes cover different oracles:\n"
        f"  only in runner.dependencies:  {sorted(runner - manager)}\n"
        f"  only in manager.oracle_deps:  {sorted(manager - runner)}\n"
        "Whichever is missing an oracle silently checks nothing for it."
    )


def test_the_ast_reader_rejects_a_name_that_only_appears_in_prose():
    """Guards the guard: a regex over the file would pass on a comment mentioning the dict."""
    with pytest.raises(AssertionError, match="no dict literal named"):
        _dict_keys_at("chorus/oracles/__init__.py", "ORACLE_SPECS")
