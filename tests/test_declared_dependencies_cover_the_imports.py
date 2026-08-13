"""Every module-level third-party import in `chorus/` must be a declared dependency.

`setup.py` builds `install_requires` from **`requirements.txt`**, not from `environment.yml`. So a
package can be present in the conda env everyone develops in, imported at module level, and still
missing from the wheel — and nothing notices until someone does a clean `pip install chorus`.

Two live instances, both found on 2026-08-13:

* `chorus/core/environment/manager.py:10` imports `yaml`. #187 added `pyyaml` to
  `environment.yml` and stopped there, so the pip path was still broken — the fix looked complete
  because the conda env it was tested in already had it.
* `chorus/utils/annotations.py:21` imports `requests`, which had never been declared anywhere. It
  works only because `huggingface_hub` happens to pull it in transitively, which is not a
  guarantee anyone wrote down.

`chorus/oracles/` is exempt: those modules deliberately import their heavy per-env dependencies
(torch, tensorflow, jax) inside methods, and the few module-level ones belong to a per-oracle conda
env rather than to the base wheel. Function-level imports anywhere are exempt for the same reason —
this checks only what executes on `import chorus`.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

#: Import name -> distribution name, where they differ.
IMPORT_TO_DISTRIBUTION = {
    "yaml": "pyyaml",
    "Bio": "biopython",
    "PIL": "pillow",
    "bs4": "beautifulsoup4",
    "sklearn": "scikit_learn",
    "cv2": "opencv_python",
    "dateutil": "python_dateutil",
}


def _declared() -> set[str]:
    """Distribution names from requirements.txt, plus setup.py's hardcoded extras."""
    out = set()
    for line in (REPO / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(re.split(r"[<>=!\[;]", line)[0].strip().lower().replace("-", "_"))
    # setup.py: install_requires=requirements + ["click>=8.0"]
    setup = (REPO / "setup.py").read_text()
    for extra in re.findall(r"install_requires\s*=\s*requirements\s*\+\s*\[([^\]]*)\]", setup):
        for q in re.findall(r"[\"']([^\"'<>=!\[]+)", extra):
            out.add(q.strip().lower().replace("-", "_"))
    return out


def _module_level_imports(path: Path) -> set[str]:
    """Top-level import names only — not imports nested in functions or `try` bodies."""
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, OSError):
        return set()
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _checked_modules() -> list[Path]:
    return [p for p in sorted((REPO / "chorus").rglob("*.py"))
            if "oracles" not in p.relative_to(REPO).parts]


def _undeclared(path: Path, declared: set[str]) -> list[str]:
    out = []
    for name in sorted(_module_level_imports(path)):
        if name in sys.stdlib_module_names or name == "chorus" or name.startswith("_"):
            continue
        dist = IMPORT_TO_DISTRIBUTION.get(name, name).lower().replace("-", "_")
        if dist not in declared:
            out.append(f"{name} (would need '{dist}' in requirements.txt)")
    return out


@pytest.mark.parametrize("path", _checked_modules(),
                         ids=lambda p: str(p.relative_to(REPO)))
def test_module_level_imports_are_declared(path: Path):
    offenders = _undeclared(path, _declared())
    assert not offenders, (
        f"{path.relative_to(REPO)} imports these at module level but they are not in "
        f"requirements.txt, which is what setup.py's install_requires reads:\n  "
        + "\n  ".join(offenders)
        + "\nA clean `pip install chorus` would fail on `import chorus`. Declaring it in "
          "environment.yml is not enough — that file does not reach the wheel."
    )


def test_the_two_real_gaps_are_now_declared():
    """Named explicitly, so a future tidy-up of requirements.txt cannot quietly drop them."""
    declared = _declared()
    for dist, why in (("pyyaml", "chorus/core/environment/manager.py imports yaml"),
                      ("requests", "chorus/utils/annotations.py imports requests")):
        assert dist in declared, f"{dist} was dropped from requirements.txt — {why}"


def test_the_guard_would_have_caught_the_shipped_gaps(tmp_path):
    """Fails-without-fix: the exact import lines that were undeclared."""
    for name in ("yaml", "requests"):
        mod = tmp_path / f"uses_{name}.py"
        mod.write_text(f"import os\nimport {name}\n\nx = 1\n")
        assert _undeclared(mod, {"numpy", "pandas"}), \
            f"guard no longer flags an undeclared module-level `import {name}`"


def test_function_level_imports_are_not_flagged(tmp_path):
    """The other half — lazy imports are the documented pattern for per-env deps."""
    mod = tmp_path / "lazy.py"
    mod.write_text("def load():\n    import torch\n    return torch\n")
    assert not _undeclared(mod, set()), "a function-level import was flagged"
