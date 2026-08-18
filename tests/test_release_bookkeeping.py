"""A tagged version must describe itself, and the version must be stated once.

Two things drifted silently before 0.7.0, both found while preparing these tags:

* **Seven releases had no changelog.** v0.5.0 through v0.5.6 were tagged *and* published
  as GitHub Releases, while ``CHANGELOG.md`` stopped at ``0.4.0``. The notes existed only
  in the Releases UI, so the file in the repo was quietly incomplete for three months.
* **66 commits sat on ``main`` with no tag at all**, between v0.5.6 and 2026-08-05 —
  including a change that moved every effect percentile. There was no name for the state
  users actually had.

Neither is exotic; both are what happens when the bookkeeping is a habit rather than a
check. So it is a check now.

The version also lives in two files (``setup.py`` and ``chorus/__init__.py``) with nothing
tying them together, which is a bump waiting to be half-applied.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CHANGELOG = REPO / "CHANGELOG.md"
SETUP = REPO / "setup.py"
INIT = REPO / "chorus" / "__init__.py"


def _changelog_versions() -> list[str]:
    """Released versions, newest first, excluding [Unreleased]."""
    return re.findall(r"^## \[([0-9][^\]]*)\]", CHANGELOG.read_text(), re.M)


def _setup_version() -> str:
    m = re.search(r'version\s*=\s*"([^"]+)"', SETUP.read_text())
    assert m, "no version= in setup.py"
    return m.group(1)


def _init_version() -> str:
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', INIT.read_text(), re.M)
    assert m, "no __version__ in chorus/__init__.py"
    return m.group(1)


def _git_tags() -> list[str]:
    out = subprocess.run(
        ["git", "tag", "-l"], cwd=REPO, capture_output=True, text=True,
    )
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    return [t for t in out.stdout.split() if re.fullmatch(r"v\d+\.\d+\.\d+", t)]


def test_the_version_is_the_same_in_both_places():
    assert _setup_version() == _init_version(), (
        f"setup.py says {_setup_version()} and chorus/__init__.py says {_init_version()}. "
        f"A half-applied bump ships a package whose metadata and runtime disagree."
    )


def test_the_declared_version_has_a_changelog_section():
    version = _init_version()
    versions = _changelog_versions()
    assert version in versions, (
        f"chorus.__version__ is {version} but CHANGELOG.md has no [{version}] section "
        f"(newest is {versions[0] if versions else 'none'}). Either the bump came without "
        f"an entry, or the entry was written under the wrong number."
    )


def test_the_declared_version_is_the_newest_changelog_section():
    """Guards the ordering, which is how a section ends up stranded mid-file."""
    version = _init_version()
    versions = _changelog_versions()
    assert versions and versions[0] == version, (
        f"the newest CHANGELOG section is [{versions[0] if versions else 'none'}] but the "
        f"declared version is {version}; sections are newest-first"
    )


def test_every_tag_has_a_changelog_section():
    """The check that would have caught the 0.5.x gap the day it opened."""
    versions = set(_changelog_versions())
    missing = sorted(t for t in _git_tags() if t.lstrip("v") not in versions)
    assert not missing, (
        f"{missing} are tagged but have no CHANGELOG section. v0.5.0-v0.5.6 sat like this "
        f"for three months with their notes only in the GitHub Releases UI; backfill from "
        f"`gh release view <tag>` rather than leaving the file incomplete."
    )


def test_released_sections_are_dated():
    body = CHANGELOG.read_text()
    undated = [
        v for v in _changelog_versions()
        if not re.search(rf"^## \[{re.escape(v)}\] — \d{{4}}-\d{{2}}-\d{{2}}", body, re.M)
    ]
    # The 0.1.0 exception that used to live here is gone: that heading is no longer bracketed, so
    # `_changelog_versions()` does not yield it and there is nothing to exempt. It was never tagged
    # and never released, so a `[0.1.0]` reference-style heading with no link definition rendered as
    # literal brackets — neither a `compare/` range nor a `releases/tag/` URL exists for it.
    assert not undated, f"released sections without an ISO date: {undated}"


def test_every_bracketed_section_has_a_link_definition():
    """A `## [x]` with no `[x]:` footer entry renders as literal brackets, not a link.

    This is what `## [0.1.0] — 2025-09-XX` did for its whole life. Guarding it here means the next
    section added without a footer entry fails immediately rather than shipping looking broken.
    """
    body = CHANGELOG.read_text()
    used = set(re.findall(r"^## \[([^\]]+)\]", body, re.M))
    defined = set(re.findall(r"^\[([^\]]+)\]:", body, re.M))
    missing = sorted(used - defined)
    assert not missing, (
        f"bracketed CHANGELOG headings with no link definition: {missing}. Either add "
        f"`[x]: https://github.com/pinellolab/chorus/compare/v<prev>...v<x>` to the footer, or drop "
        f"the brackets if the version was never tagged."
    )


def test_every_released_section_states_its_artefact_revision():
    """Because a code tag alone does not pin behaviour in this project.

    Applies from 0.6.0 on: that is the first release for which a dataset revision was
    tagged, and 0.7.0 is the first that pins one in code. Earlier releases cannot state a
    revision honestly, so they are not asked to.
    """
    body = CHANGELOG.read_text()
    for version in ("0.7.0", "0.6.0"):
        i = body.index(f"## [{version}]")
        j = body.index("\n## [", i + 10)
        section = body[i:j]
        assert "Background artefacts." in section, (
            f"[{version}] does not state which background revision it pairs with. "
            f"Percentiles are a function of (code, artefacts); a version that names only "
            f"the code is not reproducible."
        )
        assert re.search(r"backgrounds-\d{4}-\d{2}-\d{2}-\w+", section), (
            f"[{version}] mentions artefacts but names no dataset tag"
        )


def test_unreleased_is_empty_when_head_is_tagged():
    """After cutting a tag, [Unreleased] should not still hold that release's entries.

    Skips on an untagged HEAD, which is the normal state mid-development.
    """
    out = subprocess.run(
        ["git", "tag", "--points-at", "HEAD"], cwd=REPO, capture_output=True, text=True,
    )
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    tags = [t for t in out.stdout.split() if re.fullmatch(r"v\d+\.\d+\.\d+", t)]
    if not tags:
        pytest.skip("HEAD is not tagged; [Unreleased] may legitimately hold entries")

    body = CHANGELOG.read_text()
    i = body.index("## [Unreleased]")
    j = body.index("\n## [", i + 10)
    section = body[i + len("## [Unreleased]"):j].strip()
    bullets = [ln for ln in section.split("\n") if ln.startswith("- ")]
    assert not bullets, (
        f"HEAD is tagged {tags} but [Unreleased] still lists {len(bullets)} entries; they "
        f"belong in the released section or they will be attributed to the next release"
    )


# ── the version as the README states it, in the two places a reader copies from ──────────

# Both of these were stale when found during the 0.7.5 fold, and neither was covered: the guard above
# ties setup.py to chorus/__init__.py, and test_citation_is_valid_and_consistent ties CITATION.cff to
# __version__ and the README BibTeX's *title and authors* to CITATION.cff — but nothing read the README's
# install tag or the BibTeX's own `version` field. So the install instruction said `git checkout v0.7.4`
# while the tree was 0.7.5, and the BibTeX said 0.7.3, two releases behind. Both are copy-paste targets:
# one decides which code a reader installs, the other decides what they cite in a paper.

README_MD = REPO / "README.md"


def _readme_install_tag() -> str:
    m = re.search(r"git checkout v([0-9][^\s]*)", README_MD.read_text())
    assert m, "the README no longer shows a `git checkout v<tag>` install step"
    return m.group(1)


def _readme_bibtex_version() -> str:
    bib = re.search(r"```bibtex\n(.*?)```", README_MD.read_text(), re.S)
    assert bib, "no ```bibtex block in README.md"
    m = re.search(r"version\s*=\s*\{([^}]+)\}", bib.group(1))
    assert m, "the README BibTeX entry has no version field"
    return m.group(1).strip()


def test_the_readme_install_tag_is_this_version():
    """What a reader is told to check out must be what this tree is."""
    assert _readme_install_tag() == _init_version(), (
        f"README says `git checkout v{_readme_install_tag()}` but this tree is "
        f"{_init_version()}. A reader following the install steps gets a different version than the "
        f"one the rest of the README's numbers were measured against."
    )


def test_the_readme_citation_version_is_this_version():
    """What a reader cites must be what they installed."""
    assert _readme_bibtex_version() == _init_version(), (
        f"README BibTeX cites {_readme_bibtex_version()} but this tree is {_init_version()}. "
        f"This block is copied into papers; a stale version there is a wrong citation of record."
    )
