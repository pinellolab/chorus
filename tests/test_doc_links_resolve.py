"""Every in-page `](#anchor)` link in the docs must point at a heading that exists.

Two of these shipped. `README.md:114` advertised the 24-tool catalogue as `](#mcp-server)` while the
heading is `## MCP server — chorus, but you talk to Claude`, so the one link from the pitch into the
tool list went nowhere. And during the v0.7.3 install-docs pass a *fix* introduced another: the new
disk prerequisite offered `](#where-chorus-puts-large-files)` two edits before that section existed.

Both are invisible in review — a dead in-page anchor renders as ordinary blue text and simply does
nothing when clicked, so nobody notices until a reader reports it.

The slug rules mirror GitHub's: lowercase, drop everything that is not word/space/hyphen, then
replace each space with a hyphen. That last step is not a collapse — `A — B` becomes `a--b`, with two
hyphens, because the em-dash is deleted and both surrounding spaces still map to hyphens. Getting that
wrong makes a correct link look dead, which is how the first draft of this test produced eight false
positives.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS = [REPO / "README.md", REPO / "CONTRIBUTING.md", *sorted((REPO / "docs").glob("*.md"))]


def github_slug(heading: str) -> str:
    """GitHub's anchor slug for a heading's text."""
    s = heading.strip().lower()
    s = re.sub(r"`", "", s)              # inline code ticks are dropped, contents kept
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)   # links render as their text
    s = re.sub(r"[^\w\s-]", "", s)       # punctuation goes
    return s.replace(" ", "-")           # each space -> one hyphen, NOT collapsed


def _headings(text: str) -> set[str]:
    return {github_slug(m.group(2)) for m in re.finditer(r"^(#{1,6})\s+(.*)$", text, re.M)}


def _in_page_links(text: str) -> list[tuple[str, str]]:
    """(anchor, label) for every `](#anchor)` link."""
    return [(m.group(2), m.group(1))
            for m in re.finditer(r"\[([^\]]*)\]\(#([^)]+)\)", text)]


def _cross_file_links(text: str) -> list[tuple[str, str, str]]:
    """(relative_path, anchor, label) for every `](other.md#anchor)` link.

    The first draft of this file matched only same-page `](#anchor)`, so a dead
    `](../README.md#mcp-search)` was structurally invisible to it — and one was live:
    `docs/MCP_WALKTHROUGH.md:15` pointed at `../README.md#mcp-server`, the same slug that had
    already been fixed *inside* README. A guard blind to a whole link shape reads as coverage it
    does not have.
    """
    return [(m.group(2), m.group(3), m.group(1))
            for m in re.finditer(r"\[([^\]]*)\]\(([^)#\s]+\.md)#([^)\s]+)\)", text)]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_in_page_anchor_points_at_a_real_heading(doc: Path):
    if not doc.exists():
        pytest.skip(f"{doc.name} not present")
    text = doc.read_text()
    heads = _headings(text)
    dead = [f"[{label}](#{anchor})" for anchor, label in _in_page_links(text)
            if anchor not in heads]
    assert not dead, (
        f"{doc.name} has in-page links whose target heading does not exist:\n  "
        + "\n  ".join(dead)
        + "\nA dead anchor renders as normal text and silently does nothing when clicked."
    )


#: Every markdown file that may contain a cross-file anchor, not just the curated DOCS list —
#: the dead one that shipped was in a file the original list did not even include.
ALL_MARKDOWN = [p for p in sorted(REPO.rglob("*.md"))
                if not any(part in {".git", "node_modules", "build", "audits"}
                           for part in p.relative_to(REPO).parts)]


@pytest.mark.parametrize("doc", ALL_MARKDOWN, ids=lambda p: str(p.relative_to(REPO)))
def test_every_cross_file_anchor_points_at_a_real_heading(doc: Path):
    """`](other.md#anchor)` must resolve in the file it names."""
    dead = []
    for rel, anchor, label in _cross_file_links(doc.read_text()):
        target = (doc.parent / rel).resolve()
        if not target.is_file():
            dead.append(f"[{label}]({rel}#{anchor}) — {rel} does not exist")
            continue
        if anchor not in _headings(target.read_text()):
            dead.append(f"[{label}]({rel}#{anchor}) — no such heading in {rel}")
    assert not dead, (
        f"{doc.relative_to(REPO)} links to anchors that do not exist:\n  " + "\n  ".join(dead)
    )


def test_the_guard_sees_cross_file_links_at_all():
    """Guards the guard: the shape that was invisible in the first draft."""
    found = _cross_file_links("see [§MCP server](../README.md#mcp-server) for details")
    assert found == [("../README.md", "mcp-server", "§MCP server")], found


def test_the_slug_rules_match_githubs():
    """The em-dash case specifically — a collapsing implementation gets this wrong."""
    assert github_slug("MCP server — chorus, but you talk to Claude") == \
        "mcp-server--chorus-but-you-talk-to-claude"
    assert github_slug("Uninstalling / starting from scratch") == \
        "uninstalling--starting-from-scratch"
    assert github_slug("Where chorus puts large files") == "where-chorus-puts-large-files"
    assert github_slug("Cherimoya / CATv1") == "cherimoya--catv1"


def test_the_guard_catches_a_dead_anchor():
    """Fails-without-fix, using the wording that actually shipped in README.md:114."""
    text = "## MCP server — chorus, but you talk to Claude\n\nsee the [full list](#mcp-server).\n"
    heads = _headings(text)
    dead = [a for a, _ in _in_page_links(text) if a not in heads]
    assert dead == ["mcp-server"], f"guard no longer catches the shipped defect: {dead}"
