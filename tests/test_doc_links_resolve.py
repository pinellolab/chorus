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
