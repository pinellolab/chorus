"""The citation must parse, and the repo must offer exactly one.

For a research release the citation is the artefact the community is *required* to use, and it was
the one block of README text no test ever exercised. Two defects shipped:

* **The BibTeX author field separated names with commas** —
  `{Dmitry Penzar , Lorenzo Ruggeri , Rosalba Giugno, Luca Pinello}`. BibTeX's separator is
  ` and `; commas inside a single name are the `von Last, Jr, First` form, which allows at most
  two. Three commas is past that, so bibtex/biber report a name-parse error or silently collapse
  four authors into one garbled name — in someone else's paper.
* **`docs/THIRD_PARTY.md` gave a second, different citation** — different title, and "Pinello Lab"
  instead of the four named authors — leaving a reader to guess which was authoritative.

There was also no `CITATION.cff`, so GitHub showed no "Cite this repository" button. These tests keep
all three in agreement.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
CFF = REPO / "CITATION.cff"


def _bibtex() -> str:
    m = re.search(r"```bibtex\n(.*?)```", README.read_text(), re.S)
    assert m, "no ```bibtex block in README.md — the citation is what a reader copies"
    return m.group(1)


def _field(name: str, bib: str) -> str:
    m = re.search(rf"{name}\s*=\s*\{{(.+?)\}},?\s*$", bib, re.M | re.S)
    assert m, f"no `{name} = {{...}}` in the README BibTeX entry"
    return m.group(1).strip()


def test_the_bibtex_author_field_uses_and_not_commas():
    """The exact defect that shipped. `and` is BibTeX's separator; commas mean something else."""
    author = _field("author", _bibtex())
    names = [n.strip() for n in author.split(" and ")]

    assert len(names) > 1, (
        f"the author field has no ' and ' separator, so BibTeX reads it as ONE name: {author!r}"
    )
    for n in names:
        assert n.count(",") <= 1, (
            f"{n!r} contains {n.count(',')} commas. Within a single name BibTeX allows at most the "
            f"`von Last, Jr, First` form; more than that is a parse error. Separate authors with "
            f"' and '."
        )
        assert n == n.strip() and "  " not in n, f"stray whitespace in author name {n!r}"


def test_the_citation_file_exists_and_parses():
    """`CITATION.cff` is what makes GitHub render a 'Cite this repository' button."""
    assert CFF.exists(), "no CITATION.cff — GitHub will not offer a citation widget"
    data = yaml.safe_load(CFF.read_text())
    for key in ("cff-version", "message", "title", "authors"):
        assert key in data, f"CITATION.cff is missing the required key {key!r}"
    assert data["authors"], "CITATION.cff lists no authors"
    for a in data["authors"]:
        assert "family-names" in a, f"author entry without family-names: {a}"


def test_the_bibtex_and_the_citation_file_agree():
    """Two citations that disagree are worse than one, because the reader has to choose."""
    bib = _bibtex()
    data = yaml.safe_load(CFF.read_text())

    assert _field("title", bib) == data["title"], (
        f"title differs: README {_field('title', bib)!r} vs CITATION.cff {data['title']!r}"
    )

    bib_names = [n.strip() for n in _field("author", bib).split(" and ")]
    cff_names = [f"{a['family-names']}, {a['given-names']}" for a in data["authors"]]
    assert bib_names == cff_names, (
        f"author lists differ:\n  README:       {bib_names}\n  CITATION.cff: {cff_names}"
    )


def test_the_cited_version_is_the_declared_one():
    import chorus

    data = yaml.safe_load(CFF.read_text())
    assert str(data.get("version")) == chorus.__version__, (
        f"CITATION.cff cites {data.get('version')} but chorus.__version__ is {chorus.__version__}; "
        f"a citation that names the wrong version misattributes the numbers in someone's paper"
    )


def test_no_second_competing_citation_in_the_docs():
    """`THIRD_PARTY.md` used to carry its own, differently-worded citation."""
    text = (REPO / "docs" / "THIRD_PARTY.md").read_text()
    section = text[text.index("## Chorus itself"):][:1200]
    assert "Cite as:" not in section or "README" in section, (
        "docs/THIRD_PARTY.md offers its own citation again. Point at the README's BibTeX instead — "
        "one canonical citation, not two that a reader has to choose between."
    )


@pytest.mark.parametrize("bad", [
    "{Dmitry Penzar , Lorenzo Ruggeri , Rosalba Giugno, Luca Pinello}",
    "{A One, B Two, C Three}",
])
def test_the_guard_catches_the_comma_separated_form(bad):
    """Fails-without-fix, using the string that actually shipped."""
    names = [n.strip() for n in bad.strip("{}").split(" and ")]
    caught = len(names) == 1 or any(n.count(",") > 1 for n in names)
    assert caught, f"guard would not catch {bad!r}"
