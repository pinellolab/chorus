"""Two documents must not disagree about whether the history was rewritten.

`audits/README.md` explained that dropping the audit artefacts from the tree would not shrink
`git clone`, and closed with: *"Making the clone smaller would require rewriting history, which
invalidates every existing clone and every published sha — deliberately not done."*

It was then done. `git-filter-repo` purged 338 paths / 142.6 MB for v0.7.3 and `CHANGELOG.md` opens
the release with a section titled "⚠ Git history was rewritten for this release". So the repo shipped
one document telling a clone-holder to re-clone and another telling them the rewrite had been
declined — and the second is the one a reader lands on from `audits/`.

This is the failure mode the rest of this release is about: a document describing something other
than what happened. It is cheap to pin, because both statements are keyword-detectable.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHANGELOG = REPO / "CHANGELOG.md"
AUDITS_README = REPO / "audits" / "README.md"


def _changelog_says_rewritten() -> bool:
    return "Git history was rewritten" in CHANGELOG.read_text()


def test_the_changelog_still_documents_the_rewrite():
    """If this ever goes false the guard below is inverted, so state the premise explicitly."""
    assert _changelog_says_rewritten(), (
        "CHANGELOG.md no longer contains a 'Git history was rewritten' section. Every sha in this "
        "repository changed at v0.7.3; a clone-holder who does not know that gets divergent "
        "histories from `git pull` with no explanation."
    )


def test_no_document_claims_the_rewrite_was_declined():
    """The exact contradiction that shipped, in the file most likely to carry it."""
    if not _changelog_says_rewritten():
        return

    text = AUDITS_README.read_text()
    # "deliberately not done", "we did not rewrite", "not been rewritten" — any phrasing that denies
    # it, within reach of the words describing a rewrite.
    for m in re.finditer(r"rewrit\w*", text, re.I):
        window = text[max(0, m.start() - 200):m.end() + 200].lower()
        denials = ("deliberately not done", "not done", "we have not", "has not been",
                   "was not done", "declined", "chose not to")
        hit = next((d for d in denials if d in window), None)
        # A denial quoted as history ("this used to end 'deliberately not done'. It was then done.")
        # is narration, not a claim. Without this the guard fired on a correct paragraph and passed
        # only by a 38-character margin, so any reword would have broken it.
        if hit and any(k in window for k in ("used to", "was then done", "it was then")):
            continue
        assert hit is None, (
            f"audits/README.md says {hit!r} within 200 characters of a mention of rewriting, while "
            f"CHANGELOG.md documents that the history WAS rewritten for v0.7.3. One of the two is "
            f"wrong, and a reader arriving from audits/ hits this one first.\n"
            f"context: ...{text[max(0, m.start() - 120):m.end() + 120]}..."
        )


def test_the_audit_readme_points_at_the_changelog_section():
    """A reader who learns the shas changed needs the instructions, which live in one place."""
    text = AUDITS_README.read_text()
    assert "CHANGELOG.md" in text, (
        "audits/README.md discusses the repository's size and history without pointing at the "
        "CHANGELOG section that tells a clone-holder what to do about it"
    )


def test_the_branch_cleanup_is_recorded_with_its_caveats():
    """Deleting 18 remote branches is not self-documenting six months later.

    Two facts are easy to lose and expensive to rediscover: GitHub's reported size does not drop when
    you delete a branch (it GCs on its own schedule), and `git branch --merged` cannot confirm a
    squash-merged branch landed — the PR record is the only authority.
    """
    # Normalise before matching: the prose wraps mid-phrase and carries `**bold**`, so a naive
    # substring check fails on text that says exactly the right thing.
    text = re.sub(r"[*`_]", "", AUDITS_README.read_text().lower())
    text = re.sub(r"\s+", " ", text)
    for want, why in (
        ("gc schedule", "otherwise the next person reads the unchanged size as a failed deletion"),
        ("--merged", "otherwise someone uses ancestry to decide what is safe to delete, and it lies"),
        ("gh pr list", "the authoritative check should be copy-pasteable"),
    ):
        assert want in text, f"audits/README.md does not mention {want!r} — {why}"


def test_the_audit_checklist_does_not_ask_for_what_the_repo_just_removed():
    """The live runbook must not instruct the next auditor to re-commit the archived artefacts.

    v0.7.3 removed 347 raw audit artefacts (~122 MB) from the tree and `audits/README.md` explains
    why. `AUDIT_CHECKLIST.md` — which `CLAUDE.md` names as the thing to run before any release — still
    listed `screenshots/*.png`, `nb_fresh_output/*.ipynb`, `cdf_check.txt` and `device_probe.txt` under
    "a full audit should leave behind, in audits/...". Followed literally, the next audit undoes the
    cleanup, and the runbook is the more likely of the two documents to be obeyed.
    """
    checklist = (REPO / "audits" / "AUDIT_CHECKLIST.md").read_text()
    i = checklist.index("## Appendix — artefacts to produce per audit")
    appendix = checklist[i:]

    lowered = appendix.lower()
    assert "commit only" in lowered or "outside the repository" in lowered, (
        "the audit checklist's artefact appendix does not distinguish what to commit from what to "
        "keep outside the repo. v0.7.3 removed 347 files / ~122 MB of exactly these artefacts; a "
        "runbook that asks for them back re-adds that weight to every clone."
    )
    assert "audits/README.md" in appendix or "README.md" in appendix, (
        "the appendix should point at audits/README.md, which records what was archived and why"
    )
