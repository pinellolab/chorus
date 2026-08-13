"""No module may claim a report needs no network (#139, and twice since).

Inlining `igv.min.js` removes the **library** from a report's network dependencies. It does not
make the report offline-capable: the reference **sequence** is still fetched, because every
igv.js version requires a sequence source and hg38 is 3 GB.

The same distinction has now been got wrong three times in this repo:

1. `_ensure_igv_local`'s comment claimed "viewable offline, through SSL-MITM proxies, on
   air-gapped hosts" — false for three months; #139 corrected it.
2. `_igv_report`'s **module docstring** still said "self-contained HTML page" and "Gene
   annotations come from hg38 automatically via IGV's built-in genome" *after* #139 replaced
   both — found by an adversarial verification pass on the v0.7.3 release.
3. `causal.py` still carried "so the rendered HTML is self-contained offline" — same pass.

**Why this file is a blocklist rather than a clever matcher.** The first version matched
"self-contained html" plus a keyword allowlist for nearby qualifiers, and it failed twice over:
it flagged three *correct* lines (base64 PNGs really are self-contained; "listing tracks works
offline" is true of vendored metadata; the `CHORUS_IGV_SEQUENCE_URL` docstring describes how to
*make* a site offline-capable), and then, once the allowlist was wide enough to let those pass,
it **stopped catching the real defect** — a reintroduced "the rendered HTML report is
self-contained offline" slipped through because a nearby line happened to contain "sequence".

A guard that cannot catch the thing it was written for is worse than none: it reads as
assurance. So this matches the specific false claims, exempts the two known-true sentences by
name, and is mutation-tested at the bottom.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MODULES = sorted((REPO / "chorus").rglob("*.py"))

#: The claims that are false about a chorus report, matched case-insensitively.
FALSE_CLAIMS = (
    re.compile(r"self-contained\s+offline", re.I),
    re.compile(r"viewable\s+offline", re.I),
    re.compile(r"(?:report|html|panel)\b[^.\n]{0,60}\b(?:works?|is|are)\s+offline", re.I),
    re.compile(r"air-?gapped\s+hosts?\b(?![^.\n]*serve)", re.I),
)

#: Sentences that DO contain an offline word and are true. Matched as substrings, so each is a
#: statement someone deliberately wrote and someone else can re-check — not a keyword heuristic.
EXEMPT = (
    # Describes how to MAKE a site offline-capable by self-hosting the sequence.
    "serve the genomes\n    directory over HTTP and reports need no internet at all",
    "directory over HTTP and reports need no internet at all",
    # About the vendored CATv1 metadata TSVs, not about a report.
    "searching tracks works offline and pins to the code version",
)


def _offending_lines(text: str) -> list:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if not any(p.search(line) for p in FALSE_CLAIMS):
            continue
        if any(ex in line or line.strip() in ex for ex in EXEMPT):
            continue
        out.append((i, line.strip()))
    return out


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(REPO)))
def test_no_module_claims_a_report_needs_no_network(path: Path):
    offenders = [f"{path.relative_to(REPO)}:{i}  {ln[:96]}" for i, ln in
                 _offending_lines(path.read_text())]
    assert not offenders, (
        "these lines claim a report needs no network. The reference sequence is still fetched "
        f"(#139), so the claim is false:\n  " + "\n  ".join(offenders)
        + "\n  Say what inlining the JS actually buys, or point at CHORUS_IGV_SEQUENCE_URL."
    )


@pytest.mark.parametrize("claim", [
    "# so the rendered HTML is self-contained offline.",
    '"""Reports are viewable offline, on air-gapped hosts."""',
    "# the report works offline once the JS is inlined",
])
def test_the_guard_catches_each_wording_that_actually_shipped(claim):
    """Fails-without-fix, for the guard itself.

    Every string here is a real or near-verbatim version of a claim this repo shipped. If a
    future tidy-up loosens the patterns above, this notices — which the first version of this
    file did not, and it silently stopped working.
    """
    assert _offending_lines(f"x = 1\n{claim}\ny = 2\n"), f"guard no longer catches: {claim}"


def test_the_guard_leaves_the_true_statements_alone():
    """The other half: three correct lines that an earlier draft flagged."""
    for ok in (
        "Both are returned as base64-encoded PNGs for embedding in self-contained HTML.",
        "searching tracks works offline and pins to the code version.  Only the",
        "directory over HTTP and reports need no internet at all:",
        '"""Generate a self-contained HTML report.',
    ):
        assert not _offending_lines(ok), f"false positive on a true statement: {ok}"
