"""A report must not misdescribe the population its percentiles rank against.

Every report Chorus generated said the effect percentile was "ranked against ~10K
random SNPs". That was true of a much older build. This release anchors every
effect null on assay-matched regulatory strata — cCREs, DHS summits, promoters,
gene features — and the per-track background holds **17,805 to 225,253** effects
drawn from ~18,000 positions, depending on the oracle and how the layer fans out
across genes.

So the sentence was wrong in three ways at once: the count (by up to 22x), the
sampling (regulatory, not uniform), and the implied claim that a random genomic
position is the comparison. It appeared in 28 committed artefacts and would have
appeared in every future user report. Worse, the same file that generated it also
told users the opposite elsewhere, and `docs/NORMALIZATION_GUIDE.md` gave a third
answer ("~10K common SNPs from gnomAD") for a pipeline that samples no gnomAD at
all.

These tests are deliberately about the PROSE, not the numbers. A number that
drifts is caught by the artefact-consistency tests; a *description* that drifts is
caught by nothing, because it is not derived from anything — which is exactly why
it survived a release that changed the thing it describes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Files that generate user-facing prose about the reference class.
GENERATORS = [
    REPO / "chorus" / "analysis" / "variant_report.py",
    REPO / "chorus" / "analysis" / "batch_scoring.py",
    REPO / "chorus" / "analysis" / "_report_glossary.py",
    REPO / "chorus" / "analysis" / "multi_oracle_report.py",
    REPO / "chorus" / "analysis" / "causal.py",
]

DOCS = [
    REPO / "README.md",
    REPO / "docs" / "NORMALIZATION_GUIDE.md",
    REPO / "docs" / "BACKGROUND_NULL_PROTOCOL.md",
    REPO / "docs" / "variant_analysis_framework.md",
]

# Each claim that is false of the shipped nulls, with why.
FORBIDDEN = [
    (r"~?10[,.]?000 random SNPs", "the effect null is not 10,000 uniformly random SNPs"),
    (r"~?10K random SNPs", "same claim, abbreviated"),
    (r"vs random SNPs", "same claim, in a tooltip"),
    (r"~?10K common SNPs", "and it is not gnomAD common SNPs either"),
    (r"sampled uniformly across chr1[-–]chr22", "sampling is stratified, not uniform"),
    # Scoped: naming gnomAD as a COORDINATE convention alongside dbSNP/UCSC/IGV is
    # correct and common (README:80 does it). What is false is claiming the background
    # is drawn from it, so require a background-ish word nearby.
    # "SNP" is deliberately NOT a trigger here: it matches inside "dbSNP", and
    # "matching dbSNP / gnomAD / UCSC / IGV" (README:80) is a correct statement about
    # COORDINATE conventions. Only a background-ish word near gnomAD is the false claim.
    (r"gnomAD[^.\n]{0,80}(background|null|percentile|sampl)"
     r"|(background|null|percentile|sampl)[^.\n]{0,80}gnomAD",
     "no code path samples gnomAD; saying the null comes from it sends users looking"),
]


@pytest.mark.parametrize("path", GENERATORS, ids=lambda p: p.name)
def test_report_generators_do_not_claim_a_uniform_random_reference_class(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} absent")
    src = path.read_text()
    for pattern, why in FORBIDDEN:
        hits = [m.group(0) for m in re.finditer(pattern, src, re.I)]
        assert not hits, (
            f"{path.name} tells users {hits[0]!r} — {why}. The shipped nulls hold "
            f"17,805-225,253 effects from ~18,000 assay-matched regulatory positions. "
            f"See docs/BACKGROUND_NULL_PROTOCOL.md §3."
        )


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_docs_do_not_claim_a_uniform_random_reference_class(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} absent")
    src = path.read_text()
    for pattern, why in FORBIDDEN:
        # The protocol and the guide are allowed to QUOTE the retired claim while
        # explaining that it is retired, so only flag it outside a correction.
        for m in re.finditer(pattern, src, re.I):
            line_start = src.rfind("\n", 0, m.start()) + 1
            line_end = src.find("\n", m.end())
            line = src[line_start:line_end if line_end > 0 else len(src)]
            if re.search(r"not\b|never|no code|retired|used to|previously|wrong", line, re.I):
                continue
            pytest.fail(
                f"{path.name} says {m.group(0)!r} as a live claim — {why}.\n  {line.strip()}"
            )


def test_the_committed_artefacts_do_not_carry_the_retired_claim():
    """Catches the case where the source is fixed but nothing was regenerated.

    That is the ordinary outcome of fixing prose in a generator: 28 artefacts kept
    the old sentence until they were rebuilt.
    """
    stale = []
    for p in (REPO / "examples").rglob("*"):
        if p.suffix not in {".md", ".html", ".json"} or not p.is_file():
            continue
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        for m in re.finditer(r"~?10K random SNPs|~?10,000 random SNPs|vs random SNPs",
                             txt, re.I):
            # A README documenting that the claim WAS wrong is doing the right thing;
            # SORT1_rs12740374/README.md quotes it inside exactly such a correction.
            ls = txt.rfind("\n", 0, m.start()) + 1
            le = txt.find("\n", m.end())
            line = txt[ls:le if le > 0 else len(txt)]
            if re.search(r"were wrong|was wrong|not\b|never|retired|earlier revision"
                         r"|used to|previously", line, re.I):
                continue
            stale.append(f"{p.relative_to(REPO)}: {line.strip()[:90]}")
            break
    assert not stale, (
        f"{len(stale)} committed artefact(s) still describe the old reference class; "
        f"regenerate them. First few: {stale[:5]}"
    )


def test_the_true_range_is_what_the_shipped_nulls_actually_hold():
    """Pin the numbers the prose now quotes, so the prose and the artefacts agree.

    If a future rebuild moves the position count outside this range, the reports'
    "~18,000" becomes wrong and this fails rather than the prose silently drifting
    again.
    """
    np = pytest.importorskip("numpy")
    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR

    files = sorted(CHORUS_BACKGROUNDS_DIR.glob("*_pertrack.npz"))
    if not files:
        pytest.skip("no downloaded backgrounds")

    lo, hi = None, None
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            if "effect_counts" not in d.files:
                continue
            c = np.asarray(d["effect_counts"])
            lo = int(c.min()) if lo is None else min(lo, int(c.min()))
            hi = int(c.max()) if hi is None else max(hi, int(c.max()))
    assert lo is not None, "no effect_counts in any shipped background"
    # ~18,000 positions is the claim; per-track effect COUNTS fan out above that for
    # layers scored per gene (RNA, CAGE), which is why the upper bound is much larger.
    assert 15_000 <= lo <= 20_000, (
        f"minimum effect count is {lo}, so reports claiming a background of ~18,000 "
        f"variants are no longer accurate"
    )
    assert hi <= 400_000, f"maximum effect count {hi} is outside the documented fan-out"
