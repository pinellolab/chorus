"""The README's normalization example must run, and its numbers must be the current ones.

Two defects shipped in one six-line snippet, and nothing caught either because no test had ever
executed a README example:

* `norm.activity_percentile("alphagenome", track_id, ref_value=512.0)` — there is no `ref_value`
  keyword. The real signature is `activity_percentile(oracle_name, track_id, raw_signal)`, so the
  documented call raises `TypeError: got an unexpected keyword argument 'ref_value'`. A new user
  copy-pasting the quickstart hit an exception on line 2 of it.
* the effect value was documented as `0.962` and measures **0.9811** against the pinned artefact —
  it moved in the 2026-08 background rebuild and the prose did not follow.

This is the class of defect worth a test rather than a proofread: a snippet either runs or it does
not, and a percentile either matches the shipped CDFs or it does not. Both are cheap to check and
neither survives a rebuild silently again.

Integration-marked because it loads a real 279 MB NPZ.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"

TRACK = "DNASE/EFO:0001187 DNase-seq/."


def _snippet() -> str:
    """The normalization example block, located by its distinctive call."""
    text = README.read_text()
    i = text.index('norm.effect_percentile("alphagenome"')
    start = text.rindex("```python", 0, i)
    end = text.index("```", i)
    return text[start + len("```python"):end]


def _documented_keywords() -> dict[str, set[str]]:
    """Keyword names the README snippet passes, per method, parsed rather than regexed.

    The first version matched `{method}\\([^)]*?(\\w+)=`, which cannot cross the `)` of the nested
    `abs(0.45)` in the snippet. It therefore found **zero** keywords and the loop below never ran:
    renaming the real `signed` parameter to `is_signed` left the test green. An AST walk sees the call
    structure instead of guessing at it, and nested calls stop mattering.
    """
    import ast

    tree = ast.parse(_snippet().strip())
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name in ("effect_percentile", "activity_percentile"):
            out.setdefault(name, set()).update(
                kw.arg for kw in node.keywords if kw.arg is not None
            )
    return out


def test_the_documented_calls_use_the_real_signatures():
    """Static half: no keyword the API does not accept. Runs without loading an artefact."""
    import inspect

    from chorus.analysis.normalization import PerTrackNormalizer

    documented = _documented_keywords()
    assert documented, (
        "no percentile calls were parsed out of the README snippet, so this test would pass "
        "vacuously -- the same way it did when a regex could not cross a nested paren"
    )
    for method, kws in documented.items():
        params = set(inspect.signature(getattr(PerTrackNormalizer, method)).parameters)
        for kw in kws:
            assert kw in params, (
                f"README calls {method}(…{kw}=…) but the signature has no `{kw}` parameter "
                f"(accepts: {sorted(params - {'self'})}). The documented call would raise TypeError."
            )


def test_the_signature_check_sees_the_keyword_the_snippet_actually_passes():
    """Fails-without-fix for the vacuous-regex bug: `signed=` must be among the parsed keywords."""
    documented = _documented_keywords()
    assert "signed" in documented.get("effect_percentile", set()), (
        f"the snippet passes signed=False to effect_percentile, but the parser found "
        f"{documented} -- if that keyword is invisible, no signature drift can be caught"
    )


def test_the_documented_call_shape_is_positional_for_raw_signal():
    """`ref_value=` specifically — the keyword that shipped and raised.

    Checks the *code* lines only. The snippet deliberately carries a comment naming `ref_value=` to
    warn readers off it, and a bare substring check flagged that comment — the first version of this
    test did exactly that, failing on the fix rather than on the defect.
    """
    code = "\n".join(ln.split("#", 1)[0] for ln in _snippet().splitlines())
    assert "ref_value" not in code, (
        "README calls activity_percentile with `ref_value=` again; the third parameter is "
        "`raw_signal` and the example passes it positionally"
    )


@pytest.mark.integration
def test_the_documented_percentiles_match_the_shipped_artefact():
    """Dynamic half: run the example and compare to the numbers the README prints."""
    from chorus.analysis.normalization import get_pertrack_normalizer

    norm = get_pertrack_normalizer("alphagenome")
    if norm is None:
        pytest.skip("alphagenome backgrounds not available on this host")

    snippet = _snippet()
    documented = [float(m) for m in re.findall(r"#\s*→\s*([0-9.]+)", snippet)]
    assert len(documented) == 2, f"expected two documented values, found {documented}"

    eff = norm.effect_percentile("alphagenome", TRACK, abs(0.45), signed=False)
    act = norm.activity_percentile("alphagenome", TRACK, 512.0)

    assert eff is not None and act is not None, (
        f"the documented track {TRACK!r} returned None — the README example would print nothing"
    )
    assert round(eff, 4) == pytest.approx(documented[0], abs=5e-4), (
        f"README documents an effect percentile of {documented[0]} but the shipped CDFs give "
        f"{eff:.4f}. Percentiles move when backgrounds are rebuilt; the prose has to move with them."
    )
    assert round(act, 4) == pytest.approx(documented[1], abs=5e-4), (
        f"README documents an activity percentile of {documented[1]} but the shipped CDFs give "
        f"{act:.4f}."
    )


def test_the_stratification_table_matches_the_code():
    """The README described the null's composition with every fraction exactly 2x reality.

    It listed a gene-anchored mixture summing to 100% with **no cCRE stratum at all**, while
    `DEFAULT_REGION_STRATA` puts **half** the null inside cCREs. Renormalising the remainder is what
    doubled every published fraction. This is the methods description for every percentile chorus
    reports, so it is worth pinning to the source of truth.
    """
    from chorus.utils.annotations import DEFAULT_REGION_STRATA

    text = README.read_text()
    i = text.index("| oracle | effect reference population |")
    # Scope to the table itself. The first version took a flat 1400-character slice, which reached
    # ~700 characters past the table into prose containing "50 % of Enformer's accessibility rows" --
    # so the cCRE assertion was satisfied by an unrelated sentence, and deleting the cCRE stratum
    # from the table (the exact regression this test names) still passed.
    table = text[i:text.index("\n\n", i)]

    assert abs(sum(DEFAULT_REGION_STRATA.values()) - 1.0) < 1e-9, DEFAULT_REGION_STRATA

    # Each percentage anchored to its own label, and all six strata covered. Bare "10 %" was
    # previously satisfied for tss_near by tss_far's identical figure in the sibling clause.
    LABELS = {
        "ccre": "inside cCREs",
        "tss_near": "within ±1 kb of a TSS",
        "tss_far": "at 1–10 kb",
        "junction": "within ±100 bp of an exon/intron boundary",
        "gene_body": "elsewhere in a gene body",
        "random": "uniformly random",
    }
    assert set(LABELS) == set(DEFAULT_REGION_STRATA), (
        f"strata changed in code: {sorted(DEFAULT_REGION_STRATA)} vs documented {sorted(LABELS)}"
    )
    for name, label in LABELS.items():
        pct = DEFAULT_REGION_STRATA[name] * 100
        anchored = f"{pct:g} % {label}"
        assert anchored in table, (
            f"the README stratification table does not state {anchored!r} for `{name}`.\n"
            f"Every percentage must sit next to its own label: the table once omitted the cCRE "
            f"stratum entirely and doubled every other fraction, and a bare percentage check could "
            f"not see either problem.\ntable was:\n{table}"
        )


# ── README variant-effect recipes ────────────────────────────────────────────────

def _readme_allele_blocks() -> list[tuple[int, str, int, str]]:
    """(line_no, chrom, pos, ref_allele) for every `alleles=[...]` list near a variant position."""
    lines = README.read_text().splitlines()
    out = []
    for i, ln in enumerate(lines):
        m = re.search(r"\[\s*'([ACGT])'\s*,", ln)
        if not m:
            continue
        pos = None
        for j in range(max(0, i - 6), i):
            pm = re.search(r"'(chr[\dXYM]+):(\d+)'", lines[j])
            if pm:
                pos = pm.groups()
        if pos:
            out.append((i + 1, pos[0], int(pos[1]), m.group(1)))
    return out


@pytest.mark.integration
def test_every_readme_ref_allele_matches_the_genome():
    """`predict_variant_effect` is strict about the reference allele, and the README got it wrong twice.

    `strict_ref=True` is the default and deliberate (#128): a mismatched reference raises rather than
    silently substituting. The README's TLDR shipped `['A','G','C','T']` at chr11:5247500 where hg38
    has **C**; that was corrected in b5a8403 — and the identical block 600 lines later, in the
    "9 runnable recipes" section, was missed. Because it raises before binding `variant_effects`, two
    *later* recipes then died with `NameError`, so **3 of the 9 advertised recipes could not be run**.

    Needs the FASTA, hence integration-marked.
    """
    from chorus.utils.genome import GenomeManager

    fasta = Path(GenomeManager().get_genome_path("hg38") or "")
    if not fasta.is_file():
        pytest.skip("hg38.fa not available")

    from chorus.utils.sequence import extract_sequence

    blocks = _readme_allele_blocks()
    assert blocks, "no allele lists found in README — has the recipe section moved?"

    wrong = []
    for line_no, chrom, pos, ref in blocks:
        actual = extract_sequence(f"{chrom}:{pos}-{pos}", str(fasta)).upper()
        if actual != ref:
            wrong.append(f"README.md:{line_no}  says ref '{ref}' at {chrom}:{pos}, genome has '{actual}'")
    assert not wrong, (
        "README variant recipes name a reference allele the genome does not have. strict_ref=True "
        "means these raise ReferenceAlleleMismatchError rather than degrading:\n  "
        + "\n  ".join(wrong)
    )
