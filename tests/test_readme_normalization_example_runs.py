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


def test_the_documented_calls_use_the_real_signatures():
    """Static half: no keyword the API does not accept. Runs without loading an artefact."""
    import inspect

    from chorus.analysis.normalization import PerTrackNormalizer

    snippet = _snippet()
    for method in ("effect_percentile", "activity_percentile"):
        params = set(inspect.signature(getattr(PerTrackNormalizer, method)).parameters)
        for kw in re.findall(rf"{method}\([^)]*?(\w+)=", snippet, re.S):
            assert kw in params, (
                f"README calls {method}(…{kw}=…) but the signature has no `{kw}` parameter "
                f"(accepts: {sorted(params - {'self'})}). The documented call would raise TypeError."
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
    table = text[i:i + 1400]

    assert abs(sum(DEFAULT_REGION_STRATA.values()) - 1.0) < 1e-9, DEFAULT_REGION_STRATA
    for name, pct in (("ccre", 50), ("tss_near", 10), ("junction", 16.5), ("random", 7.5)):
        expected = DEFAULT_REGION_STRATA[name] * 100
        assert abs(expected - pct) < 1e-9, f"{name} moved in the code: {expected}% vs doc {pct}%"
        rendered = f"{pct:g} %"
        assert rendered in table, (
            f"the README stratification table no longer states {rendered} for `{name}`. It once "
            f"omitted the cCRE stratum entirely and doubled every other fraction; keep it in step "
            f"with DEFAULT_REGION_STRATA."
        )
