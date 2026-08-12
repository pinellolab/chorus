"""A percentile must be ranked against a null built by the same model.

Cherimoya ships two nulls because the folds are not interchangeable. Measured on
``DNASE:ENCSR149XIL`` at chr1:109,274,968, the five fold peaks are 8.24 / 15.47 / 15.34 /
11.08 / 7.65 against an ensemble peak of 11.10 -- a 2.02x spread, with any single fold landing
between 0.69x and 1.39x of the ensemble. So ranking a fold-0 prediction against the ensemble's
null (or the reverse) does not return an approximation; it returns a number that looks
completely normal and is wrong.

The default is fold 0, chosen so Cherimoya's scores are comparable with ChromBPNet, which also
defaults to fold 0 and whose null is built on the same reference sets (both reproduce
``effect_counts=18672`` and ``summary_counts=34004``). That comparison is the point of the
cross-oracle report.

Two mechanisms keep this honest and both are pinned here:

  * ``normalization_key`` resolves the CDF key from the fold the prediction was made with,
    read off the prediction's own metadata;
  * a load-time guard refuses any artefact whose stamped fold disagrees with the key asking
    for it, raising ``BackgroundFoldMismatch`` rather than degrading quietly.

The enumeration test is the important one. Wiring a fold-aware lookup at four sites and
missing a fifth is exactly the failure mode that shipped when three duplicated IGV render
paths each needed the same patch and only one got it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "chorus" / "analysis"


def test_the_default_fold_is_zero_and_matches_chrombpnet():
    """Comparability with ChromBPNet is the reason for the default, so pin both ends."""
    from chorus.oracles.cherimoya_source.catv1_globals import CATV1_DEFAULT_FOLD

    assert CATV1_DEFAULT_FOLD == 0, (
        "Cherimoya's default fold drives which null ships under the plain filename and "
        "whether its scores are comparable with ChromBPNet"
    )

    # ChromBPNet's own default, read from the signature rather than assumed.
    import inspect

    from chorus.oracles.chrombpnet import ChromBPNetOracle

    sig = inspect.signature(ChromBPNetOracle.load_pretrained_model)
    assert sig.parameters["fold"].default == 0, (
        "ChromBPNet no longer defaults to fold 0, so Cherimoya's fold-0 default no longer "
        "buys comparability -- the reason recorded in CHANGELOG and README is now false"
    )


def test_the_key_follows_the_fold_the_prediction_was_made_with():
    from chorus.analysis.normalization import normalization_key

    class _Pred:
        def __init__(self, fold):
            self.metadata = {"fold": fold, "atlas": "CATv1"}

    assert normalization_key("cherimoya", _Pred(0)) == "cherimoya"
    assert normalization_key("cherimoya", _Pred("ensemble")) == "cherimoya_ensemble"
    assert normalization_key("cherimoya", fold="ensemble") == "cherimoya_ensemble"
    # an oracle with no folds is untouched, whatever its metadata says
    assert normalization_key("alphagenome", _Pred("ensemble")) == "alphagenome"
    # and a prediction with no fold recorded falls to the default null, not the ensemble
    assert normalization_key("cherimoya", _Pred(None)) == "cherimoya"


def test_folds_without_a_null_are_refused_rather_than_approximated():
    """Refusing is the only option that cannot silently mislead.

    Allowing fold 3 to be ranked against fold 0's null would return a plausible percentile
    from a different model's distribution, and the 2.02x spread means that is not a rounding
    difference.
    """
    from chorus.core.exceptions import InvalidAssayError
    from chorus.oracles.cherimoya import CherimoyaOracle

    o = CherimoyaOracle.__new__(CherimoyaOracle)   # no env/model needed to reach the check
    for bad in (1, 2, 3, 4):
        with pytest.raises(InvalidAssayError) as exc:
            o.load_pretrained_model(encode_id="ENCSR149XIL", fold=bad)
        msg = str(exc.value)
        assert "no matching background null" in msg, msg
        assert "ensemble" in msg, "the error should name the modes that DO have a null"


def test_every_percentile_lookup_resolves_the_key_from_a_fold():
    """The enumeration guard: no lookup may pass a bare oracle_name.

    ``effect_percentile`` / ``activity_percentile`` / ``perbin_percentile_batch`` take the CDF
    key as their first argument. If any call site passes a raw ``oracle_name`` where a fold is
    available, a Cherimoya ensemble prediction gets ranked against fold 0's null. Patching
    four sites and missing a fifth is precisely how the three-render-path pooling bug shipped.
    """
    lookups = ("effect_percentile", "activity_percentile", "perbin_percentile_batch",
               "summary_percentile")
    offenders = []
    for path in sorted(ANALYSIS.glob("*.py")):
        if path.name == "normalization.py":       # where the methods are defined
            continue
        src = path.read_text()
        for fn in lookups:
            for m in re.finditer(rf"\.{fn}\(\s*([A-Za-z_][A-Za-z_0-9]*)", src):
                arg = m.group(1)
                if arg in ("norm_key", "key", "cdf_key", "normalization_key"):
                    continue
                if arg == "oracle_name" and "normalization_key" in src:
                    # the module resolves it -- confirm the resolution reaches this call by
                    # requiring the rebinding form used in variant_report
                    if re.search(r"oracle_name\s*=\s*_?normalization_key\(", src):
                        continue
                line = src[:m.start()].count("\n") + 1
                offenders.append(f"{path.name}:{line} .{fn}({arg})")
    assert not offenders, (
        "these percentile lookups pass a bare oracle name instead of a fold-resolved CDF "
        f"key, so a Cherimoya ensemble prediction would be ranked against fold 0's null: "
        f"{offenders}. Resolve with chorus.analysis.normalization.normalization_key()."
    )


def test_a_mislabelled_artefact_is_refused_not_used():
    """The last line of defence, for a stale cache or a hand-copied file."""
    import shutil
    import tempfile

    from chorus.analysis.normalization import (
        BackgroundFoldMismatch,
        CHORUS_BACKGROUNDS_DIR,
        PerTrackNormalizer,
    )

    ensemble = Path(CHORUS_BACKGROUNDS_DIR) / "cherimoya_ensemble_pertrack.npz"
    if not ensemble.exists():
        pytest.skip("ensemble artefact not present")

    d = Path(tempfile.mkdtemp())
    shutil.copy(ensemble, d / "cherimoya_pertrack.npz")     # ensemble under the fold-0 name
    with pytest.raises(BackgroundFoldMismatch) as exc:
        PerTrackNormalizer(cache_dir=str(d))._ensure_loaded("cherimoya")
    assert "fold" in str(exc.value)


def test_both_nulls_ship_and_describe_themselves():
    from chorus.analysis.normalization import CHORUS_BACKGROUNDS_DIR, PerTrackNormalizer

    n = PerTrackNormalizer()
    expected = {"cherimoya": 0, "cherimoya_ensemble": "ensemble"}
    for key, fold in expected.items():
        if not (Path(CHORUS_BACKGROUNDS_DIR) / f"{key}_pertrack.npz").exists():
            pytest.skip(f"{key} artefact not present")
        entry = n._ensure_loaded(key)
        assert entry is not None and len(entry["track_index"]) == 1518, key
    assert PerTrackNormalizer._EXPECTED_FOLD == expected
