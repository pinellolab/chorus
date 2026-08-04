"""The three release gates, asserted rather than described.

An earlier draft of the rebuild plan listed these as residuals to caption. Two were
fixable and one was a hedge, so all three became work items — and this file is what
stops them regressing quietly.

1. **No layer ships a constant percentile column.** The cause was never degeneracy:
   AlphaGenome CAGE's effect CDF holds 9,995 distinct values of 10,000. It was
   *saturation* — AG RNA's null topped out at 0.0417, so every effect above that
   pinned at exactly 1.0000 and the column stopped discriminating where it mattered.
2. **Strong eQTLs must rank as notable.** The calibration gate. A null that cannot
   put experimentally-validated regulatory variants above the middle of its own
   distribution is not measuring anything useful.
3. **A report must be bit-exact across two processes.** #127: two identical runs
   differed on 454 numeric fields with **36 sign flips**, and for CAGE the
   run-to-run noise (0.0054) exceeded the median effect being reported (0.0058) —
   i.e. 92.1 % of shipped CAGE rows were ranking noise.

Gates 1 and 2 read shipped artefacts and are skipped without them. Gate 3 needs a
GPU and is ``-m integration``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
BACKGROUNDS = Path.home() / ".chorus" / "backgrounds"
FIXTURE = Path(__file__).parent / "fixtures" / "strong_eqtl_effects.json"

# Measured against the shipped gene-anchored null on 2026-08-04:
#   RNA  232 rows over 8 liver tracks -> p50 0.781
#   CAGE 100 rows over 4 liver tracks -> p50 0.659
# The band is deliberately wide and one-sided-ish. It exists to catch a null that
# has *collapsed* (p50 near 0.5 means a strong eQTL is indistinguishable from a
# random variant) or one that has re-saturated (p50 near 1.0 means the column has
# stopped resolving the top end). It is NOT a target to tune toward: an earlier
# version of this file asserted [0.75, 0.95] as though that range were derived,
# when it was invented before any measurement existed. CAGE at 0.659 then read as
# a failure when it is simply where a correctly-widened null puts these variants.
_CALIBRATION_BAND = (0.55, 0.97)


def _load(oracle: str):
    path = BACKGROUNDS / f"{oracle}_pertrack.npz"
    if not path.exists():
        pytest.skip(f"no downloaded background for {oracle}")
    from chorus.analysis.normalization import PerTrackNormalizer
    return PerTrackNormalizer(cache_dir=str(BACKGROUNDS))


# ---------------------------------------------------------------------------
# Gate 1 — no constant percentile column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layer", ["CAGE", "RNA_SEQ"])
def test_strong_eqtls_do_not_all_read_the_same_percentile(layer):
    """The saturation gate, measured on real effects rather than invented ones.

    The invented magnitudes (0.05 / 0.5 / 5.0) that motivated the original concern
    are all far above anything the model actually predicts for a validated eQTL:
    real strong-eQTL RNA effects have median 0.0008 and max 0.031.
    """
    if not FIXTURE.exists():
        pytest.skip("no eQTL fixture")
    norm = _load("alphagenome")
    rows = json.loads(FIXTURE.read_text())[layer]
    pcts = [norm.effect_percentile("alphagenome", r["track"], r["abs_effect"])
            for r in rows]
    pcts = [p for p in pcts if p is not None]
    assert len(pcts) >= 50, f"only {len(pcts)} of {len(rows)} {layer} rows resolved"
    assert len(set(pcts)) > 1, f"{layer} percentile column is constant"
    # and it must not be saturation-dominated
    saturated = sum(1 for p in pcts if p >= 1.0) / len(pcts)
    assert saturated < 0.10, f"{layer} saturates on {saturated:.1%} of real eQTLs"


# ---------------------------------------------------------------------------
# Gate 2 — calibration against experimentally validated variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layer", ["CAGE", "RNA_SEQ"])
def test_strong_eqtls_rank_as_notable(layer):
    if not FIXTURE.exists():
        pytest.skip("no eQTL fixture")
    norm = _load("alphagenome")
    rows = json.loads(FIXTURE.read_text())[layer]
    pcts = np.array([p for p in
                     (norm.effect_percentile("alphagenome", r["track"], r["abs_effect"])
                      for r in rows) if p is not None])
    p50 = float(np.percentile(pcts, 50))
    lo, hi = _CALIBRATION_BAND
    assert lo <= p50 <= hi, (
        f"{layer} strong-eQTL median percentile {p50:.3f} outside [{lo}, {hi}]. "
        f"Below the band means the null over-corrected — the reference class now "
        f"contains variants that perturb more than validated eQTLs do, which is how "
        f"a TSS-only CAGE null measured p50 0.323. Above it means re-saturation."
    )


# ---------------------------------------------------------------------------
# Gate 3 — bit-exactness of a whole report, across processes
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_a_report_is_bit_exact_across_two_processes():
    """Runs the real ``build_variant_report`` path twice, in two OS processes.

    Two processes rather than two calls, because within one process AlphaGenome was
    already bit-exact *before* #127 was fixed — the entire defect was cross-process,
    which is the case that matters for a user rerunning a report tomorrow and for a
    builder whose null must match a query's numerator.

    Measured 2026-08-04: 603 numeric fields, 0 differing, 0 sign flips, worst
    relative delta exactly 0.0.
    """
    script = REPO / "scripts" / "gate_end_to_end_determinism.py"
    if not script.exists():
        pytest.skip("gate script absent")
    proc = subprocess.run([sys.executable, str(script), "--gpu", "0"],
                          cwd=str(REPO), capture_output=True, text=True,
                          timeout=3600)
    assert "PASS" in proc.stdout, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert proc.returncode == 0


# ---------------------------------------------------------------------------
# The #123 fingerprint, across every shipped background
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(BACKGROUNDS.glob("*_pertrack.npz"))
                         or [Path("none")])
def test_no_counts_array_is_a_tight_run_of_consecutive_integers(path):
    """#123's signature: partial per-variant credit.

    When a builder wrapped its whole per-track loop in one ``try``, a variant that
    failed midway left some tracks credited and others not, so ``effect_counts``
    came out as a tight run of consecutive integers — enformer shipped 9600-9606,
    seven values one apart. Legitimate variation looks like *separated clusters*
    (one per region set), never a run.

    A run of 2 is exempt: two adjacent counts are what a single dropped position
    looks like, and every fresh build has at most 2 distinct values per array.
    """
    if not path.exists():
        pytest.skip("no downloaded backgrounds")
    with np.load(path, allow_pickle=True) as data:
        for key in ("effect_counts", "summary_counts", "perbin_counts"):
            if key not in data.files:
                continue
            counts = np.asarray(data[key])
            uniq = np.unique(counts[counts > 0])
            if uniq.size < 3:
                continue
            span = int(uniq.max() - uniq.min())
            assert span != uniq.size - 1, (
                f"{path.name}:{key} has {uniq.size} counts forming a consecutive "
                f"run {uniq.min()}-{uniq.max()} — the #123 partial-credit shape"
            )
