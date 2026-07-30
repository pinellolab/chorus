"""Integration tests for the Cherimoya / CATv1 oracle.

These run from the **base chorus environment** and drive the oracle through
its isolated ``chorus-cherimoya`` environment, matching how
``test_smoke_predict.py`` exercises the other oracles.  They are marked
``integration`` (see ``pytest.ini``) so the fast suite stays offline, and
they need:

  * ``chorus setup --oracle cherimoya`` to have been run, and
  * a GRCh38 FASTA at ``genomes/hg38.fa``.

Run with:

    pytest tests/test_cherimoya_integration.py -v -m integration

The load-bearing test is
:func:`test_predict_matches_direct_window_scoring` — it asserts the 501 bp
window sum taken from ``oracle.predict()`` equals the one computed straight
from the model's head outputs via ``cherimoya_source.scoring``.  That is
the invariant the background CDFs rest on: a CDF is only meaningful if the
value it was built from is computed the way the query path computes it.
Chorus has been bitten by the opposite before (the pre-0.4 ChromBPNet CDF
rebuild), so it is pinned before the builder exists rather than after.

Note that each prediction in environment mode spawns a subprocess that
re-imports torch and reloads the checkpoint, so this file is slow by
construction — that is a property of the env-isolation design, not of
Cherimoya.
"""

import os

import numpy
import pytest

from chorus.oracles.cherimoya import CherimoyaOracle
from chorus.oracles.cherimoya_source.catv1_globals import (
    CATV1_INPUT_LENGTH,
    CATV1_OUTPUT_LENGTH,
    CATV1_TRIMMING,
)
from chorus.oracles.cherimoya_source.scoring import (
    expected_counts_profile,
    score_window_sum,
)

pytestmark = pytest.mark.integration

REFERENCE_FASTA = "genomes/hg38.fa"

# One of the nine experiments ChromBPNet also covers, so this doubles as
# the cross-oracle comparison track.
TEST_ACCESSION = "ENCSR000EOT"   # DNase, K562
TEST_TRACK = f"DNASE:{TEST_ACCESSION}"
TEST_REGION = ("chr1", 1_000_000, 1_000_000 + CATV1_INPUT_LENGTH)


def _require_prerequisites():
    """Skip with an actionable message rather than failing obscurely."""
    from chorus.core.environment import EnvironmentManager

    if not os.path.exists(REFERENCE_FASTA):
        pytest.skip(
            f"{REFERENCE_FASTA} not found — run `chorus genome get hg38` or "
            f"symlink a GRCh38 FASTA there."
        )
    if not EnvironmentManager().environment_exists("cherimoya"):
        pytest.skip(
            "chorus-cherimoya environment missing — run "
            "`chorus setup --oracle cherimoya`."
        )


@pytest.fixture(scope="module")
def oracle():
    _require_prerequisites()
    # Device is pinned rather than auto-detected. Cherimoya's Triton
    # kernels and its pure-PyTorch CPU fallback agree only to ~1e-2 on the
    # profile logits, so if auto-detect resolved to CUDA for one call and
    # CPU for another (a momentarily busy GPU is enough), the
    # builder-vs-predict equality test below would fail for reasons
    # unrelated to what it is testing. The same argument applies to the
    # background build: pin the device there too.
    oracle = CherimoyaOracle(
        use_environment=True,
        reference_fasta=REFERENCE_FASTA,
        device=_pinned_device(),
    )
    oracle.load_pretrained_model(assay="DNASE", cell_type="K562")
    return oracle


def _pinned_device() -> str:
    """Resolve a device once, in the oracle env, and pin it for the module."""
    import subprocess

    from chorus.core.environment import EnvironmentManager

    python_exe = EnvironmentManager().get_python_executable("cherimoya")
    probe = subprocess.run(
        [python_exe, "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True, text=True, timeout=300,
    )
    return "cuda" if probe.stdout.strip() == "True" else "cpu"


@pytest.fixture(scope="module")
def reference_sequence():
    _require_prerequisites()
    import pysam

    chrom, start, end = TEST_REGION
    with pysam.FastaFile(REFERENCE_FASTA) as fasta:
        sequence = fasta.fetch(chrom, start, end).upper()
    assert len(sequence) == CATV1_INPUT_LENGTH
    return sequence


@pytest.fixture(scope="module")
def prediction_values(oracle):
    """One prediction, reused — each env-mode call is a fresh subprocess."""
    return oracle.predict(TEST_REGION)[TEST_TRACK].values


# ── loading ──────────────────────────────────────────────────────────

def test_default_resolution_is_the_chrombpnet_matched_experiment(oracle):
    assert oracle.encode_id == TEST_ACCESSION
    assert oracle.assay == "DNASE"
    assert oracle.cell_type == "K562"
    assert oracle.track_id == TEST_TRACK
    assert oracle.fold == 0
    assert oracle.loaded


def test_loaded_checkpoint_reports_catv1_geometry(oracle):
    """The load template echoes geometry back; _check_geometry gates on it."""
    info = oracle._model_info
    assert info["trimming"] == CATV1_TRIMMING
    assert info["n_control_tracks"] == 0
    assert info["signal_groups"] == [1]
    assert info["n_parameters"] == 613892


# ── prediction ───────────────────────────────────────────────────────

def test_predict_returns_finite_values(oracle, prediction_values):
    track = oracle.predict(TEST_REGION)[TEST_TRACK]

    assert numpy.all(numpy.isfinite(track.values))
    assert numpy.all(track.values >= 0), "accessibility is non-negative"
    assert track.values.sum() > 0, "prediction is not identically zero"
    assert track.assay_type == "DNASE"
    assert track.cell_type == "K562"
    assert track.resolution == 1
    assert track.metadata["encode_id"] == TEST_ACCESSION
    assert track.metadata["fold"] == 0


def test_predicted_window_lands_at_the_trim_offset(prediction_values):
    """Only the central output_length bases should carry signal."""
    assert len(prediction_values) == CATV1_INPUT_LENGTH
    assert prediction_values[:CATV1_TRIMMING].sum() == 0.0
    assert prediction_values[CATV1_TRIMMING + CATV1_OUTPUT_LENGTH:].sum() == 0.0
    assert prediction_values[CATV1_TRIMMING:CATV1_TRIMMING + CATV1_OUTPUT_LENGTH].sum() > 0


def test_predict_matches_direct_window_scoring(
    oracle, reference_sequence, prediction_values,
):
    """The invariant the background CDFs rest on.

    Scoring the prediction the way the builder will (501 bp central window
    sum) must equal scoring the raw head outputs through the shared
    helper.  If these diverge, every percentile is silently wrong.
    """
    # Path A: through the oracle's public predict().
    profile_from_predict = prediction_values[
        CATV1_TRIMMING:CATV1_TRIMMING + CATV1_OUTPUT_LENGTH
    ]
    via_predict = score_window_sum(profile_from_predict)

    # Path B: straight from the head outputs, as the builder will do.
    logits, log_counts = oracle._forward_windows([reference_sequence])
    profile_direct = expected_counts_profile(logits, log_counts)[0]
    via_scoring = score_window_sum(profile_direct)

    # Both calls must have run on the same device, or the comparison is
    # meaningless -- Triton and the CPU fallback differ by ~1e-2.
    assert oracle._last_device == _pinned_device()

    numpy.testing.assert_allclose(
        profile_from_predict, profile_direct, rtol=1e-5, atol=1e-6,
    )
    assert via_predict == pytest.approx(via_scoring, rel=1e-6)
    assert via_predict > 0


def test_counts_are_recovered_with_expm1(oracle, reference_sequence):
    """Total predicted counts must equal expm1(log_counts), not exp()."""
    logits, log_counts = oracle._forward_windows([reference_sequence])
    profile = expected_counts_profile(logits, log_counts)[0]

    expected = float(numpy.expm1(log_counts[0][0]))
    wrong = float(numpy.exp(log_counts[0][0]))

    assert profile.sum() == pytest.approx(expected, rel=1e-6)
    # The two differ by exactly 1.0 count; assert we picked the right one.
    assert wrong - expected == pytest.approx(1.0, abs=1e-2)


def test_variant_effect_runs_end_to_end(oracle):
    """predict_variant_effect through the base-class machinery."""
    import pysam

    chrom = "chr1"
    position = 1_000_000 + CATV1_INPUT_LENGTH // 2
    with pysam.FastaFile(REFERENCE_FASTA) as fasta:
        ref_base = fasta.fetch(chrom, position - 1, position).upper()
    alt_base = "A" if ref_base != "A" else "G"

    result = oracle.predict_variant_effect(
        genomic_region=f"{chrom}:{position}-{position}",
        variant_position=f"{chrom}:{position}",
        alleles=[ref_base, alt_base],
        assay_ids=None,
    )

    assert "predictions" in result and "effect_sizes" in result
    effects = result["effect_sizes"]["alt_1"][TEST_TRACK]
    assert numpy.all(numpy.isfinite(effects))
    # A single substitution should perturb the profile, not leave it identical.
    assert numpy.abs(effects).sum() > 0


def test_predict_sliding_covers_the_query(oracle):
    """Wide-region stitching returns exactly the query span."""
    width = 3 * CATV1_OUTPUT_LENGTH
    region = ("chr1", 1_000_000, 1_000_000 + width)

    values = oracle.predict_sliding(region)[TEST_TRACK].values

    assert len(values) == width
    assert numpy.all(numpy.isfinite(values))
    assert numpy.all(values >= 0)
    assert values.sum() > 0


def test_describe_track_recovers_biosample(oracle):
    """Track ids carry no biosample, so this is how it is recovered."""
    row = oracle.describe_track(oracle.track_id)
    assert row["biosample"] == "K562"
    assert row["assay"] == "DNASE"
    assert row["assembly"] == "GRCh38"
    assert 0 <= row["count_pearson"] <= 1


def test_geometry_guard_rejects_a_mismatched_checkpoint(tmp_path):
    """_check_geometry must fail loudly rather than emit misaligned tracks.

    Needs cherimoya locally to build a wrong-shaped checkpoint, so it is
    skipped when running from the base environment.
    """
    _require_prerequisites()
    cherimoya = pytest.importorskip(
        "cherimoya", reason="needs cherimoya importable to forge a checkpoint",
    )
    from chorus.core.exceptions import ModelNotLoadedError

    # Fewer layers -> smaller receptive field -> different trimming.
    small = cherimoya.Cherimoya(n_filters=16, n_layers=2)
    path = tmp_path / "wrong_geometry.torch"
    small.save(str(path))

    oracle = CherimoyaOracle(use_environment=False)
    with pytest.raises(ModelNotLoadedError, match="trimming"):
        oracle.load_pretrained_model(assay="DNASE", weights=str(path))
