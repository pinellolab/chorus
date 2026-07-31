"""ChromBPNet count-head inversion: expm1, not exp.

ChromBPNet is trained against ``log(1 + count)``
(``chrombpnet/training/data_generators/batchgen_generator.py`` feeds
``np.log(1+batch_cts.sum(-1, keepdims=True))`` as the count target), so
recovering counts is :func:`numpy.expm1`.  Chorus used ``np.exp`` in
three places, which inflated every recovered count by exactly +1 —
negligible at a peak (~0.1% at 1,000 counts) but up to 100% at a
low-activity site, which is the regime the activity CDFs are built from.

Mirrors ``tests/test_cherimoya.py::test_expected_counts_uses_expm1_not_exp``.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

from chorus.oracles.chrombpnet import ChromBPNetOracle

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts" / "build_backgrounds_chrombpnet.py"

# ChromBPNet geometry; see ChromBPNetOracle.__init__.
INPUT_LENGTH = 2114
OUTPUT_LENGTH = 1000
INSERTION_START = (INPUT_LENGTH - OUTPUT_LENGTH) // 2


def _stub():
    """A minimal self for the pure-arithmetic transforms: no model, no env."""
    stub = SimpleNamespace(
        sequence_length=INPUT_LENGTH, output_length=OUTPUT_LENGTH
    )
    stub._insert_into_output = ChromBPNetOracle._insert_into_output.__get__(stub)
    stub._counts_from_log = ChromBPNetOracle._counts_from_log
    return stub


def _transform(counts, n_positions=OUTPUT_LENGTH, seq_len=INPUT_LENGTH):
    """Call the transform with a stub self, so no model/env is needed."""
    stub = _stub()
    probabilities = numpy.zeros((1, n_positions))  # uniform after softmax
    return ChromBPNetOracle._transform_predictions_to_tracks(
        stub, probabilities, numpy.asarray(counts, dtype=numpy.float64), seq_len
    )


def _profile(out):
    """The 1000 bp ChromBPNet window inside the padded output array."""
    return out[INSERTION_START:INSERTION_START + OUTPUT_LENGTH]


def test_transform_uses_expm1_not_exp():
    """The count head predicts log(count + 1); the inverse is expm1."""
    out = _transform([[numpy.log(101.0)]])  # log(100 + 1)

    profile = _profile(out)
    numpy.testing.assert_allclose(profile.sum(), 100.0, rtol=1e-9)
    numpy.testing.assert_allclose(profile, 100.0 / OUTPUT_LENGTH, rtol=1e-9)

    # exp() would have given 101 -- assert we are not doing that.
    assert not numpy.isclose(profile.sum(), 101.0)


def test_transform_low_activity_site_is_not_inflated():
    """Where exp() hurt most: a 1-count site was reported as 2."""
    out = _transform([[numpy.log1p(1.0)]])

    numpy.testing.assert_allclose(_profile(out).sum(), 1.0, rtol=1e-9)


@pytest.mark.parametrize("count", [0.0, 1.0, 5.0, 100.0, 10_000.0])
def test_transform_round_trips_log1p(count):
    """expm1(log1p(c)) == c across the dynamic range, including zero."""
    out = _transform([[numpy.log1p(count)]])

    numpy.testing.assert_allclose(_profile(out).sum(), count, atol=1e-8)


def test_predict_sliding_inverts_counts_with_expm1():
    """Guard the sliding-window site, which needs a live TF model to hit.

    ``predict_sliding`` scales its softmaxed profile by the count head
    inside a batch loop, so a behavioural test would require TensorFlow
    plus weights.  Assert on the source instead: the profile softmax
    legitimately uses ``np.exp``, the count inversion must not.
    """
    src = inspect.getsource(ChromBPNetOracle.predict_sliding)

    assert "np.expm1(counts[b][0])" in src
    assert "np.exp(counts" not in src


def test_builder_and_oracle_invert_counts_identically():
    """A CDF is only meaningful if built the way predict() computes.

    The background builder and the oracle must use the same count
    inversion; if they drift, ChromBPNet percentiles are silently wrong.
    See chorus/oracles/cherimoya_source/scoring.py for why this is
    load-bearing.
    """
    builder_src = BUILDER.read_text()

    # Single-track models keep expm1; multi-track subtract n_strands. A bare
    # exp(counts) with nothing subtracted would be upstream's own predict.py
    # bug (off by +n_tracks), so require the guarded form.
    assert "np.expm1(counts[:, 0:1]) if n_strands == 1" in builder_src
    assert "np.exp(counts[:, 0:1]) - n_strands" in builder_src

    oracle_src = Path(inspect.getfile(ChromBPNetOracle)).read_text()
    # The oracle routes every inversion through the shared helper rather than
    # inlining exp/expm1 at each call site.
    assert "_counts_from_log" in oracle_src
    assert "np.exp(counts) - n_tracks" in oracle_src


# ── CHIP two-strand handling ─────────────────────────────────────────

def _chip_strands(counts, n_positions=OUTPUT_LENGTH, seq_len=INPUT_LENGTH):
    """Call the CHIP strand split with a stub self (no model/env needed)."""
    stub = _stub()
    probabilities = numpy.zeros((1, n_positions, 2))  # uniform after softmax
    return ChromBPNetOracle._transform_chip_strands(
        stub, probabilities, numpy.asarray(counts, dtype=numpy.float64), seq_len
    )


def _chip_log_target(total_counts, n_tracks=2, split=None):
    """Build the count target the way bpnet-refactor trains it.

    The generator stores a PER-TRACK log1p (generators.py: log(sum over
    positions + 1)) and the count loss pools a task's tracks with
    reduce_logsumexp (custommodel.py:57), so

        C = log( sum_t (1 + c_t) ) = log(n_tracks + total_counts)
    """
    if split is None:
        split = [total_counts / n_tracks] * n_tracks
    per_track = numpy.log1p(numpy.asarray(split, dtype=numpy.float64))
    return float(numpy.log(numpy.exp(per_track).sum()))


def test_chip_log_target_helper_matches_the_closed_form():
    """Sanity-check the helper: C == log(n_tracks + total)."""
    for n, c in ((1, 100.0), (2, 100.0), (2, 3.0)):
        assert _chip_log_target(c, n) == pytest.approx(
            float(numpy.log(n + c)), abs=1e-12
        )


def test_chip_strands_together_carry_the_predicted_counts():
    """Two things at once: the joint split, and the right inverse.

    The count target is a per-track log1p pooled across strands with
    logsumexp, i.e. log(n_tracks + total), so the inverse is
    exp(C) - n_tracks. `expm1` — right for the single-track ATAC/DNASE
    models — leaves exactly ONE read of inflation on these two-track CHIP
    models (median 1.78x on background 501bp window sums).

    And because the target pools both strands, the two emitted tracks must
    SUM to that total rather than each carrying it; per-strand scaling by the
    full total measured exactly 2.00x on a real BPNet model.
    """
    C = _chip_log_target(100.0)                 # log(2 + 100) = log(102)
    plus, minus = _chip_strands([[C]])

    total = _profile(plus).sum() + _profile(minus).sum()
    numpy.testing.assert_allclose(total, 100.0, rtol=1e-9)

    # Uniform logits ⇒ the mass splits evenly between the strands.
    numpy.testing.assert_allclose(_profile(plus).sum(), 50.0, rtol=1e-9)
    numpy.testing.assert_allclose(_profile(minus).sum(), 50.0, rtol=1e-9)

    # expm1 would have recovered 101, and per-strand would have put the whole
    # total on each track.
    assert not numpy.isclose(total, float(numpy.expm1(C)))
    assert not numpy.isclose(_profile(plus).sum(), 100.0)


@pytest.mark.parametrize("count", [0.0, 1.0, 5.0, 100.0, 10_000.0])
def test_chip_strand_split_conserves_counts_across_the_range(count):
    plus, minus = _chip_strands([[_chip_log_target(count)]])

    total = _profile(plus).sum() + _profile(minus).sum()
    numpy.testing.assert_allclose(total, count, atol=1e-7)


def test_single_track_inverse_is_unchanged_expm1():
    """The 42 ATAC/DNASE rows must not move: exp(C) - 1 IS expm1(C).

    Keeping expm1 for n_tracks == 1 is both bit-identical and more accurate
    for small C, so PR #113's fix and the shipped ATAC/DNASE CDF rows stay
    exactly valid.
    """
    for c in (0.0, 1.0, 5.0, 100.0, 10_000.0):
        C = numpy.asarray([[_chip_log_target(c, n_tracks=1)]])
        got = ChromBPNetOracle._counts_from_log(C, 1)
        numpy.testing.assert_allclose(got, [[c]], atol=1e-8)
        numpy.testing.assert_allclose(got, numpy.expm1(C), rtol=0, atol=0)


def test_two_track_inverse_removes_exactly_one_read():
    for c in (1.0, 5.0, 100.0, 10_000.0):
        C = numpy.asarray([[_chip_log_target(c, n_tracks=2)]])
        numpy.testing.assert_allclose(
            ChromBPNetOracle._counts_from_log(C, 2), [[c]], atol=1e-7
        )
        # expm1 overstates by exactly one read
        numpy.testing.assert_allclose(numpy.expm1(C), [[c + 1.0]], atol=1e-7)


def test_chip_strand_split_respects_asymmetric_logits():
    """A strand with higher logits must receive proportionally more mass."""
    stub = _stub()
    probabilities = numpy.zeros((1, OUTPUT_LENGTH, 2))
    probabilities[0, :, 0] = numpy.log(3.0)  # plus strand 3x as likely

    plus, minus = ChromBPNetOracle._transform_chip_strands(
        stub, probabilities, numpy.array([[_chip_log_target(100.0)]]), INPUT_LENGTH
    )
    p, m = _profile(plus).sum(), _profile(minus).sum()

    numpy.testing.assert_allclose(p + m, 100.0, rtol=1e-9)
    numpy.testing.assert_allclose(p / m, 3.0, rtol=1e-9)


def test_builder_uses_the_same_joint_softmax_as_the_oracle():
    """A CDF is only meaningful if built from what predict() returns.

    The builder used to sum the strand logits before one softmax — a
    different quantity from either strand, and sequence-dependently so.
    """
    builder_src = BUILDER.read_text()

    assert "probabilities.sum(axis=-1)" not in builder_src
    assert "length * n_strands" in builder_src
    assert "for strand in range(" in builder_src


# ── BPNet bias-input shapes ──────────────────────────────────────────

TEMPLATE = (
    REPO_ROOT / "chorus" / "oracles" / "chrombpnet_source"
    / "templates" / "predict_template.py"
)


class _FakeInput:
    def __init__(self, shape):
        self.shape = shape


def test_zero_bias_inputs_match_the_models_declared_shapes():
    """The count bias is (None, 2); a hardcoded (N, 1) is silently broadcast.

    BPNet reduces the count-bias input with a logsumexp before concatenating
    it with the count head, so feeding width 1 instead of 2 makes that term
    log(1)=0 rather than log(2), shifting every predicted log-count down by a
    constant 0.849035 * log(2) = 0.5885 — counts 1.80x too low at a peak and
    up to 3.04x at a quiet site. The CDF builder derives these shapes, so the
    oracle silently disagreed with its own backgrounds.
    """
    stub = SimpleNamespace(
        model=SimpleNamespace(
            inputs=[
                _FakeInput((None, 2114, 4)),
                _FakeInput((None, 1000, 2)),
                _FakeInput((None, 2)),
            ]
        )
    )
    bias = ChromBPNetOracle._zero_bias_inputs(stub, 7)

    assert [b.shape for b in bias] == [(7, 1000, 2), (7, 2)]
    assert all((b == 0).all() for b in bias)


def test_no_path_hardcodes_the_count_bias_width():
    """All three CHIP predict paths must derive the bias shapes."""
    oracle_src = Path(inspect.getfile(ChromBPNetOracle)).read_text()
    template_src = TEMPLATE.read_text()

    for src, label in ((oracle_src, "chrombpnet.py"), (template_src, "predict_template.py")):
        assert "np.zeros((num_windows, 1)" not in src, label
        assert "np.zeros((B, 1)" not in src, label
        assert "inp.shape[1:]" in src, label


def test_predict_sliding_uses_the_joint_strand_softmax():
    """predict_sliding must not disagree with _predict for CHIP.

    It softmaxed the *summed logits*, a third quantity again. It now takes the
    joint softmax and sums the strands, so the sliding track is the
    per-position both-strand total and integrates to expm1(counts).
    """
    src = inspect.getsource(ChromBPNetOracle.predict_sliding)

    assert "flat = p.reshape(-1)" in src
    assert "p.sum(axis=-1)" not in src
    assert "self._zero_bias_inputs(B)" in src
