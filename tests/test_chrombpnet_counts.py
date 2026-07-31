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


def _transform(counts, n_positions=OUTPUT_LENGTH, seq_len=INPUT_LENGTH):
    """Call the transform with a stub self, so no model/env is needed."""
    stub = SimpleNamespace(
        sequence_length=INPUT_LENGTH, output_length=OUTPUT_LENGTH
    )
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

    assert "np.expm1(counts[:, 0:1])" in builder_src
    assert "np.exp(counts" not in builder_src

    oracle_src = Path(inspect.getfile(ChromBPNetOracle)).read_text()
    assert "np.exp(counts" not in oracle_src
