"""Every copy of the count-head arithmetic must produce the same numbers (#125).

Three defects fixed on 2026-07-31 were all one shape: **two copies of the same four
operations disagreeing**, with nothing comparing them.

    exp vs expm1 on the count head        4 call sites, fixed one at a time
                                          +1 read: ~0.1% at a peak, up to 100% at a quiet
                                          site, which is where the activity CDFs live
    per-strand vs joint softmax           the two emitted tracks together claimed 2.00x the
                                          predicted counts (BPNet CHIP:K562:REST)
    count bias hardcoded (N, 1)           Keras broadcast it silently; every log-count
                                          shifted by 0.5885, i.e. 1.80x low at a peak and
                                          3.04x at a quiet site

None crashed. Each produced a plausible number and each shipped, and the reason all three
were possible is that the arithmetic existed in five places at once.

This file is the instrument the extraction is verified against. It does **not** grep for the
formula — a source assertion cannot tell you whether two implementations agree, only whether
they look alike, and the ``exp``/``expm1`` bug lived through exactly that kind of check. It
feeds identical inputs to every copy and compares outputs.

Two copies stay outside the shared module on purpose, and this file is what makes that safe
rather than merely stated:

* the **torch** path in ``build_backgrounds_cherimoya.py`` runs inside the batch loop on the
  accelerator, and routing it through numpy would add a device round-trip per batch to a job
  measured in hours;
* **EPInformer-seq** scales by ``10 ** log_count``, not ``expm1``, because it is a different
  model trained against a different target. Sweeping it into a "unification" would be the
  same class of mistake as the three above, so the difference is pinned below.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chorus.core.count_head import (
    counts_from_log,
    expected_counts_profile,
    joint_softmax,
)

REPO = Path(__file__).resolve().parent.parent

# ChromBPNet geometry, matching ChromBPNetOracle.__init__.
INPUT_LENGTH = 2114
OUTPUT_LENGTH = 1000
INSERT_START = (INPUT_LENGTH - OUTPUT_LENGTH) // 2


@pytest.fixture
def heads():
    """One reproducible (logits, log_counts) pair, shaped as the models emit them."""
    rng = np.random.default_rng(20260812)
    logits = rng.normal(0.0, 2.0, size=(3, OUTPUT_LENGTH))
    # Spread over the range that matters: a quiet site, a middling one, a peak.
    log_counts = np.array([[0.05], [2.5], [7.0]])
    return logits, log_counts


# ──────────────────────────────────────────────────────────────────────
# The canonical implementation, against the closed form
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("total", [0.0, 0.5, 1.0, 10.0, 1_000.0, 100_000.0])
def test_the_single_track_inverse_round_trips_log1p(total):
    """``expm1(log1p(c)) == c`` is the whole claim for the 1,560 single-track models."""
    assert counts_from_log(np.log1p(total), 1) == pytest.approx(total, rel=1e-12, abs=1e-9)


@pytest.mark.parametrize("total", [0.0, 1.0, 10.0, 1_000.0])
def test_the_two_track_inverse_removes_exactly_one_read(total):
    """The CHIP convention, and the reason ``n_tracks`` exists.

    ``C = log(sum_t (1 + c_t))`` for a two-track task, so the inverse subtracts 2. Using
    ``expm1`` there leaves one whole read of inflation — measured median 1.78x on background
    501 bp window sums, up to 4.17x at quiet sites.
    """
    per_track = np.log1p(np.array([total / 2, total / 2]))
    pooled = float(np.log(np.exp(per_track).sum()))
    assert counts_from_log(pooled, 2) == pytest.approx(total, rel=1e-9, abs=1e-9)
    if total > 0:
        left_over = counts_from_log(pooled, 1) - total
        assert left_over == pytest.approx(1.0, rel=1e-6), (
            "expm1 on a two-track target should leave exactly one read behind"
        )


def test_the_profile_sums_to_the_predicted_total(heads):
    """The invariant that makes a track "expected counts" rather than a shape."""
    logits, log_counts = heads
    profile = expected_counts_profile(logits, log_counts)
    for row, expected in zip(profile, counts_from_log(log_counts[:, 0], 1)):
        assert row.sum() == pytest.approx(expected, rel=1e-9)


def test_two_strands_share_one_distribution(heads):
    """Joint, not per-strand: together the strands carry the total exactly once."""
    logits, log_counts = heads
    two = np.stack([logits, logits * 0.5], axis=-1)          # (B, L, 2), asymmetric
    profile = expected_counts_profile(two, log_counts)
    totals = counts_from_log(log_counts[:, 0], 2)
    for row, expected in zip(profile, totals):
        assert row.sum() == pytest.approx(expected, rel=1e-9)
    # And the strands must differ -- a bug that symmetrises them would pass the sum check.
    assert not np.allclose(profile[..., 0], profile[..., 1])


def test_centring_the_logits_does_not_change_the_answer(heads):
    """Centring exists for overflow, not for the result; a shift must cancel."""
    logits, log_counts = heads
    a = expected_counts_profile(logits, log_counts)
    b = expected_counts_profile(logits + 137.0, log_counts)
    assert np.allclose(a, b, rtol=1e-12, atol=0)


# ──────────────────────────────────────────────────────────────────────
# Copy vs copy
# ──────────────────────────────────────────────────────────────────────

def _stub_oracle():
    """A ChromBPNetOracle shell: these transforms need no model and no environment.

    Same construction as ``tests/test_chrombpnet_counts.py::_stub`` -- bound methods rather
    than lambdas, because ``_insert_into_output`` reads both geometry attributes off self.
    """
    from chorus.oracles.chrombpnet import ChromBPNetOracle

    stub = SimpleNamespace(sequence_length=INPUT_LENGTH, output_length=OUTPUT_LENGTH)
    stub._insert_into_output = ChromBPNetOracle._insert_into_output.__get__(stub)
    stub._counts_from_log = ChromBPNetOracle._counts_from_log
    return ChromBPNetOracle, stub


# The oracle transforms flatten the batch into one track (``_insert_into_output``), so they
# are batch-of-one in production. Comparing row by row rather than reshaping keeps the test
# honest about what the production path actually computes.

def test_the_chrombpnet_oracle_agrees_with_the_shared_helper(heads):
    """``_transform_predictions_to_tracks`` is the DNASE/ATAC production path."""
    logits, log_counts = heads
    cls, stub = _stub_oracle()

    for row, count in zip(logits, log_counts):
        got = cls._transform_predictions_to_tracks(
            stub, row[None, :], count[None, :], INPUT_LENGTH)
        want = expected_counts_profile(row[None, :], count[None, :], n_tracks=1)[0]
        window = np.asarray(got)[INSERT_START:INSERT_START + OUTPUT_LENGTH]
        assert np.allclose(window, want, rtol=1e-12, atol=0), (
            f"the oracle's inline arithmetic has drifted from chorus.core.count_head "
            f"at log_count={float(count[0]):.3f}"
        )


def test_the_chip_strand_split_agrees_with_the_shared_helper(heads):
    """``_transform_chip_strands`` is the two-strand production path."""
    logits, log_counts = heads
    cls, stub = _stub_oracle()

    for row, count in zip(logits, log_counts):
        two = np.stack([row, row * 0.5], axis=-1)[None, ...]     # (1, L, 2), asymmetric
        plus, minus = cls._transform_chip_strands(stub, two, count[None, :], INPUT_LENGTH)
        want = expected_counts_profile(two, count[None, :], n_tracks=2)[0]
        for track, emitted in enumerate((plus, minus)):
            window = np.asarray(emitted)[INSERT_START:INSERT_START + OUTPUT_LENGTH]
            assert np.allclose(window, want[:, track], rtol=1e-12, atol=0), (
                f"strand {track} disagrees at log_count={float(count[0]):.3f}"
            )


def test_cherimoyas_scoring_module_agrees_with_the_shared_helper(heads):
    """CATv1's copy, which is where the ``expm1`` lesson was first written down."""
    from chorus.oracles.cherimoya_source.scoring import expected_counts_profile as catv1

    logits, log_counts = heads
    assert np.allclose(catv1(logits, log_counts),
                       expected_counts_profile(logits, log_counts, n_tracks=1),
                       rtol=1e-12, atol=0)


def _function_from_script(script: str, name: str):
    """Compile ONE function out of a builder script, without importing the script.

    The builders cannot be imported at all. ``build_backgrounds_chrombpnet.py`` parses
    ``sys.argv`` at module level *and* runs its ``scope_violations`` preflight there, so an
    import in the wrong env either dies on ``ModuleNotFoundError: tensorflow`` or — in the
    right env — correctly refuses with "planning 9 tracks against 753 in the shipped
    background" and exits. Both are the builder behaving properly; neither lets a test read
    it.

    So the function is lifted from the AST and compiled on its own. That runs the **shipped**
    code rather than a copy of it, which is the whole point: a test that reimplements the
    thing it is checking proves nothing.
    """
    import ast

    path = REPO / "scripts" / script
    if not path.exists():
        pytest.skip(f"{script} not present")
    tree = ast.parse(path.read_text())
    node = next((n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if node is None:
        pytest.fail(f"{script} no longer defines {name}()")
    namespace: dict = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_the_chrombpnet_builder_agrees_with_the_oracle(heads):
    """A CDF is only meaningful if it was built from what ``predict()`` returns.

    Executed, not grepped. The builder and the oracle used to hold line-for-line copies of
    this arithmetic and were compared by asserting that both files contained matching source
    strings — which is precisely the check that ``exp`` vs ``expm1`` walked past four times.
    """
    profiles_from_heads = _function_from_script(
        "build_backgrounds_chrombpnet.py", "profiles_from_heads")

    logits, log_counts = heads
    for n_strands in (1, 2):
        stacked = (logits[..., None] if n_strands == 1
                   else np.stack([logits, logits * 0.5], axis=-1))
        got = profiles_from_heads(stacked, log_counts)
        want = expected_counts_profile(stacked, log_counts, n_tracks=n_strands)
        assert np.array_equal(got, want), (
            f"the ChromBPNet builder and the oracle disagree at n_strands={n_strands}; "
            f"every percentile from this oracle is then ranked against the wrong null"
        )


def test_the_torch_path_agrees_with_numpy(heads):
    """The one copy that legitimately stays duplicated, held to the same numbers.

    ``build_backgrounds_cherimoya.py`` does this on the accelerator inside the batch loop;
    calling numpy there would add a device round-trip per batch to a multi-hour job. The
    expression is reproduced here from that file, and float32 is why the tolerance is 1e-6
    rather than 0.

    Skips in the base env, which has no torch, so it does not run in the fast suite. Verified
    2026-08-12 in ``chorus-cherimoya`` (torch 2.13.0+cu130): **max relative difference
    1.25e-07**. Run it there when touching either side:

        /home/nvidia/miniforge3/envs/chorus-cherimoya/bin/python -m pytest \\
            tests/test_count_head_copies_agree.py -k torch
    """
    torch = pytest.importorskip(
        "torch", reason="no torch in the base env; run this in chorus-cherimoya")

    logits, log_counts = heads
    t_logits = torch.tensor(logits, dtype=torch.float32)
    t_counts = torch.tensor(log_counts[:, 0], dtype=torch.float32)

    # Exactly the lines in the builder's batch loop.
    probs = torch.softmax(t_logits - t_logits.mean(dim=1, keepdim=True), dim=1)
    counts = torch.expm1(t_counts)
    got = (probs * counts[:, None]).numpy()

    want = expected_counts_profile(logits, log_counts, n_tracks=1)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-6), (
        "the accelerator path in the cherimoya builder no longer matches "
        "chorus.core.count_head; the shipped CATv1 CDFs would rank against a different "
        "quantity than the oracle emits"
    )


# ──────────────────────────────────────────────────────────────────────
# The extraction moved no numbers
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_shared_helper_is_bit_identical_to_the_pre_extraction_expressions(dtype):
    """The claim that makes #125 a refactor rather than a release: **no number moved.**

    The pre-#125 expressions are written out below verbatim, from
    ``ChromBPNetOracle._transform_predictions_to_tracks`` and ``_transform_chip_strands`` as
    they stood at ee3325d. Not ``allclose`` — ``array_equal``. Same operations in the same
    order on the same dtype is bit-identical arithmetic, and that is what lets the shipped
    CDFs and every committed example stay untouched.

    Both dtypes matter because the two copies disagreed about precision: Cherimoya cast to
    float64 first, ChromBPNet ran on whatever TensorFlow returned. The shared helper
    preserves the caller's dtype for exactly that reason, so this must hold in each.
    """
    rng = np.random.default_rng(7)
    logits = rng.normal(0.0, 2.0, size=(2, OUTPUT_LENGTH)).astype(dtype)
    counts = np.array([[0.05], [6.0]], dtype=dtype)

    # -- verbatim, the DNASE/ATAC path before the extraction
    norm = logits - np.mean(logits, axis=1, keepdims=True)
    softmax_probs = np.exp(norm) / np.sum(np.exp(norm), axis=1, keepdims=True)
    old_dnase = softmax_probs * (np.expand_dims(np.expm1(counts)[:, 0], axis=1))
    assert np.array_equal(old_dnase, expected_counts_profile(logits, counts, n_tracks=1))

    # -- verbatim, the two-strand CHIP path before the extraction
    two = np.stack([logits, logits * 0.5], axis=-1)
    flat = two.reshape(two.shape[0], -1)
    norm = flat - np.mean(flat, axis=1, keepdims=True)
    joint = np.exp(norm) / np.sum(np.exp(norm), axis=1, keepdims=True)
    totals = np.exp(counts) - 2
    old_chip = (joint * np.expand_dims(totals[:, 0], axis=1)).reshape(two.shape[0], -1, 2)
    assert np.array_equal(old_chip, expected_counts_profile(two, counts, n_tracks=2))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_helper_preserves_the_callers_precision(dtype):
    """Promoting float32 to float64 here would move every ChromBPNet number."""
    logits = np.zeros((1, 8), dtype=dtype)
    out = expected_counts_profile(logits, np.array([[1.0]], dtype=dtype), n_tracks=1)
    assert out.dtype == dtype, (
        "the shared helper changed the precision of the arithmetic; whichever direction, "
        "one of ChromBPNet or Cherimoya just had every value shifted in its last bits"
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cherimoya_computes_in_double_precision_whatever_it_is_handed(dtype):
    """The other half of the precision contract, and it cost a real drift to find.

    CATv1 has always run this in float64 — its own wrapper cast both arrays before doing
    anything. When the shared helper started preserving the caller's dtype (so ChromBPNet's
    float32 stayed float32), Cherimoya's ``log_counts`` stopped being promoted, and the SORT1
    example's ``ref_value`` moved from 603.3464052301788 to 603.3464123072064 — 1.2e-8, small
    and completely unjustified. The unit tests missed it because they pass float64 fixtures;
    only regenerating the example caught it. Hence this test.
    """
    from chorus.oracles.cherimoya_source.scoring import expected_counts_profile as catv1

    out = catv1(np.zeros((1, 16), dtype=dtype), np.array([[2.5]], dtype=dtype))
    assert out.dtype == np.float64, (
        "CATv1 must promote to float64 before delegating, whatever dtype the model returned"
    )


# ──────────────────────────────────────────────────────────────────────
# The convention that must NOT be unified
# ──────────────────────────────────────────────────────────────────────

def test_epinformerseq_uses_a_different_convention_on_purpose():
    """``10 ** log_count``, not ``expm1``. Pinned so a tidy-up cannot quietly merge it.

    EPInformer-seq is a different model trained against a different count target. At a
    log-count of 2.5 the two conventions differ by 26x, so a "unification" here would not
    be a rounding difference — it would silently rescale every EPInformer activity value.
    """
    source = (REPO / "chorus" / "oracles" / "epinformerseq_source" / "model_usage.py")
    text = source.read_text()
    assert "10**" in text or "10 **" in text, (
        "EPInformer-seq's count convention appears to have changed; if it was routed "
        "through chorus.core.count_head, every activity value it produces moved"
    )
    assert "expm1" not in text, (
        "EPInformer-seq now uses expm1; it is trained on log10 counts, so this is a 26x "
        "error at a log-count of 2.5, not a refactor"
    )
    ten, expm1 = 10.0 ** 2.5, float(np.expm1(2.5))
    assert ten / expm1 > 25, "the two conventions really are that far apart"


def test_no_new_copy_of_the_inverse_appears_outside_the_shared_module():
    """The enumeration guard: three copies drifted once, so growth is what to prevent.

    Allowlisted sites are the ones that cannot import the helper — the torch builder — plus
    the module that defines it. Anything else raising ``expm1`` on a count head has to
    justify itself here, in a test, rather than in a comment nobody reads.
    """
    allowed = {
        "chorus/core/count_head.py",                     # the definition
        "scripts/build_backgrounds_cherimoya.py",        # torch, equivalence pinned above
    }
    import ast

    offenders = []
    for path in sorted(list((REPO / "chorus").rglob("*.py"))
                       + list((REPO / "scripts").glob("*.py"))):
        rel = str(path.relative_to(REPO))
        if rel in allowed or "chrombpnet_source/templates" in rel:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:                        # a template that is not importable
            continue
        # Parsed, not grepped: half these files *discuss* expm1 in a docstring explaining
        # the bug, and a text search flags the explanation along with the code.
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("expm1", "log1p_inverse")):
                offenders.append(f"{rel}:{node.lineno}  {ast.unparse(node)[:70]}")
    assert not offenders, (
        "these sites invert a count head without going through "
        f"chorus.core.count_head.counts_from_log:\n  " + "\n  ".join(offenders)
    )
