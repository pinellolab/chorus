"""The alphagenome_pt head-output dicts are keyed by two unrelated things.

Most heads key by resolution (``{1: ..., 128: ...}``); the splice-site classification head keys by
tensor kind (``{"logits": ..., "probs": ...}``). The extraction loop used to reassign its resolution
variable to whichever key it picked and then call ``int()`` on it, so any SPLICE_SITES track raised
``invalid literal for int() with base 10: 'logits'``. Because the default ``assay_ids=None`` means all
5,168 tracks, and the 4 SPLICE_SITES tracks are always among them, *every* default-argument call to
``predict_variant_effect`` failed on this backend while an explicit list of ATAC ids worked.

These tests exercise ``_select_head_tensor`` and ``_as_resolution`` -- the two functions the production
loop actually calls, not a reimplementation of them. They need neither torch nor a GPU because the
"tensors" are opaque to the selection logic, which is what makes it worth isolating: the equivalent
guard written for Sei's ``assay_ids=None`` defect asserted at the *validation* layer and consequently
did not catch this one, which lives in the execution path.
"""
import pytest

from chorus.oracles.alphagenome_pt import (
    _ACTIVATED_TENSOR_KEYS,
    _as_resolution,
    _select_head_tensor,
)


class TestResolutionSurvivesTensorSelection:
    """The original crash, stated as the thing that must not happen again."""

    def test_a_classification_head_does_not_turn_its_key_into_a_resolution(self):
        # Exactly the shape SpliceSitesClassificationHead returns, and the resolution SPLICE_SITES
        # tracks carry in the metadata.
        head_out = {"logits": "LOGITS", "probs": "PROBS"}
        res = 1

        _select_head_tensor(head_out, res)

        # The pre-fix code reassigned res to "logits" here; int("logits") was the reported failure.
        assert res == 1
        assert _as_resolution(res) == 1

    def test_int_of_a_head_key_is_what_used_to_raise(self):
        # Pins the failure mode itself, so the test explains the bug even if the fix is rewritten.
        with pytest.raises(ValueError, match="invalid literal for int"):
            int("logits")


class TestSelectHeadTensor:
    def test_resolution_keyed_head_selects_by_resolution(self):
        head_out = {1: "one_bp", 128: "onetwentyeight_bp"}
        assert _select_head_tensor(head_out, 128) == "onetwentyeight_bp"
        assert _select_head_tensor(head_out, 1) == "one_bp"

    def test_classification_head_prefers_probabilities_over_logits(self):
        # The correctness half. The JAX reference returns {'logits', 'predictions'} and treats
        # 'predictions' (the softmax) as the prediction -- alphagenome_research reads that key to
        # derive splice junctions. The pt port calls the same tensor 'probs'. Relying on dict
        # insertion order picked 'logits' instead, which would have made the two backends disagree on
        # the 4 SPLICE_SITES tracks: unbounded logits vs probabilities in [0, 1].
        head_out = {"logits": "LOGITS", "probs": "PROBS"}
        assert _select_head_tensor(head_out, 1) == "PROBS"

    def test_insertion_order_does_not_decide(self):
        # Same dict, logits first vs probs first -- the answer must not move.
        logits_first = {"logits": "LOGITS", "probs": "PROBS"}
        probs_first = {"probs": "PROBS", "logits": "LOGITS"}
        assert _select_head_tensor(logits_first, 1) == _select_head_tensor(probs_first, 1) == "PROBS"

    def test_a_bare_tensor_passes_straight_through(self):
        assert _select_head_tensor("TENSOR", 1) == "TENSOR"

    def test_an_unrecognised_dict_falls_back_and_warns(self, caplog):
        # Don't hard-fail on an upstream shape change, but don't be silent either.
        with caplog.at_level("WARNING"):
            got = _select_head_tensor({"something_new": "X"}, 1, "SPLICE_SITES")
        assert got == "X"
        assert "keyed by neither resolution" in caplog.text

    def test_both_upstream_key_vocabularies_are_covered(self):
        # The port is inconsistent: the classification head says "probs", the usage head says
        # "predictions". Missing either one sends that head's tracks back to raw logits, so pin both.
        assert set(_ACTIVATED_TENSOR_KEYS) == {"predictions", "probs"}

    def test_the_usage_head_shape_resolves_to_predictions_not_logits(self):
        # SpliceSitesUsageHead returns {"logits", "predictions", "track_mask"} — 734 tracks. The first
        # version of this fix only knew "probs", so these fell through to logits: measured -13.8..-11.3
        # where JAX gives 1e-6..1.2e-5, i.e. log-space values ranked against a sigmoid-space null.
        head_out = {"logits": "LOGITS", "predictions": "SIGMOID", "track_mask": "MASK"}
        assert _select_head_tensor(head_out, 1) == "SIGMOID"

    def test_a_mask_is_never_returned_as_signal(self):
        # If the activated key is ever absent, a mask must not be sliced as if it were a prediction.
        head_out = {"logits": "LOGITS", "track_mask": "MASK"}
        assert _select_head_tensor(head_out, 1, "SPLICE_SITE_USAGE") == "LOGITS"

    def test_logits_lose_to_any_activated_key_regardless_of_order(self):
        for head_out in (
            {"logits": "L", "predictions": "P"},
            {"predictions": "P", "logits": "L"},
            {"logits": "L", "probs": "P"},
            {"probs": "P", "logits": "L"},
        ):
            assert _select_head_tensor(head_out, 1) == "P"


class TestAsResolution:
    def test_ints_and_numeric_strings_both_work(self):
        assert _as_resolution(128) == 128
        assert _as_resolution("128") == 128

    @pytest.mark.parametrize("bad", ["logits", None, "probs", object()])
    def test_non_numeric_degrades_to_one_bp_with_a_warning(self, bad, caplog):
        # 1 bp is the finest resolution the model emits, so it is the conservative default. A raise
        # here would resurrect the original user-visible failure.
        with caplog.at_level("WARNING"):
            assert _as_resolution(bad) == 1
        assert "Non-numeric resolution" in caplog.text


class TestTheEnvModeTemplateDoesNotHoldASecondCopy:
    """env mode is this backend's default, and it runs a *template*, not the oracle method.

    The first fix for this bug corrected `_predict_raw` only. The template carried its own copy of the
    same six lines, so `predict_variant_effect(..., assay_ids=None)` still raised
    `invalid literal for int() with base 10: 'logits'` end-to-end while every unit test passed. Two
    copies of one rule is the silent-divergence failure mode this repo keeps rediscovering (the Sei
    normalizer and the Jupyter kernel were both correct code that nothing called).

    Source-level assertions, because the template is executed in a per-oracle conda env that this test
    process cannot import torch from.
    """

    @staticmethod
    def _template_source() -> str:
        from pathlib import Path

        import chorus

        path = (
            Path(chorus.__file__).parent
            / "oracles"
            / "alphagenome_pt_source"
            / "templates"
            / "predict_template.py"
        )
        return path.read_text()

    def test_the_template_imports_the_shared_helpers(self):
        src = self._template_source()
        assert "_select_head_tensor" in src, "template must reuse the selection rule, not restate it"
        assert "_as_resolution" in src, "template must reuse the resolution coercion"

    def test_the_template_does_not_call_int_on_a_head_key(self):
        # The exact expression that raised. Its absence is the regression guard.
        assert "int(res)" not in self._template_source()

    def test_the_template_does_not_reimplement_the_first_key_fallback(self):
        # `next(iter(head_out.keys()))` was the line that silently preferred logits over probs.
        assert "next(iter(head_out" not in self._template_source()
