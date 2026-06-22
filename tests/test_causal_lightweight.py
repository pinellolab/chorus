"""Regression tests for the lightweight causal fine-mapping scoring path.

These tests are model-free: a mock oracle returns deterministic prediction
arrays so we can assert two correctness properties of the efficiency fix in
``prioritize_causal_variants``:

1. An N-variant fine-map builds a FULL :class:`VariantReport` (one carrying
   the IGV ``_predictions`` payload) for **at most** ``report_top_n``
   variants — the rest stay on the cheap lightweight path.

2. The lightweight per-track ``allele_scores`` are **identical** to the
   per-track scores produced by the default (full) ``build_variant_report``
   path. The efficiency fix must not move a single number.
"""

import numpy as np
import pytest

from unittest.mock import MagicMock

from chorus.analysis.causal import (
    prioritize_causal_variants,
    _extract_component_scores,
    CausalWeights,
)
from chorus.analysis.variant_report import build_variant_report
from chorus.utils.ld import LDVariant


# ---------------------------------------------------------------------------
# Mock prediction plumbing (mirrors tests/test_analysis.py helpers)
# ---------------------------------------------------------------------------

_PRED_START = 1_000_000


def _make_track(assay_id, values, resolution=1, chrom="chr1", pred_start=_PRED_START):
    from chorus.core.result import OraclePredictionTrack

    n_bins = len(values)
    pred_end = pred_start + n_bins * resolution

    interval = MagicMock()
    interval.reference = MagicMock()
    interval.reference.chrom = chrom
    interval.reference.start = pred_start
    interval.reference.end = pred_end

    at = assay_id.split(":")[0] if ":" in assay_id else assay_id
    ct = assay_id.split(":")[-1] if ":" in assay_id else ""

    t = OraclePredictionTrack.__new__(OraclePredictionTrack)
    t.source_model = "mock"
    t.assay_id = assay_id
    t.assay_type = at
    t.cell_type = ct
    t.query_interval = interval
    t.prediction_interval = interval
    t.input_interval = interval
    t.resolution = resolution
    t.values = np.asarray(values, dtype=np.float32)
    t.preferred_aggregation = "sum"
    t.preferred_interpolation = "linear_divided"
    t.preferred_scoring_strategy = "mean"
    t.metadata = {}
    t.track_id = None
    return t


def _make_variant_result(seed, position):
    """Deterministic mock of oracle.predict_variant_effect output.

    Two non-expression tracks (DNASE + ChIP-TF) so the report has scorable
    layers without needing a GTF / nearby genes.
    """
    from chorus.core.result import OraclePrediction

    rng = np.random.default_rng(seed)
    n = 600  # a few bins around the variant window
    ref = {
        "DNASE:K562": rng.random(n).astype(np.float32) + 0.5,
        "CHIP:CTCF:K562": rng.random(n).astype(np.float32) + 0.5,
    }
    # Alt = ref with a deterministic, variant-specific perturbation near center
    alt = {}
    for aid, rv in ref.items():
        av = rv.copy()
        center = n // 2
        av[center - 5:center + 5] *= (1.0 + 0.3 * ((seed % 5) + 1))
        alt[aid] = av

    def _build(vals):
        p = OraclePrediction.__new__(OraclePrediction)
        p.tracks = {aid: _make_track(aid, v) for aid, v in vals.items()}
        return p

    return {
        "predictions": {"reference": _build(ref), "alt_1": _build(alt)},
        "variant_info": {"position": position, "alleles": ["A", "G"]},
    }


class _MockOracle:
    """Minimal oracle exposing only what prioritize_causal_variants touches."""

    name = "mock"
    sequence_length = 1024
    reference_fasta = None  # disables allele orientation (no FASTA)

    def __init__(self):
        self.n_predict_calls = 0

    def predict_variant_effect(self, genomic_region, variant_position, alleles, assay_ids):
        self.n_predict_calls += 1
        # Seed off the variant position so each proxy is deterministic and the
        # same position re-predicted in the top-N pass yields identical arrays.
        pos = int(variant_position.split(":")[1])
        return _make_variant_result(seed=pos % 1000, position=variant_position)


def _make_proxies(n):
    proxies = []
    for i in range(n):
        proxies.append(LDVariant(
            variant_id=f"rs{1000 + i}",
            chrom="chr1",
            position=_PRED_START + 300 + i * 7,
            ref="A",
            alt="G",
            r2=1.0 - i * 0.01,
            is_sentinel=(i == 0),
        ))
    return proxies


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _scores_signature(report):
    """Stable, comparable signature of a report's per-track allele_scores."""
    sig = {}
    for allele, tracks in report.allele_scores.items():
        rows = []
        for ts in tracks:
            rows.append((
                ts.assay_id, ts.layer, ts.region_label,
                ts.ref_value, ts.alt_value, ts.raw_score,
            ))
        sig[allele] = sorted(rows, key=lambda r: (r[0], str(r[2])))
    return sig


def test_lightweight_scores_equal_full_scores():
    """Lightweight allele_scores must equal the default full-path scores.

    The mock has no CAGE/RNA tracks, so no per-gene display rows are
    produced and the two paths are literally row-for-row identical here;
    the CAGE-TSS-skip equivalence on a real locus is covered by the GPU
    verification (lead rs9504151 + proxies).
    """
    vr = _make_variant_result(seed=7, position="chr1:1000300")

    full = build_variant_report(vr, oracle_name="mock", lightweight=False)
    # Rebuild predictions: build_variant_report consumes nothing destructively,
    # but use a fresh result to be safe.
    vr2 = _make_variant_result(seed=7, position="chr1:1000300")
    light = build_variant_report(vr2, oracle_name="mock", lightweight=True)

    assert _scores_signature(full) == _scores_signature(light)
    # Lightweight skips the IGV payload; full attaches it.
    assert light._predictions is None
    assert full._predictions is not None


def test_lightweight_composite_scores_equal_full():
    """The composite-driving CausalVariantScore must match between paths.

    Runs both reports through ``_extract_component_scores`` (the exact path
    the ranking uses) and asserts every component the composite reads is
    identical: per-layer max-|effect|, overall max_effect, convergence,
    ref_activity, and per-track raw scores.
    """
    ldv = LDVariant(variant_id="rs1", chrom="chr1", position=1000500,
                    ref="A", alt="G", r2=1.0, is_sentinel=True)
    weights = CausalWeights()

    vr_full = _make_variant_result(seed=11, position="chr1:1000500")
    full = build_variant_report(vr_full, oracle_name="mock", lightweight=False)
    vr_light = _make_variant_result(seed=11, position="chr1:1000500")
    light = build_variant_report(vr_light, oracle_name="mock", lightweight=True)

    cs_full = _extract_component_scores(ldv, full, weights)
    cs_light = _extract_component_scores(ldv, light, weights)

    assert cs_full.per_layer_scores == cs_light.per_layer_scores
    assert cs_full.max_effect == cs_light.max_effect
    assert cs_full.convergence_score == cs_light.convergence_score
    assert cs_full.ref_activity == cs_light.ref_activity
    raw_full = {k: v.raw_score for k, v in cs_full.track_scores.items()}
    raw_light = {k: v.raw_score for k, v in cs_light.track_scores.items()}
    assert raw_full == raw_light


def test_finemap_builds_at_most_report_top_n_full_reports():
    """Fine-map builds full reports (with _predictions) for <= report_top_n variants."""
    oracle = _MockOracle()
    proxies = _make_proxies(8)

    result = prioritize_causal_variants(
        oracle, {"id": "rs1000", "chrom": "chr1", "pos": _PRED_START + 300,
                 "ref": "A", "alt": "G"},
        proxies, assay_ids=None, oracle_name="mock", report_top_n=3,
    )

    n_full = sum(
        1 for s in result.scores
        if s._variant_report is not None and s._variant_report._predictions is not None
    )
    assert n_full <= 3
    # All 8 proxies score non-zero here, so exactly 3 full reports expected.
    assert n_full == 3
    # 8 lightweight scoring passes + 3 top-N re-predictions = 11 forward passes.
    assert oracle.n_predict_calls == 8 + 3


def test_finemap_ranking_matches_full_path():
    """The composite ranking is identical whether top-N reports are built or not.

    report_top_n only controls how many IGV payloads are attached after
    ranking; it must never change the order or the per-track scores.
    """
    proxies = _make_proxies(6)
    lead = {"id": "rs1000", "chrom": "chr1", "pos": _PRED_START + 300,
            "ref": "A", "alt": "G"}

    res_topn = prioritize_causal_variants(
        _MockOracle(), lead, _make_proxies(6), assay_ids=None,
        oracle_name="mock", report_top_n=3,
    )
    res_none = prioritize_causal_variants(
        _MockOracle(), lead, _make_proxies(6), assay_ids=None,
        oracle_name="mock", report_top_n=0,
    )

    order_topn = [(s.variant_id, round(s.composite, 9)) for s in res_topn.scores]
    order_none = [(s.variant_id, round(s.composite, 9)) for s in res_none.scores]
    assert order_topn == order_none

    # And per-track raw scores match variant-by-variant.
    by_id_topn = {s.variant_id: s for s in res_topn.scores}
    for s in res_none.scores:
        t = by_id_topn[s.variant_id]
        raw_none = {k: v.raw_score for k, v in s.track_scores.items()}
        raw_topn = {k: v.raw_score for k, v in t.track_scores.items()}
        assert raw_none == raw_topn

    # report_top_n=0 attaches no IGV payloads.
    assert all(
        s._variant_report is None or s._variant_report._predictions is None
        for s in res_none.scores
    )


def test_finemap_report_top_n_capped_by_nonzero():
    """When fewer variants score non-zero than report_top_n, fewer full reports build."""
    oracle = _MockOracle()
    # Only one proxy; report_top_n=3 but at most 1 full report can be built.
    proxies = _make_proxies(1)
    result = prioritize_causal_variants(
        oracle, {"id": "rs1000", "chrom": "chr1", "pos": _PRED_START + 300,
                 "ref": "A", "alt": "G"},
        proxies, assay_ids=None, oracle_name="mock", report_top_n=3,
    )
    n_full = sum(
        1 for s in result.scores
        if s._variant_report is not None and s._variant_report._predictions is not None
    )
    assert n_full == 1
