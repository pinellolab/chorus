"""``score_variant_effect_at_region`` contradicted itself on every single call.

The tool attaches a ``score_notes`` entry to explain a null score — the rule
``score_prediction_region`` follows, and worth having. But the lookup that decided
whether a score WAS null read the payload at the wrong depth:
``score_variant_effect`` returns ``{allele: {assay_id: {ref,alt,effect}}}``
(chorus/core/result.py) and the guard did ``scores.get(assay_id)`` against the
top level, i.e. against allele names. It never hit, the "is anything non-null?"
test was therefore always False, and the note was emitted unconditionally:

    scores.alt_1.ENCFF413AHU = {ref 2.0686, alt 2.4550, effect 0.3864}
    score_notes.ENCFF413AHU = "no score: the scored slice spans fewer than one
                               128 bp bin"

An agent reading that payload has two contradictory answers and no way to tell
which is real. Worse, a *genuine* null got the sub-bin explanation regardless of
the actual reason, because the non-overlap branch that ``score_prediction_region``
has was missing here.

Model-free by construction: a stub oracle returning real ``OraclePrediction`` /
``OraclePredictionTrack`` objects exercises the tool body, the real scorer in
``result.py`` and the note logic without a GPU or model weights — the defect lives
in dict shapes, which no amount of real signal would have made visible.
"""
from __future__ import annotations

import numpy as np
import pytest

from chorus.core.interval import GenomeRef, Interval
from chorus.core.result import OraclePrediction, OraclePredictionTrack

CHROM, START, RES = "chr1", 1_000_000, 128
NBINS = 5
AID = "DNASE:K562"
POSITION = f"{CHROM}:{START + 300}"
REGION = f"{CHROM}:{START + 1}-{START + NBINS * RES}"


def _track(values, *, assay_id=AID, resolution=RES, start=START, chrom=CHROM,
           span_bins=None):
    """``span_bins`` decouples the declared interval from ``len(values)``.

    Default (None) keeps them consistent. Passing a larger value reproduces the
    fabricated-``resolution`` track the repo documents (LegNet: one value over a
    200 bp interval declared at 50 bp), which is the case that distinguishes a
    geometry null from a variant genuinely outside the window.
    """
    span = len(values) if span_bins is None else span_bins
    iv = Interval.make(GenomeRef(
        chrom=chrom, start=start, end=start + span * resolution, fasta=None,
    ))
    return OraclePredictionTrack(
        source_model="stub", assay_id=assay_id, track_id="T", assay_type="DNASE",
        cell_type="K562", query_interval=iv, prediction_interval=iv, input_interval=iv,
        resolution=resolution, values=np.asarray(values, dtype=float), metadata=None,
    )


class StubOracle:
    """Returns a fixed ref/alt pair of predictions for whatever it is asked.

    ``tracks`` is a list of (assay_id, ref_values, alt_values, kwargs) so a test can
    place a track's window where it likes — that is what makes a genuine null
    reachable without a model.
    """

    reference_fasta = None
    sequence_length = NBINS * RES

    def __init__(self, tracks=None):
        self.tracks = tracks or [
            (AID, [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.5, 3.5, 4.0, 5.0], {}),
        ]

    def predict_variant_effect(self, genomic_region, variant_position, alleles,
                               assay_ids=None, genome=None):
        ref = OraclePrediction()
        for aid, ref_vals, _alt_vals, kw in self.tracks:
            ref.add(aid, _track(ref_vals, assay_id=aid, **kw))
        predictions = {"reference": ref}
        for i in range(len(alleles) - 1):          # one prediction per alt allele
            alt = OraclePrediction()
            for aid, _ref_vals, alt_vals, kw in self.tracks:
                alt.add(aid, _track(alt_vals, assay_id=aid, **kw))
            predictions[f"alt_{i + 1}"] = alt
        return {
            "predictions": predictions,
            "variant_info": {"position": variant_position, "ref": alleles[0],
                             "alts": list(alleles[1:])},
        }


@pytest.fixture
def call(monkeypatch):
    """Inject a stub oracle (and optional normalizer) into the MCP state manager."""
    import chorus.mcp.server as server

    state = server._state()
    saved_oracles = dict(state._oracles)
    saved_norms = dict(state._normalizers)

    fn = server.score_variant_effect_at_region
    for attr in ("fn", "__wrapped__"):
        fn = getattr(fn, attr, fn)

    def _call(oracle=None, normalizer=None, **kw):
        state._oracles["stub"] = oracle or StubOracle()
        state._normalizers["stub"] = normalizer
        kw.setdefault("oracle_name", "stub")
        kw.setdefault("position", POSITION)
        kw.setdefault("ref_allele", "A")
        kw.setdefault("alt_alleles", ["T"])
        kw.setdefault("assay_ids", [AID])
        kw.setdefault("region", REGION)
        return fn(**kw)

    yield _call
    state._oracles.clear(); state._oracles.update(saved_oracles)
    state._normalizers.clear(); state._normalizers.update(saved_norms)


# ---------------------------------------------------------------------------
# The contradiction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [
    {"at_variant": True},
    {"score_region": f"{CHROM}:{START + 200}-{START + 500}"},
])
def test_no_note_is_attached_when_a_score_exists(call, mode):
    """The assertion the defect failed: in BOTH modes, and on every call.

    Before the fix this returned ref 3.0 / alt 3.333 / effect 0.333 *and*
    "no score: the scored slice spans fewer than one 128 bp bin".
    """
    out = call(**mode)
    inner = out["scores"]["alt_1"][AID]
    assert inner["ref_score"] is not None and inner["effect"] is not None, inner
    assert "score_notes" not in out, (
        f"a non-null score carries a null explanation: {out.get('score_notes')}"
    )


def test_a_note_survives_only_while_every_allele_is_null(call):
    """A note is per-assay, so any allele scoring is enough to withdraw it."""
    two_alleles = call(alt_alleles=["T", "G"], at_variant=True)
    per_allele = two_alleles["scores"]
    assert set(per_allele) == {"alt_1", "alt_2"}
    assert all(v[AID]["effect"] is not None for v in per_allele.values())
    assert "score_notes" not in two_alleles


# ---------------------------------------------------------------------------
# A genuine null must name its own reason
# ---------------------------------------------------------------------------


def test_a_non_overlapping_score_region_says_so(call):
    """The branch ``score_prediction_region`` already had, missing here.

    The null is real; the old note blamed the bin width, which is not why.
    """
    out = call(score_region="chr9:1-1000")
    assert out["scores"]["alt_1"][AID]["ref_score"] is None
    note = out.get("score_notes", {}).get(AID, "")
    assert "does not overlap" in note and "chr9:1-1000" in note, note
    assert "fewer than one" not in note, f"wrong reason for a non-overlap null: {note}"


def test_at_variant_names_the_track_whose_window_excludes_the_variant(call):
    """Mixed windows: the first track fixes the bin, a later one may not span it.

    ``result.py`` hands that track empty slices and nulls, which used to be blamed
    on the bin width. Only reachable with >1 track, because a first track that does
    not span the variant raises instead.
    """
    far = START + 50 * RES
    oracle = StubOracle(tracks=[
        (AID, [1.0] * NBINS, [2.0] * NBINS, {}),
        ("DNASE:HepG2", [1.0] * NBINS, [2.0] * NBINS, {"start": far}),
    ])
    out = call(oracle=oracle, assay_ids=[AID, "DNASE:HepG2"], at_variant=True)
    assert out["scores"]["alt_1"][AID]["effect"] is not None
    assert out["scores"]["alt_1"]["DNASE:HepG2"]["effect"] is None
    notes = out.get("score_notes", {})
    assert AID not in notes, notes
    assert "maps outside this track's prediction window" in notes.get("DNASE:HepG2", ""), notes


def test_a_fabricated_resolution_null_is_blamed_on_the_geometry_not_the_variant(call):
    """``pos2bin`` returns None for two different reasons, and they read differently.

    A track whose declared ``resolution`` overstates its sampling (one value, 128 bp
    declared over 640 bp) derives an in-range genomic position into an out-of-range
    array index, so ``pos2bin`` is None for a variant that is plainly *inside* the
    window. The at_variant note therefore has to come after the geometry check, or
    it contradicts itself: "chr1:1000300 maps outside chr1:1000000-1000640".
    """
    oracle = StubOracle(tracks=[
        (AID, [1.0] * NBINS, [2.0] * NBINS, {}),
        ("DNASE:HepG2", [1.0], [2.0], {"span_bins": NBINS}),
    ])
    out = call(oracle=oracle, assay_ids=[AID, "DNASE:HepG2"], at_variant=True)
    assert out["scores"]["alt_1"]["DNASE:HepG2"]["effect"] is None
    note = out.get("score_notes", {}).get("DNASE:HepG2", "")
    assert "1 value(s)" in note and "implies 5 bins" in note, note
    assert "maps outside" not in note, f"the variant is inside the window: {note}"


def test_a_null_is_explained_under_the_assay_id_the_payload_reports(call):
    """Oracles rename assays (ChromBPNet answers "ATAC" as "ATAC:K562").

    The note loop iterated the REQUESTED ids, so a renamed track's genuine null
    went unexplained. It now follows the ids the payload actually carries.
    """
    oracle = StubOracle(tracks=[("ATAC:K562", [1.0] * NBINS, [2.0] * NBINS, {})])
    out = call(oracle=oracle, assay_ids=["ATAC"], score_region="chr9:1-1000")
    assert out["scores"]["alt_1"]["ATAC:K562"]["ref_score"] is None
    assert "does not overlap" in out.get("score_notes", {}).get("ATAC:K562", ""), out


# ---------------------------------------------------------------------------
# ref_activity_percentiles, dead for the same reason
# ---------------------------------------------------------------------------


def test_ref_activity_percentiles_are_reported(call):
    """The percentile block read ``scores[assay_id]["reference"]`` — allele-keyed
    dict, assay-shaped lookup — so ``ref_val`` was always None and the key never
    appeared. It also called ``.get()`` on an ``OraclePrediction``, which has no
    ``.get``: fixing only the outer loop would have raised AttributeError.
    """
    class StubNormalizer:
        def __init__(self): self.seen = []

        def normalize_baseline(self, oracle_name, layer, raw_signal):
            self.seen.append((oracle_name, layer, raw_signal))
            return 0.75

    norm = StubNormalizer()
    out = call(normalizer=norm, at_variant=True)
    assert out.get("ref_activity_percentiles") == {AID: 0.75}, out
    # ...and it was handed the ref score from the payload, not something else.
    assert norm.seen and norm.seen[0][2] == out["scores"]["alt_1"][AID]["ref_score"]
