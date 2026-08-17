"""Every oracle must answer "what can you predict?" the same way.

Before `describe_tracks`, that took four different calls: `get_all_assay_ids()` on enformer, borzoi and
the two AlphaGenome backends; `list_tracks()` on cherimoya; a **private** `_get_all_assay_ids()` on sei;
and nothing at all on chrombpnet, legnet or epinformerseq. An audit needed four attempts to obtain one
track id, and `get_track_info()` returned a DataFrame or a dict depending on whether you passed an
argument.

The cost was not only friction. Because no uniform call existed, consumers grew per-oracle branches
that drifted from the oracles: `chorus/analysis/discovery.py:752` calls `get_all_assay_ids()` for sei,
which sei does not have, and the MCP `list_tracks` tool answers for sei from a hardcoded literal
without ever consulting the oracle.

These tests are the thing that stops it drifting again. They run over **all nine** oracles, so a tenth
cannot ship with the question unanswered.
"""
from __future__ import annotations

import pytest

ORACLES = [
    "enformer", "borzoi", "alphagenome", "alphagenome_pt",
    "cherimoya", "sei", "chrombpnet", "legnet", "epinformerseq",
]

#: Counts as published in README / docs and as shipped in each `*_pertrack.npz`. Pinned here because
#: a catalogue that silently disagrees with the background is the failure mode this whole area has:
#: chrombpnet's raw JASPAR table offers 1,268 TF x cell combinations against a 753-row null, and
#: reading it directly instead of the builder's own enumerator reproduced exactly that gap.
EXPECTED_COUNTS = {
    "enformer": 5_313, "borzoi": 7_611, "alphagenome": 5_168, "alphagenome_pt": 5_168,
    "cherimoya": 1_518, "sei": 21_947, "chrombpnet": 753, "legnet": 3, "epinformerseq": 33,
}


def _oracle(name):
    import chorus

    return chorus.create_oracle(name)


@pytest.mark.parametrize("name", ORACLES)
def test_every_oracle_describes_its_tracks(name):
    """The whole point. No oracle may leave this unimplemented."""
    from chorus.core.tracks import TrackRecord

    recs = _oracle(name).describe_tracks()
    assert recs, f"{name}.describe_tracks() returned nothing"
    assert all(isinstance(r, TrackRecord) for r in recs), (
        f"{name} returned something other than TrackRecord objects"
    )


@pytest.mark.parametrize("name", ORACLES)
def test_the_catalogue_size_matches_what_is_published(name):
    recs = _oracle(name).describe_tracks()
    assert len(recs) == EXPECTED_COUNTS[name], (
        f"{name} describes {len(recs):,} tracks; README and the shipped null say "
        f"{EXPECTED_COUNTS[name]:,}. A catalogue that disagrees with the background is worse than no "
        f"catalogue: callers filter on it and then find no percentile."
    )


@pytest.mark.parametrize("name", ORACLES)
def test_track_ids_are_unique_and_non_empty(name):
    recs = _oracle(name).describe_tracks()
    ids = [r.track_id for r in recs]
    assert all(ids), f"{name} produced an empty track_id"
    assert len(set(ids)) == len(ids), (
        f"{name} produced {len(ids) - len(set(ids))} duplicate track_ids; ids are the key callers pass "
        f"to predict() and index by"
    )


@pytest.mark.parametrize("name", ORACLES)
def test_it_works_before_the_model_is_loaded(name):
    """Discovery must not cost a multi-GB model load.

    ChromBPNet is the reason this is asserted rather than assumed: it cannot even be constructed
    without `assay` and `cell_type`, so a catalogue that needed a loaded model would be unusable for
    exactly the question people ask first.
    """
    o = _oracle(name)
    assert getattr(o, "_model", None) is None, f"{name} loaded a model during construction"
    assert _oracle(name).describe_tracks(), f"{name} needs a loaded model to describe its tracks"


@pytest.mark.parametrize("name", ["enformer", "sei", "chrombpnet"])
def test_query_filters_and_limit_caps(name):
    """Filtering is shared by the base wrapper, so all nine get identical search semantics."""
    o = _oracle(name)
    everything = o.describe_tracks()
    assert len(o.describe_tracks(limit=5)) == 5

    probe = (everything[0].assay or everything[0].track_id)[:4]
    hits = o.describe_tracks(query=probe)
    assert hits, f"{name}: query {probe!r} matched nothing although it came from a real record"
    assert len(hits) <= len(everything)
    assert all(r.matches(probe) for r in hits)


def test_the_base_default_refuses_rather_than_returning_empty():
    """"No tracks" and "nobody implemented this" must not look alike.

    A silent empty list is how a tenth oracle would ship invisible to every consumer.
    """
    from chorus.core.base import OracleBase

    class _Bare(OracleBase):
        def load_pretrained_model(self, weights=None): pass
        def list_assay_types(self): return []
        def list_cell_types(self): return []
        def fine_tune(self, tracks, track_names, **kw): pass
        def _predict(self, seq, assay_ids): pass
        def _get_context_size(self): return 1
        def _get_sequence_length_bounds(self): return (1, 2)
        def _get_bin_size(self): return 1

    bare = _Bare()
    with pytest.raises(NotImplementedError, match="_describe_tracks"):
        bare.describe_tracks()


def test_adding_the_method_did_not_make_it_abstract():
    """Seven test-double subclasses across five files implement only the original abstract set.

    If `describe_tracks` or `_describe_tracks` ever becomes abstract, every one of them fails at
    instantiation — which is why the base provides a concrete default that raises when *called*.
    """
    from chorus.core.base import OracleBase

    abstract = set(getattr(OracleBase, "__abstractmethods__", set()))
    assert "describe_tracks" not in abstract and "_describe_tracks" not in abstract, (
        f"describe_tracks became abstract; that breaks the existing test doubles. abstract set: "
        f"{sorted(abstract)}"
    )


def test_sei_reports_both_kinds_of_track():
    """Sei's two id families are the reason `extra["kind"]` exists.

    It predicts 21,907 chromatin profiles *and* 40 projected sequence classes; `predict()` accepts
    both. Until the 2026-08-16 rebuild only the 40 had background rows, so the profiles returned real
    values whose percentile was always None.
    """
    recs = _oracle("sei").describe_tracks()
    kinds = {r.extra.get("kind") for r in recs}
    assert kinds == {"chromatin_profile", "sequence_class"}, kinds
    profiles = [r for r in recs if r.extra.get("kind") == "chromatin_profile"]
    classes = [r for r in recs if r.extra.get("kind") == "sequence_class"]
    assert len(profiles) == 21_907 and len(classes) == 40
    assert all(r.track_id.startswith("TA#") for r in profiles)
    assert all(r.track_id.startswith("CA#") for r in classes)


# ── the four defects the missing method was causing ──────────────────────────────

#: Oracles discovery cannot yet score, with the reason. `cherimoya` is genuinely per-model — one
#: track per loaded model, and its `predict()` ignores `assay_ids` — so routing it needs an
#: enumeration branch over 1,518 models, not a set-membership entry. Listing it here rather than
#: adding it to a set it would silently return `models = []` from.
#: Empty, and that is the point: cherimoya was the last entry and now has a real enumeration branch
#: (one model per biosample, 407 of 1,518, logged at runtime). Kept as a mechanism so a future gap is
#: recorded with its reason instead of being absorbed into a set that would silently return no models.
DISCOVERY_UNSUPPORTED: dict = {}


def test_discovery_routes_every_oracle_it_claims_to():
    """`alphagenome_pt` was in neither oracle set, so discovery returned "Unsupported oracle".

    Asserted against an explicit exception list so a newly-unrouted oracle fails here, while the one
    genuine gap stays visible instead of being quietly absorbed into a set that would make discovery
    return an empty model list.
    """
    from chorus.analysis.discovery import _MULTI_TRACK_ORACLES, _PER_MODEL_ORACLES

    routed = _MULTI_TRACK_ORACLES | _PER_MODEL_ORACLES
    missing = [o for o in ORACLES if o not in routed and o not in DISCOVERY_UNSUPPORTED]
    assert not missing, (
        f"these oracles are in neither discovery set and fall through to "
        f'{{"error": "Unsupported oracle"}}: {missing}'
    )
    # and the known gap must stay a *loud* failure, not an empty result
    for name in DISCOVERY_UNSUPPORTED:
        assert name not in routed, (
            f"{name} was added to a discovery set; if that is intentional it also needs a real "
            f"enumeration branch, or discovery will return an empty model list for it"
        )


def test_discovery_enumerates_through_describe_tracks():
    """It called `get_all_assay_ids()`, which Sei does not have — an AttributeError on a live path."""
    import inspect

    from chorus.analysis import discovery

    src = inspect.getsource(discovery)
    assert "oracle.get_all_assay_ids()" not in src, (
        "discovery calls oracle.get_all_assay_ids(); Sei has only the private _get_all_assay_ids, so "
        "that raises AttributeError for an oracle listed in _MULTI_TRACK_ORACLES"
    )
    assert "describe_tracks()" in src


@pytest.mark.parametrize("module", ["chorus/core/exceptions.py",
                                    "chorus/analysis/variant_report.py"])
def test_error_hints_name_a_method_every_oracle_has(module):
    """Both told users to call `get_all_assay_ids()`, which 5 of 9 oracles lack."""
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / module).read_text()
    assert "get_all_assay_ids" not in text, (
        f"{module} points users at oracle.get_all_assay_ids(), which only 4 of 9 oracles implement"
    )


def test_epinformerseq_lists_every_cell_type():
    """It returned `[self.cell_type]` — 1 of 11 — while its null carries all 33 rows."""
    from chorus.oracles.epinformerseq_source.globals import EPINFORMERSEQ_AVAILABLE_CELLTYPES

    listed = _oracle("epinformerseq").list_cell_types()
    assert set(listed) == set(EPINFORMERSEQ_AVAILABLE_CELLTYPES), (
        f"epinformerseq lists {len(listed)} of {len(EPINFORMERSEQ_AVAILABLE_CELLTYPES)} cell types. "
        f"LegNet had this exact bug and fixed it at legnet.py:192; the answer must describe the oracle, "
        f"not the instance."
    )
    assert len(listed) == 11


#: Oracles whose tracks are cell-type specific, with the distinct count published for each. Sei's
#: sequence classes are genuinely not cell-typed, so it is absent by design rather than by omission.
EXPECTED_CELL_TYPES = {"cherimoya": 407, "legnet": 3, "epinformerseq": 11}


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_CELL_TYPES.items()))
def test_cell_type_is_actually_populated(name, expected):
    """Counts being right is not the same as the fields being right.

    cherimoya's first implementation read `biosample_term_name`, a key CATv1-metadata.tsv does not
    have, so `cell_type` was None on all 1,518 records. Every count assertion still passed — and the
    field the catalogue exists to let you filter on was empty. It surfaced only when discovery tried to
    dedupe by biosample and found one.
    """
    recs = _oracle(name).describe_tracks()
    populated = [r for r in recs if r.cell_type]
    assert populated, f"{name}: no record carries a cell_type at all"
    distinct = {r.cell_type for r in populated}
    assert len(distinct) == expected, (
        f"{name} reports {len(distinct)} distinct cell types, expected {expected}. A catalogue whose "
        f"cell_type is empty or collapsed cannot answer the question it exists for."
    )


@pytest.mark.parametrize("name", ORACLES)
def test_assay_is_populated_for_every_track(name):
    """`assay` drives layer classification, so a null one is a track that cannot be scored."""
    recs = _oracle(name).describe_tracks()
    missing = [r.track_id for r in recs if not r.assay]
    assert not missing, (
        f"{name}: {len(missing)} of {len(recs)} records carry no assay, e.g. {missing[:3]}. "
        f"classify_track_layer dispatches on it, so those tracks would classify as 'other'."
    )


def test_cherimoya_is_routed_by_biosample_not_by_experiment():
    """Discovery loads one model at a time; 1,518 experiments is 6.5 h, 407 biosamples is 1.7 h.

    Measured 15.3 s per cherimoya model (5.0 s load + 10.3 s predict). Deduping to biosamples is a
    dedupe rather than a sample — discovery asks which cell types respond, and a biosample with four
    ATAC experiments answers that once — but the count is logged at runtime either way, because a
    silent reduction reads as full coverage.
    """
    from chorus.analysis.discovery import _PER_MODEL_ORACLES

    assert "cherimoya" in _PER_MODEL_ORACLES, (
        "cherimoya is unrouted again, so discovery returns {'error': 'Unsupported oracle'}"
    )
    recs = _oracle("cherimoya").describe_tracks()
    biosamples = {r.cell_type for r in recs if r.cell_type}
    assert len(biosamples) < len(recs), "the dedupe cannot help if every experiment is its own biosample"
