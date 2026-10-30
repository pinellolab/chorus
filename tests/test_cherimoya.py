"""Tests for the Cherimoya / CATv1 oracle.

The tests here need neither weights nor a GPU: metadata is vendored, and
the scoring transforms are exercised with synthetic head outputs.  Tests
that require a real checkpoint are marked ``integration`` (see
``pytest.ini``) so the fast suite stays offline.
"""

from pathlib import Path

import numpy
import pytest

from chorus.oracles.cherimoya_source.batv1_defaults import BATV1_DEFAULT_EXPERIMENT
from chorus.oracles.cherimoya_source.batv1_globals import (
    BATV1_ASSAY_TYPES,
    BATV1_N_CONTROL_TRACKS,
    BATV1_SIGNAL_GROUPS,
)
from chorus.oracles.cherimoya_source.catv1_defaults import CATV1_DEFAULT_EXPERIMENT
from chorus.oracles.cherimoya_source.catv1_globals import (
    CATV1_ASSAY_TYPES,
    CATV1_INPUT_LENGTH,
    CATV1_OUTPUT_LENGTH,
    CATV1_SCORING_WINDOW_BP,
    CATV1_TRIMMING,
    catv1_track_id,
)
from chorus.oracles.cherimoya_source.catv1_metadata import get_metadata
from chorus.oracles.cherimoya_source.scoring import (
    compute_effect,
    expected_counts_profile,
    score_window_sum,
)


def _oracle():
    from chorus.oracles.cherimoya import CherimoyaOracle
    return CherimoyaOracle(use_environment=False)


# ── geometry ─────────────────────────────────────────────────────────

def test_geometry_matches_chrombpnet():
    """CATv1 shares ChromBPNet's geometry; the ported offsets rely on it."""
    from chorus.oracles.chrombpnet import ChromBPNetOracle

    cherimoya = _oracle()
    assert cherimoya.sequence_length == CATV1_INPUT_LENGTH == 2114
    assert cherimoya.output_length == CATV1_OUTPUT_LENGTH == 1000
    assert cherimoya.bin_size == 1
    assert cherimoya.output_size == 1000
    assert CATV1_TRIMMING == 557

    chrombpnet = ChromBPNetOracle(use_environment=False)
    assert cherimoya.sequence_length == chrombpnet.sequence_length
    assert cherimoya.output_length == chrombpnet.output_length
    assert cherimoya.bin_size == chrombpnet.bin_size


def test_oracle_name_and_contract():
    oracle = _oracle()
    assert oracle.oracle_name == "cherimoya"
    assert oracle._get_context_size() == 2114
    assert oracle._get_bin_size() == 1
    assert oracle._get_sequence_length_bounds() == (500, 2114)
    assert oracle.track_id is None  # nothing loaded yet


def test_assay_types_are_the_load_bearing_strings():
    """`assay_type` must dispatch to the DNase/ATAC track classes.

    These exact strings drive OraclePredictionTrack.create and
    classify_track_layer; a typo would silently land tracks in the
    'other' layer with no background normalization.
    """
    from chorus.analysis.scorers import classify_track_layer
    from chorus.core.result import OraclePredictionTrack

    assert _oracle().list_assay_types() == ["ATAC", "CHIP", "DNASE"]

    for assay in CATV1_ASSAY_TYPES:
        assert assay in OraclePredictionTrack._registry

        class _Stub:
            assay_type = assay
            assay_id = "ENCSR000EOT"
            metadata = None

        assert classify_track_layer(_Stub()) == "chromatin_accessibility"

    # BATv1 tracks are CHIP, and every target is a TF-family protein, so they must land in
    # tf_binding (the 501 bp window) rather than histone_marks (2001 bp) -- the layer is
    # decided from the id and the description the oracle stamps.
    assert "CHIP" in OraclePredictionTrack._registry
    oracle = _oracle()
    oracle.assay, oracle.encode_id, oracle.cell_type, oracle.tf = "CHIP", "ENCSR000EGM", "K562", "CTCF"

    class _ChipStub:
        assay_type = "CHIP"
        assay_id = oracle.track_id
        metadata = oracle._track_metadata()

    assert _ChipStub.assay_id == "CHIP:ENCSR000EGM"
    assert _ChipStub.metadata["description"] == "CTCF binding in K562"
    assert _ChipStub.metadata["atlas"] == "BATv1"
    assert classify_track_layer(_ChipStub()) == "tf_binding"


def test_no_batv1_target_reads_as_a_histone_mark():
    """classify_chip_layer pattern-matches mark names in the id and description; a target
    that matched would be scored at 2001 bp against a 501 bp null."""
    from chorus.analysis.scorers import classify_chip_layer

    df = get_metadata().tracks_df
    targets = sorted(df.loc[df["atlas"] == "BATv1", "target"].unique())
    assert len(targets) == 1139
    histone = [t for t in targets if classify_chip_layer("CHIP:ENCSR000000", f"{t} binding in K562") != "tf_binding"]
    assert histone == []


def test_fine_tune_raises():
    with pytest.raises(NotImplementedError):
        _oracle().fine_tune([], [])


# ── track ids ────────────────────────────────────────────────────────

def test_track_id_format_and_uniqueness():
    assert catv1_track_id("DNASE", "ENCSR000EOT") == "DNASE:ENCSR000EOT"
    assert catv1_track_id("CHIP", "ENCSR000EGM") == "CHIP:ENCSR000EGM"

    ids = get_metadata().list_track_ids()
    assert len(ids) == 1518 + 3429
    assert len(set(ids)) == 1518 + 3429, "track ids must be unique"
    assert all(i.split(":")[0] in CATV1_ASSAY_TYPES + BATV1_ASSAY_TYPES for i in ids)
    assert sum(i.startswith("CHIP:") for i in ids) == 3429
    # Exactly two colon-separated fields: anything else would break the
    # normalizer's last-component fallback.
    assert all(len(i.split(":")) == 2 for i in ids)


def test_accession_alone_is_unique():
    """The rationale for dropping biosample from the id."""
    df = get_metadata().tracks_df
    assert df["experiment_accession"].nunique() == len(df) == 1518 + 3429
    assert (df.groupby("experiment_accession")["assay"].nunique() == 1).all()
    assert df["atlas"].value_counts().to_dict() == {"BATv1": 3429, "CATv1": 1518}


def test_assay_biosample_is_ambiguous():
    """Documents *why* the id is accession-based, so a future change to
    ASSAY:cell_type fails here rather than silently aliasing CDF rows."""
    df = get_metadata().tracks_df
    catv1 = df[df["atlas"] == "CATv1"]
    sizes = catv1.groupby(["assay", "biosample"]).size()
    assert len(sizes) == 492
    assert (sizes > 1).sum() == 162
    assert sizes.max() == 83
    # Even the flagship lines are ambiguous.
    assert sizes[("ATAC", "K562")] == 4


def test_biosample_target_is_ambiguous_in_batv1():
    """The same argument for BATv1: (biosample, target) does not name a model either."""
    df = get_metadata().tracks_df
    batv1 = df[df["atlas"] == "BATv1"]
    sizes = batv1.groupby(["biosample", "target"]).size()
    assert len(sizes) == 2474
    assert (sizes > 1).sum() == 437
    assert sizes.max() == 59
    assert sizes[("K562", "CTCF")] == 5
    assert sizes[("K562", "GATA1")] == 2


# ── defaults table ───────────────────────────────────────────────────

def test_defaults_cover_every_pair_and_are_valid():
    df = get_metadata().tracks_df
    df = df[df["atlas"] == "CATv1"]
    pairs = set(map(tuple, df[["assay", "biosample"]].drop_duplicates().values))
    assert set(CATV1_DEFAULT_EXPERIMENT) == pairs

    valid = set(df["experiment_accession"])
    for (assay, biosample), (accession, n, reason) in CATV1_DEFAULT_EXPERIMENT.items():
        assert accession in valid
        row = df[df["experiment_accession"] == accession].iloc[0]
        assert row["assay"] == assay
        assert row["biosample"] == biosample
        assert n >= 1
        assert reason in {"chrombpnet-parity", "best-count-pearson", "only-candidate"}


def test_batv1_defaults_cover_every_pair_and_are_valid():
    df = get_metadata().tracks_df
    batv1 = df[df["atlas"] == "BATv1"]
    pairs = set(map(tuple, batv1[["biosample", "target"]].drop_duplicates().values))
    assert set(BATV1_DEFAULT_EXPERIMENT) == pairs

    valid = set(batv1["experiment_accession"])
    status = dict(zip(batv1["experiment_accession"], batv1["experiment_status"]))
    for (biosample, target), (accession, n, reason) in BATV1_DEFAULT_EXPERIMENT.items():
        assert accession in valid
        row = batv1[batv1["experiment_accession"] == accession].iloc[0]
        assert row["biosample"] == biosample
        assert row["target"] == target
        assert n >= 1
        assert reason in {"bpnet-parity", "best-count-pearson", "only-candidate"}
    # A revoked experiment is never the default when a released one exists.
    for (biosample, target), (accession, n, _) in BATV1_DEFAULT_EXPERIMENT.items():
        if status[accession] == "revoked":
            siblings = batv1[(batv1["biosample"] == biosample) & (batv1["target"] == target)]
            assert set(siblings["experiment_status"]) == {"revoked"}, (biosample, target)
    assert BATV1_DEFAULT_EXPERIMENT[("K562", "ZNF143")][0] == "ENCSR427WZJ"


def test_bpnet_matched_pairs_are_pinned():
    """Every BPNet CHIP model chorus serves was trained on a BATv1 experiment, so the
    (biosample, target) pairs they cover resolve to the model trained on the same ENCODE
    experiment -- `CHIP:K562:CTCF` means the same experiment on both oracles."""
    import pandas
    from chorus.oracles.chrombpnet_source.chrombpnet_globals import iter_unique_bpnet_models

    pinned = {k: v[0] for k, v in BATV1_DEFAULT_EXPERIMENT.items() if v[2] == "bpnet-parity"}
    assert len(pinned) == 743
    assert pinned[("K562", "CTCF")] == "ENCSR000DMA"
    assert pinned[("HepG2", "CTCF")] == "ENCSR000DUG"

    jaspar_path = (Path(__file__).resolve().parents[1] / "chorus" / "oracles"
                   / "chrombpnet_source" / "chrombpnet_JASPAR_metadata.tsv")
    jaspar = pandas.read_csv(jaspar_path, sep="\t").drop_duplicates(["TF_NAME", "CELL_LINE"])
    sources = set(jaspar["SOURCE_ID"])
    assert len(sources) == 744
    assert set(pinned.values()) <= sources
    assert len(list(iter_unique_bpnet_models())) == 744
    # The one pin not taken is the revoked K562 / ZNF143 experiment.
    assert sources - set(pinned.values()) == {"ENCSR000EGP"}


def test_chrombpnet_matched_pairs_are_pinned():
    """The 9 pairs ChromBPNet also covers must resolve to the CATv1 model
    trained on the same ENCODE experiment, so `ATAC:K562` means the same
    experiment on both oracles."""
    pinned = {
        k: v[0] for k, v in CATV1_DEFAULT_EXPERIMENT.items()
        if v[2] == "chrombpnet-parity"
    }
    assert pinned == {
        ("ATAC", "K562"): "ENCSR868FGK",
        ("ATAC", "HepG2"): "ENCSR291GJU",
        ("ATAC", "GM12878"): "ENCSR637XSC",
        ("ATAC", "IMR-90"): "ENCSR200OML",
        ("DNASE", "HepG2"): "ENCSR149XIL",
        ("DNASE", "IMR-90"): "ENCSR477RTP",
        ("DNASE", "GM12878"): "ENCSR000EMT",
        ("DNASE", "K562"): "ENCSR000EOT",
        ("DNASE", "H1"): "ENCSR000EMU",
    }


# ── resolution ───────────────────────────────────────────────────────

def test_resolve_by_pair_and_accession():
    meta = get_metadata()
    assert meta.resolve(assay="ATAC", cell_type="K562") == ("ATAC", "ENCSR868FGK")
    assert meta.resolve(assay="atac", cell_type="K562") == ("ATAC", "ENCSR868FGK")
    assert meta.resolve(encode_id="ENCSR000EOT") == ("DNASE", "ENCSR000EOT")
    # BATv1: the pair needs a target too, and an accession alone is enough.
    assert meta.resolve(assay="CHIP", cell_type="K562", tf="CTCF") == ("CHIP", "ENCSR000DMA")
    assert meta.resolve(assay="chip", cell_type="K562", tf="GATA1") == ("CHIP", "ENCSR000EFT")
    assert meta.resolve(encode_id="ENCSR000EGM") == ("CHIP", "ENCSR000EGM")
    assert meta.resolve(encode_id="ENCSR000EGM", assay="CHIP", tf="CTCF") == ("CHIP", "ENCSR000EGM")


def test_resolve_rejects_bad_input():
    meta = get_metadata()
    with pytest.raises(KeyError):
        meta.resolve(encode_id="ENCSR_NOT_REAL")
    with pytest.raises(KeyError):
        meta.resolve(assay="DNASE", cell_type="not a biosample")
    with pytest.raises(KeyError):
        meta.resolve(assay="HISTONE", cell_type="K562")
    with pytest.raises(ValueError):
        # CHIP without a target is underspecified, not a lookup miss
        meta.resolve(assay="CHIP", cell_type="K562")
    with pytest.raises(KeyError):
        meta.resolve(assay="CHIP", cell_type="K562", tf="not a target")
    with pytest.raises(ValueError):
        meta.resolve(assay="DNASE")
    with pytest.raises(ValueError):
        # accession/assay disagreement should not pass silently
        meta.resolve(assay="ATAC", encode_id="ENCSR000EOT")
    with pytest.raises(ValueError):
        meta.resolve(assay="DNASE", encode_id="ENCSR000EGM")
    with pytest.raises(ValueError):
        # accession/target disagreement either
        meta.resolve(encode_id="ENCSR000EGM", tf="GATA1")


def test_load_pretrained_model_validates_input():
    from chorus.core.exceptions import InvalidAssayError
    oracle = _oracle()
    with pytest.raises(InvalidAssayError):
        oracle.load_pretrained_model(assay="DNASE", cell_type="K562", fold=5)
    with pytest.raises(InvalidAssayError):
        oracle.load_pretrained_model(assay="DNASE", cell_type="nope")
    with pytest.raises(InvalidAssayError, match="TF="):
        oracle.load_pretrained_model(assay="CHIP", cell_type="K562")
    with pytest.raises(InvalidAssayError):
        oracle.load_pretrained_model(assay="CHIP", cell_type="K562", TF="nope")


# ── metadata surface ─────────────────────────────────────────────────

def test_metadata_summary_and_search():
    meta = get_metadata()
    assert meta.get_track_summary() == {"ATAC": 369, "CHIP": 3429, "DNASE": 1149}
    assert meta.list_assay_types() == ["ATAC", "CHIP", "DNASE"]
    assert len(meta.list_cell_types()) == 407            # accessibility biosamples, as before
    assert len(meta.list_cell_types("CHIP")) == 183
    assert len(meta.list_cell_types("DNASE")) < 407
    assert len(meta.list_targets()) == 1139
    assert len(meta.search_tracks("ENCSR000EOT")) == 1
    assert len(meta.search_tracks("K562")) >= 5 + 760
    gata1 = meta.search_tracks("GATA1")
    assert len(gata1) >= 2 and set(gata1["assay"]) == {"CHIP"}
    assert meta.search_tracks("definitely-not-present").empty


def test_describe_accepts_track_id_or_accession():
    meta = get_metadata()
    by_track = meta.describe("DNASE:ENCSR000EOT")
    by_acc = meta.describe("ENCSR000EOT")
    assert by_track["experiment_accession"] == by_acc["experiment_accession"]
    assert by_track["biosample"] == "K562"
    assert by_track["assay"] == "DNASE"
    assert by_track["atlas"] == "CATv1"
    assert by_track["target"] == ""
    assert 0 <= by_track["count_pearson"] <= 1

    chip = meta.describe("CHIP:ENCSR000EGM")
    assert chip == meta.describe("ENCSR000EGM")
    assert (chip["atlas"], chip["assay"], chip["target"], chip["biosample"]) == ("BATv1", "CHIP", "CTCF", "K562")
    assert chip["experiment_status"] == "released" and chip["quality_flags"] == ""
    assert 0 <= chip["count_pearson"] <= 1
    assert "early_checkpoint" in meta.describe("ENCSR343IFJ")["quality_flags"]


def test_atlas_is_grch38_only():
    df = get_metadata().tracks_df
    assert set(df["assembly"]) == {"GRCh38"}


# ── scoring transforms ───────────────────────────────────────────────

def test_expected_counts_uses_expm1_not_exp():
    """The single most consequential line in the integration.

    Cherimoya's count head is trained against log(count + 1), so the
    inverse is expm1.  ChromBPNet's builder used exp() until 2026-07-31;
    porting that verbatim would have inflated every value.
    """
    logits = numpy.zeros((1, 1, 4))                  # uniform -> 0.25 each
    log_counts = numpy.array([[numpy.log(101.0)]])   # log(100 + 1)

    profile = expected_counts_profile(logits, log_counts)

    assert profile.shape == (1, 4)
    numpy.testing.assert_allclose(profile.sum(), 100.0, rtol=1e-9)
    numpy.testing.assert_allclose(profile, 25.0, rtol=1e-9)

    # exp() would have given 101 -- assert we are not doing that.
    assert not numpy.isclose(profile.sum(), 101.0)


def test_expected_counts_shape_handling():
    logits2d = numpy.zeros((2, 8))
    logits3d = numpy.zeros((2, 1, 8))
    counts2d = numpy.log1p(numpy.array([[10.0], [20.0]]))
    counts1d = numpy.log1p(numpy.array([10.0, 20.0]))

    a = expected_counts_profile(logits3d, counts2d)
    b = expected_counts_profile(logits2d, counts1d)
    numpy.testing.assert_allclose(a, b)
    numpy.testing.assert_allclose(a.sum(axis=1), [10.0, 20.0], rtol=1e-9)


def test_expected_counts_rejects_more_than_two_tracks():
    with pytest.raises(ValueError):
        expected_counts_profile(numpy.zeros((1, 3, 8)), numpy.zeros((1, 1)))


def test_expected_counts_sums_two_strands_under_one_softmax():
    """BATv1's head: one softmax over both strands and all positions, a count head of
    log1p(total over both strands), strands summed into one track.

    Two consequences pin the arithmetic. The summed track must total expm1(log_counts),
    not twice it -- per-strand softmaxes each scaled by the total would (the 2.00x BPNet
    defect). And the strands must share one distribution: a strand whose logits sit 100
    units lower contributes nothing, which a per-strand softmax would not show.
    """
    log_counts = numpy.log1p(numpy.array([[100.0]]))
    uniform = numpy.zeros((1, 2, 4))
    profile = expected_counts_profile(uniform, log_counts)
    assert profile.shape == (1, 4)
    numpy.testing.assert_allclose(profile.sum(), 100.0, rtol=1e-9)
    numpy.testing.assert_allclose(profile, 25.0, rtol=1e-9)

    lopsided = numpy.zeros((1, 2, 4))
    lopsided[0, 1, :] = -100.0
    profile = expected_counts_profile(lopsided, log_counts)
    numpy.testing.assert_allclose(profile.sum(), 100.0, rtol=1e-9)
    one_strand = expected_counts_profile(lopsided[:, :1, :], log_counts)
    numpy.testing.assert_allclose(profile, one_strand, rtol=1e-9)

    # And the sum is of the shared helper's own (N, L, 2) output, so the two agree.
    from chorus.core.count_head import expected_counts_profile as shared
    rng = numpy.random.RandomState(1)
    logits = rng.randn(3, 2, 16)
    counts = numpy.log1p(numpy.full((3, 1), 50.0))
    numpy.testing.assert_allclose(
        expected_counts_profile(logits, counts),
        shared(logits.transpose(0, 2, 1), counts, n_tracks=1).sum(axis=2),
        rtol=1e-12,
    )


def test_expected_counts_is_shift_invariant_in_logits():
    """Mean-centring must not change the result."""
    rng = numpy.random.RandomState(0)
    logits = rng.randn(3, 1, 16)
    counts = numpy.log1p(numpy.full((3, 1), 50.0))
    numpy.testing.assert_allclose(
        expected_counts_profile(logits, counts),
        expected_counts_profile(logits + 7.5, counts),
        rtol=1e-9,
    )


def test_score_window_sum_matches_chrombpnet_window():
    """Must reproduce the ChromBPNet builder's window exactly, including
    its centre-of-output (not centre-of-variant) placement."""
    profile = numpy.zeros(CATV1_OUTPUT_LENGTH)
    profile[250:751] = 1.0
    assert score_window_sum(profile) == pytest.approx(501.0)
    assert CATV1_SCORING_WINDOW_BP == 501

    # One base outside the window on either side must be excluded.
    edges = numpy.zeros(CATV1_OUTPUT_LENGTH)
    edges[249] = 1.0
    edges[751] = 1.0
    assert score_window_sum(edges) == pytest.approx(0.0)

    ones = numpy.ones(CATV1_OUTPUT_LENGTH)
    assert score_window_sum(ones) == pytest.approx(501.0)


def test_compute_effect_log2fc():
    assert compute_effect(0.0, 0.0) == pytest.approx(0.0)
    # (3+1)/(1+1) = 2 -> log2 = 1
    assert compute_effect(1.0, 3.0) == pytest.approx(1.0)
    assert compute_effect(3.0, 1.0) == pytest.approx(-1.0)


def test_compute_effect_matches_chorus_scorer():
    """The builder's effect formula and the runtime scorer's must agree."""
    from chorus.analysis.scorers import _compute_effect
    from chorus.oracles.cherimoya_source.scoring import PSEUDOCOUNT

    for ref, alt in [(0.0, 5.0), (5.0, 0.0), (12.5, 13.75), (100.0, 3.0)]:
        assert compute_effect(ref, alt) == pytest.approx(
            _compute_effect(ref, alt, "log2fc", PSEUDOCOUNT)
        )


def test_layer_config_agrees_with_our_constants():
    """Our window/pseudocount must match the layer chorus will score with."""
    from chorus.analysis.scorers import LAYER_CONFIGS
    from chorus.oracles.cherimoya_source.scoring import PSEUDOCOUNT

    config = LAYER_CONFIGS["chromatin_accessibility"]
    assert config.window_bp == CATV1_SCORING_WINDOW_BP
    assert config.pseudocount == PSEUDOCOUNT
    assert config.formula == "log2fc"
    assert config.aggregation == "sum"
    assert config.signed is False  # -> signed_flags=False in the NPZ


# ── track assembly ───────────────────────────────────────────────────

def test_transform_places_values_at_trim_offset():
    """Values must start at the trim offset so they line up with the
    prediction interval; a mis-offset here is the failure mode that
    collapsed ChromBPNet variant effects ~4x (see core/base.py)."""
    oracle = _oracle()
    log_counts = numpy.log1p(numpy.array([[1000.0]]))

    for n_strands in (1, 2):
        logits = numpy.zeros((1, n_strands, CATV1_OUTPUT_LENGTH))
        values = oracle._transform_predictions_to_tracks(
            logits, log_counts, CATV1_INPUT_LENGTH,
        )

        assert values.shape == (CATV1_INPUT_LENGTH,)
        assert values[:CATV1_TRIMMING].sum() == 0.0
        assert values[CATV1_TRIMMING + CATV1_OUTPUT_LENGTH:].sum() == 0.0
        window = values[CATV1_TRIMMING:CATV1_TRIMMING + CATV1_OUTPUT_LENGTH]
        assert window.sum() == pytest.approx(1000.0)


def test_geometry_guard_is_per_atlas():
    """A BATv1 checkpoint declares two control tracks and two strands; a CATv1 one none
    and one. Each is accepted under its own assay and refused under the other's, so a
    checkpoint cannot be run with the wrong null or the wrong control."""
    from chorus.core.exceptions import ModelNotLoadedError

    oracle = _oracle()
    oracle.assay = "DNASE"
    oracle._check_geometry(CATV1_TRIMMING, 0, [1])
    with pytest.raises(ModelNotLoadedError, match="control tracks"):
        oracle._check_geometry(CATV1_TRIMMING, BATV1_N_CONTROL_TRACKS, BATV1_SIGNAL_GROUPS)

    oracle.assay = "CHIP"
    assert oracle.atlas == "BATv1"
    oracle._check_geometry(CATV1_TRIMMING, BATV1_N_CONTROL_TRACKS, BATV1_SIGNAL_GROUPS)
    with pytest.raises(ModelNotLoadedError, match="control tracks"):
        oracle._check_geometry(CATV1_TRIMMING, 0, [1])
    with pytest.raises(ModelNotLoadedError, match="signal_groups"):
        oracle._check_geometry(CATV1_TRIMMING, BATV1_N_CONTROL_TRACKS, [1])
    with pytest.raises(ModelNotLoadedError, match="trimming"):
        oracle._check_geometry(CATV1_TRIMMING - 1, BATV1_N_CONTROL_TRACKS, BATV1_SIGNAL_GROUPS)


def test_normalization_key_follows_the_atlas_and_the_fold():
    """Four nulls, selected by what the oracle holds: ranking a CHIP prediction against the
    accessibility null would be the same class of silent error as fold 0 against the
    ensemble."""
    from chorus.oracles.cherimoya_source.catv1_globals import CATV1_ENSEMBLE

    oracle = _oracle()
    assert oracle.atlas is None
    oracle.assay = "DNASE"
    assert oracle.normalization_key == "cherimoya"
    oracle.fold = CATV1_ENSEMBLE
    assert oracle.normalization_key == "cherimoya_ensemble"
    oracle.assay = "CHIP"
    assert oracle.normalization_key == "batv1_ensemble"
    oracle.fold = 0
    assert oracle.normalization_key == "batv1"
    assert oracle._track_metadata()["atlas"] == "BATv1"


# ── windowing / sliding ──────────────────────────────────────────────

def test_window_sequences_geometry():
    """The sliding-window formula lives in one place; pin it."""
    oracle = _oracle()

    # Exactly one window for anything up to the input length, right-padded.
    for length in (500, 1000, CATV1_INPUT_LENGTH):
        windows = oracle._window_sequences("A" * length)
        assert len(windows) == 1
        assert len(windows[0]) == CATV1_INPUT_LENGTH
        assert windows[0].endswith("N" * (CATV1_INPUT_LENGTH - length))

    # Default step is output_length, so windows advance 1000 bp.
    seq = "A" * (CATV1_INPUT_LENGTH + 2 * CATV1_OUTPUT_LENGTH)
    windows = oracle._window_sequences(seq)
    assert len(windows) == 3
    assert all(len(w) == CATV1_INPUT_LENGTH for w in windows)

    # A finer step yields proportionally more windows.
    assert len(oracle._window_sequences(seq, step=500)) == 5


def test_window_sequences_are_offset_correctly():
    oracle = _oracle()
    seq = "".join("ACGT"[i % 4] for i in range(CATV1_INPUT_LENGTH + CATV1_OUTPUT_LENGTH))
    windows = oracle._window_sequences(seq, step=CATV1_OUTPUT_LENGTH)
    assert windows[0] == seq[:CATV1_INPUT_LENGTH]
    assert windows[1] == seq[CATV1_OUTPUT_LENGTH:CATV1_OUTPUT_LENGTH + CATV1_INPUT_LENGTH]


def test_predict_sliding_rejects_bad_step():
    oracle = _oracle()
    oracle.loaded = True  # bypass load; step validation happens first
    oracle.assay, oracle.encode_id = "DNASE", "ENCSR000EOT"
    wide = "A" * (CATV1_INPUT_LENGTH + 4 * CATV1_OUTPUT_LENGTH)
    for bad in (0, -1, CATV1_OUTPUT_LENGTH + 1):
        with pytest.raises(ValueError):
            oracle.predict_sliding(wide, step=bad)


def test_predict_sliding_stitch_alignment():
    """Pin the sliding-window registration without needing a checkpoint.

    Each window is given a distinct constant profile, so the stitched
    track must show those constants in order, in 1000 bp blocks, covering
    exactly the query. This is the offset that ChromBPNet gets wrong for
    bare-string (non-extendible) queries.
    """
    oracle = _oracle()
    oracle.loaded = True
    oracle.assay, oracle.encode_id, oracle.cell_type = "DNASE", "ENCSR000EOT", "K562"

    n_blocks = 4
    query_length = n_blocks * CATV1_OUTPUT_LENGTH
    query = "ACGT" * (query_length // 4)

    # Window k gets a flat profile summing to (k+1)*1000 -> mean k+1.
    def fake_forward(windows):
        n = len(windows)
        logits = numpy.zeros((n, 1, CATV1_OUTPUT_LENGTH))
        totals = numpy.array([[(k + 1) * CATV1_OUTPUT_LENGTH] for k in range(n)])
        return logits, numpy.log1p(totals)

    oracle._forward_windows = fake_forward

    prediction = oracle.predict_sliding(query)
    values = prediction["DNASE:ENCSR000EOT"].values

    assert len(values) == query_length
    for k in range(n_blocks):
        block = values[k * CATV1_OUTPUT_LENGTH:(k + 1) * CATV1_OUTPUT_LENGTH]
        numpy.testing.assert_allclose(block, k + 1, rtol=1e-9)


def test_predict_sliding_left_offset_is_exactly_trimming():
    """The stitch offset assumes the input starts side_pad left of the
    query, for extendible and non-extendible references alike."""
    oracle = _oracle()
    query_length = 3 * CATV1_OUTPUT_LENGTH
    query_interval_len = query_length + CATV1_TRIMMING

    from chorus.core.interval import Interval, Sequence
    interval = Interval.make(Sequence(sequence="A" * query_length))
    staged = interval.extend(query_interval_len, how="left")
    assert len(staged) == query_interval_len
    # The query's first base must sit at index CATV1_TRIMMING.
    assert staged.sequence[:CATV1_TRIMMING] == "N" * CATV1_TRIMMING


# ── registration ─────────────────────────────────────────────────────

def test_registered_in_oracles_dict():
    from chorus.oracles import ORACLES, get_oracle
    from chorus.oracles.cherimoya import CherimoyaOracle

    assert ORACLES["cherimoya"] is CherimoyaOracle
    assert get_oracle("cherimoya") is CherimoyaOracle
    assert get_oracle("CHERIMOYA") is CherimoyaOracle


def test_create_oracle_resolves_cherimoya():
    import chorus
    from chorus.oracles.cherimoya import CherimoyaOracle

    oracle = chorus.create_oracle("cherimoya", use_environment=False)
    assert isinstance(oracle, CherimoyaOracle)
    assert oracle.use_environment is False
    assert chorus.CherimoyaOracle is CherimoyaOracle


def test_registered_in_environment_maps():
    """Cherimoya resolves to its own env, and its dependency probe actually probes something.

    This used to also assert `ORACLE_CLASS_MAP["cherimoya"] == "CherimoyaOracle"`. That map was
    removed: its only consumer assigned the lookup to a local and never read it, because the
    generated script asks the oracle for its own `__class__.__name__` in the child process. The
    assertion was pinning a value nothing consulted.

    The dependency-probe check replaces it, and is the assertion that would have caught a real
    defect — cherimoya was absent from the runner's probe, so `chorus health` reported it Healthy
    without importing torch or cherimoya.
    """
    import sys
    from pathlib import Path

    from chorus.core.environment.manager import EnvironmentManager

    assert EnvironmentManager().get_environment_name("cherimoya") == "chorus-cherimoya"

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_registries_cover_every_oracle import _runner_dependencies

    assert "cherimoya" in _runner_dependencies()


def test_environment_yaml_exists_and_pins_cherimoya():
    """The env pin is part of the reproducibility contract: the CDFs are
    only valid for the checkpoint loader that produced them."""
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[1] / "environments" / "chorus-cherimoya.yml"
    assert path.exists()
    spec = yaml.safe_load(path.read_text())
    assert spec["name"] == "chorus-cherimoya"

    pip_deps = []
    for dep in spec["dependencies"]:
        if isinstance(dep, dict) and "pip" in dep:
            pip_deps = dep["pip"]
    assert any(d.startswith("cherimoya==") for d in pip_deps), (
        "cherimoya must be pinned exactly, not floated"
    )


def test_registered_in_backgrounds_and_probes():
    from chorus.cli._backgrounds import _KNOWN_ORACLES
    from chorus.core.weights_probe import _ARTIFACT_PROBES

    assert "cherimoya" in _KNOWN_ORACLES
    assert "batv1" in _KNOWN_ORACLES, "the BATv1 null is keyed 'batv1'; status must list it"
    assert "cherimoya" in _ARTIFACT_PROBES


def test_prefetch_load_kwargs_are_valid():
    """A bare load_pretrained_model() would raise, so setup needs kwargs.

    Guards against `chorus setup --oracle cherimoya` failing at the
    weight-prefetch step.
    """
    from chorus.cli._setup_prefetch import _DEFAULT_LOAD_KWARGS
    from chorus.oracles.cherimoya_source.catv1_metadata import get_metadata

    entries = _DEFAULT_LOAD_KWARGS["cherimoya"]
    assert isinstance(entries, list) and entries

    meta = get_metadata()
    assert {e["assay"] for e in entries} == {"ATAC", "DNASE", "CHIP"}
    for entry in entries:
        assert {"assay", "cell_type"} <= set(entry)
        if entry["assay"] == "CHIP":
            assert "TF" in entry
        # Must actually resolve, or setup fails at prefetch time.
        assay, accession = meta.resolve(
            assay=entry["assay"], cell_type=entry["cell_type"], tf=entry.get("TF"),
        )
        assert assay == entry["assay"]
        assert accession.startswith("ENCSR")


def test_mcp_oracle_spec_matches_the_oracle():
    from chorus.mcp.server import ORACLE_SPECS

    spec = ORACLE_SPECS["cherimoya"]
    oracle = _oracle()
    assert spec["framework"] == "PyTorch"
    assert spec["input_size_bp"] == oracle.sequence_length
    assert spec["output_bins"] == oracle.output_length
    assert spec["resolution_bp"] == oracle.bin_size
    assert sorted(spec["assay_types"]) == sorted(oracle.list_assay_types())


def test_mcp_list_tracks_search_and_summary():
    from chorus.mcp.server import list_tracks

    summary = list_tracks("cherimoya")
    assert summary["num_tracks"] == 1518 + 3429
    assert summary["track_summary"] == {"ATAC": 369, "CHIP": 3429, "DNASE": 1149}
    assert len(summary["cell_types"]) == 407
    assert len(summary["chip_cell_types"]) == 183
    assert summary["num_targets"] == 1139
    assert "query" in summary["note"] and "TF=" in summary["note"]

    hit = list_tracks("cherimoya", query="ENCSR000EOT")
    assert hit["num_results"] == 1
    assert hit["tracks"][0]["track_id"] == "DNASE:ENCSR000EOT"
    assert hit["tracks"][0]["biosample"] == "K562"
    assert hit["tracks"][0]["atlas"] == "CATv1"

    chip = list_tracks("cherimoya", query="ENCSR000EGM")
    assert chip["num_results"] == 1
    assert chip["tracks"][0]["track_id"] == "CHIP:ENCSR000EGM"
    assert (chip["tracks"][0]["target"], chip["tracks"][0]["atlas"]) == ("CTCF", "BATv1")
    assert chip["tracks"][0]["quality_flags"] == ""
    gata1 = list_tracks("cherimoya", query="GATA1")
    assert gata1["num_results"] >= 2 and all(t["assay"] == "CHIP" for t in gata1["tracks"])

    # Results are capped so a broad query cannot flood an LLM context.
    broad = list_tracks("cherimoya", query="cell")
    assert len(broad["tracks"]) <= 200
    assert broad["num_results"] >= len(broad["tracks"])


def test_mcp_load_oracle_can_pin_an_experiment():
    """An agent that finds a track via list_tracks must be able to load it.

    `(assay, cell_type)` resolves to the committed default, so without an
    `encode_id` parameter the other 1,026 experiments are unreachable
    through the MCP surface even though `list_tracks` advertises them.
    """
    import inspect

    from chorus.mcp.server import load_oracle

    params = inspect.signature(load_oracle).parameters
    assert "encode_id" in params, (
        "load_oracle must accept encode_id or non-default CATv1 experiments "
        "cannot be reached via MCP"
    )
    assert "Cherimoya" in load_oracle.__doc__


def test_mcp_docstrings_mention_cherimoya():
    """MCP tool docstrings are LLM-facing: an omitted oracle is invisible."""
    from chorus.mcp.server import list_tracks, load_oracle

    for tool in (list_tracks, load_oracle):
        assert "cherimoya" in tool.__doc__.lower()


# ── background builder ───────────────────────────────────────────────

def _builder_source():
    from pathlib import Path
    path = (Path(__file__).resolve().parents[1] / "scripts"
            / "build_backgrounds_cherimoya.py")
    assert path.exists(), "background builder script is missing"
    return path.read_text()


def test_builder_exists_and_parses():
    import ast
    ast.parse(_builder_source())


def test_builder_defaults_to_dhs_on():
    """DHS sampling must stay ON by default.

    The published ChromBPNet CDFs contain the DHS components
    (effect_counts=18672, summary_counts=34004). Building without them
    yields 9609/29004, which is silently non-comparable to every other
    oracle -- the exact failure the shared sampling exists to prevent.
    """
    source = _builder_source()
    assert "parser.set_defaults(dhs=True)" in source
    assert '"--no-dhs"' in source
    # An `action="store_true"` --dhs flag would mean off-by-default again.
    assert '"--dhs", dest="dhs", action="store_true"' not in source


def test_builder_reuses_the_shared_scoring_transform():
    """The builder must not reimplement the count transform.

    A CDF is only meaningful if the value it was built from is computed the
    way the query path computes it; importing the same helpers is what
    makes that structural rather than a convention.
    """
    source = _builder_source()
    assert "from chorus.oracles.cherimoya_source.scoring import" in source
    for symbol in ("expected_counts_profile", "score_window_sum", "compute_effect"):
        assert symbol in source
    # And it must cross-check its fast device-side path against them.
    assert "assert_allclose" in source


def test_builder_pins_the_device():
    """Auto-detect could silently fall back to CPU, which disagrees with
    the Triton path by ~1e-2 and would corrupt individual CDF rows."""
    source = _builder_source()
    assert '"--device", default="cuda"' in source


def test_builder_replicates_chrombpnet_seeds():
    """Comparability depends on identical sampling, not merely similar.

    The position/allele seeds are still literals in the builder. The RESERVOIR
    seed moved: it used to be a `random.Random(12345)` literal here, and now
    lives once in `chorus.analysis.background_sampling.DEFAULT_SEED`, which both
    this builder and ChromBPNet's get by importing `ReservoirSampler` (#125).
    Asserting the shared import is a stronger guarantee than grepping for the
    literal — the two builders can no longer drift apart at all — so this checks
    the import plus the shared default, rather than a string that is now absent
    by design.
    """
    source = _builder_source()
    for seed in ("random.seed(42)", "seed=43", "random.seed(44)", "seed=456",
                 "random.seed(789)", "random.Random(111)", "seed=567",
                 "RandomState(999)"):
        assert seed in source, f"missing ChromBPNet seed: {seed}"

    assert "from chorus.analysis.background_sampling import" in source, (
        "builder must get its reservoir from the shared module so its seed "
        "cannot drift from ChromBPNet's"
    )
    from chorus.analysis.background_sampling import DEFAULT_SEED
    assert DEFAULT_SEED == 12345, (
        "the shared reservoir seed changed; every oracle's background would now "
        "be sampled differently from the ones on HuggingFace"
    )


def test_builder_accepts_the_flags_the_cli_passes():
    """`chorus backgrounds build` hands the script a fixed flag set.

    cli/_backgrounds.py builds `mamba run -n <env> python <script> --part both
    [--only-missing] [--gpu N]`. A build script that does not accept those
    dies with 'unrecognized arguments', so the documented CLI path fails even
    though the script works when invoked directly. Regression for --gpu.
    """
    source = _builder_source()
    for flag in ('"--part"', '"--gpu"', '"--only-missing"'):
        assert flag in source, f"builder must accept {flag} for the CLI path"


def test_builder_gpu_flag_is_applied_before_torch_import():
    """--gpu has to set CUDA_VISIBLE_DEVICES at argparse time.

    Setting it after torch initialises CUDA has no effect, so the pinning
    would silently do nothing.
    """
    source = _builder_source()
    env_line = source.index('os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)')
    build_def = source.index("def build(")
    assert env_line < build_def, (
        "CUDA_VISIBLE_DEVICES must be set at module scope, before torch is "
        "imported inside build()"
    )


# ── Triton autotune cache ────────────────────────────────────────────

def _template_source(name):
    from pathlib import Path
    path = (Path(__file__).resolve().parents[1] / "chorus" / "oracles"
            / "cherimoya_source" / "templates" / name)
    assert path.exists(), f"{name} is missing"
    return path.read_text()


def test_enable_autotune_cache_returns_bool_and_never_raises():
    """The helper must degrade to a no-op rather than fail.

    It runs on every load, including on macOS and CPU-only installs where
    Triton is absent, so an ImportError here would break loading entirely.
    """
    from chorus.oracles.cherimoya_source._triton_autotune import (
        enable_autotune_cache,
    )
    assert isinstance(enable_autotune_cache(), bool)


@pytest.mark.parametrize("name", ["load_template.py", "predict_template.py"])
def test_templates_set_autotune_cache_before_cherimoya_import(name):
    """The knob must be set before `import cherimoya`.

    triton.autotune reads knobs.autotuning.cache in Autotuner.__init__,
    which runs when the decorators are evaluated at import time. Setting it
    afterwards is silently useless -- every call would go back to
    re-benchmarking (~7.2s per call on CATv1), with no error to notice.
    """
    source = _template_source(name)
    knob = source.index("knobs.autotuning.cache = True")
    import_cherimoya = source.index("from cherimoya import Cherimoya")
    assert knob < import_cherimoya, (
        f"{name}: knobs.autotuning.cache must be set before cherimoya is "
        f"imported, otherwise the autotune cache never takes effect"
    )


def test_load_direct_enables_autotune_cache_before_import():
    """Same ordering constraint on the in-process path (builder + tests)."""
    import inspect
    from chorus.oracles.cherimoya import CherimoyaOracle

    source = inspect.getsource(CherimoyaOracle._load_direct)
    enable = source.index("enable_autotune_cache()")
    import_cherimoya = source.index("from cherimoya import Cherimoya")
    assert enable < import_cherimoya, (
        "_load_direct must enable the autotune cache before importing "
        "cherimoya"
    )
