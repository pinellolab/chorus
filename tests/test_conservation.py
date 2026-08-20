"""Tests for chorus.analysis.conservation (GPN-Star conservation tracks).

Model-free and network-free: the download/cache logic is exercised against
a mocked ``hf_hub_download`` that drops a tiny fixture file at the same
repo-relative nested path HuggingFace actually uses, and the bigwig-reading
logic is exercised against a small locally-written bigwig fixture — no real
9.9 GB download or network access needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chorus.analysis import conservation


def _write_fixture_bigwig(path, chrom="chr1", chrom_size=1000, start=100, values=None):
    import pyBigWig

    if values is None:
        values = [float(i) for i in range(20)]
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([(chrom, chrom_size)])
    bw.addEntries(chrom, [start + i for i in range(len(values))], values=values, span=1, step=1)
    bw.close()


def test_gpn_star_bigwig_path_downloads_and_flattens(tmp_path, monkeypatch):
    downloads_dir = tmp_path / "downloads"
    cfg = conservation._TRACK_SOURCES["gpn_star"]

    calls = {"n": 0}

    # Mimic hf_hub_download's real behaviour: it preserves the repo-relative
    # subpath under local_dir, landing the file nested rather than flat.
    def fake_hf_hub_download(repo_id, filename, repo_type, local_dir):
        calls["n"] += 1
        assert repo_id == cfg["hf_repo"]
        assert filename == cfg["hf_filename"]
        assert repo_type == "dataset"
        nested = Path(local_dir) / filename
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_bytes(b"fake-bigwig-bytes")
        return str(nested)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    assert not conservation.has_gpn_star_bigwig(downloads_dir)

    path = conservation.gpn_star_bigwig_path(downloads_dir)

    assert path == downloads_dir / "gpn_star" / "entropy.bw"
    assert path.exists()
    assert path.read_bytes() == b"fake-bigwig-bytes"
    # The nested HF repo-relative directories should be cleaned up.
    assert not (downloads_dir / "gpn_star" / "bigwig").exists()
    assert calls["n"] == 1
    assert conservation.has_gpn_star_bigwig(downloads_dir)

    # Second call must be a pure cache hit — no network call.
    path2 = conservation.gpn_star_bigwig_path(downloads_dir)
    assert path2 == path
    assert calls["n"] == 1


@pytest.mark.parametrize(
    "track,path_fn,has_fn",
    [
        ("phylop100way", conservation.phylop_bigwig_path, conservation.has_phylop_bigwig),
        ("phastcons100way", conservation.phastcons_bigwig_path, conservation.has_phastcons_bigwig),
    ],
)
def test_url_source_bigwig_path_downloads(tmp_path, monkeypatch, track, path_fn, has_fn):
    downloads_dir = tmp_path / "downloads"
    cfg = conservation._TRACK_SOURCES[track]

    calls = {"n": 0}

    def fake_download_with_resume(url, dest, **kwargs):
        calls["n"] += 1
        assert url == cfg["url"]
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"fake-ucsc-bigwig-bytes")

    import chorus.utils.http as http_mod
    monkeypatch.setattr(http_mod, "download_with_resume", fake_download_with_resume)

    assert not has_fn(downloads_dir)
    path = path_fn(downloads_dir)

    assert path == downloads_dir / cfg["local_subdir"] / cfg["local_filename"]
    assert path.read_bytes() == b"fake-ucsc-bigwig-bytes"
    assert calls["n"] == 1
    assert has_fn(downloads_dir)

    # Second call is a pure cache hit — no network call.
    path_fn(downloads_dir)
    assert calls["n"] == 1


def test_list_tracks_reports_status_without_downloading(tmp_path, monkeypatch):
    downloads_dir = tmp_path / "downloads"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("list_tracks must not trigger a download")

    import huggingface_hub
    import chorus.utils.http as http_mod
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail_if_called)
    monkeypatch.setattr(http_mod, "download_with_resume", fail_if_called)

    info = conservation.list_tracks(downloads_dir)

    assert set(info.keys()) == {
        "gpn_star", "gpn_star_llr_a", "gpn_star_llr_c", "gpn_star_llr_g", "gpn_star_llr_t",
        "phylop100way", "phastcons100way",
    }
    for track, status in info.items():
        assert status["downloaded"] is False
        assert status["size_bytes"] is None
        assert status["size_note"]  # non-empty human-readable size estimate
        assert status["source"] in ("hf", "url")

    # Pre-create one track's file to confirm it flips to downloaded=True
    # with the right size, while the others remain not-downloaded.
    gpn_path = info["gpn_star"]["path"]
    gpn_path.parent.mkdir(parents=True, exist_ok=True)
    gpn_path.write_bytes(b"x" * 1234)

    info2 = conservation.list_tracks(downloads_dir)
    assert info2["gpn_star"]["downloaded"] is True
    assert info2["gpn_star"]["size_bytes"] == 1234
    assert info2["phylop100way"]["downloaded"] is False


def test_download_track_dispatches_by_name(tmp_path, monkeypatch):
    downloads_dir = tmp_path / "downloads"
    calls = []

    def fake_hf_hub_download(repo_id, filename, repo_type, local_dir):
        calls.append("hf")
        nested = Path(local_dir) / filename
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_bytes(b"data")
        return str(nested)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    path = conservation.download_track("gpn_star", downloads_dir)
    assert path == downloads_dir / "gpn_star" / "entropy.bw"
    assert calls == ["hf"]


def test_download_track_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown conservation track"):
        conservation.download_track("not_a_real_track")


def test_read_entropy_values_fills_gaps_with_zero(tmp_path):
    bw_path = tmp_path / "entropy.bw"
    values = [0.1 * i for i in range(10)]  # positions 100..109 (0-based)
    _write_fixture_bigwig(bw_path, start=100, values=values)

    # 1-based inclusive region spanning the covered stretch plus gaps on
    # both sides.
    result = conservation.read_entropy_values("chr1", 95, 115, bw_path=bw_path)

    assert len(result) == 115 - 95 + 1
    assert not np.any(np.isnan(result))
    # Positions 95-100 (1-based) => 0-based 94-99, all uncovered -> 0.
    assert np.allclose(result[:5], 0.0)
    # 0-based position 100 is 1-based position 101 -> index 101-95=6.
    assert result[6] == pytest.approx(values[0])
    assert result[15] == pytest.approx(values[9])
    # Tail beyond the covered stretch is uncovered -> 0.
    assert np.allclose(result[16:], 0.0)


def test_draw_logo_draws_only_positive_importance(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ref_seq = "ACGTN"
    importance = [0.2, 0.0, 0.8, -0.1, 0.5]
    conservation._draw_logo(ax, ref_seq, importance, start=1000)

    # Zero and negative importances are skipped; 3 of 5 positions draw.
    assert len(ax.patches) == 3
    assert ax.get_xlim() == (1000.0, 1005.0)
    assert ax.get_ylim()[0] == pytest.approx(0.0)
    assert ax.get_ylim()[1] == pytest.approx(0.8 * 1.05)
    plt.close(fig)


def test_conservation_igv_features_no_aggregation_for_small_window(tmp_path):
    bw_path = tmp_path / "entropy.bw"
    values = [1.0] * 50
    _write_fixture_bigwig(bw_path, start=0, values=values)

    features = conservation.conservation_igv_features(
        "chr1", 1, 50, bw_path=bw_path, max_window_bp=20_000,
    )

    # 1bp resolution, no skip_zeros -> one feature per base, none dropped.
    assert len(features) == 50
    assert all(f["end"] - f["start"] == 1 for f in features)
    # Our start/end args are 1-based inclusive; IGV feature coords must be
    # 0-based (matches every other track built in _igv_report.py) — the
    # first feature for region [1, 50] should start at 0-based position 0.
    assert features[0]["start"] == 0
    assert features[-1]["end"] == 50


def test_conservation_igv_features_clips_large_window_around_center(tmp_path):
    # Regression test: a real bug report showed identical entropy values
    # repeating in 52bp blocks when AlphaGenome's ~1Mb prediction window
    # (1_048_576 // 20_000 == 52) was mean-aggregated down to a feature-count
    # cap. Verify the fix — clipping to a bounded window at true 1bp
    # resolution — instead of ever averaging blocks of bases together.
    bw_path = tmp_path / "entropy.bw"
    n = 40_000
    # A ramp, not a constant, so aggregation vs. clipping is distinguishable:
    # if any two adjacent raw values got averaged, the returned values
    # would repeat; a ramp read at 1bp resolution never repeats.
    values = [float(i) for i in range(n)]
    _write_fixture_bigwig(bw_path, chrom_size=n + 1000, start=0, values=values)

    center = 20_000
    features = conservation.conservation_igv_features(
        "chr1", 1, n, bw_path=bw_path, center=center, max_window_bp=1_000,
    )

    # Clipped to ~max_window_bp (1001 = the inclusive-both-ends [c-500, c+500]
    # span), not mean-aggregated across the full 40,000bp.
    assert len(features) == 1_001
    # Every returned value is 1bp of true resolution: no two consecutive
    # features share a value (would happen if bases got averaged together).
    vals = [f["value"] for f in features]
    assert len(set(vals)) == len(vals)
    # Clipped window is centered on `center` (1-based -> 0-based feature coords).
    assert features[0]["start"] == center - 1 - 500
    assert features[-1]["end"] == center + 500


def test_conservation_igv_features_clip_respects_region_bounds(tmp_path):
    # When center is near the edge of [start, end], the clipped window must
    # not extend past the requested region.
    bw_path = tmp_path / "entropy.bw"
    n = 5_000
    values = [float(i) for i in range(n)]
    _write_fixture_bigwig(bw_path, chrom_size=n + 1000, start=0, values=values)

    features = conservation.conservation_igv_features(
        "chr1", 1, n, bw_path=bw_path, center=50, max_window_bp=1_000,
    )

    assert features[0]["start"] == 0  # clamped to region start, not center-500
    assert features[-1]["end"] <= n


def test_conservation_igv_features_default_caps_to_default_window(tmp_path):
    # Default max_window_bp=DEFAULT_MAX_WINDOW_BP: caller doesn't need to
    # pass anything to get a bounded window around the variant (matches the
    # user-facing "cap the conservation track" behavior). Uses the real
    # module default directly rather than monkeypatching it — the default
    # is bound into the function signature at import time, so patching the
    # module attribute afterward would not change already-bound defaults.
    assert conservation.DEFAULT_MAX_WINDOW_BP == 100_000
    n = conservation.DEFAULT_MAX_WINDOW_BP + 40_000
    bw_path = tmp_path / "entropy.bw"
    values = [float(i) for i in range(n)]
    _write_fixture_bigwig(bw_path, chrom_size=n + 1000, start=0, values=values)

    center = n // 2
    features = conservation.conservation_igv_features(
        "chr1", 1, n, bw_path=bw_path, center=center,
    )

    # Capped to ~DEFAULT_MAX_WINDOW_BP, not the full region.
    assert len(features) < n
    assert abs(len(features) - conservation.DEFAULT_MAX_WINDOW_BP) <= 1
    half = conservation.DEFAULT_MAX_WINDOW_BP // 2
    assert features[0]["start"] == center - 1 - half
    assert features[-1]["end"] == center + half


def test_conservation_igv_features_no_cap_below_default_window(tmp_path):
    # A region smaller than DEFAULT_MAX_WINDOW_BP is never clipped.
    bw_path = tmp_path / "entropy.bw"
    n = 40_000
    values = [1.0] * n
    _write_fixture_bigwig(bw_path, chrom_size=n + 1000, start=0, values=values)

    features = conservation.conservation_igv_features("chr1", 1, n, bw_path=bw_path)

    assert features[0]["start"] == 0
    assert features[-1]["end"] == n


def test_conservation_igv_features_explicit_none_disables_cap(tmp_path):
    n = conservation.DEFAULT_MAX_WINDOW_BP + 40_000
    bw_path = tmp_path / "entropy.bw"
    values = [1.0] * n
    _write_fixture_bigwig(bw_path, chrom_size=n + 1000, start=0, values=values)

    features = conservation.conservation_igv_features(
        "chr1", 1, n, bw_path=bw_path, max_window_bp=None,
    )

    assert features[0]["start"] == 0
    assert features[-1]["end"] == n


def test_apply_transform_raw_is_identity():
    values = np.array([0.3, 0.8, 1.1, 0.0])
    result = conservation._apply_transform(values, "raw")
    assert list(result) == list(values)


def test_apply_transform_invert_is_clip_one_minus_entropy():
    # invert: clip(1 - entropy, 0, 1) — a FIXED conservation score using
    # the documented "entropy ~1.0 = neutral" reference point, not a
    # window-relative max. entropy=1 -> 0 (neutral); entropy=0 -> 1 (fully
    # constrained); entropy>1 clips to 0; entropy<0 clips to 1.
    values = np.array([0.3, 0.8, 1.1, 0.0, -0.2])
    result = conservation._apply_transform(values, "invert")

    assert list(result) == pytest.approx([0.7, 0.2, 0.0, 1.0, 1.0])
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)
    # Among values still within [0, 1], lowest raw entropy -> tallest.
    assert result[:3].argmax() == values[:3].argmin()
    assert ((result >= 0) & (result <= 1)).all()  # always in [0, 1]


def test_apply_transform_invert_is_window_independent():
    # Unlike a window-relative max, the same raw entropy value must
    # transform identically regardless of what else is in the window —
    # this is exactly the property that fixes the "baseline around 0.5"
    # symptom (a window-relative transform shifted the baseline depending
    # on the window's own max).
    same_value = 0.4
    low_context = np.array([same_value, 0.1, 0.2])
    high_context = np.array([same_value, 1.6, 1.8])

    r_low = conservation._apply_transform(low_context, "invert")
    r_high = conservation._apply_transform(high_context, "invert")

    assert r_low[0] == pytest.approx(r_high[0])
    assert r_low[0] == pytest.approx(1.0 - same_value)


def test_apply_transform_unknown_raises():
    with pytest.raises(ValueError):
        conservation._apply_transform(np.array([1.0]), "bogus")


def test_conservation_igv_features_transform(tmp_path):
    bw_path = tmp_path / "entropy.bw"
    values = [0.3, 0.8, 1.1, 0.0]
    _write_fixture_bigwig(bw_path, start=0, values=values)

    inverted = conservation.conservation_igv_features(
        "chr1", 1, 4, bw_path=bw_path, transform="invert",
    )
    raw_features = conservation.conservation_igv_features(
        "chr1", 1, 4, bw_path=bw_path, transform="raw",
    )

    assert [f["value"] for f in inverted] == pytest.approx([0.7, 0.2, 0.0, 1.0])
    assert [f["value"] for f in raw_features] == pytest.approx(values)
    assert all(0 <= f["value"] <= 1 for f in inverted)


@pytest.mark.parametrize(
    "feature_fn",
    [conservation.phylop_igv_features, conservation.phastcons_igv_features],
)
def test_phylop_phastcons_igv_features_are_raw_and_capped(tmp_path, feature_fn):
    n = conservation.DEFAULT_MAX_WINDOW_BP + 40_000
    bw_path = tmp_path / "track.bw"
    values = [float(i % 7) for i in range(n)]  # non-trivial, no accidental negatives
    _write_fixture_bigwig(bw_path, chrom_size=n + 1000, start=0, values=values)

    center = n // 2
    features = feature_fn("chr1", 1, n, bw_path=bw_path, center=center)

    # Capped by default (same DEFAULT_MAX_WINDOW_BP as GPN-Star), and never
    # negated — "no transformation, just raw values".
    assert len(features) < n
    assert all(f["value"] >= 0 for f in features)


@pytest.mark.parametrize(
    "track_fn",
    [
        conservation.conservation_coolbox_track,
        conservation.phylop_coolbox_track,
        conservation.phastcons_coolbox_track,
    ],
)
def test_coolbox_tracks_render(tmp_path, track_fn):
    import matplotlib
    matplotlib.use("Agg")
    from coolbox.api import XAxis

    bw_path = tmp_path / "track.bw"
    _write_fixture_bigwig(bw_path, start=0, values=[float(i) for i in range(20)])

    frame = track_fn(bw_path=bw_path) + XAxis()
    fig = frame.plot("chr1:1-20")
    assert fig is not None


def _write_llr_fixtures(tmp_path, chrom, values_by_base, chrom_size=1000):
    paths = {}
    for base, values in values_by_base.items():
        p = tmp_path / f"llr_{base}.bw"
        _write_fixture_bigwig(p, chrom=chrom, chrom_size=chrom_size, start=0, values=values)
        paths[base] = p
    return paths


def test_read_llr_values_returns_dict_keyed_by_base(tmp_path):
    paths = _write_llr_fixtures(tmp_path, "chr1", {
        "A": [0.0] * 5, "C": [1.0] * 5, "G": [2.0] * 5, "T": [3.0] * 5,
    })

    result = conservation.read_llr_values("chr1", 1, 5, bw_paths=paths)

    assert set(result.keys()) == {"A", "C", "G", "T"}
    assert list(result["A"]) == pytest.approx([0.0] * 5)
    assert list(result["G"]) == pytest.approx([2.0] * 5)
    assert list(result["T"]) == pytest.approx([3.0] * 5)


def test_compute_stacked_logo_heights_matches_documented_softmax_rule(tmp_path):
    chrom = "chr1"
    entropy_values = [0.0, 1.0, 2.0, 0.5]  # positions 1..4 (1-based): H=0, neutral, max, mid
    entropy_path = tmp_path / "entropy.bw"
    _write_fixture_bigwig(entropy_path, chrom=chrom, start=0, values=entropy_values)

    # Position index 2 (1-based pos 3) is ref='G' but llr_G is a large
    # nonzero 1.0 there — deliberately not already zero, to prove this
    # function forces the reference logit to 0 itself rather than trusting
    # the bigwig to already encode that.
    llr_values = {
        "A": [0.0, -1.0, 3.0, 2.0],
        "C": [-1.0, 0.0, -2.0, -0.5],
        "G": [-2.0, 2.0, 1.0, 0.0],
        "T": [-3.0, -0.5, 0.0, -1.0],
    }
    llr_paths = _write_llr_fixtures(tmp_path, chrom, llr_values)

    fa_path = tmp_path / "ref.fa"
    ref_seq = "ACGT"
    fa_path.write_text(f">{chrom}\n" + ref_seq + "N" * 996 + "\n")

    heights = conservation.compute_stacked_logo_heights(
        chrom, 1, 4, genome_fasta=fa_path, entropy_bw_path=entropy_path, llr_bw_paths=llr_paths,
    )

    assert set(heights.keys()) == {"A", "C", "G", "T"}
    base_order = ["A", "C", "G", "T"]

    # Independently recompute expected values from the documented rule
    # (github: songlab/gpn-star-scores) rather than re-deriving from the
    # code under test: ref base -> logit 0; alternates -> their llr;
    # stable softmax; height = p(base) * (2 - H).
    for i in range(4):
        logits = np.array([0.0 if b == ref_seq[i] else llr_values[b][i] for b in base_order])
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        h_bits = min(max(entropy_values[i], 0.0), 2.0)
        expected = p * (2.0 - h_bits)
        for j, base in enumerate(base_order):
            assert heights[base][i] == pytest.approx(expected[j]), (i, base)


def test_compute_stacked_logo_heights_forces_reference_logit_to_zero(tmp_path):
    # The upstream bigwigs are documented to already store 0 at the
    # reference base's own position, but this function must not merely
    # assume that — it forces it explicitly. Prove the override is real
    # (not a no-op) by deliberately setting the ref base's own llr to a
    # large nonzero value that would otherwise swamp the softmax.
    chrom = "chr1"
    entropy_path = tmp_path / "entropy.bw"
    _write_fixture_bigwig(entropy_path, chrom=chrom, start=0, values=[1.0])  # (2-H)=1, nonzero

    llr_values = {"A": [5.0], "C": [-1.0], "G": [-2.0], "T": [-3.0]}  # ref='A', llr_A wrongly nonzero
    llr_paths = _write_llr_fixtures(tmp_path, chrom, llr_values)

    fa_path = tmp_path / "ref.fa"
    fa_path.write_text(f">{chrom}\n" + "A" + "N" * 999 + "\n")

    heights = conservation.compute_stacked_logo_heights(
        chrom, 1, 1, genome_fasta=fa_path, entropy_bw_path=entropy_path, llr_bw_paths=llr_paths,
    )

    base_order = ["A", "C", "G", "T"]
    correct_logits = np.array([0.0, -1.0, -2.0, -3.0])  # A forced to 0, not 5.0
    correct_p = np.exp(correct_logits - correct_logits.max())
    correct_p = correct_p / correct_p.sum()
    for j, base in enumerate(base_order):
        assert heights[base][0] == pytest.approx(correct_p[j] * 1.0)

    # Without the override, A's raw llr=5.0 would swamp the softmax —
    # a very different (and wrong) result from the one asserted above.
    buggy_logits = np.array([5.0, -1.0, -2.0, -3.0])
    buggy_p = np.exp(buggy_logits - buggy_logits.max())
    buggy_p = buggy_p / buggy_p.sum()
    assert heights["A"][0] != pytest.approx(buggy_p[0] * 1.0)


def test_compute_stacked_logo_heights_clips_entropy_to_0_2_bits(tmp_path):
    chrom = "chr1"
    entropy_path = tmp_path / "entropy.bw"
    _write_fixture_bigwig(entropy_path, chrom=chrom, start=0, values=[-0.5, 2.5])
    llr_paths = _write_llr_fixtures(tmp_path, chrom, {b: [0.0, 0.0] for b in "ACGT"})

    fa_path = tmp_path / "ref.fa"
    fa_path.write_text(f">{chrom}\n" + "AA" + "N" * 998 + "\n")

    heights = conservation.compute_stacked_logo_heights(
        chrom, 1, 2, genome_fasta=fa_path, entropy_bw_path=entropy_path, llr_bw_paths=llr_paths,
    )

    total_pos0 = sum(heights[b][0] for b in "ACGT")
    total_pos1 = sum(heights[b][1] for b in "ACGT")
    # Uniform logits (all 0) -> probabilities always sum to 1, so the
    # stack total is exactly (2 - clipped_H) regardless of which base wins.
    assert total_pos0 == pytest.approx(2.0)  # H=-0.5 clipped to 0 -> 2-0=2
    assert total_pos1 == pytest.approx(0.0)  # H=2.5 clipped to 2 -> 2-2=0


def test_conservation_stacked_logo_igv_features_shape_and_keys(tmp_path):
    chrom = "chr1"
    n = 10
    entropy_path = tmp_path / "entropy.bw"
    _write_fixture_bigwig(entropy_path, chrom=chrom, start=0, values=[0.5] * n)
    llr_paths = _write_llr_fixtures(tmp_path, chrom, {b: [0.0] * n for b in "ACGT"})

    fa_path = tmp_path / "ref.fa"
    fa_path.write_text(f">{chrom}\n" + "A" * n + "N" * (1000 - n) + "\n")

    features = conservation.conservation_stacked_logo_igv_features(
        chrom, 1, n, genome_fasta=fa_path, entropy_bw_path=entropy_path, llr_bw_paths=llr_paths,
        max_window_bp=None,
    )

    assert len(features) == n
    assert all({"chr", "start", "end", "pA", "pC", "pG", "pT"} <= set(f.keys()) for f in features)
    # 0-based feature coordinates, matches every other IGV track built here.
    assert features[0]["start"] == 0
    assert features[-1]["end"] == n
    # Uniform LLRs -> heights sum to exactly (2 - H) = 1.5 at every position.
    for f in features:
        assert f["pA"] + f["pC"] + f["pG"] + f["pT"] == pytest.approx(1.5)


def test_conservation_stacked_logo_igv_features_clips_large_window(tmp_path):
    chrom = "chr1"
    n = conservation.DEFAULT_MAX_WINDOW_BP + 2_000
    entropy_path = tmp_path / "entropy.bw"
    _write_fixture_bigwig(entropy_path, chrom=chrom, chrom_size=n + 1000, start=0, values=[0.5] * n)
    llr_paths = _write_llr_fixtures(tmp_path, chrom, {b: [0.0] * n for b in "ACGT"}, chrom_size=n + 1000)

    fa_path = tmp_path / "ref.fa"
    fa_path.write_text(f">{chrom}\n" + "A" * n + "\n")

    center = n // 2
    features = conservation.conservation_stacked_logo_igv_features(
        chrom, 1, n, genome_fasta=fa_path, entropy_bw_path=entropy_path, llr_bw_paths=llr_paths,
        center=center,
    )

    assert len(features) < n
    assert abs(len(features) - conservation.DEFAULT_MAX_WINDOW_BP) <= 1


def test_draw_stacked_logo_draws_one_patch_per_nonzero_base(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    heights = {
        "A": [0.5, 0.0],
        "C": [0.3, 0.0],
        "G": [0.0, 1.5],
        "T": [0.2, 0.0],
    }
    conservation._draw_stacked_logo(ax, heights, start=1000)

    # Position 0 has 3 nonzero bases (A, C, T); position 1 has 1 (G).
    assert len(ax.patches) == 4
    assert ax.get_xlim() == (1000.0, 1002.0)
    plt.close(fig)


def test_gpn_star_logo_track_js_has_valid_syntax():
    # The custom IGV.js track class is hand-written (not minified/bundled,
    # unlike igv.min.js) and gets inlined verbatim into every report's
    # <script> tag — a syntax error here would silently break the whole
    # report page, not just the one track, so it's worth a cheap check.
    import shutil
    import subprocess

    node = shutil.which("node") or shutil.which("nodejs")
    if node is None:
        pytest.skip("node not available")

    js_path = Path(conservation.__file__).parent / "static" / "gpn_star_logo_track.js"
    result = subprocess.run([node, "--check", str(js_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_gpn_star_logo_track_js_draws_letter_at_correct_y(tmp_path):
    # Regression test for a real rendering bug: drawLetter translated to
    # (x, y + h) instead of (x, y) before scaling+fillText, which shifts
    # the glyph's ink down by its own height h -- global ink ends up at
    # [y+h, y+2h] instead of [y, y+h]. Harmless for short letters (h~1-2px,
    # unnoticeable) but for the tallest/dominant letter in a stack (h can
    # be ~80% of the whole track height) it pushes the glyph almost
    # entirely below the visible track -- exactly why a variant's dominant
    # reference letter disappeared while the three minor letters (small h)
    # still looked fine. Uses a mock canvas context (records
    # translate/scale/fillText calls, stubs measureText) rather than a
    # real renderer, so this doesn't need node-canvas as a dependency.
    import shutil
    import subprocess

    node = shutil.which("node") or shutil.which("nodejs")
    if node is None:
        pytest.skip("node not available")

    js_path = Path(conservation.__file__).parent / "static" / "gpn_star_logo_track.js"
    harness = tmp_path / "harness.js"
    harness.write_text(f"""
const fs = require('fs');
const registry = {{}};
global.igv = {{
    TrackBase: class {{
        constructor(config, browser) {{ this.browser = browser; this.init(config); }}
        init(config) {{ this.config = config; this.height = config.height; }}
    }},
    registerTrackClass(name, cls) {{ registry[name] = cls; }},
}};
eval(fs.readFileSync({str(js_path)!r}, 'utf8'));
const TrackClass = registry['gpnstarstackedlogo'];

// Single base with nonzero height -> sort order is irrelevant, isolating
// just the geometry bug. pA=1.0, MAX_STACK_HEIGHT=2.0, pixelHeight=100
// -> h = 50, and since it's the only nonzero base, y ends up 100-50=50.
const feature = {{ chr: 'chr1', start: 0, end: 1, pA: 1.0, pC: 0, pG: 0, pT: 0 }};

let translateY = null, scaleY = null;
const ctx = {{
    save() {{}}, restore() {{}}, fillText() {{}},
    set fillStyle(v) {{}}, get fillStyle() {{ return undefined; }},
    set font(v) {{}}, get font() {{ return undefined; }},
    translate(x, y) {{ translateY = y; }},
    scale(sx, sy) {{ scaleY = sy; }},
    measureText() {{
        return {{ actualBoundingBoxAscent: 80, actualBoundingBoxDescent: 20, actualBoundingBoxLeft: 5, actualBoundingBoxRight: 65 }};
    }},
}};

const track = new TrackClass({{ height: 100, features: [feature] }}, {{}});
track.draw({{ features: [feature], context: ctx, bpPerPixel: 1, bpStart: 0, pixelWidth: 100, pixelHeight: 100 }});

const expectedY = 50; // the box's intended top -- see comment above
if (translateY !== expectedY) {{
    console.error(`translate y = ${{translateY}}, expected ${{expectedY}} (box top) -- got y+h=${{translateY}} instead, the glyph is shifted down by its own height`);
    process.exit(1);
}}
""")
    result = subprocess.run([node, str(harness)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_gpn_star_logo_track_js_skips_fillText_when_zoomed_out(tmp_path):
    # Perf regression guard: fillText is expensive (font shaping +
    # rasterization) compared to fillRect. At the full ~100,000bp
    # conservation window that's up to 4 fillText calls x 100,001
    # positions on every single redraw (every pan/zoom) if letters are
    # always drawn regardless of zoom -- exactly what made the report
    # sluggish to redraw. Below LETTER_ZOOM_THRESHOLD each position is
    # sub-pixel wide anyway (illegible as text), so the track must fall
    # back to cheap fillRect bars -- same threshold igv.js's own native
    # "dynseq" wig graph type uses for the same reason.
    import shutil
    import subprocess

    node = shutil.which("node") or shutil.which("nodejs")
    if node is None:
        pytest.skip("node not available")

    js_path = Path(conservation.__file__).parent / "static" / "gpn_star_logo_track.js"
    harness = tmp_path / "harness.js"
    harness.write_text(f"""
const fs = require('fs');
const registry = {{}};
global.igv = {{
    TrackBase: class {{
        constructor(config, browser) {{ this.browser = browser; this.init(config); }}
        init(config) {{ this.config = config; this.height = config.height; }}
    }},
    registerTrackClass(name, cls) {{ registry[name] = cls; }},
}};
eval(fs.readFileSync({str(js_path)!r}, 'utf8'));
const TrackClass = registry['gpnstarstackedlogo'];

const features = [{{ chr: 'chr1', start: 0, end: 1, pA: 1.0, pC: 0.3, pG: 0.2, pT: 0.1 }}];

function makeContext() {{
    let fillTextCalls = 0, fillRectCalls = 0;
    const ctx = {{
        save() {{}}, restore() {{}}, translate() {{}}, scale() {{}},
        fillText() {{ fillTextCalls++; }},
        fillRect() {{ fillRectCalls++; }},
        set fillStyle(v) {{}}, get fillStyle() {{ return undefined; }},
        set font(v) {{}}, get font() {{ return undefined; }},
        measureText() {{
            return {{ actualBoundingBoxAscent: 80, actualBoundingBoxDescent: 20, actualBoundingBoxLeft: 5, actualBoundingBoxRight: 65 }};
        }},
    }};
    return {{ ctx, counts: () => ({{ fillTextCalls, fillRectCalls }}) }};
}}

const track = new TrackClass({{ height: 60, features }}, {{}});

const zoomedIn = makeContext();
track.draw({{ features, context: zoomedIn.ctx, bpPerPixel: 0.1, bpStart: 0, pixelWidth: 100, pixelHeight: 60 }});
const inCounts = zoomedIn.counts();

const zoomedOut = makeContext();
track.draw({{ features, context: zoomedOut.ctx, bpPerPixel: 50, bpStart: 0, pixelWidth: 100, pixelHeight: 60 }});
const outCounts = zoomedOut.counts();

if (inCounts.fillTextCalls === 0) {{
    console.error('expected fillText calls when zoomed in (bpPerPixel=0.1), got 0');
    process.exit(1);
}}
if (outCounts.fillTextCalls !== 0) {{
    console.error(`expected NO fillText calls when zoomed out (bpPerPixel=50), got ${{outCounts.fillTextCalls}}`);
    process.exit(1);
}}
if (outCounts.fillRectCalls === 0) {{
    console.error('expected fillRect calls (cheap bar fallback) when zoomed out, got 0');
    process.exit(1);
}}
""")
    result = subprocess.run([node, str(harness)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_stacked_logo_track_type_matches_js_registration(monkeypatch):
    # Regression test for a real bug: igv.js lowercases config.type before
    # looking it up in its track-class registry (createTrack does
    # `t.type.toLowerCase()`), so the "type" string emitted into the track
    # config and the string passed to igv.registerTrackClass in
    # gpn_star_logo_track.js must agree once lowercased. A mismatch here
    # raises no Python-side error at all — igv.js just silently fails at
    # render time with "Error creating track. Could not determine track
    # type for file: [object Object]".
    import json
    import re

    from chorus.analysis import _igv_report

    js_path = Path(conservation.__file__).parent / "static" / "gpn_star_logo_track.js"
    m = re.search(r'igv\.registerTrackClass\(\s*["\']([^"\']+)["\']', js_path.read_text())
    assert m, "could not find an igv.registerTrackClass(...) call in gpn_star_logo_track.js"
    registered_type = m.group(1)

    class FakeInterval:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class FakeReference:
        def __init__(self, start, end):
            self.reference = FakeInterval(start, end)

    class FakeTrack:
        def __init__(self, n):
            self.values = np.zeros(n)
            self.resolution = 1
            self.prediction_interval = FakeReference(1_000_000, 1_000_000 + n)
            self.source_model = "fake"
            self.metadata = {"description": "fake track"}

    track = FakeTrack(50)
    monkeypatch.setattr(conservation, "conservation_igv_features", lambda *a, **k: [])
    monkeypatch.setattr(
        conservation, "conservation_stacked_logo_igv_features",
        lambda *a, **k: [{"chr": "chr1", "start": 0, "end": 1, "pA": 0.1, "pC": 0.2, "pG": 0.3, "pT": 0.4}],
    )
    monkeypatch.setattr(conservation, "phylop_igv_features", lambda *a, **k: [])
    monkeypatch.setattr(conservation, "phastcons_igv_features", lambda *a, **k: [])

    html = _igv_report.build_igv_html(
        {"assay1": track}, {"assay1": track}, "chr1", 1_000_010, "A", "T", show_conservation=True,
    )

    m2 = re.search(
        r'igv\.createBrowser\(\s*document\.getElementById\("igv-div"\),\s*(\{.*?\})\s*\);', html, re.S,
    )
    opts = json.loads(m2.group(1))
    logo_tracks = [t for t in opts["tracks"] if "sequence logo" in t.get("name", "")]
    assert len(logo_tracks) == 1
    configured_type = logo_tracks[0]["type"]

    assert configured_type.lower() == registered_type.lower(), (
        f"track config type {configured_type!r} does not match "
        f"igv.registerTrackClass({registered_type!r}, ...) once lowercased "
        "-- igv.js would fail to create this track at runtime."
    )


def test_sequence_logo_track_fetch_data_returns_stacked_heights(tmp_path):
    from coolbox.utilities.genome import GenomeRange

    chrom = "chr1"
    entropy_path = tmp_path / "entropy.bw"
    _write_fixture_bigwig(entropy_path, chrom=chrom, start=0, values=[0.5] * 4)
    llr_paths = _write_llr_fixtures(tmp_path, chrom, {b: [0.0] * 4 for b in "ACGT"})

    fa_path = tmp_path / "ref.fa"
    fa_path.write_text(f">{chrom}\n" + "ACGT" + "N" * 996 + "\n")

    track = conservation.conservation_logo_track(
        str(fa_path), entropy_bw_path=entropy_path, llr_bw_paths=llr_paths,
    )
    heights = track.fetch_data(GenomeRange("chr1", 1, 4))

    assert set(heights.keys()) == {"A", "C", "G", "T"}
    # Uniform (all-zero) LLRs -> p=0.25 per base; height = 0.25 * (2-0.5) = 0.375.
    for base in "ACGT":
        assert list(heights[base]) == pytest.approx([0.375] * 4)
