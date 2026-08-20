"""Tests for chorus.utils.annotation_store.AnnotationStore.

Model-free and network-free: HuggingFace/URL downloads are mocked (never touch the
network), and genome-build verification uses small locally-written bigwig fixtures
(same helper style as tests/test_conservation.py) rather than the real ~9.9 GB
GPN-Star file.

AnnotationStore is deliberately an *aggregator* — it reads conservation.py's and
AnnotationManager's existing registries at call time rather than duplicating them,
so several tests assert passthrough (it delegates, it doesn't reimplement).
"""
from __future__ import annotations

import pytest
import yaml

from chorus.analysis import conservation
from chorus.core.exceptions import GenomeAssemblyMismatchError
from chorus.utils.annotation_store import AnnotationStore
from chorus.utils.annotations import AnnotationManager
from chorus.utils.genome import ASSEMBLY_CHR1_LENGTH


def _write_fixture_bigwig(path, chrom="chr1", chrom_size=1000, values=None):
    import pyBigWig

    if values is None:
        values = [1.0] * 20
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([(chrom, chrom_size)])
    bw.addEntries(chrom, list(range(len(values))), values=values, span=1, step=1)
    bw.close()


def _make_store(tmp_path) -> AnnotationStore:
    return AnnotationStore(
        annotations_dir=tmp_path / "annotations",
        downloads_dir=tmp_path / "downloads",
    )


def _fail_if_called(*args, **kwargs):
    raise AssertionError("must not trigger a download")


# ──────────────────────────────────────────────────────────────────────
# list_annotations
# ──────────────────────────────────────────────────────────────────────

def test_list_annotations_merges_all_three_sources_without_downloading(tmp_path, monkeypatch):
    import huggingface_hub
    import chorus.utils.http as http_mod

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fail_if_called)
    monkeypatch.setattr(http_mod, "download_with_resume", _fail_if_called)
    monkeypatch.setattr("requests.get", _fail_if_called)

    store = _make_store(tmp_path)
    entries = store.list_annotations()
    by_id = {e.id: e for e in entries}

    for track_id in conservation._TRACK_SOURCES:
        assert by_id[track_id].origin == "conservation"
        assert by_id[track_id].genome_build == "hg38"
        assert by_id[track_id].downloaded is False

    for ann_id in AnnotationManager.ANNOTATION_SOURCES:
        assert by_id[ann_id].origin == "gtf"
        assert by_id[ann_id].downloaded is False


def test_list_annotations_includes_custom_entries(tmp_path):
    store = _make_store(tmp_path)
    store.add_annotation(
        "my_custom_track",
        description="A custom track",
        genome_build="hg38",
        url="https://example.org/my_custom_track.bed",
    )
    by_id = {e.id: e for e in store.list_annotations()}
    assert by_id["my_custom_track"].origin == "custom"
    assert by_id["my_custom_track"].format == "bed"
    assert by_id["my_custom_track"].downloaded is False


# ──────────────────────────────────────────────────────────────────────
# add_annotation validation
# ──────────────────────────────────────────────────────────────────────

def test_add_annotation_requires_exactly_one_source(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(ValueError, match="exactly one source"):
        store.add_annotation("x", description="d", genome_build="hg38")
    with pytest.raises(ValueError, match="exactly one source"):
        store.add_annotation(
            "x", description="d", genome_build="hg38",
            url="https://example.org/a.bed", local_path=str(tmp_path / "a.bed"),
        )


def test_add_annotation_requires_hf_revision_when_hf_repo_given(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(ValueError, match="hf_revision"):
        store.add_annotation(
            "x", description="d", genome_build="hg38",
            hf_repo="someorg/somerepo", hf_filename="a.bw",
        )


@pytest.mark.parametrize("bad_revision", ["main", "master", "HEAD", "Main"])
def test_add_annotation_rejects_moving_branch_revisions(tmp_path, bad_revision):
    store = _make_store(tmp_path)
    with pytest.raises(ValueError, match="hf_revision"):
        store.add_annotation(
            "x", description="d", genome_build="hg38",
            hf_repo="someorg/somerepo", hf_filename="a.bw", hf_revision=bad_revision,
        )


def test_add_annotation_rejects_unknown_genome_build(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(ValueError, match="not an assembly"):
        store.add_annotation(
            "x", description="d", genome_build="GRCh38",
            url="https://example.org/a.bed",
        )


def test_add_annotation_rejects_collision_with_builtin_id(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(ValueError, match="builtin"):
        store.add_annotation(
            "gpn_star", description="d", genome_build="hg38",
            url="https://example.org/a.bw",
        )
    with pytest.raises(ValueError, match="builtin"):
        store.add_annotation(
            "gencode_v48_basic", description="d", genome_build="hg38",
            url="https://example.org/a.gtf.gz",
        )


def test_add_annotation_rejects_duplicate_custom_id_without_overwrite(tmp_path):
    store = _make_store(tmp_path)
    store.add_annotation("dup", description="d", genome_build="hg38", url="https://example.org/a.bed")
    with pytest.raises(ValueError, match="overwrite"):
        store.add_annotation("dup", description="d2", genome_build="hg38", url="https://example.org/b.bed")

    # overwrite=True replaces it.
    store.add_annotation("dup", description="d2", genome_build="hg38", url="https://example.org/b.bed", overwrite=True)
    entry = store.describe_annotation("dup")
    assert entry.description == "d2"


def test_add_annotation_persists_to_yaml(tmp_path):
    store = _make_store(tmp_path)
    store.add_annotation(
        "my_track", description="desc", genome_build="hg19",
        hf_repo="someorg/somerepo", hf_filename="track.bw", hf_revision="v1.0",
    )
    yaml_path = tmp_path / "annotations" / "custom_annotations.yaml"
    assert yaml_path.exists()
    data = yaml.safe_load(yaml_path.read_text())
    assert data["version"] == 1
    entry = data["annotations"]["my_track"]
    assert entry["genome_build"] == "hg19"
    assert entry["hf_repo"] == "someorg/somerepo"
    assert entry["hf_revision"] == "v1.0"
    assert entry["format"] == "bigwig"


# ──────────────────────────────────────────────────────────────────────
# Genome-build verification (bigwig-only)
# ──────────────────────────────────────────────────────────────────────

def test_add_annotation_with_local_bigwig_matching_build_succeeds(tmp_path):
    bw_path = tmp_path / "matching.bw"
    _write_fixture_bigwig(bw_path, chrom_size=ASSEMBLY_CHR1_LENGTH["hg38"])

    store = _make_store(tmp_path)
    entry = store.add_annotation(
        "matching_track", description="d", genome_build="hg38", local_path=str(bw_path),
    )
    assert entry.verified_genome_build == "hg38"
    assert entry.warning is None


def test_add_annotation_with_local_bigwig_mismatched_build_raises(tmp_path):
    bw_path = tmp_path / "mismatched.bw"
    _write_fixture_bigwig(bw_path, chrom_size=ASSEMBLY_CHR1_LENGTH["mm10"])

    store = _make_store(tmp_path)
    with pytest.raises(GenomeAssemblyMismatchError):
        store.add_annotation(
            "mismatched_track", description="d", genome_build="hg38", local_path=str(bw_path),
        )
    # The rejected entry must not have been persisted.
    assert "mismatched_track" not in store._load_custom_yaml().get("annotations", {})


def test_describe_annotation_verifies_downloaded_custom_bigwig(tmp_path):
    store = _make_store(tmp_path)
    # Register with a URL source (not yet downloaded), then drop the file at the
    # deterministic download path by hand to simulate a completed download.
    store.add_annotation(
        "url_track", description="d", genome_build="hg38",
        url="https://example.org/url_track.bw",
    )
    local_path = store._custom_dir("url_track") / "url_track.bw"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _write_fixture_bigwig(local_path, chrom_size=ASSEMBLY_CHR1_LENGTH["hg38"])

    entry = store.describe_annotation("url_track")
    assert entry.downloaded is True
    assert entry.verified_genome_build == "hg38"


def test_describe_annotation_raises_on_mismatched_downloaded_bigwig(tmp_path):
    store = _make_store(tmp_path)
    store.add_annotation(
        "url_track2", description="d", genome_build="hg38",
        url="https://example.org/url_track2.bw",
    )
    local_path = store._custom_dir("url_track2") / "url_track2.bw"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _write_fixture_bigwig(local_path, chrom_size=ASSEMBLY_CHR1_LENGTH["mm10"])

    with pytest.raises(GenomeAssemblyMismatchError):
        store.describe_annotation("url_track2")


def test_describe_annotation_unknown_id_raises(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(ValueError, match="Unknown annotation"):
        store.describe_annotation("does_not_exist")


# ──────────────────────────────────────────────────────────────────────
# download_annotation — passthrough for builtin origins, real dispatch for custom
# ──────────────────────────────────────────────────────────────────────

def test_download_annotation_delegates_to_conservation_for_conservation_ids(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    calls = []
    monkeypatch.setattr(
        conservation, "download_track",
        lambda track, downloads_dir=None: calls.append((track, downloads_dir)) or (tmp_path / "fake.bw"),
    )
    path = store.download_annotation("gpn_star")
    assert calls == [("gpn_star", store.downloads_dir)]
    assert path == tmp_path / "fake.bw"


def test_download_annotation_delegates_to_annotation_manager_for_gtf_ids(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    calls = []
    monkeypatch.setattr(
        AnnotationManager, "download_annotation",
        lambda self, annotation_id, force=False: calls.append(annotation_id) or (tmp_path / "fake.gtf"),
    )
    path = store.download_annotation("gencode_v48_basic")
    assert calls == ["gencode_v48_basic"]
    assert path == tmp_path / "fake.gtf"


def test_download_annotation_custom_url_uses_download_with_resume(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    store.add_annotation("custom_url", description="d", genome_build="hg38", url="https://example.org/track.bed")

    calls = []

    def fake_download(url, dest, label=None):
        calls.append((url, dest, label))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("fake bed content")

    import chorus.utils.http as http_mod
    monkeypatch.setattr(http_mod, "download_with_resume", fake_download)

    path = store.download_annotation("custom_url")
    assert calls == [("https://example.org/track.bed", store._custom_dir("custom_url") / "track.bed", "custom_url")]
    assert path.exists()


def test_download_annotation_custom_hf_pins_revision(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    store.add_annotation(
        "custom_hf", description="d", genome_build="hg38",
        hf_repo="someorg/somerepo", hf_filename="track.bw", hf_revision="v2026-08-19",
    )

    calls = []

    def fake_hf_hub_download(repo_id, filename, repo_type, revision, local_dir):
        calls.append(dict(repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision))
        dest = store._custom_dir("custom_hf") / "track.bw"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("fake bw content")
        return str(dest)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    path = store.download_annotation("custom_hf")
    assert len(calls) == 1
    assert calls[0]["revision"] == "v2026-08-19"
    assert path.exists()


def test_download_annotation_custom_local_is_a_noop(tmp_path):
    local_file = tmp_path / "already_here.bed"
    local_file.write_text("data")

    store = _make_store(tmp_path)
    store.add_annotation("local_one", description="d", genome_build="hg38", local_path=str(local_file))

    path = store.download_annotation("local_one")
    assert path == local_file.resolve()


def test_download_annotation_unknown_id_raises(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(ValueError, match="Unknown annotation"):
        store.download_annotation("does_not_exist")


# ──────────────────────────────────────────────────────────────────────
# remove_custom_annotation
# ──────────────────────────────────────────────────────────────────────

def test_remove_custom_annotation_removes_yaml_entry(tmp_path):
    store = _make_store(tmp_path)
    store.add_annotation("to_remove", description="d", genome_build="hg38", url="https://example.org/a.bed")
    store.remove_custom_annotation("to_remove")
    assert "to_remove" not in store._load_custom_yaml().get("annotations", {})


def test_remove_custom_annotation_rejects_builtin_id(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(ValueError, match="not a custom annotation"):
        store.remove_custom_annotation("gpn_star")


def test_remove_custom_annotation_delete_file_removes_downloaded_file(tmp_path):
    store = _make_store(tmp_path)
    local_file = tmp_path / "to_delete.bed"
    local_file.write_text("data")
    store.add_annotation("with_file", description="d", genome_build="hg38", local_path=str(local_file))

    store.remove_custom_annotation("with_file", delete_file=True)
    assert not local_file.exists()
