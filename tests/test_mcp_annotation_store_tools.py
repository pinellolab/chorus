"""Tests for the AnnotationStore-backed MCP tools: list_annotations,
describe_annotation, download_annotation.

Distinct from tests/test_mcp_annotation_tools.py, which covers list_genomes /
get_genes_in_region / get_gene_tss (the pre-existing GTF/genome tools). These
three are new, backed by chorus.utils.annotation_store.AnnotationStore.

Uses a tmp_path-scoped AnnotationStore (via a monkeypatched get_annotation_store)
so nothing here touches the real, already-populated data directory or the network.
"""
from __future__ import annotations

import pytest

import chorus.mcp.server as server
import chorus.utils.annotation_store as annotation_store_mod
from chorus.utils.annotation_store import AnnotationStore


def _call(name: str, /, **kw):
    fn = getattr(server, name)
    for attr in ("fn", "__wrapped__"):
        fn = getattr(fn, attr, fn)
    return fn(**kw)


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    store = AnnotationStore(
        annotations_dir=tmp_path / "annotations",
        downloads_dir=tmp_path / "downloads",
    )
    monkeypatch.setattr(annotation_store_mod, "get_annotation_store", lambda: store)
    return store


def test_list_annotations_reports_capping_fields(isolated_store):
    out = _call("list_annotations")
    assert set(["num_annotations", "showing", "truncated", "annotations"]) <= out.keys()
    assert out["truncated"] is False
    assert out["num_annotations"] == len(out["annotations"])
    ids = [a["id"] for a in out["annotations"]]
    assert "gpn_star" in ids
    assert "gencode_v48_basic" in ids


def test_describe_annotation_known_id(isolated_store):
    out = _call("describe_annotation", annotation_id="gpn_star")
    assert out["id"] == "gpn_star"
    assert out["genome_build"] == "hg38"
    assert out["downloaded"] is False  # isolated tmp dir, nothing downloaded


def test_describe_annotation_unknown_id_via_safe_tool_returns_error_dict(isolated_store):
    # Call through the actual decorated tool (not unwrapped) so _safe_tool's
    # error-formatting is exercised, matching what a real MCP client sees.
    out = server.describe_annotation(annotation_id="does_not_exist")
    assert out["error_type"] == "ValueError"
    assert "does_not_exist" in out["error"]
    assert out["tool"] == "describe_annotation"


def test_download_annotation_delegates_to_the_store(isolated_store, monkeypatch, tmp_path):
    fake_path = tmp_path / "fake_downloaded.bw"
    calls = []
    monkeypatch.setattr(
        AnnotationStore, "download_annotation",
        lambda self, annotation_id: calls.append(annotation_id) or fake_path,
    )

    out = _call("download_annotation", annotation_id="gpn_star")
    assert calls == ["gpn_star"]
    assert out == {"annotation_id": "gpn_star", "path": str(fake_path)}
