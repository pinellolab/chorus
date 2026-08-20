"""Tests for the `chorus annotation` CLI subcommand.

Network-free: exercises the CLI handler functions directly against a
monkeypatched `chorus.utils.annotation_store.AnnotationStore`, plus a real
argparse pass to confirm the subcommand wires in correctly. Mirrors the
two-layer shape of tests/test_cli_conservation.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from chorus.cli import _annotation
from chorus.utils.annotation_store import AnnotationEntry, AnnotationStore


def _entry(**overrides):
    defaults = dict(
        id="gpn_star",
        origin="conservation",
        description="GPN-Star entropy conservation score (hg38)",
        genome_build="hg38",
        format="bigwig",
        downloaded=True,
        path=Path("/fake/downloads/gpn_star/entropy.bw"),
        size_bytes=9_897_991_293,
        size_note="~9.9 GB",
        source={"kind": "hf"},
    )
    defaults.update(overrides)
    return AnnotationEntry(**defaults)


def test_annotation_list_prints_downloaded_and_missing(monkeypatch, capsys):
    entries = [
        _entry(),
        _entry(id="gencode_v48_basic", origin="gtf", genome_build="hg38",
               downloaded=False, path=None, size_bytes=None, size_note=None),
    ]
    monkeypatch.setattr(AnnotationStore, "list_annotations", lambda self: entries)

    rc = _annotation.annotation_list(argparse.Namespace())

    assert rc == 0
    out = capsys.readouterr().out
    assert "gpn_star" in out
    assert "GB" in out
    assert "gencode_v48_basic" in out
    assert "not downloaded" in out


def test_annotation_describe_prints_fields(monkeypatch, capsys):
    entry = _entry(verified_genome_build="hg38")
    monkeypatch.setattr(AnnotationStore, "describe_annotation", lambda self, annotation_id: entry)

    rc = _annotation.annotation_describe(argparse.Namespace(annotation_id="gpn_star"))

    assert rc == 0
    out = capsys.readouterr().out
    assert "genome_build: hg38" in out
    assert "verified_genome_build: hg38" in out


def test_annotation_describe_reports_failure(monkeypatch):
    def raise_error(self, annotation_id):
        raise ValueError("Unknown annotation: 'x'")
    monkeypatch.setattr(AnnotationStore, "describe_annotation", raise_error)

    rc = _annotation.annotation_describe(argparse.Namespace(annotation_id="x"))
    assert rc == 1


def test_annotation_download_success(monkeypatch):
    monkeypatch.setattr(AnnotationStore, "download_annotation", lambda self, annotation_id: Path("/fake/path"))
    rc = _annotation.annotation_download(argparse.Namespace(annotation_id="gpn_star"))
    assert rc == 0


def test_annotation_download_reports_failure(monkeypatch):
    def raise_error(self, annotation_id):
        raise RuntimeError("network down")
    monkeypatch.setattr(AnnotationStore, "download_annotation", raise_error)

    rc = _annotation.annotation_download(argparse.Namespace(annotation_id="gpn_star"))
    assert rc == 1


def test_annotation_add_success(monkeypatch):
    monkeypatch.setattr(AnnotationStore, "add_annotation", lambda self, annotation_id, **kw: _entry(id=annotation_id))

    rc = _annotation.annotation_add(argparse.Namespace(
        annotation_id="my_track", description="d", genome_build="hg38", format=None,
        hf_repo=None, hf_filename=None, hf_revision=None,
        url="https://example.org/a.bed", local_path=None, local_filename=None,
        overwrite=False,
    ))
    assert rc == 0


def test_annotation_add_reports_failure(monkeypatch):
    def raise_error(self, annotation_id, **kw):
        raise ValueError("bad input")
    monkeypatch.setattr(AnnotationStore, "add_annotation", raise_error)

    rc = _annotation.annotation_add(argparse.Namespace(
        annotation_id="my_track", description="d", genome_build="hg38", format=None,
        hf_repo=None, hf_filename=None, hf_revision=None,
        url=None, local_path=None, local_filename=None, overwrite=False,
    ))
    assert rc == 1


def test_annotation_remove_success(monkeypatch):
    calls = []
    monkeypatch.setattr(
        AnnotationStore, "remove_custom_annotation",
        lambda self, annotation_id, delete_file=False: calls.append((annotation_id, delete_file)),
    )
    rc = _annotation.annotation_remove(argparse.Namespace(annotation_id="my_track", delete_file=True))
    assert rc == 0
    assert calls == [("my_track", True)]


def test_annotation_remove_reports_failure(monkeypatch):
    def raise_error(self, annotation_id, delete_file=False):
        raise ValueError("not a custom annotation")
    monkeypatch.setattr(AnnotationStore, "remove_custom_annotation", raise_error)

    rc = _annotation.annotation_remove(argparse.Namespace(annotation_id="gpn_star", delete_file=False))
    assert rc == 1


def test_register_annotation_subcommand_parses_all_subcommands():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    _annotation.register_annotation_subcommand(subparsers)

    list_args = parser.parse_args(["annotation", "list"])
    assert list_args.func is _annotation.annotation_list

    describe_args = parser.parse_args(["annotation", "describe", "gpn_star"])
    assert describe_args.func is _annotation.annotation_describe
    assert describe_args.annotation_id == "gpn_star"

    download_args = parser.parse_args(["annotation", "download", "gpn_star"])
    assert download_args.func is _annotation.annotation_download

    add_args = parser.parse_args([
        "annotation", "add", "my_track",
        "--description", "d", "--genome-build", "hg38",
        "--url", "https://example.org/a.bed",
    ])
    assert add_args.func is _annotation.annotation_add
    assert add_args.annotation_id == "my_track"
    assert add_args.genome_build == "hg38"
    assert add_args.url == "https://example.org/a.bed"

    remove_args = parser.parse_args(["annotation", "remove", "my_track", "--delete-file"])
    assert remove_args.func is _annotation.annotation_remove
    assert remove_args.delete_file is True
