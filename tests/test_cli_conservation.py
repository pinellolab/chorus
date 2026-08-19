"""Tests for the `chorus conservation` CLI subcommand.

Network-free: exercises the CLI handler functions directly against a
monkeypatched `chorus.analysis.conservation` (list_tracks/download_track),
plus a real argparse pass to confirm the subcommand wires in correctly.
"""
from __future__ import annotations

import argparse

import pytest

from chorus.cli import _conservation


def _fake_status(downloaded_tracks=()):
    return {
        "gpn_star": {
            "downloaded": "gpn_star" in downloaded_tracks,
            "size_bytes": 123456789 if "gpn_star" in downloaded_tracks else None,
            "size_note": "~9.9 GB",
            "source": "hf",
            "path": _FakePath("/fake/downloads/gpn_star/entropy.bw"),
        },
        "phylop20way": {
            "downloaded": "phylop20way" in downloaded_tracks,
            "size_bytes": 234567890 if "phylop20way" in downloaded_tracks else None,
            "size_note": "~7.3 GB",
            "source": "url",
            "path": _FakePath("/fake/downloads/phylop20way/hg38.phyloP20way.bw"),
        },
        "phastcons7way": {
            "downloaded": "phastcons7way" in downloaded_tracks,
            "size_bytes": None,
            "size_note": "~7.2 GB",
            "source": "url",
            "path": _FakePath("/fake/downloads/phastcons7way/hg38.phastCons7way.bw"),
        },
    }


class _FakePath:
    """Minimal stand-in for Path with just what conservation_status touches."""

    def __init__(self, s):
        self._s = s

    def __str__(self):
        return self._s

    def stat(self):
        class _Stat:
            st_mtime = 0
        return _Stat()


def test_conservation_status_prints_downloaded_and_missing(monkeypatch, capsys):
    from chorus.analysis import conservation

    monkeypatch.setattr(conservation, "list_tracks", lambda: _fake_status(["gpn_star"]))

    rc = _conservation.conservation_status(argparse.Namespace())

    assert rc == 0
    out = capsys.readouterr().out
    assert "gpn_star" in out
    assert "GB" in out  # downloaded track shows a size
    assert "phylop20way" in out
    assert "not downloaded" in out  # missing tracks say so
    assert "~7.3 GB" in out  # missing tracks show the size estimate


def test_conservation_download_requires_track_or_all():
    rc = _conservation.conservation_download(argparse.Namespace(track=None, all=False))
    assert rc == 1


def test_conservation_download_rejects_unknown_track(monkeypatch):
    from chorus.analysis import conservation
    monkeypatch.setattr(conservation, "list_tracks", lambda: _fake_status())

    rc = _conservation.conservation_download(argparse.Namespace(track="not_real", all=False))
    assert rc == 1


def test_conservation_download_skips_already_downloaded(monkeypatch):
    from chorus.analysis import conservation
    monkeypatch.setattr(conservation, "list_tracks", lambda: _fake_status(["gpn_star"]))

    def fail_if_called(*a, **kw):
        raise AssertionError("download_track must not be called for an already-downloaded track")
    monkeypatch.setattr(conservation, "download_track", fail_if_called)

    rc = _conservation.conservation_download(argparse.Namespace(track="gpn_star", all=False))
    assert rc == 0


def test_conservation_download_all_calls_each_missing_track(monkeypatch):
    from chorus.analysis import conservation
    monkeypatch.setattr(conservation, "list_tracks", lambda: _fake_status(["gpn_star"]))

    downloaded = []
    monkeypatch.setattr(conservation, "download_track", lambda t: downloaded.append(t))

    rc = _conservation.conservation_download(argparse.Namespace(track=None, all=True))

    assert rc == 0
    # gpn_star already downloaded -> skipped; the other two get downloaded.
    assert set(downloaded) == {"phylop20way", "phastcons7way"}


def test_conservation_download_llr_alias_expands_to_all_four_tracks(monkeypatch):
    # gpn_star_llr is a bundle alias (chorus/cli/_conservation.py::_TRACK_ALIASES)
    # for the four calibrated-LLR bigwigs that jointly feed the stacked
    # sequence-logo track — a user shouldn't need to invoke download 4 times.
    from chorus.analysis import conservation

    status = _fake_status()
    for base in "ACGT":
        name = f"gpn_star_llr_{base.lower()}"
        status[name] = {
            "downloaded": False,
            "size_bytes": None,
            "size_note": "~11 GB",
            "source": "hf",
            "path": _FakePath(f"/fake/downloads/gpn_star_llr/llr_{base}.bw"),
        }
    monkeypatch.setattr(conservation, "list_tracks", lambda: status)

    downloaded = []
    monkeypatch.setattr(conservation, "download_track", lambda t: downloaded.append(t))

    rc = _conservation.conservation_download(argparse.Namespace(track="gpn_star_llr", all=False))

    assert rc == 0
    assert set(downloaded) == {"gpn_star_llr_a", "gpn_star_llr_c", "gpn_star_llr_g", "gpn_star_llr_t"}


def test_conservation_download_reports_failure(monkeypatch):
    from chorus.analysis import conservation
    monkeypatch.setattr(conservation, "list_tracks", lambda: _fake_status())

    def raise_error(track):
        raise RuntimeError("network down")
    monkeypatch.setattr(conservation, "download_track", raise_error)

    rc = _conservation.conservation_download(argparse.Namespace(track="gpn_star", all=False))
    assert rc == 1


def test_register_conservation_subcommand_parses_status_and_download():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    _conservation.register_conservation_subcommand(subparsers)

    status_args = parser.parse_args(["conservation", "status"])
    assert status_args.func is _conservation.conservation_status

    download_args = parser.parse_args(["conservation", "download", "--track", "gpn_star"])
    assert download_args.func is _conservation.conservation_download
    assert download_args.track == "gpn_star"

    download_all_args = parser.parse_args(["conservation", "download", "--all"])
    assert download_all_args.all is True
