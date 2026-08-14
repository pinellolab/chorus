"""`chorus config data-dir` must report each file's bytes once.

It reported the HuggingFace cache as **41.5 GB** where `du -shL` gave 20.7 GB. The hub stores every
file once under `<repo>/blobs/<sha>` and exposes it as a relative symlink from
`<repo>/snapshots/<rev>/<name>`, so *both endpoints are inside the tree being walked* — the old code
added the real file and then added it again through the dereferenced symlink.

Two things this pins that a casual fix would get wrong:

* **Symlinks must still be followed.** Skipping them is the tempting one-liner and it drops
  `genomes` from 3.3 GB to 170 bytes, because those entries are symlinks to FASTAs stored *outside*
  the tree — their bytes are reachable only through the link.
* **Dedup is by `(st_dev, st_ino)`, not by name**, so it covers hardlinks too. The real cache on the
  box where this was found had none (`find -links +1` → nothing, which is why the first write-up
  blaming hardlinks was wrong), but `huggingface_hub` hardlinks in some configurations and the
  inode key handles both without caring which.

Directory sizes are inherently machine-specific, so the assertions here are built on temp trees with
known link structure rather than on whatever this host happens to hold.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from chorus.cli._datadir import _fmt_size


def _bytes_of(path: Path) -> float:
    """Parse `_fmt_size`'s human string back to bytes (it uses decimal units)."""
    text = _fmt_size(path)
    value, unit = text.split()
    return float(value) * {"KB": 1e3, "MB": 1e6, "GB": 1e9}[unit]


def test_a_symlink_inside_the_tree_is_not_counted_twice(tmp_path):
    """The HuggingFace blobs/snapshots layout, in miniature."""
    blobs, snaps = tmp_path / "blobs", tmp_path / "snapshots"
    blobs.mkdir(), snaps.mkdir()
    payload = blobs / "deadbeef"
    payload.write_bytes(b"x" * 4_000_000)
    (snaps / "model.safetensors").symlink_to(payload)

    assert _bytes_of(tmp_path) == pytest.approx(4e6, rel=0.01), (
        "a blob reachable both directly and through an internal symlink was counted twice — "
        "this is exactly the 41.5-vs-20.7 GB defect"
    )


def test_a_symlink_pointing_outside_the_tree_is_still_counted(tmp_path):
    """The `genomes` case. Skipping symlinks would report ~0 here."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    real = outside / "hg38.fa"
    real.write_bytes(b"A" * 3_000_000)

    tree = tmp_path / "genomes"
    tree.mkdir()
    (tree / "hg38.fa").symlink_to(real)

    assert _bytes_of(tree) == pytest.approx(3e6, rel=0.01), (
        "a symlink to a file stored outside the tree contributed nothing — `genomes` would read "
        "as 170 bytes instead of 3.3 GB"
    )


def test_hardlinks_are_counted_once(tmp_path):
    """Same inode by another name. Covered by the same `(st_dev, st_ino)` key."""
    a = tmp_path / "a.bin"
    a.write_bytes(b"y" * 2_000_000)
    try:
        os.link(a, tmp_path / "b.bin")
    except OSError:
        pytest.skip("filesystem does not support hardlinks")

    assert _bytes_of(tmp_path) == pytest.approx(2e6, rel=0.01)


def test_broken_symlinks_do_not_abort_the_walk(tmp_path):
    """A dangling link must be skipped, not turn the whole size into '?'."""
    (tmp_path / "real.bin").write_bytes(b"z" * 1_000_000)
    (tmp_path / "dangling").symlink_to(tmp_path / "gone")

    assert _bytes_of(tmp_path) == pytest.approx(1e6, rel=0.01)


def test_directories_and_absent_paths(tmp_path):
    assert _fmt_size(tmp_path / "nope") == "absent"
    (tmp_path / "sub").mkdir()
    assert _fmt_size(tmp_path) == "0 KB", "an empty tree should not count directory inodes"


@pytest.mark.integration
def test_it_agrees_with_du_on_the_real_data_dir():
    """On a real install, match `du -sL --apparent-size`, which dedups by inode the same way."""
    from chorus.core.globals import describe_layout

    layout = describe_layout()
    checked = 0
    for key in ("hf_cache", "genomes", "downloads", "backgrounds", "annotations"):
        path = Path(layout[key])
        if not path.is_dir() or not any(path.iterdir()):
            continue
        out = subprocess.run(["du", "-sL", "--apparent-size", "-B1", str(path)],
                             capture_output=True, text=True)
        if out.returncode != 0:
            continue
        du_bytes = int(out.stdout.split()[0])
        assert _bytes_of(path) == pytest.approx(du_bytes, rel=0.02), (
            f"{key}: chorus says {_fmt_size(path)}, du -sL --apparent-size says "
            f"{du_bytes / 1e9:.1f} GB"
        )
        checked += 1
    if not checked:
        pytest.skip("no populated data directories on this host")
