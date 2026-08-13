"""``chorus config data-dir`` — show, set, or migrate where bulk data is stored.

Everything large used to land in ``$HOME``: 7.8 GB of per-track backgrounds plus,
because nothing set ``HF_HOME``, another 12 GB of model weights under
``~/.cache/huggingface``. That is the wrong filesystem on a shared machine or anywhere
with a home quota. The default is now the installation directory, and this command is
how a user points it elsewhere and moves what is already downloaded.

Three ways to choose, in precedence order — the same order
``chorus.core.globals.resolve_data_dir`` applies:

1. ``CHORUS_DATA_DIR=/path`` in the environment (per-shell, per-job, highest priority)
2. ``chorus setup --data-dir /path`` or ``chorus config data-dir --set /path``,
   recorded in ``<install>/chorus_data_dir.txt``
3. the installation directory (default), or ``~/.chorus`` when the install tree is
   not writable
"""
from __future__ import annotations

import logging
import shutil
import stat as stat_module
from pathlib import Path

logger = logging.getLogger(__name__)


def _fmt_size(path: Path) -> str:
    """Size of a tree, counting each underlying file once — i.e. ``du -L`` semantics.

    The dedup is the whole point. A HuggingFace hub cache stores every file once under
    ``<repo>/blobs/<sha>`` and exposes it as a relative symlink from
    ``<repo>/snapshots/<rev>/<name>``, so **both endpoints live inside the tree being walked**.
    Counting the real file and then again through the dereferenced symlink reported the cache here
    as **41.5 GB** against a true 20.7 GB — 6,927 symlinks, all resolving internally, double-counted
    to the byte. (An earlier write-up blamed hardlinks; measured, this tree has none —
    ``find -links +1`` returns nothing. Deduping on ``(st_dev, st_ino)`` covers both anyway, and
    `huggingface_hub` does hardlink in some configurations.)

    Symlinks are still **followed**, deliberately. Simply skipping them — the obvious one-line fix —
    drops ``genomes`` from 3.3 GB to 170 bytes, because its two entries are symlinks to FASTAs
    stored outside the tree, so their bytes are only reachable through the link.
    """
    if not path.exists():
        return "absent"
    total = 0
    seen: set[tuple[int, int]] = set()
    try:
        for p in path.rglob("*"):
            try:
                st = p.stat()                          # follows symlinks
            except OSError:
                continue                               # broken link, or vanished mid-walk
            if not stat_module.S_ISREG(st.st_mode):
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += st.st_size
    except Exception:
        return "?"
    if total >= 1e9:
        return f"{total / 1e9:.1f} GB"
    if total >= 1e6:
        return f"{total / 1e6:.0f} MB"
    return f"{total / 1e3:.0f} KB"


def _write_marker(target: Path) -> None:
    from chorus.core.globals import CHORUS_ROOT, DATA_DIR_MARKER

    target.mkdir(parents=True, exist_ok=True)
    probe = target / ".chorus-write-probe"
    probe.touch()
    probe.unlink()
    marker = CHORUS_ROOT / DATA_DIR_MARKER
    marker.write_text(f"{target.resolve()}\n")
    print(f"Recorded data directory in {marker}")


def data_dir_command(args) -> int:
    from chorus.core.globals import (
        CHORUS_BACKGROUNDS_DIR,
        DATA_DIR_MARKER,
        CHORUS_ROOT,
        describe_layout,
    )

    if getattr(args, "set", None):
        target = Path(args.set).expanduser().resolve()
        try:
            _write_marker(target)
        except Exception as exc:
            logger.error("cannot use %s: %s", target, exc)
            return 1
        print("Re-run any chorus command for the new location to take effect "
              "(paths resolve at import time).")
        if not getattr(args, "migrate", False):
            print(f"Existing data was NOT moved. Re-run with --migrate to move it, "
                  f"or leave it and chorus will re-download on demand.")

    if getattr(args, "migrate", False):
        layout = describe_layout()
        src = CHORUS_BACKGROUNDS_DIR
        dst = Path(layout["data_dir"]) / "backgrounds"
        if getattr(args, "set", None):
            dst = Path(args.set).expanduser().resolve() / "backgrounds"
        if src.resolve() == dst.resolve():
            print(f"Nothing to migrate: backgrounds already at {src}")
        elif not src.exists():
            print(f"Nothing to migrate: {src} does not exist")
        else:
            files = sorted(src.glob("*_pertrack.npz"))
            print(f"Moving {len(files)} background file(s), {_fmt_size(src)}, "
                  f"from {src} to {dst}")
            dst.mkdir(parents=True, exist_ok=True)
            for f in files:
                target = dst / f.name
                if target.exists():
                    print(f"  skip {f.name} (already present at the destination)")
                    continue
                # copy-then-remove rather than move: src may be a symlink into
                # another filesystem, where rename() fails with EXDEV, and a failed
                # move must never leave the only copy half-written.
                shutil.copy2(f, target)
                if target.stat().st_size != f.stat().st_size:
                    target.unlink(missing_ok=True)
                    logger.error("size mismatch copying %s — aborting", f.name)
                    return 1
                f.unlink()
                print(f"  moved {f.name}")
            print("Migration complete.")

    layout = describe_layout()
    print("\nResolved chorus data layout")
    print(f"  data_dir      {layout['data_dir']}")
    print(f"  chosen via    {layout['source']}")
    print()
    for key in ("backgrounds", "downloads", "genomes", "annotations", "hf_cache"):
        p = Path(layout[key])
        print(f"  {key:13} {layout[key]}  [{_fmt_size(p)}]")
    print()
    print(f"  environments  {layout['environments']}  (always with the install)")
    print(f"  credentials   {layout['config_secrets']}  (always in $HOME, never "
          f"in a shared data dir)")
    print()
    print("To change it:")
    print("  export CHORUS_DATA_DIR=/path/to/data        # per-shell, highest priority")
    print("  chorus config data-dir --set /path/to/data  # persist for this install")
    print("  chorus setup --data-dir /path/to/data       # choose at install time")
    print(f"  (persisted in {CHORUS_ROOT / DATA_DIR_MARKER})")
    return 0


def register_config_subcommand(subparsers):
    config_parser = subparsers.add_parser(
        "config",
        help="Show or change where chorus stores downloaded data",
    )
    config_sub = config_parser.add_subparsers(dest="config_command")
    dd = config_sub.add_parser(
        "data-dir",
        help="Show, set, or migrate the data directory",
        description=(
            "Downloaded data (backgrounds, weights, genomes, annotations, HF cache) "
            "defaults to the chorus installation directory. Point it elsewhere for "
            "shared installs or when $HOME is too small.\n\n"
            "Credentials always stay in $HOME and are never moved."
        ),
    )
    dd.add_argument("--set", metavar="PATH",
                    help="Persist this data directory for the installation")
    dd.add_argument("--migrate", action="store_true",
                    help="Move existing background files to the resolved location")
    dd.set_defaults(func=data_dir_command)
    return config_parser
