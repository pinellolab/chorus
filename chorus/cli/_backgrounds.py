"""CLI subcommands for managing per-track CDF backgrounds.

Usage:
    chorus backgrounds status [--oracle NAME]
    chorus backgrounds build  --oracle NAME [--track ID] [--only-missing] [--gpu N]
    chorus backgrounds add-tracks --oracle NAME --npz PATH
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_BG_DIR = Path.home() / ".chorus" / "backgrounds"

# Oracles that have per-track NPZ backgrounds
_KNOWN_ORACLES = [
    "enformer", "borzoi", "chrombpnet", "sei", "legnet", "alphagenome",
]


def backgrounds_status(args):
    """Show per-oracle background CDF status."""
    oracles = [args.oracle] if args.oracle else _KNOWN_ORACLES

    for oracle in oracles:
        npz_path = _BG_DIR / f"{oracle}_pertrack.npz"
        if not npz_path.exists():
            print(f"  {oracle:14s}  --  no backgrounds found")
            continue

        data = np.load(str(npz_path), allow_pickle=False)
        ids = [str(t) for t in data["track_ids"]]
        n = len(ids)
        size_mb = npz_path.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(npz_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

        keys_present = [k for k in ("effect_cdfs", "summary_cdfs", "perbin_cdfs") if k in data]

        print(f"  {oracle:14s}  {n:>6d} tracks  {size_mb:6.1f} MB  {mtime}  CDFs: {', '.join(keys_present)}")

        # ChromBPNet: show ATAC/DNASE vs CHIP breakdown
        if oracle == "chrombpnet" and n > 0:
            n_chip = sum(1 for i in ids if i.startswith("CHIP:"))
            n_other = n - n_chip
            print(f"  {'':14s}         ATAC/DNASE: {n_other}  CHIP: {n_chip}")

        data.close()

    return 0


def backgrounds_build(args):
    """Build CDF backgrounds for an oracle (delegates to build script)."""
    import subprocess

    oracle = args.oracle
    script = Path(__file__).resolve().parents[2] / "scripts" / f"build_backgrounds_{oracle}.py"
    if not script.exists():
        logger.error(f"No build script found for oracle '{oracle}' (expected {script})")
        return 1

    # Determine the conda env
    env_name = f"chorus-{oracle}" if oracle != "base" else "chorus"

    # Build the command as a list (avoid shell quoting issues)
    cmd_parts = ["mamba", "run", "-n", env_name, "python", str(script), "--part", "both"]
    if args.only_missing:
        cmd_parts.append("--only-missing")
    if args.gpu is not None:
        cmd_parts.extend(["--gpu", str(args.gpu)])

    logger.info(f"Running: {' '.join(cmd_parts)}")
    result = subprocess.run(cmd_parts, check=False)

    if result.returncode == 0:
        logger.info("Build completed successfully.")
    return result.returncode


def backgrounds_add_tracks(args):
    """Append tracks from a user-provided NPZ into the main oracle NPZ."""
    from ..analysis.normalization import PerTrackNormalizer

    oracle = args.oracle
    src_path = Path(args.npz)
    if not src_path.exists():
        logger.error(f"Source NPZ not found: {src_path}")
        return 1

    src = np.load(str(src_path), allow_pickle=False)
    new_ids = [str(t) for t in src["track_ids"]]
    n_new = len(new_ids)
    logger.info(f"Source NPZ has {n_new} tracks to append.")

    path, n_added = PerTrackNormalizer.append_tracks(
        oracle_name=oracle,
        new_track_ids=new_ids,
        new_effect_cdfs=src.get("effect_cdfs"),
        new_summary_cdfs=src.get("summary_cdfs"),
        new_perbin_cdfs=src.get("perbin_cdfs"),
        new_signed_flags=src.get("signed_flags"),
        new_effect_counts=src.get("effect_counts"),
        new_summary_counts=src.get("summary_counts"),
        new_perbin_counts=src.get("perbin_counts"),
    )
    src.close()

    if n_added == 0:
        print(f"All {n_new} tracks already present in {oracle} backgrounds. Nothing added.")
    else:
        print(f"Added {n_added} new tracks to {path} ({n_new - n_added} duplicates skipped).")

    return 0


def register_backgrounds_subcommand(subparsers):
    """Register the 'backgrounds' subcommand group on the main CLI parser."""
    bg_parser = subparsers.add_parser(
        "backgrounds",
        help="Manage per-track CDF background distributions",
        description=(
            "View, build, and extend the per-track CDF backgrounds used for\n"
            "percentile-normalised variant scoring. Each oracle stores one\n"
            "NPZ file (~/.chorus/backgrounds/{oracle}_pertrack.npz) with\n"
            "effect, summary, and per-bin CDFs for every track."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bg_sub = bg_parser.add_subparsers(dest="bg_command", help="Backgrounds commands")

    # --- status ---
    status_p = bg_sub.add_parser("status", help="Show background CDF status per oracle")
    status_p.add_argument("--oracle", help="Show only this oracle")
    status_p.set_defaults(func=backgrounds_status)

    # --- build ---
    build_p = bg_sub.add_parser(
        "build",
        help="Build CDF backgrounds for an oracle",
        description=(
            "Run the CDF build pipeline for an oracle. By default builds all\n"
            "tracks; use --only-missing to skip tracks already in the NPZ."
        ),
    )
    build_p.add_argument("--oracle", required=True, help="Oracle name (e.g. chrombpnet)")
    build_p.add_argument("--only-missing", action="store_true", help="Skip tracks already in the NPZ")
    build_p.add_argument("--gpu", type=int, help="GPU index to use")
    build_p.set_defaults(func=backgrounds_build)

    # --- add-tracks ---
    add_p = bg_sub.add_parser(
        "add-tracks",
        help="Append tracks from a user-built NPZ",
        description=(
            "Incrementally append new tracks from a source NPZ file into\n"
            "the main per-oracle NPZ. Duplicate track IDs are skipped."
        ),
    )
    add_p.add_argument("--oracle", required=True, help="Oracle name (e.g. chrombpnet)")
    add_p.add_argument("--npz", required=True, help="Path to source NPZ with new tracks")
    add_p.set_defaults(func=backgrounds_add_tracks)

    return bg_parser
