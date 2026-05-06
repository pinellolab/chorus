"""chorus cleanup — remove conda envs, downloaded weights, CDFs, and genomes."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

from ..core.globals import CHORUS_DOWNLOADS_DIR, CHORUS_GENOMES_DIR
from ..core.weights_probe import SETUP_MARKER_NAME

_BACKGROUNDS_DIR = Path.home() / ".chorus" / "backgrounds"

_ALL_ORACLES = [
    "enformer", "borzoi", "chrombpnet", "sei", "legnet",
    "alphagenome", "alphagenome_pt",
]


def _dry(msg: str, dry_run: bool) -> None:
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"{prefix}{msg}")


def _remove_path(p: Path, dry_run: bool) -> bool:
    """Remove a file or directory tree. Returns True if something was deleted."""
    if not p.exists():
        return False
    _dry(f"  remove {p}", dry_run)
    if not dry_run:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    return True


def _cleanup_oracle(
    oracle: str, dry_run: bool, manager
) -> Tuple[int, int]:
    """Remove env + weight dir for one oracle. Returns (envs_removed, dirs_removed)."""
    envs = 0
    dirs = 0

    # Conda environment
    if manager.environment_exists(oracle):
        _dry(f"  remove conda env chorus-{oracle}", dry_run)
        if not dry_run:
            manager.remove_environment(oracle)
        envs += 1

    # Downloaded weights + setup marker
    weight_dir = CHORUS_DOWNLOADS_DIR / oracle.lower()
    if _remove_path(weight_dir, dry_run):
        dirs += 1

    # Per-oracle background CDF
    for pattern in [f"{oracle}_pertrack.npz", f"{oracle}_*.npy"]:
        for f in _BACKGROUNDS_DIR.glob(pattern):
            _remove_path(f, dry_run)

    return envs, dirs


def cleanup_resources(args) -> int:
    from ..core.environment import EnvironmentManager
    from ..utils.genome import GenomeManager

    manager = EnvironmentManager()
    dry_run: bool = args.dry_run

    do_oracle: str | None = getattr(args, "oracle", None)
    do_backgrounds: bool = getattr(args, "backgrounds", False)
    do_genomes: bool = getattr(args, "genomes", False)
    do_all: bool = getattr(args, "all", False)

    if do_all:
        do_oracle = "all"
        do_backgrounds = True
        do_genomes = True

    if not any([do_oracle, do_backgrounds, do_genomes]):
        print(
            "Nothing to clean up. Specify at least one of:\n"
            "  --oracle {name|all}   conda env + weights\n"
            "  --backgrounds         background CDFs (~/.chorus/backgrounds/)\n"
            "  --genomes             downloaded reference genomes\n"
            "  --all                 everything above\n"
            "\nAdd --dry-run to preview without deleting."
        )
        return 1

    total_envs = 0
    total_dirs = 0
    total_files = 0

    # ── Oracle envs + weights ──────────────────────────────────────────
    if do_oracle:
        oracles: List[str] = _ALL_ORACLES if do_oracle == "all" else [do_oracle]
        unknown = [o for o in oracles if o not in _ALL_ORACLES]
        if unknown:
            print(
                f"Unknown oracle(s): {', '.join(unknown)}. "
                f"Valid: {', '.join(_ALL_ORACLES)}"
            )
            return 1

        print(f"{'[DRY RUN] ' if dry_run else ''}Cleaning oracle(s): {', '.join(oracles)}")
        for oracle in oracles:
            e, d = _cleanup_oracle(oracle, dry_run, manager)
            total_envs += e
            total_dirs += d

    # ── Background CDFs ────────────────────────────────────────────────
    if do_backgrounds and _BACKGROUNDS_DIR.exists():
        print(f"{'[DRY RUN] ' if dry_run else ''}Cleaning backgrounds: {_BACKGROUNDS_DIR}")
        for f in sorted(_BACKGROUNDS_DIR.glob("*.npz")) + sorted(_BACKGROUNDS_DIR.glob("*.npy")):
            _remove_path(f, dry_run)
            total_files += 1

    # ── Genomes ────────────────────────────────────────────────────────
    if do_genomes:
        print(f"{'[DRY RUN] ' if dry_run else ''}Cleaning genomes: {CHORUS_GENOMES_DIR}")
        gm = GenomeManager()
        for genome_id in gm.list_downloaded_genomes():
            _dry(f"  remove genome {genome_id}", dry_run)
            if not dry_run:
                gm.remove_genome(genome_id)
            total_files += 1

    # ── Summary ────────────────────────────────────────────────────────
    parts = []
    if total_envs:
        parts.append(f"{total_envs} environment(s)")
    if total_dirs:
        parts.append(f"{total_dirs} weight dir(s)")
    if total_files:
        parts.append(f"{total_files} file(s)")

    verb = "Would remove" if dry_run else "Removed"
    summary = f"{verb}: {', '.join(parts)}" if parts else f"{verb}: nothing (already clean)"
    print(summary)
    return 0
