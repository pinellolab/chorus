"""chorus cleanup — remove conda envs, downloaded weights, CDFs, genomes, and the HF cache."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

from ..core.globals import CHORUS_DOWNLOADS_DIR, CHORUS_GENOMES_DIR
from ..core.weights_probe import SETUP_MARKER_NAME
from chorus.core.globals import CHORUS_BACKGROUNDS_DIR

_BACKGROUNDS_DIR = CHORUS_BACKGROUNDS_DIR

# Keep in sync with chorus.oracles.ORACLES (hardcoded here so cleanup stays
# lightweight and does not import the oracle classes / heavy deps).
_ALL_ORACLES = [
    "enformer", "borzoi", "chrombpnet", "cherimoya", "sei", "legnet",
    "epinformerseq", "alphagenome", "alphagenome_pt",
]


def _dry(msg: str, dry_run: bool) -> None:
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"{prefix}{msg}")


def _hf_cache_targets() -> Tuple[List[Path], Path | None]:
    """The HuggingFace directories chorus owns, and the parent it must never delete.

    Returns ``(targets, credential_parent)``.

    Resolution goes through ``huggingface_hub.constants.HF_HUB_CACHE`` rather than
    ``describe_layout()["hf_cache"]``, and the difference is not cosmetic:

    * with ``HF_HOME`` set, ``describe_layout()`` returns the HF **home**, which contains the
      ``token`` and ``stored_tokens`` files written by ``huggingface-cli login``. Deleting that
      would destroy the user's credential and any non-chorus model cache.
    * with ``HF_HUB_CACHE`` set and ``HF_HOME`` unset, ``describe_layout()`` reports
      ``<data-dir>/huggingface`` while the real cache is elsewhere — so a cleanup keyed on it would
      delete the wrong directory and leave the bytes behind.

    Only ``hub/`` is removed unconditionally. Its sibling ``xet/`` is chorus-owned too, but is only
    taken when the parent sits inside the chorus data dir, so a shared ``HF_HOME`` keeps everything
    except the hub cache. The parent itself is never a target.
    """
    from ..core.globals import CHORUS_DATA_DIR

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        hub = Path(HF_HUB_CACHE)
    except Exception:
        hub = Path.home() / ".cache" / "huggingface" / "hub"

    parent = hub.parent
    targets = [hub]
    inside_chorus = False
    try:
        inside_chorus = parent.resolve().is_relative_to(Path(CHORUS_DATA_DIR).resolve())
    except (OSError, ValueError):
        pass
    if inside_chorus:
        xet = parent / "xet"
        if xet.exists():
            targets.append(xet)
    return targets, parent


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
    do_hf_cache: bool = getattr(args, "hf_cache", False)
    do_all: bool = getattr(args, "all", False)

    if do_all:
        do_oracle = "all"
        do_backgrounds = True
        do_genomes = True
        # Deliberately NOT do_hf_cache. `--all` means "everything chorus put in its data dir",
        # and the HF cache can legitimately live outside it: with HF_HOME pointing at a shared
        # cache, deleting it would take other projects' weights and the user's stored token with
        # it. Keeping it also makes an upgrade cheap, which is what README's Upgrading section
        # recommends. The notice below makes the omission visible rather than silent.

    if not any([do_oracle, do_backgrounds, do_genomes, do_hf_cache]):
        print(
            "Nothing to clean up. Specify at least one of:\n"
            "  --oracle {name|all}   conda env + weights\n"
            f"  --backgrounds         background CDFs ({_BACKGROUNDS_DIR}/)\n"
            "  --genomes             downloaded reference genomes\n"
            "  --hf-cache            the HuggingFace hub cache (most oracle weights live here)\n"
            "  --all                 everything above except --hf-cache\n"
            "\nAdd --dry-run to preview without deleting."
        )
        return 1

    total_envs = 0
    total_dirs = 0
    total_files = 0
    total_caches = 0

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

    # ── HuggingFace cache ──────────────────────────────────────────────
    if do_hf_cache:
        targets, parent = _hf_cache_targets()
        print(f"{'[DRY RUN] ' if dry_run else ''}Cleaning HuggingFace cache")
        for t in targets:
            if _remove_path(t, dry_run):      # prints the "remove <path>" line itself
                total_caches += 1
        if parent is not None:
            _dry(f"  keeping {parent} itself — it can hold your `huggingface-cli login` token",
                 dry_run)
    elif do_all:
        targets, _ = _hf_cache_targets()
        hub = targets[0]
        if hub.exists():
            # _dry(), not print(): under --dry-run every other line carries the [DRY RUN] prefix,
            # and one unprefixed line in the transcript reads as something that already happened.
            _dry(f"Note: the HuggingFace cache at {hub} was NOT removed — most oracle weights "
                 f"live there, and keeping it makes a re-install fast. Pass --hf-cache to remove "
                 f"it too.", dry_run)

    # ── Summary ────────────────────────────────────────────────────────
    parts = []
    if total_envs:
        parts.append(f"{total_envs} environment(s)")
    if total_dirs:
        parts.append(f"{total_dirs} weight dir(s)")
    if total_files:
        parts.append(f"{total_files} file(s)")
    if total_caches:
        parts.append(f"{total_caches} HF cache dir(s)")

    verb = "Would remove" if dry_run else "Removed"
    summary = f"{verb}: {', '.join(parts)}" if parts else f"{verb}: nothing (already clean)"
    print(summary)
    return 0
