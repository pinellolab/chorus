"""Where chorus puts things on disk, and how a user redirects that.

THE PROBLEM THIS SOLVES
-----------------------
Everything large used to land in ``$HOME``: 7.8 GB of per-track backgrounds under
``~/.chorus/backgrounds/`` and, because nothing set ``HF_HOME``, a further 12 GB of
model weights under ``~/.cache/huggingface/``. On a shared machine, or any box with a
small home quota, that fills the wrong filesystem — and on this one it did.

So the default is now the **chorus installation directory**, with a single switch to
point it anywhere else.

RESOLUTION ORDER
----------------
1. ``CHORUS_DATA_DIR`` environment variable — explicit, wins over everything.
2. ``data_dir`` recorded in ``<install>/chorus_data_dir.txt`` by
   ``chorus setup --data-dir PATH`` — the install-time choice.
3. The installation directory itself, if writable. **This is the new default.**
4. ``~/.chorus`` — only when the install directory is *not* writable, which is the
   normal case for a ``pip install`` into a system ``site-packages``. Writing into a
   read-only or root-owned tree would fail at the first download, and quietly
   choosing a location that works beats crashing.

WHAT DOES *NOT* MOVE, AND WHY
-----------------------------
Secrets stay in ``$HOME``: the LDlink token in ``~/.chorus/config.toml`` and the
HuggingFace token wherever the HF libraries keep it. The point of the data directory
is that it can be **shared between users** — a group-readable install tree is exactly
the wrong place for a personal API token. So the split is deliberate: bulk data
follows ``CHORUS_DATA_DIR``, credentials follow the user.

Conda environments also stay with the installation. A shared *data* directory must
not imply shared conda prefixes.

MIGRATION
---------
An existing install already has data in ``~/.chorus``. If the legacy directory holds
backgrounds and the resolved one does not, the legacy path is used and a one-time
message says so — otherwise upgrading would silently re-download 7.8 GB and orphan
the old copy. ``chorus config data-dir --migrate`` moves it deliberately.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

CHORUS_ROOT = Path(__file__).parent.parent.parent

#: Name of the file that records an install-time ``--data-dir`` choice.
DATA_DIR_MARKER = "chorus_data_dir.txt"

#: The pre-2026-08 location. Still honoured when it already holds data.
LEGACY_DATA_DIR = Path.home() / ".chorus"

_ENV_VAR = "CHORUS_DATA_DIR"


def _is_writable(path: Path) -> bool:
    """Can we actually create files under *path*?

    Checked by trying, not by reading permission bits: a directory can be mode 777
    and still be unwritable (read-only mount, full filesystem, SELinux denial), and
    ``os.access`` reports the wrong answer for all three.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".chorus-write-probe"
        probe.touch()
        probe.unlink()
        return True
    except Exception:
        return False


def _read_marker() -> Path | None:
    marker = CHORUS_ROOT / DATA_DIR_MARKER
    try:
        if marker.is_file():
            raw = marker.read_text().strip()
            if raw:
                return Path(raw).expanduser()
    except Exception as exc:                                    # pragma: no cover
        logger.warning("could not read %s: %s", marker, exc)
    return None


def _legacy_has_data() -> bool:
    """True if the old ``~/.chorus`` holds backgrounds worth not abandoning."""
    bg = LEGACY_DATA_DIR / "backgrounds"
    try:
        return bg.is_dir() and any(bg.glob("*_pertrack.npz"))
    except Exception:                                           # pragma: no cover
        return False


def resolve_data_dir(*, explicit: str | os.PathLike | None = None) -> Path:
    """Where bulk data lives. See the module docstring for the order."""
    if explicit:
        return Path(explicit).expanduser().resolve()

    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()

    marker = _read_marker()
    if marker:
        return marker.resolve()

    return (CHORUS_ROOT if _is_writable(CHORUS_ROOT) else LEGACY_DATA_DIR).resolve()


def resolve_backgrounds_dir(data_dir: Path) -> Path:
    """Where the per-track NPZs live, with legacy compat applied PER KIND.

    Backgrounds get their own resolver because they are the only kind that used to
    live under ``~/.chorus``. Annotations, downloads and genomes were always in the
    installation tree, so a whole-directory legacy fallback would drag those three
    *out* of the install dir on any machine that had ever downloaded a background —
    relocating data that was never in the wrong place. That is a regression an
    earlier version of this module actually had.
    """
    resolved = data_dir / "backgrounds"
    if os.environ.get(_ENV_VAR) or _read_marker():
        return resolved                      # an explicit choice is never overridden
    has_new = resolved.is_dir() and any(resolved.glob("*_pertrack.npz"))
    if has_new or not _legacy_has_data():
        return resolved
    logger.info(
        "Using the legacy background directory %s because it already holds "
        "*_pertrack.npz. New installs default to %s; run "
        "`chorus config data-dir --migrate` to move, or set %s to choose.",
        LEGACY_DATA_DIR / "backgrounds", resolved, _ENV_VAR,
    )
    return (LEGACY_DATA_DIR / "backgrounds").resolve()


CHORUS_DATA_DIR = resolve_data_dir()

# ---------------------------------------------------------------------------
# Per-kind directories. All of these follow CHORUS_DATA_DIR.
# ---------------------------------------------------------------------------

CHORUS_ANNOTATIONS_DIR = CHORUS_DATA_DIR / "annotations"
CHORUS_DOWNLOADS_DIR = CHORUS_DATA_DIR / "downloads"
CHORUS_GENOMES_DIR = CHORUS_DATA_DIR / "genomes"
CHORUS_BACKGROUNDS_DIR = resolve_backgrounds_dir(CHORUS_DATA_DIR)
CHORUS_LIB_DIR = CHORUS_DATA_DIR / "lib"
CHORUS_HF_CACHE_DIR = CHORUS_DATA_DIR / "huggingface"

# Conda prefixes belong to the installation, not to the data directory.
CHORUS_ENVIRONMENTS_DIR = CHORUS_ROOT / "environments"

# Credentials stay with the USER, never in a possibly-shared data directory.
CHORUS_CONFIG_PATH = LEGACY_DATA_DIR / "config.toml"

for _d in (CHORUS_ANNOTATIONS_DIR, CHORUS_DOWNLOADS_DIR, CHORUS_GENOMES_DIR,
           CHORUS_BACKGROUNDS_DIR, CHORUS_ENVIRONMENTS_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception as exc:                                    # pragma: no cover
        logger.warning("could not create %s: %s", _d, exc)


def point_hf_cache_at_data_dir() -> str:
    """Send HuggingFace downloads to the data directory instead of ``~/.cache``.

    ``huggingface_hub`` reads ``HF_HOME`` when its constants module is first
    imported, so this has to run **before** anything imports it — which is why it is
    called at the bottom of this module rather than lazily from an oracle. Chorus
    imports ``core.globals`` early enough for that to hold.

    A pre-existing ``HF_HOME`` / ``HF_HUB_CACHE`` is left alone: someone who has
    already pointed the HF cache somewhere deliberately does not want chorus
    second-guessing it.
    """
    for var in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        CHORUS_HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:                                    # pragma: no cover
        logger.warning("could not create %s: %s", CHORUS_HF_CACHE_DIR, exc)
        return ""
    os.environ["HF_HOME"] = str(CHORUS_HF_CACHE_DIR)
    return str(CHORUS_HF_CACHE_DIR)


point_hf_cache_at_data_dir()


def describe_layout() -> dict:
    """Every resolved path plus how it was chosen — for ``chorus config data-dir``."""
    if os.environ.get(_ENV_VAR):
        source = f"{_ENV_VAR} environment variable"
    elif _read_marker():
        source = f"{DATA_DIR_MARKER} in the installation directory"
    elif CHORUS_DATA_DIR == LEGACY_DATA_DIR.resolve():
        source = ("legacy ~/.chorus (already holds backgrounds)"
                  if _legacy_has_data()
                  else "~/.chorus fallback (installation directory not writable)")
    else:
        source = "installation directory (default)"
    return {
        "data_dir": str(CHORUS_DATA_DIR),
        "source": source,
        "install_dir": str(CHORUS_ROOT),
        "annotations": str(CHORUS_ANNOTATIONS_DIR),
        "downloads": str(CHORUS_DOWNLOADS_DIR),
        "genomes": str(CHORUS_GENOMES_DIR),
        "backgrounds": str(CHORUS_BACKGROUNDS_DIR),
        "hf_cache": os.environ.get("HF_HOME", str(CHORUS_HF_CACHE_DIR)),
        "environments": str(CHORUS_ENVIRONMENTS_DIR),
        "config_secrets": str(CHORUS_CONFIG_PATH),
    }
