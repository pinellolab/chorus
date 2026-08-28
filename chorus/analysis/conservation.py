"""Conservation-score tracks for variant-effect-prediction plots.

Three hg38 conservation sources are wrapped, each as a single genome-wide
bigwig, bulk-downloaded once (mirrors how oracle weights and per-track
backgrounds are cached) rather than streamed per-region:

- **GPN-Star entropy** (``huggingface.co/datasets/songlab/gpn-star-scores``,
  ``bigwig/gpn-star-hg38-v100-200m/entropy.bw``, ~9.9 GB) — shown as two
  tracks: a plain coverage view and a per-nucleotide "sequence logo"
  (reference-base letters scaled by a transformed score). Both display
  ``clip(1 - entropy, 0, 1)`` rather than raw entropy — a fixed 0-1 scale
  using the documented "entropy ~1.0 = neutral" reference point, so the
  most-conserved (lowest entropy) positions get the tallest letters/
  highest values and the baseline sits at a consistent 0 across windows
  (see :func:`_apply_transform`). GPN-Star ships three hg38 models, one per
  multi-species alignment — ``v100`` (100-way **vertebrate**, used here),
  ``m447`` (447-way mammalian), and ``p243`` (243-way primate); the
  ``-hg38-v100-200m`` path segment below is that vertebrate model.
- **PhyloP 100-way** (UCSC ``hg38/phyloP100way/hg38.phyloP100way.bw``,
  ~9.2 GB) — plain coverage only, raw values, no transform. Same 100-way
  vertebrate alignment as GPN-Star's ``v100`` model above.
- **PhastCons 100-way** (UCSC ``hg38/phastCons100way/hg38.phastCons100way.bw``,
  ~5.5 GB) — plain coverage only, raw values, no transform. Same 100-way
  vertebrate alignment as GPN-Star's ``v100`` model above.

All three share the same download/cache/read/IGV-feature machinery; only
the source location and (for GPN-Star) the extra logo rendering differ.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Default width of the window embedded per conservation track in an IGV
# report, centered on the variant. Oracle prediction windows can be far
# larger (AlphaGenome reaches ~1 Mb) than is practical to embed at 1bp
# resolution in a self-contained HTML file; None disables capping (embeds
# the exact region the caller asks for, whatever its size).
DEFAULT_MAX_WINDOW_BP = 100_000

#: Every source wrapped here is hg38, and a bigwig read against the wrong assembly
#: returns plausible numbers rather than an error — chr1:3,000,000 exists in mm10 and in
#: hg38. So the assembly is named once, and callers that plot these tracks check it.
CONSERVATION_ASSEMBLY = "hg38"

#: Pinned so two users on the same chorus commit read the same scores. The repo is a
#: moving dataset: fetching its head would let a re-upload change conservation values
#: with nothing in the tree recording that it happened (which is what
#: `chorus-backgrounds` did on 2026-08-10). Bump deliberately, with the values re-checked.
GPN_STAR_HF_REVISION = "a7b13bbf0d2338d74a7e5f0f8466e41ac0722f50"

_TRACK_SOURCES = {
    "gpn_star": dict(
        kind="hf",
        hf_repo="songlab/gpn-star-scores",
        hf_revision=GPN_STAR_HF_REVISION,
        hf_filename="bigwig/gpn-star-hg38-v100-200m/entropy.bw",
        local_subdir="gpn_star",
        local_filename="entropy.bw",
        size_note="~9.9 GB",
    ),
    "gpn_star_llr_a": dict(
        kind="hf",
        hf_repo="songlab/gpn-star-scores",
        hf_revision=GPN_STAR_HF_REVISION,
        hf_filename="bigwig/gpn-star-hg38-v100-200m/llr_A.bw",
        local_subdir="gpn_star_llr",
        local_filename="llr_A.bw",
        size_note="~11 GB",
    ),
    "gpn_star_llr_c": dict(
        kind="hf",
        hf_repo="songlab/gpn-star-scores",
        hf_revision=GPN_STAR_HF_REVISION,
        hf_filename="bigwig/gpn-star-hg38-v100-200m/llr_C.bw",
        local_subdir="gpn_star_llr",
        local_filename="llr_C.bw",
        size_note="~11.6 GB",
    ),
    "gpn_star_llr_g": dict(
        kind="hf",
        hf_repo="songlab/gpn-star-scores",
        hf_revision=GPN_STAR_HF_REVISION,
        hf_filename="bigwig/gpn-star-hg38-v100-200m/llr_G.bw",
        local_subdir="gpn_star_llr",
        local_filename="llr_G.bw",
        size_note="~11.6 GB",
    ),
    "gpn_star_llr_t": dict(
        kind="hf",
        hf_repo="songlab/gpn-star-scores",
        hf_revision=GPN_STAR_HF_REVISION,
        hf_filename="bigwig/gpn-star-hg38-v100-200m/llr_T.bw",
        local_subdir="gpn_star_llr",
        local_filename="llr_T.bw",
        size_note="~11 GB",
    ),
    "phylop100way": dict(
        kind="url",
        url="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw",
        local_subdir="phylop100way",
        local_filename="hg38.phyloP100way.bw",
        size_note="~9.2 GB",
    ),
    "phastcons100way": dict(
        kind="url",
        url="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw",
        local_subdir="phastcons100way",
        local_filename="hg38.phastCons100way.bw",
        size_note="~5.5 GB",
    ),
}

BASE_COLORS = {
    "A": "#109648",
    "C": "#255C99",
    "G": "#F7B32B",
    "T": "#D62839",
    "N": "#AAAAAA",
}


def _default_downloads_dir() -> Path:
    from chorus.core.globals import CHORUS_DOWNLOADS_DIR
    return CHORUS_DOWNLOADS_DIR


def _local_bigwig_path(track: str, downloads_dir: Path | None = None) -> Path:
    cfg = _TRACK_SOURCES[track]
    downloads_dir = downloads_dir or _default_downloads_dir()
    return Path(downloads_dir) / cfg["local_subdir"] / cfg["local_filename"]


def _has_bigwig(track: str, downloads_dir: Path | None = None) -> bool:
    return _local_bigwig_path(track, downloads_dir).exists()


def _bigwig_path(track: str, downloads_dir: Path | None = None) -> Path:
    """Return the local path for *track*, downloading it on first use.

    No-ops — no network touched — if the flat target already exists.
    """
    cfg = _TRACK_SOURCES[track]
    local_path = _local_bigwig_path(track, downloads_dir)
    if local_path.exists():
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg["kind"] == "hf":
        logger.info(
            "Downloading %s conservation track (%s, %s) from HuggingFace "
            "— this is a one-time download.",
            track, cfg["hf_filename"], cfg["size_note"],
        )
        from chorus.utils.annotation_store import hf_download_flat

        hf_download_flat(
            repo=cfg["hf_repo"],
            filename=cfg["hf_filename"],
            dest=local_path,
            revision=cfg.get("hf_revision"),
        )
    elif cfg["kind"] == "url":
        from chorus.utils.http import download_with_resume

        logger.info(
            "Downloading %s conservation track (%s) from UCSC — this is a "
            "one-time download.",
            track, cfg["size_note"],
        )
        download_with_resume(cfg["url"], local_path, label=track)
    else:
        raise ValueError(f"Unknown track source kind: {cfg['kind']!r}")

    logger.info("%s conservation track cached at %s", track, local_path)
    _verify_assembly(local_path, track)
    return local_path


def _verify_assembly(local_path: Path, track: str) -> None:
    """Check a freshly downloaded bigwig really is the assembly we think it is.

    The store verifies this in ``describe_annotation``/``add_annotation``, but a report
    never calls those — it calls :func:`_bigwig_path`. So a truncated download, a
    substituted mirror, or an hg19 file dropped into ``downloads/phylop100way/`` was read
    and plotted unchecked. Warn-not-raise for an unrecognised file, matching
    :func:`chorus.utils.genome.require_assembly`: refusing an assembly we simply have no
    chr1 length for would be worse than saying so.
    """
    from chorus.core.exceptions import GenomeAssemblyMismatchError

    try:
        from chorus.utils.genome import require_assembly_for_bigwig

        require_assembly_for_bigwig(
            local_path, CONSERVATION_ASSEMBLY, context=f"{track} conservation track",
        )
    except GenomeAssemblyMismatchError:
        # A *confident* mismatch is the case this exists for: an hg19 file where hg38 is
        # expected reads without error and yields plausible numbers about the wrong DNA.
        raise
    except Exception as exc:
        # Anything else — pyBigWig missing, a file that will not open, no chr1 — is not a
        # mismatch claim, and turning it into one would make the download path fail on
        # files the reader would report on far more clearly a moment later.
        logger.warning(
            "Could not verify the assembly of the %s track at %s (%s: %s); continuing.",
            track, local_path, type(exc).__name__, exc,
        )


# ---------------------------------------------------------------------
# Public per-source path/existence helpers
# ---------------------------------------------------------------------

def has_gpn_star_bigwig(downloads_dir: Path | None = None) -> bool:
    """Cheap on-disk existence check — no download attempt."""
    return _has_bigwig("gpn_star", downloads_dir)


def gpn_star_bigwig_path(downloads_dir: Path | None = None) -> Path:
    """Local path to the GPN-Star entropy bigwig, downloading on first use."""
    return _bigwig_path("gpn_star", downloads_dir)


# Per-base calibrated-LLR bigwigs (``llr_{A,C,G,T}.bw``): the reference
# nucleotide at a given position is written as an explicit 0 in its own
# track; the three alternates carry ``llr_calibrated`` (mutation-rate
# calibrated log-likelihood ratio, alt vs. ref). Together with entropy.bw
# these four tracks are what :func:`compute_stacked_logo_heights` needs to
# build a per-base probability distribution.
_LLR_TRACK_BY_BASE = {"A": "gpn_star_llr_a", "C": "gpn_star_llr_c", "G": "gpn_star_llr_g", "T": "gpn_star_llr_t"}


def has_gpn_star_llr_bigwigs(downloads_dir: Path | None = None) -> bool:
    """Cheap on-disk existence check for all four LLR bigwigs — no download attempt."""
    return all(_has_bigwig(track, downloads_dir) for track in _LLR_TRACK_BY_BASE.values())


def gpn_star_llr_bigwig_paths(downloads_dir: Path | None = None) -> dict[str, Path]:
    """Local paths to the four per-base calibrated-LLR bigwigs, keyed by base.

    Downloads each on first use (~11 GB apiece, ~44 GB total) — same
    bulk-download-once behaviour as :func:`gpn_star_bigwig_path`.
    """
    return {base: _bigwig_path(track, downloads_dir) for base, track in _LLR_TRACK_BY_BASE.items()}


def has_phylop_bigwig(downloads_dir: Path | None = None) -> bool:
    return _has_bigwig("phylop100way", downloads_dir)


def phylop_bigwig_path(downloads_dir: Path | None = None) -> Path:
    """Local path to the UCSC PhyloP 100-way bigwig, downloading on first use."""
    return _bigwig_path("phylop100way", downloads_dir)


def has_phastcons_bigwig(downloads_dir: Path | None = None) -> bool:
    return _has_bigwig("phastcons100way", downloads_dir)


def phastcons_bigwig_path(downloads_dir: Path | None = None) -> Path:
    """Local path to the UCSC PhastCons 100-way bigwig, downloading on first use."""
    return _bigwig_path("phastcons100way", downloads_dir)


def list_tracks(downloads_dir: Path | None = None) -> dict[str, dict]:
    """Status info for every known conservation track (for CLI/health use).

    Keys are the internal track identifiers (``gpn_star``, ``phylop100way``,
    ``phastcons100way``). Each value has ``path``, ``downloaded`` (bool),
    ``size_bytes`` (if downloaded, else ``None``), ``size_note``
    (approximate expected size), and ``source`` (``"hf"`` or ``"url"``).
    Pure filesystem check — never triggers a download.
    """
    info = {}
    for track, cfg in _TRACK_SOURCES.items():
        path = _local_bigwig_path(track, downloads_dir)
        downloaded = path.exists()
        info[track] = {
            "path": path,
            "downloaded": downloaded,
            "size_bytes": path.stat().st_size if downloaded else None,
            "size_note": cfg["size_note"],
            "source": cfg["kind"],
        }
    return info


def download_track(track: str, downloads_dir: Path | None = None) -> Path:
    """Download the named conservation track by its :func:`list_tracks` key.

    No-ops — no network touched — if already downloaded. Raises
    ``ValueError`` for an unknown track name.
    """
    if track not in _TRACK_SOURCES:
        raise ValueError(
            f"Unknown conservation track: {track!r}. Valid: {sorted(_TRACK_SOURCES)}"
        )
    return _bigwig_path(track, downloads_dir)


# ---------------------------------------------------------------------
# Reading raw per-base values
# ---------------------------------------------------------------------

def read_bigwig_values(
    chrom: str, start: int, end: int, *, bw_path: str | Path, preserve_nan: bool = False,
) -> np.ndarray:
    """Read per-base values from a local bigwig for a 1-based inclusive region.

    Returns a float array of length ``end - start + 1``. Positions with no coverage are
    NaN in the bigwig; by default they are mapped to 0.0.

    ``preserve_nan=True`` keeps them, and callers that *invert* the scale must use it.
    Mapping no-coverage to 0.0 and then applying ``transform="invert"`` yields
    ``clip(1 - 0, 0, 1) == 1.0`` — the maximum — so an assembly gap rendered as a
    solid full-height bar, indistinguishable from a perfectly constrained base.
    """
    import pyBigWig

    with pyBigWig.open(str(bw_path)) as bw:
        values = bw.values(chrom, start - 1, end, numpy=True)
    if preserve_nan:
        return np.asarray(values, dtype=float)
    return np.nan_to_num(values, nan=0.0)


def read_entropy_values(chrom: str, start: int, end: int, *, bw_path: str | Path | None = None) -> np.ndarray:
    """Read per-base GPN-Star entropy scores. Auto-downloads on first call."""
    path = Path(bw_path) if bw_path is not None else gpn_star_bigwig_path()
    return read_bigwig_values(chrom, start, end, bw_path=path)


def read_llr_values(
    chrom: str, start: int, end: int, *, bw_paths: dict[str, str | Path] | None = None,
) -> dict[str, np.ndarray]:
    """Read per-base calibrated LLR scores from all four ``llr_{A,C,G,T}.bw`` tracks.

    Returns a dict keyed by base (``"A"``, ``"C"``, ``"G"``, ``"T"``), each a
    float array of length ``end - start + 1`` (1-based inclusive region,
    matching :func:`read_bigwig_values`). Auto-downloads on first call.
    """
    paths = bw_paths if bw_paths is not None else gpn_star_llr_bigwig_paths()
    return {base: read_bigwig_values(chrom, start, end, bw_path=path) for base, path in paths.items()}


# ---------------------------------------------------------------------
# LLR-derived per-base probabilities (stacked sequence logo)
# ---------------------------------------------------------------------

_LOGO_BASE_ORDER = ("A", "C", "G", "T")


def compute_stacked_logo_heights(
    chrom: str,
    start: int,
    end: int,
    *,
    genome_fasta: str | Path | None = None,
    entropy_bw_path: str | Path | None = None,
    llr_bw_paths: dict[str, str | Path] | None = None,
) -> dict[str, np.ndarray]:
    """Per-base logo heights from calibrated LLR scores, for a 1-based inclusive region.

    Follows the rule documented at
    ``huggingface.co/datasets/songlab/gpn-star-scores``: the reference
    nucleotide gets logit zero and the three alternates get their
    independently supplied ``llr_calibrated`` values; a stable float64
    softmax over those four logits gives ``p(base)``; each base's logo
    height is ``p(base) * (2 - H)`` for base-2 entropy ``H``.

    The reference-gets-zero rule is enforced explicitly here (not merely
    assumed from the upstream bigwigs already encoding it that way) — this
    keeps it an invariant of this function rather than a fact about
    someone else's file.

    Returns a dict keyed by base, each a float array of length
    ``end - start + 1``, aligned 1:1 with the region's positions. Heights
    are always >= 0 and sum (per position) to at most 2.0.

    *genome_fasta* defaults to the shared hg38 reference (auto-downloaded
    via ``chorus.utils.genome.get_genome``, same as every oracle) — GPN-Star
    is hg38-only, so this doesn't take a ``genome`` param the way
    ``build_igv_html`` does.
    """
    import pyfaidx

    if genome_fasta is None:
        from chorus.utils.genome import get_genome
        genome_fasta = get_genome("hg38")

    entropy = read_entropy_values(chrom, start, end, bw_path=entropy_bw_path)
    llr = read_llr_values(chrom, start, end, bw_paths=llr_bw_paths)

    fasta = pyfaidx.Fasta(str(genome_fasta))
    ref_seq = str(fasta[chrom][start - 1:end]).upper()

    n = len(entropy)
    logits = np.stack([llr[base].astype(np.float64) for base in _LOGO_BASE_ORDER], axis=0)  # (4, n)
    ref_mask = np.zeros((4, n), dtype=bool)
    for i, base in enumerate(ref_seq):
        row = _LOGO_BASE_ORDER.index(base) if base in _LOGO_BASE_ORDER else None
        if row is not None:
            ref_mask[row, i] = True
    logits = np.where(ref_mask, 0.0, logits)

    # Stable float64 softmax: subtract the per-position max before exp().
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=0, keepdims=True)

    h_bits = np.clip(entropy, 0.0, 2.0)
    heights = probs * (2.0 - h_bits)[np.newaxis, :]

    return {base: heights[i] for i, base in enumerate(_LOGO_BASE_ORDER)}


def read_phylop_values(chrom: str, start: int, end: int, *, bw_path: str | Path | None = None) -> np.ndarray:
    """Read per-base PhyloP 100-way scores. Auto-downloads on first call."""
    path = Path(bw_path) if bw_path is not None else phylop_bigwig_path()
    return read_bigwig_values(chrom, start, end, bw_path=path)


def read_phastcons_values(chrom: str, start: int, end: int, *, bw_path: str | Path | None = None) -> np.ndarray:
    """Read per-base PhastCons 100-way scores. Auto-downloads on first call."""
    path = Path(bw_path) if bw_path is not None else phastcons_bigwig_path()
    return read_bigwig_values(chrom, start, end, bw_path=path)


# ---------------------------------------------------------------------
# CoolBox line tracks (plain coverage, raw values)
# ---------------------------------------------------------------------

def _bigwig_coolbox_track(bw_path, *, title, color, height, coolbox_kwargs):
    from coolbox.api import BigWig, Color, Title, TrackHeight

    return (
        BigWig(str(bw_path), **coolbox_kwargs)
        + Color(color)
        + TrackHeight(height)
        + Title(title)
    )


def conservation_coolbox_track(
    *,
    bw_path: str | Path | None = None,
    title: str = "GPN-Star conservation (entropy)",
    color: str = "#2a9d8f",
    height: float = 2.0,
    **coolbox_kwargs,
):
    """Build a CoolBox frame with the GPN-Star entropy track as a bigwig line.

    Reads the downloaded bigwig directly (coolbox's own ``BigWig`` track
    fetches whatever region the frame is displayed at, same as any other
    coolbox track) — no temp bedgraph needed, unlike
    ``OraclePredictionTrack.get_coolbox_representation`` whose data lives
    in memory rather than on disk. Chain with ``+`` alongside other frames,
    e.g. ``pred.get_coolbox_representation() + conservation_coolbox_track()``.
    """
    path = Path(bw_path) if bw_path is not None else gpn_star_bigwig_path()
    return _bigwig_coolbox_track(path, title=title, color=color, height=height, coolbox_kwargs=coolbox_kwargs)


def phylop_coolbox_track(
    *,
    bw_path: str | Path | None = None,
    title: str = "PhyloP 100-way",
    color: str = "#6a4c93",
    height: float = 2.0,
    **coolbox_kwargs,
):
    """Build a CoolBox frame with the raw PhyloP 100-way track as a bigwig line."""
    path = Path(bw_path) if bw_path is not None else phylop_bigwig_path()
    return _bigwig_coolbox_track(path, title=title, color=color, height=height, coolbox_kwargs=coolbox_kwargs)


def phastcons_coolbox_track(
    *,
    bw_path: str | Path | None = None,
    title: str = "PhastCons 100-way",
    color: str = "#1982c4",
    height: float = 2.0,
    **coolbox_kwargs,
):
    """Build a CoolBox frame with the raw PhastCons 100-way track as a bigwig line."""
    path = Path(bw_path) if bw_path is not None else phastcons_bigwig_path()
    return _bigwig_coolbox_track(path, title=title, color=color, height=height, coolbox_kwargs=coolbox_kwargs)


# ---------------------------------------------------------------------
# Sequence logo (GPN-Star only)
# ---------------------------------------------------------------------

def _draw_logo(
    ax,
    ref_seq: str,
    importance: Sequence[float],
    start: int,
    resolution: int = 1,
    ymax: float | None = None,
) -> None:
    """Draw ref-base letters scaled by *importance* onto *ax*.

    Standard sequence-logo glyph technique: each letter is a
    ``matplotlib.textpath.TextPath`` scaled/translated via ``Affine2D`` to
    fill ``[x, x+resolution] x [0, height]``. A single-letter counterpart
    to :func:`_draw_stacked_logo` (used by :class:`SequenceLogoTrack`) for
    any ``(ref_seq, importance)`` pair that isn't a full per-base
    probability distribution — e.g. an ISM-importance logo.
    """
    import matplotlib.transforms as mtransforms
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath

    importance = np.clip(np.asarray(importance, dtype=float), 0.0, None)
    if ymax is None:
        ymax = float(importance.max()) if importance.size and importance.max() > 0 else 1.0

    font = FontProperties(family="monospace", weight="bold")
    for i, base in enumerate(ref_seq):
        h = float(importance[i]) if i < len(importance) else 0.0
        if h <= 0:
            continue
        x = start + i * resolution
        text_path = TextPath((0, 0), base.upper(), size=1, prop=font)
        bbox = text_path.get_extents()
        if bbox.width == 0 or bbox.height == 0:
            continue
        transform = (
            mtransforms.Affine2D()
            .translate(-bbox.x0, -bbox.y0)
            .scale(resolution / bbox.width, h / bbox.height)
            .translate(x, 0)
        )
        patch = PathPatch(
            text_path,
            transform=transform + ax.transData,
            facecolor=BASE_COLORS.get(base.upper(), BASE_COLORS["N"]),
            edgecolor="none",
        )
        ax.add_patch(patch)

    ax.set_xlim(start, start + len(ref_seq) * resolution)
    ax.set_ylim(0, ymax * 1.05)


def _draw_stacked_logo(
    ax,
    heights_by_base: dict[str, Sequence[float]],
    start: int,
    resolution: int = 1,
    ymax: float | None = None,
) -> None:
    """Draw a stacked, multi-letter sequence logo onto *ax*.

    Standard WebLogo-style stacking: at each position, all four bases are
    drawn bottom-to-top in ascending height order (most-likely base on
    top), each a ``TextPath`` glyph scaled to its own height slice —
    unlike :func:`_draw_logo`, which draws only a single (reference)
    letter per position. *heights_by_base* is the
    ``{"A": arr, "C": arr, "G": arr, "T": arr}`` output of
    :func:`compute_stacked_logo_heights`.
    """
    import matplotlib.transforms as mtransforms
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath

    bases = list(heights_by_base.keys())
    arrays = {b: np.clip(np.asarray(heights_by_base[b], dtype=float), 0.0, None) for b in bases}
    n = len(next(iter(arrays.values()))) if arrays else 0

    if ymax is None:
        totals = sum(arrays.values()) if arrays else np.zeros(n)
        ymax = float(totals.max()) if totals.size and totals.max() > 0 else 1.0

    font = FontProperties(family="monospace", weight="bold")
    for i in range(n):
        x = start + i * resolution
        y = 0.0
        for base in sorted(bases, key=lambda b: arrays[b][i]):
            h = float(arrays[base][i])
            if h <= 0:
                continue
            text_path = TextPath((0, 0), base.upper(), size=1, prop=font)
            bbox = text_path.get_extents()
            if bbox.width == 0 or bbox.height == 0:
                y += h
                continue
            transform = (
                mtransforms.Affine2D()
                .translate(-bbox.x0, -bbox.y0)
                .scale(resolution / bbox.width, h / bbox.height)
                .translate(x, y)
            )
            patch = PathPatch(
                text_path,
                transform=transform + ax.transData,
                facecolor=BASE_COLORS.get(base.upper(), BASE_COLORS["N"]),
                edgecolor="none",
            )
            ax.add_patch(patch)
            y += h

    ax.set_xlim(start, start + n * resolution)
    ax.set_ylim(0, ymax * 1.05)


_sequence_logo_track_cls = None


def _get_sequence_logo_track_cls():
    """Lazily build the ``SequenceLogoTrack`` class.

    Every coolbox usage elsewhere in chorus imports it function-locally
    rather than at module top (see ``chorus/core/result.py``), even though
    it's a hard dependency of the base env — this keeps that same
    convention so importing ``chorus.analysis.conservation`` never forces
    coolbox's import-time side effects.
    """
    global _sequence_logo_track_cls
    if _sequence_logo_track_cls is None:
        from coolbox.core.track.base import Track

        class SequenceLogoTrack(Track):
            """CoolBox track: LLR-derived stacked sequence logo.

            Reads GPN-Star entropy, the four calibrated-LLR bigwigs, and
            reference sequence (pyfaidx) for whatever GenomeRange the
            enclosing frame is displayed at — same lazy, render-time-region
            model as coolbox's own ``BigWig`` track. Composes into
            existing frames via ``+`` like any other CoolBox track. Per
            position, height is ``p(base) * (2 - H)`` for each of the four
            bases (see :func:`compute_stacked_logo_heights`) — a fixed
            0-2 bit scale so the baseline is consistent across windows.
            """

            DEFAULT_PROPERTIES = {"height": 2, "title": "", "ymax": 2.0}

            def __init__(
                self,
                genome_fasta: str | Path | None = None,
                entropy_bw_path: str | Path | None = None,
                llr_bw_paths: dict[str, str | Path] | None = None,
                **kwargs,
            ):
                properties = SequenceLogoTrack.DEFAULT_PROPERTIES.copy()
                properties.update(kwargs)
                super().__init__(properties)

                self._genome_fasta = genome_fasta
                self._entropy_bw_path = entropy_bw_path
                self._llr_bw_paths = llr_bw_paths

            def fetch_data(self, gr, **kwargs):
                # coolbox GenomeRange is 0-based half-open (its own to_gr("chr1:0-1000")
                # gives start 0); compute_stacked_logo_heights takes 1-based inclusive.
                # Passing gr.start straight through fetched one extra base, one to the
                # left, so the letters sat a base off from every coolbox track above.
                return compute_stacked_logo_heights(
                    gr.chrom, gr.start + 1, gr.end,
                    genome_fasta=self._genome_fasta,
                    entropy_bw_path=self._entropy_bw_path,
                    llr_bw_paths=self._llr_bw_paths,
                )

            def plot(self, ax, gr, **kwargs) -> None:
                self.ax = ax
                heights = self.fetch_data(gr)
                # gr.start + 1 is the 1-based coordinate of the first fetched value, which
                # is what _draw_stacked_logo lays letters out from.
                _draw_stacked_logo(ax, heights, gr.start + 1, ymax=self.properties.get("ymax"))
                self.plot_label()

        _sequence_logo_track_cls = SequenceLogoTrack
    return _sequence_logo_track_cls


def __getattr__(name):
    if name == "SequenceLogoTrack":
        return _get_sequence_logo_track_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def conservation_logo_track(
    genome_fasta: str | Path | None = None,
    *,
    entropy_bw_path: str | Path | None = None,
    llr_bw_paths: dict[str, str | Path] | None = None,
    **kwargs,
):
    """Convenience constructor for a ``SequenceLogoTrack``.

    No region args needed — like ``conservation_coolbox_track``, the track
    fetches data for whatever region the enclosing CoolBox frame is
    displayed at. *genome_fasta* defaults to the shared hg38 reference
    (see :func:`compute_stacked_logo_heights`).
    """
    cls = _get_sequence_logo_track_cls()
    return cls(genome_fasta, entropy_bw_path=entropy_bw_path, llr_bw_paths=llr_bw_paths, **kwargs)


# ---------------------------------------------------------------------
# IGV wig feature builders
# ---------------------------------------------------------------------

def _clip_window(
    chrom: str, start: int, end: int, *, center: int | None, max_window_bp: int | None,
) -> tuple[int, int]:
    window_bp = end - start + 1
    if max_window_bp is None or window_bp <= max_window_bp:
        return start, end
    c = center if center is not None else (start + end) // 2
    half = max_window_bp // 2
    clipped_start = max(start, c - half)
    clipped_end = min(end, c + half)
    logger.info(
        "Conservation track window (%d bp) exceeds the %d bp cap; "
        "clipping to %s:%d-%d (centered on %d) to keep true per-base "
        "resolution and a reasonably sized report.",
        window_bp, max_window_bp, chrom, clipped_start, clipped_end, c,
    )
    return clipped_start, clipped_end


def _apply_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return values
    if transform == "invert":
        # Remap entropy to a fixed [0, 1] conservation score: clip(-entropy,
        # -1, 0) + 1, equivalently clip(1 - entropy, 0, 1). Uses the fixed
        # reference point "entropy ~1.0 = neutral" documented for GPN-Star,
        # NOT a window-relative max — an earlier window-relative version
        # (max(window) - entropy) made "neutral" positions land wherever
        # the window's own max happened to be (e.g. 0.6 in a window
        # topping out at 1.6), which read as a shifted/noisy baseline
        # instead of a clean 0. With a fixed reference, entropy=1
        # (neutral) always maps to 0 and entropy=0 (fully constrained)
        # always maps to 1, regardless of what else is in the window.
        # Always in [0, 1] by construction (clipped), so — unlike a plain
        # negation — this is safe to feed directly to IGV's dynseq graph
        # type: no negative values means letters never render
        # upside-down, and the same array can drive both a coverage view
        # and the sequence-logo heights.
        return np.clip(1.0 - values, 0.0, 1.0)
    raise ValueError(f"Unknown transform: {transform!r}")


def _bigwig_igv_features(
    chrom: str,
    start: int,
    end: int,
    *,
    bw_path: str | Path,
    center: int | None = None,
    max_window_bp: int | None = DEFAULT_MAX_WINDOW_BP,
    transform: str = "raw",
) -> list[dict]:
    from ._igv_report import _downsample_to_features

    start, end = _clip_window(chrom, start, end, center=center, max_window_bp=max_window_bp)
    # preserve_nan so no-coverage stays distinguishable from a real 0 through the
    # transform; the uncovered positions are dropped below rather than plotted.
    values = read_bigwig_values(chrom, start, end, bw_path=bw_path, preserve_nan=True)
    values = _apply_transform(values, transform)
    uncovered = np.isnan(values)
    # _downsample_to_features expects a 0-based feature-coordinate origin
    # (matches every other IGV track built in _igv_report.py); our own
    # start/end args are 1-based inclusive. Always 1bp resolution — no
    # mean-aggregation — so per-base rendering (e.g. IGV's dynseq graph
    # type) never shows averaged, misleadingly-flat blocks.
    features = _downsample_to_features(
        np.nan_to_num(values, nan=0.0), chrom, start - 1,
        resolution=1, bin_size=1, skip_zeros=False,
    )
    if not uncovered.any():
        return features
    # 1bp resolution with skip_zeros=False gives one feature per input value, so the
    # mask lines up positionally. Assert it rather than trust it: a future change to
    # either argument would otherwise silently drop the wrong positions.
    if len(features) != len(values):
        logger.warning(
            "conservation: %d features for %d values — cannot map no-coverage positions, "
            "leaving them in", len(features), len(values),
        )
        return features
    return [f for f, missing in zip(features, uncovered) if not missing]


def conservation_igv_features(
    chrom: str,
    start: int,
    end: int,
    *,
    bw_path: str | Path | None = None,
    center: int | None = None,
    max_window_bp: int | None = DEFAULT_MAX_WINDOW_BP,
    transform: str = "raw",
) -> list[dict]:
    """Build IGV wig feature dicts for the GPN-Star entropy track.

    By default the region is capped to ``DEFAULT_MAX_WINDOW_BP`` centered
    on *center* (see :func:`_clip_window`); pass ``max_window_bp=None`` for
    no cap. ``transform="invert"`` remaps entropy to a fixed
    ``clip(1 - entropy, 0, 1)`` conservation score so the *most conserved*
    (lowest entropy) position gets the *highest* value/letter-height — see
    :func:`_apply_transform`. Because that's always in [0, 1], the same
    values can safely drive both a coverage view and IGV's ``dynseq``
    sequence logo (a plain negation could go negative, which makes dynseq
    render letters upside-down).
    """
    path = Path(bw_path) if bw_path is not None else gpn_star_bigwig_path()
    return _bigwig_igv_features(
        chrom, start, end, bw_path=path, center=center, max_window_bp=max_window_bp, transform=transform,
    )


def conservation_stacked_logo_igv_features(
    chrom: str,
    start: int,
    end: int,
    *,
    genome_fasta: str | Path | None = None,
    entropy_bw_path: str | Path | None = None,
    llr_bw_paths: dict[str, str | Path] | None = None,
    center: int | None = None,
    max_window_bp: int | None = DEFAULT_MAX_WINDOW_BP,
) -> list[dict]:
    """Build IGV feature dicts for the LLR-derived stacked-base logo track.

    Unlike every other track in this module, each feature carries four
    per-base heights (``pA``/``pC``/``pG``/``pT``, see
    :func:`compute_stacked_logo_heights`) instead of a single ``value`` —
    consumed client-side by the custom ``gpnStarStackedLogo`` IGV.js track
    type (``chorus/analysis/static/gpn_star_logo_track.js``), which draws
    four stacked colored rectangles per position. Same window-clipping
    discipline as :func:`conservation_igv_features`: always 1bp resolution,
    capped to ``max_window_bp`` around *center*.
    """
    start, end = _clip_window(chrom, start, end, center=center, max_window_bp=max_window_bp)
    heights = compute_stacked_logo_heights(
        chrom, start, end,
        genome_fasta=genome_fasta,
        entropy_bw_path=entropy_bw_path,
        llr_bw_paths=llr_bw_paths,
    )
    n = len(heights[_LOGO_BASE_ORDER[0]])
    features = []
    for i in range(n):
        pos0 = start - 1 + i  # 0-based feature coordinate, matches every other IGV track here
        features.append({
            "chr": chrom,
            "start": pos0,
            "end": pos0 + 1,
            "pA": float(heights["A"][i]),
            "pC": float(heights["C"][i]),
            "pG": float(heights["G"][i]),
            "pT": float(heights["T"][i]),
        })
    return features


def phylop_igv_features(
    chrom: str,
    start: int,
    end: int,
    *,
    bw_path: str | Path | None = None,
    center: int | None = None,
    max_window_bp: int | None = DEFAULT_MAX_WINDOW_BP,
) -> list[dict]:
    """Build IGV wig feature dicts for the raw PhyloP 100-way track (no transform)."""
    path = Path(bw_path) if bw_path is not None else phylop_bigwig_path()
    return _bigwig_igv_features(
        chrom, start, end, bw_path=path, center=center, max_window_bp=max_window_bp, transform="raw",
    )


def phastcons_igv_features(
    chrom: str,
    start: int,
    end: int,
    *,
    bw_path: str | Path | None = None,
    center: int | None = None,
    max_window_bp: int | None = DEFAULT_MAX_WINDOW_BP,
) -> list[dict]:
    """Build IGV wig feature dicts for the raw PhastCons 100-way track (no transform)."""
    path = Path(bw_path) if bw_path is not None else phastcons_bigwig_path()
    return _bigwig_igv_features(
        chrom, start, end, bw_path=path, center=center, max_window_bp=max_window_bp, transform="raw",
    )
