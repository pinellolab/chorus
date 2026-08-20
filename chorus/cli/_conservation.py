"""CLI subcommands for managing conservation-track bigwigs.

Usage:
    chorus conservation status
    chorus conservation download [--track NAME] [--all]
"""

import argparse
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# The four calibrated-LLR bigwigs are only useful together (they jointly
# feed the stacked sequence-logo track), so `download --track gpn_star_llr`
# expands to all four rather than making users invoke download 4 times.
_TRACK_ALIASES = {
    "gpn_star_llr": ["gpn_star_llr_a", "gpn_star_llr_c", "gpn_star_llr_g", "gpn_star_llr_t"],
}


def conservation_status(args):
    """Show download status for every conservation track."""
    from ..analysis import conservation

    info = conservation.list_tracks()

    for track, status in info.items():
        if status["downloaded"]:
            size_mb = status["size_bytes"] / (1024 * 1024)
            mtime = datetime.fromtimestamp(status["path"].stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  {track:16s}  {size_mb / 1024:6.1f} GB  {mtime}  {status['path']}")
        else:
            print(f"  {track:16s}  -- not downloaded ({status['size_note']}) --")

    return 0


def conservation_download(args):
    """Download one, several, or all conservation tracks."""
    from ..analysis import conservation

    if args.all:
        tracks = list(conservation.list_tracks().keys())
    elif args.track:
        tracks = _TRACK_ALIASES.get(args.track, [args.track])
    else:
        logger.error("Specify --track NAME or --all. Run `chorus conservation status` to see names.")
        return 1

    known = conservation.list_tracks()
    unknown = [t for t in tracks if t not in known]
    if unknown:
        logger.error(
            f"Unknown track(s): {', '.join(unknown)}. Valid: {', '.join(sorted(known))}."
        )
        return 1

    all_ok = True
    for track in tracks:
        status = known[track]
        if status["downloaded"]:
            logger.info(f"{track}: already downloaded at {status['path']}")
            continue
        logger.info(f"Downloading {track} ({status['size_note']})...")
        try:
            path = conservation.download_track(track)
            logger.info(f"✓ {track} ready at {path}")
        except Exception as exc:
            logger.error(f"✗ Failed to download {track}: {exc}")
            all_ok = False

    return 0 if all_ok else 1


def register_conservation_subcommand(subparsers):
    """Register the 'conservation' subcommand group on the main CLI parser."""
    cons_parser = subparsers.add_parser(
        "conservation",
        help="Manage conservation-track bigwigs (GPN-Star, PhyloP, PhastCons)",
        description=(
            "View and pre-download the hg38 conservation bigwigs used by "
            "show_conservation=True in variant reports: GPN-Star entropy, "
            "vertebrate-alignment model (HuggingFace, ~9.9 GB — GPN-Star "
            "also ships mammalian and primate models, not fetched here), "
            "the four GPN-Star calibrated-LLR tracks that feed the stacked "
            "sequence-logo track (HuggingFace, ~11 GB each, ~44 GB total — "
            "download together with `--track gpn_star_llr`), and PhyloP "
            "100-way / PhastCons 100-way, the same 100-way vertebrate "
            "alignment (UCSC, ~5.5-9.2 GB each). Each is bulk-downloaded "
            "once into downloads/<track>/ and cached for reuse — without "
            "this command they still download lazily the first time a "
            "report requests them."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cons_sub = cons_parser.add_subparsers(dest="conservation_command", help="Conservation commands")

    status_p = cons_sub.add_parser("status", help="Show download status for every conservation track")
    status_p.set_defaults(func=conservation_status)

    download_p = cons_sub.add_parser(
        "download",
        help="Pre-download conservation track bigwig(s)",
        description=(
            "Pre-download one or all conservation tracks so the first "
            "show_conservation=True report doesn't pay the download cost. "
            "No-ops for tracks already downloaded."
        ),
    )
    download_p.add_argument(
        "--track",
        help=(
            "Track to download (see `chorus conservation status` for names); "
            "`gpn_star_llr` is a bundle alias for all four calibrated-LLR tracks"
        ),
    )
    download_p.add_argument("--all", action="store_true", help="Download every conservation track")
    download_p.set_defaults(func=conservation_download)

    return cons_parser
