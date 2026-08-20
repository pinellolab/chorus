"""CLI subcommands for the generic annotation catalog (AnnotationStore).

Usage:
    chorus annotation list
    chorus annotation describe <id>
    chorus annotation download <id>
    chorus annotation add <id> --description ... --genome-build ...
        (--hf-repo ... --hf-filename ... --hf-revision ... | --url ... | --local-path ...)
        [--format bigwig|gtf|bed|other] [--overwrite]
    chorus annotation remove <id> [--delete-file]
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def annotation_list(args):
    """List every known annotation (conservation tracks, GTFs, custom entries)."""
    from ..utils.annotation_store import AnnotationStore

    entries = AnnotationStore().list_annotations()
    for entry in sorted(entries, key=lambda e: (e.origin, e.id)):
        status = "downloaded" if entry.downloaded else "not downloaded"
        size = f"{entry.size_bytes / (1024**3):.1f} GB" if entry.size_bytes else (entry.size_note or "")
        print(f"  [{entry.origin:12s}] {entry.id:24s} {entry.genome_build or '?':6s} {status:14s} {size}")
    return 0


def annotation_describe(args):
    """Show full metadata for one annotation, verifying genome build for bigwigs."""
    from ..utils.annotation_store import AnnotationStore

    try:
        entry = AnnotationStore().describe_annotation(args.annotation_id)
    except Exception as exc:
        logger.error(f"{exc}")
        return 1

    for key, value in entry.as_dict().items():
        print(f"  {key}: {value}")
    return 0


def annotation_download(args):
    """Download (or confirm cached) one annotation by id."""
    from ..utils.annotation_store import AnnotationStore

    try:
        path = AnnotationStore().download_annotation(args.annotation_id)
    except Exception as exc:
        logger.error(f"Failed to download {args.annotation_id}: {exc}")
        return 1

    logger.info(f"✓ {args.annotation_id} ready at {path}")
    return 0


def annotation_add(args):
    """Register a new custom annotation."""
    from ..utils.annotation_store import AnnotationStore

    try:
        entry = AnnotationStore().add_annotation(
            args.annotation_id,
            description=args.description,
            genome_build=args.genome_build,
            format=args.format,
            hf_repo=args.hf_repo,
            hf_filename=args.hf_filename,
            hf_revision=args.hf_revision,
            url=args.url,
            local_path=args.local_path,
            local_filename=args.local_filename,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        logger.error(f"Failed to add {args.annotation_id}: {exc}")
        return 1

    logger.info(f"✓ Added custom annotation {entry.id!r} ({entry.genome_build}, {entry.format})")
    return 0


def annotation_remove(args):
    """Remove a custom annotation entry."""
    from ..utils.annotation_store import AnnotationStore

    try:
        AnnotationStore().remove_custom_annotation(args.annotation_id, delete_file=args.delete_file)
    except Exception as exc:
        logger.error(f"Failed to remove {args.annotation_id}: {exc}")
        return 1

    logger.info(f"✓ Removed custom annotation {args.annotation_id!r}")
    return 0


def register_annotation_subcommand(subparsers):
    """Register the 'annotation' subcommand group on the main CLI parser."""
    ann_parser = subparsers.add_parser(
        "annotation",
        help="List, describe, download, or add annotations (conservation tracks, GTFs, custom)",
        description=(
            "A unified catalog over the conservation tracks (GPN-Star, PhyloP, "
            "PhastCons), GENCODE gene annotations, and any custom annotation you "
            "register yourself. Each entry records its reference genome build; "
            "bigwig-format annotations are physically verified against it on "
            "`describe`, not just trusted."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ann_sub = ann_parser.add_subparsers(dest="annotation_command", help="Annotation commands")

    list_p = ann_sub.add_parser("list", help="List every known annotation")
    list_p.set_defaults(func=annotation_list)

    describe_p = ann_sub.add_parser("describe", help="Show full metadata for one annotation")
    describe_p.add_argument("annotation_id", help="Annotation id (see `chorus annotation list`)")
    describe_p.set_defaults(func=annotation_describe)

    download_p = ann_sub.add_parser("download", help="Download (or confirm cached) one annotation")
    download_p.add_argument("annotation_id", help="Annotation id (see `chorus annotation list`)")
    download_p.set_defaults(func=annotation_download)

    add_p = ann_sub.add_parser(
        "add",
        help="Register a new custom annotation",
        description=(
            "Register a custom annotation, persisted to "
            "<annotations dir>/custom_annotations.yaml. Exactly one source must be "
            "given: --hf-repo/--hf-filename/--hf-revision, --url, or --local-path."
        ),
    )
    add_p.add_argument("annotation_id", help="Id to register the annotation under")
    add_p.add_argument("--description", required=True, help="Human-readable description")
    add_p.add_argument("--genome-build", required=True, dest="genome_build", help="e.g. hg38, hg19, mm10")
    add_p.add_argument("--format", choices=["bigwig", "gtf", "bed", "other"], default=None,
                        help="Inferred from filename if omitted")
    add_p.add_argument("--hf-repo", dest="hf_repo", help="HuggingFace dataset repo id")
    add_p.add_argument("--hf-filename", dest="hf_filename", help="Filename within the HF repo")
    add_p.add_argument("--hf-revision", dest="hf_revision", help="Pinned HF revision (tag/commit, not main/master/HEAD)")
    add_p.add_argument("--url", help="Plain HTTP(S) URL to download from")
    add_p.add_argument("--local-path", dest="local_path", help="Path to a file already on disk")
    add_p.add_argument("--local-filename", dest="local_filename",
                        help="Override the on-disk filename for --hf-repo/--url sources")
    add_p.add_argument("--overwrite", action="store_true", help="Replace an existing custom entry with this id")
    add_p.set_defaults(func=annotation_add)

    remove_p = ann_sub.add_parser("remove", help="Remove a custom annotation entry")
    remove_p.add_argument("annotation_id", help="Custom annotation id to remove")
    remove_p.add_argument("--delete-file", action="store_true", dest="delete_file",
                           help="Also delete the downloaded file, if present")
    remove_p.set_defaults(func=annotation_remove)

    return ann_parser
