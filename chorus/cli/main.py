"""Main CLI entry point for Chorus."""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

from ..core.environment import EnvironmentManager
from ..utils.genome import GenomeManager

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def setup_environments(args):
    """Set up oracle environments + pre-download weights, backgrounds, genome."""
    manager = EnvironmentManager()

    if args.oracle == "base":
        logger.error(
            "'base' is not an oracle — the base 'chorus' environment is "
            "installed directly from the root environment.yml, not via "
            "`chorus setup`. Run:\n\n"
            "    mamba env create -f environment.yml\n"
            "    mamba activate chorus\n"
            "    pip install -e ."
        )
        return 1

    # Route both "chorus setup --oracle all" AND "chorus setup" (no --oracle)
    # through the same setup-all flow so the LDlink prompt and HF-token
    # gate apply uniformly.
    if args.oracle == "all" or args.oracle is None:
        from ._setup_all import setup_all_oracles
        return setup_all_oracles(args)

    oracles = [args.oracle]

    # AlphaGenome is gated; resolve the HF token up front so we don't
    # build a multi-GB env only to fail on auth.
    if "alphagenome" in [o.lower() for o in oracles] and not args.no_weights:
        from ._tokens import resolve_hf_token
        if not resolve_hf_token(cli_token=getattr(args, "hf_token", None), interactive=True):
            logger.error(
                "AlphaGenome requires a HuggingFace token. Set HF_TOKEN, run "
                "'huggingface-cli login', or pass --hf-token."
            )
            return 1

    from ..core.environment import EnvironmentRunner
    from ..core.weights_probe import write_setup_marker
    from ._setup_prefetch import prefetch_for_oracle
    runner = EnvironmentRunner(manager)

    success_count = 0
    for oracle in oracles:
        logger.info(f"Setting up environment for {oracle}...")

        if args.force:
            # Drop any stale marker up front so a mid-run failure doesn't
            # leave `chorus health` reporting Healthy for a half-rebuilt
            # env. The marker is re-created at the end only on success.
            from ..core.weights_probe import setup_marker_path
            stale = setup_marker_path(oracle)
            if stale.exists():
                stale.unlink()

        if not manager.create_environment(oracle, force=args.force):
            logger.error(f"✗ Failed to build env for {oracle}")
            continue
        logger.info(f"✓ env for {oracle}")

        if args.no_weights and args.no_backgrounds and args.no_genome:
            logger.info(
                f"Skipping all data prefetch for {oracle} (--no-weights, "
                "--no-backgrounds, --no-genome). Setup marker NOT written."
            )
            success_count += 1
            continue

        ok, errors = prefetch_for_oracle(
            oracle,
            runner,
            skip_weights=args.no_weights,
            skip_backgrounds=args.no_backgrounds,
            skip_genome=args.no_genome,
        )
        if not ok:
            logger.error(f"✗ prefetch failed for {oracle}:")
            for err in errors:
                logger.error(f"  - {err}")
            continue

        if args.no_weights:
            # Don't claim a setup is complete when the user explicitly
            # opted out of the download that makes `chorus health` pass.
            logger.info(
                f"✓ {oracle} env ready (weights skipped — setup marker NOT written)"
            )
        else:
            write_setup_marker(oracle)
            logger.info(f"✓ {oracle} ready")
        success_count += 1

    logger.info(f"\nSetup complete: {success_count}/{len(oracles)} oracles ready.")
    return 0 if success_count == len(oracles) else 1


def list_environments(args):
    """List available oracle environments."""
    manager = EnvironmentManager()
    
    available_oracles = manager.list_available_oracles()
    
    if not available_oracles:
        print("No oracle environment definitions found.")
        return 0
    
    print("Available oracle environments:")
    print("-" * 50)
    
    for oracle in available_oracles:
        env_name = manager.get_environment_name(oracle)
        exists = manager.environment_exists(oracle)
        status = "✓ Installed" if exists else "✗ Not installed"
        
        print(f"{oracle:<20} {env_name:<25} {status}")
        
        if args.verbose and exists:
            info = manager.get_environment_info(oracle)
            if info:
                print(f"  Path: {info['path']}")
                print(f"  Packages: {len(info.get('packages', []))}")
    
    print("-" * 50)
    return 0


def validate_environments(args):
    """Validate oracle environments."""
    manager = EnvironmentManager()
    
    if args.oracle:
        oracles = [args.oracle]
    else:
        oracles = [o for o in manager.list_available_oracles() 
                  if manager.environment_exists(o)]
    
    if not oracles:
        logger.info("No installed environments to validate.")
        return 0
    
    all_valid = True
    
    for oracle in oracles:
        is_valid, issues = manager.validate_environment(oracle)
        
        if is_valid:
            logger.info(f"✓ {oracle}: Valid")
        else:
            logger.error(f"✗ {oracle}: Invalid")
            for issue in issues:
                logger.error(f"  - {issue}")
            all_valid = False
    
    return 0 if all_valid else 1


def remove_environments(args):
    """Remove oracle environments."""
    manager = EnvironmentManager()
    
    if not args.oracle:
        logger.error("Please specify an oracle to remove with --oracle")
        return 1
    
    if not manager.environment_exists(args.oracle):
        logger.error(f"Environment for {args.oracle} does not exist.")
        return 1
    
    # Confirm removal
    if not args.yes:
        response = input(f"Remove environment for {args.oracle}? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Removal cancelled.")
            return 0
    
    if manager.remove_environment(args.oracle):
        logger.info(f"Successfully removed environment for {args.oracle}")
        return 0
    else:
        logger.error(f"Failed to remove environment for {args.oracle}")
        return 1


def check_health(args):
    """Check health of oracle environments."""
    manager = EnvironmentManager()
    from ..core.environment import EnvironmentRunner
    
    runner = EnvironmentRunner(manager)
    
    if args.oracle:
        oracles = [args.oracle]
    else:
        oracles = [o for o in manager.list_available_oracles() 
                  if manager.environment_exists(o)]
    
    if not oracles:
        logger.info("No installed environments to check.")
        return 0
    
    all_ok = True

    for oracle in oracles:
        logger.info(f"\nChecking {oracle}...")
        health = runner.check_environment_health(oracle, timeout=args.timeout)

        if health.get('weights_status') == 'missing':
            logger.warning(
                f"⚠ {oracle}: Not installed — run `chorus setup {oracle}`"
            )
            for reason in health.get('missing_artifacts', []):
                logger.warning(f"  - {reason}")
            all_ok = False
        elif health['errors']:
            logger.error(f"✗ {oracle}: Unhealthy")
            for error in health['errors']:
                logger.error(f"  - {error}")
            all_ok = False
        else:
            logger.info(f"✓ {oracle}: Healthy")

            if args.verbose and health.get('metadata'):
                metadata = health['metadata']
                logger.info(f"  Class: {metadata.get('class_name')}")
                logger.info(f"  Assay types: {len(metadata.get('assay_types', []))}")
                logger.info(f"  Cell types: {len(metadata.get('cell_types', []))}")

    return 0 if all_ok else 1


def list_genomes(args):
    """List available and downloaded genomes."""
    manager = GenomeManager()
    
    available = manager.list_available_genomes()
    downloaded = manager.list_downloaded_genomes()
    
    print("Available genomes:")
    print("-" * 70)
    
    for genome_id, description in available.items():
        status = "✓ Downloaded" if genome_id in downloaded else "✗ Not downloaded"
        print(f"{genome_id:<10} {description:<45} {status}")
        
        if args.verbose and genome_id in downloaded:
            info = manager.get_genome_info(genome_id)
            if info:
                print(f"  Path: {info['path']}")
                print(f"  Size: {info['size_mb']:.1f} MB")
                if 'num_chromosomes' in info:
                    print(f"  Chromosomes: {info['num_chromosomes']}")
    
    print("-" * 70)
    print(f"Total: {len(downloaded)}/{len(available)} genomes downloaded")
    return 0


def download_genome(args):
    """Download a reference genome."""
    manager = GenomeManager()
    
    if args.genome not in manager.list_available_genomes():
        logger.error(f"Unknown genome: {args.genome}")
        logger.info(f"Available genomes: {', '.join(manager.list_available_genomes().keys())}")
        return 1
    
    if manager.is_genome_downloaded(args.genome) and not args.force:
        logger.info(f"Genome {args.genome} is already downloaded.")
        logger.info("Use --force to re-download.")
        return 0
    
    logger.info(f"Downloading {args.genome}...")
    if manager.download_genome(args.genome, force=args.force):
        logger.info(f"Successfully downloaded {args.genome}")
        return 0
    else:
        logger.error(f"Failed to download {args.genome}")
        return 1


def remove_genome(args):
    """Remove a downloaded genome."""
    manager = GenomeManager()
    
    if not manager.is_genome_downloaded(args.genome):
        logger.error(f"Genome {args.genome} is not downloaded.")
        return 1
    
    # Get genome info before removal
    info = manager.get_genome_info(args.genome)
    
    # Confirm removal
    if not args.yes:
        if info:
            print(f"Genome: {args.genome}")
            print(f"Path: {info['path']}")
            print(f"Size: {info['size_mb']:.1f} MB")
        response = input(f"Remove genome {args.genome}? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Removal cancelled.")
            return 0
    
    if manager.remove_genome(args.genome):
        logger.info(f"Successfully removed genome {args.genome}")
        return 0
    else:
        logger.error(f"Failed to remove genome {args.genome}")
        return 1


def genome_info(args):
    """Show detailed information about a genome."""
    manager = GenomeManager()
    
    if not manager.is_genome_downloaded(args.genome):
        logger.error(f"Genome {args.genome} is not downloaded.")
        return 1
    
    info = manager.get_genome_info(args.genome)
    if not info:
        logger.error(f"Could not get information for genome {args.genome}")
        return 1
    
    print(f"\nGenome: {info['id']}")
    print(f"Description: {info['description']}")
    print(f"Path: {info['path']}")
    print(f"Size: {info['size_mb']:.1f} MB")
    
    if 'num_chromosomes' in info:
        print(f"\nChromosomes: {info['num_chromosomes']}")
        print(f"Total length: {info['total_length']:,} bp")
        
        if args.verbose and 'chromosomes' in info:
            print("\nChromosome details:")
            print("-" * 40)
            for chrom in info['chromosomes'][:10]:  # Show first 10
                print(f"{chrom['name']:<15} {chrom['length']:>15,} bp")
            if len(info['chromosomes']) > 10:
                print(f"... and {len(info['chromosomes']) - 10} more chromosomes")
    
    return 0


def main(argv: Optional[List[str]] = None):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chorus: Modular framework for genomic foundation models"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser(
        'setup',
        help='Set up oracle environments + pre-download weights, backgrounds, hg38',
        description=(
            "Build the oracle's conda environment AND pre-download its "
            "default weights, background CDFs, and the hg38 reference so "
            "the oracle is ready to predict after setup finishes.\n\n"
            "Tokens:\n"
            "  HF_TOKEN / --hf-token  required for alphagenome (gated HF model)\n"
            "  LDLINK_TOKEN           optional, only for fine_map_causal_variant"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    setup_parser.add_argument(
        '--oracle',
        help='Specific oracle to set up, or "all" to set up every oracle (default: all if omitted)'
    )
    setup_parser.add_argument(
        '--force',
        action='store_true',
        help='Force recreation of existing environments'
    )
    setup_parser.add_argument(
        '--no-weights',
        action='store_true',
        help="Skip weight pre-download (env only — setup marker NOT written)"
    )
    setup_parser.add_argument(
        '--no-backgrounds',
        action='store_true',
        help='Skip pre-download of background CDFs from HuggingFace'
    )
    setup_parser.add_argument(
        '--no-genome',
        action='store_true',
        help='Skip pre-download of the hg38 reference'
    )
    setup_parser.add_argument(
        '--hf-token',
        default=None,
        help='HuggingFace token (required for alphagenome if not already logged in)'
    )
    setup_parser.set_defaults(func=setup_environments)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available environments')
    list_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information'
    )
    list_parser.set_defaults(func=list_environments)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate environments')
    validate_parser.add_argument(
        '--oracle',
        help='Specific oracle to validate (default: all installed)'
    )
    validate_parser.set_defaults(func=validate_environments)
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove an environment')
    remove_parser.add_argument(
        '--oracle',
        required=True,
        help='Oracle environment to remove'
    )
    remove_parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )
    remove_parser.set_defaults(func=remove_environments)
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check environment health')
    health_parser.add_argument(
        '--oracle',
        help='Specific oracle to check (default: all installed)'
    )
    health_parser.add_argument(
        '--timeout',
        help='Timeout for health check per single oracle (default: 120 seconds)',
        default=120,
        type=int
    )
    health_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information'
    )
    health_parser.set_defaults(func=check_health)
    
    # Genome commands
    genome_parser = subparsers.add_parser('genome', help='Manage reference genomes')
    genome_subparsers = genome_parser.add_subparsers(dest='genome_command', help='Genome commands')
    
    # Genome list
    genome_list_parser = genome_subparsers.add_parser('list', help='List available genomes')
    genome_list_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information'
    )
    genome_list_parser.set_defaults(func=list_genomes)
    
    # Genome download
    genome_download_parser = genome_subparsers.add_parser('download', help='Download a genome')
    genome_download_parser.add_argument(
        'genome',
        help='Genome to download (e.g., hg38, hg19, mm10)'
    )
    genome_download_parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if genome exists'
    )
    genome_download_parser.set_defaults(func=download_genome)
    
    # Genome remove
    genome_remove_parser = genome_subparsers.add_parser('remove', help='Remove a genome')
    genome_remove_parser.add_argument(
        'genome',
        help='Genome to remove'
    )
    genome_remove_parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )
    genome_remove_parser.set_defaults(func=remove_genome)
    
    # Genome info
    genome_info_parser = genome_subparsers.add_parser('info', help='Show genome information')
    genome_info_parser.add_argument(
        'genome',
        help='Genome to show information for'
    )
    genome_info_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show chromosome details'
    )
    genome_info_parser.set_defaults(func=genome_info)
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Handle genome subcommand without action
    if args.command == 'genome' and not hasattr(args, 'func'):
        genome_parser.print_help()
        return 0
    
    # Execute command
    return args.func(args)


# Create cli alias for setuptools entry point
cli = main

if __name__ == '__main__':
    sys.exit(main())