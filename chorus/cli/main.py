"""Main CLI entry point for Chorus."""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

from ..core.environment import EnvironmentManager

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def setup_environments(args):
    """Set up oracle environments."""
    manager = EnvironmentManager()
    
    if args.oracle:
        # Set up specific oracle
        oracles = [args.oracle]
    else:
        # Set up all oracles
        oracles = manager.list_available_oracles()
        if not oracles:
            logger.error("No oracle environment definitions found.")
            return 1
    
    success_count = 0
    for oracle in oracles:
        logger.info(f"Setting up environment for {oracle}...")
        
        if manager.create_environment(oracle, force=args.force):
            logger.info(f"✓ Successfully set up {oracle}")
            success_count += 1
        else:
            logger.error(f"✗ Failed to set up {oracle}")
    
    logger.info(f"\nSetup complete: {success_count}/{len(oracles)} environments created.")
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
    
    all_healthy = True
    
    for oracle in oracles:
        logger.info(f"\nChecking {oracle}...")
        health = runner.check_environment_health(oracle)
        
        if health['errors']:
            logger.error(f"✗ {oracle}: Unhealthy")
            for error in health['errors']:
                logger.error(f"  - {error}")
            all_healthy = False
        else:
            logger.info(f"✓ {oracle}: Healthy")
            
            if args.verbose and health.get('metadata'):
                metadata = health['metadata']
                logger.info(f"  Class: {metadata.get('class_name')}")
                logger.info(f"  Assay types: {len(metadata.get('assay_types', []))}")
                logger.info(f"  Cell types: {len(metadata.get('cell_types', []))}")
    
    return 0 if all_healthy else 1


def main(argv: Optional[List[str]] = None):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chorus: Modular framework for genomic foundation models"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up oracle environments')
    setup_parser.add_argument(
        '--oracle', 
        help='Specific oracle to set up (default: all)'
    )
    setup_parser.add_argument(
        '--force', 
        action='store_true',
        help='Force recreation of existing environments'
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
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information'
    )
    health_parser.set_defaults(func=check_health)
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Execute command
    return args.func(args)


# Create cli alias for setuptools entry point
cli = main

if __name__ == '__main__':
    sys.exit(main())