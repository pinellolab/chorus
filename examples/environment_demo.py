"""
Demo script showing how to use Chorus with modular environments.

This script demonstrates:
1. Setting up oracle-specific conda environments
2. Running predictions in isolated environments
3. Managing multiple oracles with different dependencies
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chorus.core.environment import EnvironmentManager, EnvironmentRunner
from chorus.oracles.enformer_env import EnformerOracleEnv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_environments():
    """Set up conda environments for oracles."""
    logger.info("=== Setting Up Oracle Environments ===")
    
    manager = EnvironmentManager()
    
    # List available oracle environments
    oracles = manager.list_available_oracles()
    logger.info(f"Found {len(oracles)} oracle environment definitions: {oracles}")
    
    # Check which environments exist
    for oracle in oracles:
        env_name = manager.get_environment_name(oracle)
        exists = manager.environment_exists(oracle)
        logger.info(f"{oracle}: {env_name} - {'Exists' if exists else 'Not installed'}")
    
    # Set up enformer environment (as an example)
    logger.info("\nSetting up Enformer environment...")
    success = manager.create_environment('enformer', force=False)
    
    if success:
        logger.info("✓ Enformer environment created successfully!")
        
        # Get environment info
        info = manager.get_environment_info('enformer')
        if info:
            logger.info(f"  Path: {info['path']}")
            logger.info(f"  Python: {manager.get_python_executable('enformer')}")
    else:
        logger.error("✗ Failed to create Enformer environment")
    
    return manager


def test_environment_runner(manager):
    """Test running code in isolated environments."""
    logger.info("\n=== Testing Environment Runner ===")
    
    runner = EnvironmentRunner(manager)
    
    # Test 1: Simple function execution
    def get_numpy_version():
        import numpy as np
        return np.__version__
    
    try:
        version = runner.run_in_environment('enformer', get_numpy_version)
        logger.info(f"NumPy version in enformer environment: {version}")
    except Exception as e:
        logger.error(f"Failed to run in environment: {e}")
    
    # Test 2: Check TensorFlow availability
    def check_tensorflow():
        try:
            import tensorflow as tf
            return {
                'available': True,
                'version': tf.__version__,
                'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0
            }
        except ImportError:
            return {'available': False}
    
    try:
        tf_info = runner.run_in_environment('enformer', check_tensorflow)
        logger.info(f"TensorFlow in enformer environment: {tf_info}")
    except Exception as e:
        logger.error(f"Failed to check TensorFlow: {e}")
    
    # Test 3: Run a script
    script = """
import sys
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

# List key packages
packages = ['numpy', 'pandas', 'tensorflow', 'tensorflow_hub']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f"{pkg}: {version}")
    except ImportError:
        print(f"{pkg}: not installed")
"""
    
    try:
        result = runner.run_script_in_environment('enformer', script, timeout=30)
        logger.info("Script output:")
        print(result.stdout)
    except Exception as e:
        logger.error(f"Failed to run script: {e}")
    
    # Test 4: Check environment health
    health = runner.check_environment_health('enformer')
    logger.info(f"\nEnvironment health check:")
    logger.info(f"  Exists: {health['environment_exists']}")
    logger.info(f"  Can import: {health['can_import']}")
    logger.info(f"  Dependencies OK: {health['dependencies_ok']}")
    if health['errors']:
        logger.error(f"  Errors: {health['errors']}")


def test_oracle_with_environment():
    """Test using an oracle with environment support."""
    logger.info("\n=== Testing Oracle with Environment Support ===")
    
    # Create oracle with environment support
    oracle = EnformerOracleEnv(use_environment=True)
    
    # Get oracle status
    status = oracle.get_status()
    logger.info(f"Oracle status: {status}")
    
    # Try to load model (this would actually download and load Enformer)
    # For demo purposes, we'll skip this as it requires significant resources
    logger.info("\nSkipping model loading for demo (would download ~1GB model)")
    
    # Demonstrate prediction workflow (without actually loading model)
    logger.info("\nDemonstrating prediction workflow:")
    
    # Example sequence
    test_seq = "ATCGATCGATCG" * 100  # Short test sequence
    
    # If model were loaded, we could do:
    # predictions = oracle.predict_region_replacement(
    #     genomic_region="chr1:1000-2000",
    #     seq=test_seq,
    #     assay_ids=["DNase", "ATAC-seq"],
    #     create_tracks=False
    # )
    
    logger.info("Would predict for assays: DNase, ATAC-seq")
    logger.info(f"Sequence length: {len(test_seq)}")
    logger.info(f"Using environment: {oracle.use_environment}")


def compare_environments():
    """Compare running with and without environment isolation."""
    logger.info("\n=== Comparing Environment Modes ===")
    
    # Oracle with environment
    oracle_env = EnformerOracleEnv(use_environment=True)
    logger.info(f"Oracle with environment: {oracle_env.get_status()}")
    
    # Oracle without environment
    oracle_direct = EnformerOracleEnv(use_environment=False)
    logger.info(f"Oracle without environment: {oracle_direct.get_status()}")
    
    # The key difference is that oracle_env runs all model operations
    # in an isolated conda environment, while oracle_direct uses the
    # current Python environment


def main():
    """Main demo function."""
    logger.info("Chorus Modular Environment System Demo")
    logger.info("=" * 50)
    
    # Step 1: Set up environments
    manager = setup_environments()
    
    # Step 2: Test environment runner
    if manager.environment_exists('enformer'):
        test_environment_runner(manager)
    else:
        logger.warning("Enformer environment not found. Skipping runner tests.")
    
    # Step 3: Test oracle with environment support
    test_oracle_with_environment()
    
    # Step 4: Compare environment modes
    compare_environments()
    
    logger.info("\n" + "=" * 50)
    logger.info("Demo complete!")
    logger.info("\nTo use the CLI:")
    logger.info("  chorus list              # List environments")
    logger.info("  chorus setup --oracle enformer  # Set up specific oracle")
    logger.info("  chorus setup             # Set up all oracles")
    logger.info("  chorus validate          # Validate environments")
    logger.info("  chorus health            # Check environment health")


if __name__ == "__main__":
    main()