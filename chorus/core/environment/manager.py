"""Environment manager for handling oracle-specific conda environments."""

import os
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml
import sys

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Manages conda environments for different oracles."""
    
    def __init__(self, base_path: Optional[Path] = None, conda_exe: Optional[str] = None):
        """
        Initialize the environment manager.
        
        Args:
            base_path: Base path for environments directory
            conda_exe: Path to conda executable
        """
        if base_path is None:
            # Default to package root/environments
            module_path = Path(__file__).parent.parent.parent.parent
            base_path = module_path / "environments"
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Try to find conda executable
        self.conda_exe = conda_exe or self._find_conda_executable()
        if not self.conda_exe:
            raise RuntimeError("Could not find conda executable. Please install conda or mamba.")
        
        # Cache for environment status
        self._env_cache = {}
        
    def _find_conda_executable(self) -> Optional[str]:
        """Find conda or mamba executable."""
        # First check CONDA_EXE environment variable (set by conda when activated)
        conda_exe = os.environ.get('CONDA_EXE')
        if conda_exe and os.path.exists(conda_exe):
            logger.info(f"Found conda via CONDA_EXE: {conda_exe}")
            # Check if mamba exists in the same directory
            conda_dir = os.path.dirname(conda_exe)
            mamba_exe = os.path.join(conda_dir, 'mamba')
            if os.path.exists(mamba_exe):
                logger.info(f"Found mamba at: {mamba_exe}")
                return mamba_exe
            return conda_exe
        
        # Check MAMBA_EXE environment variable
        mamba_exe = os.environ.get('MAMBA_EXE')
        if mamba_exe and os.path.exists(mamba_exe):
            logger.info(f"Found mamba via MAMBA_EXE: {mamba_exe}")
            return mamba_exe
        
        # Try using the base conda installation path
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            # Go up to the base conda directory
            # If we're in an env, CONDA_PREFIX points to the env, not base
            # Check for CONDA_PREFIX_1 which points to base when in an env
            base_prefix = os.environ.get('CONDA_PREFIX_1', conda_prefix)
            for cmd in ['mamba', 'conda']:
                exe_path = os.path.join(base_prefix, 'bin', cmd)
                if os.path.exists(exe_path):
                    logger.info(f"Found {cmd} via CONDA_PREFIX at: {exe_path}")
                    return exe_path
        
        # Try common conda/mamba commands (may work via shell functions)
        for cmd in ['mamba', 'conda', 'micromamba']:
            try:
                result = subprocess.run(
                    [cmd, '--version'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                if result.returncode == 0:
                    logger.info(f"Found {cmd}: {result.stdout.strip()}")
                    return cmd
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        # Try common installation paths
        common_paths = [
            os.path.expanduser("~/miniforge3/bin/conda"),
            os.path.expanduser("~/miniconda3/bin/conda"),
            os.path.expanduser("~/anaconda3/bin/conda"),
            "/opt/conda/bin/conda",
            "/usr/local/bin/conda",
            # Add path specific to your setup
            "/data/pinello/SHARED_SOFTWARE/miniforge3/bin/mamba",
            "/data/pinello/SHARED_SOFTWARE/miniforge3/bin/conda"
        ]
        
        # Also check for mamba in these locations
        for base_path in common_paths:
            base_dir = os.path.dirname(base_path)
            mamba_path = os.path.join(base_dir, 'mamba')
            if os.path.exists(mamba_path):
                return mamba_path
            elif os.path.exists(base_path):
                return base_path
        
        return None
    
    def list_available_oracles(self) -> List[str]:
        """List all oracles with environment definitions."""
        oracle_envs = []
        env_files = self.base_path.glob("*.yml")
        
        for env_file in env_files:
            if env_file.stem.startswith("chorus-"):
                oracle_name = env_file.stem.replace("chorus-", "")
                oracle_envs.append(oracle_name)
        
        return sorted(oracle_envs)
    
    def get_environment_name(self, oracle: str) -> str:
        """Get the conda environment name for an oracle."""
        return f"chorus-{oracle}"
    
    def get_environment_file(self, oracle: str) -> Path:
        """Get the environment file path for an oracle."""
        return self.base_path / f"chorus-{oracle}.yml"
    
    def environment_exists(self, oracle: str) -> bool:
        """Check if an environment exists for the given oracle."""
        env_name = self.get_environment_name(oracle)
        
        # Check cache first
        if env_name in self._env_cache:
            return self._env_cache[env_name]
        
        # Check with conda
        try:
            result = subprocess.run(
                [self.conda_exe, 'env', 'list', '--json'],
                capture_output=True,
                text=True,
                check=True
            )
            
            env_data = json.loads(result.stdout)
            env_names = [os.path.basename(env) for env in env_data.get('envs', [])]
            exists = env_name in env_names
            
            # Update cache
            self._env_cache[env_name] = exists
            return exists
            
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"Error checking environment existence: {e}")
            return False
    
    def create_environment(self, oracle: str, force: bool = False) -> bool:
        """
        Create a conda environment for the given oracle.
        
        Args:
            oracle: Name of the oracle
            force: If True, recreate environment even if it exists
            
        Returns:
            True if successful, False otherwise
        """
        env_name = self.get_environment_name(oracle)
        env_file = self.get_environment_file(oracle)
        
        if not env_file.exists():
            logger.error(f"Environment file not found: {env_file}")
            return False
        
        # Check if environment already exists
        if self.environment_exists(oracle) and not force:
            logger.info(f"Environment {env_name} already exists")
            return True
        
        # Remove existing environment if force is True
        if force and self.environment_exists(oracle):
            logger.info(f"Removing existing environment {env_name}")
            self.remove_environment(oracle)
        
        # Create environment
        logger.info(f"Creating environment {env_name} from {env_file}")
        try:
            cmd = [self.conda_exe, 'env', 'create', '-f', str(env_file)]
            
            # Use mamba if available for faster solving
            if 'mamba' in self.conda_exe:
                cmd = [self.conda_exe, 'env', 'create', '-f', str(env_file)]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully created environment {env_name}")
                self._env_cache[env_name] = True
                return True
            else:
                logger.error(f"Failed to create environment: {result.stderr}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating environment: {e}")
            return False
    
    def remove_environment(self, oracle: str) -> bool:
        """Remove a conda environment for the given oracle."""
        env_name = self.get_environment_name(oracle)
        
        if not self.environment_exists(oracle):
            logger.info(f"Environment {env_name} does not exist")
            return True
        
        try:
            result = subprocess.run(
                [self.conda_exe, 'env', 'remove', '-n', env_name, '-y'],
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Successfully removed environment {env_name}")
            self._env_cache[env_name] = False
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error removing environment: {e}")
            return False
    
    def update_environment(self, oracle: str) -> bool:
        """Update an existing conda environment."""
        env_name = self.get_environment_name(oracle)
        env_file = self.get_environment_file(oracle)
        
        if not self.environment_exists(oracle):
            logger.info(f"Environment {env_name} does not exist, creating it")
            return self.create_environment(oracle)
        
        if not env_file.exists():
            logger.error(f"Environment file not found: {env_file}")
            return False
        
        try:
            cmd = [self.conda_exe, 'env', 'update', '-n', env_name, '-f', str(env_file)]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully updated environment {env_name}")
                return True
            else:
                logger.error(f"Failed to update environment: {result.stderr}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error updating environment: {e}")
            return False
    
    def get_environment_info(self, oracle: str) -> Optional[Dict]:
        """Get detailed information about an environment."""
        env_name = self.get_environment_name(oracle)
        
        if not self.environment_exists(oracle):
            return None
        
        try:
            # Get environment location
            result = subprocess.run(
                [self.conda_exe, 'env', 'list', '--json'],
                capture_output=True,
                text=True,
                check=True
            )
            
            env_data = json.loads(result.stdout)
            env_path = None
            
            for env in env_data.get('envs', []):
                if os.path.basename(env) == env_name:
                    env_path = env
                    break
            
            if not env_path:
                return None
            
            # Get package list
            result = subprocess.run(
                [self.conda_exe, 'list', '-n', env_name, '--json'],
                capture_output=True,
                text=True,
                check=True
            )
            
            packages = json.loads(result.stdout)
            
            # Parse environment file
            env_file = self.get_environment_file(oracle)
            env_config = {}
            if env_file.exists():
                with open(env_file, 'r') as f:
                    env_config = yaml.safe_load(f)
            
            return {
                'name': env_name,
                'path': env_path,
                'oracle': oracle,
                'packages': packages,
                'config': env_config,
                'exists': True
            }
            
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"Error getting environment info: {e}")
            return None
    
    def get_python_executable(self, oracle: str) -> Optional[str]:
        """Get the path to the Python executable for an oracle's environment."""
        env_info = self.get_environment_info(oracle)
        if not env_info or not env_info.get('path'):
            return None
        
        # Construct Python path
        env_path = env_info['path']
        python_path = os.path.join(env_path, 'bin', 'python')
        
        if os.path.exists(python_path):
            return python_path
        
        # Try Windows path
        python_path = os.path.join(env_path, 'Scripts', 'python.exe')
        if os.path.exists(python_path):
            return python_path
        
        return None
    
    def validate_environment(self, oracle: str) -> Tuple[bool, List[str]]:
        """
        Validate that an environment has all required dependencies.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        env_file = self.get_environment_file(oracle)
        issues = []
        
        if not env_file.exists():
            issues.append(f"Environment file not found: {env_file}")
            return False, issues
        
        if not self.environment_exists(oracle):
            issues.append(f"Environment not created for {oracle}")
            return False, issues
        
        # Check Python executable
        python_exe = self.get_python_executable(oracle)
        if not python_exe:
            issues.append("Python executable not found in environment")
            return False, issues
        
        # Check oracle-specific dependencies instead of importing chorus
        oracle_deps = {
            'enformer': ['tensorflow', 'tensorflow_hub'],
            'sei': ['torch'],
            'borzoi': ['tensorflow'],
            'chrombpnet': ['tensorflow']
        }
        
        if oracle in oracle_deps:
            for dep in oracle_deps[oracle]:
                try:
                    result = subprocess.run(
                        [python_exe, '-c', f'import {dep}'],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode != 0:
                        issues.append(f"Missing dependency {dep}: {result.stderr}")
                except subprocess.TimeoutExpired:
                    issues.append(f"Timeout while checking dependency {dep}")
                except Exception as e:
                    issues.append(f"Error checking dependency {dep}: {str(e)}")
        
        return len(issues) == 0, issues
    
    def setup_all_environments(self, force: bool = False) -> Dict[str, bool]:
        """
        Set up all available oracle environments.
        
        Args:
            force: If True, recreate all environments
            
        Returns:
            Dictionary mapping oracle names to success status
        """
        results = {}
        oracles = self.list_available_oracles()
        
        for oracle in oracles:
            logger.info(f"Setting up environment for {oracle}")
            success = self.create_environment(oracle, force=force)
            results[oracle] = success
        
        return results