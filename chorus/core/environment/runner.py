"""Runner for executing code in oracle-specific environments."""

import os
import shutil
import sys
import json
import pickle
import tempfile
import subprocess
import logging
from typing import Any, Dict, Optional, Callable
import shlex

logger = logging.getLogger(__name__)


def _find_mamba() -> str:
    """Locate the mamba executable, falling back to conda."""
    exe = shutil.which("mamba")
    if exe:
        return exe
    # Common install locations
    for candidate in [
        "/data/pinello/SHARED_SOFTWARE/miniforge3/bin/mamba",
        os.path.expanduser("~/miniforge3/bin/mamba"),
        os.path.expanduser("~/mambaforge/bin/mamba"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    # Fall back to conda
    exe = shutil.which("conda")
    if exe:
        return exe
    raise FileNotFoundError(
        "Neither mamba nor conda found on PATH. "
        "Set PATH to include your miniforge3/bin directory."
    )


ORACLE_CLASS_MAP = {
    "chrombpnet": "ChromBPNetOracle",
    "borzoi": "BorzoiOracle",
    "enformer": "EnformerOracle",
    "sei": "SeiOracle",
    "legnet": "LegNetOracle",
    "alphagenome": "AlphaGenomeOracle",
}


class EnvironmentRunner:
    """Executes code in oracle-specific conda environments."""

    def __init__(self, environment_manager):
        """
        Initialize the runner.

        Args:
            environment_manager: Instance of EnvironmentManager
        """
        self.env_manager = environment_manager

    @staticmethod
    def _detect_python_version(env_prefix: str) -> str:
        """Detect the Python major.minor version in a conda env."""
        lib_dir = os.path.join(env_prefix, "lib")
        if os.path.isdir(lib_dir):
            # Sort descending by length so "python3.10" is preferred over
            # "python3.1" (both may exist as directories).
            candidates = sorted(
                (name for name in os.listdir(lib_dir)
                 if name.startswith("python3.") and os.path.isdir(os.path.join(lib_dir, name))),
                key=len, reverse=True,
            )
            if candidates:
                return candidates[0].replace("python", "")  # e.g. "3.10"
        return "3"  # fallback

    def _prepare_env(self, oracle: str) -> dict:
        """Build an env dict with LD_PRELOAD and LD_LIBRARY_PATH for an oracle.

        Handles two common Linux issues:
        1. libstdc++ version mismatch (LD_PRELOAD the env's copy).
        2. TensorFlow <2.15 not auto-discovering nvidia-* pip packages
           (add their lib dirs to LD_LIBRARY_PATH).
        """
        env = os.environ.copy()
        env_info = self.env_manager.get_environment_info(oracle)
        if not env_info:
            return env
        env_prefix = env_info["path"]

        # 1. LD_PRELOAD for libstdc++
        env_libstdcpp = os.path.join(env_prefix, "lib", "libstdc++.so.6")
        if os.path.exists(env_libstdcpp):
            env["LD_PRELOAD"] = env_libstdcpp

        # 2. LD_LIBRARY_PATH for nvidia pip packages (TF <2.15 CUDA discovery)
        pyver = self._detect_python_version(env_prefix)
        nvidia_base = os.path.join(
            env_prefix, "lib", f"python{pyver}", "site-packages", "nvidia",
        )
        if os.path.isdir(nvidia_base):
            nvidia_lib_dirs = [
                os.path.join(nvidia_base, pkg, "lib")
                for pkg in os.listdir(nvidia_base)
                if os.path.isdir(os.path.join(nvidia_base, pkg, "lib"))
            ]
            if nvidia_lib_dirs:
                existing = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = ":".join(nvidia_lib_dirs) + (
                    ":" + existing if existing else ""
                )

        # 3. Remove MPLBACKEND to avoid matplotlib conflicts
        env.pop("MPLBACKEND", None)

        return env
    
    def run_code_in_environment(
        self,
        oracle: str,
        code: str,
        timeout: Optional[int] = None
    ) -> Any:
        """
        Run Python code in the oracle's conda environment and return the result.
        
        The code should set a variable named 'result' that will be returned.
        
        Args:
            oracle: Name of the oracle
            code: Python code to execute
            timeout: Timeout in seconds
            
        Returns:
            The value of the 'result' variable from the executed code
        """
        # Check if environment exists
        if not self.env_manager.environment_exists(oracle):
            raise RuntimeError(f"Environment for {oracle} does not exist. Run setup first.")
        
        # Get Python executable
        python_exe = self.env_manager.get_python_executable(oracle)
        if not python_exe:
            raise RuntimeError(f"Could not find Python executable for {oracle}")
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as code_file:
            code_path = code_file.name
            
            # Get the chorus package path
            import chorus
            chorus_path = os.path.dirname(os.path.dirname(chorus.__file__))
            
            # Write the wrapped code to file
            # Properly indent the user code
            indented_code = '\n'.join('    ' + line for line in code.split('\n'))
            
            wrapped_code = f"""import sys
import pickle
import traceback

# Add chorus to Python path
sys.path.insert(0, {repr(chorus_path)})

# Initialize result
result = None
error = None

try:
    # Execute user code
{indented_code}
    
    # Save result
    with open({repr(code_path + '.pkl')}, 'wb') as f:
        pickle.dump({{'success': True, 'result': result, 'error': None}}, f)
except Exception as e:
    # Save error
    with open({repr(code_path + '.pkl')}, 'wb') as f:
        pickle.dump({{'success': False, 'result': None, 'error': str(e), 'traceback': traceback.format_exc()}}, f)
"""
            code_file.write(wrapped_code)
        
        output_path = code_path + '.pkl'
        
        try:
            # Run the script file instead of passing code as argument
            env_name = self.env_manager.get_environment_name(oracle)
            mamba_exe = getattr(self.env_manager, 'conda_exe', None) or _find_mamba()
            running_command = shlex.split(f"{mamba_exe} run -n {env_name} python {code_path}")

            env = self._prepare_env(oracle)
                
            result = subprocess.run(
                running_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Execution failed: {result.stderr}")
            
            # Load result
            if os.path.exists(output_path):
                with open(output_path, 'rb') as f:
                    output_data = pickle.load(f)
                
                if output_data['success']:
                    return output_data['result']
                else:
                    raise RuntimeError(f"Code execution failed: {output_data['error']}\n{output_data.get('traceback', '')}")
            else:
                raise RuntimeError(f"No output file created. Stdout: {result.stdout}, Stderr: {result.stderr}")
                
        finally:
            # Clean up
            for path in [code_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def run_in_environment(
        self,
        oracle: str,
        function: Callable,
        args: tuple = (),
        kwargs: dict = None,
        timeout: Optional[int] = None
    ) -> Any:
        """
        Run a function in the oracle's conda environment.
        
        Args:
            oracle: Name of the oracle
            function: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            timeout: Timeout in seconds
            
        Returns:
            Result of the function execution
        """
        if kwargs is None:
            kwargs = {}
        
        # Check if environment exists
        if not self.env_manager.environment_exists(oracle):
            raise RuntimeError(f"Environment for {oracle} does not exist. Run setup first.")
        
        # Get Python executable
        python_exe = self.env_manager.get_python_executable(oracle)
        if not python_exe:
            raise RuntimeError(f"Could not find Python executable for {oracle}")
        
        # Create temporary files for communication
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as input_file:
            input_path = input_file.name
            # Serialize function and arguments
            pickle.dump({
                'function': function,
                'args': args,
                'kwargs': kwargs
            }, input_file)
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            # Create execution script
            script = self._create_execution_script(input_path, output_path)
            
            # Run in environment
            result = subprocess.run(
                [python_exe, '-c', script],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Execution failed: {result.stderr}")
            
            # Load result
            with open(output_path, 'rb') as f:
                output_data = pickle.load(f)
            
            if output_data['success']:
                return output_data['result']
            else:
                raise RuntimeError(f"Function execution failed: {output_data['error']}")
                
        finally:
            # Clean up temporary files
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def run_script_in_environment(
        self,
        oracle: str,
        script: str,
        timeout: Optional[int] = None,
        capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Run a Python script in the oracle's conda environment.
        
        Args:
            oracle: Name of the oracle
            script: Python script to execute
            timeout: Timeout in seconds
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            CompletedProcess instance
        """
        # Check if environment exists
        if not self.env_manager.environment_exists(oracle):
            raise RuntimeError(f"Environment for {oracle} does not exist. Run setup first.")
        
        # Get Python executable
        python_exe = self.env_manager.get_python_executable(oracle)
        if not python_exe:
            raise RuntimeError(f"Could not find Python executable for {oracle}")
        
        env = self._prepare_env(oracle)

        # Run script
        return subprocess.run(
            [python_exe, '-c', script],
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            env=env,
        )
    
    def import_oracle_in_environment(
        self,
        oracle: str,
        timeout: Optional[int] = 120
    ) -> Dict[str, Any]:
        """
        Import an oracle module in its environment and get metadata.

        Args:
            oracle: Name of the oracle
            timeout: Timeout in seconds (default: 120, NFS can be slow)
            
        Returns:
            Dictionary with oracle metadata
        """
        class_name = ORACLE_CLASS_MAP.get(oracle.lower(), f"{oracle.capitalize()}Oracle")
        script = f"""
import json
import sys
import chorus
try:    
    oracle_instance = chorus.create_oracle('{oracle}', use_environment=True)
    # Get metadata
    metadata = {{
        'class_name': oracle_instance.__class__.__name__,
        'assay_types': oracle_instance.list_assay_types(),
        'cell_types': oracle_instance.list_cell_types(),
        'has_model': hasattr(oracle_instance, 'model'),
        'is_loaded': getattr(oracle_instance, 'loaded', False)
    }}
    
    print(json.dumps({{'success': True, 'metadata': metadata}}))
    
except Exception as e:
    print(json.dumps({{'success': False, 'error': str(e)}}))
"""
        
        result = self.run_script_in_environment(oracle, script, timeout=timeout)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to import oracle: {result.stderr}")
        
        try:
            output = json.loads(result.stdout)
            if output['success']:
                return output['metadata']
            else:
                raise RuntimeError(f"Import failed: {output['error']}")
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid output: {result.stdout}")
    
    def _create_execution_script(self, input_path: str, output_path: str) -> str:
        """Create a script for executing a function in a subprocess."""
        # Get the chorus package path
        import chorus
        chorus_path = os.path.dirname(os.path.dirname(chorus.__file__))
        
        return f"""
import pickle
import sys
import traceback
import os

# Add chorus to Python path so isolated environment can access it
sys.path.insert(0, {repr(chorus_path)})

# Load input
with open('{input_path}', 'rb') as f:
    data = pickle.load(f)

function = data['function']
args = data['args']
kwargs = data['kwargs']

# Execute function
try:
    result = function(*args, **kwargs)
    output = {{'success': True, 'result': result}}
except Exception as e:
    output = {{
        'success': False,
        'error': str(e),
        'traceback': traceback.format_exc()
    }}

# Save output
with open('{output_path}', 'wb') as f:
    pickle.dump(output, f)
"""
    
    def run_oracle_method(
        self,
        oracle: str,
        method_name: str,
        args: tuple = (),
        kwargs: dict = None,
        timeout: Optional[int] = None
    ) -> Any:
        """
        Run a specific oracle method in its environment.
        
        Args:
            oracle: Name of the oracle
            method_name: Name of the method to call
            args: Positional arguments
            kwargs: Keyword arguments
            timeout: Timeout in seconds
            
        Returns:
            Result of the method call
        """
        if kwargs is None:
            kwargs = {}
        
        # Create a wrapper function that imports and calls the oracle method
        def wrapper(*args, **kwargs):
            import importlib
            
            # Import oracle module
            module_name = f"chorus.oracles.{oracle}"
            module = importlib.import_module(module_name)
            
            # Get oracle class
            oracle_class = getattr(module, f"{oracle.capitalize()}Oracle")
            
            # Create instance
            instance = oracle_class()
            
            # Get method
            method = getattr(instance, method_name)
            
            # Call method
            return method(*args, **kwargs)
        
        return self.run_in_environment(oracle, wrapper, args, kwargs, timeout)
    
    def check_environment_health(self, oracle: str, timeout: int) -> Dict[str, Any]:
        """
        Check the health of an oracle's environment.
        
        Args:
            oracle: Name of the oracle
            
        Returns:
            Dictionary with health check results
        """
        health = {
            'oracle': oracle,
            'environment_exists': self.env_manager.environment_exists(oracle),
            'python_executable': None,
            'can_import': False,
            'dependencies_ok': False,
            'errors': []
        }
        
        if not health['environment_exists']:
            health['errors'].append("Environment does not exist")
            return health
        
        # Check Python executable
        python_exe = self.env_manager.get_python_executable(oracle)
        if python_exe:
            health['python_executable'] = python_exe
        else:
            health['errors'].append("Python executable not found")
            return health
        
        # Try to import oracle
        try:
            metadata = self.import_oracle_in_environment(oracle, timeout=timeout)  # Increased for slower systems
            health['can_import'] = True
            health['metadata'] = metadata
        except Exception as e:
            health['errors'].append(f"Cannot import oracle: {str(e)}")
        
        # Check key dependencies
        deps_script = f"""
import json
import importlib

dependencies = {{
    'enformer': ['tensorflow', 'tensorflow_hub'],
    'borzoi': ['torch'],
    'sei': ['torch'],
    'chrombpnet': ['tensorflow'],
    'legnet': ['torch'],
    'alphagenome': ['jax']
}}

oracle_deps = dependencies.get('{oracle}', [])
missing = []

for dep in oracle_deps:
    try:
        importlib.import_module(dep)
    except ImportError:
        missing.append(dep)

print(json.dumps({{'missing': missing}}))
"""
        
        try:
            result = self.run_script_in_environment(oracle, deps_script, timeout=120)  # TensorFlow import can take >30s
            if result.returncode == 0:
                deps_data = json.loads(result.stdout)
                if not deps_data['missing']:
                    health['dependencies_ok'] = True
                else:
                    health['errors'].append(f"Missing dependencies: {deps_data['missing']}")
        except Exception as e:
            health['errors'].append(f"Error checking dependencies: {str(e)}")
        
        return health