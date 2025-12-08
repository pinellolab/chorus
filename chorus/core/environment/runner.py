"""Runner for executing code in oracle-specific environments."""

import os
import sys
import json
import pickle
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
import shlex
logger = logging.getLogger(__name__)


class EnvironmentRunner:
    """Executes code in oracle-specific conda environments."""
    
    def __init__(self, environment_manager):
        """
        Initialize the runner.
        
        Args:
            environment_manager: Instance of EnvironmentManager
        """
        self.env_manager = environment_manager
    
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
            running_command = shlex.split(f"mamba run -n {env_name} python {code_path}")

            env = os.environ.copy()
            
            # Force the loader to use the env's libstdc++

            env_prefix = self.env_manager.get_environment_info(oracle)['path']
            env_libstdcpp = f"{env_prefix}/lib/libstdc++.so.6"
            if os.path.exists(env_libstdcpp):
                env["LD_PRELOAD"] = env_libstdcpp
          
            env['PATH'].replace("chorus", env_name)
            if 'MPLBACKEND' in env:
                env.pop('MPLBACKEND') # remove matplotlib backend to avoid conflict with matplotlib inline backend

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