"""Borzoi oracle implementation"""

from ..core.base import OracleBase
from ..core.track import Track
from ..core.exceptions import ModelNotLoadedError, InvalidSequenceError
from typing import List, Tuple, Union, Optional, Dict
import numpy as np
import os
import logging
import tempfile
import subprocess
import h5py
import json
import shutil
import pandas as pd
import time
from typing import List, Tuple, Union, Optional, Dict, Any

logger = logging.getLogger(__name__)


class BorzoiOracle(OracleBase):
    
    def __init__(self, use_environment: bool = True, reference_fasta: Optional[str] = None,
                model_path: Optional[str] = None, params_path: Optional[str] = None, 
                fold: int = 0):
        """
        Initialize Borzoi oracle.
        
        Args:
            use_environment: Whether to use isolated conda environment
            reference_fasta: Path to reference FASTA file (required for predictions)
            model_path: Path to Borzoi model (.h5 file)
            params_path: Path to Borzoi parameters file (params.json)
            fold: Which model fold to use (0-3) if using default paths
        """
        self.oracle_name = 'borzoi'
        super().__init__(use_environment=use_environment)
        
        # Borzoi parameters from the real implementation
        self.sequence_length = 524288  # 524kb input
        self.target_length = 6144      # Will be updated from actual model
        self.bin_size = 32             # 32bp bins
        
        self.fold = fold
        self.reference_fasta = reference_fasta
        
        borzoi_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'oracles', 'borzoi_material')
        os.environ['BORZOI_DIR'] = borzoi_dir
        os.makedirs(borzoi_dir, exist_ok=True)
        if model_path is None:
            self.model_path = os.path.join(borzoi_dir, f"saved_models/f{fold}c0/train/model0_best.h5")
        else:
            self.model_path = model_path
            
        if params_path is None:
            self.params_path = os.path.join(borzoi_dir, "params.json")
        else:
            self.params_path = params_path
        
        # Add targets_path from metadata
        from .borzoi_metadata import get_metadata
        metadata = get_metadata()
        self.targets_path = metadata.metadata_file
        # Print some example track IDs (first 10)
        print("\nExample track IDs:")
        for i, track_id in enumerate(list(metadata._track_index_map.keys())[:10]):
            print(f"  {i+1}. {track_id}")

        
        borzoi_dir = os.environ.get('BORZOI_DIR', './borzoi_material')
        self.predict_script = os.path.join(borzoi_dir, "borzoi_predict.py")

        # Load targets and parameters
        self._load_model_params()

    def _load_model_params(self):
        """Load model parameters to get actual dimensions."""
        if os.path.exists(self.params_path):
            try:
                with open(self.params_path, 'r') as f:
                    params = json.load(f)
                
                model_params = params.get('model', {})
                self.sequence_length = model_params.get('seq_length', self.sequence_length)
            except Exception as e:
                pass

    def _download_file_if_needed(self, url: str, local_path: str, description: str) -> bool:
        """Download a file if it doesn't exist locally."""
        if os.path.exists(local_path):
            logger.info(f"{description} found at {local_path}")
            return True
            
        # Create directory structure if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        logger.info(f"Downloading {description} from {url}...")
        try:
            import urllib.request
            # Use a temporary file during download
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
                
            # Download to temporary file
            urllib.request.urlretrieve(url, temp_path)
            
            # Move to final location
            shutil.move(temp_path, local_path)
            
            logger.info(f"Successfully downloaded {description} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {description}: {str(e)}")
            
            # Clean up temp file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
                
            return False

    def _download_borzoi_files_if_needed(self) -> bool:
        """Download required Borzoi files if they don't exist."""
        downloads_needed = []
        
        # Check model file
        if not os.path.exists(self.model_path):
            # Create the directory structure for the model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            model_url = f"https://storage.googleapis.com/seqnn-share/borzoi/f{self.fold}/model0_best.h5"
            downloads_needed.append((model_url, self.model_path, f"Borzoi model (fold {self.fold})"))

        # Check params file (try downloading example params)
        if not os.path.exists(self.params_path):
            # Create the directory for params file
            os.makedirs(os.path.dirname(self.params_path), exist_ok=True)
            params_url = "https://github.com/calico/borzoi/raw/main/examples/params.json"
            downloads_needed.append((params_url, self.params_path, "Borzoi parameters"))

        # Download all needed files
        success = True
        for url, path, description in downloads_needed:
            if not self._download_file_if_needed(url, path, description):
                success = False
        
        return success

    def load_pretrained_model(self, weights: str = None) -> None:
        """Load Borzoi model and download required files if needed."""
        
        if weights is not None:
            self.model_path = weights
        
        # Try to download missing files first
        download_success = self._download_borzoi_files_if_needed()
        
        # Also try to download the targets file if needed
        from .borzoi_metadata import get_metadata
        metadata = get_metadata()
        
        # Check what files we have after attempting downloads
        required_files = {
            'model': self.model_path,
            'params': self.params_path,
            'predict_script': self.predict_script,
            'targets': self.targets_path
        }
        
        missing = []
        for name, path in required_files.items():
            if not os.path.exists(path):
                missing.append(f"{name}: {path}")
        
        if missing:
            error_msg = f"Missing required files: {', '.join(missing)}\n"
            error_msg += "To download manually:\n"
            error_msg += f"1. Model: https://storage.googleapis.com/seqnn-share/borzoi/f{self.fold}/model0_best.h5\n"
            error_msg += "2. Params: https://github.com/calico/borzoi/raw/main/examples/params.json\n"
            error_msg += "3. Targets: https://github.com/calico/borzoi/raw/main/examples/targets_human.txt"
            raise ModelNotLoadedError(error_msg)
        
        if self.reference_fasta and not os.path.exists(self.reference_fasta):
            raise ModelNotLoadedError(f"Reference FASTA not found: {self.reference_fasta}")


    def list_assay_types(self) -> List[str]:
        """Return all unique assay types from Borzoi metadata."""
        from .borzoi_metadata import get_metadata
        metadata = get_metadata()
        return metadata.list_assay_types()

    def list_cell_types(self) -> List[str]:
        """Return all unique cell types from Borzoi metadata."""
        from .borzoi_metadata import get_metadata
        metadata = get_metadata()
        return metadata.list_cell_types()
    
    def _predict(self, input_data: Union[str, Tuple[str, int, int]], assay_ids: List[str]) -> np.ndarray:
        """Internal prediction method used by the base class's predict method."""
        # Handle genomic coordinates (chrom, start, end)
        if isinstance(input_data, tuple) and len(input_data) == 3:
            chrom, start, end = input_data
            
            # Check if reference genome is available
            if not self.reference_fasta:
                raise ValueError("Reference genome FASTA required for genomic coordinates prediction")
            
            # Extract sequence from reference genome
            from ..utils.sequence import extract_sequence
            sequence = extract_sequence(f"{chrom}:{start}-{end}", self.reference_fasta)
            
            # Make prediction on the extracted sequence
            return self._predict_sequence(sequence, assay_ids)
        
        # Otherwise, it should be a sequence string
        return self._predict_sequence(input_data, assay_ids)

    def _predict_sequence(self, sequence: str, assay_ids: List[str]) -> np.ndarray:
        """Predict for raw sequence using borzoi_predict.py"""
        # Create a temporary FASTA file with the sequence
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as fasta_file:
            fasta_file.write(f">sequence\n{sequence}\n")
            fasta_path = fasta_file.name
        
        output_dir = tempfile.mkdtemp()
        
        try:
            # Build the command to run borzoi_predict.py
            cmd = [
                'python', self.predict_script,
                '-o', output_dir,
                '--rc',
                '-t', self.targets_path,
                self.params_path,
                self.model_path,
                fasta_path
            ]
                        
            if self.use_environment:
                result = self._run_predict_in_environment(cmd, output_dir, assay_ids)
            else:
                result = self._run_predict_direct(cmd, output_dir, assay_ids)
            
            return result
            
        finally:
            # Clean up
            if os.path.exists(fasta_path):
                try:
                    os.unlink(fasta_path)
                except:
                    pass
            if os.path.exists(output_dir):
                try:
                    shutil.rmtree(output_dir)
                except:
                    pass
    

    def _run_predict_direct(self, cmd: List[str], output_dir: str, assay_ids: List[str]) -> np.ndarray:
        """Run borzoi_predict.py directly."""
        try:            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
                        
            if result.returncode != 0:
                raise RuntimeError(f"borzoi_predict.py failed: {result.stderr}")
            
            # Parse the output
            return self._parse_predict_output(output_dir, assay_ids)
        
        except Exception as e:
            raise

    def _run_predict_in_environment(self, cmd: List[str], output_dir: str, assay_ids: List[str]) -> np.ndarray:
        """Run borzoi_predict.py in conda environment."""
        env_code = f"""
    import subprocess
    import os

    # Run borzoi_predict.py 
    result = subprocess.run({repr(cmd)}, capture_output=True, text=True, timeout=600, env=env)

    {{
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'output_dir': {repr(output_dir)}
    }}
    """
        
        try:
            result_info = self.run_code_in_environment(env_code, timeout=700)
            
            return self._parse_predict_output(output_dir, assay_ids)
            
        except Exception as e:
            raise


    def _parse_predict_output(self, output_dir: str, assay_ids: List[str]) -> np.ndarray:
        """Parse predictions.h5 output file."""
        pred_file = os.path.join(output_dir, 'predictions.h5')
                
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Predictions output file not found: {pred_file}")
        
        try:
            with h5py.File(pred_file, 'r') as f:
                
                # Get target indices
                if 'target_ids' in f:
                    target_ids = [tid.decode() if isinstance(tid, bytes) else str(tid) 
                                for tid in f['target_ids'][:]]
                    track_indices = self._map_assay_ids_to_indices(assay_ids, target_ids)
                else:
                    track_indices = [0]  # Default to first track
                
                # Get predictions
                if 'preds' in f:
                    preds_array = f['preds'][:]  # Load the entire array
                    
                    # Handle different prediction formats based on dimensions
                    if len(preds_array.shape) == 3:  # Shape: (batch, positions, targets)
                        predictions = np.zeros((preds_array.shape[1], len(assay_ids)))
                        
                        for i, track_idx in enumerate(track_indices):
                            if track_idx < preds_array.shape[2]:
                                predictions[:, i] = preds_array[0, :, track_idx]
                                track_id = target_ids[track_idx] if track_idx < len(target_ids) else f"track_{track_idx}"
                    
                    elif len(preds_array.shape) == 2:  # Shape: (positions, targets)
                        predictions = np.zeros((preds_array.shape[0], len(assay_ids)))
                        
                        for i, track_idx in enumerate(track_indices):
                            if track_idx < preds_array.shape[1]:
                                predictions[:, i] = preds_array[:, track_idx]
                                track_id = target_ids[track_idx] if track_idx < len(target_ids) else f"track_{track_idx}"
                    return predictions
                
                return np.zeros((1, len(assay_ids)))
                
        except Exception as e:
            #print(f"DEBUG: Error parsing predictions output: {str(e)}")
            import traceback
            #print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return np.zeros((1, len(assay_ids)))


    def _map_assay_ids_to_indices(self, assay_ids: List[str], target_ids: List[str] = None) -> List[int]:
        """Map requested assay IDs to target indices using metadata."""
        from .borzoi_metadata import get_metadata
        
        metadata = get_metadata()
        indices = []
        
        for assay_id in assay_ids:
            # Check if it's a direct identifier
            idx = metadata.get_track_by_identifier(assay_id)
            if idx is not None:
                indices.append(idx)
                continue
            
            # Search by description
            matches = metadata.get_tracks_by_description(assay_id)
            if matches:
                # Use the first match
                indices.append(matches[0][0])  # First match's index
                continue
            
            # If no match, use index 0
            logger.warning(f"No matching track found for '{assay_id}', using index 0")
            indices.append(0)
        
        return indices
    def fine_tune(self, tracks: List[Track], track_names: List[str], **kwargs) -> None:
        """Fine-tuning not implemented for Borzoi."""
        raise NotImplementedError("Borzoi fine-tuning not implemented")

    def _get_context_size(self) -> int:
        """Return sequence context size."""
        return self.sequence_length

    def _get_sequence_length_bounds(self) -> Tuple[int, int]:
        """Return sequence length bounds."""
        return (1000, self.sequence_length)

    def _get_bin_size(self) -> int:
        """Return the bin size for predictions."""
        return self.bin_size
    
    def get_status(self) -> Dict[str, Any]:
        """Get oracle status including environment info."""
        status = {
            'name': self.__class__.__name__,
            'loaded': self.loaded,
            'use_environment': self.use_environment,
            'environment_info': None
        }
        
        if self.use_environment:
            status['environment_info'] = self.get_environment_info()
        
        return status
    
    def get_track_info(self, query: str = None) -> Union[pd.DataFrame, Dict[str, int]]:
        """Get information about available tracks.
        
        Args:
            query: Optional search query. If None, returns summary by assay type.
        
        Returns:
            If query is provided: DataFrame of matching tracks
            If no query: Dictionary with counts by assay type
        """
        from .enformer_metadata import get_metadata
        metadata = get_metadata()
        
        if query:
            return metadata.search_tracks(query)
        else:
            return metadata.get_track_summary()
    
    def get_output_window_coords(self, region_center: int) -> Tuple[int, int]:
        """Calculate Enformer's output window coordinates for a given region center.
        
        Enformer has a specific architecture:
        - Input: 393,216 bp
        - Output: 114,688 bp (896 bins × 128 bp)
        - The output is centered within the input with 139,264 bp offset on each side
        
        Args:
            region_center: Genomic coordinate of the region center
            
        Returns:
            Tuple of (output_start, output_end) genomic coordinates
        """
        output_size = self.target_length * self.bin_size
        output_offset = (self.sequence_length - output_size) // 2
        
        # Calculate input window
        input_start = region_center - self.sequence_length // 2
        
        # Calculate output window
        output_start = input_start + output_offset
        output_end = output_start + output_size
        
        return output_start, output_end
    
    def map_predictions_to_coords(self, predictions: np.ndarray, 
                                chrom: str, start: int, end: int) -> List[Dict[str, Any]]:
        """Map prediction values to genomic coordinates.
        
        This handles the complex coordinate mapping needed for Enformer's architecture,
        where the output window is a subset of the input window.
        
        Args:
            predictions: Array of prediction values (896 bins)
            chrom: Chromosome name
            start: Start coordinate of the original query region
            end: End coordinate of the original query region
            
        Returns:
            List of dictionaries with genomic coordinates and values for BedGraph
        """
        # Get the region center
        region_center = (start + end) // 2
        
        # Get output window coordinates
        output_start, output_end = self.get_output_window_coords(region_center)
        
        # Map predictions to coordinates
        mapped_predictions = []
        for i, value in enumerate(predictions):
            bin_start = output_start + i * self.bin_size
            bin_end = bin_start + self.bin_size
            
            mapped_predictions.append({
                'chrom': chrom,
                'start': bin_start,
                'end': bin_end,
                'value': float(value)
            })
        
        return mapped_predictions
    
    def analyze_gene_expression(self, predictions: Dict[str, np.ndarray], 
                              gene_name: str, 
                              chrom: str, start: int, end: int,
                              gtf_file: str,
                              cage_track_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze predicted gene expression using CAGE signal at TSS.
        
        For Enformer, we analyze gene expression by looking at CAGE signal
        around the transcription start sites (TSS) of the gene.
        
        Args:
            predictions: Dictionary of track predictions
            gene_name: Name of the gene to analyze
            chrom: Chromosome of the predicted region
            start: Start of the predicted region
            end: End of the predicted region  
            gtf_file: Path to GTF file with gene annotations
            cage_track_ids: List of CAGE track IDs to analyze
                          If None, uses all CAGE tracks in predictions
                          
        Returns:
            Dictionary with gene expression analysis:
            - tss_positions: List of TSS positions
            - cage_signals: Dict of track_id -> signals at each TSS
            - mean_expression: Dict of track_id -> mean expression
            - max_expression: Dict of track_id -> max expression
            
        Note:
            For Borzoi, we would sum RNA-seq signal over coding exons
            as described in their paper, but Enformer doesn't have RNA-seq tracks.
        """
        from ..utils.annotations import get_gene_tss
        
        # Get TSS positions for the gene
        tss_df = get_gene_tss(gene_name, annotation=gtf_file)
        
        if len(tss_df) == 0:
            logger.warning(f"No TSS found for gene {gene_name}")
            return {
                'tss_positions': [],
                'cage_signals': {},
                'mean_expression': {},
                'max_expression': {}
            }
        
        # Filter TSS positions to those in our region
        region_center = (start + end) // 2
        output_start, output_end = self.get_output_window_coords(region_center)
        
        tss_in_region = tss_df[
            (tss_df['chrom'] == chrom) &
            (tss_df['tss'] >= output_start) &
            (tss_df['tss'] <= output_end)
        ]
        
        if len(tss_in_region) == 0:
            logger.warning(f"No TSS for {gene_name} in output window")
            return {
                'tss_positions': [],
                'cage_signals': {},
                'mean_expression': {},
                'max_expression': {}
            }
        
        # Identify CAGE tracks if not specified
        if cage_track_ids is None:
            cage_track_ids = [
                track_id for track_id in predictions.keys()
                if 'CAGE' in track_id.upper() or track_id.startswith('CNhs')
            ]
        
        # Analyze CAGE signal at TSS positions
        cage_signals = {}
        mean_expression = {}
        max_expression = {}
        
        for track_id in cage_track_ids:
            if track_id not in predictions:
                continue
                
            track_signals = []
            
            for _, tss_info in tss_in_region.iterrows():
                tss_pos = tss_info['tss']
                
                # Convert TSS position to bin index
                tss_bin = (tss_pos - output_start) // self.bin_size
                
                # Get signal in window around TSS (e.g., +/- 5 bins = +/- 640bp)
                window_size = 5
                start_bin = max(0, tss_bin - window_size)
                end_bin = min(len(predictions[track_id]), tss_bin + window_size + 1)
                
                # Take max signal in window (TSS can be somewhat imprecise)
                if start_bin < end_bin:
                    window_signal = predictions[track_id][start_bin:end_bin]
                    track_signals.append(np.max(window_signal))
            
            cage_signals[track_id] = track_signals
            
            if track_signals:
                mean_expression[track_id] = np.mean(track_signals)
                max_expression[track_id] = np.max(track_signals)
            else:
                mean_expression[track_id] = 0.0
                max_expression[track_id] = 0.0
        
        return {
            'gene_name': gene_name,
            'tss_positions': tss_in_region['tss'].tolist(),
            'tss_info': tss_in_region.to_dict('records'),
            'cage_signals': cage_signals,
            'mean_expression': mean_expression,
            'max_expression': max_expression,
            'n_tss': len(tss_in_region)
        }
    
    def save_predictions_as_bedgraph(self, 
                                   predictions: Dict[str, np.ndarray],
                                   chrom: str,
                                   start: int,
                                   output_dir: str = ".",
                                   prefix: str = "",
                                   bin_size: Optional[int] = None,
                                   track_colors: Optional[Dict[str, str]] = None,
                                   end: Optional[int] = None) -> List[str]:
        """Save predictions as BedGraph files with Enformer-specific coordinate mapping.
        
        This overrides the base class method to handle Enformer's specific architecture
        where the output window is offset within the input window.
        
        Args:
            predictions: Dictionary mapping track names to prediction arrays
            chrom: Chromosome name
            start: Start coordinate of the query region
            output_dir: Directory to save files
            prefix: Prefix for output filenames
            bin_size: Bin size (uses self.bin_size if not provided)
            track_colors: Optional dict mapping track names to colors
            end: End coordinate of the query region (optional, used for coordinate mapping)
            
        Returns:
            List of created file paths
        """
        # If end is not provided, we need to calculate it based on the predictions
        if end is None:
            # Assume the predictions cover the full output window
            # This might not be ideal for all use cases
            logger.warning("End coordinate not provided for save_predictions_as_bedgraph. "
                         "Using start as region center for coordinate mapping.")
            region_center = start
        else:
            region_center = (start + end) // 2
        
        # Get the actual genomic coordinates for Enformer's output window
        output_start, _ = self.get_output_window_coords(region_center)
        
        # Use the base class method with the corrected start coordinate
        return super().save_predictions_as_bedgraph(
            predictions=predictions,
            chrom=chrom,
            start=output_start,  # Use the mapped output start
            output_dir=output_dir,
            prefix=prefix,
            bin_size=bin_size,
            track_colors=track_colors
        )
