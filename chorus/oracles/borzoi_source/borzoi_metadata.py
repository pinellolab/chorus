"""
Borzoi track metadata loader based on the targets file.
"""

import os
import pandas as pd
import urllib.request
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class BorzoiMetadata:
    """Class to handle Borzoi track metadata."""
    
    def __init__(self):
        # Path to the metadata file (in the same directory as this module)
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.metadata_file = os.path.join(module_dir, 'borzoi_human_targets.txt')
        
        # Alternative simpler metadata file from the Borzoi repo
        self.simple_metadata_file = "./borzoi_human_targets.txt"
        
        self.tracks_df = None
        self._track_index_map = {}
        self._load_metadata()
    
    def _download_metadata_if_needed(self):
        """Download metadata file if not present."""
        if os.path.exists(self.metadata_file):
            return True
            
        # Create directory if needed
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        
        # URL for the gzipped metadata file
        metadata_url = "https://github.com/calico/borzoi/raw/main/data/targets_human.txt.gz"
        
        logger.info(f"Downloading metadata file from {metadata_url}...")
        try:
            # Create a temporary file for download
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Download to temporary file
            urllib.request.urlretrieve(metadata_url, temp_path)
            
            # Decompress the gzipped file directly to the correct location
            import gzip
            with gzip.open(temp_path, 'rt') as f_in:
                with open(self.metadata_file, 'w') as f_out:
                    f_out.write(f_in.read())
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            logger.info(f"Successfully downloaded and decompressed metadata file to {self.metadata_file}")
            return True
                
        except Exception as e:
            logger.warning(f"Failed to download metadata file: {str(e)}")
            
            # Clean up temp file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return False
    
    def _load_metadata(self):
        """Load the metadata file."""
        # Try to download if not present
        if not os.path.exists(self.metadata_file):
            self._download_metadata_if_needed()
        
        # Try to load the structured metadata file
        if os.path.exists(self.metadata_file):
            try:
                # Read the file to check its format
                with open(self.metadata_file, 'r') as f:
                    first_line = f.readline().strip()
                
                # Check if the first line is a header
                if first_line.startswith('identifier') or 'description' in first_line:
                    # This is a header row, use pandas to read with header
                    self.tracks_df = pd.read_csv(self.metadata_file, sep='\t')
                else:
                    # No header, read with column names
                    column_names = ['identifier', 'file', 'clip', 'clip_soft', 'scale', 
                                   'sum_stat', 'strand_pair', 'description']
                    self.tracks_df = pd.read_csv(self.metadata_file, sep='\t', 
                                               names=column_names)

                # Get the identifier column - check different possible names
                identifier_col = None
                for col_name in ['identifier', '0', 0, 1]:
                    if col_name in self.tracks_df.columns:
                        identifier_col = col_name
                        break
                
                if identifier_col is not None:
                    # Create index mapping using the identifier column
                    self._track_index_map = {
                        str(row[identifier_col]): i 
                        for i, (_, row) in enumerate(self.tracks_df.iterrows())
                        if pd.notna(row[identifier_col])
                    }
                    logger.info(f"Created mapping for {len(self._track_index_map)} track identifiers")
                    logger.info(f"First few identifiers: {list(self._track_index_map.keys())[:5]}")
                    
                    # Add a numeric 'index' column to the DataFrame that matches row position
                    # This will be used by get_tracks_by_description and other methods
                    self.tracks_df['index'] = range(len(self.tracks_df))
                    
                    # Extract track_type and cell_type from description if available
                    if 'description' in self.tracks_df.columns:
                        # Initialize new columns
                        self.tracks_df['track_type'] = ""
                        self.tracks_df['cell_type'] = ""
                        
                        # Extract from description format like "ASSAY:CELL"
                        for idx, row in self.tracks_df.iterrows():
                            if pd.notna(row['description']) and ':' in row['description']:
                                parts = row['description'].split(':', 1)
                                self.tracks_df.at[idx, 'track_type'] = parts[0]
                                
                                # Extract cell type (everything after assay up to first space)
                                if len(parts) > 1:
                                    cell_parts = parts[1].split(' ', 1)
                                    self.tracks_df.at[idx, 'cell_type'] = cell_parts[0]
                else:
                    logger.warning(f"No suitable identifier column found in metadata. Columns: {self.tracks_df.columns.tolist()}")
                
                return
            except Exception as e:
                logger.warning(f"Failed to load structured metadata: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        logger.error("Failed to load metadata file. Make sure borzoi_human_targets.txt exists and is properly formatted.")
    
    def get_track_by_identifier(self, identifier: str) -> Optional[int]:
        """Get track index by identifier (e.g., K562_RNA-seq_plus)."""
        return self._track_index_map.get(identifier)
    
    def get_tracks_by_description(self, description: str) -> List[Tuple[int, str, str]]:
        """
        Get all tracks matching a description.
        
        Args:
            description: Description to search for (e.g., "RNA-seq:K562")
            
        Returns:
            List of (index, identifier, full_description) tuples
        """
        if self.tracks_df is None or self.tracks_df.empty:
            return []
        
        matches = []
        
        # Split description into parts for more flexible matching
        search_parts = description.lower().split(':')
        
        if len(search_parts) > 1 and 'track_type' in self.tracks_df.columns and 'cell_type' in self.tracks_df.columns:
            assay_type = search_parts[0]
            cell_type = search_parts[1]
            
            # Match both assay type and cell type
            for idx, row in self.tracks_df.iterrows():
                if (pd.notna(row['track_type']) and pd.notna(row['cell_type']) and
                    row['track_type'].lower() == assay_type.lower() and
                    row['cell_type'].lower() == cell_type.lower()):
                    matches.append((int(row['index']), row['identifier'], row['description']))
            
            if matches:
                return matches
        
        # Fall back to flexible text matching in description
        if 'description' in self.tracks_df.columns:
            for idx, row in self.tracks_df.iterrows():
                if pd.notna(row['description']) and description.lower() in row['description'].lower():
                    matches.append((int(row['index']), row['identifier'], row['description']))
        
        # Try matching in identifier field if still no matches
        if not matches and 'identifier' in self.tracks_df.columns:
            for idx, row in self.tracks_df.iterrows():
                if pd.notna(row['identifier']) and description.lower() in str(row['identifier']).lower():
                    matches.append((int(row['index']), row['identifier'], 
                                  row['description'] if 'description' in row and pd.notna(row['description']) else ""))
        
        return matches
    
    def get_track_info(self, index: int) -> Optional[Dict]:
        """Get full information for a track by index."""
        if self.tracks_df is None or self.tracks_df.empty:
            return None
        
        track = self.tracks_df[self.tracks_df['index'] == index]
        if track.empty:
            return None
        
        return track.iloc[0].to_dict()

    def id2index(self, ids: List[str]) -> List[int]:
        """Get track indices by identifiers."""
        return [self._track_index_map.get(id) for id in ids]
    
    def parse_description(self, desc: str) -> dict[str]:
      
        parts = desc.split(':')
        if len(parts) > 0:
            assay_type = parts[0]
        else:
            assay_type = self.DEFAULT_ASSAY_TYPE
        if len(parts) > 1:
            cell_type_part = parts[1].strip()
            # Handle different formats:
            # Simple: "K562"
            # Complex: "H1-hESC", "HeLa-S3"
            # With modifiers: "K562 treated with...", "GM12878 male adult..."
            # Just take the first token (which might include hyphens)
            tokens = cell_type_part.split()
            if tokens:
                # The cell type is the first token (may include hyphens)
                cell_type = tokens[0]
                # Remove trailing commas or other punctuation
                cell_type = cell_type.rstrip(',.')
        else:
            cell_type = self.DEFAULT_CELL_TYPE 

        return {'assay_type': assay_type, 'cell_type': cell_type}
    
    def get_track_summary(self) -> Dict[str, int]:
        """Get summary of available tracks by assay type."""
        if self.tracks_df is None or self.tracks_df.empty:
            return {}
        
        summary = {}
        for assay_type in self.list_assay_types():
            count = len(self.tracks_df[self.tracks_df['description'].str.startswith(f"{assay_type}:")])
            summary[assay_type] = count
        
        return summary
    
    def list_cell_types(self) -> List[str]:
        """List all unique cell types."""
        if self.tracks_df is None or self.tracks_df.empty or 'cell_type' not in self.tracks_df.columns:
            return []
        
        return sorted(self.tracks_df['cell_type'].dropna().unique().tolist())
    
    def list_assay_types(self) -> List[str]:
        """List all unique assay types."""
        if self.tracks_df is None or self.tracks_df.empty or 'track_type' not in self.tracks_df.columns:
            return []
        
        return sorted(self.tracks_df['track_type'].dropna().unique().tolist())
    
    def search_tracks(self, query: str) -> pd.DataFrame:
        """Search tracks by any field."""
        if self.tracks_df is None or self.tracks_df.empty:
            return pd.DataFrame()
        
        # Search in all text columns
        mask = False
        for col in ['description', 'identifier', 'track_type', 'cell_type']:
            if col in self.tracks_df.columns:
                mask = mask | self.tracks_df[col].astype(str).str.contains(query, case=False)
        
        return self.tracks_df[mask]

# Global instance
_metadata = None

def get_metadata():
    """Get the global metadata instance."""
    global _metadata
    if _metadata is None:
        _metadata = BorzoiMetadata()
    return _metadata
