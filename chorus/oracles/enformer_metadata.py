"""
Enformer track metadata loader based on the official targets file.
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class EnformerMetadata:
    """Class to handle Enformer track metadata."""
    
    def __init__(self):
        self.metadata_file = os.path.join(os.path.dirname(__file__), 'enformer_human_targets.txt')
        self.tracks_df = None
        self._track_index_map = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load the metadata file."""
        try:
            self.tracks_df = pd.read_csv(self.metadata_file, sep='\t')
            # Create index mapping
            for idx, row in self.tracks_df.iterrows():
                self._track_index_map[row['identifier']] = row['index']
            logger.info(f"Loaded {len(self.tracks_df)} track metadata entries")
        except Exception as e:
            logger.error(f"Failed to load Enformer metadata: {e}")
            self.tracks_df = pd.DataFrame()
    
    def get_track_by_identifier(self, identifier: str) -> Optional[int]:
        """Get track index by ENCODE identifier (e.g., ENCFF413AHU)."""
        return self._track_index_map.get(identifier)
    
    def get_tracks_by_description(self, description: str) -> List[Tuple[int, str, str]]:
        """
        Get all tracks matching a description.
        
        Args:
            description: Description to search for (e.g., "DNASE:K562")
            
        Returns:
            List of (index, identifier, full_description) tuples
        """
        if self.tracks_df is None or self.tracks_df.empty:
            return []
        
        # Exact match first
        exact_matches = self.tracks_df[self.tracks_df['description'] == description]
        if not exact_matches.empty:
            return [(row['index'], row['identifier'], row['description']) 
                    for _, row in exact_matches.iterrows()]
        
        # Partial match
        partial_matches = self.tracks_df[self.tracks_df['description'].str.contains(description, case=False)]
        return [(row['index'], row['identifier'], row['description']) 
                for _, row in partial_matches.iterrows()]
    
    def get_track_info(self, index: int) -> Optional[Dict]:
        """Get full information for a track by index."""
        if self.tracks_df is None or self.tracks_df.empty:
            return None
        
        track = self.tracks_df[self.tracks_df['index'] == index]
        if track.empty:
            return None
        
        return track.iloc[0].to_dict()
    
    def list_cell_types(self) -> List[str]:
        """List all unique cell types."""
        if self.tracks_df is None or self.tracks_df.empty:
            return []
        
        cell_types = set()
        for desc in self.tracks_df['description']:
            # Extract cell type from description
            parts = desc.split(':')
            if len(parts) > 1:
                cell_types.add(parts[1].split()[0])  # Get first word after colon
        
        return sorted(list(cell_types))
    
    def list_assay_types(self) -> List[str]:
        """List all unique assay types."""
        if self.tracks_df is None or self.tracks_df.empty:
            return []
        
        assay_types = set()
        for desc in self.tracks_df['description']:
            # Extract assay type from description
            parts = desc.split(':')
            if len(parts) > 0:
                assay_types.add(parts[0])
        
        return sorted(list(assay_types))
    
    def search_tracks(self, query: str) -> pd.DataFrame:
        """Search tracks by any field."""
        if self.tracks_df is None or self.tracks_df.empty:
            return pd.DataFrame()
        
        # Search in description and identifier
        mask = (self.tracks_df['description'].str.contains(query, case=False) | 
                self.tracks_df['identifier'].str.contains(query, case=False))
        
        return self.tracks_df[mask]

# Global instance
_metadata = None

def get_metadata():
    """Get the global metadata instance."""
    global _metadata
    if _metadata is None:
        _metadata = EnformerMetadata()
    return _metadata