"""
ChromBPNet metadata on JASPAR models.
"""

import os
import pandas as pd
import logging
from typing import List

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class BPNetMetadata:

    def __init__(self):
        self.metadata_path = os.path.join(os.path.dirname(__file__), "chrombpnet_JASPAR_metadata.tsv")
        self.TF_df = None
        self._load_metadata()

    def _load_metadata(self):
        """Loads metadata file"""
        if not os.path.exists(self.metadata_path):
            logger.error(f"BPNet metadata file not found: {self.metadata_file}")
            logger.error("This file should be included with the Chorus package.")
            logger.error("If missing, you can download it from:")
            logger.error("https://mencius.uio.no/JASPAR/JASPAR_metadata/2026/")
            self.TF_df = pd.DataFrame()
            return
        
        try:
            self.TF_df = pd.read_csv(self.metadata_path, sep="\t")
            logger.info("BPNet metadata loaded.")
        except Exception as e:
            logger.error(f"Failed to load BPNet metadata: {e}")
            self.TF_df = pd.DataFrame()

    def list_cell_types(self) -> List[str]:
        if self.TF_df is None or self.TF_df.empty:
            return []
        
        cell_types = self.TF_df["CELL_LINE"].unique().tolist()
        return cell_types
    
    def list_TFs(self) -> List[str]:
        if self.TF_df is None or self.TF_df.empty:
            return []
        
        TFs = self.TF_df["TF_NAME"].unique().tolist()
        return TFs
    
    def list_TFs_by_cell_type(self, cell_type: str) -> List[str]:
        if self.TF_df is None or self.TF_df.empty:
            return []

        TFs_by_cell_type = self.TF_df[self.TF_df["CELL_LINE"] == cell_type]["TF_NAME"].tolist()
        return TFs_by_cell_type
    
    def list_cell_types_by_TF(self, TF: str) -> List[str]:
        if self.TF_df is None or self.TF_df.empty:
            return []
        
        cells_by_TF = self.TF_df[self.TF_df["TF_NAME"] == TF]["CELL_LINE"].tolist()
        return cells_by_TF
    
    def get_weights_by_cell_and_tf(self, tf: str, cell_type: str) -> str:
        models = self.TF_df[
            (self.TF_df["TF_NAME"] == tf) & (self.TF_df["CELL_LINE"] == cell_type)
        ]

        if models.empty:
            error_msg = f"""Incorrect combination of cell type and TF
Available cell types for {tf}: {self.list_cell_types_by_TF(tf)}
Available TFs for {cell_type}: {self.list_TFs_by_cell_type(cell_type)}
            """
            raise ValueError(error_msg)
        
        if len(models) > 1:
            logger.info("More than one model match the input combination, returning the first one.")
            
        return models["MODEL_URL"].values[0]
    

# TESTING
if __name__ == "__main__":
    bpnet_metadata = BPNetMetadata()
    print(bpnet_metadata.list_TFs())
    print(bpnet_metadata.list_cell_types())
    print(bpnet_metadata.list_cell_types_by_TF("REST"))
    print(bpnet_metadata.list_TFs_by_cell_type("K562"))
    print(bpnet_metadata.get_weights_by_cell_and_tf(tf="REST", cell_type="K562"))

    # More than one model
    print(bpnet_metadata.get_weights_by_cell_and_tf(tf="YY1", cell_type="K562"))

    # Should raise ValueError
    print(bpnet_metadata.get_weights_by_cell_and_tf(tf="REST", cell_type="opla"))


