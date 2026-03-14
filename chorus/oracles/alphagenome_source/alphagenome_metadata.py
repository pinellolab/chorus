"""
AlphaGenome track metadata.

AlphaGenome predicts 5,930+ human functional genomic tracks across multiple
modalities.  Track information is obtained programmatically from the
alphagenome_research metadata module.

When the alphagenome packages are not installed (e.g. in the main chorus
environment) we fall back to a cached JSON file that is generated on first
use inside the oracle conda environment.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Map AlphaGenome OutputType names to chorus track type names.
OUTPUT_TYPE_TO_CHORUS = {
    "RNA_SEQ": "RNA",
    "CAGE": "CAGE",
    "ATAC": "ATAC",
    "DNASE": "DNASE",
    "CHIP_HISTONE": "CHIP",
    "CHIP_TF": "CHIP",
    "SPLICE_SITES": "SPLICE_SITES",
    "SPLICE_SITE_USAGE": "SPLICE_SITES",
    "PROCAP": "PRO_CAP",
}

# Modalities we skip for now (complex output formats).
SKIPPED_OUTPUT_TYPES = {"CONTACT_MAPS", "SPLICE_JUNCTIONS"}

# Resolution per output type (bp per bin).
OUTPUT_TYPE_RESOLUTION = {
    "ATAC": 1,
    "CAGE": 1,
    "DNASE": 1,
    "RNA_SEQ": 1,
    "CHIP_HISTONE": 128,
    "CHIP_TF": 128,
    "SPLICE_SITES": 1,
    "SPLICE_SITE_USAGE": 1,
    "PROCAP": 1,
}

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(MODULE_DIR, "alphagenome_tracks.json")


def _build_track_list_from_package() -> List[dict]:
    """Build track metadata from the installed alphagenome_research package."""
    from alphagenome.models.dna_output import OutputType
    from alphagenome.models.dna_model import Organism
    from alphagenome_research.model.metadata import metadata as metadata_lib

    output_metadata = metadata_lib.load(Organism.HOMO_SAPIENS)

    tracks: List[dict] = []
    idx = 0
    for output_type in OutputType:
        ot_name = output_type.name
        if ot_name in SKIPPED_OUTPUT_TYPES:
            continue
        chorus_type = OUTPUT_TYPE_TO_CHORUS.get(ot_name, ot_name)
        resolution = OUTPUT_TYPE_RESOLUTION.get(ot_name, 1)

        track_meta = output_metadata.get(output_type)
        if track_meta is None:
            continue

        for local_idx, (_, row) in enumerate(track_meta.iterrows()):
            name = row.get("name", "")
            biosample = row.get("biosample_name", "")
            if not isinstance(biosample, str):
                biosample = ""
            strand = row.get("strand", ".")
            # Build a unique identifier: output_type/name/strand
            # Append local_index for padding tracks to avoid collisions
            identifier = f"{ot_name}/{name}/{strand}"
            if name.lower() == "padding":
                identifier = f"{identifier}/{local_idx}"
            tracks.append({
                "index": idx,
                "identifier": identifier,
                "name": name,
                "output_type": ot_name,
                "chorus_type": chorus_type,
                "cell_type": biosample,
                "strand": strand,
                "resolution": resolution,
                "local_index": local_idx,
                "description": f"{chorus_type}:{biosample}" if biosample else chorus_type,
            })
            idx += 1

    return tracks


def _save_cache(tracks: List[dict]) -> None:
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(tracks, f)
        logger.info(f"Cached {len(tracks)} AlphaGenome tracks to {CACHE_FILE}")
    except OSError as e:
        logger.warning(f"Could not write track cache: {e}")


def _load_cache() -> Optional[List[dict]]:
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Could not read track cache: {e}")
        return None


class AlphaGenomeMetadata:
    """AlphaGenome track metadata manager."""

    DEFAULT_ASSAY_TYPE = "UNKNOWN"
    DEFAULT_CELL_TYPE = "UNKNOWN"

    def __init__(self):
        self._tracks: List[dict] = []
        self._track_index_map: Dict[str, int] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        # Try installed package first
        try:
            self._tracks = _build_track_list_from_package()
            _save_cache(self._tracks)
        except Exception as e:
            logger.debug(f"Could not build metadata from package: {e}")
            # Fall back to cache
            cached = _load_cache()
            if cached is not None:
                self._tracks = cached
            else:
                logger.warning(
                    "alphagenome packages not available and no track cache found. "
                    "Metadata will be empty until the oracle environment is used."
                )
                return

        self._track_index_map = {
            t["identifier"]: t["index"] for t in self._tracks
        }
        logger.info(f"Loaded {len(self._tracks)} AlphaGenome tracks")

    # ------------------------------------------------------------------
    # Public API (mirrors BorzoiMetadata interface)
    # ------------------------------------------------------------------
    def get_track_by_identifier(self, identifier: str) -> Optional[int]:
        return self._track_index_map.get(identifier)

    def get_track_info(self, index: int) -> Optional[dict]:
        for t in self._tracks:
            if t["index"] == index:
                return t
        return None

    def id2index(self, ids: List[str]) -> List[Optional[int]]:
        return [self._track_index_map.get(i) for i in ids]

    def parse_description(self, desc: str) -> dict:
        parts = desc.split(":", 1)
        assay_type = parts[0] if parts else self.DEFAULT_ASSAY_TYPE
        cell_type = (
            parts[1].split()[0].rstrip(",.")
            if len(parts) > 1 and parts[1].strip()
            else self.DEFAULT_CELL_TYPE
        )
        return {"assay_type": assay_type, "cell_type": cell_type}

    def get_tracks_by_description(
        self, description: str
    ) -> List[Tuple[int, str, str]]:
        query = description.lower()
        matches = []
        for t in self._tracks:
            if (
                query in t.get("description", "").lower()
                or query in t.get("identifier", "").lower()
            ):
                matches.append(
                    (t["index"], t["identifier"], t.get("description", ""))
                )
        return matches

    def list_assay_types(self) -> List[str]:
        return sorted({t["chorus_type"] for t in self._tracks})

    def list_cell_types(self) -> List[str]:
        return sorted(
            {
                t["cell_type"]
                for t in self._tracks
                if t.get("cell_type") and isinstance(t["cell_type"], str)
            }
        )

    def get_track_summary(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for t in self._tracks:
            ct = t["chorus_type"]
            summary[ct] = summary.get(ct, 0) + 1
        return summary

    def search_tracks(self, query: str):
        import pandas as pd

        query_lower = query.lower()
        matches = [
            t
            for t in self._tracks
            if query_lower in t.get("identifier", "").lower()
            or query_lower in t.get("name", "").lower()
            or query_lower in t.get("description", "").lower()
            or query_lower in t.get("cell_type", "").lower()
        ]
        return pd.DataFrame(matches) if matches else pd.DataFrame()


# Global singleton
_metadata: Optional[AlphaGenomeMetadata] = None


def get_metadata() -> AlphaGenomeMetadata:
    global _metadata
    if _metadata is None:
        _metadata = AlphaGenomeMetadata()
    return _metadata
