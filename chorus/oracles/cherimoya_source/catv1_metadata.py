"""Metadata index over the CATv1 experiment atlas.

Shaped after :class:`~chorus.oracles.borzoi_source.borzoi_metadata.BorzoiMetadata`
rather than ChromBPNet's flat dict, because CATv1 is a large atlas (1,518
tracks) and needs to be *searched* rather than enumerated — Borzoi already
ships 7,612 tracks behind this interface and ``list_tracks`` in the MCP
server routes it through ``search_tracks`` with a result cap.

The two TSVs are vendored next to this module (281 KB total, smaller than
either the Borzoi or ChromBPNet vendored metadata) so that listing and
searching tracks works offline and pins to the code version.  Only the
model checkpoints are fetched lazily from HuggingFace.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas

from .catv1_defaults import CATV1_DEFAULT_EXPERIMENT
from .catv1_globals import (
    ASSAY_TERM_TO_CHORUS,
    CATV1_ASSAY_TYPES,
    catv1_track_id,
)

logger = logging.getLogger(__name__)

_SOURCE_DIR = Path(__file__).resolve().parent
_METADATA_TSV = _SOURCE_DIR / "CATv1-metadata.tsv"
_PERFORMANCE_TSV = _SOURCE_DIR / "CATv1-performance-fold0.tsv"

# Columns searched by `search_tracks`.
_SEARCH_COLUMNS = [
    "track_id",
    "experiment_accession",
    "assay",
    "biosample",
    "biosample_classification",
    "biosample_summary",
]


class CATv1Metadata:
    """Searchable index over the 1,518 CATv1 experiments."""

    def __init__(self):
        self.tracks_df = self._load()

    def _load(self) -> pandas.DataFrame:
        meta = pandas.read_csv(_METADATA_TSV, sep="\t")

        df = pandas.DataFrame({
            "experiment_accession": meta["experiment_accession"],
            "annotation_accession": meta["annotation_accession"],
            "assay": meta["assay_term_name"].map(ASSAY_TERM_TO_CHORUS),
            "biosample": meta["experiment_biosample_term_name"],
            "biosample_classification": meta["biosample_classification"],
            "biosample_summary": meta["biosample_simple_summary"],
            "biosample_term_id": meta["biosample_term_id"],
            "assembly": meta["assembly"],
            "perturbed": meta["experiment_peturbed"].astype(bool),
        })

        unmapped = df["assay"].isna()
        if unmapped.any():
            # Loud rather than silent: an unmapped assay would produce a
            # track whose assay_type doesn't dispatch to a DNase/ATAC
            # track class, and would land in the 'other' layer with no
            # background normalization.
            raise ValueError(
                f"{int(unmapped.sum())} CATv1 rows have an assay_term_name "
                f"outside {sorted(ASSAY_TERM_TO_CHORUS)}: "
                f"{sorted(meta.loc[unmapped, 'assay_term_name'].unique())}"
            )

        df["track_id"] = [
            catv1_track_id(a, e)
            for a, e in zip(df["assay"], df["experiment_accession"])
        ]

        perf = pandas.read_csv(_PERFORMANCE_TSV, sep="\t")
        perf = perf[["experiment_accession", "profile_pearson", "count_pearson"]]
        df = df.merge(perf, on="experiment_accession", how="left")

        return df

    # ── lookup ───────────────────────────────────────────────────────

    def resolve(
        self,
        assay: Optional[str] = None,
        cell_type: Optional[str] = None,
        encode_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Resolve a track request to ``(assay, experiment_accession)``.

        ``encode_id`` wins when given.  Otherwise ``(assay, cell_type)`` is
        looked up in the committed defaults table; for the 162 ambiguous
        pairs this logs which experiment was chosen and how many
        alternatives exist, so a user who cares can pin ``encode_id``.

        Args:
            assay: ``'ATAC'`` or ``'DNASE'`` (case-insensitive).
            cell_type: Biosample term name, e.g. ``'K562'``.
            encode_id: ENCODE experiment accession; overrides the pair.

        Returns:
            ``(assay, experiment_accession)``.

        Raises:
            KeyError: unknown accession, or no default for the pair.
            ValueError: neither ``encode_id`` nor both of assay/cell_type.
        """
        if encode_id is not None:
            row = self.tracks_df[self.tracks_df["experiment_accession"] == encode_id]
            if row.empty:
                raise KeyError(
                    f"Unknown CATv1 experiment accession {encode_id!r}. "
                    f"Search with CATv1Metadata().search_tracks(...)."
                )
            resolved_assay = row.iloc[0]["assay"]
            if assay is not None and assay.upper() != resolved_assay:
                raise ValueError(
                    f"{encode_id} is a {resolved_assay} experiment, but "
                    f"assay={assay!r} was requested."
                )
            return resolved_assay, encode_id

        if assay is None or cell_type is None:
            raise ValueError(
                "Provide either encode_id, or both assay and cell_type."
            )

        assay = assay.upper()
        if assay not in CATV1_ASSAY_TYPES:
            raise KeyError(
                f"CATv1 covers {CATV1_ASSAY_TYPES}, not {assay!r}."
            )

        entry = CATV1_DEFAULT_EXPERIMENT.get((assay, cell_type))
        if entry is None:
            raise KeyError(
                f"No CATv1 experiment for assay={assay!r}, "
                f"cell_type={cell_type!r}. Search with "
                f"CATv1Metadata().search_tracks({cell_type!r})."
            )

        chosen, n_candidates, reason = entry
        if n_candidates > 1:
            logger.info(
                "%s:%s maps to %d CATv1 experiments; using %s (%s). "
                "Pass encode_id=... to pick a different one.",
                assay, cell_type, n_candidates, chosen, reason,
            )
        return assay, chosen

    def describe(self, track_or_accession: str) -> Dict:
        """Return the metadata row for a track id or accession as a dict.

        The track id intentionally carries only assay + accession, so this
        is how a caller recovers the biosample and fold-0 metrics.
        """
        key = track_or_accession
        df = self.tracks_df
        row = df[(df["track_id"] == key) | (df["experiment_accession"] == key)]
        if row.empty:
            raise KeyError(f"Unknown CATv1 track or accession {key!r}.")
        return row.iloc[0].to_dict()

    # ── Borzoi-shaped interface, used by the MCP `list_tracks` branch ──

    def list_assay_types(self) -> List[str]:
        """List assay types present in the atlas."""
        return sorted(self.tracks_df["assay"].dropna().unique().tolist())

    def list_cell_types(self) -> List[str]:
        """List every biosample term name in the atlas."""
        return sorted(self.tracks_df["biosample"].dropna().unique().tolist())

    def list_track_ids(self) -> List[str]:
        """List every canonical track id, sorted."""
        return sorted(self.tracks_df["track_id"].tolist())

    def get_track_summary(self) -> Dict[str, int]:
        """Count experiments per assay type."""
        return {
            assay: int((self.tracks_df["assay"] == assay).sum())
            for assay in self.list_assay_types()
        }

    def search_tracks(self, query: str) -> pandas.DataFrame:
        """Case-insensitive substring search across the descriptive columns.

        Args:
            query: e.g. ``'K562'``, ``'ENCSR000EOT'``, ``'T cell'``, ``'ATAC'``.

        Returns:
            Matching rows. Empty frame when nothing matches.
        """
        df = self.tracks_df
        if not query:
            return df

        mask = pandas.Series(False, index=df.index)
        for col in _SEARCH_COLUMNS:
            if col in df.columns:
                mask |= df[col].astype(str).str.contains(query, case=False, regex=False)
        return df[mask]


_metadata: Optional[CATv1Metadata] = None


def get_metadata() -> CATv1Metadata:
    """Return the process-wide :class:`CATv1Metadata`, loading on first use."""
    global _metadata
    if _metadata is None:
        _metadata = CATv1Metadata()
    return _metadata
