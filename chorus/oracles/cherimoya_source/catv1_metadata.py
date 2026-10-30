"""Metadata index over the Cherimoya atlases: CATv1 (accessibility) and BATv1 (ChIP-seq).

Shaped after :class:`~chorus.oracles.borzoi_source.borzoi_metadata.BorzoiMetadata`
rather than ChromBPNet's flat dict, because the atlases are large (1,518 + 3,429
tracks) and need to be *searched* rather than enumerated — Borzoi already
ships 7,611 tracks behind this interface and ``list_tracks`` in the MCP
server routes it through ``search_tracks`` with a result cap.

The four TSVs are vendored next to this module (about 1 MB total) so that
listing and searching tracks works offline and pins to the code version.
Only the model checkpoints are fetched lazily from HuggingFace.

The two atlases share one frame and one ``ASSAY:ENCSR`` id scheme.  A row's
``atlas`` column says which it is, ``assay`` is ``ATAC`` / ``DNASE`` for CATv1
and ``CHIP`` for BATv1, and ``target`` is the ChIP-seq target for BATv1 rows and
empty for CATv1 rows.  The module keeps its CATv1 name because every importer
reaches it by that path; :class:`CATv1Metadata` is kept as an alias of
:class:`CherimoyaMetadata`.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas

from .batv1_defaults import BATV1_DEFAULT_EXPERIMENT
from .batv1_globals import BATV1_ASSAY_TERM_TO_CHORUS, BATV1_ASSAY_TYPES
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
_BATV1_METADATA_TSV = _SOURCE_DIR / "BATv1-metadata.tsv"
_BATV1_PERFORMANCE_TSV = _SOURCE_DIR / "BATv1-performance-fold0.tsv"

# Columns searched by `search_tracks`.
_SEARCH_COLUMNS = [
    "track_id",
    "experiment_accession",
    "assay",
    "target",
    "biosample",
    "biosample_classification",
    "biosample_summary",
]

# Column order of the shared frame. Columns BATv1 adds are empty for CATv1 rows.
_COLUMNS = [
    "atlas",
    "track_id",
    "experiment_accession",
    "annotation_accession",
    "assay",
    "target",
    "biosample",
    "biosample_classification",
    "biosample_summary",
    "biosample_term_id",
    "assembly",
    "perturbed",
    "target_investigated_as",
    "experiment_status",
    "n_peaks",
    "quality_flags",
    "profile_pearson",
    "count_pearson",
]


def _load_atlas(
    atlas: str,
    metadata_tsv: Path,
    performance_tsv: Path,
    assay_map: Dict[str, str],
    perturbed_column: str,
) -> pandas.DataFrame:
    meta = pandas.read_csv(metadata_tsv, sep="\t", dtype=str, keep_default_na=False)

    df = pandas.DataFrame({
        "atlas": atlas,
        "experiment_accession": meta["experiment_accession"],
        "annotation_accession": meta["annotation_accession"],
        "assay": meta["assay_term_name"].map(assay_map),
        "target": meta["target_label"] if "target_label" in meta else "",
        "biosample": meta["experiment_biosample_term_name"],
        "biosample_classification": meta["biosample_classification"],
        "biosample_summary": meta["biosample_simple_summary"],
        "biosample_term_id": meta["biosample_term_id"],
        "assembly": meta["assembly"],
        "perturbed": meta[perturbed_column].str.upper() == "TRUE",
        "target_investigated_as": meta["target_investigated_as"] if "target_investigated_as" in meta else "",
        "experiment_status": meta["experiment_status"] if "experiment_status" in meta else "released",
        "n_peaks": meta["n_peaks"] if "n_peaks" in meta else "",
        "quality_flags": meta["quality_flags"].replace("none", "") if "quality_flags" in meta else "",
    })

    unmapped = df["assay"].isna()
    if unmapped.any():
        # Loud rather than silent: an unmapped assay would produce a
        # track whose assay_type doesn't dispatch to a DNase/ATAC/CHIP
        # track class, and would land in the 'other' layer with no
        # background normalization.
        raise ValueError(
            f"{int(unmapped.sum())} {atlas} rows have an assay_term_name "
            f"outside {sorted(assay_map)}: "
            f"{sorted(meta.loc[unmapped, 'assay_term_name'].unique())}"
        )

    df["track_id"] = [
        catv1_track_id(a, e)
        for a, e in zip(df["assay"], df["experiment_accession"])
    ]

    perf = pandas.read_csv(performance_tsv, sep="\t")
    perf = perf[["experiment_accession", "profile_pearson", "count_pearson"]]
    df = df.merge(perf, on="experiment_accession", how="left")
    return df[_COLUMNS]


class CherimoyaMetadata:
    """Searchable index over the 1,518 CATv1 and 3,429 BATv1 experiments."""

    def __init__(self):
        self.tracks_df = self._load()

    def _load(self) -> pandas.DataFrame:
        catv1 = _load_atlas(
            "CATv1", _METADATA_TSV, _PERFORMANCE_TSV, ASSAY_TERM_TO_CHORUS,
            # The release misspells the column; BATv1 spells it correctly.
            perturbed_column="experiment_peturbed",
        )
        batv1 = _load_atlas(
            "BATv1", _BATV1_METADATA_TSV, _BATV1_PERFORMANCE_TSV, BATV1_ASSAY_TERM_TO_CHORUS,
            perturbed_column="experiment_perturbed",
        )
        df = pandas.concat([catv1, batv1], ignore_index=True)
        if df["track_id"].duplicated().any():
            raise ValueError("CATv1 and BATv1 share an experiment accession; track ids collide.")
        return df

    # ── lookup ───────────────────────────────────────────────────────

    def resolve(
        self,
        assay: Optional[str] = None,
        cell_type: Optional[str] = None,
        encode_id: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Resolve a track request to ``(assay, experiment_accession)``.

        ``encode_id`` wins when given.  Otherwise ``(assay, cell_type)`` -- plus
        ``tf`` for ``CHIP`` -- is looked up in the committed defaults table; for
        the ambiguous pairs this logs which experiment was chosen and how many
        alternatives exist, so a user who cares can pin ``encode_id``.

        Args:
            assay: ``'ATAC'``, ``'DNASE'`` or ``'CHIP'`` (case-insensitive).
            cell_type: Biosample term name, e.g. ``'K562'``.
            encode_id: ENCODE experiment accession; overrides the pair.
            tf: ChIP-seq target as ENCODE names it, e.g. ``'GATA1'``.
                Required with ``assay='CHIP'`` unless ``encode_id`` is given.

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
                    f"Unknown Cherimoya experiment accession {encode_id!r}. "
                    f"Search with CherimoyaMetadata().search_tracks(...)."
                )
            resolved_assay = row.iloc[0]["assay"]
            if assay is not None and assay.upper() != resolved_assay:
                raise ValueError(
                    f"{encode_id} is a {resolved_assay} experiment, but "
                    f"assay={assay!r} was requested."
                )
            if tf is not None and resolved_assay == "CHIP" and row.iloc[0]["target"] != tf:
                raise ValueError(
                    f"{encode_id} is a {row.iloc[0]['target']} experiment, but "
                    f"TF={tf!r} was requested."
                )
            return resolved_assay, encode_id

        if assay is None or cell_type is None:
            raise ValueError(
                "Provide either encode_id, or both assay and cell_type "
                "(and TF for assay='CHIP')."
            )

        assay = assay.upper()
        if assay in BATV1_ASSAY_TYPES:
            if tf is None:
                raise ValueError(
                    "assay='CHIP' needs TF=<target> as well as cell_type, or an "
                    "encode_id. Search with CherimoyaMetadata().search_tracks(...)."
                )
            entry = BATV1_DEFAULT_EXPERIMENT.get((cell_type, tf))
            if entry is None:
                raise KeyError(
                    f"No BATv1 experiment for cell_type={cell_type!r}, TF={tf!r}. "
                    f"Search with CherimoyaMetadata().search_tracks({tf!r})."
                )
            label = f"{assay}:{cell_type}:{tf}"
        elif assay in CATV1_ASSAY_TYPES:
            entry = CATV1_DEFAULT_EXPERIMENT.get((assay, cell_type))
            if entry is None:
                raise KeyError(
                    f"No CATv1 experiment for assay={assay!r}, "
                    f"cell_type={cell_type!r}. Search with "
                    f"CherimoyaMetadata().search_tracks({cell_type!r})."
                )
            label = f"{assay}:{cell_type}"
        else:
            raise KeyError(
                f"Cherimoya covers {sorted(CATV1_ASSAY_TYPES + BATV1_ASSAY_TYPES)}, "
                f"not {assay!r}."
            )

        chosen, n_candidates, reason = entry
        if n_candidates > 1:
            logger.info(
                "%s maps to %d experiments; using %s (%s). "
                "Pass encode_id=... to pick a different one.",
                label, n_candidates, chosen, reason,
            )
        return assay, chosen

    def describe(self, track_or_accession: str) -> Dict:
        """Return the metadata row for a track id or accession as a dict.

        The track id intentionally carries only assay + accession, so this
        is how a caller recovers the biosample, the target and fold-0 metrics.
        """
        key = track_or_accession
        df = self.tracks_df
        row = df[(df["track_id"] == key) | (df["experiment_accession"] == key)]
        if row.empty:
            raise KeyError(f"Unknown Cherimoya track or accession {key!r}.")
        return row.iloc[0].to_dict()

    # ── Borzoi-shaped interface, used by the MCP `list_tracks` branch ──

    def list_assay_types(self) -> List[str]:
        """List assay types present across both atlases."""
        return sorted(self.tracks_df["assay"].dropna().unique().tolist())

    def list_cell_types(self, assay: Optional[str] = None) -> List[str]:
        """List biosample term names.

        Like ChromBPNet's, the default answers "which biosamples can I ask about
        accessibility in?" -- the CATv1 biosamples.  Pass ``assay="CHIP"`` for
        the BATv1 biosamples, or ``"ATAC"`` / ``"DNASE"`` to narrow further.
        """
        df = self.tracks_df
        if assay is None:
            df = df[df["atlas"] == "CATv1"]
        else:
            df = df[df["assay"] == assay.upper()]
        return sorted(df["biosample"].dropna().unique().tolist())

    def list_targets(self) -> List[str]:
        """List every ChIP-seq target in BATv1 (1,139 of them)."""
        df = self.tracks_df
        return sorted(df.loc[df["atlas"] == "BATv1", "target"].unique().tolist())

    def list_track_ids(self) -> List[str]:
        """List every canonical track id across both atlases, sorted."""
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
            query: e.g. ``'K562'``, ``'ENCSR000EOT'``, ``'T cell'``, ``'GATA1'``.

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


#: Kept for importers that predate BATv1; the class covers both atlases.
CATv1Metadata = CherimoyaMetadata

_metadata: Optional[CherimoyaMetadata] = None


def get_metadata() -> CherimoyaMetadata:
    """Return the process-wide :class:`CherimoyaMetadata`, loading on first use."""
    global _metadata
    if _metadata is None:
        _metadata = CherimoyaMetadata()
    return _metadata
