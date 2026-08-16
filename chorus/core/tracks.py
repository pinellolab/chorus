"""One shape for "what can this oracle predict?".

Before this, answering that question took a different call per oracle — `get_all_assay_ids()` on four
of them, `list_tracks()` on cherimoya, a *private* `_get_all_assay_ids()` on sei, and nothing at all on
chrombpnet, legnet or epinformerseq. `get_track_info()` existed on four oracles and returned either a
DataFrame or a dict depending on whether you passed an argument, and a *different* `get_track_info` on
the metadata classes took an integer index. An audit needed four attempts to obtain one track id
(`audits/2026-08-15_gpu_cdf_determinism_reproducibility.md:82-99`).

The cost was not only friction. Because there was no uniform call, consumers hardcoded per-oracle
branches, and those branches drifted from the oracles: `chorus/analysis/discovery.py` calls
`get_all_assay_ids()` for sei, which sei does not have, and the MCP `list_tracks` tool answers for sei
from a hardcoded literal without ever asking the oracle.

`TrackRecord` is deliberately a plain frozen dataclass rather than a DataFrame: these cross the
subprocess boundary into per-oracle conda envs, where pandas is not guaranteed and where a stable,
JSON-serialisable shape matters more than tabular convenience. Call `.as_dict()` when a payload is
needed and build a DataFrame at the edge if you want one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrackRecord:
    """One predictable track, described uniformly across oracles.

    ``track_id`` is the contract: it is exactly what you pass to ``predict(..., assay_ids=[...])`` and
    exactly what appears as a key in the returned :class:`OraclePrediction`. It is **not** necessarily
    the id under which a background row is stored — chrombpnet emits per-strand
    ``CHIP:cell:TF:+`` while its null is strand-merged, and legnet emits ``LentiMPRA:HepG2`` against a
    null keyed bare ``HepG2``. ``chorus.analysis.normalization.PerTrackNormalizer._match_track_id``
    is the bridge; :attr:`has_background` records the outcome of asking it.
    """

    #: What you pass to ``predict()``. Unique within one oracle.
    track_id: str

    #: Assay or output type, in whatever vocabulary the oracle uses (``"DNASE"``, ``"H3K27ac"``,
    #: ``"sequence-class"``, ``"LentiMPRA"``). Deliberately not normalised here — the layer mapping
    #: lives in ``chorus.analysis.scorers.classify_track_layer``, which is the one place that decides
    #: what an assay means.
    assay: Optional[str] = None

    #: Biosample / cell type, where the oracle has one. ``None`` for oracles whose tracks are not
    #: cell-type specific (sei's sequence classes, for instance).
    cell_type: Optional[str] = None

    #: Human-readable label, when the oracle's metadata carries one.
    description: Optional[str] = None

    #: Whether a background row exists for this track, i.e. whether a percentile is available.
    #: ``None`` means "not checked" — distinct from ``False``, which means checked and absent.
    #:
    #: This field exists because Sei made the distinction unavoidable: it predicts 21,907 chromatin
    #: profiles that for a long time had no null at all, so ``predict()`` returned a real value whose
    #: percentile was silently ``None``. Rather than hide that, a caller can now see it.
    has_background: Optional[bool] = None

    #: Anything oracle-specific worth surfacing without inventing a field for it (fold, ENCODE
    #: accession, model filename, …).
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form, for MCP payloads and subprocess boundaries."""
        return asdict(self)

    def matches(self, query: str) -> bool:
        """Case-insensitive substring search over the fields a human would search.

        Used by the default filtering in ``OracleBase.describe_tracks`` so every oracle gets the same
        search semantics rather than nine hand-rolled ones.
        """
        if not query:
            return True
        q = query.lower()
        return any(
            q in str(v).lower()
            for v in (self.track_id, self.assay, self.cell_type, self.description)
            if v is not None
        )
