"""A generic catalog of every annotation chorus knows about.

Three registries already exist, each with its own shape and no shared interface:

- :mod:`chorus.analysis.conservation` — GPN-Star / PhyloP / PhastCons bigwig tracks,
  downloaded from HuggingFace or UCSC.
- :class:`chorus.utils.annotations.AnnotationManager` — GENCODE GTF gene annotations,
  downloaded over plain HTTP.
- Nothing, for a user's own custom annotation.

``AnnotationStore`` does not replace either of the first two — it reads them at call
time (:func:`list_annotations`, :func:`download_annotation` delegate straight through)
and adds a third, user-editable source: a flat YAML file of custom entries, plus the
methods to list/describe/download/add/remove across all three uniformly.

Every entry records a declared ``genome_build``. For bigwig-format annotations this is
physically verified against the file's own chromosome-1 length (see
:func:`chorus.utils.genome.require_assembly_for_bigwig`) rather than trusted blindly —
a wrong build produces a plausible-looking answer about the wrong DNA, not a crash, so
:func:`describe_annotation` raises rather than warns on a confident mismatch. GTF/BED/
other formats have no comparably reliable chromosome-length header, so their declared
genome is recorded as metadata only.
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

from .annotations import AnnotationManager
from .genome import ASSEMBLY_CHR1_LENGTH

logger = logging.getLogger(__name__)

#: Conservation tracks carry no description in ``conservation._TRACK_SOURCES`` (it's a
#: bare download-dispatch table), so descriptions live here instead.
_CONSERVATION_DESCRIPTIONS = {
    "gpn_star": "GPN-Star entropy conservation score (hg38)",
    "gpn_star_llr_a": "GPN-Star calibrated log-likelihood ratio, base A (hg38)",
    "gpn_star_llr_c": "GPN-Star calibrated log-likelihood ratio, base C (hg38)",
    "gpn_star_llr_g": "GPN-Star calibrated log-likelihood ratio, base G (hg38)",
    "gpn_star_llr_t": "GPN-Star calibrated log-likelihood ratio, base T (hg38)",
    "phylop20way": "UCSC PhyloP 20-way conservation score (hg38)",
    "phastcons7way": "UCSC PhastCons 7-way conservation score (hg38)",
}

#: Every conservation track shipped today is hg38-only (see conservation.py's module
#: docstring) — conservation.py itself never records this per-track, so it's fixed here.
_CONSERVATION_GENOME_BUILD = "hg38"


@dataclasses.dataclass(frozen=True)
class AnnotationEntry:
    """One row of the merged annotation catalog."""

    id: str
    origin: str  # "conservation" | "gtf" | "custom"
    description: str
    genome_build: Optional[str]
    format: Optional[str]  # "bigwig" | "gtf" | "bed" | "other" | None
    downloaded: bool
    path: Optional[Path]
    size_bytes: Optional[int]
    size_note: Optional[str]
    source: dict
    verified_genome_build: Optional[str] = None
    warning: Optional[str] = None

    def as_dict(self) -> dict:
        """JSON-safe dict, for CLI printing and MCP tool returns."""
        d = dataclasses.asdict(self)
        if d.get("path") is not None:
            d["path"] = str(d["path"])
        return d


class AnnotationStore:
    """Unified list/describe/download/add interface over every annotation source."""

    def __init__(
        self,
        *,
        annotations_dir: Optional[Union[str, Path]] = None,
        downloads_dir: Optional[Union[str, Path]] = None,
        annotation_manager: Optional[AnnotationManager] = None,
    ) -> None:
        from ..core.globals import CHORUS_ANNOTATIONS_DIR, CHORUS_DOWNLOADS_DIR

        self.annotations_dir = Path(annotations_dir) if annotations_dir else Path(CHORUS_ANNOTATIONS_DIR)
        self.downloads_dir = Path(downloads_dir) if downloads_dir else Path(CHORUS_DOWNLOADS_DIR)
        self._annotation_manager = annotation_manager or AnnotationManager(self.annotations_dir)
        self._custom_yaml_path = self.annotations_dir / "custom_annotations.yaml"

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_annotations(self) -> List[AnnotationEntry]:
        """Every known annotation, merged from all three sources.

        Pure filesystem/metadata check — never triggers a download. Returns the
        full list; capping/truncation for display is a caller (CLI/MCP) concern.
        """
        return [*self._conservation_entries(), *self._gtf_entries(), *self._custom_entries()]

    def _conservation_entries(self) -> List[AnnotationEntry]:
        from chorus.analysis import conservation

        entries = []
        info = conservation.list_tracks(self.downloads_dir)
        for track_id, status in info.items():
            cfg = dict(conservation._TRACK_SOURCES.get(track_id, {}))
            entries.append(AnnotationEntry(
                id=track_id,
                origin="conservation",
                description=_CONSERVATION_DESCRIPTIONS.get(track_id, f"Conservation track {track_id}"),
                genome_build=_CONSERVATION_GENOME_BUILD,
                format="bigwig",
                downloaded=status["downloaded"],
                path=status["path"] if status["downloaded"] else None,
                size_bytes=status["size_bytes"],
                size_note=status["size_note"],
                source=cfg,
            ))
        return entries

    def _gtf_entries(self) -> List[AnnotationEntry]:
        entries = []
        for ann_id, meta in self._annotation_manager.list_annotations().items():
            path = meta.get("path")
            downloaded = bool(meta.get("downloaded"))
            size_bytes = None
            if downloaded and path:
                try:
                    size_bytes = Path(path).stat().st_size
                except OSError:
                    size_bytes = None
            genome_build = meta.get("genome")
            entries.append(AnnotationEntry(
                id=ann_id,
                origin="gtf",
                description=meta.get("description", f"GTF annotation {ann_id}"),
                genome_build=None if genome_build == "unknown" else genome_build,
                format="gtf",
                downloaded=downloaded,
                path=Path(path) if downloaded and path else None,
                size_bytes=size_bytes,
                size_note=None,
                source={k: v for k, v in meta.items() if k not in ("downloaded", "path")},
            ))
        return entries

    def _custom_entries(self) -> List[AnnotationEntry]:
        entries = []
        custom = self._load_custom_yaml().get("annotations", {})
        for ann_id, meta in custom.items():
            path = self._custom_local_path(ann_id, meta)
            downloaded = path is not None and path.exists()
            size_bytes = path.stat().st_size if downloaded else None
            entries.append(AnnotationEntry(
                id=ann_id,
                origin="custom",
                description=meta.get("description", f"Custom annotation {ann_id}"),
                genome_build=meta.get("genome_build"),
                format=meta.get("format"),
                downloaded=downloaded,
                path=path if downloaded else None,
                size_bytes=size_bytes,
                size_note=None,
                source=dict(meta),
            ))
        return entries

    # ------------------------------------------------------------------
    # Describe (with physical genome-build verification for bigwigs)
    # ------------------------------------------------------------------

    def describe_annotation(self, annotation_id: str) -> AnnotationEntry:
        """Full metadata for one annotation.

        If it's downloaded and its format is ``"bigwig"``, physically verifies
        ``genome_build`` against the file's own chr1 length. A confident mismatch
        raises :class:`~chorus.core.exceptions.GenomeAssemblyMismatchError` — this
        is intentional; the whole point of this check is to not swallow it.
        """
        by_id = {e.id: e for e in self.list_annotations()}
        if annotation_id not in by_id:
            raise ValueError(
                f"Unknown annotation: {annotation_id!r}. Valid: {sorted(by_id)}"
            )
        entry = by_id[annotation_id]
        if entry.downloaded and entry.path is not None and entry.format == "bigwig":
            verified, warning = self._verify_bigwig(
                entry.path, entry.genome_build, context=f"annotation {annotation_id!r}"
            )
            entry = dataclasses.replace(entry, verified_genome_build=verified, warning=warning)
        return entry

    @staticmethod
    def _verify_bigwig(path: Path, genome_build: Optional[str], *, context: str):
        if not genome_build:
            return None, "no declared genome_build to verify against"
        from .genome import require_assembly_for_bigwig

        found = require_assembly_for_bigwig(path, genome_build, context=context)
        if found is None:
            return None, "assembly unrecognized from bigwig header; proceeding unverified"
        return found, None

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_annotation(self, annotation_id: str) -> Path:
        """Ensure *annotation_id* is on disk locally; return its path.

        Delegates to :func:`conservation.download_track`/
        :meth:`AnnotationManager.download_annotation` for builtin entries (no
        duplicate download-dispatch logic); dispatches custom entries by
        ``kind`` (``hf``/``url``/``local``).
        """
        from chorus.analysis import conservation

        if annotation_id in conservation._TRACK_SOURCES:
            return conservation.download_track(annotation_id, self.downloads_dir)

        if annotation_id in self._annotation_manager.ANNOTATION_SOURCES:
            return Path(self._annotation_manager.download_annotation(annotation_id))

        gtf_info = self._annotation_manager.list_annotations()
        if annotation_id in gtf_info:
            # A loose GTF file AnnotationManager found on disk by globbing, not a
            # registered source — nothing to download, it's already there.
            return Path(gtf_info[annotation_id]["path"])

        custom = self._load_custom_yaml().get("annotations", {})
        if annotation_id in custom:
            return self._download_custom(annotation_id, custom[annotation_id])

        raise ValueError(
            f"Unknown annotation: {annotation_id!r}. Run list_annotations() to see valid ids."
        )

    def _download_custom(self, annotation_id: str, meta: dict) -> Path:
        kind = meta.get("kind")
        local_path = self._custom_local_path(annotation_id, meta)

        if kind == "local":
            if local_path is None or not local_path.exists():
                raise FileNotFoundError(
                    f"Custom annotation {annotation_id!r} local_path does not exist: {local_path}"
                )
            return local_path

        if local_path.exists():
            return local_path

        local_path.parent.mkdir(parents=True, exist_ok=True)

        if kind == "hf":
            from huggingface_hub import hf_hub_download

            revision = meta["hf_revision"]
            logger.info(
                "Downloading custom annotation %s (%s, revision=%s) from HuggingFace...",
                annotation_id, meta["hf_filename"], revision,
            )
            downloaded = Path(hf_hub_download(
                meta["hf_repo"],
                filename=meta["hf_filename"],
                repo_type="dataset",
                revision=revision,
                local_dir=str(local_path.parent),
            ))
            if downloaded != local_path:
                shutil.move(str(downloaded), str(local_path))
                nested_dir = downloaded.parent
                while nested_dir != local_path.parent and nested_dir.exists() and not any(nested_dir.iterdir()):
                    emptied = nested_dir
                    nested_dir = nested_dir.parent
                    emptied.rmdir()
        elif kind == "url":
            from .http import download_with_resume

            logger.info("Downloading custom annotation %s from %s...", annotation_id, meta["url"])
            download_with_resume(meta["url"], local_path, label=annotation_id)
        else:
            raise ValueError(f"Unknown custom annotation source kind: {kind!r}")

        logger.info("%s custom annotation cached at %s", annotation_id, local_path)
        return local_path

    # ------------------------------------------------------------------
    # Add / remove custom entries
    # ------------------------------------------------------------------

    def add_annotation(
        self,
        annotation_id: str,
        *,
        description: str,
        genome_build: str,
        format: Optional[str] = None,
        hf_repo: Optional[str] = None,
        hf_filename: Optional[str] = None,
        hf_revision: Optional[str] = None,
        url: Optional[str] = None,
        local_path: Optional[Union[str, Path]] = None,
        local_filename: Optional[str] = None,
        overwrite: bool = False,
    ) -> AnnotationEntry:
        """Register a new custom annotation and persist it to ``custom_annotations.yaml``.

        Exactly one of ``(hf_repo & hf_filename)``, ``url``, or ``local_path`` must be
        given. An HF source requires an explicit ``hf_revision`` (not ``main``/
        ``master``/``HEAD``) so the download stays reproducible. ``genome_build`` must
        be a name chorus can identify (a typo-catch, not yet a physical check). If the
        annotation is already on disk (``local_path``, or a previously-cached download)
        and looks like a bigwig, its declared ``genome_build`` is verified immediately
        against the file's own chr1 length — this can raise
        :class:`~chorus.core.exceptions.GenomeAssemblyMismatchError`.
        """
        n_sources = sum([bool(hf_repo or hf_filename), bool(url), bool(local_path)])
        if n_sources != 1:
            raise ValueError(
                "add_annotation requires exactly one source: (hf_repo & hf_filename), url, or local_path."
            )

        if hf_repo or hf_filename:
            if not (hf_repo and hf_filename):
                raise ValueError("hf_repo and hf_filename must both be given together.")
            if not hf_revision or hf_revision.strip().lower() in ("main", "master", "head"):
                raise ValueError(
                    "hf_revision is required and must not be 'main'/'master'/'HEAD' -- pin an "
                    "explicit tag or commit so the download stays reproducible."
                )
            kind = "hf"
        elif url:
            kind = "url"
        else:
            kind = "local"
            local_path = str(Path(local_path).expanduser().resolve())

        if genome_build not in ASSEMBLY_CHR1_LENGTH:
            raise ValueError(
                f"genome_build={genome_build!r} is not an assembly chorus can identify; "
                f"known: {sorted(ASSEMBLY_CHR1_LENGTH)}."
            )

        from chorus.analysis import conservation

        if annotation_id in conservation._TRACK_SOURCES or annotation_id in self._annotation_manager.ANNOTATION_SOURCES:
            raise ValueError(f"{annotation_id!r} is a builtin annotation id and cannot be overridden.")

        data = self._load_custom_yaml()
        existing = data.setdefault("annotations", {})
        if annotation_id in existing and not overwrite:
            raise ValueError(
                f"{annotation_id!r} already exists as a custom annotation. Pass overwrite=True to replace it."
            )

        if format is None:
            format = self._infer_format(local_filename or hf_filename or url or local_path)

        entry_meta = {
            "description": description,
            "genome_build": genome_build,
            "format": format,
            "kind": kind,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
        if kind == "hf":
            entry_meta.update(hf_repo=hf_repo, hf_filename=hf_filename, hf_revision=hf_revision)
            if local_filename:
                entry_meta["local_filename"] = local_filename
        elif kind == "url":
            entry_meta["url"] = url
            if local_filename:
                entry_meta["local_filename"] = local_filename
        else:
            entry_meta["local_path"] = local_path

        resolved_path = self._custom_local_path(annotation_id, entry_meta)
        if format == "bigwig" and resolved_path is not None and resolved_path.exists():
            from .genome import require_assembly_for_bigwig

            require_assembly_for_bigwig(
                resolved_path, genome_build, context=f"annotation {annotation_id!r}"
            )  # raises GenomeAssemblyMismatchError on a confident mismatch

        existing[annotation_id] = entry_meta
        self._save_custom_yaml(data)

        return self.describe_annotation(annotation_id)

    def remove_custom_annotation(self, annotation_id: str, *, delete_file: bool = False) -> None:
        """Remove a custom entry. Raises ``ValueError`` for a non-custom (or unknown) id."""
        data = self._load_custom_yaml()
        existing = data.get("annotations", {})
        if annotation_id not in existing:
            raise ValueError(
                f"{annotation_id!r} is not a custom annotation (builtin entries cannot be removed)."
            )
        meta = existing.pop(annotation_id)
        self._save_custom_yaml(data)

        if delete_file:
            path = self._custom_local_path(annotation_id, meta)
            if path is not None and path.exists():
                try:
                    path.unlink()
                except OSError as exc:
                    logger.warning(
                        "Could not delete %s for removed custom annotation %s: %s", path, annotation_id, exc
                    )

    # ------------------------------------------------------------------
    # Custom-entries flat file
    # ------------------------------------------------------------------

    def _load_custom_yaml(self) -> dict:
        if not self._custom_yaml_path.exists():
            return {"version": 1, "annotations": {}}
        import yaml

        with open(self._custom_yaml_path) as f:
            data = yaml.safe_load(f) or {}
        data.setdefault("version", 1)
        data.setdefault("annotations", {})
        return data

    def _save_custom_yaml(self, data: dict) -> None:
        import yaml

        self._custom_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._custom_yaml_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    def _custom_dir(self, annotation_id: str) -> Path:
        return self.annotations_dir / "custom" / annotation_id

    def _custom_local_path(self, annotation_id: str, meta: dict) -> Optional[Path]:
        kind = meta.get("kind")
        if kind == "local":
            local_path = meta.get("local_path")
            return Path(local_path) if local_path else None
        filename = meta.get("local_filename") or self._default_local_filename(meta)
        if not filename:
            return None
        return self._custom_dir(annotation_id) / filename

    @staticmethod
    def _default_local_filename(meta: dict) -> Optional[str]:
        if meta.get("kind") == "hf" and meta.get("hf_filename"):
            return Path(meta["hf_filename"]).name
        if meta.get("kind") == "url" and meta.get("url"):
            name = Path(urlparse(meta["url"]).path).name
            return name or None
        return None

    @staticmethod
    def _infer_format(name_hint: Optional[str]) -> str:
        if not name_hint:
            return "other"
        parsed = urlparse(str(name_hint))
        raw = parsed.path if parsed.scheme else str(name_hint)
        name = Path(raw).name.lower()
        if name.endswith(".bw") or name.endswith(".bigwig"):
            return "bigwig"
        if name.endswith(".gtf") or name.endswith(".gtf.gz"):
            return "gtf"
        if name.endswith(".bed") or name.endswith(".bed.gz"):
            return "bed"
        return "other"


_store: Optional[AnnotationStore] = None


def get_annotation_store() -> AnnotationStore:
    """The global :class:`AnnotationStore` instance."""
    global _store
    if _store is None:
        _store = AnnotationStore()
    return _store
