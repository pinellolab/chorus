"""Batch variant scoring — process multiple variants and produce ranked results.

Scores each variant via build_variant_report, extracts aggregate per-layer
scores, and returns a ranked table sorted by maximum effect magnitude.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .normalization import QuantileNormalizer
from .variant_report import build_variant_report

logger = logging.getLogger(__name__)


@dataclass
class BatchVariantScore:
    """Aggregate score for a single variant across all tracks and layers."""

    chrom: str
    position: int
    ref: str
    alt: str
    variant_id: str
    max_effect: float          # max |raw_score| across all tracks/layers
    top_layer: str             # layer with the largest effect
    top_track: str             # track with the largest effect
    per_layer_scores: dict = field(default_factory=dict)  # {layer: max_score}
    gene_name: str = ""
    cell_type: str = ""
    max_quantile: float | None = None


@dataclass
class BatchResult:
    """Ranked results from batch variant scoring."""

    scores: list[BatchVariantScore]  # sorted by |max_effect| descending

    def to_dataframe(self):
        """Convert to a pandas DataFrame."""
        import pandas as pd

        rows = []
        for s in self.scores:
            row = {
                "chrom": s.chrom,
                "position": s.position,
                "ref": s.ref,
                "alt": s.alt,
                "variant_id": s.variant_id,
                "gene_name": s.gene_name,
                "cell_type": s.cell_type,
                "max_effect": s.max_effect,
                "max_quantile": s.max_quantile,
                "top_layer": s.top_layer,
                "top_track": s.top_track,
            }
            for layer, score in s.per_layer_scores.items():
                row[f"layer_{layer}"] = score
            rows.append(row)
        return pd.DataFrame(rows)

    def to_tsv(self, output_path: str | None = None) -> str:
        """Generate a TSV string of ranked variants.

        Args:
            output_path: If provided, write the TSV to this file path.

        Returns:
            TSV string with header row.
        """
        df = self.to_dataframe()
        tsv = df.to_csv(sep="\t", index=False)
        if output_path is not None:
            from pathlib import Path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(tsv, encoding="utf-8")
        return tsv

    def to_markdown(self) -> str:
        """Generate a markdown table of ranked variants."""
        has_quantile = any(s.max_quantile is not None for s in self.scores)
        lines = [
            "## Batch Variant Scoring Results",
            "",
            f"**{len(self.scores)} variants scored**",
            "",
        ]
        header = "| Rank | Variant | ID | Gene | Cell Type | Max Effect |"
        sep =    "|------|---------|-----|------|-----------|-----------|"
        if has_quantile:
            header += " Quantile |"
            sep += "----------|"
        header += " Top Layer | Top Track |"
        sep += "-----------|-----------|"
        lines.append(header)
        lines.append(sep)
        for i, s in enumerate(self.scores, 1):
            sign = "+" if s.max_effect >= 0 else ""
            var_str = f"{s.chrom}:{s.position} {s.ref}>{s.alt}"
            gene = s.gene_name or "—"
            ct = s.cell_type or "—"
            row = (
                f"| {i} | {var_str} | {s.variant_id} "
                f"| {gene} | {ct} "
                f"| {sign}{s.max_effect:.3f} |"
            )
            if has_quantile:
                q = f"{s.max_quantile:.2f}" if s.max_quantile is not None else "—"
                row += f" {q} |"
            row += f" {s.top_layer} | {s.top_track} |"
            lines.append(row)
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        return {
            "num_variants": len(self.scores),
            "scores": [
                {
                    "chrom": s.chrom,
                    "position": s.position,
                    "ref": s.ref,
                    "alt": s.alt,
                    "variant_id": s.variant_id,
                    "gene_name": s.gene_name,
                    "cell_type": s.cell_type,
                    "max_effect": s.max_effect,
                    "max_quantile": s.max_quantile,
                    "top_layer": s.top_layer,
                    "top_track": s.top_track,
                    "per_layer_scores": s.per_layer_scores,
                }
                for s in self.scores
            ],
        }


def score_variant_batch(
    oracle,
    variants: list[dict],
    assay_ids: list[str],
    gene_name: str | None = None,
    normalizer: QuantileNormalizer | None = None,
) -> BatchResult:
    """Score multiple variants and rank by effect magnitude.

    Args:
        oracle: A loaded Chorus oracle.
        variants: List of variant dicts, each with keys:
            ``chrom``, ``pos``, ``ref``, ``alt``, and optional ``id``.
        assay_ids: List of assay identifiers.
        gene_name: Optional gene name for expression scoring.
        normalizer: Optional quantile normalizer.

    Returns:
        A :class:`BatchResult` with variants ranked by |max_effect|.
    """
    batch_scores: list[BatchVariantScore] = []

    for i, var in enumerate(variants):
        chrom = var["chrom"]
        pos = int(var["pos"])
        ref = var["ref"]
        alt = var["alt"]
        vid = var.get("id", f"{chrom}:{pos}_{ref}>{alt}")

        logger.info("Scoring variant %d/%d: %s", i + 1, len(variants), vid)

        try:
            position = f"{chrom}:{pos}"
            region = f"{chrom}:{pos}-{pos + 1}"
            alleles = [ref, alt]

            variant_result = oracle.predict_variant_effect(
                genomic_region=region,
                variant_position=position,
                alleles=alleles,
                assay_ids=assay_ids,
            )

            report = build_variant_report(
                variant_result,
                oracle_name=getattr(oracle, "name", "oracle"),
                gene_name=gene_name,
                normalizer=normalizer,
            )

            # Extract aggregate scores from the report
            max_effect = 0.0
            top_layer = ""
            top_track = ""
            per_layer: dict[str, float] = {}
            max_quantile: float | None = None

            # Use first alt allele
            allele_key = alt
            if allele_key not in report.allele_scores:
                # Fallback: use first available allele key
                keys = [k for k in report.allele_scores]
                allele_key = keys[0] if keys else alt

            # Collect cell types from track scores
            var_cell_types: set[str] = set()

            for ts in report.allele_scores.get(allele_key, []):
                if ts.cell_type:
                    var_cell_types.add(ts.cell_type)
                if ts.raw_score is None:
                    continue
                abs_score = abs(ts.raw_score)
                if abs_score > abs(max_effect):
                    max_effect = ts.raw_score
                    top_layer = ts.layer
                    top_track = ts.assay_id

                # Track max quantile
                if ts.quantile_score is not None:
                    abs_q = abs(ts.quantile_score)
                    if max_quantile is None or abs_q > abs(max_quantile):
                        max_quantile = ts.quantile_score

                # Track max per layer
                layer_key = ts.layer
                if layer_key not in per_layer or abs_score > abs(per_layer[layer_key]):
                    per_layer[layer_key] = ts.raw_score

            batch_scores.append(BatchVariantScore(
                chrom=chrom,
                position=pos,
                ref=ref,
                alt=alt,
                variant_id=vid,
                max_effect=max_effect,
                top_layer=top_layer,
                top_track=top_track,
                per_layer_scores=per_layer,
                gene_name=report.gene_name or "",
                cell_type=", ".join(sorted(var_cell_types)),
                max_quantile=max_quantile,
            ))

        except Exception as exc:
            logger.error("Failed to score variant %s: %s", vid, exc)
            batch_scores.append(BatchVariantScore(
                chrom=chrom,
                position=pos,
                ref=ref,
                alt=alt,
                variant_id=vid,
                max_effect=0.0,
                top_layer="error",
                top_track=str(exc),
                per_layer_scores={},
            ))

    # Sort by |max_effect| descending
    batch_scores.sort(key=lambda s: abs(s.max_effect), reverse=True)

    return BatchResult(scores=batch_scores)
