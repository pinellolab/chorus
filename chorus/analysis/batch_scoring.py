"""Batch variant scoring — process multiple variants and produce ranked results.

Scores each variant via build_variant_report, extracts aggregate per-layer
scores, and returns a ranked table sorted by maximum effect magnitude.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .analysis_request import AnalysisRequest
from .normalization import QuantileNormalizer
from .variant_report import build_variant_report, _describe_normalizer

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
    analysis_request: AnalysisRequest | None = None

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

    def to_html(self, output_path: str | None = None) -> str:
        """Render the ranked results as a self-contained HTML page.

        Matches the output-format set of the other analysis tools (MD,
        JSON, TSV, HTML) so ``batch_scoring/`` is not the odd one out.
        The HTML includes the Analysis Request header and a sortable
        ranking table.
        """
        import html as _html

        parts: list[str] = [
            "<!DOCTYPE html><html lang='en'><head>",
            "<meta charset='utf-8'>",
            f"<title>Batch Variant Scoring — {len(self.scores)} variants</title>",
            "<style>",
            "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "max-width:1100px;margin:24px auto;padding:0 16px;color:#24292e;}",
            "h1{font-size:1.4em;margin-top:0;}",
            "table{border-collapse:collapse;width:100%;margin-top:12px;font-size:0.9em;}",
            "th,td{border:1px solid #d0d7de;padding:6px 10px;text-align:left;}",
            "th{background:#f6f8fa;}",
            "tr:nth-child(even){background:#fafbfc;}",
            ".pos{color:#cf222e;font-weight:600;}.neg{color:#116329;font-weight:600;}",
            "</style></head><body>",
            "<h1>Batch Variant Scoring Results</h1>",
        ]
        if self.analysis_request is not None:
            parts.append(self.analysis_request.to_html_fragment())

        parts.append(f"<p><b>{len(self.scores)} variants scored</b></p>")

        has_quantile = any(s.max_quantile is not None for s in self.scores)
        cols = ["Rank", "Variant", "ID", "Gene", "Cell Type", "Max Effect"]
        if has_quantile:
            cols.append("Quantile")
        cols += ["Top Layer", "Top Track"]

        parts.append("<table><thead><tr>")
        for c in cols:
            parts.append(f"<th>{_html.escape(c)}</th>")
        parts.append("</tr></thead><tbody>")

        for i, s in enumerate(self.scores, 1):
            var_str = f"{s.chrom}:{s.position} {s.ref}>{s.alt}"
            sign = "+" if s.max_effect >= 0 else ""
            effect_class = "pos" if s.max_effect >= 0 else "neg"
            row = [
                f"<td>{i}</td>",
                f"<td>{_html.escape(var_str)}</td>",
                f"<td>{_html.escape(s.variant_id)}</td>",
                f"<td>{_html.escape(s.gene_name or '—')}</td>",
                f"<td>{_html.escape(s.cell_type or '—')}</td>",
                f"<td class='{effect_class}'>{sign}{s.max_effect:.3f}</td>",
            ]
            if has_quantile:
                q = f"{s.max_quantile:.2f}" if s.max_quantile is not None else "—"
                row.append(f"<td>{q}</td>")
            row.append(f"<td>{_html.escape(s.top_layer)}</td>")
            row.append(f"<td>{_html.escape(s.top_track)}</td>")
            parts.append("<tr>" + "".join(row) + "</tr>")

        parts.append("</tbody></table>")
        parts.append("</body></html>")
        html = "\n".join(parts)

        if output_path is not None:
            from pathlib import Path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(html, encoding="utf-8")
            logger.info("Batch HTML report written to %s", output_path)
        return html

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
        lines: list[str] = []
        if self.analysis_request is not None:
            lines.append(self.analysis_request.to_markdown())
        lines += [
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
        out: dict = {
            "num_variants": len(self.scores),
        }
        if self.analysis_request is not None:
            out["analysis_request"] = self.analysis_request.to_dict()
        out["scores"] = [
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
        ]
        return out


def score_variant_batch(
    oracle,
    variants: list[dict],
    assay_ids: list[str] | None,
    gene_name: str | None = None,
    normalizer: QuantileNormalizer | None = None,
    analysis_request: AnalysisRequest | None = None,
) -> BatchResult:
    """Score multiple variants and rank by effect magnitude.

    Each entry in ``variants`` must be a dict with keys:

    - ``chrom`` (str, e.g. "chr1")
    - ``pos`` (int, 1-based genomic coordinate)
    - ``ref`` (str, reference allele)
    - ``alt`` (str, alternate allele)
    - ``id`` (str, optional — used as the label; defaults to
      ``chrom:pos_ref>alt``)

    Args:
        oracle: A loaded Chorus oracle.
        variants: List of variant dicts (schema above).
        assay_ids: List of track identifiers to score. Pass ``None`` to let
            the oracle score all available tracks (recommended for
            AlphaGenome and other multi-track oracles).
        gene_name: Optional gene name for expression scoring.
        normalizer: Optional quantile normalizer.
        analysis_request: Optional :class:`AnalysisRequest` carrying the
            user's original prompt and context; rendered at the top of
            every report produced from this batch.

    Returns:
        A :class:`BatchResult` with variants ranked by |max_effect|.
        Supports ``to_markdown()``, ``to_tsv()``, ``to_dict()``, and
        ``to_dataframe()``.
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

            # Find the cell type of the single strongest-effect track — that's
            # the biologically meaningful tissue context for this variant.
            # Collecting *all* cell types (thousands on AlphaGenome) blows up
            # the markdown table and is useless for ranking.
            top_cell_type = ""

            for ts in report.allele_scores.get(allele_key, []):
                if ts.raw_score is None:
                    continue
                abs_score = abs(ts.raw_score)
                if abs_score > abs(max_effect):
                    max_effect = ts.raw_score
                    top_layer = ts.layer
                    top_track = ts.assay_id
                    top_cell_type = ts.cell_type or ""

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
                cell_type=top_cell_type,
                max_quantile=max_quantile,
            ))

        except Exception as exc:
            logger.error("Failed to score variant %s: %s", vid, exc, exc_info=True)
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

    # Synthesize a minimal AnalysisRequest if the caller didn't supply one
    if analysis_request is None:
        analysis_request = AnalysisRequest(
            tool_name="score_variant_batch",
            oracle_name=getattr(oracle, "name", None),
            normalizer_name=_describe_normalizer(normalizer),
            tracks_requested=(
                "all oracle tracks" if not assay_ids else f"{len(assay_ids)} tracks"
            ),
        )
    else:
        if analysis_request.oracle_name is None:
            analysis_request.oracle_name = getattr(oracle, "name", None)
        if analysis_request.normalizer_name is None:
            analysis_request.normalizer_name = _describe_normalizer(normalizer)

    return BatchResult(scores=batch_scores, analysis_request=analysis_request)
