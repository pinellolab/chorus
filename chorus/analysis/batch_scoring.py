"""Batch variant scoring — process multiple variants and produce ranked results.

Scores each variant via build_variant_report, preserves **per-track** scores
(raw + percentile + activity percentile), and returns a ranked table.

Two display modes for the rendered output:

- **by_assay** (default): one column per assay type, all in the same cell type.
  Best when the user has a fixed cell type and wants to compare across assays.
- **by_cell_type**: one column per cell type, all for the same assay.
  Best when the user has a fixed assay and wants to compare across tissues.

In both modes, every cell shows the raw effect score and its effect percentile.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .analysis_request import AnalysisRequest
from .normalization import QuantileNormalizer
from .variant_report import TrackScore, build_variant_report, _describe_normalizer

logger = logging.getLogger(__name__)


@dataclass
class BatchVariantScore:
    """Per-variant scores including individual track-level detail."""

    chrom: str
    position: int
    ref: str
    alt: str
    variant_id: str

    # Aggregate
    max_effect: float = 0.0
    top_layer: str = ""
    top_track: str = ""
    per_layer_scores: dict = field(default_factory=dict)
    gene_name: str = ""
    cell_type: str = ""
    max_quantile: float | None = None

    # Per-track detail — keyed by assay_id
    track_scores: dict[str, TrackScore] = field(default_factory=dict)


def _track_display_name(ts: TrackScore) -> str:
    """Human-readable short name for a track (e.g. 'DNASE:HepG2').

    When two tracks share the same description (e.g. CAGE+/CAGE- have
    `description='CAGE:HepG2'` but different assay_ids ending in `/+` vs
    `/-`), a strand suffix is appended so table columns remain unique.
    """
    desc = ts.description or ""
    if not desc and ts.assay_type and ts.cell_type:
        desc = f"{ts.assay_type}:{ts.cell_type}"
    if not desc:
        return ts.assay_id
    # Append strand suffix when the assay_id carries one (common for
    # CAGE, PRO-CAP, splicing) to distinguish otherwise-identical labels.
    aid = ts.assay_id or ""
    if aid.endswith("/+"):
        return f"{desc} (+)"
    if aid.endswith("/-"):
        return f"{desc} (-)"
    return desc


@dataclass
class BatchResult:
    """Ranked results from batch variant scoring."""

    scores: list[BatchVariantScore]
    analysis_request: AnalysisRequest | None = None

    # ── Per-track dataframe ─────────────────────────────────────────────

    def to_dataframe(self):
        """One row per variant. Per-track columns with ``_ref``, ``_alt``, ``_log2fc``,
        ``_effect_pctile``, and ``_activity_pctile`` suffixes."""
        import pandas as pd

        # Collect the union of all track IDs across all variants
        all_track_ids: list[str] = []
        seen: set[str] = set()
        for s in self.scores:
            for tid in s.track_scores:
                if tid not in seen:
                    all_track_ids.append(tid)
                    seen.add(tid)

        rows = []
        for s in self.scores:
            row: dict = {
                "variant_id": s.variant_id,
                "chrom": s.chrom,
                "position": s.position,
                "ref": s.ref,
                "alt": s.alt,
                "gene_name": s.gene_name,
                "max_effect": s.max_effect,
                "max_quantile": s.max_quantile,
            }
            for tid in all_track_ids:
                ts = s.track_scores.get(tid)
                display = _track_display_name(ts) if ts else tid
                if ts and ts.raw_score is not None:
                    row[f"{display}_ref"] = ts.ref_value
                    row[f"{display}_alt"] = ts.alt_value
                    row[f"{display}_log2fc"] = ts.raw_score
                    row[f"{display}_effect_pctile"] = ts.quantile_score
                    row[f"{display}_activity_pctile"] = ts.ref_signal_percentile
                else:
                    row[f"{display}_ref"] = None
                    row[f"{display}_alt"] = None
                    row[f"{display}_log2fc"] = None
                    row[f"{display}_effect_pctile"] = None
                    row[f"{display}_activity_pctile"] = None
            rows.append(row)
        return pd.DataFrame(rows)

    # ── TSV ─────────────────────────────────────────────────────────────

    def to_tsv(self, output_path: str | None = None) -> str:
        df = self.to_dataframe()
        tsv = df.to_csv(sep="\t", index=False)
        if output_path is not None:
            from pathlib import Path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(tsv, encoding="utf-8")
        return tsv

    # ── Markdown ────────────────────────────────────────────────────────

    def to_markdown(self, display_mode: str = "by_assay") -> str:
        """Render a markdown table.

        Args:
            display_mode: ``"by_assay"`` (default) — one column per assay,
                ``"by_cell_type"`` — one column per cell type,
                ``"summary"`` — legacy aggregate table.
        """
        lines: list[str] = []
        if self.analysis_request is not None:
            lines.append(self.analysis_request.to_markdown())
        lines += ["## Batch Variant Scoring Results", "",
                   f"**{len(self.scores)} variants scored**", ""]

        if display_mode == "summary" or not self._has_track_scores():
            return self._markdown_summary(lines)
        return self._markdown_per_track(lines, display_mode)

    def _has_track_scores(self) -> bool:
        return any(s.track_scores for s in self.scores)

    def _markdown_summary(self, lines: list[str]) -> str:
        """Legacy aggregate table."""
        has_q = any(s.max_quantile is not None for s in self.scores)
        header = "| Rank | Variant | ID | Max Effect |"
        sep = "|------|---------|-----|-----------|"
        if has_q:
            header += " Effect %ile |"
            sep += "------------|"
        header += " Top Layer | Top Track |"
        sep += "-----------|-----------|"
        lines += [header, sep]
        for i, s in enumerate(self.scores, 1):
            sign = "+" if s.max_effect >= 0 else ""
            row = f"| {i} | {s.chrom}:{s.position} {s.ref}>{s.alt} | {s.variant_id} | {sign}{s.max_effect:.3f} |"
            if has_q:
                q = f"{s.max_quantile:.2f}" if s.max_quantile is not None else "—"
                row += f" {q} |"
            row += f" {s.top_layer} | {s.top_track} |"
            lines.append(row)
        lines.append("")
        return "\n".join(lines)

    def _markdown_per_track(self, lines: list[str], mode: str) -> str:
        """Per-track table: each column is one assay (by_assay) or one cell type (by_cell_type)."""
        # Build ordered column list from all variants' track_scores
        col_order: list[str] = []  # assay_ids in order
        col_labels: dict[str, str] = {}  # assay_id → display name
        seen: set[str] = set()
        for s in self.scores:
            for tid, ts in s.track_scores.items():
                if tid not in seen:
                    col_order.append(tid)
                    col_labels[tid] = _track_display_name(ts)
                    seen.add(tid)

        # Header — one group of columns per track: Ref | Alt | log2FC | %ile
        header = "| Variant | ID |"
        sep = "|---------|-----|"
        for tid in col_order:
            label = col_labels[tid]
            header += f" {label} Ref | {label} Alt | {label} log2FC | {label} Effect %ile |"
            sep += "---|---|---|---|"
        lines += [header, sep]

        # Rows
        for s in self.scores:
            row = f"| {s.chrom}:{s.position} {s.ref}>{s.alt} | {s.variant_id} |"
            for tid in col_order:
                ts = s.track_scores.get(tid)
                if ts and ts.raw_score is not None:
                    ref_str = f"{ts.ref_value:.3g}" if ts.ref_value is not None else "—"
                    alt_str = f"{ts.alt_value:.3g}" if ts.alt_value is not None else "—"
                    sign = "+" if ts.raw_score >= 0 else ""
                    fc_str = f"{sign}{ts.raw_score:.3f}"
                    from chorus.analysis.variant_report import _fmt_percentile
                    pct_str = _fmt_percentile(ts.quantile_score)
                    row += f" {ref_str} | {alt_str} | {fc_str} | {pct_str} |"
                else:
                    row += " — | — | — | — |"
            lines.append(row)

        # Score guide + track ID footnote
        lines += [
            "",
            "Each track shows: **Ref** (reference allele prediction), **Alt** (alternate allele prediction), "
            "**log2FC** (log2 fold-change alt/ref), **Effect %ile** (ranked against ~10K random SNPs).",
            "",
            "**Track identifiers** (for tracing back to oracle data):",
            "",
        ]
        for tid in col_order:
            lines.append(f"- {col_labels[tid]}: `{tid}`")
        lines.append("")
        return "\n".join(lines)

    # ── HTML ────────────────────────────────────────────────────────────

    def to_html(self, output_path: str | None = None,
                display_mode: str = "by_assay") -> str:
        import html as _html

        parts: list[str] = [
            "<!DOCTYPE html><html lang='en'><head>",
            "<meta charset='utf-8'>",
            f"<title>Batch Variant Scoring — {len(self.scores)} variants</title>",
            "<style>",
            "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "max-width:1400px;margin:24px auto;padding:0 16px;color:#24292e;}",
            "h1{font-size:1.4em;margin-top:0;}",
            "table{border-collapse:collapse;width:100%;margin-top:12px;font-size:0.85em;}",
            "th,td{border:1px solid #d0d7de;padding:5px 8px;text-align:center;}",
            "th{background:#f6f8fa;font-size:0.85em;}",
            "td:first-child,td:nth-child(2){text-align:left;}",
            "tr:nth-child(even){background:#fafbfc;}",
            ".gain{background:#dcfce7;color:#166534;}",
            ".loss{background:#fee2e2;color:#991b1b;}",
            ".neutral{color:#6b7280;}",
            ".pctile{font-size:0.8em;color:#6b7280;}",
            "</style></head><body>",
            "<h1>Batch Variant Scoring Results</h1>",
        ]
        if self.analysis_request is not None:
            parts.append(self.analysis_request.to_html_fragment())
        parts.append(f"<p><b>{len(self.scores)} variants scored</b></p>")

        if self._has_track_scores() and display_mode != "summary":
            self._html_per_track(parts)
        else:
            self._html_summary(parts)

        parts += ["</body></html>"]
        html = "\n".join(parts)
        if output_path is not None:
            from pathlib import Path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(html, encoding="utf-8")
            logger.info("Batch HTML report written to %s", output_path)
        return html

    def _html_summary(self, parts: list[str]) -> None:
        import html as _html
        cols = ["Rank", "Variant", "ID", "Max Effect", "Top Layer", "Top Track"]
        parts.append("<table><thead><tr>")
        for c in cols:
            parts.append(f"<th>{_html.escape(c)}</th>")
        parts.append("</tr></thead><tbody>")
        for i, s in enumerate(self.scores, 1):
            sign = "+" if s.max_effect >= 0 else ""
            cls = "gain" if s.max_effect >= 0 else "loss"
            parts.append(
                f"<tr><td>{i}</td><td>{s.chrom}:{s.position} {s.ref}>{s.alt}</td>"
                f"<td>{_html.escape(s.variant_id)}</td>"
                f"<td class='{cls}'>{sign}{s.max_effect:.3f}</td>"
                f"<td>{_html.escape(s.top_layer)}</td>"
                f"<td>{_html.escape(s.top_track)}</td></tr>"
            )
        parts.append("</tbody></table>")

    def _html_per_track(self, parts: list[str]) -> None:
        import html as _html
        col_order: list[str] = []
        col_labels: dict[str, str] = {}
        seen: set[str] = set()
        for s in self.scores:
            for tid, ts in s.track_scores.items():
                if tid not in seen:
                    col_order.append(tid)
                    col_labels[tid] = _track_display_name(ts)
                    seen.add(tid)

        parts.append("<table><thead><tr>")
        parts.append("<th rowspan='2'>Variant</th><th rowspan='2'>ID</th>")
        for tid in col_order:
            parts.append(f"<th colspan='4'>{_html.escape(col_labels[tid])}</th>")
        parts.append("</tr><tr>")
        for _ in col_order:
            parts.append("<th>Ref</th><th>Alt</th><th>log2FC</th><th>Effect %ile</th>")
        parts.append("</tr></thead><tbody>")

        for s in self.scores:
            parts.append(
                f"<tr><td>{s.chrom}:{s.position} {s.ref}>{s.alt}</td>"
                f"<td>{_html.escape(s.variant_id)}</td>"
            )
            for tid in col_order:
                ts = s.track_scores.get(tid)
                if ts and ts.raw_score is not None:
                    ref_str = f"{ts.ref_value:.3g}" if ts.ref_value is not None else "—"
                    alt_str = f"{ts.alt_value:.3g}" if ts.alt_value is not None else "—"
                    sign = "+" if ts.raw_score >= 0 else ""
                    cls = "gain" if ts.raw_score > 0.1 else ("loss" if ts.raw_score < -0.1 else "neutral")
                    from chorus.analysis.variant_report import _fmt_percentile
                    pct_str = _fmt_percentile(ts.quantile_score)
                    parts.append(f"<td>{ref_str}</td><td>{alt_str}</td>"
                                 f"<td class='{cls}'>{sign}{ts.raw_score:.3f}</td>"
                                 f"<td>{pct_str}</td>")
                else:
                    parts.append("<td>—</td><td>—</td><td>—</td><td>—</td>")
            parts.append("</tr>")
        parts.append("</tbody></table>")
        parts.append("<p style='font-size:0.85em;color:#6b7280;margin-top:8px;'>"
                     "Columns per track: <b>Ref</b> (reference allele prediction), "
                     "<b>Alt</b> (alternate allele prediction), "
                     "<b>log2FC</b> (log2 fold-change alt/ref), "
                     "<b>Effect %ile</b> (ranked against ~10K random SNPs). "
                     "Green = gain, red = loss.</p>")

    # ── JSON ────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        out: dict = {"num_variants": len(self.scores)}
        if self.analysis_request is not None:
            out["analysis_request"] = self.analysis_request.to_dict()
        out["scores"] = []
        for s in self.scores:
            entry = {
                "chrom": s.chrom,
                "position": s.position,
                "ref": s.ref,
                "alt": s.alt,
                "variant_id": s.variant_id,
                "gene_name": s.gene_name,
                "max_effect": s.max_effect,
                "max_quantile": s.max_quantile,
                "top_layer": s.top_layer,
                "top_track": s.top_track,
                "per_layer_scores": s.per_layer_scores,
            }
            if s.track_scores:
                entry["track_scores"] = {
                    tid: {
                        "description": _track_display_name(ts),
                        "layer": ts.layer,
                        "cell_type": ts.cell_type,
                        "ref_value": ts.ref_value,
                        "alt_value": ts.alt_value,
                        "raw_score": ts.raw_score,
                        "quantile_score": ts.quantile_score,
                        "ref_signal_percentile": ts.ref_signal_percentile,
                    }
                    for tid, ts in s.track_scores.items()
                }
            out["scores"].append(entry)
        return out


# ── Scoring function ────────────────────────────────────────────────────────

def score_variant_batch(
    oracle,
    variants: list[dict],
    assay_ids: list[str] | None,
    gene_name: str | None = None,
    normalizer: QuantileNormalizer | None = None,
    analysis_request: AnalysisRequest | None = None,
    oracle_name: str | None = None,
) -> BatchResult:
    """Score multiple variants and rank by effect magnitude.

    Each entry in ``variants`` must be a dict with keys:

    - ``chrom`` (str, e.g. "chr1")
    - ``pos`` (int, 1-based genomic coordinate)
    - ``ref`` (str, reference allele)
    - ``alt`` (str, alternate allele)
    - ``id`` (str, optional — label; defaults to ``chrom:pos_ref>alt``)

    Args:
        oracle: A loaded Chorus oracle.
        variants: List of variant dicts (schema above).
        assay_ids: Track identifiers to score. Pass ``None`` for all tracks
            (not recommended for batch scoring — use specific tracks for
            interpretable per-track output).
        gene_name: Optional gene name for expression scoring.
        normalizer: Optional quantile normalizer.
        analysis_request: Optional :class:`AnalysisRequest`.

    Returns:
        :class:`BatchResult` with per-track scores for each variant.
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

            variant_result = oracle.predict_variant_effect(
                genomic_region=region,
                variant_position=position,
                alleles=[ref, alt],
                assay_ids=assay_ids,
            )

            _oname = oracle_name or getattr(oracle, "name", None) or oracle.__class__.__name__.lower().replace("oracle", "")
            report = build_variant_report(
                variant_result,
                oracle_name=_oname,
                gene_name=gene_name,
                normalizer=normalizer,
            )

            # Collect per-track scores and aggregates from the report
            max_effect = 0.0
            top_layer = ""
            top_track = ""
            top_cell_type = ""
            per_layer: dict[str, float] = {}
            max_quantile: float | None = None
            track_scores: dict[str, TrackScore] = {}

            allele_key = alt
            if allele_key not in report.allele_scores:
                keys = list(report.allele_scores)
                allele_key = keys[0] if keys else alt

            for ts in report.allele_scores.get(allele_key, []):
                # Store every track score
                track_scores[ts.assay_id] = ts

                if ts.raw_score is None:
                    continue
                abs_score = abs(ts.raw_score)
                if abs_score > abs(max_effect):
                    max_effect = ts.raw_score
                    top_layer = ts.layer
                    top_track = ts.assay_id
                    top_cell_type = ts.cell_type or ""
                if ts.quantile_score is not None:
                    if max_quantile is None or abs(ts.quantile_score) > abs(max_quantile):
                        max_quantile = ts.quantile_score
                layer_key = ts.layer
                if layer_key not in per_layer or abs_score > abs(per_layer[layer_key]):
                    per_layer[layer_key] = ts.raw_score

            batch_scores.append(BatchVariantScore(
                chrom=chrom, position=pos, ref=ref, alt=alt,
                variant_id=vid, max_effect=max_effect,
                top_layer=top_layer, top_track=top_track,
                per_layer_scores=per_layer,
                gene_name=report.gene_name or "",
                cell_type=top_cell_type,
                max_quantile=max_quantile,
                track_scores=track_scores,
            ))

        except Exception as exc:
            logger.error("Failed to score variant %s: %s", vid, exc, exc_info=True)
            batch_scores.append(BatchVariantScore(
                chrom=chrom, position=pos, ref=ref, alt=alt,
                variant_id=vid, top_layer="error", top_track=str(exc),
            ))

    batch_scores.sort(key=lambda s: abs(s.max_effect), reverse=True)

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
