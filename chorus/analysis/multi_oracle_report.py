"""Multi-oracle cross-validation reports.

A :class:`MultiOracleReport` groups the per-oracle :class:`VariantReport`
objects for the **same variant** scored by several oracles (e.g. ChromBPNet
for chromatin accessibility, LegNet for MPRA, AlphaGenome as a generalist
model) and renders a single cross-oracle validation HTML.

The report answers three questions a new user cares about when comparing
oracles on a classic causal variant:

1. Do the oracles **agree on direction** for each regulatory layer?
2. Which **specific assay / cell type** drove each oracle's headline call?
3. What are the **units** of each number (log2FC vs lnFC vs Δ), which may
   differ across layers even within a single oracle?

The cross-oracle "Consensus matrix" at the top of the page gives a fast
visual answer; the per-oracle collapsible sections below preserve the full
evidence trail and link out to each oracle's standalone report.
"""
from __future__ import annotations

import html as _html_mod
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .analysis_request import AnalysisRequest
from ._report_glossary import (
    HOW_TO_READ_CSS,
    formula_chip,
    formula_label,
    render_how_to_read,
)
from .scorers import LAYER_CONFIGS
from .variant_report import TrackScore, VariantReport, _CSS, _fmt_percentile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MultiOracleReport:
    """One variant, multiple oracles, one cross-validation view.

    Parameters
    ----------
    reports
        Mapping ``oracle_name -> VariantReport``. All reports must describe
        the same variant (same chrom/position/ref). Order of iteration
        matches insertion order and is preserved in the rendered HTML.
    variant_id
        Optional human-readable label (e.g. an rsID). If omitted we fall back
        to ``chrom:position``.
    per_oracle_report_paths
        Optional mapping ``oracle_name -> relative_html_path``; when present,
        the per-oracle sections link out to those standalone reports so users
        can drill further.
    """

    chrom: str
    position: int
    ref_allele: str
    alt_alleles: list[str]
    gene_name: Optional[str]
    reports: dict[str, VariantReport]
    variant_id: Optional[str] = None
    analysis_request: Optional[AnalysisRequest] = None
    per_oracle_report_paths: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_reports(
        cls,
        reports: Iterable[VariantReport],
        *,
        variant_id: str | None = None,
        analysis_request: AnalysisRequest | None = None,
        per_oracle_report_paths: dict[str, str] | None = None,
    ) -> "MultiOracleReport":
        """Build a :class:`MultiOracleReport` from per-oracle reports.

        Raises :class:`ValueError` when the reports disagree on variant
        identity — a mismatch is always a bug at the caller, never a silent
        downstream rendering problem.
        """
        report_list = list(reports)
        if not report_list:
            raise ValueError("At least one VariantReport is required")

        first = report_list[0]
        for r in report_list[1:]:
            if (r.chrom, r.position, r.ref_allele) != (
                first.chrom, first.position, first.ref_allele,
            ):
                raise ValueError(
                    f"Variant mismatch: {first.chrom}:{first.position} "
                    f"{first.ref_allele} != {r.chrom}:{r.position} {r.ref_allele}"
                )

        # Alt alleles may differ if oracles were run on different allele sets;
        # union them rather than erroring — the consensus matrix works on
        # shared layers, not shared alleles.
        alt_union: list[str] = list(first.alt_alleles)
        for r in report_list[1:]:
            for a in r.alt_alleles:
                if a not in alt_union:
                    alt_union.append(a)

        gene_name = first.gene_name or next(
            (r.gene_name for r in report_list[1:] if r.gene_name), None
        )

        report_map: dict[str, VariantReport] = {}
        for r in report_list:
            key = r.oracle_name or "unknown"
            # Guard against duplicate oracle names — pick the one with more
            # scored tracks.
            if key in report_map:
                existing = report_map[key]
                n_existing = sum(len(s) for s in existing.allele_scores.values())
                n_new = sum(len(s) for s in r.allele_scores.values())
                if n_new > n_existing:
                    report_map[key] = r
                continue
            report_map[key] = r

        return cls(
            chrom=first.chrom,
            position=first.position,
            ref_allele=first.ref_allele,
            alt_alleles=alt_union,
            gene_name=gene_name,
            reports=report_map,
            variant_id=variant_id,
            analysis_request=analysis_request,
            per_oracle_report_paths=per_oracle_report_paths or {},
        )

    @classmethod
    def from_json_files(
        cls,
        paths: Iterable[str | Path],
        *,
        variant_id: str | None = None,
        analysis_request: AnalysisRequest | None = None,
        per_oracle_report_paths: dict[str, str] | None = None,
    ) -> "MultiOracleReport":
        """Rehydrate from on-disk JSON files (one per oracle)."""
        reports: list[VariantReport] = []
        for p in paths:
            path = Path(p)
            with path.open() as fh:
                data = json.load(fh)
            reports.append(VariantReport.from_dict(data))
        return cls.from_reports(
            reports,
            variant_id=variant_id,
            analysis_request=analysis_request,
            per_oracle_report_paths=per_oracle_report_paths,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def display_variant_id(self) -> str:
        return self.variant_id or f"{self.chrom}:{self.position}"

    def layers_present(self) -> list[str]:
        """Ordered list of regulatory layers across all oracles."""
        order: list[str] = []
        for rep in self.reports.values():
            for scores in rep.allele_scores.values():
                for ts in scores:
                    if ts.layer not in order:
                        order.append(ts.layer)
        return order

    def _best_per_layer(
        self, report: VariantReport,
    ) -> dict[str, TrackScore]:
        """Return the max-|effect| :class:`TrackScore` per layer in a report."""
        out: dict[str, TrackScore] = {}
        for scores in report.allele_scores.values():
            for ts in scores:
                if ts.raw_score is None:
                    continue
                cur = out.get(ts.layer)
                if cur is None or abs(ts.raw_score) > abs(cur.raw_score or 0):
                    out[ts.layer] = ts
        return out

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d: dict = {
            "variant_id": self.display_variant_id,
            "variant": {
                "chrom": self.chrom,
                "position": self.position,
                "ref_allele": self.ref_allele,
                "alt_alleles": self.alt_alleles,
            },
            "gene_name": self.gene_name,
            "oracles": list(self.reports.keys()),
            "per_oracle_report_paths": dict(self.per_oracle_report_paths),
            "consensus": self._consensus_rows(),
        }
        if self.analysis_request is not None:
            d["analysis_request"] = self.analysis_request.to_dict()
        return d

    def _consensus_rows(self) -> list[dict]:
        """Machine-readable cross-oracle matrix (one row per layer)."""
        rows = []
        for layer in self.layers_present():
            entry: dict = {"layer": layer, "oracles": {}}
            directions: list[int] = []
            for name, rep in self.reports.items():
                best = self._best_per_layer(rep).get(layer)
                if best is None or best.raw_score is None:
                    entry["oracles"][name] = None
                    continue
                entry["oracles"][name] = {
                    "raw_score": best.raw_score,
                    "assay_id": best.assay_id,
                    "cell_type": best.cell_type,
                    "quantile_score": best.quantile_score,
                    # Prefer the enriched display name (e.g.
                    # ``CHIP:CEBPA:HepG2``) over the raw AlphaGenome
                    # assay_id (``CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA
                    # genetically modified…``) so the consensus matrix
                    # matches what the per-variant reports show.
                    "description": best.description or best.assay_id,
                }
                directions.append(1 if best.raw_score > 0 else -1)
            # "Consensus" requires at least two voting oracles; a single
            # voter is recorded as ``single_gain`` / ``single_loss`` so the
            # rendered label doesn't read "all ↑" when only one oracle
            # actually scored the layer.
            if directions:
                pos = sum(1 for d in directions if d > 0)
                neg = len(directions) - pos
                if len(directions) == 1:
                    entry["agreement"] = "single_gain" if pos == 1 else "single_loss"
                elif pos == len(directions):
                    entry["agreement"] = "consensus_gain"
                elif neg == len(directions):
                    entry["agreement"] = "consensus_loss"
                else:
                    entry["agreement"] = "disagree"
            else:
                entry["agreement"] = "no_data"
            rows.append(entry)
        return rows

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def to_html(self, output_path: str | Path | None = None) -> str:
        html = _build_multioracle_html(self)
        if output_path is not None:
            path = Path(output_path)
            if path.suffix == "" or path.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                fname = f"{self._fname_stub()}_multioracle_report.html"
                path = path / fname
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
            logger.info("Multi-oracle report written to %s", path)
        return html

    def _fname_stub(self) -> str:
        stub = self.variant_id or f"{self.chrom}_{self.position}"
        if self.gene_name:
            stub += f"_{self.gene_name}"
        return stub.replace(":", "_").replace("/", "_").replace(" ", "_")

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# Multi-oracle validation — {self.display_variant_id}")
        lines.append("")
        lines.append(f"- **Variant:** {self.chrom}:{self.position:,} "
                     f"{self.ref_allele}>{','.join(self.alt_alleles)}")
        if self.gene_name:
            lines.append(f"- **Gene:** {self.gene_name}")
        lines.append(f"- **Oracles:** {', '.join(self.reports.keys())}")
        lines.append("")
        lines.append("## Cross-oracle consensus")
        lines.append("")
        header = ["Layer"] + list(self.reports.keys()) + ["Agreement"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for row in self._consensus_rows():
            layer = row["layer"]
            cfg = LAYER_CONFIGS.get(layer)
            # Use the canonical display label (log2FC / lnFC / Δ) rather
            # than the raw lowercase LayerConfig key so the markdown
            # matches the HTML formula chips exactly.
            layer_label = (
                f"{cfg.description} ({formula_label(cfg.formula)})" if cfg
                else layer
            )
            cells = [layer_label]
            for name in self.reports.keys():
                entry = row["oracles"].get(name)
                if entry is None:
                    cells.append("—")
                else:
                    sign = "+" if entry["raw_score"] >= 0 else ""
                    # Prefer enriched description (e.g. CHIP:CEBPA:HepG2)
                    # over raw assay_id so the matrix matches per-variant
                    # reports.
                    track_label = entry.get("description") or entry["assay_id"]
                    cells.append(
                        f"{sign}{entry['raw_score']:.3f} · "
                        f"{track_label} · {entry['cell_type'] or '—'}"
                    )
            agree = {
                "consensus_gain": "all ↑",
                "consensus_loss": "all ↓",
                "single_gain": "only ↑ (n=1)",
                "single_loss": "only ↓ (n=1)",
                "disagree": "disagree",
                "no_data": "—",
            }[row["agreement"]]
            cells.append(agree)
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_EXTRA_CSS = """
.moracle-header { margin: 0.4rem 0; }
.consensus-table .agree-gain { background: #dcfce7; color: #166534;
                               font-weight: 600; }
.consensus-table .agree-loss { background: #fee2e2; color: #991b1b;
                               font-weight: 600; }
.consensus-table .agree-mixed { background: #fff3cd; color: #856404;
                                font-weight: 600; }
/* single-voter — neutral grey so users see that only one oracle
   reported a direction (no real consensus possible). */
.consensus-table .agree-single { background: #f3f4f6; color: #4b5563;
                                 font-weight: 500; }
.consensus-table .agree-none { color: #adb5bd; }
.consensus-table td.effect-cell { font-family: ui-monospace, SFMono-Regular,
                                  Menlo, monospace; font-size: 0.85rem; }
.consensus-table td.effect-cell .track { display: block; font-size: 0.75rem;
                                         color: #6b7280; font-family: inherit; }
.consensus-table td.effect-pos { color: #166534; }
.consensus-table td.effect-neg { color: #991b1b; }
details.oracle-block { margin: 0.6rem 0; border: 1px solid #dee2e6;
                       border-radius: 4px; background: white; }
details.oracle-block summary { padding: 0.6rem 1rem; cursor: pointer;
                               font-weight: 600; color: #343a40;
                               background: #f8f9fa; border-radius: 4px; }
details.oracle-block[open] summary { border-bottom: 1px solid #dee2e6;
                                     border-radius: 4px 4px 0 0; }
details.oracle-block .oracle-body { padding: 0.8rem 1rem; }
details.oracle-block table { font-size: 0.85rem; }
details.oracle-block table th { background: #6c757d; }
"""


def _build_multioracle_html(report: "MultiOracleReport") -> str:
    esc = _html_mod.escape
    p: list[str] = []
    p.append("<!DOCTYPE html><html lang='en'><head>")
    p.append("<meta charset='utf-8'>")
    title_parts = [
        "Multi-oracle validation",
        report.display_variant_id,
    ]
    if report.gene_name:
        title_parts.append(report.gene_name)
    p.append(f"<title>{esc(' — '.join(title_parts))}</title>")
    p.append(f"<style>{_CSS}{_EXTRA_CSS}</style>")
    p.append("</head><body>")

    # Header -----------------------------------------------------------
    header_bits = ["Multi-oracle validation"]
    if report.variant_id:
        header_bits.append(esc(report.variant_id))
    header_bits.append(f"{esc(report.chrom)}:{report.position:,}")
    p.append(f"<h1 class='moracle-header'>{' — '.join(header_bits)}</h1>")
    if report.analysis_request is not None:
        p.append(report.analysis_request.to_html_fragment())
    p.append(f"<p class='meta'><b>Variant:</b> {esc(report.chrom)}:"
             f"{report.position:,} {esc(report.ref_allele)}&gt;"
             f"{esc(','.join(report.alt_alleles))}</p>")
    if report.gene_name:
        p.append(f"<p class='meta'><b>Gene:</b> {esc(report.gene_name)}</p>")
    p.append(f"<p class='meta'><b>Oracles compared:</b> "
             f"{esc(', '.join(report.reports.keys()))}</p>")

    # Glossary ---------------------------------------------------------
    layers_all = report.layers_present()
    any_quantile = any(
        ts.quantile_score is not None
        for rep in report.reports.values()
        for scores in rep.allele_scores.values()
        for ts in scores
    )
    any_activity = any(
        ts.ref_signal_percentile is not None
        for rep in report.reports.values()
        for scores in rep.allele_scores.values()
        for ts in scores
    )
    p.append(render_how_to_read(
        layers_present=layers_all,
        include_percentile=any_quantile,
        include_activity=any_activity,
        lead_sentence=(
            "<b>How to read this report.</b> "
            "A single variant has been scored independently by "
            f"<b>{len(report.reports)}</b> different deep-learning oracles. "
            "The consensus matrix below shows each oracle's strongest call per "
            "regulatory layer, and flags whether the oracles agree on "
            "direction — use it to separate signal from single-model "
            "artefacts."
        ),
    ))

    # Cross-oracle consensus -------------------------------------------
    p.append("<h2>Cross-oracle consensus</h2>")
    p.append("<p class='meta'>Each cell is the <b>single strongest track</b> "
             "that oracle reported for that layer. '—' means the oracle did "
             "not score any track in that layer.</p>")
    p.append("<table class='consensus-table'><thead><tr>")
    p.append("<th>Layer</th>")
    for name in report.reports.keys():
        p.append(f"<th>{esc(name)}</th>")
    p.append("<th>Agreement</th></tr></thead><tbody>")

    for row in report._consensus_rows():
        layer = row["layer"]
        cfg = LAYER_CONFIGS.get(layer)
        layer_label = cfg.description if cfg else layer
        chip = formula_chip(cfg.formula) if cfg else ""
        p.append(f"<tr><td>{esc(layer_label)} {chip}</td>")
        for name in report.reports.keys():
            entry = row["oracles"].get(name)
            if entry is None:
                p.append("<td class='agree-none'>—</td>")
                continue
            score = entry["raw_score"]
            sign_char = "+" if score >= 0 else ""
            cls = "effect-pos" if score >= 0 else "effect-neg"
            # Use the same percentile display helper as every other
            # chorus report so "≥99th" / "≤1st" / "near-zero" are used
            # uniformly (not "+100.0%").
            q = entry.get("quantile_score")
            q_str = f" · {_fmt_percentile(q)}" if q is not None else ""
            # Prefer enriched label (CHIP:CEBPA:HepG2) over raw assay_id
            # so the consensus matrix matches per-variant reports.
            track_line = entry.get("description") or entry["assay_id"]
            if entry.get("cell_type") and entry["cell_type"] not in track_line:
                track_line += f" · {entry['cell_type']}"
            p.append(
                f"<td class='effect-cell {cls}'>{sign_char}{score:.3f}{q_str}"
                f"<span class='track'>{esc(track_line)}</span></td>"
            )
        agree = row["agreement"]
        agree_label, agree_cls = {
            "consensus_gain": ("✅ all ↑", "agree-gain"),
            "consensus_loss": ("✅ all ↓", "agree-loss"),
            "single_gain": ("↑ only (n=1)", "agree-single"),
            "single_loss": ("↓ only (n=1)", "agree-single"),
            "disagree": ("⚠ disagree", "agree-mixed"),
            "no_data": ("—", "agree-none"),
        }[agree]
        p.append(f"<td class='{agree_cls}'>{agree_label}</td>")
        p.append("</tr>")
    p.append("</tbody></table>")

    # Per-oracle drill-down --------------------------------------------
    p.append("<h2>Per-oracle evidence</h2>")
    p.append("<p class='meta'>Each block expands to the strongest track per "
             "layer for that oracle. For the full signal overlays and every "
             "track scored, open the standalone per-oracle report.</p>")
    for name, rep in report.reports.items():
        # Derive a compact per-oracle summary: strongest layer + n tracks.
        best_map = report._best_per_layer(rep)
        n_tracks = sum(len(s) for s in rep.allele_scores.values())
        strongest_layer = None
        strongest_score = 0.0
        for layer, ts in best_map.items():
            if ts.raw_score is not None and abs(ts.raw_score) > abs(strongest_score):
                strongest_score = ts.raw_score
                strongest_layer = layer
        layer_desc = LAYER_CONFIGS.get(strongest_layer).description \
            if strongest_layer and strongest_layer in LAYER_CONFIGS else (
                strongest_layer or "no tracks"
            )
        sign_s = "+" if strongest_score >= 0 else ""

        summary_bits = [
            f"<b>{esc(name)}</b>",
            f"{n_tracks} tracks",
        ]
        if strongest_layer:
            summary_bits.append(
                f"strongest: {esc(layer_desc)} "
                f"{sign_s}{strongest_score:.3f}"
            )
        p.append("<details class='oracle-block'>")
        p.append(f"<summary>{' · '.join(summary_bits)}</summary>")
        p.append("<div class='oracle-body'>")

        # Per-layer breakdown using the winning track per layer.
        if best_map:
            p.append("<table><thead><tr><th>Layer</th><th>Top assay</th>"
                     "<th>Cell type</th><th>Ref</th><th>Alt</th>"
                     "<th>Effect</th><th>%ile</th></tr></thead><tbody>")
            # Sort layers by |effect| descending so the readable story leads.
            sorted_items = sorted(
                best_map.items(),
                key=lambda kv: abs(kv[1].raw_score or 0),
                reverse=True,
            )
            for layer, ts in sorted_items:
                cfg = LAYER_CONFIGS.get(layer)
                layer_label = cfg.description if cfg else layer
                chip = formula_chip(cfg.formula) if cfg else ""
                sc = ts.raw_score or 0.0
                sign_s = "+" if sc >= 0 else ""
                cls = "effect-pos" if sc >= 0 else "effect-neg"
                ref_s = f"{ts.ref_value:.3g}" if ts.ref_value is not None else "—"
                alt_s = f"{ts.alt_value:.3g}" if ts.alt_value is not None else "—"
                q = ts.quantile_score
                q_str = _fmt_percentile(q) if q is not None else "—"
                # Prefer enriched description (e.g. "CHIP:CEBPA:HepG2")
                # over the raw AlphaGenome assay_id so the per-oracle
                # drill-down table matches what the variant-report pages
                # show.
                track_display = ts.description or ts.assay_id
                p.append(
                    f"<tr><td>{esc(layer_label)} {chip}</td>"
                    f"<td>{esc(track_display)}</td>"
                    f"<td>{esc(ts.cell_type or '—')}</td>"
                    f"<td>{ref_s}</td><td>{alt_s}</td>"
                    f"<td class='effect-cell {cls}'>"
                    f"{sign_s}{sc:.3f}</td>"
                    f"<td>{q_str}</td></tr>"
                )
            p.append("</tbody></table>")
        else:
            p.append("<p class='note'>No tracks scored for this oracle.</p>")

        link = report.per_oracle_report_paths.get(name)
        if link:
            p.append(f"<p><a href='{esc(link)}'>Open {esc(name)} "
                     f"standalone report →</a></p>")
        p.append("</div></details>")

    # Footer -----------------------------------------------------------
    p.append("<div class='footer'>Generated by Chorus "
             "<code>MultiOracleReport</code></div>")
    p.append("</body></html>")
    return "\n".join(p)
