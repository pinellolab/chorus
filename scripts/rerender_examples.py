"""Re-render every example HTML report from its saved JSON.

When the report-rendering code changes (e.g. a new glossary block, formula
chips on column headers, per-track provenance in summaries) the example
HTML artefacts ship in the repo need to be refreshed. *Most* of them can be
refreshed without touching any oracle: the underlying predictions are
already on disk in ``example_output.json``, and every renderer supports
``from_dict`` round-tripping.

This script walks the examples directory, rehydrates every supported JSON,
and re-renders the HTML in place. It runs in any environment — no GPU,
no oracle model downloads.

Coverage:

* ``variant_analysis/**`` — uses :meth:`VariantReport.from_dict`
* ``validation/**``       — VariantReport (per-oracle) + MultiOracleReport
                             consolidator is refreshed from the stored
                             per-oracle JSONs.
* ``sequence_engineering/region_swap`` and ``integration_simulation``
                           — also VariantReport.to_dict format.
* ``discovery/**``        — multi-cell-type VariantReport JSONs.

Not covered (require re-running the oracle):

* ``causal_prioritization/**`` — CausalResult.to_dict doesn't preserve
  the full per-track allele scores needed for the drill-down table.
  Re-run ``scripts/regenerate_remaining_examples.py --only causal``
  to refresh. This script will warn but not error.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rerender")

EXAMPLES = REPO_ROOT / "examples" / "applications"


# ---------------------------------------------------------------------------
# Rehydrate a VariantReport-style JSON back to HTML
# ---------------------------------------------------------------------------

def _rehydrate_variant_report(json_path: Path) -> int:
    """Load a VariantReport JSON and write HTML back to its dir.

    Returns the number of HTML files written.
    """
    from chorus.analysis.variant_report import VariantReport

    with json_path.open() as fh:
        data = json.load(fh)

    # Some of our example JSONs are the *causal* format; detect by key.
    if "rankings" in data and "sentinel" in data:
        return 0  # handled elsewhere

    # Some are MultiOracleReport summaries; detect by 'consensus' key.
    if "consensus" in data and "oracles" in data:
        return 0

    if "alleles" not in data or "variant" not in data:
        return 0

    report = VariantReport.from_dict(data)
    # Derive the original HTML filename convention.
    html_name = report.default_filename("html")
    out_path = json_path.parent / html_name

    # When an explicit filename was used before (non-default), keep the
    # previous HTML file name if we can unambiguously identify it. This
    # preserves link stability across README references.
    existing_htmls = sorted(
        p for p in json_path.parent.glob("*_report.html")
        if "multioracle" not in p.name.lower()
    )
    if len(existing_htmls) == 1:
        out_path = existing_htmls[0]
    elif len(existing_htmls) > 1:
        # Multiple candidates means the dir holds sibling reports from
        # different oracles or runs.  Pick the one that belongs to THIS
        # JSON by oracle-name match; prefer documented "validation_report"
        # names when present.
        oracle = report.oracle_name.lower()
        validation_matches = [p for p in existing_htmls
                              if "validation_report" in p.name.lower()]
        oracle_matches = [p for p in existing_htmls if oracle in p.name.lower()]
        if validation_matches and oracle in validation_matches[0].name.lower():
            out_path = validation_matches[0]
        elif len(oracle_matches) == 1:
            out_path = oracle_matches[0]
        elif validation_matches:
            out_path = validation_matches[0]

    report.to_html(output_path=out_path)
    logger.info("  rerendered %s", out_path.relative_to(REPO_ROOT))
    return 1


# ---------------------------------------------------------------------------
# Main walk
# ---------------------------------------------------------------------------

def walk_examples(only: str | None = None) -> int:
    total = 0
    skipped = 0

    category_dirs = sorted(p for p in EXAMPLES.iterdir() if p.is_dir())
    for cat in category_dirs:
        if only and cat.name != only:
            continue
        logger.info("Category: %s", cat.name)

        # Causal examples can't be rehydrated without re-running oracles.
        if cat.name == "causal_prioritization":
            logger.info("  [skipped — re-run scripts/regenerate_remaining_examples.py --only causal]")
            skipped += 1
            continue

        # batch_scoring has its own JSON format (flat BatchResult) — skip for
        # now; that report already includes the glossary.
        if cat.name == "batch_scoring":
            logger.info("  [skipped — batch_scoring example is regenerated separately]")
            skipped += 1
            continue

        for sub in sorted(cat.rglob("*.json")):
            # Only look at example_output.json and per-oracle JSONs.
            if sub.name not in {"example_output.json"} \
                    and not sub.name.endswith("_variant_report.json"):
                continue
            try:
                total += _rehydrate_variant_report(sub)
            except Exception as exc:
                logger.warning("  FAILED %s: %s", sub.relative_to(REPO_ROOT), exc)

        # Multi-oracle dir: refresh the consolidated report too.
        if cat.name == "validation":
            for sub in sorted(cat.iterdir()):
                if sub.is_dir() and "multioracle" in sub.name.lower():
                    total += _refresh_multioracle(sub)

    logger.info("Done — %d HTML files rewritten, %d categories skipped.",
                total, skipped)
    return total


def _refresh_multioracle(dir_path: Path) -> int:
    """Recompute the multi-oracle consolidated HTML from per-oracle JSONs."""
    from chorus.analysis import MultiOracleReport
    from chorus.analysis.analysis_request import AnalysisRequest

    # Preserve the canonical oracle ordering used by
    # scripts/regenerate_multioracle.py (specialists → generalist) so the
    # consensus-matrix columns don't shuffle between runs.
    _ORACLE_ORDER = ["chrombpnet", "legnet", "alphagenome", "enformer",
                     "borzoi", "sei"]
    all_jsons = {
        p.stem.replace("_variant_report", ""): p
        for p in dir_path.glob("*_variant_report.json")
    }
    per_oracle_jsons = [
        all_jsons[name] for name in _ORACLE_ORDER if name in all_jsons
    ]
    per_oracle_jsons += [
        p for name, p in sorted(all_jsons.items())
        if name not in _ORACLE_ORDER
    ]
    if not per_oracle_jsons:
        return 0

    per_oracle_paths = {}
    for jp in per_oracle_jsons:
        oracle = jp.stem.replace("_variant_report", "")
        html_candidate = dir_path / f"rs12740374_SORT1_{oracle}_report.html"
        if html_candidate.exists():
            per_oracle_paths[oracle] = html_candidate.name

    # Synthesise a multi-oracle AnalysisRequest that describes the *combined*
    # comparison, not the first per-oracle run — matches what
    # scripts/regenerate_multioracle.py writes so the rendered prompt block
    # at the top of the page is consistent whether the report is produced
    # from a full regen or a pure-JSON rerender.
    with per_oracle_jsons[0].open() as fh:
        first = json.load(fh)
    oracle_names = [
        p.stem.replace("_variant_report", "") for p in per_oracle_jsons
    ]
    ar = AnalysisRequest(
        user_prompt=(
            "Validate rs12740374 (the classic SORT1 LDL-cholesterol causal "
            "variant) by scoring it with three independent deep-learning "
            "oracles: ChromBPNet for chromatin accessibility, LegNet for MPRA "
            "promoter activity, and AlphaGenome as a generalist model "
            "covering ChIP, histones and CAGE. A new user should be able to "
            "see at a glance whether the three oracles agree on direction, "
            "and which assay/cell type drove each call."
        ),
        tool_name="MultiOracleReport",
        oracle_name=", ".join(oracle_names),
        normalizer_name="per-oracle chorus per-track v1",
        tracks_requested="assay_ids as listed in each per-oracle request",
    )

    moracle = MultiOracleReport.from_json_files(
        per_oracle_jsons,
        variant_id=first.get("gene_name") and "rs12740374" or None,
        analysis_request=ar,
        per_oracle_report_paths=per_oracle_paths,
    )
    html_path = dir_path / f"{moracle._fname_stub()}_multioracle_report.html"
    moracle.to_html(output_path=html_path)
    with (dir_path / "example_output.md").open("w") as fh:
        fh.write(moracle.to_markdown())
    with (dir_path / "example_output.json").open("w") as fh:
        json.dump(moracle.to_dict(), fh, indent=2, default=str)
    logger.info("  refreshed multi-oracle: %s",
                html_path.relative_to(REPO_ROOT))
    return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        help="Limit to one category name (variant_analysis, validation, …).",
    )
    args = p.parse_args()
    walk_examples(only=args.only)
