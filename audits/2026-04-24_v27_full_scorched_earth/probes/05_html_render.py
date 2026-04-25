"""Render every shipped HTML walkthrough at 1600×4500, screenshot it,
and audit against audit-checklist §7.

Per §7:
- Renders at 1600×4500 without JS errors in the browser console.       [P0]
- IGV browser block shows real signal tracks (not just placeholder).   [P0]
- Glossary block present with log2FC/lnFC/Δ formula legend.            [P1]
- Every per-layer table has Track · Cell Type · Ref · Alt · Effect [formula
  badge] · Ref%ile · Activity%ile · Interpretation.                    [P1]
- Formula badges match layer: log2FC chromatin/TF/histone/TSS,
  lnFC RNA-seq/CAGE gene expression, Δ MPRA.                           [P0]
- Cell-type column doesn't duplicate text already in the track label.  [P1]

Output:
- screenshots/<basename>.png — 1600-wide full-page render
- render_log.json — per-file pass/fail + counts of console errors and
  flagged keywords
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path("examples/walkthroughs")
OUT = Path("audits/2026-04-24_v27_full_scorched_earth/screenshots")
OUT.mkdir(parents=True, exist_ok=True)


def audit_one(page, html_path: Path) -> dict:
    """Render one HTML, return audit dict."""
    url = f"file://{html_path.resolve()}"
    # Two pairs of HTMLs (variant_analysis/SORT1_rs12740374 vs
    # validation/SORT1_rs12740374_multioracle) share basenames; prefix
    # with the immediate parent dir so screenshots don't collide.
    name = f"{html_path.parent.name}__{html_path.stem}"
    out_png = OUT / f"{name}.png"

    console_msgs: list[dict] = []
    page.on(
        "console",
        lambda msg: console_msgs.append(
            {"type": msg.type, "text": msg.text[:200]}
        ),
    )
    pageerrors: list[str] = []
    page.on("pageerror", lambda exc: pageerrors.append(str(exc)[:200]))

    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
    except Exception as exc:
        return {
            "file": str(html_path),
            "loaded": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    # Give IGV / dynamic JS up to 12 s extra (per §7 selenium wait).
    page.wait_for_timeout(12000)

    page.set_viewport_size({"width": 1600, "height": 4500})
    page.screenshot(path=str(out_png), full_page=True)

    body_text = page.inner_text("body")
    body_lower = body_text.lower()
    body_size = len(body_text)

    # Heuristics for the §7 criteria (text-only — exact rendering details
    # need a human eyeball on the screenshot but these catch wholesale
    # regressions).
    #
    # IGV is rendered into a <div> by JavaScript (`igv.createBrowser`),
    # so its tracks don't show up in body innerText. Check both the
    # source for the setup call AND the live DOM for igv-class
    # elements after the 12 s wait.
    page_html = page.content()
    has_igv_setup_in_source = (
        "igv.createBrowser" in page_html or "igv.min.js" in page_html
    )
    has_igv_dom = page.evaluate(
        "() => document.querySelectorAll('[id*=igv], [class*=igv]').length"
    )
    has_igv_browser_block = has_igv_setup_in_source and has_igv_dom > 0
    has_glossary = (
        "glossary" in body_lower
        or "how to read" in body_lower
        or "log2fc" in body_lower
    )
    has_log2fc_badge = "log2fc" in body_lower or "log2 fold" in body_lower
    has_lnfc_badge = "lnfc" in body_lower
    has_delta_badge = (
        "Δ" in body_text or "alt - ref" in body_lower or "alt-ref" in body_lower
    )
    has_percentile_columns = (
        "%ile" in body_text or "percentile" in body_text.lower()
    )

    severe_console = [
        m for m in console_msgs if m["type"] in ("error",) and m["text"]
    ]

    return {
        "file": str(html_path),
        "screenshot": str(out_png),
        "loaded": True,
        "body_chars": body_size,
        "has_igv_setup": has_igv_setup_in_source,
        "igv_dom_nodes": has_igv_dom,
        "has_igv_block": has_igv_browser_block,
        "has_glossary": has_glossary,
        "has_log2fc": has_log2fc_badge,
        "has_lnfc": has_lnfc_badge,
        "has_delta": has_delta_badge,
        "has_percentile_columns": has_percentile_columns,
        "console_errors": len(severe_console),
        "page_errors": len(pageerrors),
        "first_page_error": pageerrors[0] if pageerrors else None,
    }


def main():
    htmls = sorted(ROOT.glob("**/*.html"))
    print(f"[render] {len(htmls)} HTML files to audit")
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1600, "height": 4500})
        page = ctx.new_page()
        for i, html in enumerate(htmls, 1):
            print(f"  [{i}/{len(htmls)}] {html}")
            try:
                results.append(audit_one(page, html))
            except Exception as exc:
                results.append({
                    "file": str(html),
                    "loaded": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        browser.close()

    summary_path = OUT.parent / "render_log.json"
    summary_path.write_text(json.dumps(results, indent=2))

    fails = [r for r in results if not r.get("loaded")]
    no_igv = [r for r in results if r.get("loaded") and not r.get("has_igv_block")]
    no_glossary = [r for r in results if r.get("loaded") and not r.get("has_glossary")]
    js_err = [r for r in results if r.get("page_errors", 0) > 0]

    print(f"\n[render] summary: {len(results)} total, {len(fails)} failed-to-load")
    print(f"  no IGV block: {len(no_igv)}")
    print(f"  no glossary:  {len(no_glossary)}")
    print(f"  JS errors:    {len(js_err)}")
    if fails:
        print("\nFailed to load:")
        for r in fails:
            print(f"  - {r['file']}: {r.get('error')}")
    if js_err:
        print("\nFiles with JS errors:")
        for r in js_err[:5]:
            print(f"  - {r['file']}: {r['first_page_error']}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
