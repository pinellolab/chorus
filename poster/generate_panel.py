"""Generate a multi-oracle poster panel from a variant config (JSON).

The panel format is the one at `poster/rs12740374_SORT1_panel_compact.html`:
six-row regulatory-layer consensus matrix + two-column mechanism box
(predictions vs published ground truth, if available) + live IGV iframe.

Usage
-----
    python poster/generate_panel.py poster/variants/rs12740374.json

Produces:
    poster/<rsid>_<gene>_panel.html   (the panel)

Configs live in `poster/variants/*.json`. See `rs12740374.json` for the
canonical schema. The user is expected to:

1.  Run chorus predictions for the variant (e.g. `verify_predictions.py`).
2.  Hand-curate the row values + ground-truth bullets in a config JSON.
3.  Run this script to render the HTML.

If paperclip is installed and the config sets `ground_truth.pmc_id`,
this script will try `paperclip lookup pmc <id>` and inline the paper
title in the citation. Manual curation of bullets is still required;
this is a rendering tool, not a literature-extraction LLM.

The IGV iframe (igv_only.html) is shared across panels; regenerate it
with `extract_igv.py` against the canonical chorus multi-oracle example.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

POSTER_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = POSTER_DIR / "_panel_template_compact.html"


# ---------------------------------------------------------------------------
# Row renderer
# ---------------------------------------------------------------------------

ORACLES = ("chrombpnet", "legnet", "alphagenome")

VERDICT_CLASS = {
    "match": "tag match",
    "single": "tag single",
    "soft": "tag soft",
}


def _cell(oracle_data: dict | None) -> str:
    """Render one consensus-matrix cell."""
    if oracle_data is None:
        return '<td class="num empty">·<span class="why">not scored</span></td>'
    value = oracle_data["value"]
    pct = oracle_data.get("pct", "")
    track = oracle_data.get("track", "")
    return (
        f'<td class="num">'
        f'<span class="v">{value}</span>'
        f'<span class="pct">{pct}</span>'
        f'<span class="track">{track}</span>'
        f'</td>'
    )


def _row(row: dict) -> str:
    """Render one regulatory-layer row of the consensus matrix."""
    verdict = row["verdict"]
    klass = VERDICT_CLASS.get(verdict["tag"], "tag single")
    layer_cell = (
        f'<td class="layer">'
        f'<span class="lname">{row["layer_name"]}</span>'
        f'<span class="unit">{row["unit"]}</span>'
        f'</td>'
    )
    cells = "\n          ".join(_cell(row["values"].get(o)) for o in ORACLES)
    verdict_cell = f'<td class="agree"><span class="{klass}">{verdict["text"]}</span></td>'
    return f"<tr>\n          {layer_cell}\n          {cells}\n          {verdict_cell}\n        </tr>"


def _bullets(items: list[str], tag: str = "li") -> str:
    return "\n        ".join(f"<{tag}>{item}</{tag}>" for item in items)


# ---------------------------------------------------------------------------
# Paper source resolution (PMC, PubMed, DOI, URL, paperclip-native ID)
# ---------------------------------------------------------------------------

def _paperclip_lookup(field: str, value: str) -> bool:
    """Best-effort: does paperclip have this paper indexed?"""
    if not shutil.which("paperclip"):
        return False
    try:
        out = subprocess.run(
            ["paperclip", "lookup", field, value],
            capture_output=True, text=True, timeout=15,
        )
        return out.returncode == 0 and "No documents found" not in out.stdout
    except Exception:
        return False


def _paper_links(paper: dict) -> tuple[str, str | None]:
    """Build the citation_html block from a paper-source dict.

    Accepts any combination of (in priority order for the canonical link):
      - paperclip_id  e.g. "PMC3062476", "bio_abc123", "med_xyz", "arx_2403.01"
      - pmc_id        e.g. "PMC3062476"
      - pubmed_id     e.g. "20686566"
      - doi           e.g. "10.1038/nature09266"
      - url           any URL string (takes precedence as the canonical link)
      - citation_text human-readable, e.g. "Musunuru et al., Nature 466, 714 (2010)"

    Returns (citation_html, paperclip_indicator_html_or_None).
    """
    pid  = paper.get("paperclip_id") or paper.get("pmc_id")  # both look up via /papers/<id>
    pmc  = paper.get("pmc_id")
    pmid = paper.get("pubmed_id")
    doi  = paper.get("doi")
    url  = paper.get("url")
    txt  = paper.get("citation_text", "")

    # Build the canonical link in priority order.
    if not url:
        if pmc:
            url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/"
        elif pmid:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        elif doi:
            url = f"https://doi.org/{doi}"

    parts: list[str] = []
    if txt:
        parts.append(txt)
    label = pmc or pmid or doi or pid
    if url:
        if label:
            parts.append(f'<a href="{url}">{label}</a>')
        else:
            parts.append(f'<a href="{url}">link</a>')
    elif label:
        parts.append(str(label))
    citation_html = " · ".join(parts) if parts else ""

    # paperclip best-effort lookup across whatever IDs we have.
    in_paperclip = False
    if pid and pid.startswith(("PMC", "bio_", "med_", "arx_")):
        # paperclip native lookups: pmc_id matches the /papers/ dir name
        if pid.startswith("PMC"):
            in_paperclip = _paperclip_lookup("pmc", pid)
        # bio_/med_/arx_ have no `paperclip lookup` keyword; fall through.
    if not in_paperclip and pmc:
        in_paperclip = _paperclip_lookup("pmc", pmc)
    if not in_paperclip and pmid:
        in_paperclip = _paperclip_lookup("pmid", pmid)
    if not in_paperclip and doi:
        in_paperclip = _paperclip_lookup("doi", doi)

    indicator = (
        '<span style="margin-left:6px;color:#2F8F58">✓ in paperclip</span>'
        if in_paperclip else None
    )
    return citation_html, indicator


# ---------------------------------------------------------------------------
# Panel renderer
# ---------------------------------------------------------------------------

def render(config: dict) -> str:
    with open(TEMPLATE_PATH) as f:
        tpl = f.read()

    v = config["variant"]
    ct = config["cell_type"]

    # Variant header
    location = f'{v["chrom"]}:{v["position"]:,}'
    ref_alt = f'{v["ref"]} → {v["alt"]}'

    # Consensus matrix rows
    rows_html = "\n\n        ".join(_row(r) for r in config["rows"])

    # Mechanism bullets
    pred_bullets = _bullets(config["mechanism"]["predictions"])

    # Ground-truth column (optional)
    gt = config.get("ground_truth")
    if gt:
        gt_bullets = _bullets(gt["bullets"])

        # New schema: `paper` sub-object with PMC / PubMed / DOI / URL.
        # Back-compat: old flat `pmc_id` + `citation_html` still works.
        paper = gt.get("paper")
        if paper:
            citation_html, indicator = _paper_links(paper)
        else:
            citation_html = gt.get("citation_html", "")
            indicator = None
            if gt.get("pmc_id") and _paperclip_lookup("pmc", gt["pmc_id"]):
                indicator = '<span style="margin-left:6px;color:#2F8F58">✓ in paperclip</span>'
        if indicator:
            citation_html = f"{citation_html} {indicator}" if citation_html else indicator

        gt_html = f"""
    <div class="col">
      <h3>{gt.get("title", "Ground truth")}</h3>
      <ul>
        {gt_bullets}
      </ul>
      <div class="ref">
        {citation_html}
      </div>
    </div>"""
    else:
        gt_html = """
    <div class="col">
      <h3>Ground truth</h3>
      <p style="font-size:11px;color:#8A93A8;margin:0;line-height:1.4">
        No paper cited. Use <code>paperclip search</code> to find a
        published reporter / qPCR / eQTL study at this locus, then add
        verified quotes to the config's <code>ground_truth.bullets</code>.
      </p>
    </div>"""

    igv_src = config.get("igv_src", "igv_only.html")

    out = tpl
    out = out.replace("{{VARIANT_RSID}}", v["rsid"])
    out = out.replace("{{VARIANT_GENE}}", v["gene"])
    out = out.replace("{{VARIANT_LOCATION}}", location)
    out = out.replace("{{VARIANT_REF_ALT}}", ref_alt)
    out = out.replace("{{CELL_TYPE}}", ct)
    out = out.replace("{{ROWS_HTML}}", rows_html)
    out = out.replace("{{PREDICTION_BULLETS}}", pred_bullets)
    out = out.replace("{{GROUND_TRUTH_BLOCK}}", gt_html)
    out = out.replace("{{IGV_SRC}}", igv_src)
    out = out.replace(
        "{{IGV_META}}",
        config.get("igv_meta", f'{v["gene"]} ±150 kb'),
    )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", help="path to variant config JSON")
    ap.add_argument("-o", "--out", help="output HTML path (default: poster/<rsid>_<gene>_panel.html)")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    out_path = args.out or str(
        POSTER_DIR / f'{config["variant"]["rsid"]}_{config["variant"]["gene"]}_panel_compact.html'
    )
    html = render(config)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
