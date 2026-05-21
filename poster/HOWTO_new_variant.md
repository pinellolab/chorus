# How to build a multi-oracle poster panel for a new variant

The chorus poster panel format
(`rs12740374_SORT1_panel_compact.html` is the canonical example) is
reusable. Each panel needs (a) verified chorus predictions across the
relevant regulatory layers, (b) optional published-ground-truth bullets
extracted from a paper. The generator takes a JSON config and emits the
final HTML.

```
poster/
  generate_panel.py             # the renderer
  _panel_template_compact.html  # HTML template with {{KEY}} placeholders
  variants/
    rs12740374.json             # canonical example (Musunuru 2010)
    <your_variant>.json         # ← your new variant here
  verify_predictions.py         # rerun ChromBPNet / LegNet / AlphaGenome
  verify_chrombpnet_cebpa.py    # ChromBPNet TF head (BPNet architecture)
  verify_legnet_narrow.py       # LegNet at the SNP site (single fragment)
  extract_igv.py                # regenerate igv_only.html from chorus example
```

## Recipe (per variant)

### 1. Run chorus predictions

Edit the variant coordinates into a small wrapper or just modify the
`VARIANT = dict(...)` block at the top of each `verify_*.py` script.
Each script runs in its own mamba env:

```bash
mamba run -n chorus-alphagenome python poster/verify_predictions.py --oracle alphagenome
mamba run -n chorus-chrombpnet  python poster/verify_predictions.py --oracle chrombpnet
mamba run -n chorus-legnet      python poster/verify_predictions.py --oracle legnet
mamba run -n chorus-chrombpnet  python poster/verify_chrombpnet_cebpa.py    # if TF panel includes a relevant TF
mamba run -n chorus-legnet      python poster/verify_legnet_narrow.py        # for the narrow-window MPRA at the SNP
```

Each writes `_verified_<oracle>*.json` with raw_score, quantile_score,
and per-track metadata. These are the source of truth for the panel
numbers.

### 2. Look for a paper (optional)

```bash
# If paperclip is installed in your env:
paperclip search "rs12740374 SORT1 luciferase liver" -n 5
paperclip lookup pmc PMC3062476
```

For papers not in paperclip's index, fall back to PMC search +
WebFetch. Save verbatim quotes for the ground-truth bullets to a small
markdown file next to your config (e.g.
`musunuru_2010_extracted_claims.md` for the SORT1 example).

If no paper is available, leave `ground_truth` out of the config; the
generator will emit a small "no paper cited" placeholder card.

**Paper source flexibility.** The `ground_truth.paper` block accepts
any combination of `pmc_id`, `pubmed_id`, `doi`, `url`, and
`paperclip_id`. The generator picks the best canonical link in this
priority order: `url` ▸ `pmc_id` ▸ `pubmed_id` ▸ `doi`. The clickable
label shown to the reader is whichever ID you provided. If `paperclip`
is on PATH, the generator also runs `paperclip lookup` against PMC /
PubMed / DOI and adds a "✓ in paperclip" tag when the paper is indexed
locally.

### 3. Write the config

Copy `poster/variants/rs12740374.json` and edit. The schema:

```jsonc
{
  "variant": {
    "rsid":     "rsXXX",
    "chrom":    "chr1",
    "position": 109274968,
    "ref":      "G",
    "alt":      "T",
    "gene":     "SORT1"
  },
  "cell_type": "HepG2",
  "prompt": "free-text user prompt that motivated this panel",

  "rows": [
    {
      "layer_name": "1 · Chromatin",
      "unit":       "DNase",
      "values": {
        "chrombpnet":  { "value": "1.00", "pct": "≥99<sup>th</sup>", "track": "ChromBPNet head<br/>log₂FC +1.24" },
        "legnet":      null,
        "alphagenome": { "value": "1.00", "pct": "≥99<sup>th</sup>", "track": "log₂FC +1.34" }
      },
      "verdict": { "tag": "match", "text": "✓ both ↑" }
    },
    /* ... up to ~6 rows */
  ],

  "mechanism": {
    "predictions": [
      "Minor T allele <strong>opens chromatin</strong> (ChromBPNet +1.24, AG +1.34).",
      "Gains a <strong>C/EBPα binding</strong> site (BPNet TF +1.99, AG +2.7).",
      /* ... */
    ]
  },

  "ground_truth": {
    "title": "Ground truth (Author Year)",
    "paper": {
      "citation_text": "Author et al., <em>Journal</em> vol, page (year)",
      /* Provide ONE OR MORE of the following; generator picks the best link: */
      "pmc_id":        "PMC3062476",          // resolves to pmc.ncbi.nlm.nih.gov/articles/PMC3062476/
      "pubmed_id":     "20686566",            // resolves to pubmed.ncbi.nlm.nih.gov/20686566/
      "doi":           "10.1038/nature09266", // resolves to doi.org/...
      "url":           "https://...",         // any URL, takes precedence as the canonical link
      "paperclip_id":  "PMC3062476"           // paperclip /papers/<id> directory; falls back to pmc_id
    },
    "bullets": [
      "Verbatim claim 1 ...",
      "Verbatim claim 2 ..."
    ]
  },

  "igv_src":  "igv_only.html",
  "igv_meta": "SORT1 ±150 kb · zoomable to ±500 kb"
}
```

Verdict tag classes: `"match"` (green), `"single"` (blue), `"soft"`
(amber). Use `"soft"` for honest caveats (e.g. "magnitude underestimated").

Each `values.<oracle>` can be:
- a `{value, pct, track}` dict → renders a populated cell
- `null` → renders a "not scored" placeholder

### 4. Regenerate the IGV iframe target

`igv_only.html` is variant-specific (it embeds the per-bin signal data
from the canonical chorus example). For a new variant:

```bash
# Edit scripts/regenerate_multioracle.py's VARIANT dict to your variant,
# then run all three oracles and the consolidator. The resulting report
# is the source for extract_igv.py.
mamba run -n chorus-chrombpnet  python scripts/regenerate_multioracle.py --oracle chrombpnet
mamba run -n chorus-legnet      python scripts/regenerate_multioracle.py --oracle legnet
mamba run -n chorus-alphagenome python scripts/regenerate_multioracle.py --oracle alphagenome
mamba run -n chorus             python scripts/regenerate_multioracle.py --consolidate

python poster/extract_igv.py
```

The generator points at `poster/igv_only.html` by default; if you want
to keep multiple variants' IGV blobs side-by-side, rename and set
`igv_src` in each variant's config.

### 5. Render

```bash
python poster/generate_panel.py poster/variants/<your_variant>.json
# wrote poster/<rsid>_<gene>_panel_compact.html (~12 KB)

open poster/<rsid>_<gene>_panel_compact.html
```

Screenshot the rendered panel at high DPI (or `chromium --headless
--screenshot --window-size=620,1800` for reproducible captures) and
drop it into the right column of your poster.

## Suggested next step: promote to chorus examples

The natural home for this pattern is
`examples/walkthroughs/visualization/multi_oracle_poster_panel/`,
which would package up:

- `generate_panel.py`, `_panel_template_compact.html`
- `verify_predictions.py`, `verify_chrombpnet_cebpa.py`, `verify_legnet_narrow.py`
- `extract_igv.py`
- `variants/rs12740374.json` as the canonical example
- `notebook.ipynb` matching the chorus walkthrough convention (load a
  config → run predictions → render panel → display screenshot inline)
- `README.md` linking to `expected_numbers.json` for verifier-style
  reproduction

That promotion is a one-PR job; happy to do it on this branch if you
green-light.
