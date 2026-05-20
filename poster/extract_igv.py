"""Regenerate poster/igv_only.html from the canonical chorus multi-oracle report.

The poster panel iframes this file. We extract the live IGV.js bundle +
inlined per-bin signal data from the chorus example report on disk and
wrap it in a minimal page that posts its rendered height to the parent
window (so the iframe auto-sizes with no scrollbar).

Source : examples/walkthroughs/validation/SORT1_rs12740374_multioracle/
           rs12740374_SORT1_multioracle_report.html
Output : poster/igv_only.html

Requires the chorus repo at v0.5.6 or later (sliding-window prediction
extends ChromBPNet + LegNet signal across the full window).
"""
from __future__ import annotations
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(
    REPO_ROOT,
    "examples/walkthroughs/validation/SORT1_rs12740374_multioracle/"
    "rs12740374_SORT1_multioracle_report.html",
)
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "igv_only.html")

# Default locus for the poster panel: zoom in on SORT1.
# The full prediction window is left accessible via IGV zoom controls.
NEW_LOCUS = '"locus":"chr1:109150000-109450000"'


def main() -> int:
    if not os.path.exists(SRC):
        sys.exit(f"FATAL: canonical report not found: {SRC}\n"
                 "Run `mamba run -n chorus python scripts/regenerate_multioracle.py "
                 "--consolidate` after each oracle has produced its pickle.")

    with open(SRC) as f:
        s = f.read()

    igv_div_start = s.find('<div id="igv-multioracle"')
    igv_div_end   = s.find('</div>', igv_div_start) + len('</div>')
    bundle_start  = s.find('<script>!function(e,t)', igv_div_end)
    bundle_end    = s.find('</script>', bundle_start) + len('</script>')
    init_start    = s.find('<script>', bundle_end)
    init_end      = s.find('</script>', init_start) + len('</script>')

    if min(igv_div_start, bundle_start, init_start) < 0:
        sys.exit("FATAL: could not locate IGV blocks in canonical report")

    div_html    = s[igv_div_start:igv_div_end]
    bundle_html = s[bundle_start:bundle_end]
    init_html   = s[init_start:init_end]

    # Swap the default locus.
    m = re.search(r'"locus":"[^"]+"', init_html)
    if m:
        init_html = init_html.replace(m.group(0), NEW_LOCUS)

    page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Chorus IGV - rs12740374 / SORT1</title>
<style>
  html, body {{
    margin: 0; padding: 0; background: #fff;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', Segoe UI,
                 Helvetica, Arial, sans-serif;
    overflow: hidden;
  }}
  #igv-multioracle {{ padding: 4px 8px 8px; }}
</style>
<script>
  // Tell the parent (poster panel) the real rendered height so the
  // iframe can fit every track with no scrollbar.
  function reportHeight() {{
    const el = document.getElementById('igv-multioracle');
    const h = Math.max(
      document.documentElement.scrollHeight,
      document.body.scrollHeight,
      (el ? el.scrollHeight + 24 : 0)
    );
    try {{ window.parent.postMessage({{ type: 'igv-height', height: h }}, '*'); }} catch (e) {{}}
  }}
  window.addEventListener('load', () => {{
    [80, 250, 600, 1200, 2500, 5000].forEach(t => setTimeout(reportHeight, t));
    if ('ResizeObserver' in window) {{
      const ro = new ResizeObserver(() => reportHeight());
      ro.observe(document.body);
      const el = document.getElementById('igv-multioracle');
      if (el) ro.observe(el);
    }}
  }});
</script>
</head>
<body>
{div_html}
{bundle_html}
{init_html}
</body>
</html>
'''
    with open(DST, "w") as f:
        f.write(page)
    print(f"wrote {DST} ({os.path.getsize(DST) // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
