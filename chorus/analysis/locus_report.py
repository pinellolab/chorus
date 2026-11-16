"""One igv.js locus report, built from predictions that earlier MCP calls already made.

The usual conversation scores a variant with several oracles (``predict_variant_effect``, once per oracle, for a
chosen set of tracks) and then wants them side by side over the same stretch of genome. This module is that last
step: it reuses what is on disk and runs no model.

How earlier results are found
-----------------------------
``predict_variant_effect`` saves a reference and an alternate bedgraph per track, named only by track id, so a later
call for another variant overwrites them and nothing says which oracle or variant a file belongs to. It therefore also
writes a **manifest** per (variant, oracle) -- ``predictions_<chrom>_<pos>_<ref>_<alt>_<oracle>.json`` -- recording,
for every track, the files, the genomic interval predicted, the resolution and a fingerprint (size and mtime) of each
file. :func:`build_locus_report` selects tracks by oracle and assay id from those manifests and refuses a file whose
fingerprint no longer matches, because that means a later call replaced it with another variant's prediction.

What it draws
-------------
Over one window -- AlphaGenome's 1,048,576 bp output window by default (taken from the selected AlphaGenome track's own
interval if there is one, else centred on the variant) -- one merged track per selection, reference in blue and
alternate in orange (reference drawn last and opaque, so orange shows only where the alternate is higher), binned to
128 bp, plus the GENCODE genes and a band on the variant. Tracks may combine several assay ids (the mean of the + and
- strands of CAGE or RNA-seq), and GPN-Star conservation can be added over the whole window. A prediction that covers
less than the window (a 2 kb ChromBPNet call) is drawn where it exists, and the coverage is reported.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: AlphaGenome's output window, the default width of the report.
WINDOW_BP = 1_048_576
BIN_BP = 128
AGGREGATES = ("mean", "max", "sum")
MAX_TRACKS = 30
ALPHAGENOME = ("alphagenome", "alphagenome_pt")
MANIFEST_SCHEMA = "chorus_predictions/1"
REF_RGBA, ALT_RGBA, GREY = "rgba(0,114,178,1)", "rgba(213,94,0,1)", "rgb(122,135,148)"

_BAD_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


# ---------------------------------------------------------------------------
# Manifest: written by predict_variant_effect, read here
# ---------------------------------------------------------------------------

def _variant_tuple(info: dict) -> tuple[str, int, str, str]:
    chrom, pos = str(info["position"]).split(":")
    if "alleles" in info:
        ref, alts = info["alleles"][0], list(info["alleles"][1:])
    else:
        ref, alts = info.get("ref", ""), list(info.get("alts", []))
    return chrom, int(str(pos).replace(",", "")), str(ref), str(alts[0]) if alts else ""


def manifest_path(directory: str | Path, oracle: str, chrom: str, pos: int, ref: str, alt: str) -> Path:
    name = _BAD_NAME.sub("_", f"predictions_{chrom}_{pos}_{ref}_{alt}_{oracle}") + ".json"
    return Path(directory) / name


def _fingerprint(path: str | Path) -> list[int]:
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns]


def _describe(track) -> str:
    try:
        from .variant_report import _track_description
        d = _track_description(track)
        if d:
            return str(d)
    except Exception:  # noqa: BLE001 - a missing description is cosmetic
        pass
    at, ct = getattr(track, "assay_type", ""), getattr(track, "cell_type", "")
    return f"{at}:{ct}" if at and ct else str(getattr(track, "assay_id", ""))


def record_predictions(output_dir: str | Path, oracle_name: str, variant_info: dict, predictions: dict,
                       saved_files: dict) -> Optional[str]:
    """Add the tracks of one ``predict_variant_effect`` call to the (variant, oracle) manifest; return its path.

    Never raises into the caller: the prediction has been made and its bedgraphs written, and a missing manifest
    only means ``build_locus_report`` cannot find them (it says so, and explicit bedgraph paths still work).
    """
    try:
        chrom, pos, ref, alt = _variant_tuple(variant_info)
        alt_key = next(k for k in predictions if k != "reference")
        path = manifest_path(output_dir, oracle_name, chrom, pos, ref, alt)
        doc = {"schema": MANIFEST_SCHEMA, "oracle": oracle_name,
               "variant": {"chrom": chrom, "pos": pos, "ref": ref, "alt": alt}, "alt_key": alt_key, "tracks": {}}
        if path.exists():
            try:
                old = json.loads(path.read_text())
                if old.get("schema") == MANIFEST_SCHEMA:
                    doc["tracks"] = old.get("tracks", {})
            except (OSError, ValueError):
                pass
        ref_files, alt_files = saved_files.get("reference", {}), saved_files.get(alt_key, {})
        for assay_id, track in predictions[alt_key].items():
            if assay_id not in ref_files or assay_id not in alt_files:
                continue
            iv = getattr(getattr(track, "prediction_interval", None), "reference", None)
            doc["tracks"][assay_id] = {
                "assay_type": getattr(track, "assay_type", None), "cell_type": getattr(track, "cell_type", None),
                "description": _describe(track), "resolution": int(getattr(track, "resolution", 1) or 1),
                "interval": ({"chrom": str(iv.chrom), "start": int(iv.start), "end": int(iv.end)}
                             if iv is not None and hasattr(iv, "chrom") else None),
                "files": {"ref": ref_files[assay_id], "alt": alt_files[assay_id]},
                "fingerprint": {"ref": _fingerprint(ref_files[assay_id]), "alt": _fingerprint(alt_files[assay_id])},
            }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        os.replace(tmp, path)
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not write the prediction manifest: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Reading and binning
# ---------------------------------------------------------------------------

def parse_region(region: str) -> tuple[str, int, int]:
    """``chrom:start-end`` (1-based, inclusive, commas allowed) -> ``(chrom, lo, hi)`` as 0-based half-open."""
    m = re.fullmatch(r"\s*([\w.]+):([\d,]+)-([\d,]+)\s*", region)
    if not m:
        raise ValueError(f"region must look like chr1:108750680-109799255, got {region!r}")
    lo, hi = int(m.group(2).replace(",", "")) - 1, int(m.group(3).replace(",", ""))
    if hi <= lo:
        raise ValueError(f"region end must be after its start: {region!r}")
    return m.group(1), lo, hi


def centred_window(pos: int, width: int = WINDOW_BP) -> tuple[int, int]:
    """AlphaGenome's own interval for a variant at 1-based ``pos``: ``[pos - 1 - width/2, + width)``."""
    lo = pos - 1 - width // 2
    return lo, lo + width


def read_bedgraph(path: str | Path, chrom: str, lo: int, hi: int) -> np.ndarray:
    """Per-base values of a bedgraph over ``[lo, hi)`` (0-based); NaN where the file has no data."""
    first = Path(path).read_text(errors="ignore")[:200]
    skip = 1 if first.lstrip().startswith(("track", "browser")) else 0
    df = pd.read_csv(path, sep="\t", header=None, names=["c", "s", "e", "v"], usecols=[0, 1, 2, 3], skiprows=skip,
                     dtype={"c": str, "s": np.int64, "e": np.int64, "v": np.float64}, comment="#")
    df = df[df.c == chrom]
    s, e = np.maximum(df.s.to_numpy(), lo), np.minimum(df.e.to_numpy(), hi)
    keep = e > s
    s, e, v = s[keep], e[keep], df.v.to_numpy()[keep]
    out = np.full(hi - lo, np.nan, dtype=np.float32)
    if len(s) == 0:
        return out
    length = e - s
    starts = np.repeat(np.cumsum(length) - length, length)
    idx = np.repeat(s - lo, length) + (np.arange(int(length.sum())) - starts)
    out[idx] = np.repeat(v, length)
    return out


def bin_values(a: np.ndarray, bin_bp: int, how: str = "mean") -> np.ndarray:
    """Aggregate per-base values into ``bin_bp`` bins (the last bin is padded with NaN); NaN for an empty bin."""
    if how not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}, got {how!r}")
    n = -(-len(a) // bin_bp)
    g = np.full(n * bin_bp, np.nan, dtype=np.float64)
    g[:len(a)] = a
    g = g.reshape(n, bin_bp)
    ok = ~np.isnan(g)
    cnt = ok.sum(axis=1)
    if how == "max":
        out = np.where(ok, g, -np.inf).max(axis=1)
    else:
        out = np.where(ok, g, 0.0).sum(axis=1)
        if how == "mean":
            out = out / np.maximum(cnt, 1)
    return np.where(cnt > 0, out, np.nan)


# ---------------------------------------------------------------------------
# Track selection
# ---------------------------------------------------------------------------

def _oracle_label(oracle: str) -> str:
    from .layer_ranking import MODEL_LABELS
    return MODEL_LABELS.get(oracle.lower(), oracle)


class _Manifests:
    """The manifests of one variant, loaded on demand."""

    def __init__(self, directory: Path, chrom: str, pos: int, ref: str, alt: str):
        self.dir, self.v, self._cache = directory, (chrom, pos, ref, alt), {}

    def get(self, oracle: str) -> dict:
        if oracle not in self._cache:
            p = manifest_path(self.dir, oracle, *self.v)
            if not p.exists():
                chrom, pos, ref, alt = self.v
                raise ValueError(
                    f"no saved predictions for oracle {oracle!r} at {chrom}:{pos} {ref}>{alt} in {self.dir}. Run "
                    f"predict_variant_effect(oracle_name={oracle!r}, position='{chrom}:{pos}', ref_allele='{ref}', "
                    f"alt_alleles=['{alt}'], assay_ids=[...], region=<the window>) first; its bedgraphs and manifest are "
                    f"what this report reuses")
            self._cache[oracle] = json.loads(p.read_text())
        return self._cache[oracle]

    def entry(self, oracle: str, assay_id: str) -> dict:
        m = self.get(oracle)
        if assay_id not in m["tracks"]:
            have = sorted(m["tracks"])
            raise ValueError(f"{oracle}: no saved prediction for track {assay_id!r}; saved: "
                             + (", ".join(have[:12]) + (" ..." if len(have) > 12 else "") if have else "none"))
        e = m["tracks"][assay_id]
        for which in ("ref", "alt"):
            f = e["files"][which]
            if not Path(f).exists():
                raise ValueError(f"{oracle}: the saved {which} bedgraph for {assay_id!r} is gone ({f})")
            if _fingerprint(f) != e["fingerprint"][which]:
                raise ValueError(f"{oracle}: the saved {which} bedgraph for {assay_id!r} changed since it was recorded "
                                 f"-- a later predict_variant_effect call overwrote it (the files are named by track id "
                                 f"only). Run predict_variant_effect for these tracks again, then rebuild")
        return e


def _normalise_spec(spec: dict, i: int) -> dict:
    if not isinstance(spec, dict):
        raise ValueError(f"track {i + 1}: a track must be an object")
    out = dict(spec)
    ids = out.get("assay_ids") or ([out["assay_id"]] if out.get("assay_id") else None)
    explicit = out.get("ref_bedgraph") or out.get("alt_bedgraph")
    if explicit:
        if not (out.get("ref_bedgraph") and out.get("alt_bedgraph")):
            raise ValueError(f"track {i + 1}: give both ref_bedgraph and alt_bedgraph")
        if not out.get("name"):
            raise ValueError(f"track {i + 1}: a track made of explicit bedgraphs needs a name")
    else:
        if not out.get("oracle"):
            raise ValueError(f"track {i + 1}: oracle is required (the oracle_name used in predict_variant_effect)")
        if not ids:
            raise ValueError(f"track {i + 1}: assay_id (or assay_ids) is required")
        if len(ids) > 1 and out.get("combine", "mean") not in ("mean", "sum"):
            raise ValueError(f"track {i + 1}: combine must be 'mean' or 'sum'")
    out["assay_ids"] = ids
    out["aggregate"] = out.get("aggregate", "mean")
    if out["aggregate"] not in AGGREGATES:
        raise ValueError(f"track {i + 1}: aggregate must be one of {AGGREGATES}")
    out["scale"] = float(out.get("scale", 1.0))
    return out


def _resolve(spec: dict, manifests: _Manifests) -> dict:
    """Entries (files, interval, resolution, description) the spec points at."""
    if spec.get("ref_bedgraph"):
        return {"oracle": spec.get("oracle") or "custom", "entries": [
            {"assay_id": spec["name"], "files": {"ref": spec["ref_bedgraph"], "alt": spec["alt_bedgraph"]},
             "interval": None, "resolution": None, "description": spec["name"]}]}
    entries = []
    for aid in spec["assay_ids"]:
        e = dict(manifests.entry(spec["oracle"], aid))
        e["assay_id"] = aid
        entries.append(e)
    return {"oracle": spec["oracle"], "entries": entries}


def _read_pair(entries: list[dict], chrom: str, lo: int, hi: int, combine: str):
    refs, alts = [], []
    for e in entries:
        refs.append(read_bedgraph(e["files"]["ref"], chrom, lo, hi))
        alts.append(read_bedgraph(e["files"]["alt"], chrom, lo, hi))
    if len(entries) == 1:
        return refs[0], alts[0]
    f = np.nanmean if combine == "mean" else np.nansum
    with np.errstate(all="ignore"):
        return f(np.stack(refs), axis=0), f(np.stack(alts), axis=0)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def _features(chrom: str, lo: int, hi: int, bin_bp: int, vals: np.ndarray) -> list[dict]:
    return [{"chr": chrom, "start": lo + i * bin_bp, "end": min(lo + (i + 1) * bin_bp, hi), "value": float(f"{v:.4g}")}
            for i, v in enumerate(vals) if np.isfinite(v)]


def _merged_track(name: str, chrom: str, lo: int, hi: int, bin_bp: int, ref: np.ndarray, alt: np.ndarray, height: int) -> dict:
    top = float(f"{np.nanmax(np.maximum(np.nan_to_num(ref), np.nan_to_num(alt))) * 1.05:.2g}") or 1.0
    subs = [{"type": "wig", "name": f"{name} {a}", "color": c, "windowFunction": "max", "autoscale": False, "min": 0,
             "max": top, "features": _features(chrom, lo, hi, bin_bp, v)}
            for a, c, v in (("alt", ALT_RGBA, alt), ("ref", REF_RGBA, ref))]       # ref last, opaque: orange where alt is higher
    return {"name": name, "type": "merged", "alpha": 1, "height": height, "tracks": subs}


_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #fff; font-family: 'DejaVu Sans', 'Open Sans', Arial, sans-serif; color: #1F2933; }}
  .page {{ width: {width}px; max-width: 100%; padding: 18px 26px 14px 26px; background: #fff; }}
  .title {{ font-size: 22px; font-weight: 800; line-height: 1.15; margin: 0; }}
  .sub {{ font-size: 13px; color: #5F6B7A; margin: 6px 0 10px 0; display: flex; gap: 22px; flex-wrap: wrap; align-items: center; }}
  .sw {{ display: inline-block; width: 30px; height: 8px; margin-right: 7px; vertical-align: middle; }}
  .note {{ font-size: 12px; color: #5F6B7A; margin-top: 8px; }}
</style></head>
<body>
<div class="page" id="page">
  <h1 class="title">{title}</h1>
  <div class="sub"><span><span class="sw" style="background:{ref_c}"></span>{ref} (reference)</span>
    <span><span class="sw" style="background:{alt_c}"></span>{alt} (alternate)</span>{subs}</div>
  <div id="igv-locus"></div>
  {notes}
</div>
{igv_tag}
<script>
(async function() {{
  try {{
    window.igvBrowser = await igv.createBrowser(
      document.getElementById("igv-locus"),
      {config});
    document.body.setAttribute("data-ready", "1");
  }} catch (e) {{ console.error(e); document.body.setAttribute("data-ready", "error: " + e.message); }}
}})();
</script></body></html>
"""

_HIDE_CHROME_JS = """() => { const hide = ['igv-gear-menu-column', 'igv-track-drag-column', 'igv-scrollbar-column',
    'igv-sample-info-column', 'igv-sample-name-column', 'igv-navbar'];
  const walk = (root) => root.querySelectorAll('*').forEach(e => {
    if (hide.some(c => e.classList.contains(c))) e.style.setProperty('display', 'none', 'important');
    if (e.shadowRoot) walk(e.shadowRoot); });
  walk(document); }"""


def _igv_tag() -> str:
    from ._igv_report import _IGV_CDN, _ensure_igv_local
    local = _ensure_igv_local()
    return f"<script>{local.read_text()}</script>" if local is not None else f'<script src="{_IGV_CDN}"></script>'


def _local_genome_dir() -> Optional[Path]:
    """The directory holding a usable ``hg38.fa`` and ``hg38.fa.fai``, if this install has them."""
    try:
        from chorus.core.globals import CHORUS_GENOMES_DIR
        d = Path(CHORUS_GENOMES_DIR)
    except Exception:  # noqa: BLE001
        return None
    return d if (d / "hg38.fa").exists() and (d / "hg38.fa.fai").exists() else None


def _screenshot(png_path: Path, scale: float, render, expected_bp: int) -> str:
    """Render the report in headless Chromium and save the page as a PNG; returns how the sequence was sourced.

    igv.js needs a sequence source to start. The remote 2bit works online and fails at once when external requests are
    blocked, and Chorus's one-contig placeholder FASTA makes igv believe chr1 is 1,000 bp long, which collapses any
    window wider than that. So when ``hg38.fa`` is on disk the page is served from a temporary directory beside it
    (same origin, the recipe documented in ``igv_reference_config``) and nothing external is needed.
    """
    import http.server
    import tempfile
    import threading

    from playwright.sync_api import sync_playwright

    genome = _local_genome_dir()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        saved = os.environ.get("CHORUS_IGV_SEQUENCE_URL")
        try:
            if genome:
                (td / "hg38.fa").symlink_to(genome / "hg38.fa")
                (td / "hg38.fa.fai").symlink_to(genome / "hg38.fa.fai")
                os.environ["CHORUS_IGV_SEQUENCE_URL"] = "hg38.fa"
            (td / "report.html").write_text(render(None))
        finally:
            if saved is None:
                os.environ.pop("CHORUS_IGV_SEQUENCE_URL", None)
            else:
                os.environ["CHORUS_IGV_SEQUENCE_URL"] = saved
        httpd = None
        if genome:
            class Handler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *a, **kw):
                    super().__init__(*a, directory=str(td), **kw)

                def log_message(self, *a):
                    pass
            httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            url = f"http://127.0.0.1:{httpd.server_address[1]}/report.html"
        else:
            url = "file://" + str(td / "report.html")
        try:
            with sync_playwright() as p:
                b = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
                pg = b.new_page(viewport={"width": 1500, "height": 1200}, device_scale_factor=scale)
                pg.goto(url)
                pg.wait_for_function("document.body.getAttribute('data-ready')", timeout=90000)
                state = pg.evaluate("document.body.getAttribute('data-ready')")
                if state != "1":
                    raise RuntimeError(f"igv.js could not start ({state}). It needs a sequence source: online access, "
                                       f"or hg38.fa + hg38.fa.fai in the chorus genomes directory")
                pg.wait_for_timeout(3000)
                shown = pg.evaluate("() => { const f = window.igvBrowser.referenceFrameList[0]; return f.end - f.start; }")
                if abs(shown - expected_bp) > 0.01 * expected_bp:
                    # igv clamps the view to the chromosome length it was given; a placeholder sequence makes that 1 kb
                    raise RuntimeError(f"igv.js is showing {shown:,.0f} bp instead of the {expected_bp:,} bp window "
                                       f"(its reference gave chr1 the wrong length); a real hg38.fa is needed")
                pg.evaluate(_HIDE_CHROME_JS)
                pg.wait_for_timeout(500)
                pg.locator("#page").screenshot(path=str(png_path))
                b.close()
        finally:
            if httpd is not None:
                httpd.shutdown()
    return "local hg38.fa" if genome else "remote reference"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_locus_report(
    position: str,
    ref_allele: str,
    alt_allele: str,
    tracks: Sequence[dict],
    output_dir: str | Path,
    *,
    source_dir: str | Path | None = None,
    region: Optional[str] = None,
    bin_bp: int = BIN_BP,
    conservation: bool = False,
    title: Optional[str] = None,
    clip_negative: bool = True,
    track_height: int = 80,
    screenshot: bool = False,
    screenshot_scale: float = 2.0,
) -> dict:
    """Build one igv.js report over a ~1 Mb window from predictions saved by earlier ``predict_variant_effect`` calls.

    Args:
        position: ``"chr1:109274968"``.
        ref_allele / alt_allele: The variant the predictions were made for (they select the manifests).
        tracks: What to draw, one entry per track, in order. Each is an object with
            ``oracle`` (the ``oracle_name`` of the earlier call) and ``assay_id``, or ``assay_ids`` plus
            ``combine`` (``"mean"`` or ``"sum"``) to merge several -- the + and - strands of a CAGE or
            RNA-seq track; or explicit ``ref_bedgraph`` and ``alt_bedgraph`` paths with a ``name``. Optional:
            ``name``, ``aggregate`` (``mean`` | ``max`` | ``sum``, how per-base values fall into a bin; default
            ``mean``; ``sum`` turns per-base counts into counts per bin, ``max`` keeps sharp peaks) and
            ``scale`` (a factor applied after binning).
        output_dir: Where the HTML (and PNG) are written.
        source_dir: Where the earlier calls saved their files (default ``output_dir``).
        region: ``"chr1:108750680-109799255"`` (1-based, inclusive). Default: the interval of the first
            AlphaGenome track selected, else AlphaGenome's 1,048,576 bp window centred on the variant.
        bin_bp: Bin width; 128 matches AlphaGenome's coarse resolution.
        conservation: Add GPN-Star conservation (1 - entropy) over the window if its bigwig is already
            downloaded (it is never downloaded here: it is ~25 GB).
        title: Page title.
        clip_negative: Draw values below 0 as 0 (a count cannot be negative; BPNet's count head dips below zero
            in the background). The number of clipped bins is reported.
        screenshot: Also write a PNG (needs Playwright and Chromium).

    Returns:
        ``files``, ``window`` (chrom, lo, hi, source), ``tracks`` (name, source tracks, coverage of the window,
        bin aggregation) and ``notes``.
    """
    chrom, pos_s = position.split(":")
    pos = int(pos_s.replace(",", ""))
    if not 1 <= int(bin_bp) <= 5000:
        raise ValueError("bin_bp must be between 1 and 5000")
    bin_bp = int(bin_bp)
    tracks = list(tracks or [])
    if not tracks:
        raise ValueError("tracks is empty: select at least one saved prediction to draw")
    if len(tracks) > MAX_TRACKS:
        raise ValueError(f"at most {MAX_TRACKS} tracks per report, got {len(tracks)}")
    source = Path(source_dir or output_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifests = _Manifests(source, chrom, pos, ref_allele, alt_allele)
    specs = [_normalise_spec(t, i) for i, t in enumerate(tracks)]
    resolved = [_resolve(s, manifests) for s in specs]
    notes: list[str] = []

    # the window
    if region:
        w_chrom, lo, hi = parse_region(region)
        if w_chrom != chrom:
            raise ValueError(f"region is on {w_chrom} but the variant is on {chrom}")
        origin = "region argument"
    else:
        ag = next((e["interval"] for r in resolved if r["oracle"].lower() in ALPHAGENOME for e in r["entries"]
                   if e.get("interval") and e["interval"]["end"] - e["interval"]["start"] == WINDOW_BP), None)
        if ag:
            lo, hi, origin = int(ag["start"]), int(ag["end"]), "interval of the selected AlphaGenome track"
        else:
            lo, hi = centred_window(pos)
            origin = "AlphaGenome's 1,048,576 bp window centred on the variant"
    if not lo <= pos - 1 < hi:
        raise ValueError(f"the window {chrom}:{lo + 1}-{hi} does not contain the variant at {pos}")

    igv_tracks, summary = [], []
    for spec, res in zip(specs, resolved):
        ents = res["entries"]
        combine = spec.get("combine", "mean")
        ref1, alt1 = _read_pair(ents, chrom, lo, hi, combine)
        ref = bin_values(ref1, bin_bp, spec["aggregate"]) * spec["scale"]
        alt = bin_values(alt1, bin_bp, spec["aggregate"]) * spec["scale"]
        if not (np.isfinite(ref).any() or np.isfinite(alt).any()):
            raise ValueError(f"{spec.get('name') or ents[0]['assay_id']}: the saved prediction has no data in "
                             f"{chrom}:{lo + 1}-{hi} (it was predicted for a different region)")
        coverage = float(np.isfinite(ref).mean())
        name = spec.get("name")
        if not name:
            desc = ents[0].get("description") or ents[0]["assay_id"]
            name = f"{_oracle_label(res['oracle'])} · {desc}" + (f" ({len(ents)}-track {combine})" if len(ents) > 1 else "")
        if clip_negative:
            neg = float(np.nanmean(ref < 0)) if np.isfinite(ref).any() else 0.0
            if neg > 0:
                notes.append(f"{name}: {neg:.1%} of bins were below 0 and are drawn at 0")
            ref, alt = np.where(np.isfinite(ref), np.maximum(ref, 0), np.nan), np.where(np.isfinite(alt), np.maximum(alt, 0), np.nan)
        if coverage < 0.99:
            notes.append(f"{name}: the saved prediction covers {coverage:.0%} of the window; predict_variant_effect "
                         f"with region={chrom}:{lo + 1}-{hi} to draw all of it")
        igv_tracks.append(_merged_track(name, chrom, lo, hi, bin_bp, ref, alt, track_height))
        summary.append({"name": name, "oracle": res["oracle"], "assay_ids": [e["assay_id"] for e in ents],
                        "combine": combine if len(ents) > 1 else None, "aggregate": spec["aggregate"],
                        "scale": spec["scale"], "coverage": round(coverage, 4), "n_bins": int(np.isfinite(ref).sum())})

    if conservation:
        try:
            from . import conservation as cons
            if not cons._has_bigwig("gpn_star"):
                notes.append("conservation skipped: the GPN-Star bigwig is not downloaded (download_annotation "
                             "fetches it; ~25 GB, never done implicitly here)")
            else:
                e = cons.read_bigwig_values(chrom, lo + 1, hi, bw_path=cons.gpn_star_bigwig_path(), preserve_nan=True)
                v = bin_values(np.clip(1.0 - e, 0.0, 1.0), bin_bp, "mean")        # NaN (assembly gap) stays NaN, not 1
                igv_tracks.append({"name": "GPN-Star conservation", "type": "wig", "color": GREY, "height": 48, "min": 0,
                                   "max": 1, "autoscale": False, "windowFunction": "mean",
                                   "features": _features(chrom, lo, hi, bin_bp, v)})
                summary.append({"name": "GPN-Star conservation", "oracle": None, "assay_ids": ["gpn_star"],
                                "coverage": round(float(np.isfinite(v).mean()), 4)})
        except Exception as exc:  # noqa: BLE001
            notes.append(f"conservation skipped: {exc}")

    from ._igv_report import igv_browser_config

    locus = f"{chrom}:{lo + 1}-{hi}"
    roi = [{"name": "Variant", "color": "rgba(213,94,0,0.18)", "features": [{"chr": chrom, "start": pos - 1, "end": pos}]}]
    span = (hi - lo) / 1000
    title = title or f"{chrom}:{pos:,} {ref_allele}>{alt_allele} across a {span:,.0f} kb window"
    sub = "".join(f"<span>{s}</span>" for s in
                  [f"{len(summary)} tracks", f"{bin_bp} bp bins"] + (["conservation: GPN-Star"] if conservation and
                                                                      any(x["name"] == "GPN-Star conservation" for x in summary) else []))
    note_html = "".join(f'<div class="note">{n}</div>' for n in notes)

    def render(bundle_sequence: Optional[bool] = None) -> str:
        cfg = igv_browser_config(locus, igv_tracks, roi, genome="hg38", gene_track=True, bundle_sequence=bundle_sequence)
        cfg["showCenterGuide"] = False
        return _PAGE.format(title=title, width=1400, ref_c=REF_RGBA, alt_c=ALT_RGBA, ref=ref_allele, alt=alt_allele,
                            subs=sub, notes=note_html, igv_tag=_igv_tag(), config=json.dumps(cfg, separators=(",", ":")))

    stem = _BAD_NAME.sub("_", f"{chrom}_{pos}_{ref_allele}_{alt_allele}_locus_report")
    html_path = out / f"{stem}.html"
    html_path.write_text(render())
    files = {"html": str(html_path)}
    if screenshot:
        png = out / f"{stem}.png"
        try:
            _screenshot(png, screenshot_scale, render, hi - lo)
            files["png"] = str(png)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"screenshot failed: {exc}")
    return {"files": files, "window": {"chrom": chrom, "lo": lo, "hi": hi, "bp": hi - lo, "source": origin,
                                       "locus": locus},
            "variant": {"chrom": chrom, "pos": pos, "ref": ref_allele, "alt": alt_allele},
            "tracks": summary, "bin_bp": bin_bp, "notes": notes}
