"""Generate an IGV.js-based interactive genome browser for HTML reports.

Embeds signal tracks as inline feature arrays in a self-contained HTML
page.  The user can zoom, pan, and interact with the browser.  Gene
annotations come from hg38 automatically via IGV's built-in genome.

Track data is downsampled to keep the HTML file size manageable while
preserving the shape of peaks and effects.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# IGV.js: prefer local cached copy (inlined), fall back to CDN.
#
# The committed HTML reports need IGV.js to render the embedded genome
# browser. To stay robust when the viewer is offline / behind a proxy / on
# a network that MITMs TLS (observed during the 2026-04-16 audit:
# `net::ERR_CERT_AUTHORITY_INVALID` on 2/19 reports), we lazy-download
# the bundle once into ``~/.chorus/lib/igv.min.js`` the first time
# ``build_igv_html`` runs on a machine, then inline the JS into every
# subsequent report. If the download fails we silently fall back to the
# CDN <script> tag so report generation still succeeds.
_IGV_CDN = "https://cdn.jsdelivr.net/npm/igv@3.1.1/dist/igv.min.js"
_IGV_LOCAL = Path.home() / ".chorus" / "lib" / "igv.min.js"


def _ensure_igv_local() -> Path | None:
    """Ensure ``_IGV_LOCAL`` exists; download it from the CDN on first use.

    Returns the local path when the file is available, ``None`` if the
    download failed (callers then fall back to the CDN <script> tag).
    """
    if _IGV_LOCAL.exists() and _IGV_LOCAL.stat().st_size > 0:
        return _IGV_LOCAL
    try:
        from chorus.utils.http import download_with_resume
        _IGV_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        download_with_resume(_IGV_CDN, _IGV_LOCAL, label="igv.min.js")
        if _IGV_LOCAL.exists() and _IGV_LOCAL.stat().st_size > 0:
            logger.info("Cached igv.min.js to %s — future reports will inline it.", _IGV_LOCAL)
            return _IGV_LOCAL
    except Exception as exc:
        logger.warning(
            "Could not pre-cache igv.min.js (%s); reports will reference %s at view time.",
            exc, _IGV_CDN,
        )
    return None

# Vivid alt colours that contrast strongly with the grey ref
_LAYER_COLORS = {
    "chromatin_accessibility": "0,100,220",    # bright blue (DNASE/ATAC)
    "tf_binding":              "220,30,30",     # bright red (ChIP-TF)
    "histone_marks":           "200,50,160",    # magenta (ChIP-Histone)
    "tss_activity":            "230,120,0",     # bright orange (CAGE)
    "gene_expression":         "120,50,200",    # purple (RNA)
    "promoter_activity":       "230,120,0",     # orange (LentiMPRA)
    "splicing":                "140,86,75",     # brown
    "regulatory_classification": "0,170,190",   # teal (Sei)
}

_REF_COLOR = "180,180,180"  # light grey — strong contrast with vivid alt

# Layer-aware CDF percentile thresholds for IGV visualization.
# floor_pctile = noise threshold (anything below maps to 0).
# peak_pctile  = "1.0" reference point.
#
# Sharp signals (CAGE, TF, DNASE) use floor=p95 / peak=p99 — captures
# all real peaks while suppressing model noise.  Broad histone marks
# use floor=p90 / peak=p99 to preserve their domain shape.
_LAYER_FLOOR_PCTILE = {
    "tss_activity":              0.95,  # CAGE/PRO-CAP — sharp TSS peaks
    "tf_binding":                0.95,  # ChIP-TF — sharp binding peaks
    "chromatin_accessibility":   0.95,  # DNASE/ATAC — focused peaks
    "splicing":                  0.95,  # SPLICE — sharp signals
    "histone_marks":             0.90,  # ChIP-Histone — broad domains
    "gene_expression":           0.90,  # RNA-seq — broad coverage
    "promoter_activity":         0.95,
    "regulatory_classification": 0.95,
}
_PEAK_PCTILE = 0.99
_DEFAULT_FLOOR_PCTILE = 0.95
# Display max: tall enough to show strong peaks (>>p99) without
# saturating most bins.  1.0 = p99 (top 1% threshold), so 3.0 captures
# 3x stronger than the genome-wide top 1%.  Bins above 3.0 clip but
# this is rare for real biology.
_DISPLAY_MAX = 3.0


def build_igv_html(
    ref_pred,
    alt_pred,
    variant_chrom: str,
    variant_pos: int,
    ref_allele: str = "",
    alt_allele: str = "",
    gene_name: Optional[str] = None,
    genome: str = "hg38",
    bin_size: int = 0,
    normalizer=None,
    oracle_name: Optional[str] = None,
    modification_region: Optional[tuple[int, int]] = None,
) -> str:
    """Build the IGV.js browser configuration as an HTML fragment.

    Args:
        ref_pred: Reference OraclePrediction.
        alt_pred: Alternate OraclePrediction.
        variant_chrom: Chromosome.
        variant_pos: Variant position.
        ref_allele: Reference allele string.
        alt_allele: Alternate allele string.
        gene_name: Gene to mention in the header.
        genome: IGV genome identifier (default hg38).
        bin_size: Downsample bin size in bp.  0 = auto-detect.
        normalizer: Optional QuantileNormalizer with baseline backgrounds.
            When provided, signal values are mapped to genome-wide activity
            percentiles [0, 1], making all tracks directly comparable.
        oracle_name: Oracle name for baseline lookup (required if normalizer given).

    Returns:
        HTML string containing the IGV.js browser div + script.
    """
    from .scorers import classify_track_layer

    assay_ids = list(ref_pred.keys())
    if not assay_ids:
        return ""

    # Determine prediction window
    first = ref_pred[assay_ids[0]]
    pred_start = first.prediction_interval.reference.start
    pred_end = first.prediction_interval.reference.end
    window_bp = pred_end - pred_start

    # Auto bin size: target ~3000 features per track
    if bin_size <= 0:
        bin_size = max(1, window_bp // 3000)

    # Build tracks
    tracks = []

    # Variant / modification annotation track.
    # For region swaps and insertions, highlight the full affected region.
    # For point variants, highlight the single nucleotide position.
    if modification_region is not None:
        marker_start, marker_end = modification_region
        marker_label = f"{variant_chrom}:{marker_start+1:,}-{marker_end:,} ({ref_allele}>{alt_allele})"
    else:
        marker_start = variant_pos - 1
        marker_end = variant_pos + max(len(ref_allele), 1)
        marker_label = f"{variant_chrom}:{variant_pos:,} {ref_allele}>{alt_allele}"

    tracks.append({
        "name": f"Modification: {ref_allele}>{alt_allele}",
        "type": "annotation",
        "displayMode": "EXPANDED",
        "height": 25,
        "color": "red",
        "features": [{
            "chr": variant_chrom,
            "start": marker_start,
            "end": marker_end,
            "name": marker_label,
        }],
    })

    # When a PerTrackNormalizer is available, rescale raw bin values
    # using CDF-derived noise floor (p95) and peak threshold (p99).
    # This preserves peak shape (linear transform) while making tracks
    # comparable across cell types: 1.0 = top 1% of bins genome-wide.
    # Falls back to raw autoscale when no normalizer is available.
    use_floor = normalizer is not None and oracle_name is not None

    for assay_id in assay_ids:
        ref_track = ref_pred[assay_id]
        alt_track = alt_pred[assay_id]

        layer = classify_track_layer(ref_track)
        rgb = _LAYER_COLORS.get(layer, "70,130,180")

        t_start = ref_track.prediction_interval.reference.start
        t_res = ref_track.resolution

        ref_vals = ref_track.values
        alt_vals = alt_track.values

        # Apply layer-aware floor-subtract + rescale when available
        floor_ok = False
        if use_floor:
            from .normalization import PerTrackNormalizer
            if isinstance(normalizer, PerTrackNormalizer):
                floor_p = _LAYER_FLOOR_PCTILE.get(layer, _DEFAULT_FLOOR_PCTILE)
                ref_fl = normalizer.perbin_floor_rescale_batch(
                    oracle_name, assay_id, ref_vals,
                    floor_pctile=floor_p,
                    peak_pctile=_PEAK_PCTILE,
                    max_value=_DISPLAY_MAX,
                )
                if ref_fl is not None:
                    alt_fl = normalizer.perbin_floor_rescale_batch(
                        oracle_name, assay_id, alt_vals,
                        floor_pctile=floor_p,
                        peak_pctile=_PEAK_PCTILE,
                        max_value=_DISPLAY_MAX,
                    )
                    ref_vals = ref_fl
                    alt_vals = alt_fl
                    floor_ok = True

        ref_features = _downsample_to_features(
            ref_vals, variant_chrom, t_start, t_res, bin_size,
            skip_zeros=not floor_ok,
        )
        alt_features = _downsample_to_features(
            alt_vals, variant_chrom, t_start, t_res, bin_size,
            skip_zeros=not floor_ok,
        )

        group_id = assay_id.replace(":", "_").replace(" ", "_")
        if floor_ok:
            scale_cfg = {"min": 0, "max": _DISPLAY_MAX, "autoscale": False}
            name_suffix = ""
        else:
            scale_cfg = {"autoscale": True, "autoscaleGroup": group_id}
            name_suffix = ""

        # Build a human-readable display name from track metadata
        display_name = assay_id
        meta = getattr(ref_track, "metadata", None)
        if meta and isinstance(meta, dict) and meta.get("description"):
            display_name = meta["description"]
        elif hasattr(ref_track, "assay_type") and hasattr(ref_track, "cell_type"):
            display_name = f"{ref_track.assay_type}:{ref_track.cell_type}"

        # Merged overlay: ref (grey) + alt (coloured) on same panel
        tracks.append({
            "name": f"{display_name}{name_suffix}",
            "type": "merged",
            "height": 80,
            "tracks": [
                {
                    "type": "wig",
                    "name": f"{display_name} ref",
                    "color": f"rgb({_REF_COLOR})",
                    **scale_cfg,
                    "features": ref_features,
                },
                {
                    "type": "wig",
                    "name": f"{display_name} alt",
                    "color": f"rgb({rgb})",
                    **scale_cfg,
                    "features": alt_features,
                },
            ],
        })

    # ROI: red stripe across all tracks highlighting the modification
    roi = [{
        "name": "Modification",
        "color": "rgba(255, 0, 0, 0.12)",
        "features": [{
            "chr": variant_chrom,
            "start": marker_start,
            "end": marker_end,
        }],
    }]

    # Initial locus: full prediction window
    locus = f"{variant_chrom}:{pred_start}-{pred_end}"

    igv_options = {
        "genome": genome,
        "locus": locus,
        "showRuler": True,
        "showNavigation": True,
        "showCenterGuide": True,
        "roi": roi,
        "tracks": tracks,
    }

    # Build HTML fragment
    options_json = json.dumps(igv_options, separators=(",", ":"))

    # Inline IGV.js from local cache (no network needed) or fall back to CDN.
    local = _ensure_igv_local()
    if local is not None:
        igv_js = local.read_text()
        igv_script_tag = f"<script>{igv_js}</script>"
    else:
        igv_script_tag = f'<script src="{_IGV_CDN}"></script>'

    html = f"""
<div id="igv-div" style="margin: 1rem 0; min-height: 400px;"></div>
{igv_script_tag}
<script>
(async function() {{
    try {{
        const browser = await igv.createBrowser(
            document.getElementById("igv-div"),
            {options_json}
        );
        console.log("IGV browser created successfully");
    }} catch(e) {{
        console.error("IGV error:", e);
        document.getElementById("igv-div").innerHTML =
            '<p style="color:red;padding:1rem">Error loading IGV browser: ' + e.message + '</p>';
    }}
}})();
</script>
"""
    return html


def _downsample_to_features(
    values: np.ndarray,
    chrom: str,
    start: int,
    resolution: int,
    bin_size: int,
    skip_zeros: bool = True,
) -> list[dict]:
    """Downsample a signal array into IGV wig features.

    Aggregates bins by taking the mean over each output bin.
    When *skip_zeros* is True (default for raw data), bins with near-zero
    signal are omitted to reduce JSON size.  Set to False for
    percentile-normalized data to avoid gaps.
    """
    n = len(values)
    vals = values.astype(np.float64)

    # Number of original bins per output bin
    bins_per = max(1, bin_size // resolution)

    features = []
    if skip_zeros:
        threshold = float(np.percentile(np.abs(vals[vals != 0]), 5)) if np.any(vals != 0) else 0
    else:
        threshold = -1  # never skip

    for i in range(0, n, bins_per):
        chunk = vals[i:i + bins_per]
        v = float(np.mean(chunk))

        # Skip near-zero bins to reduce JSON size (only for raw data)
        if skip_zeros and abs(v) < threshold * 0.1:
            continue

        feat_start = start + i * resolution
        feat_end = start + min(i + bins_per, n) * resolution

        features.append({
            "chr": chrom,
            "start": feat_start,
            "end": feat_end,
            "value": round(v, 4),
        })

    return features
