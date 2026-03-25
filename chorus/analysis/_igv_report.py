"""Generate an IGV.js-based interactive genome browser for HTML reports.

Embeds signal tracks as inline feature arrays in a self-contained HTML
page.  The user can zoom, pan, and interact with the browser.  Gene
annotations come from hg38 automatically via IGV's built-in genome.

Track data is downsampled to keep the HTML file size manageable while
preserving the shape of peaks and effects.
"""

import json
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# IGV.js CDN
_IGV_CDN = "https://cdn.jsdelivr.net/npm/igv@3.1.1/dist/igv.min.js"

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

    # Variant annotation track
    tracks.append({
        "name": f"Variant: {ref_allele}>{alt_allele}",
        "type": "annotation",
        "displayMode": "EXPANDED",
        "height": 25,
        "color": "red",
        "features": [{
            "chr": variant_chrom,
            "start": variant_pos - 1,
            "end": variant_pos + max(len(ref_allele), 1),
            "name": f"{variant_chrom}:{variant_pos:,} {ref_allele}>{alt_allele}",
        }],
    })

    # Signal tracks grouped by assay
    for assay_id in assay_ids:
        ref_track = ref_pred[assay_id]
        alt_track = alt_pred[assay_id]

        layer = classify_track_layer(ref_track)
        rgb = _LAYER_COLORS.get(layer, "70,130,180")

        t_start = ref_track.prediction_interval.reference.start
        t_res = ref_track.resolution

        ref_features = _downsample_to_features(
            ref_track.values, variant_chrom, t_start, t_res, bin_size,
        )
        alt_features = _downsample_to_features(
            alt_track.values, variant_chrom, t_start, t_res, bin_size,
        )

        group_id = assay_id.replace(":", "_").replace(" ", "_")

        # Merged overlay: ref (grey) + alt (coloured) on same panel
        tracks.append({
            "name": assay_id,
            "type": "merged",
            "height": 80,
            "tracks": [
                {
                    "type": "wig",
                    "name": f"{assay_id} ref",
                    "color": f"rgb({_REF_COLOR})",
                    "autoscale": True,
                    "autoscaleGroup": group_id,
                    "features": ref_features,
                },
                {
                    "type": "wig",
                    "name": f"{assay_id} alt",
                    "color": f"rgb({rgb})",
                    "autoscale": True,
                    "autoscaleGroup": group_id,
                    "features": alt_features,
                },
            ],
        })

    # ROI: variant position as red stripe across all tracks
    roi = [{
        "name": "Variant",
        "color": "rgba(255, 0, 0, 0.12)",
        "features": [{
            "chr": variant_chrom,
            "start": variant_pos - 1,
            "end": variant_pos + max(len(ref_allele), 1),
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

    html = f"""
<div id="igv-div" style="margin: 1rem 0; min-height: 400px;"></div>
<script src="{_IGV_CDN}"></script>
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
) -> list[dict]:
    """Downsample a signal array into IGV wig features.

    Aggregates bins by taking the mean over each output bin.
    Skips bins with near-zero signal to reduce JSON size.
    """
    n = len(values)
    vals = values.astype(np.float64)

    # Number of original bins per output bin
    bins_per = max(1, bin_size // resolution)

    features = []
    threshold = float(np.percentile(np.abs(vals[vals != 0]), 5)) if np.any(vals != 0) else 0

    for i in range(0, n, bins_per):
        chunk = vals[i:i + bins_per]
        v = float(np.mean(chunk))

        # Skip near-zero bins to reduce JSON size
        if abs(v) < threshold * 0.1:
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
