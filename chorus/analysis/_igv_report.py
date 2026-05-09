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

# IGV.js is bundled as a package resource at
# ``chorus/analysis/static/igv.min.js`` so every install has an offline-
# usable copy without any network round-trip. The legacy CDN + HF
# fallback paths remain as secondary options in case a downstream
# consumer stripped the static file from the wheel.
#
# Inlining the JS into every report makes the committed HTMLs
# self-contained (viewable offline, through SSL-MITM proxies, on air-gapped
# hosts). The CDN <script> fallback is the last resort and only triggers
# when both the bundled copy and both network paths fail.
_IGV_CDN = "https://cdn.jsdelivr.net/npm/igv@3.1.1/dist/igv.min.js"
_IGV_LOCAL = Path.home() / ".chorus" / "lib" / "igv.min.js"
_IGV_BUNDLED = Path(__file__).parent / "static" / "igv.min.js"
# HuggingFace mirror — tertiary fallback for unusual installs where the
# bundled resource is missing (e.g. stripped by a packer) and stdlib
# urllib is blocked by a MITM proxy.
_IGV_HF_REPO = "lucapinello/chorus-backgrounds"
_IGV_HF_FILENAME = "igv.min.js"


def _ensure_igv_local() -> Path | None:
    """Return a path to ``igv.min.js`` that callers can read + inline.

    Resolution order:
      1. ``chorus/analysis/static/igv.min.js`` — bundled with the
         package. Always present in a standard install; no network
         touched.
      2. ``~/.chorus/lib/igv.min.js`` — legacy cache from earlier chorus
         versions. Kept for continuity.
      3. CDN via stdlib ``urllib`` (``download_with_resume``).
      4. HuggingFace dataset mirror via ``huggingface_hub``.

    Returns the local path when the file is available, ``None`` if all
    four sources failed (callers then fall back to a CDN ``<script>``
    tag in the rendered HTML — reports remain viewable online).
    """
    # 1. Bundled package resource (fast path — no I/O beyond the stat).
    if _IGV_BUNDLED.exists() and _IGV_BUNDLED.stat().st_size > 0:
        return _IGV_BUNDLED

    # 2. Legacy user cache from pre-v13 installs.
    if _IGV_LOCAL.exists() and _IGV_LOCAL.stat().st_size > 0:
        return _IGV_LOCAL

    # Bundled file missing (stripped by a packer?) and no legacy cache.
    # Fall back to the download paths to stay functional.
    _IGV_LOCAL.parent.mkdir(parents=True, exist_ok=True)

    # 3. CDN via stdlib urllib.
    try:
        from chorus.utils.http import download_with_resume
        download_with_resume(_IGV_CDN, _IGV_LOCAL, label="igv.min.js")
        if _IGV_LOCAL.exists() and _IGV_LOCAL.stat().st_size > 0:
            logger.info("Cached igv.min.js from CDN to %s.", _IGV_LOCAL)
            return _IGV_LOCAL
    except Exception as exc:
        logger.debug("CDN fetch of igv.min.js failed (%s); trying HF mirror.", exc)

    # 4. HuggingFace mirror (works through SSL-MITM proxies where stdlib
    # urllib fails — huggingface_hub uses httpx + certifi).
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            _IGV_HF_REPO,
            filename=_IGV_HF_FILENAME,
            repo_type="dataset",
            local_dir=str(_IGV_LOCAL.parent),
        )
        dp = Path(downloaded)
        if dp != _IGV_LOCAL and dp.exists():
            dp.replace(_IGV_LOCAL)
        if _IGV_LOCAL.exists() and _IGV_LOCAL.stat().st_size > 0:
            logger.info("Cached igv.min.js from HuggingFace mirror to %s.", _IGV_LOCAL)
            return _IGV_LOCAL
    except Exception as exc:
        logger.warning(
            "igv.min.js unavailable: bundled resource missing, CDN and HF "
            "mirror both failed (%s); reports will reference %s at view time.",
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
    "chromatin_accessibility":   0.90,  # DNASE/ATAC — lowered from 0.95 so the peak base/shoulder displays alongside the top
    "splicing":                  0.95,  # SPLICE — sharp signals
    "histone_marks":             0.90,  # ChIP-Histone — broad domains
    "gene_expression":           0.90,  # RNA-seq — broad coverage
    "promoter_activity":         0.85,  # LentiMPRA via LegNet — predictions are even sparser than chromatin (most of genome is not a strong promoter); floor at p85 keeps moderately-active promoters visible.  Note: LegNet's summary_cdfs is signed, so repressive values still clip to 0; lowering the floor expands only the positive half.
    "regulatory_classification": 0.95,
}
_PEAK_PCTILE = 0.99
_DEFAULT_FLOOR_PCTILE = 0.95
# Display max: tall enough to show strong peaks (>>p99) without
# saturating most bins.  1.0 = p99 (top 1% threshold), so 3.0 captures
# 3x stronger than the genome-wide top 1%.  Bins above 3.0 clip but
# this is rare for real biology.
_DISPLAY_MAX = 3.0
_HIGH_RES_ORACLES = ["chrombpnet", "legnet"] # for visualization mean vs max pooling


def rescale_for_display(
    values,
    layer: str,
    normalizer=None,
    oracle_name: str | None = None,
    assay_id: str | None = None,
):
    """Single-track display rescale.  Canonical helper used by every
    track-rendering path (IGV WIG, matplotlib PNG, CoolBox, notebooks)
    so they share one source of truth for normalization semantics.

    Returns ``(out_values, cfg)`` where ``cfg`` is a dict with:

    - ``rescaled`` (bool): True iff CDF-based rescale was applied.
      False means the values were returned unchanged and the caller
      should autoscale per-track.
    - ``signed`` (bool): True iff the layer is signed (Borzoi RNA, Sei,
      LentiMPRA).  Signed tracks use symmetric ``[-DISPLAY_MAX, +DISPLAY_MAX]``;
      unsigned use ``[0, DISPLAY_MAX]``.
    - ``ymin`` / ``ymax`` (float): suggested y-axis limits.  Renderers
      can use these to set IGV ``min``/``max``, matplotlib ``set_ylim``,
      or CoolBox ``MinValue``/``MaxValue``.
    - ``floor_pctile`` / ``peak_pctile`` / ``display_max`` (float): the
      thresholds used (informational; same for every rendering path).

    All semantics:
      - 1.0 (unsigned) or ±1.0 (signed) = genome-wide p99 of |signal|
      - DISPLAY_MAX = 3.0 = 3× p99 above the floor (cap)
      - 0.0 (unsigned) = below the layer floor (genome-wide noise)

    Pass ``normalizer=None`` to opt out (returns values unchanged with
    ``rescaled=False``, ``ymin/ymax`` set to data min/max for autoscale).
    """
    import numpy as np

    if normalizer is None or oracle_name is None or assay_id is None:
        v = np.asarray(values)
        return values, {
            "rescaled": False, "signed": False,
            "ymin": float(v.min()) if v.size else 0.0,
            "ymax": float(v.max()) if v.size else 1.0,
            "floor_pctile": None, "peak_pctile": None,
            "display_max": _DISPLAY_MAX,
        }

    from .normalization import PerTrackNormalizer
    if not isinstance(normalizer, PerTrackNormalizer):
        v = np.asarray(values)
        return values, {
            "rescaled": False, "signed": False,
            "ymin": float(v.min()) if v.size else 0.0,
            "ymax": float(v.max()) if v.size else 1.0,
            "floor_pctile": None, "peak_pctile": None,
            "display_max": _DISPLAY_MAX,
        }

    signed = normalizer.is_signed(oracle_name, assay_id)
    if signed:
        out = normalizer.signed_floor_rescale_batch(
            oracle_name, assay_id, values,
            peak_pctile=_PEAK_PCTILE, max_value=_DISPLAY_MAX,
        )
        if out is None:
            v = np.asarray(values)
            return values, {
                "rescaled": False, "signed": True,
                "ymin": float(v.min()) if v.size else -1.0,
                "ymax": float(v.max()) if v.size else 1.0,
                "floor_pctile": None, "peak_pctile": _PEAK_PCTILE,
                "display_max": _DISPLAY_MAX,
            }
        return out, {
            "rescaled": True, "signed": True,
            "ymin": -_DISPLAY_MAX, "ymax": _DISPLAY_MAX,
            "floor_pctile": None, "peak_pctile": _PEAK_PCTILE,
            "display_max": _DISPLAY_MAX,
        }

    floor_p = _LAYER_FLOOR_PCTILE.get(layer, _DEFAULT_FLOOR_PCTILE)
    out = normalizer.perbin_floor_rescale_batch(
        oracle_name, assay_id, values,
        floor_pctile=floor_p, peak_pctile=_PEAK_PCTILE, max_value=_DISPLAY_MAX,
    )
    if out is None:
        v = np.asarray(values)
        return values, {
            "rescaled": False, "signed": False,
            "ymin": float(v.min()) if v.size else 0.0,
            "ymax": float(v.max()) if v.size else 1.0,
            "floor_pctile": floor_p, "peak_pctile": _PEAK_PCTILE,
            "display_max": _DISPLAY_MAX,
        }
    return out, {
        "rescaled": True, "signed": False,
        "ymin": 0.0, "ymax": _DISPLAY_MAX,
        "floor_pctile": floor_p, "peak_pctile": _PEAK_PCTILE,
        "display_max": _DISPLAY_MAX,
    }


def apply_floor_rescale(
    normalizer,
    oracle_name: str | None,
    assay_id: str,
    layer: str,
    ref_vals,
    alt_vals,
):
    """Floor-subtract + rescale a ref/alt value pair using the normalizer.

    Returns ``(rescaled, ref_out, alt_out, signed)``.

    - ``rescaled=True, signed=False``: unsigned floor-rescale, values map
      to ``[0, _DISPLAY_MAX]`` with layer-aware thresholds (p95/p99 for
      sharp signals, p90/p99 for broad domains).  1.0 = genome-wide p99
      peak.  IGV scale_cfg should be ``{min: 0, max: _DISPLAY_MAX}``.
    - ``rescaled=True, signed=True``: signed symmetric rescale, values
      map to ``[-_DISPLAY_MAX, +_DISPLAY_MAX]`` using ``p99(|cdf|)`` as
      the unit.  ±1.0 = genome-wide top-1% absolute effect.  IGV
      scale_cfg should be ``{min: -_DISPLAY_MAX, max: +_DISPLAY_MAX}``.
    - ``rescaled=False``: no normalizer / no CDF / lookup miss.  Caller
      should fall back to raw autoscale.

    Used by every IGV-rendering path so panels share the same semantics.
    """

    # Delegate to the unified single-track rescaler (rescale_for_display)
    # so IGV, matplotlib, CoolBox and notebook callers all share the same
    # semantics — only the wrapper differs (this one returns a 4-tuple
    # for the ref/alt pair instead of (values, cfg)).
    ref_out, cfg_ref = rescale_for_display(
        ref_vals, layer, normalizer=normalizer,
        oracle_name=oracle_name, assay_id=assay_id,
    )
    alt_out, cfg_alt = rescale_for_display(
        alt_vals, layer, normalizer=normalizer,
        oracle_name=oracle_name, assay_id=assay_id,
    )
    # Both ref/alt should have identical scale_cfg (same track, same CDF).
    # If either failed to rescale, fall back to passthrough.
    if not (cfg_ref["rescaled"] and cfg_alt["rescaled"]):
        return False, ref_vals, alt_vals, cfg_ref["signed"]
    return True, ref_out, alt_out, cfg_ref["signed"]

def _calculate_track_bin_size(
    resolution: int,
    window_bp: int,
    source_oracle: str,
) -> tuple[int, str]:
    """Calculate appropriate bin size and aggregation method.
    
    Returns:
        (bin_size, aggregation_method) where aggregation is "mean" or "max"
    """

    # For chrombpnet or legnet models, apply max pooling
    # For any other oracle, apply mean pooling.
    # ChromBPNet's 1-bp output produces narrow tall peaks; mean-pooling
    # over 20 bp dilutes a single sharp peak with 19 near-zero neighbors,
    # then the p95 floor-rescale clips the diluted value below zero —
    # the panel ends up 97 % empty.  Max-pooling preserves the peak top
    # within each 20-bp bin so the panel actually shows the peak shape.
    # (PR #79's description says "max pooling preserves peak signals for
    # ChromBPNet"; the original code path returned "mean" — taking the
    # description as ground truth.)
    if source_oracle == "chrombpnet":
        bin_size = 20
        return bin_size, "max"
    elif source_oracle == "legnet":
        return resolution, "max"
    
    # Fallback: return 3_000 features per bin
    num_features = 3_000
    bin_size = window_bp // num_features
    return bin_size, "mean"

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

        t_res = ref_track.resolution
        actual_bp_in_array = len(ref_track.values) * t_res
        t_start = variant_pos - (actual_bp_in_array // 2)

        ref_vals = ref_track.values
        alt_vals = alt_track.values

        # Apply layer-aware floor-subtract + rescale when available
        floor_ok = False
        signed_track = False
        if use_floor:
            floor_ok, ref_vals, alt_vals, signed_track = apply_floor_rescale(
                normalizer, oracle_name, assay_id, layer, ref_vals, alt_vals,
            )

        track_bin_size, agg_method = _calculate_track_bin_size(
            t_res, window_bp, first.source_model,
        )

        # Signed tracks have negative values that ``skip_zeros`` would
        # incorrectly count as background — disable the threshold drop
        # so the repressive half stays in the wig features.
        ref_features = _downsample_to_features(
            ref_vals, variant_chrom, t_start, t_res, track_bin_size,
            skip_zeros=not (floor_ok or signed_track),
            aggregation_method=agg_method
        )
        alt_features = _downsample_to_features(
            alt_vals, variant_chrom, t_start, t_res, track_bin_size,
            skip_zeros=not (floor_ok or signed_track),
            aggregation_method=agg_method
        )

        group_id = assay_id.replace(":", "_").replace(" ", "_")
        if floor_ok and signed_track:
            # Symmetric signed scale: ±1.0 = genome-wide top-1% |effect|.
            scale_cfg = {"min": -_DISPLAY_MAX, "max": _DISPLAY_MAX, "autoscale": False}
            name_suffix = ""
        elif floor_ok:
            scale_cfg = {"min": 0, "max": _DISPLAY_MAX, "autoscale": False}
            name_suffix = ""
        else:
            scale_cfg = {"autoscale": True, "autoscaleGroup": group_id}
            name_suffix = ""

        # Build a human-readable display name from track metadata.
        # Use _track_description from variant_report for enriched CHIP names
        # (e.g. "CHIP:CEBPA:HepG2" instead of generic "CHIP:HepG2").
        from chorus.analysis.variant_report import _track_description
        display_name = _track_description(ref_track) or assay_id
        if display_name == assay_id:
            meta = getattr(ref_track, "metadata", None)
            if meta and isinstance(meta, dict) and meta.get("description"):
                display_name = meta["description"]
            elif hasattr(ref_track, "assay_type") and hasattr(ref_track, "cell_type"):
                display_name = f"{ref_track.assay_type}:{ref_track.cell_type}"

        # Merged overlay: ref (grey) + alt (coloured) on same panel
        source_model = first.source_model
        tracks.append({
            "name": f"{display_name}{name_suffix}",
            "type": "merged",
            "height": 80,
            "tracks": [
                {
                    "type": "wig",
                    "name": f"{display_name} ref",
                    "color": f"rgb({_REF_COLOR})",
                    "windowFunction": "max" if source_model in _HIGH_RES_ORACLES else "mean",
                    **scale_cfg,
                    "features": ref_features,
                },
                {
                    "type": "wig",
                    "name": f"{display_name} alt",
                    "color": f"rgb({rgb})",
                    "windowFunction": "max" if source_model in _HIGH_RES_ORACLES else "mean",
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
    aggregation_method: str = "mean"
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

        if aggregation_method == "mean":
            v = float(np.mean(chunk))
        else:
            v = float(np.max(chunk))

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
