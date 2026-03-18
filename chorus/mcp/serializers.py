"""Serialize OraclePrediction objects into JSON-safe dicts for MCP responses."""

import numpy as np
from pathlib import Path
from typing import Optional


def _downsample(values: np.ndarray, max_bins: int = 500) -> list[float]:
    """Downsample an array to at most *max_bins* evenly-spaced points."""
    n = len(values)
    if n <= max_bins:
        return [float(v) for v in values]
    indices = np.linspace(0, n - 1, max_bins, dtype=int)
    return [float(values[i]) for i in indices]


def _track_summary(track) -> dict:
    """Compute summary statistics for a single OraclePredictionTrack."""
    vals = track.values
    return {
        "assay_id": track.assay_id,
        "assay_type": track.assay_type,
        "cell_type": track.cell_type,
        "chrom": track.chrom,
        "start": int(track.start),
        "end": int(track.end),
        "resolution": int(track.resolution),
        "num_bins": int(len(vals)),
        "mean": float(np.mean(vals)),
        "max": float(np.max(vals)),
        "min": float(np.min(vals)),
        "std": float(np.std(vals)),
    }


def serialize_prediction(
    prediction,
    output_dir: Optional[str] = None,
    prefix: str = "",
    inline_threshold: int = 500,
) -> dict:
    """Convert an OraclePrediction into a JSON-safe dict.

    - Always includes per-track summary stats.
    - If a track has ≤ *inline_threshold* bins, values are included inline.
    - Otherwise values are downsampled for a preview and the full data is
      saved as a bedgraph file (path returned in the response).
    """
    tracks_out: list[dict] = []
    saved_files: dict[str, str] = {}

    for assay_id, track in prediction.items():
        info = _track_summary(track)
        n_bins = len(track.values)

        if n_bins <= inline_threshold:
            info["values"] = [float(v) for v in track.values]
        else:
            info["values_preview"] = _downsample(track.values, inline_threshold)
            info["values_preview_note"] = (
                f"Downsampled from {n_bins} to {inline_threshold} bins. "
                "Full data saved as bedgraph."
            )
            # Save full bedgraph
            if output_dir is not None:
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                files = prediction.save_predictions_as_bedgraph(
                    str(out), prefix=prefix
                )
                saved_files.update(files)
                if assay_id in files:
                    info["bedgraph_path"] = files[assay_id]

        tracks_out.append(info)

    result: dict = {"tracks": tracks_out}
    if saved_files:
        result["saved_files"] = saved_files
    return result


def serialize_variant_effect(variant_result: dict, output_dir: Optional[str] = None) -> dict:
    """Serialize the dict returned by OracleBase.predict_variant_effect()."""
    out: dict = {"variant_info": variant_result["variant_info"], "tracks": {}}

    # Per-allele predictions
    for allele_name, pred in variant_result["predictions"].items():
        out["tracks"][allele_name] = {}
        for assay_id, track in pred.items():
            out["tracks"][allele_name][assay_id] = _track_summary(track)

    # Effect sizes
    effect_summaries: dict = {}
    for allele_name, effects in variant_result["effect_sizes"].items():
        effect_summaries[allele_name] = {}
        for assay_id, diff_arr in effects.items():
            effect_summaries[allele_name][assay_id] = {
                "mean_effect": float(np.mean(diff_arr)),
                "max_effect": float(np.max(diff_arr)),
                "min_effect": float(np.min(diff_arr)),
                "abs_max_effect": float(np.max(np.abs(diff_arr))),
                "std_effect": float(np.std(diff_arr)),
            }
    out["effect_sizes"] = effect_summaries

    # Optionally save bedgraphs
    if output_dir is not None:
        saved: dict = {}
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for allele_name, pred in variant_result["predictions"].items():
            files = pred.save_predictions_as_bedgraph(
                str(out_path), prefix=allele_name
            )
            saved[allele_name] = files
        out["saved_files"] = saved

    return out


def serialize_replacement_or_insertion(result: dict, output_dir: Optional[str] = None, prefix: str = "") -> dict:
    """Serialize results from predict_region_replacement / predict_region_insertion_at."""
    out: dict = {}

    raw = result.get("raw_predictions")
    if raw is not None:
        out["raw_predictions"] = serialize_prediction(raw, output_dir=output_dir, prefix=prefix + "raw_")

    norm = result.get("normalized_scores")
    if norm is not None:
        out["normalized_scores"] = serialize_prediction(norm, output_dir=output_dir, prefix=prefix + "norm_")

    return out
