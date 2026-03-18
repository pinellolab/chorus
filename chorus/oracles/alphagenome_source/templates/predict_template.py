"""Run AlphaGenome prediction in the conda environment.

This script is executed inside the chorus-alphagenome conda environment.
The placeholder ``__ARGS_FILE_NAME__`` is replaced at runtime with the
path to a JSON file containing prediction arguments.
"""

import json
import os
import platform as _platform
import numpy as np

with open("__ARGS_FILE_NAME__") as inp:
    args = json.load(inp)

# Pre-import device routing: JAX Metal is too experimental for AlphaGenome
# (missing default_memory_space etc.), so force CPU on macOS unless the user
# explicitly requests Metal.
device_str = args.get("device")
if _platform.system() == "Darwin" and (device_str is None or device_str.startswith("cpu")):
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax

if device_str is not None and device_str.startswith("cpu"):
    jax_device = jax.devices("cpu")[0]
elif device_str is not None and device_str.startswith("gpu"):
    jax_device = jax.devices("gpu")[0]
elif device_str is not None and device_str.startswith("metal"):
    jax_device = jax.devices("METAL")[0]
else:
    # Auto-detect: prefer CUDA GPU > CPU
    available_platforms = {d.platform for d in jax.devices()}
    if "gpu" in available_platforms:
        jax_device = jax.devices("gpu")[0]
    else:
        jax_device = jax.devices("cpu")[0]

from alphagenome.models.dna_output import OutputType
from alphagenome_research.model.dna_model import create_from_huggingface
from chorus.oracles.alphagenome_source.alphagenome_metadata import (
    get_metadata,
    SKIPPED_OUTPUT_TYPES,
)

# Load model
fold = args.get("fold", "all_folds")
model = create_from_huggingface(fold, device=jax_device)

# Prepare sequence
sequence = args["sequence"]

# Determine which OutputTypes we need for the requested assay_ids
metadata = get_metadata()
assay_ids = args["assay_ids"]

# Find which output types are needed
needed_output_types = set()
for aid in assay_ids:
    idx = metadata.get_track_by_identifier(aid)
    if idx is None:
        raise ValueError(f"Assay ID not found in metadata: {aid}")
    info = metadata.get_track_info(idx)
    if info is None:
        raise ValueError(f"No track info for index {idx} (assay {aid})")
    needed_output_types.add(info["output_type"])

# Map output type names to OutputType enum
requested_outputs = []
for ot in OutputType:
    if ot.name in needed_output_types and ot.name not in SKIPPED_OUTPUT_TYPES:
        requested_outputs.append(ot)

# Run prediction
output = model.predict_sequence(
    sequence,
    requested_outputs=requested_outputs,
    ontology_terms=None,
)

# Extract per-assay predictions
collected = []
resolutions = []
for aid in assay_ids:
    idx = metadata.get_track_by_identifier(aid)
    if idx is None:
        raise ValueError(f"Assay ID not found in metadata: {aid}")
    info = metadata.get_track_info(idx)
    if info is None:
        raise ValueError(f"No track info for index {idx} (assay {aid})")
    ot_name = info["output_type"]
    local_idx = info["local_index"]

    ot_enum = OutputType[ot_name]
    track_data = output.get(ot_enum)

    if track_data is None:
        raise ValueError(
            f"No prediction data for output type {ot_name} (assay {aid})"
        )

    values = np.asarray(track_data.values)
    # values shape: (positional_bins, num_tracks)
    if local_idx >= values.shape[1]:
        raise ValueError(
            f"local_idx {local_idx} out of bounds for output type {ot_name} "
            f"with {values.shape[1]} tracks (assay {aid})"
        )
    track_values = values[:, local_idx]
    collected.append(track_values.tolist())
    resolutions.append(info["resolution"])

result = {
    "values": collected,
    "resolutions": resolutions,
}
