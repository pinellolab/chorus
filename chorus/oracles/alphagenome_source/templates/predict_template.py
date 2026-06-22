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
if _platform.system() == "Darwin" and not (device_str is not None and device_str.startswith("metal")):
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

# AlphaGenome fetches its GCS reference tables (gencode GTF / splice-site /
# polyA feathers) fresh on every load via urllib — these are not cached, and
# the GCS endpoint intermittently fails mid-load, aborting the whole load:
#   - ssl.SSLEOFError (UNEXPECTED_EOF) at connect time, and
#   - http.client.IncompleteRead while reading the body of the large
#     (~330 MB) gencode GTF feather.
# Retry at both the urlopen (connect) and pd.read_feather (download+parse)
# levels; the latter is what covers the mid-stream IncompleteRead.
import http.client as _http
import ssl as _ssl
import time as _time
import urllib.error as _urlerr
import urllib.request as _urlreq
import pandas as _pd

_TRANSIENT = (_ssl.SSLError, _urlerr.URLError, OSError,
              _http.IncompleteRead, _http.HTTPException)


def _retry(_fn, *a, **k):
    last = None
    for _attempt in range(8):
        try:
            return _fn(*a, **k)
        except _TRANSIENT as exc:
            last = exc
            _time.sleep(2 * (_attempt + 1))
    raise last


_orig_urlopen = _urlreq.urlopen
_urlreq.urlopen = lambda *a, **k: _retry(_orig_urlopen, *a, **k)
_orig_read_feather = _pd.read_feather
_pd.read_feather = lambda *a, **k: _retry(_orig_read_feather, *a, **k)

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

# Extract per-assay predictions.
#
# Two efficiency fixes here (both byte-identical to the prior behaviour;
# the parent re-wraps each entry with np.array(..., dtype=np.float32)):
#   1. Convert each OutputType's full (bins × tracks) array to NumPy ONCE
#      and cache it. The old code called np.asarray(track_data.values) per
#      assay — i.e. re-materialising the same large JAX array thousands of
#      times (≈ O(assays × bins × tracks)), which dominated the in-env
#      forward pass while the GPU sat idle.
#   2. Keep each track's signal as a float32 NumPy array instead of
#      .tolist(). The result dict is returned to the parent via pickle,
#      and pickling ndarrays (raw buffers) is far cheaper than pickling
#      ~5000 Python float lists; the parent then builds the same float32
#      arrays it did before.
collected = []
resolutions = []
_ot_cache = {}
for aid in assay_ids:
    idx = metadata.get_track_by_identifier(aid)
    if idx is None:
        raise ValueError(f"Assay ID not found in metadata: {aid}")
    info = metadata.get_track_info(idx)
    if info is None:
        raise ValueError(f"No track info for index {idx} (assay {aid})")
    ot_name = info["output_type"]
    local_idx = info["local_index"]

    values = _ot_cache.get(ot_name)
    if values is None:
        ot_enum = OutputType[ot_name]
        track_data = output.get(ot_enum)
        if track_data is None:
            raise ValueError(
                f"No prediction data for output type {ot_name} (assay {aid})"
            )
        values = np.asarray(track_data.values)  # (positional_bins, num_tracks)
        _ot_cache[ot_name] = values

    if local_idx >= values.shape[1]:
        raise ValueError(
            f"local_idx {local_idx} out of bounds for output type {ot_name} "
            f"with {values.shape[1]} tracks (assay {aid})"
        )
    # np.ascontiguousarray so the pickled buffer is a compact 1-D float32
    # array rather than a strided view into the big 2-D output.
    track_values = np.ascontiguousarray(values[:, local_idx], dtype=np.float32)
    collected.append(track_values)
    resolutions.append(info["resolution"])

result = {
    "values": collected,
    "resolutions": resolutions,
}
