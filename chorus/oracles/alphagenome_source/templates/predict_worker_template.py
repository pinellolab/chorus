"""Persistent AlphaGenome prediction worker.

Executed inside the ``chorus-alphagenome`` conda environment by
:class:`chorus.oracles.alphagenome.AlphaGenomePredictWorker`. Unlike
``predict_template.py`` (which loads the model + fetches the ~330 MB GCS
reference tables on EVERY forward pass), this worker loads the model ONCE
and then serves an arbitrary number of prediction requests over stdin/stdout.

Protocol (stdout carries ONLY these control lines; everything else — model
load logs, library chatter — goes to stderr):

  * On startup, after the model has loaded, print exactly ``READY``.
  * Then loop, reading one path per line from stdin:
      - ``__STOP__``            -> exit 0.
      - ``<request_args_path>`` -> a JSON file ``{"sequence", "assay_ids"}``.
        On success print ``OK <result_pickle_path>``; the result pickle holds
        the SAME ``{"values": [...], "resolutions": [...]}`` dict that
        ``predict_template.py`` produces. On a per-request error print
        ``ERR <single-line-repr>`` and KEEP serving.

The placeholder ``__INIT_ARGS_FILE__`` is replaced at launch with the path to
a JSON file holding ``{"device", "fold", "length"}`` (same init args the
per-call template reads from its args file).
"""

import json
import os
import pickle
import platform as _platform
import sys
import tempfile
import traceback

import numpy as np


def _log(*a):
    """All diagnostic output goes to stderr — stdout is control-only."""
    print(*a, file=sys.stderr, flush=True)


with open("__INIT_ARGS_FILE__") as _inp:
    _init = json.load(_inp)

# Pre-import device routing: JAX Metal is too experimental for AlphaGenome
# (missing default_memory_space etc.), so force CPU on macOS unless the user
# explicitly requests Metal.
device_str = _init.get("device")
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

# Load model ONCE.
fold = _init.get("fold", "all_folds")
_log("worker: create_from_huggingface(%r) on %r" % (fold, jax_device))
model = create_from_huggingface(fold, device=jax_device)
_log("worker: model loaded")

metadata = get_metadata()


def _predict_one(request):
    """Run one forward pass + per-assay extraction.

    Byte-identical to predict_template.py: same requested_outputs derivation,
    same _ot_cache, same np.ascontiguousarray(values[:, local_idx], float32).
    """
    sequence = request["sequence"]
    assay_ids = request["assay_ids"]

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

    # Extract per-assay predictions (cache each OutputType array once;
    # keep float32 ndarrays for cheap pickling — identical to template).
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
        track_values = np.ascontiguousarray(values[:, local_idx], dtype=np.float32)
        collected.append(track_values)
        resolutions.append(info["resolution"])

    return {"values": collected, "resolutions": resolutions}


# Signal readiness AFTER the (slow) model load. The parent blocks on this.
print("READY", flush=True)

# Serve requests until told to stop (or stdin closes).
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    if line == "__STOP__":
        break
    try:
        with open(line) as _rf:
            request = json.load(_rf)
        result = _predict_one(request)
        fd, result_path = tempfile.mkstemp(suffix=".pkl", prefix="ag_worker_res_")
        with os.fdopen(fd, "wb") as _wf:
            pickle.dump(result, _wf)
        print("OK " + result_path, flush=True)
    except Exception as exc:  # one bad request must not kill the worker
        _log("worker: request failed\n" + traceback.format_exc())
        rep = repr(exc).replace("\n", " ").replace("\r", " ")
        print("ERR " + rep, flush=True)

sys.exit(0)
