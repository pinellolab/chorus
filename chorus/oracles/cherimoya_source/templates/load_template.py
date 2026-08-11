"""Load a CATv1 checkpoint inside the chorus-cherimoya environment.

Run by ``CherimoyaOracle._load_in_environment`` via
``run_code_in_environment``.  ``__ARGS_FILE_NAME__`` is substituted with a
temp JSON path by the caller.

Like the other oracles' load templates this cannot hand a live model back
across the subprocess boundary, so it verifies the checkpoint loads and
reports geometry; the predict template loads it again.
"""

import json
import os

with open("__ARGS_FILE_NAME__") as inp:  # to be formatted by calling script
    args = json.load(inp)

device = args["device"]
if device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif device and device.startswith("cuda:"):
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]

import torch

# Populate/reuse Triton's on-disk autotune cache; see the note in
# predict_template.py and cherimoya_source/_triton_autotune.py.  Must
# precede `import cherimoya`.
try:
    from triton import knobs
    knobs.autotuning.cache = True
except (ImportError, AttributeError):
    pass

from cherimoya import Cherimoya

if device in (None, "", "auto"):
    resolved = "cuda" if torch.cuda.is_available() else "cpu"
elif device == "gpu":
    resolved = "cuda"
else:
    resolved = device

# compile=False is mandatory here, not a preference.  Cherimoya.load
# defaults to compile=True with compile_mode='max-autotune', and this
# template runs in a *fresh subprocess on every predict call* -- so a
# compiling load would pay a full max-autotune warmup per prediction,
# which looks like a hang rather than a slow call.
model = Cherimoya.load(args["model_weights"], device=resolved, compile=False)
model = model.eval()

n_params = sum(p.numel() for p in model.parameters())

# Everything here must be a *builtin* type. The result is pickled in this
# environment and unpickled in the caller's, which has no torch -- so any
# torch-defined object fails to load there. `torch.__version__` is the
# subtle one: it is a `torch.torch_version.TorchVersion` instance (a str
# subclass), so returning it raises `ModuleNotFoundError: No module named
# 'torch'` from `pickle.load` in the parent, which reads exactly like a
# broken environment rather than a serialization bug. Hence str().
result = {
    "loaded": True,
    "model_class": str(type(model)),
    "device": str(resolved),
    "cuda_available": bool(torch.cuda.is_available()),
    "n_parameters": int(n_params),
    "trimming": int(model.trimming),
    "n_control_tracks": int(model.n_control_tracks),
    "signal_groups": [int(g) for g in model.signal_groups],
    "torch_version": str(torch.__version__),
    "description": "Cherimoya CATv1 model loaded successfully",
}
