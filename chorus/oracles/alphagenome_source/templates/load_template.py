"""Load AlphaGenome model in the conda environment.

This script is executed inside the chorus-alphagenome conda environment.
The placeholder ``__ARGS_FILE_NAME__`` is replaced at runtime with the
path to a JSON file containing loading arguments.
"""

import json
import os
import platform as _platform

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

import os
import huggingface_hub

# Ensure HuggingFace token is available (required for gated AlphaGenome model)
try:
    huggingface_hub.whoami()
except huggingface_hub.errors.LocalTokenNotFoundError:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)
    else:
        raise RuntimeError(
            "AlphaGenome requires HuggingFace authentication. "
            "Set the HF_TOKEN environment variable or run 'huggingface-cli login'."
        )

from alphagenome_research.model.dna_model import create_from_huggingface

fold = args.get("fold", "all_folds")
model = create_from_huggingface(fold, device=jax_device)

# Build the track cache while we're inside the environment
from chorus.oracles.alphagenome_source.alphagenome_metadata import get_metadata
get_metadata()

result = {
    "loaded": True,
    "description": "AlphaGenome model loaded successfully",
    "device": str(jax_device),
    "fold": fold,
}
