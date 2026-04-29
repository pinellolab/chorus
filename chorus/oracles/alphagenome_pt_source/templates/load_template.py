"""Load the AlphaGenome PyTorch port inside the chorus-alphagenome-pytorch env.

Placeholder ``__ARGS_FILE_NAME__`` is replaced at runtime with the path to a
JSON file containing loading arguments.

Notes
-----
The PyTorch port (``genomicsxai/alphagenome-pytorch``) exposes
``AlphaGenome.from_pretrained(path, device=...)`` which accepts a local
filesystem path only. We resolve the path via ``hf_hub_download`` from
``gtca/alphagenome_pytorch`` (the upstream-maintained HF mirror).
"""

import json
import os

# conda-forge `compilers` ships its own libomp.dylib that conflicts with
# the libomp inside the pip-installed torch wheel on macOS. Setting
# KMP_DUPLICATE_LIB_OK=TRUE before importing torch is the documented
# workaround. Harmless on Linux/CUDA.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

with open("__ARGS_FILE_NAME__") as inp:
    args = json.load(inp)

import torch
import huggingface_hub

# HF auth: the AlphaGenome model terms apply to the PyTorch port too. We
# inherit the existing HF_TOKEN flow so users who already accepted the JAX
# license at huggingface.co/google/alphagenome-all-folds don't need a second
# login. The PyTorch weight repo (gtca/alphagenome_pytorch) is currently
# public, but auth keeps the flow consistent with the JAX path.
try:
    huggingface_hub.whoami()
except huggingface_hub.errors.LocalTokenNotFoundError:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)

device_str = args.get("device")
if device_str == "mps" or (device_str is None and torch.backends.mps.is_available()):
    device = torch.device("mps")
elif device_str == "cuda" or (device_str is None and torch.cuda.is_available()):
    device = torch.device("cuda")
elif device_str and device_str.startswith("cuda:"):
    device = torch.device(device_str)
else:
    device = torch.device("cpu")

repo_id = args.get("hf_repo", "gtca/alphagenome_pytorch")
filename = args.get("weights_filename", "alphagenome.pt")
weights_path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)

from alphagenome_pytorch import AlphaGenome

model = AlphaGenome.from_pretrained(weights_path, device=device)
model.eval()

result = {
    "loaded": True,
    "description": "AlphaGenome PyTorch port loaded successfully",
    "device": str(device),
    "weights_path": weights_path,
    "repo_id": repo_id,
    "filename": filename,
}
