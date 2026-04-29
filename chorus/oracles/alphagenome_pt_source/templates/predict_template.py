"""Run AlphaGenome PyTorch port prediction inside the conda environment.

Placeholder ``__ARGS_FILE_NAME__`` is replaced at runtime with the path to a
JSON file containing prediction arguments.

The PyTorch port returns a nested dict keyed by lowercase output-type name
(e.g. ``'atac'``) and then by resolution (e.g. ``outputs['atac'][1]``). We
slice each requested assay's track using the ``local_index`` field from the
shared chorus AlphaGenome metadata cache.
"""

import json
import os
import numpy as np

# Match load_template — pip torch + conda-forge compilers libomp clash.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

with open("__ARGS_FILE_NAME__") as inp:
    args = json.load(inp)

import torch
import huggingface_hub

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

# Install the MPS-compat rope patch before model construction.
from chorus.oracles.alphagenome_pt_source import _mps_compat  # noqa: F401

from alphagenome_pytorch import AlphaGenome
from chorus.oracles.alphagenome_source.alphagenome_metadata import (
    get_metadata,
    SKIPPED_OUTPUT_TYPES,
)

model = AlphaGenome.from_pretrained(weights_path, device=device)
model.eval()

# One-hot encode (A,C,G,T) → (4,) channels. PyTorch port expects
# shape (batch, length, 4) per the demo notebook conventions.
sequence = args["sequence"]
_BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
seq_arr = np.zeros((len(sequence), 4), dtype=np.float32)
for i, b in enumerate(sequence):
    j = _BASE_TO_IDX.get(b.upper(), -1)
    if j >= 0:
        seq_arr[i, j] = 1.0
dna_onehot = torch.from_numpy(seq_arr).unsqueeze(0).to(device)  # (1, L, 4)

metadata = get_metadata()
assay_ids = args["assay_ids"]

# Collect needed output types so we can request only what we need.
needed_output_types = set()
for aid in assay_ids:
    idx = metadata.get_track_by_identifier(aid)
    if idx is None:
        raise ValueError(f"Assay ID not found in metadata: {aid}")
    info = metadata.get_track_info(idx)
    if info is None:
        raise ValueError(f"No track info for index {idx} (assay {aid})")
    needed_output_types.add(info["output_type"])

# Map chorus OutputType names → PyTorch port lowercase head keys.
_OUTPUT_TYPE_TO_PT_KEY = {
    "ATAC": "atac",
    "DNASE": "dnase",
    "CAGE": "cage",
    "RNA_SEQ": "rna_seq",
    "CHIP_HISTONE": "chip_histone",
    "CHIP_TF": "chip_tf",
    "PROCAP": "procap",
    "SPLICE_SITES": "splice_sites",
    "SPLICE_SITE_USAGE": "splice_site_usage",
}

heads = tuple(
    _OUTPUT_TYPE_TO_PT_KEY[ot]
    for ot in needed_output_types
    if ot in _OUTPUT_TYPE_TO_PT_KEY and ot not in SKIPPED_OUTPUT_TYPES
)

with torch.no_grad():
    # Use forward() so we can pass heads= to skip computation for unused
    # output types. predict() is a thin no_grad wrapper without head
    # filtering.
    output = model(
        dna_onehot,
        organism_index=torch.tensor([0], dtype=torch.long, device=device),  # 0 = human
        heads=heads if heads else None,
    )

collected = []
resolutions = []
for aid in assay_ids:
    idx = metadata.get_track_by_identifier(aid)
    info = metadata.get_track_info(idx)
    ot_name = info["output_type"]
    local_idx = info["local_index"]
    res = info.get("resolution", 1)
    pt_key = _OUTPUT_TYPE_TO_PT_KEY.get(ot_name)
    if pt_key is None or pt_key not in output:
        raise ValueError(
            f"Output type {ot_name} not produced by PyTorch port (key={pt_key})"
        )
    head_out = output[pt_key]
    # head_out is either a tensor (single resolution) or a dict[res_int -> tensor]
    if isinstance(head_out, dict):
        if res not in head_out:
            # Fall back to whatever resolution is available
            res = next(iter(head_out.keys()))
        tensor = head_out[res]
    else:
        tensor = head_out
    arr = tensor.detach().cpu().numpy()
    # Expected shape: (batch=1, spatial, num_tracks)
    if arr.ndim == 3:
        track_values = arr[0, :, local_idx]
    elif arr.ndim == 2:
        track_values = arr[:, local_idx]
    else:
        raise ValueError(
            f"Unexpected output array shape {arr.shape} for {ot_name}"
        )
    collected.append(track_values.astype(np.float32).tolist())
    resolutions.append(int(res))

result = {
    "values": collected,
    "resolutions": resolutions,
}
