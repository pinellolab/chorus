
"""Direct prediction in current environment."""

from chorus.oracles.borzoi_source.helpers import perform_prediction  
from chorus.oracles.borzoi_source.borzoi_metadata import get_metadata

import torch
import json
import os 

with open("__ARGS_FILE_NAME__") as inp:  # to be formatted by calling script 
    args = json.load(inp)

from borzoi_pytorch import Borzoi
flashzoi = Borzoi.from_pretrained(f'johahi/borzoi-replicate-{args["fold"]}')

device = args['device']
if device is None or device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

device = torch.device(device)

flashzoi.to(device)

pred = perform_prediction(flashzoi, args['sequence'], args['length'], device)

meta = get_metadata()
track_indices = meta.id2index(args['assay_ids'])
if any(map(lambda x: x is None, track_indices)):
    missing = [a for a, i in zip(args['assay_ids'], track_indices) if i is None]
    # Use ValueError (not bare Exception) so the env-runner surfaces a
    # recognisable error type. v26 P1 #12.
    raise ValueError(
        f"Borzoi track identifier(s) not found in metadata: {missing}. "
        f"Use oracle.get_track_info() to list valid identifiers, or "
        f"oracle.get_track_info(pattern) to search (e.g. 'DNASE:K562')."
    )
# Extract predictions for selected tracks
selected_predictions = pred[:, track_indices]
result = selected_predictions.tolist()