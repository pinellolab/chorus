
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
if device is None:
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

device = torch.device(device)

flashzoi.to(device)

pred = perform_prediction(flashzoi, args['sequence'], args['length'], device)

meta = get_metadata()
track_indices = meta.id2index(args['assay_ids'])
if any(map(lambda x: x is None, track_indices)):
    raise Exception(f"Some assay IDs not found in metadata: {args['assay_ids']}")
# Extract predictions for selected tracks
selected_predictions = pred[:, track_indices]
result = selected_predictions.tolist()