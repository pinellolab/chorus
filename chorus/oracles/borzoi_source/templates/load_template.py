import json 
import torch 
from borzoi_pytorch import Borzoi

with open("__ARGS_FILE_NAME__") as inp:  # to be formatted by calling script 
    args = json.load(inp)

flashzoi = Borzoi.from_pretrained(f'johahi/borzoi-replicate-{args["fold"]}')
device = args['device']
if device is None:
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

device = torch.device(device)

flashzoi.to(device)

# Get model info (we can't pickle the model itself)
result = {
    'loaded': True,
    'model_class': str(type(flashzoi)),
    'description': 'Borzoi model loaded successfully',
    'device': args['device']
}