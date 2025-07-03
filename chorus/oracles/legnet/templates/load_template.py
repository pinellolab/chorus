import json 
import torch 

from chorus.oracles.legnet.model_usage import load_model
with open("__ARGS_FILE_NAME__") as inp:  # to be formatted by calling script 
    args = json.load(inp)

device = torch.device(args['device'])

model = load_model(config_path=args['config_path'], 
                   weights_path=args['model_weights'])
model.eval()
model.to(device)

result = {
    'loaded': True,
    'model_class': str(type(model)),
    'description': 'LegNet model loaded successfully',
    'device': args['device'],
    'assays': ['MPRA'],
    'celltypes': [args['cell_line']],
}   