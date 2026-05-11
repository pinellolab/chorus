import json
import torch

from chorus.oracles.epinformerseq_source.model_usage import load_model

with open("__ARGS_FILE_NAME__") as inp:  # to be formatted by calling script
    args = json.load(inp)

_dev = args['device']
if _dev is None or _dev == 'auto':
    if torch.cuda.is_available():
        _dev = 'cuda'
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        _dev = 'mps'
    else:
        _dev = 'cpu'
device = torch.device(_dev)

model = load_model(weights_path=args['model_weights'], device=device)
model.eval()

result = {
    'loaded': True,
    'model_class': str(type(model)),
    'description': 'EPInformer-seq enhancer_predictor_256bp loaded successfully',
    'device': _dev,
    'assays': [args['assay']],
    'celltypes': [args['cell_type']],
}
