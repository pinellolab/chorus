import json
import torch

from chorus.oracles.epinformerseq_source.model_usage import load_model, predict_activity

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

preds, _ = predict_activity(
    model,
    seq=args['seq'],
    average_reverse=args.get('reverse_aug', False),
    device=device,
)

result = {
    'preds': preds.tolist(),
}
