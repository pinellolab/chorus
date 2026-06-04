import json
import torch

from chorus.oracles.epinformerseq_source.model_usage import (
    load_main_model,
    load_bias_model,
    predict_activity,
)

with open("__ARGS_FILE_NAME__") as inp:
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

main = load_main_model(args['main_weights'], device=device)
bias = load_bias_model(args['bias_weights'], device=device)

preds, _ = predict_activity(
    main, bias,
    seq=args['seq'],
    cell_type=args['cell_type'],
    assay=args.get('assay', 'Enhancer_DNase'),
    average_reverse=args.get('reverse_aug', False),
    device=device,
    in_window=args.get('in_window', 2114),
)

result = {
    'preds': preds.tolist(),
}
