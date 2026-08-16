import json 
import torch 
from chorus.oracles.sei_source.sei import Sei, SeiProjector, SeiNormalizer
from chorus.oracles.sei_source.annotations import SeiClassesList, SeiTargetList
from chorus.oracles.sei_source.utils import gather_with_nones
from chorus.oracles.sei_source.exceptions import SeiError

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

model = Sei(sequence_length=args['sequence_length'], n_genomic_features=args['n_genomic_features'])
model_weights = torch.load(args['model_weights'], map_location='cpu', weights_only=True)
model_weights = {key.replace("module.model.", ""): value for key, value in model_weights.items()}
model.load_state_dict(model_weights)
model.eval()
model.to(device)

projector = SeiProjector(weights=args['projector_weights'], n_classes=args['n_classes'])

targets = SeiTargetList.load(args['targets'])
classes = SeiClassesList.load(args['classes'])

targets_inds = args['targets_inds']
classes_inds = args['classes_inds']

if targets_inds is None and classes_inds is None:
    raise SeiError("Assays or classes ids must be provided")


def _raw(sequence):
    preds, _ = model.seq_sliding_predict(sequence,
                                        reverse_aug=args['reverse_aug'],
                                        window_size=args['sequence_length'],
                                        step=args['bin_size'],
                                        batch_size=args['batch_size'])
    return preds


# Two shapes. `seqs` is the variant path: every allele is predicted here so the
# nucleosome-occupancy correction -- which is defined over the ref/alt PAIR, not over one
# sequence -- can be applied to the raw 21,907-profile vectors before projection. Doing it in
# the child avoids shipping 21,907 floats per allele back across the subprocess boundary.
# `seq` is the single-sequence path and is unchanged.
if args.get('seqs') is not None:
    seqs = args['seqs']
    raws = [_raw(s) for s in seqs]

    if args.get('histone_inds'):
        normalizer = SeiNormalizer(histone_inds=args['histone_inds'])
        raws = normalizer.equalize(raws)

    result = {
        'per_allele': [
            {
                'selected_preds': r[:, targets_inds].tolist(),
                'selected_classes': projector(r)[:, classes_inds].tolist(),
                'seq_length': len(s),
            }
            for r, s in zip(raws, seqs)
        ],
        'normalized': bool(args.get('histone_inds')),
    }
else:
    seq = args['seq']
    predictions = _raw(seq)
    class_preds = projector(predictions)

    result = {
        'selected_preds': predictions[:, targets_inds].tolist(),
        'selected_classes': class_preds[:, classes_inds].tolist(),
        'seq_length': len(seq),
    }