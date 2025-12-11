import json 
import torch 
from chorus.oracles.sei_source.sei import Sei, SeiProjector, SeiNormalizer
from chorus.oracles.sei_source.annotations import SeiClassesList, SeiTargetList
from chorus.oracles.sei_source.utils import gather_with_nones
from chorus.oracles.sei_source.exceptions import SeiError

with open("__ARGS_FILE_NAME__") as inp:  # to be formatted by calling script 
    args = json.load(inp)

device = torch.device(args['device'])

model = Sei(sequence_length=args['sequence_length'], n_genomic_features=args['n_genomic_features'])
model_weights = torch.load(args['model_weights'], map_location='cpu', weights_only=True)
model_weights = {key.replace("module.model.", ""): value for key, value in model_weights.items()}
model.load_state_dict(model_weights)
model.eval()
model.to(device)

projector = SeiProjector(weights=args['projector_weights'], n_classes=args['n_classes'])

targets = SeiTargetList.load(args['targets'])
classes = SeiClassesList.load(args['classes'])

seq = args['seq']
targets_inds = args['targets_inds']
classes_inds = args['classes_inds']

if targets_inds is None and classes_inds is None:
    raise SeiError("Assays or classes ids must be provided")

predictions, _ = model.seq_sliding_predict(seq, 
                                        reverse_aug=args['reverse_aug'],
                                        window_size=args['sequence_length'],
                                        step=args['bin_size'],
                                        batch_size=args['batch_size'])

class_preds = projector(predictions)

selected_preds = predictions[:, targets_inds] # single-seq prediction
selected_preds = selected_preds.tolist()

selected_classes = class_preds[:, classes_inds] # single-seq prediction
selected_classes = selected_classes.tolist()
        
result = {
    'selected_preds': selected_preds,
    'selected_classes': selected_classes,
    'seq_length': len(seq),
}