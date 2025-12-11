import json 
import torch 
from chorus.oracles.sei_source.sei import Sei, SeiProjector, SeiNormalizer
from chorus.oracles.sei_source.annotations import SeiClassesList, SeiTargetList

with open("__ARGS_FILE_NAME__") as inp:  # to be formatted by calling script 
    args = json.load(inp)

device = torch.device(args['device'])

model = Sei(sequence_length=args['sequence_length'], n_genomic_features=args['n_genomic_features'])
model_weights = torch.load(args['model_weights'], map_location='cpu', weights_only=True)
model_weights = {key.replace("module.model.", ""): value for key, value in model_weights.items()}
model.load_state_dict(model_weights)
model.to(device)

projector = SeiProjector(weights=args['projector_weights'], n_classes=args['n_classes'])

normalizer = SeiNormalizer(histone_inds=args['histone_inds'])

targets = SeiTargetList.load(args['targets'])
classes = SeiClassesList.load(args['classes'])

result = {
    'loaded': True,
    'model_class': str(type(model)),
    'description': 'Sei model loaded successfully',
    'device': args['device'],
    'classes': classes.list_class_types(),
    'groups': classes.list_group_types(),
    'assays': targets.list_assay_types(),
    'celltypes': targets.list_cell_types(),
}   
