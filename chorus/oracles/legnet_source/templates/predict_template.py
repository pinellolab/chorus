import json 
import torch 
from chorus.oracles.legnet_source.model_usage import load_model
from chorus.oracles.legnet_source.model_usage import predict_bigseq

with open("__ARGS_FILE_NAME__") as inp:  # to be formatted by calling script 
    args = json.load(inp)

device = torch.device(args['device'])

model = load_model(config_path=args['config_path'], 
                   weights_path=args['model_weights'])
model.eval()
model.to(device)


preds, _ = predict_bigseq(model, 
                                seq=args['seq'], 
                                reverse_aug=args['reverse_aug'],
                                window_size=args['sequence_length'],
                                step=args['bin_size'],
                                left_flank=args['left_flank'],
                                right_flank=args['right_flank'],
                                batch_size=args['batch_size'])
        
result = {
    'preds': preds.tolist(),
}