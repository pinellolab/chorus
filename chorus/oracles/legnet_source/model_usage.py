import torch
import numpy as np

from .config import TrainingConfig
from .dataset import SubSequenceDataset
from .legnet_globals import LEGNET_WINDOW, LEGNET_DEFAULT_STEP
from .agarwal_meta import LEFT_MPRA_FLANK, RIGHT_MPRA_FLANK
from torch.utils.data import DataLoader

def load_model(config_path: str, weights_path: str) -> torch.nn.Module:
    config = TrainingConfig.from_json(config_path)
    model = config.get_model()

    # LegNet was trained using pytorch-lightning framework. However, it is relatively easy to move code to raw PyTorch 
    weights = torch.load(weights_path, map_location='cpu', weights_only=False)['state_dict']
    weights = {key.replace('model.', ''): value for key, value in weights.items()}
    model.load_state_dict(weights);
    model.eval(); 
    return model

def get_device(m: torch.nn.Module) -> torch.device:
    return next(m.parameters()).device

def predict_bigseq(model, 
                   seq: str, 
                   step: int = LEGNET_DEFAULT_STEP,
                   window_size: int = LEGNET_WINDOW, 
                   reverse_aug: bool = False,
                   batch_size: int = 1,
                   left_flank: str = LEFT_MPRA_FLANK,
                   right_flank: str = RIGHT_MPRA_FLANK,):

    device = get_device(model)
    ds = SubSequenceDataset(sequence=seq, 
                        window_size=window_size,
                        reverse_aug=reverse_aug,
                        step=step, 
                        left_flank=left_flank,
                        right_flank=right_flank)
    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=False)
    offsets = []
    preds = []

    with torch.inference_mode():
        if reverse_aug:
            for X, rev_X, of in dl:
                pr = model(X.to(device)).cpu().numpy()
                pr_rev = model(rev_X.to(device)).cpu().numpy()
                pr_avg = (pr + pr_rev) / 2
                preds.append(pr_avg)
                offsets.append(of.numpy())
        else:
            for X, of in dl:
                pr = model(X.to(device)).cpu().numpy()
                preds.append(pr)
                offsets.append(of.numpy())

    preds = np.concatenate(preds)
    offsets = np.concatenate(offsets)
    return preds, offsets 