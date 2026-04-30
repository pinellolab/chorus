import json
import os
import torch
from borzoi_pytorch import Borzoi

with open("__ARGS_FILE_NAME__") as inp:  # to be formatted by calling script
    args = json.load(inp)


def _load_borzoi(fold: int):
    """Prefer the chorus-controlled HF mirror at lucapinello/chorus-borzoi
    (per-fold subdirs); fall back to johahi/borzoi-replicate-{fold} on
    any failure. Both contain the same flashzoi PyTorch weights."""
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(
            repo_id="lucapinello/chorus-borzoi",
            repo_type="model",
            allow_patterns=[f"fold_{fold}/*"],
        )
        fold_path = os.path.join(local_dir, f"fold_{fold}")
        if not os.path.isdir(fold_path):
            raise FileNotFoundError(f"fold_{fold}/ missing in chorus-borzoi mirror")
        return Borzoi.from_pretrained(fold_path)
    except Exception as exc:
        print(f"chorus-borzoi mirror unavailable ({exc}); falling back to johahi")
        return Borzoi.from_pretrained(f'johahi/borzoi-replicate-{fold}')


flashzoi = _load_borzoi(args["fold"])
device = args['device']
if device is None or device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = 'mps'
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
