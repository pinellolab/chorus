import numpy as np 
import torch
def dna_1hot(seq: str):
    """Convert DNA sequence to one-hot encoding."""
    seq = seq.upper()
    seq_len = len(seq)
    seq_1hot = np.zeros((seq_len, 4), dtype='bool')
    
    for i in range(seq_len):
        if seq[i] == 'A':
            seq_1hot[i,0] = True
        elif seq[i] == 'C':
            seq_1hot[i,1] = True
        elif seq[i] == 'G':
            seq_1hot[i,2] = True
        elif seq[i] == 'T':
            seq_1hot[i,3] = True
    
    return seq_1hot

def padseq(seq: str, window: int):
    tpad = window - len(seq) 
    lpad = tpad // 2
    rpad = tpad - lpad
    lN = 'N' * lpad
    rN = 'N' * rpad
    return f'{lN}{seq}{rN}'

def perform_prediction(model, seq: str, window: int, device):
    seq = padseq(seq=seq, window=window)
    ohe = dna_1hot(seq)
    ohe = torch.from_numpy(ohe)
    ohe_float = ohe.permute(1, 0).float()
    ohe_float = ohe_float.to(device)
    with torch.no_grad():
        pred = model(ohe_float.unsqueeze(0)).squeeze(0)
    pred = pred.permute(1, 0).cpu().numpy()

    return pred