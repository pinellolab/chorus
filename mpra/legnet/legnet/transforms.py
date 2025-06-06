import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F




CODES = {
    "A": 0,
    "T": 3,
    "G": 1,
    "C": 2,
    'N': 4
}

INV_CODES = {value: key for key, value in CODES.items()}

COMPL = {
    'A': 'T',
    'T': 'A',
    'G': 'C',
    'C': 'G',
    'N': 'N'
}

def n2id(n):
    return CODES[n.upper()]

def id2n(i):
    return INV_CODES[i]


class Seq2Tensor(nn.Module):
    '''
    Encode sequences using one-hot encoding after preprocessing.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, seq: str):
        seq = [n2id(x) for x in seq]
        code = torch.from_numpy(np.array(seq))
        code = F.one_hot(code, num_classes=5) # 5th class is N

        mask = code[:, 4] == 1
        code = code[:, :4].float() 
        code[mask] = 0.25 # encode Ns with .25
        return code.transpose(0, 1)


class AddFlanks(nn.Module):
    '''
    Add flanks to nucleotide sequence
    '''

    def __init__(self, 
                 up: str, 
                 down: str):
        super().__init__()
        self.up = up 
        self.down = down

    def forward(self, seq: str):
        return self.up + seq + self.down

    def __repr__(self):
        return f"AddFlanks(up={self.up}, down={self.down})"


class PadNs(nn.Module):
    '''
    Pad sequence with Ns up to requested size
    '''

    def __init__(self, 
                 size: int = 200):
        super().__init__()
        self.size = size

    def forward(self, seq: str):
        add_sz = self.size - len(seq)
        lp, mod = divmod(add_sz, 2)
        lp, rp = lp + mod, lp
        return lp * 'N' + seq + rp * 'N'


class ReverseComplement(nn.Module):
    '''
    Get reverse complement of the sequence
    '''
    def __init__(self):
        super().__init__()

    def forward(self, seq: str):
        return ''.join(COMPL[c] for c in reversed(seq))