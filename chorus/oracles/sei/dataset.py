import torch
import math 
import torch.nn as nn 

from torch.utils.data import Dataset 
from .seq_utils import one_hot_encode, rev_compl
from .sei_globals import SEI_WINDOW, SEI_STEP 

class Seq2Tensor(nn.Module):
    '''
    Encode sequences using one-hot encoding after preprocessing.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, seq: str):
        return one_hot_encode(seq, dtype=torch.float32)

class PadNs(nn.Module):
    '''
    Pad sequence with Ns up to requested size
    '''

    def __init__(self, 
                 size: int = SEI_WINDOW):
        super().__init__()
        self.size = size

    def forward(self, seq: str):
        add_sz = self.size - len(seq)
        lp, mod = divmod(add_sz, 2)
        lp, rp = lp + mod, lp
        return lp * 'N' + seq + rp * 'N'

DEFAULT_TRANSFORM = nn.Sequential(
    PadNs(size=SEI_WINDOW),
    Seq2Tensor()
)

class SubSequenceDataset(Dataset):
    def __init__(self, 
                 sequence: str, 
                 reverse_aug: bool = True,
                 window_size: int = SEI_WINDOW, 
                 step: int = SEI_STEP, 
                 transform=DEFAULT_TRANSFORM):
        """
            sequence: A single nucleotide sequence (e.g., 'ATGC...')
            window_size: Length of each subsequence.
            step: Step size to slide the window.
            transform (callable, optional): Optional transform to apply on each subsequence.
        """
        self.sequence = sequence

        self.window_size = window_size
        self.step = step
        self.transform = transform
        self.reverse_aug = reverse_aug
        # Calculate number of valid subsequences
        if len(self.sequence) <= self.window_size:
            self.num_samples = 1 
        else:
            self.num_samples = math.ceil(len(self.sequence) / step)

    def __len__(self):
        return self.num_samples

    def _encode(self, subseq: str):
        if self.transform is not None:
            subseq_enc = self.transform(subseq)
        return subseq_enc

    def __getitem__(self, idx):
        start = idx * self.step
        end = start + self.window_size
        subseq = self.sequence[start:end]
       
        if self.reverse_aug:
            rev_subseq = rev_compl(subseq)
            return self._encode(subseq), self._encode(rev_subseq), start
        else:
            return self._encode(subseq), start