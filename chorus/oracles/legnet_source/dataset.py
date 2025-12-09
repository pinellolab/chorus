import torch.nn as nn
import math

from torch.utils.data import Dataset 
from .transforms import Seq2Tensor, AddFlanks, PadNs, ReverseComplement
from .legnet_globals import LEGNET_WINDOW, LEGNET_DEFAULT_STEP
from .agarwal_meta import LEFT_MPRA_FLANK, RIGHT_MPRA_FLANK


class SubSequenceDataset(Dataset):
    def __init__(self, 
                 sequence: str, 
                 reverse_aug: bool = True,
                 window_size: int = LEGNET_WINDOW, 
                 step: int = LEGNET_DEFAULT_STEP, 
                 left_flank: str = LEFT_MPRA_FLANK,
                 right_flank: str = RIGHT_MPRA_FLANK):
        """
            sequence: A single nucleotide sequence (e.g., 'ATGC...')
            window_size: Length of each subsequence.
            step: Step size to slide the window.
        """
        self.sequence = sequence

        self.window_size = window_size
        self.step = step
        self.transform = nn.Sequential(AddFlanks(up=left_flank,
                                               down=right_flank),
                                      PadNs(size=self.window_size),
                                      Seq2Tensor())
        self.rev_compl = ReverseComplement()

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
            rev_subseq = self.rev_compl(subseq)
            return self._encode(subseq), self._encode(rev_subseq), start
        else:
            return self._encode(subseq), start