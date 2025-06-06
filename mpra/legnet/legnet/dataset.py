from torch.utils.data import Dataset 
from .transforms import Seq2Tensor

class SubSequenceDataset(Dataset):
    def __init__(self, 
                 sequence: str, 
                 window_size: int=200, 
                 step: int =50, 
                 transform=Seq2Tensor):
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

        # Calculate number of valid subsequences
        if len(self.sequence) <= self.window_size:
            self.num_samples = 1 
        else:
            self.num_samples = (len(self.sequence) - self.window_size) // self.step + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.step
        end = start + self.window_size
        if (len(self.sequence) - end) < self.step:
            end = len(self.sequence)

        subseq = self.sequence[start:end]

        if self.transform is not None:
            subseq = self.transform(subseq)
        return subseq, start 


class SubSequenceDataset(Dataset):
    def __init__(self, 
                 sequence: str, 
                 window_size: int=200, 
                 step: int =50, 
                 transform=Seq2Tensor):
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

        # Calculate number of valid subsequences
        if len(self.sequence) <= self.window_size:
            self.num_samples = 1 
        else:
            self.num_samples = (len(self.sequence) - self.window_size) // self.step + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.step
        end = start + self.window_size
        if (len(self.sequence) - end) < self.step:
            end = len(self.sequence)

        subseq = self.sequence[start:end]

        if self.transform is not None:
            subseq = self.transform(subseq)
        return subseq, start 
    