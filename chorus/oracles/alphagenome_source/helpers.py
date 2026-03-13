"""Utility functions for AlphaGenome oracle."""

import numpy as np


def dna_1hot(seq: str) -> np.ndarray:
    """Convert DNA sequence to one-hot encoding (N x 4, ACGT order)."""
    seq = seq.upper()
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq_len = len(seq)
    one_hot = np.zeros((seq_len, 4), dtype=np.float32)
    for i, base in enumerate(seq):
        idx = mapping.get(base)
        if idx is not None:
            one_hot[i, idx] = 1.0
    return one_hot


def padseq(seq: str, window: int) -> str:
    """Pad sequence with N's to reach *window* length, centered."""
    tpad = window - len(seq)
    if tpad <= 0:
        return seq
    lpad = tpad // 2
    rpad = tpad - lpad
    return "N" * lpad + seq + "N" * rpad
