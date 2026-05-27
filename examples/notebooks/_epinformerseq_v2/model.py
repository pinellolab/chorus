"""EPInformer-seq per-cell: profile output + bias correction.

One model per cell (no FiLM, no cell embedding). Architecture mirrors
`legnet_profile_v2/model.py` PerCellProfileNet (per-rep BAM training,
2026-05-27 sweep, 11 Roadmap cells). The previous joint-cell
`CellCondProfileNet` (FiLM + cell_emb) was retired in favor of the
per-cell variant which gives consistently equal or better test-r and
cleaner cell-specificity at validated enhancers.

Architecture:
  - Conv stem (kernel=21) -> 64-channel body
  - 9 dilated residual blocks (kernel=3, dilations 1, 2, 4, ..., 256)
    each with BN + ELU + residual; receptive field ~ 1024 bp.
  - No cell conditioning -- one model per cell.
  - Profile head: per-bp 2-channel logits (DNase, H3K27ac), softmax over position.
  - Count head: pooled trunk -> 64-d -> 2 scalar log10-counts.

Output shapes (input length L=1024):
  profile_logits : (B, 2, L)    # softmax over dim=-1 per channel for multinomial NLL
  log_counts     : (B, 2)        # log10(total + 1) per channel

BiasNet stays byte-identical to the prior implementation so per-cell
`bias.pt` checkpoints load without state-dict mismatch.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResBlock(nn.Module):
    """Dilated 1D conv + BN + ELU with residual skip. Length-preserving."""

    def __init__(self, ch: int, dilation: int, kernel: int = 3):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(self.bn(self.conv(x)) + x)


class PerCellProfileNet(nn.Module):
    """Per-cell profile + counts. 1024-bp DNA -> 2-channel profile + 2 scalar counts."""

    def __init__(self,
                 stem_ch: int = 128,
                 body_ch: int = 64,
                 n_dilated: int = 9):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, stem_ch, kernel_size=21, padding=10),
            nn.ELU(),
            nn.Conv1d(stem_ch, body_ch, kernel_size=1),
            nn.ELU(),
        )
        self.blocks = nn.ModuleList(
            [DilatedResBlock(body_ch, dilation=2 ** i) for i in range(n_dilated)]
        )
        self.profile_head = nn.Conv1d(body_ch, 2, kernel_size=1)
        self.count_head = nn.Sequential(
            nn.Linear(body_ch, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        profile_logits = self.profile_head(h)                # (B, 2, L)
        pooled = F.adaptive_avg_pool1d(h, 1).squeeze(-1)     # (B, body_ch)
        log_counts = self.count_head(pooled)                 # (B, 2)
        return profile_logits, log_counts


class BiasNet(nn.Module):
    """Per-cell bias model -- ChromBPNet-style sequence-bias subtraction.

    Trained on random non-peak regions where signal is dominated by Tn5 / MNase
    sequence preference. Frozen during main training. Architecturally identical
    to the v2 bias checkpoint shape so existing per-cell bias.pt files load.
    """

    def __init__(self, stem_ch: int = 64, body_ch: int = 32, n_dilated: int = 9):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, stem_ch, kernel_size=21, padding=10),
            nn.ELU(),
            nn.Conv1d(stem_ch, body_ch, kernel_size=1),
            nn.ELU(),
        )
        self.blocks = nn.ModuleList(
            [DilatedResBlock(body_ch, dilation=2 ** i) for i in range(n_dilated)]
        )
        self.profile_head = nn.Conv1d(body_ch, 2, kernel_size=1)
        self.count_head = nn.Sequential(
            nn.Linear(body_ch, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        profile_logits = self.profile_head(h)
        pooled = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        log_counts = self.count_head(pooled)
        return profile_logits, log_counts
