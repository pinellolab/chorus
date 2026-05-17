"""EPInformer-seq v2.1: profile output + bias correction + cell conditioning.

Changes from v2:
  - DilatedResBlock adds an SELayer (channel attention, Hu et al. 2018)
    between BN and the residual add. Borrowed from LegNet (Penzar et al.
    2023). Lets the model learn which channels matter on a per-content
    basis. Complements FiLM, which does cell-conditional channel modulation.
  - Activation switches ELU -> SiLU (Swish) throughout the main model.
    SiLU usually edges out ELU by 0.5-1% on regression tasks.
  - BiasNet kept byte-for-byte identical to v2 (uses _V2DilatedResBlock)
    so the per-cell v2 bias.pt checkpoints load with no state-dict mismatch.

Architecture:
  - Conv stem (kernel=21) -> 64-channel body
  - 9 dilated residual blocks (kernel=3, dilations 1, 2, 4, ..., 256)
    each with SE + BN + SiLU + residual; receptive field ~ 1024 bp.
  - FiLM modulation from a 32-d cell embedding applied after every block
  - Profile head: per-bp 2-channel logits (DNase, H3K27ac), softmax over position
  - Count head: pooled trunk + cell embedding -> 2 scalar log10-counts

Output shapes (input length L=1024):
  profile_logits : (B, 2, L)    # softmax over dim=-1 per channel for multinomial NLL
  log_counts     : (B, 2)        # log10(total + 1) per channel
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    """Squeeze-and-Excitation channel attention (1D variant).

    global-mean pool -> bottleneck MLP -> sigmoid gate -> multiply.
    """

    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(ch, hidden),
            nn.SiLU(),
            nn.Linear(hidden, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean(dim=2)                 # (B, C)
        y = self.fc(y).unsqueeze(-1)      # (B, C, 1)
        return x * y


class DilatedResBlock(nn.Module):
    """v2.1: dilated 1D conv + BN + SE-gate + residual + SiLU."""

    def __init__(self, ch: int, dilation: int, kernel: int = 3,
                 se_reduction: int = 4):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(ch)
        self.se = SELayer(ch, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn(self.conv(x))
        h = self.se(h)
        return F.silu(h + x)


class FiLM1d(nn.Module):
    """Per-channel (gamma, beta) modulation from a context embedding."""

    def __init__(self, emb_dim: int, ch: int):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 2 * ch)
        self.ch = ch
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)  # gamma=0 -> 1+gamma=1, beta=0 -> identity

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        gb = self.fc(emb)                       # (B, 2*ch)
        gamma = gb[:, :self.ch].unsqueeze(-1)   # (B, ch, 1)
        beta = gb[:, self.ch:].unsqueeze(-1)
        return (1.0 + gamma) * x + beta


class CellCondProfileNet(nn.Module):
    """Main model: 1024 bp DNA + cell_id -> per-bp 2-channel profile + 2 scalar counts."""

    def __init__(self,
                 n_cells: int = 11,
                 stem_ch: int = 128,
                 body_ch: int = 64,
                 n_dilated: int = 9,
                 cell_emb: int = 32,
                 se_reduction: int = 4):
        super().__init__()
        self.n_cells = n_cells
        self.cell_emb = nn.Embedding(n_cells, cell_emb)
        self.stem = nn.Sequential(
            nn.Conv1d(4, stem_ch, kernel_size=21, padding=10),
            nn.SiLU(),
            nn.Conv1d(stem_ch, body_ch, kernel_size=1),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList()
        self.films = nn.ModuleList()
        for i in range(n_dilated):
            self.blocks.append(DilatedResBlock(body_ch, dilation=2 ** i,
                                               se_reduction=se_reduction))
            self.films.append(FiLM1d(cell_emb, body_ch))
        self.profile_head = nn.Conv1d(body_ch, 2, kernel_size=1)
        self.count_head = nn.Sequential(
            nn.Linear(body_ch + cell_emb, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor, cell_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.cell_emb(cell_id)         # (B, cell_emb)
        h = self.stem(x)                     # (B, body_ch, L)
        for blk, film in zip(self.blocks, self.films):
            h = blk(h)
            h = film(h, emb)
        profile_logits = self.profile_head(h)               # (B, 2, L)
        pooled = F.adaptive_avg_pool1d(h, 1).squeeze(-1)    # (B, body_ch)
        count_in = torch.cat([pooled, emb], dim=1)
        log_counts = self.count_head(count_in)              # (B, 2)
        return profile_logits, log_counts


class _V2DilatedResBlock(nn.Module):
    """Original v2 DilatedResBlock -- preserved so v2 BiasNet checkpoints load
    with no state-dict mismatch.
    """

    def __init__(self, ch: int, dilation: int, kernel: int = 3):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(self.bn(self.conv(x)) + x)


class BiasNet(nn.Module):
    """Bias model -- architecturally identical to v2 so v2 bias.pt loads
    directly into v2.1.

    Trained on random non-peak regions where signal is dominated by Tn5/MNase
    sequence preference. No cell conditioning -- one shared bias model across
    cells (the residual per-cell count offset is absorbed in the main model).
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
            [_V2DilatedResBlock(body_ch, dilation=2 ** i) for i in range(n_dilated)]
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


# ---------------------------------------------------------------------------
# Loss functions (ChromBPNet recipe) -- identical to v2
# ---------------------------------------------------------------------------

def multinomial_nll(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Multinomial NLL per-example, per-channel.

    logits : (B, C, L)   -- raw logits, softmaxed over dim=-1 per channel
    counts : (B, C, L)   -- observed integer counts per position

    Returns scalar mean over batch + channel.
    """
    logp = F.log_softmax(logits, dim=-1)
    nll = -(counts.float() * logp).sum(dim=-1)
    total = counts.float().sum(dim=-1).clamp(min=1.0)
    return (nll / total).mean()


def count_mse(log_pred: torch.Tensor, total_obs: torch.Tensor) -> torch.Tensor:
    """MSE in log10(1 + counts) space, per channel."""
    target = torch.log10(total_obs.float() + 1.0)
    return F.mse_loss(log_pred, target)
