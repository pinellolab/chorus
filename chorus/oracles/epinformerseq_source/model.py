"""Vendored copy of the EPInformer-seq enhancer_predictor_256bp model.

Source-of-truth: https://github.com/pinellolab/EPInformer
(EPInformer/models.py, classes ``seq_256bp_encoder`` and
``enhancer_predictor_256bp``)

Only the two classes used at inference time are included here, with no
extra dependencies beyond ``torch.nn``. The model takes a one-hot 256-bp
DNA tensor of shape ``(B, 4, 256)`` (or ``(B, 4, 1, 256)``) and returns a
single scalar per sequence: ``log2(0.1 + sqrt(DNase × H3K27ac))``.
"""

from __future__ import annotations

import torch.nn as nn


class seq_256bp_encoder(nn.Module):
    """Conv stem + 4-block conv tower → (B, 128, 1, 16) latent."""

    def __init__(self, base_size: int = 4, out_dim: int = 128, conv_dim: int = 256):
        super().__init__()
        self.conv_dim = conv_dim
        self.out_dim = out_dim
        self.base_size = base_size
        self.stem_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=base_size,
                out_channels=self.conv_dim,
                kernel_size=(1, 8),
                stride=1,
                padding="same",
            ),
            nn.ELU(),
        )
        self.conv_tower = nn.ModuleList([])
        conv_dims = [self.conv_dim, 128, 64, 64, 128]
        for i in range(4):
            self.conv_tower.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=conv_dims[i],
                        out_channels=conv_dims[i + 1],
                        kernel_size=(1, 3),
                        padding=(0, 1),
                    ),
                    nn.BatchNorm2d(conv_dims[i + 1]),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                )
            )
            self.conv_tower.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=conv_dims[i + 1],
                        out_channels=conv_dims[i + 1],
                        kernel_size=(1, 1),
                    ),
                    nn.ELU(),
                )
            )

    def forward(self, enhancers_input):
        if enhancers_input.shape[2] == 1:
            x = enhancers_input
        else:
            x = enhancers_input.permute(0, 3, 1, 2).contiguous()
        x = self.stem_conv(x)
        for i in range(0, len(self.conv_tower), 2):
            x = self.conv_tower[i](x)
            x = self.conv_tower[i + 1](x) + x
        return x


class enhancer_predictor_256bp(nn.Module):
    """256-bp one-hot DNA → scalar log2 activity."""

    def __init__(self):
        super().__init__()
        self.encoder = seq_256bp_encoder()
        self.embedToAct = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(128 * 16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, enhancer_seq):
        if len(enhancer_seq.shape) < 4:
            enhancer_seq = enhancer_seq.unsqueeze(2)
        seq_embed = self.encoder(enhancer_seq)
        out = self.embedToAct(seq_embed)
        return out.squeeze(-1)
