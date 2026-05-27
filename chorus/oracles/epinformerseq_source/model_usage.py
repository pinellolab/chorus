"""Helper functions for loading and running the EPInformer-seq per-cell model.

One ``PerCellProfileNet`` checkpoint per cell + a matching frozen
``BiasNet`` per cell (ChromBPNet-style Tn5 / MNase sequence-bias
subtraction). Predict path returns either per-bp 2-channel profiles
or a scalar enhancer-activity in the units of the legacy oracle
(``sqrt(max DNase × max H3K27ac)`` over the 1024-bp window).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .model import PerCellProfileNet, BiasNet
from .globals import (
    EPINFORMERSEQ_AVAILABLE_CELLTYPES,
    EPINFORMERSEQ_WINDOW,
)


# 1024-bp ACGT one-hot (matches PerCellProfileNet's input length).
_ACGT = np.frombuffer(b"ACGT", dtype=np.uint8)
_ALPHABET = {b: i for i, b in enumerate(_ACGT)}


def one_hot_dna(seq: str, length: int = EPINFORMERSEQ_WINDOW) -> np.ndarray:
    """One-hot encode a DNA string to shape (4, length), float32.

    Upper-cases, right-pads/truncates to length, encodes non-ACGT as all-zeros.
    """
    s = seq.upper().encode("ascii", errors="ignore")
    if len(s) > length:
        # Center crop if too long (keeps the variant in the middle of the window).
        excess = len(s) - length
        s = s[excess // 2 : excess // 2 + length]
    arr = np.frombuffer(s, dtype=np.uint8)
    out = np.zeros((4, length), dtype=np.float32)
    for i, b in enumerate(arr):
        ch = _ALPHABET.get(b)
        if ch is not None:
            out[ch, i] = 1.0
    return out


def one_hot_rc(ohe: np.ndarray) -> np.ndarray:
    """Reverse-complement of a one-hot (4, L) array."""
    return ohe[::-1, ::-1].copy()


def load_main_model(weights_path: str, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Load this cell's PerCellProfileNet checkpoint."""
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model = PerCellProfileNet()
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def load_bias_model(weights_path: str, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Load one cell's frozen BiasNet checkpoint."""
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model = BiasNet()
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model.to(device)
    return model


def predict_profile(
    main: torch.nn.Module,
    bias: torch.nn.Module,
    seq: str,
    cell_type: str,
    *,
    average_reverse: bool = True,
    device: str | torch.device = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict per-bp DNase + H3K27ac profile for one ``cell_type``.

    ``cell_type`` is only used for validation -- the main/bias models passed in
    are already specific to one cell. Per-cell architecture means no cell_id
    tensor is fed to the model.

    Returns
    -------
    dnase : np.ndarray, shape (L,)
        Per-bp DNase signal (softmax(profile) * 10**log_count).
    h3k27ac : np.ndarray, shape (L,)
        Per-bp H3K27ac signal.
    counts : np.ndarray, shape (2,)
        Total predicted log10-counts (DNase, H3K27ac).
    """
    if cell_type not in EPINFORMERSEQ_AVAILABLE_CELLTYPES:
        raise ValueError(f"Unknown cell type {cell_type!r}")

    ohe = one_hot_dna(seq, length=EPINFORMERSEQ_WINDOW)
    x = torch.from_numpy(ohe).unsqueeze(0).to(device)               # (1, 4, L)

    with torch.inference_mode():
        mp, mc = main(x)                      # (1, 2, L), (1, 2)
        bp, _ = bias(x)                       # (1, 2, L)
        final = mp + bp                       # logit-space bias correction
        soft = torch.softmax(final, dim=-1)
        count = 10.0 ** mc                    # (1, 2)
        signal = soft * count.unsqueeze(-1)   # (1, 2, L)
        if average_reverse:
            ohe_rc = one_hot_rc(ohe)
            x_rc = torch.from_numpy(ohe_rc).unsqueeze(0).to(device)
            mp_r, mc_r = main(x_rc)
            bp_r, _ = bias(x_rc)
            final_r = mp_r + bp_r
            soft_r = torch.softmax(final_r, dim=-1)
            count_r = 10.0 ** mc_r
            signal_r = soft_r * count_r.unsqueeze(-1)
            signal_r = torch.flip(signal_r, dims=[-1])   # flip back to fwd
            signal = 0.5 * (signal + signal_r)
            count  = 0.5 * (count + count_r)
    sig = signal[0].cpu().numpy()             # (2, L)
    return sig[0].astype(np.float32), sig[1].astype(np.float32), count[0].cpu().numpy().astype(np.float32)


def predict_activity(
    main: torch.nn.Module,
    bias: torch.nn.Module,
    seq: str,
    cell_type: str,
    *,
    assay: str = "Enhancer_H3K27ac_DNase",
    average_reverse: bool = True,
    device: str | torch.device = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict a single scalar enhancer-activity for one cell.

    For ``assay='Enhancer_H3K27ac_DNase'`` (default) returns the geometric
    mean of the per-bp DNase and H3K27ac peak signals (sqrt(D * H)). For the
    single-assay variants (Enhancer_DNase, Enhancer_H3K27ac) returns the per-bp
    peak max of that channel only.

    The peak max is taken over the central 256 bp of the 1024-bp window
    (positions 384–639) to match the background CDF builder
    (scripts/build_backgrounds_epinformerseq_v2_percell.py); using the
    full window would let off-summit signal drift the percentile lookup.
    """
    dnase, h3, counts = predict_profile(
        main, bias, seq, cell_type, average_reverse=average_reverse, device=device
    )
    # Central 256 bp slice — must match CENTRAL_START/END in the builder.
    c_start = (dnase.shape[-1] - 256) // 2
    c_end   = c_start + 256
    d_central = dnase[c_start:c_end]
    h_central = h3[c_start:c_end]
    if assay == "Enhancer_DNase":
        scalar = float(np.max(d_central))
    elif assay == "Enhancer_H3K27ac":
        scalar = float(np.max(h_central))
    else:  # combined / default
        scalar = float(np.sqrt(np.max(d_central) * np.max(h_central) + 1e-12))
    preds = np.array([scalar], dtype=np.float32)
    return preds, np.array([0], dtype=np.int64)
