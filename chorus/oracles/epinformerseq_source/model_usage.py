"""Helper functions for loading and running the EPInformer-seq per-cell model.

One ``PerCellProfileNetWide`` checkpoint per cell (2114-bp input -> central
1024-bp crop) + a matching frozen ``BiasNet`` per cell (ChromBPNet-style Tn5 /
MNase sequence-bias subtraction). Predict path returns either the per-bp profile
or a scalar DNase activity (``max DNase`` over the central 256 bp).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .model import PerCellProfileNetWide, BiasNet
from .globals import (
    EPINFORMERSEQ_AVAILABLE_CELLTYPES,
    EPINFORMERSEQ_WINDOW,
    EPINFORMERSEQ_WIDE_WINDOW,
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


def load_main_model(
    weights_path: str,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Load this cell's main checkpoint (``PerCellProfileNetWide``, 2114->1024)."""
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model = PerCellProfileNetWide(
        in_window=EPINFORMERSEQ_WIDE_WINDOW, out_window=EPINFORMERSEQ_WINDOW
    )
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
    in_window: int = EPINFORMERSEQ_WIDE_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict per-bp DNase + H3K27ac profile for one ``cell_type``.

    ``cell_type`` is only used for validation -- the main/bias models passed in
    are already specific to one cell. Per-cell architecture means no cell_id
    tensor is fed to the model.

    ``in_window`` is the main model's input length: ``EPINFORMERSEQ_WINDOW``
    (1024) for the standard model, ``EPINFORMERSEQ_WIDE_WINDOW`` (2114) for the
    widewin variant. The profile output is always the central 1024 bp; the
    BiasNet (1024-bp input) runs on the central 1024 bp of the input, so for the
    widewin path the main sees the full 2114 bp while the bias sees the centre.

    Returns
    -------
    dnase : np.ndarray, shape (1024,)
        Per-bp DNase signal (softmax(profile) * 10**log_count).
    h3k27ac : np.ndarray, shape (1024,)
        Per-bp H3K27ac signal.
    counts : np.ndarray, shape (2,)
        Total predicted log10-counts (DNase, H3K27ac).
    """
    if cell_type not in EPINFORMERSEQ_AVAILABLE_CELLTYPES:
        raise ValueError(f"Unknown cell type {cell_type!r}")

    out_window = EPINFORMERSEQ_WINDOW
    pad = (in_window - out_window) // 2
    ohe = one_hot_dna(seq, length=in_window)               # (4, in_window)

    def _run(ohe_arr: np.ndarray):
        x_main = torch.from_numpy(ohe_arr).unsqueeze(0).to(device)     # (1, 4, in_window)
        ohe_c = ohe_arr if pad == 0 else ohe_arr[:, pad:pad + out_window].copy()
        x_bias = torch.from_numpy(ohe_c).unsqueeze(0).to(device)       # (1, 4, 1024)
        mp, mc = main(x_main)                 # (1, 2, 1024), (1, 2)
        bp, _ = bias(x_bias)                  # (1, 2, 1024)
        soft = torch.softmax(mp + bp, dim=-1)
        count = 10.0 ** mc                    # (1, 2)
        return soft * count.unsqueeze(-1), count

    with torch.inference_mode():
        signal, count = _run(ohe)
        if average_reverse:
            signal_r, count_r = _run(one_hot_rc(ohe))
            signal_r = torch.flip(signal_r, dims=[-1])   # flip back to fwd
            signal = 0.5 * (signal + signal_r)
            count  = 0.5 * (count + count_r)
    sig = signal[0].cpu().numpy()             # (2, 1024)
    return sig[0].astype(np.float32), sig[1].astype(np.float32), count[0].cpu().numpy().astype(np.float32)


def predict_activity(
    main: torch.nn.Module,
    bias: torch.nn.Module,
    seq: str,
    cell_type: str,
    *,
    assay: str = "Enhancer_DNase",
    average_reverse: bool = True,
    device: str | torch.device = "cpu",
    in_window: int = EPINFORMERSEQ_WIDE_WINDOW,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict a single scalar enhancer-activity for one cell.

    ``assay='Enhancer_DNase'`` (default) returns the per-bp peak max of the
    DNase channel; ``'Enhancer_H3K27ac'`` the H3K27ac channel; and
    ``'Enhancer_H3K27ac_DNase'`` the composite sqrt(max DNase * max H3K27ac) --
    all over the central 256 bp.

    The peak max is taken over the central 256 bp of the 1024-bp window
    (positions 384–639) to match the background CDF builder
    (scripts/build_backgrounds_epinformerseq_v2_percell.py); using the
    full window would let off-summit signal drift the percentile lookup.
    """
    dnase, h3, counts = predict_profile(
        main, bias, seq, cell_type, average_reverse=average_reverse,
        device=device, in_window=in_window,
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
