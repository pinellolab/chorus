"""Helper functions for loading and running the EPInformer-seq encoder."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .model import enhancer_predictor_256bp


# 256-bp ACGT one-hot. Order matches kipoiseq.transforms.functional.one_hot_dna
# (used in the EPInformer training pipeline).
_ACGT = np.frombuffer(b"ACGT", dtype=np.uint8)
_ALPHABET = {b: i for i, b in enumerate(_ACGT)}


def one_hot_dna(seq: str, length: int = 256) -> np.ndarray:
    """One-hot encode a DNA string to shape (4, length), float32.

    - Upper-cases input.
    - Pads/truncates to ``length`` (right-pad with zeros if too short;
      truncate from the right if too long).
    - Non-ACGT bases (N, etc.) are encoded as all-zeros (no channel set).
    - Channel order is A, C, G, T.
    """
    s = seq.upper().encode("ascii", errors="ignore")
    if len(s) > length:
        s = s[:length]
    arr = np.frombuffer(s, dtype=np.uint8)
    out = np.zeros((4, length), dtype=np.float32)
    for i, b in enumerate(arr):
        ch = _ALPHABET.get(b)
        if ch is not None:
            out[ch, i] = 1.0
    return out


def one_hot_rc(ohe: np.ndarray) -> np.ndarray:
    """Reverse-complement of a one-hot (4, L) array. ACGT → TGCA channel flip + reverse."""
    return ohe[::-1, ::-1].copy()


def load_model(weights_path: str, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Instantiate enhancer_predictor_256bp and load a state-dict checkpoint.

    Checkpoint format (from train_seqEncoder.py:444-448):
        {"model_state_dict": ..., "model_name": ..., "fold_i": ..., "cell": ...}

    For backwards compatibility we also accept a raw state_dict file.
    """
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    model = enhancer_predictor_256bp()
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def predict_activity(
    model: torch.nn.Module,
    seq: str,
    *,
    average_reverse: bool = True,
    device: str | torch.device = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict scalar activity for a 256-bp sequence.

    Returns
    -------
    preds : np.ndarray, shape (1,)
        Single scalar prediction in linear activity units (RPM-space):
        sqrt(DNase_RPM × H3K27ac_RPM), clipped to >= 0.

        The trained model emits log2(0.1 + activity); we un-transform here so
        downstream consumers see a non-negative scalar that's directly
        interpretable as enhancer activity.  RC averaging is performed in
        log2-space (matches training-time symmetry) before the un-transform.
    bins : np.ndarray, shape (1,)
        Bin index (always [0] for this single-window oracle); kept for API
        parity with sliding-window oracles like LegNet's predict_bigseq.
    """
    ohe = one_hot_dna(seq, length=256)
    x = torch.from_numpy(ohe).unsqueeze(0).to(device)  # (1, 4, 256)
    with torch.inference_mode():
        pred_log2 = model(x).cpu().numpy().reshape(-1)  # (1,)
        if average_reverse:
            ohe_rc = one_hot_rc(ohe)
            x_rc = torch.from_numpy(ohe_rc).unsqueeze(0).to(device)
            pred_rc = model(x_rc).cpu().numpy().reshape(-1)
            pred_log2 = (pred_log2 + pred_rc) / 2.0
    # Un-transform log2(0.1 + activity) -> linear activity.  Clip tiny
    # negatives that arise when pred_log2 < log2(0.1) (low-signal regions).
    pred = np.maximum(np.power(2.0, pred_log2) - 0.1, 0.0)
    return pred.astype(np.float32), np.array([0], dtype=np.int64)
