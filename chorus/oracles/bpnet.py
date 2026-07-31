"""Public one-shot BPNet helpers for advanced users.

This module exposes the loader/predictor recipe that the
:class:`ChromBPNetOracle` runs internally for CHIP (BPNet-architecture)
weights, so users can score their own variants without standing up the
full chorus oracle scaffold.

The internal recipe (``chrombpnet.py:_load_direct``) requires four
non-obvious steps that the documented Keras flow does **not** capture:

1. Add ``chorus/oracles/chrombpnet_source/templates`` to ``sys.path``
   so the bundled ``BPNet.arch`` package becomes importable.
2. Build the model from ``input_data.json`` ``tasks`` dict using
   ``BPNet(tasks, {}, name_prefix="main")`` — *not*
   ``tf.keras.models.load_model``, which silently loads a half-broken
   model with a Lambda layer that references ``bpnet.model.arch`` (the
   author's source layout, not chorus's vendored copy). Predictions
   from that half-broken model fail silently with ``except: pass``.
3. Call ``model.load_weights(h5_path)`` to populate the weights.
4. Pass the model a 3-tuple ``[one_hot, profile_bias_zeros,
   count_bias_zeros]`` at predict time — BPNet expects bias tensors
   even when the user has none.

This module is import-safe only inside the ``chorus-chrombpnet`` conda
env (TF dependency). Use ``chorus.oracles.ChromBPNetOracle`` for the
managed multi-env flow.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default ChromBPNet/BPNet I/O dimensions. Matches the canonical
# Kundaje-lab models bundled with chorus.
DEFAULT_SEQUENCE_LENGTH = 2114
DEFAULT_OUTPUT_LENGTH = 1000

_BASE_MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}


def _templates_dir() -> str:
    """Return the absolute path to chorus's bundled BPNet templates dir."""
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(here, "chrombpnet_source", "templates")


def load_bpnet_model(weights_path: str, tasks_json: str | None = None) -> Any:
    """Load a BPNet/CHIP h5 weights file as a usable Keras model.

    Args:
        weights_path: Path to a BPNet h5 weights file (e.g. a
            JASPAR-trained TF-binding model).
        tasks_json: Optional path to a ``input_data.json`` describing
            the model's task heads. Defaults to the
            ``input_data.json`` bundled with chorus.

    Returns:
        A ``tf.keras.Model`` with weights loaded.

    Raises:
        ImportError: when TensorFlow is unavailable (run inside
            ``chorus-chrombpnet`` env).
        FileNotFoundError: when ``weights_path`` or ``tasks_json`` is missing.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"BPNet weights file not found: {weights_path}")

    tdir = _templates_dir()
    if tdir not in sys.path:
        sys.path.insert(0, tdir)
    from BPNet.arch import BPNet  # type: ignore  # noqa: E402

    if tasks_json is None:
        tasks_json = os.path.join(tdir, "input_data.json")
    if not os.path.exists(tasks_json):
        raise FileNotFoundError(f"BPNet tasks JSON not found: {tasks_json}")

    with open(tasks_json) as fh:
        tasks_raw = json.load(fh)
    tasks = {int(k): v for k, v in tasks_raw.items()}

    model = BPNet(tasks, {}, name_prefix="main")
    model.load_weights(weights_path)
    logger.info("Loaded BPNet model from %s", weights_path)
    return model


def encode_sequence(sequence: str) -> np.ndarray:
    """One-hot encode a DNA string as a ``(L, 4)`` float32 array.

    Bases outside ``ACGT`` are encoded as all-zero columns (matches the
    internal oracle behaviour).
    """
    L = len(sequence)
    out = np.zeros((L, 4), dtype=np.float32)
    for i, b in enumerate(sequence.upper()):
        idx = _BASE_MAPPING.get(b)
        if idx is not None:
            out[i, idx] = 1.0
    return out


def predict_bpnet(
    model: Any,
    sequence: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    output_length: int = DEFAULT_OUTPUT_LENGTH,
) -> dict[str, np.ndarray]:
    """Run a BPNet model on a single sequence and return the raw heads.

    Args:
        model: A model returned by :func:`load_bpnet_model`.
        sequence: A DNA string of length **exactly** ``sequence_length``
            (default 2114 bp for the canonical BPNet input). Use
            :func:`chorus.utils.get_centered_window` to build the
            correctly-sized input from a 1-based variant position.
        sequence_length: BPNet's expected input length (default 2114).
        output_length: BPNet's profile output length (default 1000).

    Returns:
        ``{"profile": np.ndarray (1, output_length, 2),
           "counts":  np.ndarray (1, 1)}`` — the raw BPNet head outputs.

        To get a usable per-base signal, take a **single softmax over both
        strands jointly** (all ``2 * output_length`` logits at once) and scale
        by ``exp(counts) - 2``::

            logits = profile.reshape(1, -1)
            logits = logits - logits.mean(axis=1, keepdims=True)
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            per_bp = (probs * (np.exp(counts) - 2)).reshape(profile.shape)
            plus, minus = per_bp[..., 0], per_bp[..., 1]

        Not ``softmax(profile) * exp(counts)`` per strand: the profile
        multinomial was trained over the flattened both-strand vector, the
        count target is a per-track ``log1p`` pooled across strands with
        ``logsumexp`` (so it equals ``log(2 + total)``), and scaling each
        strand by the full total double-counts. See
        :meth:`chorus.oracles.chrombpnet.ChromBPNetOracle._counts_from_log`
        and ``_transform_chip_strands``.
    """
    one_hot = encode_sequence(sequence)
    if one_hot.shape[0] != sequence_length:
        raise ValueError(
            f"BPNet expects an input of exactly sequence_length={sequence_length} "
            f"bp; got len={one_hot.shape[0]}. Use "
            f"chorus.utils.get_centered_window(..., length={sequence_length})."
        )

    one_hot_batch = one_hot[np.newaxis]
    # Derive the bias shapes from the model. The count bias is declared
    # (None, 2) and is logsumexp-reduced inside the model, so a hardcoded
    # (1, 1) — which Keras silently broadcasts rather than rejecting — makes
    # that term log(1)=0 instead of log(2), shifting every predicted log-count
    # down by a constant 0.5885 (counts 1.8x too low at a peak, up to 3x at a
    # quiet site).
    bias_inputs = [
        np.zeros(
            [1] + [d if d is not None else 1 for d in inp.shape[1:]],
            dtype=np.float32,
        )
        for inp in model.inputs[1:]
    ]

    profile, counts = model.predict_on_batch([one_hot_batch, *bias_inputs])
    return {"profile": np.asarray(profile), "counts": np.asarray(counts)}
