"""Batched forward pass for a CATv1 checkpoint, inside chorus-cherimoya.

Run by ``CherimoyaOracle._forward_windows`` via ``run_code_in_environment``.
``__ARGS_FILE_NAME__`` is substituted with a temp JSON path by the caller.

The contract is deliberately narrow: this template receives a list of
sequences that are **already** exactly ``input_length`` bp and returns raw
head outputs.  All geometry — tiling, padding, stitching, and combining
the two heads into expected counts — lives in the parent process, where
it is unit-testable and shared with the background builder.  (ChromBPNet's
equivalent template tiles internally, which duplicates the sliding-window
formula between the template and the oracle; that formula has already
been the subject of one off-by-one fix, so it is worth having in exactly
one place.)
"""

import json
import os

with open("__ARGS_FILE_NAME__") as inp:  # to be formatted by calling script
    args = json.load(inp)

device = args["device"]
if device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# NOTE: `cuda:N` deliberately does NOT touch CUDA_VISIBLE_DEVICES. Torch already
# interprets the ordinal within the visible set, so masking here did two wrong
# things at once: it overrode any outer mask a scheduler had set -- landing the
# process on a GPU it was not granted -- and, because the resolved device string
# `cuda:N` is then handed to torch anyway, it made every N except 0 crash with
# `CUDA error: invalid device ordinal` (masking to one device, then asking for
# device N of one). Verified 2026-08-13. See audits/2026-08-12_post_v0.7.2_audit.md F1.

import numpy
import torch

# Reuse Triton's autotune choice from a previous process instead of
# re-benchmarking it for this one forward pass (~7.2s/call on CATv1).  Must
# precede `import cherimoya`, which is when the decorators read the knob.
# Kept inline rather than imported from cherimoya_source/_triton_autotune.py
# (which documents why) to keep this template self-contained -- importing
# chorus in this env costs ~0.6s, which would eat much of the saving.
try:
    from triton import knobs
    knobs.autotuning.cache = True
except (ImportError, AttributeError):
    pass

from cherimoya import Cherimoya

if device in (None, "", "auto"):
    resolved = "cuda" if torch.cuda.is_available() else "cpu"
elif device == "gpu":
    resolved = "cuda"
else:
    resolved = device

windows = args["windows"]
batch_size = args.get("batch_size", 64)

# compile=False is mandatory, not a preference.  Cherimoya.load defaults
# to compile=True with compile_mode='max-autotune', and this template runs
# in a fresh subprocess on every predict call -- so a compiling load would
# pay a full max-autotune warmup per prediction, which reads as a hang.
model = Cherimoya.load(args["model_weights"], device=resolved, compile=False)
model = model.eval()

MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot(seq):
    """(4, len(seq)) float32 one-hot; ambiguous bases stay all-zero."""
    out = numpy.zeros((4, len(seq)), dtype=numpy.float32)
    for i, base in enumerate(seq.upper()):
        j = MAPPING.get(base)
        if j is not None:
            out[j, i] = 1.0
    return out


encoded = [one_hot(s) for s in windows]

all_logits = []
all_log_counts = []
with torch.no_grad():
    for i in range(0, len(encoded), batch_size):
        batch = numpy.stack(encoded[i:i + batch_size])
        X = torch.from_numpy(batch).to(resolved)
        profile_logits, log_counts = model(X)
        all_logits.append(profile_logits.float().cpu().numpy())
        all_log_counts.append(log_counts.float().cpu().numpy())

profile_logits = numpy.concatenate(all_logits)     # (n_windows, 1, output_length)
log_counts = numpy.concatenate(all_log_counts)     # (n_windows, 1)

# The resolved device is returned, not just used. Cherimoya's Triton
# kernels and the pure-PyTorch CPU fallback agree only to ~1e-2 on the
# logits, so a silent auto-detect fallback to CPU -- a busy GPU, a failed
# CUDA init -- would quietly change the numbers *and* run ~50x slower.
# For a 1,518-model background build that has to be visible, not inferred.
result = [profile_logits.tolist(), log_counts.tolist(), str(resolved)]
