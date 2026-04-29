"""PyTorch-side runner for the JAX↔PT equivalence stress test.

Loads the alphagenome-pytorch port once, then predicts at the SORT1 locus
across multiple window sizes × heads. Saves to a single npz with the same
key scheme as run_jax.py so the diff script can compare directly.

Run inside chorus-alphagenome_pt env:
    mamba run -n chorus-alphagenome_pt python audits/.../run_pt.py <output.npz> [device]

device defaults to "cpu" — Mac MPS is faster ≤524 kb but cliffs at 1 MB,
so the stress test picks device per-window via the routing recommendation.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np

# pip torch + conda-forge compilers libomp clash — same shim as predict_template.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pyfaidx  # noqa: E402

GENOME_FASTA = "/Users/lp698/Projects/chorus.bak-2026-04-28/genomes/hg38.fa"
SORT1_TSS = ("chr1", 109_274_968)

# Same matrix as run_jax.py — keys must match exactly for diffing.
WINDOWS = [(2**17, "128k"), (2**19, "524k"), (2**20, "1m")]
HEADS = ["DNASE", "ATAC", "CAGE", "RNA_SEQ"]
_PT_KEY = {"DNASE": "dnase", "ATAC": "atac", "CAGE": "cage", "RNA_SEQ": "rna_seq"}

# Per-window device decision (mirrors recommend_alphagenome_backend heuristic):
# MPS for ≤600kb, CPU for ≥768kb. CPU is slower but stable past the cliff.
DEVICE_PER_WINDOW = {"128k": "mps", "524k": "mps", "1m": "cpu"}


def main() -> None:
    out_path = sys.argv[1]
    chrom, tss = SORT1_TSS
    fa = pyfaidx.Fasta(GENOME_FASTA)

    print("Loading PT AlphaGenome port (CPU initial; per-window switch)...", flush=True)
    t0 = time.time()
    import torch
    import huggingface_hub
    # Apply MPS rope shim before model construction
    sys.path.insert(0, "/Users/lp698/Projects/chorus.bak-2026-04-28")
    from chorus.oracles.alphagenome_pt_source import _mps_compat  # noqa: F401
    from alphagenome_pytorch import AlphaGenome

    weights_path = huggingface_hub.hf_hub_download(
        "gtca/alphagenome_pytorch", "model_all_folds.safetensors"
    )
    print(f"  weights @ {weights_path}", flush=True)

    # Load on CPU — we'll move per-window when needed (avoids reload).
    model = AlphaGenome.from_pretrained(weights_path, device=torch.device("cpu"))
    model.eval()
    print(f"  loaded (cpu) in {time.time() - t0:.1f}s", flush=True)

    pt_heads = tuple(_PT_KEY[h] for h in HEADS)
    results: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    devices: dict[str, str] = {}

    current_device_str = "cpu"
    current_device = torch.device("cpu")

    for win_bp, win_label in WINDOWS:
        target_device_str = DEVICE_PER_WINDOW[win_label]
        if target_device_str == "mps" and not torch.backends.mps.is_available():
            target_device_str = "cpu"
        if target_device_str != current_device_str:
            t_move = time.time()
            current_device = torch.device(target_device_str)
            model = model.to(current_device)
            current_device_str = target_device_str
            print(f"  moved model to {current_device_str} in {time.time()-t_move:.1f}s", flush=True)
        devices[win_label] = current_device_str

        half = win_bp // 2
        seq = str(fa[chrom][tss - half:tss + half]).upper().strip("N")
        valid = [2**p for p in range(15, 21)]
        target = max(v for v in valid if v <= len(seq))
        trim = (len(seq) - target) // 2
        seq = seq[trim:trim + target]
        # one-hot
        nuc = {"A": 0, "C": 1, "G": 2, "T": 3}
        arr = np.zeros((len(seq), 4), dtype=np.float32)
        for i, b in enumerate(seq):
            j = nuc.get(b, -1)
            if j >= 0:
                arr[i, j] = 1.0
        x = torch.from_numpy(arr).unsqueeze(0).to(current_device)
        org = torch.tensor([0], dtype=torch.long, device=current_device)

        print(f"window {win_label} ({len(seq)} bp) on {current_device_str}: predicting...", flush=True)
        t0 = time.time()
        with torch.no_grad():
            out = model(x, organism_index=org, heads=pt_heads)
        # MPS sync
        if current_device_str == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        timings[win_label] = dt
        print(f"  predict {win_label}: {dt:.2f}s", flush=True)

        for h, key in zip(HEADS, pt_heads):
            head_out = out[key]
            # Some heads return a dict {resolution: tensor}; others return tensor directly.
            if isinstance(head_out, dict):
                # Pick highest resolution available (1 = base-pair)
                res = min(head_out.keys())
                tensor = head_out[res]
            else:
                tensor = head_out
            data = tensor[0].detach().cpu().numpy().astype(np.float32)  # (length, n_tracks)
            results[f"{h.lower()}_{win_label}"] = data
            print(f"    {h:10s} shape={data.shape}", flush=True)

    np.savez_compressed(out_path, **results)
    meta = {
        "backend": "pt",
        "device_per_window": devices,
        "windows": [w[1] for w in WINDOWS],
        "heads": HEADS,
        "timings_s": timings,
        "platform": platform.platform(),
    }
    with open(out_path.replace(".npz", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({"ok": True, "n_arrays": len(results), **meta}))


if __name__ == "__main__":
    main()
