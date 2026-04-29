"""Numerical equivalence between AlphaGenome JAX and PyTorch backends.

Gated with ``@pytest.mark.integration`` — runs both backends inside their
respective conda envs, predicts a 1 MB SORT1 window with each, and asserts
the per-track outputs agree within tolerance.

Run from the repo root:

    pytest tests/test_alphagenome_backends_equivalence.py -m integration -v

Skips automatically if either env is missing.
"""
from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest


SORT1_WINDOW = (
    "chr1",
    109_274_968 - 524_288,
    109_274_968 + 524_288,
)
GENOME_FASTA = Path(__file__).parent.parent / "genomes" / "hg38.fa"


def _env_exists(env_name: str) -> bool:
    try:
        out = subprocess.run(
            ["mamba", "env", "list", "--json"],
            capture_output=True, text=True, check=True,
        )
        envs = json.loads(out.stdout).get("envs", [])
        return any(env_name == os.path.basename(p) for p in envs)
    except Exception:
        return False


PT_RUNNER = textwrap.dedent(
    r"""
    import os, sys, json, numpy as np
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    args = json.loads(sys.argv[1])
    fasta = args["fasta"]; chrom = args["chrom"]
    start = args["start"]; end = args["end"]; device_name = args["device"]

    import pyfaidx
    seq = str(pyfaidx.Fasta(fasta)[chrom][start:end]).upper()
    seq = seq.strip("N")
    # round to nearest power of 2 in [2**15, 2**20]
    valid = [2**p for p in range(15, 21)]
    target = max(v for v in valid if v <= len(seq))
    trim = (len(seq) - target) // 2
    seq = seq[trim:trim + target]

    import torch, huggingface_hub
    from alphagenome_pytorch import AlphaGenome

    weights = huggingface_hub.hf_hub_download("gtca/alphagenome_pytorch", "model_all_folds.safetensors")
    device = torch.device(device_name)
    model = AlphaGenome.from_pretrained(weights, device=device); model.eval()

    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq):
        j = "ACGT".find(b)
        if j >= 0: arr[i, j] = 1.0
    x = torch.from_numpy(arr).unsqueeze(0).to(device)
    org = torch.tensor([0], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x, organism_index=org, heads=("atac","dnase"))
    dnase_1 = out["dnase"][1][0].detach().cpu().numpy()  # (L, n_tracks)
    np.savez_compressed(args["out"], dnase_1=dnase_1)
    print(json.dumps({"length": len(seq), "shape": list(dnase_1.shape)}))
    """
)


JAX_RUNNER = textwrap.dedent(
    r"""
    import os, sys, json, platform, numpy as np
    if platform.system() == "Darwin":
        os.environ["JAX_PLATFORMS"] = "cpu"

    args = json.loads(sys.argv[1])
    fasta = args["fasta"]; chrom = args["chrom"]
    start = args["start"]; end = args["end"]

    import pyfaidx
    seq = str(pyfaidx.Fasta(fasta)[chrom][start:end]).upper()
    seq = seq.strip("N")
    valid = [2**p for p in range(15, 21)]
    target = max(v for v in valid if v <= len(seq))
    trim = (len(seq) - target) // 2
    seq = seq[trim:trim + target]

    import jax
    from alphagenome.models.dna_output import OutputType
    from alphagenome_research.model.dna_model import create_from_huggingface
    jax_device = jax.devices("cpu")[0]
    model = create_from_huggingface("all_folds", device=jax_device)
    out = model.predict_sequence(seq, requested_outputs=[OutputType.DNASE], ontology_terms=None)
    dnase = np.asarray(out[OutputType.DNASE].values)  # (L, n_tracks)
    np.savez_compressed(args["out"], dnase=dnase)
    print(json.dumps({"length": len(seq), "shape": list(dnase.shape)}))
    """
)


def _run(env: str, code: str, args: dict, tmp_out: Path) -> dict:
    args = {**args, "out": str(tmp_out)}
    r = subprocess.run(
        ["mamba", "run", "-n", env, "python", "-c", code, json.dumps(args)],
        capture_output=True, text=True, timeout=1800,
    )
    if r.returncode != 0:
        raise RuntimeError(f"subprocess in {env} failed:\n{r.stderr}")
    last = next(l for l in reversed(r.stdout.strip().splitlines()) if l.startswith("{"))
    return json.loads(last)


@pytest.mark.integration
def test_jax_pt_dnase_equivalence_at_sort1(tmp_path):
    """JAX and PyTorch backends should agree on DNase predictions at the
    SORT1 locus to within tight tolerance.

    The PyTorch port's README claims per-head + full-forward equivalence
    against the JAX checkpoint. This test pins that claim against actual
    chorus inputs (one assay, one canonical region) so a future weight
    refresh upstream doesn't silently drift our results.
    """
    if not GENOME_FASTA.exists():
        pytest.skip("hg38.fa not present — `chorus genome download hg38` first")
    if not _env_exists("chorus-alphagenome"):
        pytest.skip("chorus-alphagenome env missing")
    if not _env_exists("chorus-alphagenome_pt"):
        pytest.skip("chorus-alphagenome_pt env missing")

    chrom, start, end = SORT1_WINDOW
    pt_out = tmp_path / "pt.npz"
    jax_out = tmp_path / "jax.npz"

    base_args = {
        "fasta": str(GENOME_FASTA),
        "chrom": chrom, "start": start, "end": end,
    }
    _run("chorus-alphagenome_pt", PT_RUNNER,
         {**base_args, "device": "cpu"}, pt_out)
    _run("chorus-alphagenome", JAX_RUNNER, base_args, jax_out)

    pt = np.load(pt_out)["dnase_1"]
    jx = np.load(jax_out)["dnase"]

    # Shapes should match (L, n_tracks). PyTorch returns (L, n_tracks)
    # at resolution 1 — same as JAX's predict_sequence per-track output.
    assert pt.shape == jx.shape, f"shape mismatch: pt={pt.shape} jax={jx.shape}"

    # Per-track absolute and relative tolerances. Upstream claims numerical
    # equivalence; we pin a generous bound so weight conversions or attn
    # numerics changes get flagged but minor implementation drift doesn't.
    abs_diff = np.abs(pt - jx).max()
    rel_diff = np.abs(pt - jx).mean() / (np.abs(jx).mean() + 1e-9)
    assert abs_diff < 0.1, f"max abs diff {abs_diff:.4f} exceeds 0.1"
    assert rel_diff < 0.05, f"mean rel diff {rel_diff:.4f} exceeds 5%"
