"""Benchmark AlphaGenome JAX vs PyTorch backends across devices.

Measures forward-pass latency on a single 1 MB DNA window for each
available (backend, device) pair on the current machine.

Run from the repo root:

    python scripts/benchmark_alphagenome_backends.py

Or restrict to one backend / one length:

    python scripts/benchmark_alphagenome_backends.py --backends pt --lengths 524288

Output: a markdown table written to stdout and (optionally) to a file
via ``--output``. The audit report at
``audits/2026-04-29_alphagenome_pytorch_spike/report.md`` consumes this.

Requires the matching conda envs (``chorus-alphagenome``,
``chorus-alphagenome_pt``) — runs each backend's load+forward inside its
own subprocess via ``mamba run`` so JAX and torch never coexist in one
interpreter.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

DEFAULT_LENGTHS = [32768, 131072, 524288, 1048576]
DEFAULT_BACKENDS = ["jax", "pt"]


PT_INNER = textwrap.dedent(
    r"""
    import os, sys, time, json, numpy as np
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args = json.loads(sys.argv[1])
    import torch, huggingface_hub
    from alphagenome_pytorch import AlphaGenome

    device_name = args["device"]
    device = torch.device(device_name)

    weights = huggingface_hub.hf_hub_download(
        "gtca/alphagenome_pytorch", "model_all_folds.safetensors"
    )
    model = AlphaGenome.from_pretrained(weights, device=device)
    model.eval()

    rng = np.random.default_rng(0)
    n = args["length"]
    seq = np.zeros((n, 4), dtype=np.float32)
    seq[np.arange(n), rng.integers(0, 4, n)] = 1.0
    x = torch.from_numpy(seq).unsqueeze(0).to(device)
    org = torch.tensor([0], dtype=torch.long, device=device)

    def sync():
        if device_name == "mps": torch.mps.synchronize()
        elif device_name.startswith("cuda"): torch.cuda.synchronize()

    # 2 warm + 3 timed
    for _ in range(2):
        with torch.no_grad():
            _ = model(x, organism_index=org, heads=("atac","dnase"))
        sync()
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x, organism_index=org, heads=("atac","dnase"))
        sync()
        times.append(time.perf_counter() - t0)

    print(json.dumps({"backend":"pt","device":device_name,"length":n,"times":times}))
    """
)


JAX_INNER = textwrap.dedent(
    r"""
    import os, sys, time, json, numpy as np, platform
    args = json.loads(sys.argv[1])
    if platform.system() == "Darwin":
        os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    from alphagenome_research.model.dna_model import create_from_huggingface
    from alphagenome.models.dna_output import OutputType

    jax_device = jax.devices("gpu")[0] if "gpu" in {d.platform for d in jax.devices()} else jax.devices("cpu")[0]
    model = create_from_huggingface("all_folds", device=jax_device)

    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(["A","C","G","T"], args["length"]).tolist())

    # 2 warm + 3 timed
    for _ in range(2):
        _ = model.predict_sequence(seq, requested_outputs=[OutputType.ATAC, OutputType.DNASE], ontology_terms=None)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = model.predict_sequence(seq, requested_outputs=[OutputType.ATAC, OutputType.DNASE], ontology_terms=None)
        times.append(time.perf_counter() - t0)

    print(json.dumps({"backend":"jax","device":str(jax_device),"length":args["length"],"times":times}))
    """
)


def run_inner(env: str, code: str, args: dict, timeout: int = 1800) -> dict | None:
    """Run a subprocess and return the parsed JSON result, or None on failure."""
    cmd = ["mamba", "run", "-n", env, "python", "-c", code, json.dumps(args)]
    try:
        out = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"[{env}] FAIL ({e.returncode}):\n{e.stderr}\n")
        return None
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"[{env}] TIMEOUT after {timeout}s\n")
        return None
    # Last JSON line is the result; everything else is logs
    for line in reversed(out.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    sys.stderr.write(f"[{env}] no JSON in output:\n{out.stdout}\n")
    return None


def detect_devices() -> dict[str, list[str]]:
    """Detect which (backend, device) combos are runnable on this host."""
    available = {"pt": [], "jax": []}

    # PyTorch
    pt_check = subprocess.run(
        [
            "mamba", "run", "-n", "chorus-alphagenome_pt", "python", "-c",
            "import torch,os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE'); "
            "print('cpu', int(torch.backends.mps.is_available()), int(torch.cuda.is_available()))",
        ],
        capture_output=True, text=True,
    )
    if pt_check.returncode == 0:
        parts = pt_check.stdout.strip().split()
        available["pt"].append("cpu")
        if len(parts) >= 2 and parts[1] == "1":
            available["pt"].append("mps")
        if len(parts) >= 3 and parts[2] == "1":
            available["pt"].append("cuda")

    # JAX (always CPU on macOS in our pinning; CUDA on Linux if jax[cuda] installed)
    jax_check = subprocess.run(
        [
            "mamba", "run", "-n", "chorus-alphagenome", "python", "-c",
            "import os,platform; "
            "os.environ.setdefault('JAX_PLATFORMS','cpu') if platform.system()=='Darwin' else None; "
            "import jax; "
            "print(','.join(sorted({d.platform for d in jax.devices()})))",
        ],
        capture_output=True, text=True,
    )
    if jax_check.returncode == 0:
        for plat in jax_check.stdout.strip().split(","):
            available["jax"].append(plat)

    return available


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    ap.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    if not shutil.which("mamba"):
        sys.stderr.write("mamba not found on PATH — required to dispatch envs\n")
        return 2

    available = detect_devices()
    rows = []
    for backend in args.backends:
        for device in available.get(backend, []):
            for length in args.lengths:
                code = PT_INNER if backend == "pt" else JAX_INNER
                env_name = "chorus-alphagenome_pt" if backend == "pt" else "chorus-alphagenome"
                inner_args = {"device": device, "length": length}
                print(f"[{backend}/{device}] L={length} ...", flush=True, file=sys.stderr)
                r = run_inner(env_name, code, inner_args, timeout=args.timeout)
                if r is None:
                    rows.append({"backend": backend, "device": device, "length": length,
                                 "mean_s": None, "min_s": None, "ok": False})
                else:
                    times = r["times"]
                    rows.append({
                        "backend": backend, "device": r.get("device", device),
                        "length": length, "mean_s": sum(times)/len(times),
                        "min_s": min(times), "ok": True,
                    })

    # Render markdown table
    md = ["| backend | device | length | mean (s) | min (s) |",
          "|---|---|---:|---:|---:|"]
    for r in rows:
        if r["ok"]:
            md.append(
                f"| {r['backend']} | {r['device']} | {r['length']:,} | "
                f"{r['mean_s']:.2f} | {r['min_s']:.2f} |"
            )
        else:
            md.append(f"| {r['backend']} | {r['device']} | {r['length']:,} | FAIL | FAIL |")
    table = "\n".join(md)
    print(table)
    if args.output:
        args.output.write_text(table + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
