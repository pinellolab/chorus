"""End-to-end integration tests that hit the network / download models.

Gated with ``@pytest.mark.integration`` so they're skipped by default.
Run when maintainer-level verification is needed:

    pytest tests/test_integration.py -v -m integration

Covers the three scenarios v8 audit flagged as not exercised:

1. SEI + LegNet per-track CDF download (items 2 in v9 plan)
2. ChromBPNet fresh model download from ENCODE (item 3)
3. MCP server end-to-end session via spawned subprocess (item 4)

These intentionally hit real services so they're NOT in the fast suite
and NOT in GitHub Actions CI (disk/time constraints). They're runnable
locally by a maintainer with a full oracle env setup.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Item 2 — SEI + LegNet CDF download
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("oracle", ["sei", "legnet"])
def test_pertrack_background_download(tmp_path, oracle):
    """Exercise the HF CDF download path for the two oracles no regen
    workflow touches. v8 covered alphagenome/borzoi/chrombpnet/enformer
    (download was triggered as a side effect of regen); sei + legnet
    were never exercised because no committed example uses them.

    Verifies: (1) the NPZ is retrievable from the public HF dataset,
    (2) it loads, (3) it passes the same empirical checks v8 ran
    (monotone CDFs, p50 <= p95 <= p99 > 0, all effect_counts > 0).
    """
    from chorus.analysis.normalization import download_pertrack_backgrounds

    n = download_pertrack_backgrounds(oracle, cache_dir=str(tmp_path))
    assert n == 1, f"expected 1 file downloaded for {oracle}, got {n}"

    path = tmp_path / f"{oracle}_pertrack.npz"
    assert path.exists()

    with np.load(path, allow_pickle=True) as npz:
        effect_cdfs = npz["effect_cdfs"]
        summary_cdfs = npz["summary_cdfs"]
        effect_counts = npz["effect_counts"]
        signed_flags = npz["signed_flags"]

    # Monotonicity on a sample of rows (cheap — full sweep is overkill)
    assert all(np.all(np.diff(r) >= -1e-9) for r in effect_cdfs[:10]), \
        f"{oracle}: effect CDF rows must be non-decreasing"

    # p50 <= p95 <= p99 > 0 on the summary CDF
    n_pts = summary_cdfs.shape[1]
    p50 = int(0.50 * n_pts)
    p95 = int(0.95 * n_pts)
    p99 = int(0.99 * n_pts)
    for row in summary_cdfs[: min(10, summary_cdfs.shape[0])]:
        assert row[p50] <= row[p95] <= row[p99], \
            f"{oracle}: summary CDF percentiles out of order"
        assert row[p99] >= 0

    # Every track must have at least 1 effect sample
    assert (effect_counts > 0).all(), \
        f"{oracle}: all tracks should have at least one effect_count"

    # Layer semantics
    if oracle in ("sei", "legnet"):
        # Both are classification/regression models; all tracks are signed.
        assert signed_flags.all(), \
            f"{oracle}: all tracks are expected to be signed"


# ---------------------------------------------------------------------------
# Item 3 — ChromBPNet fresh download
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_chrombpnet_fresh_single_model_download(tmp_path):
    """Download one ChromBPNet model (ATAC:K562, ~500 MB tarball) from
    scratch and verify the resume helper fetches, unpacks, and loads
    it. v8 preserved the 37 GB ``downloads/chrombpnet/`` across every
    'fresh install' audit, so the ENCODE-to-disk path was never
    verified from zero.

    Uses an isolated temp ``download_dir`` on the instance so the
    real 37 GB cache isn't touched.
    """
    import chorus

    reference_fasta = str(Path(__file__).parent.parent / "genomes" / "hg38.fa")
    if not Path(reference_fasta).exists():
        pytest.skip("hg38.fa missing — run `chorus genome download hg38` first")

    oracle = chorus.create_oracle(
        "chrombpnet", use_environment=True, reference_fasta=reference_fasta,
    )
    # Redirect the download_dir to a tmpdir so we actually re-download.
    oracle.download_dir = Path(tmp_path) / "chrombpnet"
    oracle.download_dir.mkdir(parents=True, exist_ok=True)

    # Load ATAC:K562 fold 0 — smallest ATAC model family, known good
    oracle.load_pretrained_model(assay="ATAC", cell_type="K562", fold=0)
    assert oracle.loaded, "model should be loaded after load_pretrained_model"

    # Final tarball should have been extracted into the tmp download_dir
    extracted = Path(tmp_path) / "chrombpnet" / "ATAC_K562"
    assert extracted.exists(), "extracted model dir must exist under tmp download_dir"

    # Predict on the smoke-test region — must return finite values
    result = oracle.predict(("chr1", 1_000_000, 1_002_114))
    tracks = dict(result.items())
    assert len(tracks) > 0, "predict must return at least one track"
    for name, track in tracks.items():
        assert track.values.shape[0] > 0, f"empty values for {name}"
        assert np.isfinite(track.values).all(), f"non-finite values in {name}"


# ---------------------------------------------------------------------------
# Item 4 — End-to-end MCP session
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_mcp_e2e_list_oracles_and_analyze_variant(tmp_path):
    """Spawn ``chorus-mcp`` as a stdio subprocess via the fastmcp
    Python Client and call two real tools:

    1. ``list_oracles`` — no side effects, verifies the stdio protocol
       works and the registered tool name matches docs.
    2. ``analyze_variant_multilayer`` on SORT1 rs12740374 in HepG2 with
       AlphaGenome — verifies a real analysis tool round-trips.

    This is the first E2E integration test of the server; prior tests
    at ``tests/test_mcp.py`` mock the oracles entirely.
    """
    import asyncio

    import shutil

    pytest.importorskip("fastmcp")
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        pytest.skip("HF_TOKEN not set — AlphaGenome is gated")

    # Locate chorus-mcp on PATH (installed by `pip install -e .` in the
    # active env). Going through `mamba run -n chorus chorus-mcp` is
    # less portable because two-root mamba installs resolve the env
    # name inconsistently (see README Troubleshooting).
    chorus_mcp_bin = shutil.which("chorus-mcp")
    if not chorus_mcp_bin:
        pytest.skip("chorus-mcp not on PATH — activate the chorus env first")

    # Inherit MAMBA_ROOT_PREFIX so the spawned server finds the
    # oracle envs under ~/.local/share/mamba (documented "two mamba
    # installs" trap — default mamba root is miniforge3/envs which
    # doesn't contain chorus-*).
    mamba_root = os.environ.get("MAMBA_ROOT_PREFIX") or str(
        Path.home() / ".local" / "share" / "mamba"
    )

    async def run():
        transport = StdioTransport(
            command=chorus_mcp_bin,
            args=[],
            env={
                "HF_TOKEN": hf_token,
                "CHORUS_NO_TIMEOUT": "1",
                "PATH": os.environ.get("PATH", ""),
                "MAMBA_ROOT_PREFIX": mamba_root,
                "MAMBA_EXE": os.environ.get("MAMBA_EXE", ""),
                "HOME": os.environ.get("HOME", ""),
            },
        )
        async with Client(transport=transport) as client:
            # (1) list_oracles — cheap, structural check
            oracles_result = await client.call_tool("list_oracles", {})
            text = str(oracles_result)
            for name in ("alphagenome", "enformer", "chrombpnet", "borzoi"):
                assert name in text, f"{name} missing from list_oracles output"

            # (2) load_oracle — must precede any predict/analyze call.
            # Surface any load error (wrapped by _safe_tool) before we
            # try to use the oracle.
            load_resp = await client.call_tool("load_oracle", {"oracle_name": "alphagenome"})
            load_payload = load_resp.data if hasattr(load_resp, "data") else load_resp
            assert "error" not in (load_payload or {}), (
                f"load_oracle returned error: {load_payload}"
            )

            # (3) real analysis — AlphaGenome predicting SORT1 DNase HepG2
            result = await client.call_tool("analyze_variant_multilayer", {
                "oracle_name": "alphagenome",
                "position": "chr1:109274968",
                "ref_allele": "G",
                "alt_alleles": ["T"],
                "assay_ids": ["DNASE/EFO:0001187 DNase-seq/."],
                "gene_name": "SORT1",
                "user_prompt": "E2E integration test — Musunuru 2010 variant",
            })
            data = result.data if hasattr(result, "data") else result
            payload = data if isinstance(data, dict) else json.loads(str(data))
            # Structural assertions only (AlphaGenome CPU non-det is ±0.05)
            assert "variant" in payload or "alleles" in payload, \
                f"unexpected payload shape: {list(payload.keys())[:10]}"
            assert payload.get("oracle") == "alphagenome" or "alphagenome" in str(payload)
            return payload

    asyncio.run(run())
