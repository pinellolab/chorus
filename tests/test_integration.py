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
    scratch via the ENCODE tarball fallback path and verify the resume
    helper fetches, unpacks, and loads it. v8 preserved the 37 GB
    ``downloads/chrombpnet/`` across every 'fresh install' audit, so
    the ENCODE-to-disk path was never verified from zero.

    Uses an isolated temp ``download_dir`` on the instance so the
    real cache isn't touched.

    NB: chorus ≥ 0.3 routes the default fold-0 ``chrombpnet_nobias``
    request through the HuggingFace slim mirror and never touches the
    ENCODE tarball path. To keep this test specifically exercising the
    tarball fallback (the intent of the original v9 test), we pass
    ``model_type='chrombpnet'`` (bias-aware variant — only available
    via the full ENCODE tarball).
    """
    import chorus

    reference_fasta = str(Path(__file__).parent.parent / "genomes" / "hg38.fa")
    if not Path(reference_fasta).exists():
        pytest.skip("hg38.fa missing — run `chorus genome download hg38` first")

    from chorus.core.environment.manager import EnvironmentManager
    if not EnvironmentManager().environment_exists("chrombpnet"):
        pytest.skip(
            "chorus-chrombpnet env missing — run `chorus setup --oracle chrombpnet` first. "
            "Without it, the subprocess oracle runner falls back to direct load which needs "
            "TensorFlow in the base env (not installed by default)."
        )

    # This test exists to exercise the ENCODE tarball path specifically, so it is
    # the one test in the suite that hard-depends on encodeproject.org. When that
    # portal is down the failure surfaces 60 s later as a bare
    # "TimeoutError: The read operation timed out" from urllib, which reads like a
    # broken download helper -- and a release gate that goes red because a third
    # party is offline is a gate people learn to wave through.
    #
    # So separate the conditions the way the two skips above already do: an
    # unreachable portal is a skip that names the portal; a reachable portal plus
    # a failed download is a real failure. Observed 2026-08-08, ENCODE returning
    # HTTP 000 on a 25 s budget while huggingface.co answered in 33 ms.
    import urllib.error
    import urllib.request
    try:
        urllib.request.urlopen("https://www.encodeproject.org/", timeout=20).close()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        pytest.skip(
            f"encodeproject.org unreachable ({type(exc).__name__}: {exc}) — this test "
            f"deliberately downloads a ~500 MB ENCODE tarball to cover the fallback "
            f"path the HF slim mirror bypasses, so it cannot run offline. The download "
            f"helper itself is covered by tests/test_error_recovery.py."
        )

    oracle = chorus.create_oracle(
        "chrombpnet", use_environment=True, reference_fasta=reference_fasta,
    )
    # Redirect the download_dir to a tmpdir so we actually re-download.
    oracle.download_dir = Path(tmp_path) / "chrombpnet"
    oracle.download_dir.mkdir(parents=True, exist_ok=True)

    # Load ATAC:K562 fold 0 with the bias-aware variant to force the
    # ENCODE tarball path (slim HF mirror only ships the nobias variant).
    oracle.load_pretrained_model(
        assay="ATAC", cell_type="K562", fold=0, model_type="chrombpnet",
    )
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

    # Ask huggingface_hub, not just the environment. `get_token()` is the read-only equivalent of
    # what the AlphaGenome load paths themselves do: HF_TOKEN -> HUGGING_FACE_HUB_TOKEN -> the
    # credential file written by `huggingface-cli login`. Gating on the env vars alone was strictly
    # narrower than the runtime it guards -- both alphagenome.py and its load template try
    # `huggingface_hub.whoami()` FIRST and only fall back to the env -- so this test skipped on any
    # host authenticated the normal way, and the MCP path went unverified for months.
    #
    # Deliberately not `chorus.cli._tokens.resolve_hf_token`: that calls `huggingface_hub.login()`,
    # which writes the token to disk, and mutates os.environ on failure. A test must not do either.
    import huggingface_hub

    hf_token = huggingface_hub.get_token()
    if not hf_token:
        pytest.skip(
            "no HuggingFace token (set HF_TOKEN, HUGGING_FACE_HUB_TOKEN, or run "
            "`huggingface-cli login`) — AlphaGenome is gated"
        )

    # Locate chorus-mcp on PATH (installed by `pip install -e .` in the
    # active env). Going through `mamba run -n chorus chorus-mcp` is
    # less portable because two-root mamba installs resolve the env
    # name inconsistently (see README Troubleshooting).
    chorus_mcp_bin = shutil.which("chorus-mcp")
    if not chorus_mcp_bin:
        pytest.skip("chorus-mcp not on PATH — activate the chorus env first")

    # MAMBA_ROOT_PREFIX tells the spawned server which mamba root holds the oracle envs. It used to
    # fall back to `~/.local/share/mamba` when the variable was unset — the "two mamba installs"
    # trap this test was written on. On a host where the envs live under `miniforge3/envs` that
    # fallback is actively wrong: the child inherits a root with no `chorus-*` in it, `mamba env
    # list --json` returns envs that do not include the oracle, and the failure surfaces as
    # "Python executable not found in environment" — which sends you looking at the env's contents
    # rather than at the root you pointed mamba at. Masked for months by the HF skip above.
    #
    # Derive it from the mamba/conda binary chorus itself resolved (root = <prefix>/bin/mamba's
    # grandparent), and if that cannot be determined, pass nothing and let mamba use its own
    # default — which is correct far more often than a guess.
    mamba_root = os.environ.get("MAMBA_ROOT_PREFIX")
    if not mamba_root:
        try:
            from chorus.core.environment import EnvironmentManager

            conda_exe = Path(EnvironmentManager().conda_exe).resolve()
            if conda_exe.parent.name == "bin":
                mamba_root = str(conda_exe.parent.parent)
        except Exception:
            mamba_root = None

    async def run():
        transport = StdioTransport(
            command=chorus_mcp_bin,
            args=[],
            # A whitelist, so anything the child needs must be named explicitly. HF_TOKEN is
            # passed because the token may have come from the credential file rather than the
            # environment; the HF_* vars below are forwarded when set so a non-default credential
            # or cache location still resolves in the child (HOME alone only covers the default).
            env={
                "HF_TOKEN": hf_token,
                "CHORUS_NO_TIMEOUT": "1",
                "PATH": os.environ.get("PATH", ""),
                # Omitted entirely when it could not be derived, rather than guessed at.
                **({"MAMBA_ROOT_PREFIX": mamba_root} if mamba_root else {}),
                "MAMBA_EXE": os.environ.get("MAMBA_EXE", ""),
                "HOME": os.environ.get("HOME", ""),
                **{k: os.environ[k] for k in
                   ("HF_HOME", "HF_TOKEN_PATH", "HF_HUB_CACHE", "XDG_CACHE_HOME")
                   if k in os.environ},
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
