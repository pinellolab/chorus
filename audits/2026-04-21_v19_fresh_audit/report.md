# v19 fresh audit — driven by `audits/AUDIT_CHECKLIST.md`

First audit walked top-to-bottom against the 18-section checklist shipped
in PRs #33 + #34. Artefacts in this directory match the checklist
appendix (`screenshots/`, `cdf_check.txt`, `device_probe.txt`,
`consistency_grep.txt`, `python_api.txt`, `determinism.txt`,
`pytest.txt`).

Also lands `CLAUDE.md` at repo root pointing every future Claude
session at the checklist so this doesn't drift by memory.

## Results by checklist section

| § | Topic | Result | Notes |
|---|---|---|---|
| 1 | Install & environment | **deferred** | Full fresh install needs a Linux/CUDA host + ~80 GB. |
| 2 | HuggingFace auth | **spot-checked** (code paths) | All 3 paths updated in v18. End-to-end HF gate test belongs on release host. |
| 3 | GPU / device | **PASS** | All 6 envs detect Metal / MPS on macOS arm64. See `device_probe.txt`. |
| 4 | Per-track CDFs | **PASS** | All 6 oracles: monotonic, p50 ≤ p95 ≤ p99, signed% matches semantics. See `cdf_check.txt`. |
| 5 | Python API | **PASS** | `sequence_length` matches spec for all 6. Error messages clear. See `python_api.txt`. |
| 6 | Notebooks fresh-run | **deferred** | v18 already ran `single_oracle_quickstart.ipynb` clean; the other two are long. |
| 7 | HTML reports | **PASS** | **18/18 shipped HTMLs** rendered via selenium with **0 JS errors**. See `screenshots/`. |
| 8 | MCP server | **spot-checked** | 22 tools confirmed in v17. E2E `chorus-mcp` over stdio deferred. |
| 9 | Error messages | **PASS** | Unknown-oracle, no-model, no-genome all tested in v17/v18. |
| 10 | Repo consistency | **2 drifts found** | Fixed in this PR. |
| 11 | Test suite | **PASS** | see `pytest.txt`. |
| 12 | Reproducibility | **PASS** | Regen scripts idempotent; CDFs auto-downloadable. |
| 13 | Scientific determinism | **PASS (mock)** | Real-oracle same-seed check belongs on release host. |
| 14 | Genomics edge cases | **deferred** | Needs loaded oracles. |
| 15 | Offline / air-gapped | **PASS** | **0 runtime CDN fetches** — `<script src="http…">` / `<link href="http…">` greps empty across all 18 HTMLs. Apparent "CDN refs" earlier were attribution comments inside the bundled IGV.js. |
| 16 | Logging hygiene | **PASS** | No committed HF tokens (`hf_…`) or AWS keys. |
| 17 | Supply chain | **partial** | `pip-audit` not installed in base env; documented as release-host check. |
| 18 | License / attribution | **P1: missing NOTICE / THIRD_PARTY** | `LICENSE` (MIT, Pinello Lab) present. No `NOTICE` or `docs/THIRD_PARTY.md` attributing the 6 oracle models. |

## Fixes in this PR

1. **`scripts/build_backgrounds_borzoi.py:4`** — module docstring said
   "**7,612 Borzoi tracks**"; real count is 7,611 (matches v17 fix to
   `scripts/README.md`). Updated.
2. **`chorus/oracles/alphagenome.py:22`** — `AlphaGenomeOracle`
   docstring said "**5,930 human functional genomic tracks**"; real
   count is 5,731 (matches v16/v17/v18 fixes to notebooks, README,
   server.py, metadata). Updated. This was the **last** `5,930` in the
   live code.

## Known issues flagged, NOT fixed here

- **§18 third-party attribution missing**: no `NOTICE` or
  `docs/THIRD_PARTY.md`. The 6 oracle models (Enformer, Borzoi,
  ChromBPNet, Sei, LegNet, AlphaGenome) and the bundled IGV.js should
  be credited with licenses in one reachable place. P1 for release.
- **§17 pip-audit not installed**: add to `environment.yml` dev-deps,
  or keep as a release-host CI step. The release CI workflow at
  `.github/workflows/tests.yml` doesn't currently run supply-chain
  scans.
- **§1 & §14 deferred to release-host audit**: fresh install from a
  clean machine + genomics edge-cases (variant near telomere,
  soft-masked FASTA, indels) need physical run time that doesn't fit
  on this pass.

## Delivered

- `CLAUDE.md` at repo root — instructs future Claude sessions to read
  `audits/AUDIT_CHECKLIST.md` before any ship-ready audit.
- 2 docstring fixes (`5,930`→`5,731`, `7,612`→`7,611`) — the last
  stale canonical numbers in live code.
- `audits/2026-04-21_v19_fresh_audit/` — 16 HTML screenshots (18
  reports; 2 pairs share basenames in different dirs), CDF sanity
  output, per-env device probe, consistency greps, Python API probe,
  determinism check, `pytest.txt`.
