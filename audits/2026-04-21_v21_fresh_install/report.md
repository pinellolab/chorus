# v21 fresh-install audit — 2026-04-21

Walked `audits/AUDIT_CHECKLIST.md` top-to-bottom with **every
re-downloadable cache purged** before starting. Scope deliberately
excludes the 80 GB / 2–4 h oracle-env recreation (destructive to
ongoing work; belongs on a release-host audit) — data caches + fresh
HF pulls + full checklist exercise are what's covered.

## What was nuked before the audit

| Cache | Before | After nuke |
|---|---|---|
| `~/.chorus/backgrounds/` | 1.5 GB (6 NPZ files) | empty |
| `genomes/hg38.fa` + `.fai` | 3.0 GB | gone |

## What fresh-downloaded during the audit

| Asset | Size | Time | Source |
|---|---|---|---|
| hg38 reference genome | 3.1 GB | ~9 min | documented `chorus genome download hg38` flow → UCSC + decompress + `samtools faidx` index |
| 6 per-track CDFs | 1.5 GB total | **44 s total** across all 6 oracles | `huggingface.co/datasets/lucapinello/chorus-backgrounds` |

## Results by checklist section

| § | Topic | Result |
|---|---|---|
| 1 | Install & env | **CLI PASS** — `chorus --help` surfaces all 6 subcommands (setup, list, validate, remove, health, genome). Conda-env recreation (80 GB / 2-4 h) **deliberately deferred** to release-host audit. |
| 3 | GPU / device | **PASS** — all 6 oracle envs detect Metal/MPS/JAX-METAL on macOS arm64. See `logs/05_device_probe.txt`. |
| 4 | CDF fresh download | **PASS** — 6/6 monotonic `effect_cdfs`, `p50≤p95≤p99` on `summary_cdfs`, signed% matches semantics (0/20/0/100/100/13). See `logs/04_cdf_fresh_download.txt`. |
| 5 | Python API | **PASS** — `sequence_length` matches spec for all 6; invalid-name error names the valid options; §14.4 chrom validation fires on bad input. See `logs/06_api_sanity.txt`. |
| 6 | Notebook fresh exec | **PASS** — `single_oracle_quickstart.ipynb` re-executed: 49 cells, **0 errors**, **0 warnings**. Fresh notebook saved at `logs/09_quickstart_fresh.ipynb`. Other two notebooks require all 6 oracle envs + GPU; deferred to release host. |
| 7 | HTML reports (selenium) | **PASS** — **18/18** shipped HTMLs rendered with fresh Chrome profile, **0 JS errors** each. Screenshots in `screenshots/`. |
| 10 | Repo consistency | **PASS** — no drift: `grep '5,930\|7,612\|196 kbp\|examples/applications/'` in live code → 0 matches. See `logs/07_consistency.txt`. |
| 11 | Test suite | **PASS** — **335 passed / 1 skipped** (8m 28s). See `logs/10_pytest.txt`. |
| 14.4 | chrZZ validation | **PASS** — `GenomeRef.slop` + `extract_sequence` both raise actionable exceptions naming the bad chromosome. Regression test `test_bad_chromosome_gives_actionable_error` is in the 335 pass count. |
| 15 | Offline | **PASS** — 0 runtime CDN fetches (`<script src="http…">` / `<link href="http…">` empty across all 18 HTMLs). |
| 16 | Logging hygiene | **PASS** — no committed HF tokens or AWS keys. |
| 18 | License | **PASS** — `LICENSE` (MIT, Pinello Lab), `docs/THIRD_PARTY.md` (v19 attribution for 6 oracles + IGV.js), bundled IGV vendor license header intact. |

## Deferred to release-host audit

- **§1 full conda env recreate** — deleting + rebuilding 7 envs is 80 GB / 2–4 h and would block the user's ongoing work. Release-host job, not a routine audit.
- **§2 HF-gate end-to-end** — would need to unset `HF_TOKEN` and try AlphaGenome load; too invasive for a routine audit. Code paths already verified in v18.
- **§6 multi-oracle notebook re-execute** — `comprehensive_oracle_showcase.ipynb` + `advanced_multi_oracle_analysis.ipynb` need all 6 oracles loaded + GPU. Release-host runners.
- **§8 MCP E2E** — spawn `chorus-mcp` over stdio + `fastmcp.Client.call_tool`. Needs ~4 min AlphaGenome predict. Integration-marked test.
- **§13 real-oracle determinism** — same-input-twice check on each of 6 oracles. Needs loaded models; ~30 min. Release-host.
- **§14 remaining edge cases** (indels, chrM, telomere near-edge, multi-allelic) — need loaded oracles. §14.4 chrZZ already fixed + regression-tested here.
- **§17 pip-audit** — step already in `.github/workflows/tests.yml` as advisory (v20). Runs in CI, not locally here.

## Headline result

**No new findings.** Every section that could run on this macOS arm64
host with only data-cache purging passed. The previous audits (v15-v20)
+ fixes in PRs #32, #34, #36 have driven this repo to a ship-clean
state on mechanisable items. What remains open is release-host work
(full env build, real-oracle determinism, multi-oracle notebooks, MCP
E2E) — items explicitly scoped for that environment.

## Artefacts in this folder

- `report.md` — this summary
- `screenshots/*.png` (16) — fresh selenium-rendered reports at 1600×4500
- `logs/00_pre_nuke.txt` — state before cache purge
- `logs/01_post_nuke.txt` — empty caches confirmed
- `logs/02_cli.txt` — `chorus --help` output
- `logs/03_genome_download.txt` — hg38 fresh download trace
- `logs/04_cdf_fresh_download.txt` — 6-oracle CDF fresh pull + sanity
- `logs/05_device_probe.txt` — per-env accelerator detection
- `logs/06_api_sanity.txt` — sequence_length + invalid-oracle + chrZZ
- `logs/07_consistency.txt` — §10 + §15 + §16 greps (all empty = pass)
- `logs/08_selenium.txt` — 18 HTML renders, 0 JS errors
- `logs/09_quickstart_fresh.ipynb` — re-executed notebook
- `logs/10_pytest.txt` — fast suite output
