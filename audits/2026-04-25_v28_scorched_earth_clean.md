# Chorus v28 Scorched-Earth Audit — Clean Run

**Date**: 2026-04-25 (audit ran 23:30 → 01:35 across midnight)
**Platform**: macOS 15.7.4 / Apple M3 Ultra / 96 GB
**Branch**: `main` @ `200befd` (post v27 follow-ups)
**Auditor**: Claude Opus 4.7 (1M context)
**Outcome**: 0 findings — first scorched-earth audit in this run that
returned no new bugs.

## Why this audit

User requested a second consecutive scorched-earth pass to verify the
v27 chain of fixes (pip shadowing, FANTOM track validator, chorus-sei
solver explosion, ChromBPNet HepG2 prefetch) all hold up on a real
greenfield install. v28 is the regression check on the regression
fixes.

## What was scorched

| Item                                | Size before |
| ----------------------------------- | ----------- |
| `~/.local/share/mamba/envs/chorus*` (7 envs) | ~10.4 GB |
| `~/.chorus/`                        | 1.5 GB |
| `genomes/hg38.fa*`                  | 3.1 GB |
| `downloads/`                        | 10 GB |
| **Total**                           | **~25 GB** |

Verified empty afterwards.

## What was reinstalled (chronological)

Following README TLDR Step-by-Step exactly as a new user, *with the
v27 doc fix*:

1. **Step 1 — base env** (1.5 min total)
   ```bash
   mamba env create -f environment.yml      # ~1 min
   mamba activate chorus
   python -m pip install -e .               # ~30 s — note `python -m pip` not `pip`
   ```

   The v27 P0 #1 fix (`python -m pip install -e .` instead of bare
   `pip`) is now the documented form. Verified the README + CONTRIBUTING
   snippets match.

2. **Step 2 — `chorus setup --oracle all`** (60 min wall-clock)

   | Oracle      | env  | weights | total |
   | ----------- | ---- | ------- | ----- |
   | alphagenome | 0:36 | 1:31    | 12:13 (incl. hg38 9:48) |
   | borzoi      | 0:37 | 0:24    | 1:04 |
   | chrombpnet  | 0:31 | 8:33    | 8:51 (K562 only at first) |
   | enformer    | 0:34 | 0:25    | 1:13 |
   | legnet      | 0:53 | 0:13    | 1:14 |
   | sei         | 0:35 | 35:21   | 35:21 (3 GB Zenodo) |
   | **Total**   |      |         | **~60 min** |

   Sei env created cleanly with the v27 yml fix (PyTorch >=2.0.0,
   no cudatoolkit pin). Solver took <1 minute on macOS — Linux
   solver-explosion bug doesn't affect this platform but the fix is
   cross-compatible.

3. **Step 2.5 — re-run `chorus setup --oracle chrombpnet`** (8 min)

   The v27 P1 fix landed mid-audit (`9100964`). Re-running chrombpnet
   setup picked up the new prefetch logic that pulls **both** K562 and
   HepG2 DNase tarballs:

   ```
   downloads/chrombpnet/
   ├── DNASE_HepG2/      <- new (00:38, post-pull)
   ├── DNASE_K562/       <- v28 setup --oracle all (23:51, pre-pull)
   └── .chorus_setup_v1
   ```

   `_DEFAULT_LOAD_KWARGS["chrombpnet"]` is now a list of dicts; the
   prefetch script loops over them and triggers one ENCODE pull per
   cell type. Future runs of `chorus setup` (cold cache) will get
   both in one pass.

4. **Step 3 — README Python snippet** (12 s) — clean:
   - WT mean signal: `0.468`
   - 3 alt alleles scored, no ref-mismatch warning

5. **Step 4 — MCP** — verified via integration test (see below).

## Verification matrix

| Check | Result | Time |
|---|---|---|
| `chorus list` | 6/6 ✓ Installed, exit 0 | <1 s |
| `chorus health` | 6/6 ✓ Healthy, exit 0 | 43 s |
| `chorus --version` | `chorus 0.1.0` (new v27 P2 flag) | <1 s |
| README Step 3 snippet | WT 0.468, 3 alts | 12 s |
| `single_oracle_quickstart.ipynb` | 0 errors, 0 warnings | ~3 min |
| `advanced_multi_oracle_analysis.ipynb` | 0 errors, 0 warnings | ~10 min |
| `comprehensive_oracle_showcase.ipynb` | 0 errors, 0 warnings | ~12 min |
| MCP E2E (`-m integration`) | 1 passed | 280 s |
| SEI + LegNet CDF download (`-m integration`) | 2 passed | 2.7 s |
| Multi-oracle predict (Enformer + ChromBPNet HepG2) | both finite (means: 0.294 / 0.242) | 60 s — **HepG2 prefetched, no on-demand download** |
| `discover_variant_effects` (SORT1 rs12740374) | 4 layer rankings | 5 min |
| Fast pytest (`-m "not integration"`) | 340 passed / 1 skipped | 34 min (ran in parallel with smoke; usually ~9 min standalone) |

## Findings

**None.** No P0, no P1, no P2.

This is the first audit in the v22→v28 series to return a clean bill.
Notable things that *would* have been findings if the prior agents
hadn't already fixed them:

- ChromBPNet HepG2 missing from prefetch — fixed in `9100964`. v28
  confirms `downloads/chrombpnet/DNASE_HepG2/` is created during
  setup and the comprehensive notebook (which loads HepG2) no longer
  blocks on a 722 MB tarball mid-run.
- FANTOM CAGE identifiers (`CNhs11250`) rejected by validator — fixed
  in `2ecd998` + `5acee71`. v28 confirms the quickstart notebook
  re-executes clean and the regression test in
  `tests/test_prediction_methods.py` passes.
- `chorus-sei` solver explosion — fixed in `68f5cc3`. v28 confirms
  the macOS sei env build is fast (<1 min solver) with the new yml.
- Dead `#mcp-server-ai-assistant-integration` anchor and 20-vs-22
  tool list in MCP `--help` — fixed in `2ecd998`. v28 confirms both.

## What v28 still does NOT cover

- **Linux x86_64 / CUDA** — covered by the parallel agent's v27 audit
  on a HPC host (commits `5acee71`, `68f5cc3`, `9100964`, `200befd`).
- **HTML walkthrough render** — covered by `200befd` (playwright /
  Chromium render of all 18 walkthroughs).
- **Long-running batch / clinical workloads** — out of scope.
- **First-time interactive HF token prompt path** — v28 used
  `HF_TOKEN` env var so the interactive prompt at
  `chorus/cli/_tokens.py:127` (getpass) was not exercised.

## Time budget (v28)

```
Phase                                Wall-clock
-----------------------------------  -----------
Scorched-earth delete                ~30 s
mamba env create + python -m pip     ~1.5 min
chorus setup --oracle all            ~60 min
chorus setup --oracle chrombpnet     ~8 min   (post-pull, HepG2 prefetch)
chorus list / health / --version     ~1 min
README Step 3                        12 s
3 notebooks                          ~25 min
MCP + CDF integration tests          ~5 min
Multi-oracle + discover smoke        ~7 min
Fast pytest (parallel-throttled)     ~34 min
-----------------------------------  -----------
Total                                ~2 h 22 min
```

## Conclusion

v28 is the closing brace on the v22→v27 audit chain. With the four v27
P0/P1 fixes in place, the README TLDR walks a new user from `git clone`
to a real prediction in 60-65 minutes with zero manual intervention,
zero workarounds, and zero error messages along the way. All three
shipped notebooks execute end-to-end. The MCP server connects and
serves real predictions. Fine-mapping / discovery / multi-oracle code
paths all return finite, sensible values.

No further action recommended.
