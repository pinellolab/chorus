# Chorus v27 Scorched-Earth Audit

**Date**: 2026-04-24
**Platform**: macOS 15.7.4 / Apple M3 Ultra / 96 GB
**Branch**: `main` @ `41916af` (post v26 sweep)
**Auditor**: Claude Opus 4.7 (1M context)
**Fix branch**: `fix/2026-04-24-v27-track-id-validator`

## Why this audit

User asked for a fresh-from-zero install + every-layer test pass after
v26 was declared "done". v22, v23, v24, v25 all reported zero blockers
on macOS arm64 — but those audits each kept *something* warm
(downloads cache, env, genome) so an actual greenfield user run had
not been replayed since v21. v27 closes that gap.

## What was scorched

| Item                                | Size before |
| ----------------------------------- | ----------- |
| `~/.local/share/mamba/envs/chorus*` (7 envs) | ~10.4 GB |
| `~/.chorus/`                        | 1.5 GB |
| `genomes/hg38.fa*`                  | 3.0 GB |
| `downloads/`                        | 9.9 GB |
| **Total**                           | **~24.8 GB** |

Verified empty afterwards: `mamba env list | grep chorus` → none,
`ls ~/.chorus genomes/ downloads/` → not present.

## What was reinstalled (chronological)

Following README TLDR Step-by-Step exactly as a new user would:

1. **Step 1 — base env** (5 min)
   ```bash
   mamba env create -f environment.yml
   mamba activate chorus
   pip install -e .
   ```
   Clean. `chorus --help` works.

2. **Step 2 — `chorus setup --oracle all`** (60 min, unattended)

   Per-oracle wall-clock from setup log:

   | Oracle | env  | weights | total |
   | ------ | ---- | ------- | ----- |
   | alphagenome | 0:46 | 1:30 | 12:01 (incl. hg38 download 9:40) |
   | borzoi      | 0:37 | 0:25 | 1:02 |
   | chrombpnet  | 0:34 | 7:42 | 8:16 |
   | enformer    | 0:33 | 0:24 | 0:57 |
   | legnet      | 0:30 | 0:46 | 1:16 |
   | sei         | 0:32 | 34:46 | 35:18 |
   | **Total**   |      |       | **~60 min** |

   No errors, no warnings, no manual intervention required. HF token
   accepted on first try (`✓ HuggingFace auth ok (user: lucapinello, via env)`).

3. **Step 3 — README Python snippet** — runs in 12 s end-to-end:
   - Loads Enformer in `chorus-enformer` env
   - WT mean signal: `0.468` for ENCFF413AHU at chr11:5247000-5248000
   - Variant scan: 3 alt alleles scored (`['reference','alt_1','alt_2','alt_3']`)
   - No ref-mismatch warning (the v25 fix to `['C','A','G','T']` holds)

4. **Step 4 — MCP** — verified separately via E2E integration test:
   - `pytest tests/test_integration.py::test_mcp_e2e_list_oracles_and_analyze_variant -m integration`
   - 280 s, PASSED
   - Spawns `chorus-mcp` stdio subprocess via `fastmcp.Client`
   - Calls `list_oracles` → `load_oracle` → `analyze_variant_multilayer`
     for SORT1 rs12740374 in HepG2 with AlphaGenome
   - Real AlphaGenome predict, real DNASE/EFO:0001187 track resolution

## Verification beyond TLDR

- `chorus list` → 6/6 ✓ Installed, exit 0
- `chorus health` → 6/6 ✓ Healthy in 43 s, exit 0
- **Quickstart notebook** (`single_oracle_quickstart.ipynb`) executed end-to-end on first try (after the P0 fix below)
- **Multi-oracle smoke** — Enformer + ChromBPNet predict on chr1:109274000-109276114, both return finite values, ChromBPNet ATAC:K562 model auto-downloaded from ENCODE (728 MB) when first invoked
- **Discovery smoke** — `discover_variant_effects` for SORT1 rs12740374 returns the expected 5-key dict (`cell_type_ranking`, `layer_rankings`, `total_tracks_scored`, `selected_tracks`, `report`)
- **Fast pytest** — 340 passed, 1 skipped (regression check post-validator-fix)

## Findings

### P0 (would break a new user) — fixed in this PR

**P0-1: `_validate_assay_ids` rejects FANTOM CAGE identifiers** (`chorus/oracles/enformer.py:351`, `chorus/oracles/borzoi.py:302`)

The track-ID guard introduced in v26 PR #44 only treated `ENCFF*` strings as identifiers; everything else fell through to the description-substring path. FANTOM CAGE identifiers (`CNhs11250`, etc.) are valid identifiers but don't start with `ENCFF`, so they were silently classified as descriptions. `get_tracks_by_description("CNhs11250")` returns empty → guard raises `InvalidAssayError`. The shipped quickstart notebook uses `['ENCFF413AHU', 'CNhs11250']` and would have failed for every new user on cell In[8]:

```
InvalidAssayError: Enformer does not recognise these track IDs: ['CNhs11250']
```

**Fix**: Try `get_track_by_identifier` *first* unconditionally; fall through to description lookup only if identifier lookup returns None. Same change applied to Enformer + Borzoi (Borzoi has the same code path with the same bug). Quickstart notebook now executes clean.

This is exactly the kind of bug a kept-warm install would never catch — the previous owner's environment already had the metadata cached and the guard was never exercised on a real first call.

### P1 (would confuse but recoverable) — fixed in this PR

**P1-1: `chorus-mcp --help` lists 20 tools, MCP server registers 22** (`chorus/mcp/server.py:1598`)

The hand-maintained list in the `--help` print block missed `discover_variant` and `fine_map_causal_variant`. README and MCP_WALKTHROUGH advertise both, FastMCP registers both, but a user running `chorus-mcp --help` to confirm the surface would think the README overstates the capabilities.

**Fix**: Reordered into 4 logical groups (Discovery / Lifecycle / Predict / Analyze) with the alphabetised list explicitly tagged `(22)`.

**P1-2: Dead anchor in `docs/MCP_WALKTHROUGH.md:14`**

Link `../README.md#mcp-server-ai-assistant-integration` resolves to nothing — README's heading is `### MCP server` → slug `#mcp-server`.

**Fix**: Updated to `#mcp-server`.

### P1 (deferred to a follow-up PR — non-blocking)

These are documentation drift items that don't break workflows:

- `examples/notebooks/single_oracle_quickstart.ipynb` md cell 1 — install steps say `chorus genome download hg38` after `chorus setup --oracle enformer`, but `chorus setup` already pre-fetches hg38. Redundant but harmless.
- `examples/notebooks/advanced_multi_oracle_analysis.ipynb` md cell 2 — same redundant hg38 step + omits `chorus setup --oracle legnet` despite the "Extra: MPRA-LegNet" section requiring it.
- `examples/notebooks/comprehensive_oracle_showcase.ipynb` md cell 1 — track-count table has small mismatches: ChromBPNet says "Per-model (DNASE, ATAC)" but Operation 8 covers CHIP too; Sei row says "128 bp" resolution but Sei outputs a single 4096 bp scalar (resolution_bp=None in ORACLE_SPECS); LegNet "1 track" should be 3 (K562/HepG2/WTC11).
- `audits/AUDIT_CHECKLIST.md:88,156` — track-count line mixes "total" with "CDF-backed" without labels; AlphaGenome counted as 5,168 in one row and 5,731 in another.

### P2 (minor / informational) — not fixed

- `chorus setup` "✓ chrombpnet weights ready" is the env-probe success — it does *not* download the user-specified model (e.g. ATAC:K562). First `predict()` triggers the real ENCODE tarball pull (728 MB). Behaviour is fine, but doc could note it. Same applies to all per-model oracles (chrombpnet, sei, legnet) where setup probes one default and predict pulls the rest on demand.
- `discover_variant_effects` triggers 7 identical "Annotation file already exists" log lines — looks like 7 separate `AnnotationManager` instances each logging the same probe. Cosmetic; merge or demote to DEBUG.
- ChromBPNet ATAC:K562 download progress logs at 14.4%, 28.8%, 43.3% etc. — every ~70 s, which is fine but could be coarsened to every 25%.
- `_validate_assay_ids` allows description-substring matches (e.g. `DNASE:HepG2`) that the prediction template's `id2index` then rejects. Validator and template have inconsistent contracts. The error from the template is actionable, so this is a P2 polish item, not a blocker.

## Summary

- v27 audit found **1 P0** (track-ID validator regression, fixed) and **2 P1 docs** (fixed in same PR), plus 4 P1 doc-drift items deferred to a follow-up.
- README TLDR Steps 1-4 work end-to-end after the fix. Total green-field
  install + first-prediction time: **~62 min** (60 min setup + 2 min predict).
- Without this scorched-earth run, the v26-introduced P0 in
  `_validate_assay_ids` would have shipped to every new user.

## Verified

```bash
# After the P0 fix
pytest tests/ --ignore=tests/test_smoke_predict.py -q
→ 340 passed, 1 skipped

# Quickstart notebook (the one that broke pre-fix)
jupyter nbconvert --to notebook --execute --inplace examples/notebooks/single_oracle_quickstart.ipynb
→ exit 0, ~3 min, all 22 cells execute

# MCP E2E (already in v9 integration suite)
pytest tests/test_integration.py::test_mcp_e2e_list_oracles_and_analyze_variant -m integration
→ 1 passed in 280.74s
```

## What is NOT in this audit

- **Linux / CUDA path** — still deferred to user's other machine.
- **Self-hosted CI runner** for smoke + integration tests — out of scope.
- **Long-form notebooks** — `advanced_multi_oracle_analysis.ipynb` and
  `comprehensive_oracle_showcase.ipynb` were skipped to keep the audit
  to one fresh-day session. Smoke-tested via direct API calls instead.
- **P1 notebook md-cell drift** — captured here, deferred to a follow-up.
