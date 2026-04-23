# v25 — README TLDR effectiveness audit — 2026-04-23

User asked: "remove all the oracle envs and chorus main env, do a pull
and follow as a new user. Also I changed the readme to be easier to
follow make sure this is effective! try everything!"

## Pre-nuke state

| Asset | Size |
|---|---|
| 7 conda envs (`chorus` + 6 per-oracle) | — |
| `~/.chorus/` | 1.5 GB |
| `genomes/hg38.*` | 3.0 GB |
| `downloads/*` | 12 GB (partial from v23/v24) |

All deleted; `mamba env list | grep chorus` → empty.

## Followed the new README TLDR verbatim

### Step 1 — Install (README L18-22) ✓

```bash
mamba env create -f environment.yml    # 78 s
pip install -e .                        # 14 s
python -c "import chorus; print(chorus.__version__)"  → 0.1.0
```

Total: ~90 s.

### Step 2 — `chorus setup` (bare, README L25-34) ✓

README says: "Pulls every oracle's weights, all background CDFs, and
the hg38 reference — everything pre-downloaded so your first prediction
doesn't block on a multi-GB tarball."

Ran end-to-end with `HF_TOKEN` + `LDLINK_TOKEN` as env vars (the README
documents the interactive prompt path; non-TTY falls back to env /
flag per the v22 token-halt contract).

**Timing**: 16:28:07 → 17:27:55 = **59 min 48 s** wall clock
(consistent with "45–60 min" in README).

**Result**: 6/6 markers written, 6/6 Healthy.

```
downloads/alphagenome/.chorus_setup_v1
downloads/borzoi/.chorus_setup_v1
downloads/chrombpnet/.chorus_setup_v1
downloads/enformer/.chorus_setup_v1
downloads/legnet/.chorus_setup_v1
downloads/sei/.chorus_setup_v1

chorus health → 6/6 Healthy
```

Both Sei P1 fixes from v23 (SameFileError + materialize-cached-info)
held — fresh install reached Healthy without intervention.

### Step 3 — Prediction snippet (README L40-63) ✓ with 2 findings fixed

**Before fix** — the snippet as shipped fires a warning:

```
WARNING - Provided reference allele 'A' does not match the genome at
this position ('C'). Chorus will use the provided allele.
```

Because it passes `alleles=['A', 'G', 'C', 'T']` at `chr11:5247500` —
but the genome has **C** at that position. The first element is treated
as ref, so 'A' is substituted at a 'C' site.

Also, the printed summary `len(effects)` returns **3** (dict size:
`predictions`, `effect_sizes`, `variant_info`), not 4 as a new user
would read as "4 alleles scored".

**Fix landed in this PR** — README Step 3:
- `['A', 'G', 'C', 'T']` → `['C', 'A', 'G', 'T']` with a comment that
  the first element is the reference (= genome base).
- `len(effects)` → `len(effects['predictions']) - 1` with key-list
  printout: `"scored 3 alt alleles (['reference', 'alt_1', 'alt_2', 'alt_3'])"`.

**Post-fix output** (warning-free):
```
WT mean signal: 0.468
Variant result: scored 3 alt alleles (['reference', 'alt_1', 'alt_2', 'alt_3'])
```

### Step 4 — MCP server (README L67-75)

Not exercised interactively via Claude Code (no Claude Code CLI in this
environment), but the equivalent subprocess test passed:

```python
StdioTransport(command='chorus-mcp') + fastmcp.Client
→ list_oracles:     6/6 environments installed
→ list_tracks(AG):  13,605 char JSON response
→ list_genomes:     hg38 present + downloaded
→ get_gene_tss:     SORT1 → chr1:109393357 (- strand)
→ oracle_status:    {"loaded_oracles":[]}
```

All 5 tool calls succeed over stdio; 22 tools registered.

## All 3 notebooks run clean on the scorched install

| Notebook | Code cells | Errors | Warnings |
|---|---|---|---|
| `single_oracle_quickstart.ipynb` | 34/34 | **0** | **0** |
| `comprehensive_oracle_showcase.ipynb` | 38/38 | **0** | **0** |
| `advanced_multi_oracle_analysis.ipynb` | 57/57 | **0** | **0** |

No regressions. The notebook fixes from v23 (`108 → 109` in advanced,
Sei cache materialization) hold across fresh install.

## Fast test suite

(Running in background; outcome appended to `logs/10_pytest.txt`.)

## README effectiveness

The new TLDR is a real improvement over the previous "Fresh Install"
section:

1. **One command** for step 2 (`chorus setup`) instead of per-oracle
   loops. Saves 5–6 copy-paste cycles.
2. **Token contract explicit** up front — HF + LDlink mentioned with
   "when prompted" hint, registration URLs, and what happens if you
   skip LDlink.
3. **Runnable Step 3 snippet** that exercises both `predict()` and
   `predict_variant_effect()` in 20 lines — but had two drifts that a
   copy-paste user would hit on run 1 (fixed here).
4. **Time budgets** documented: "5 minutes" for install, "45–60 min,
   unattended" for setup. v25 measured: **~90 s + ~60 min** — matches
   exactly.

## Findings

### P2 (fixed here)

- `README.md:62-69` — Step 3 prediction snippet used `alleles=['A','G','C','T']`
  at `chr11:5247500` which triggers a ref-mismatch warning (genome has
  'C'). Corrected to `['C','A','G','T']` with inline comment.
- `README.md:69` — `len(effects)` prints 3 (dict keys) not 4 (alleles).
  Changed to `len(effects['predictions']) - 1` + key listing.

Both fixes in one edit. No behaviour change; just the shipped snippet
is now warning-free and the printout is meaningful.

## Artefacts in `audits/2026-04-23_v25_readme_tldr/`

- `report.md` — this summary
- `logs/00-10_*.txt` — pre-nuke state, env create, chorus setup,
  prediction snippet, chorus health, all 3 notebook runs, MCP E2E,
  pytest
- `nb_fresh/*.ipynb` — all 3 notebooks freshly executed

## Headline

New README TLDR works end-to-end from a zero-state host:

- **~60 min** from `git pull` → 6/6 Healthy (within the README's
  "45-60 min" estimate)
- **All 3 notebooks pass** with 0 errors / 0 warnings on the fresh
  install
- **MCP E2E passes** — 22 tools registered, 5 exercised successfully
- **Two small snippet drifts found + fixed** in Step 3 — the warning
  a new user would otherwise see on run 1 is gone
