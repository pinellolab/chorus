# v0.5.6 — Per-walkthrough reproduction notebooks

**Date**: 2026-05-15
**Branch**: `fix/v0.5.6-walkthrough-notebooks`
**Trigger**: collaborator final-touches request — every walkthrough
should ship a Jupyter notebook that reproduces the same result as the
matching MCP query.

## Context

Each `examples/walkthroughs/<category>/<name>/` already ships a
markdown report, JSON, TSV, and an HTML IGV view. What was missing
was **executable Python code** that produces those artifacts. v0.5.6
adds one `notebook.ipynb` per walkthrough — 13 in total — generated
by a single declarative script. Each notebook calls the same Python
API the MCP tool wraps, so top-to-bottom execution reproduces the
documented outputs.

## Notebook contract (per user spec)

Every generated notebook satisfies:

- Single imports cell at the top — no later imports.
- Assumes `chorus` is already installed; no `pip install` cells.
- One logical step per cell with a leading `#` comment.
- All function arguments explicit; no implicit defaults.
- Dedicated save cell that writes markdown, JSON, TSV, and HTML
  to the walkthrough dir.
- Top-to-bottom execution reproduces the MCP output.

## Implementation

### `scripts/generate_walkthrough_notebooks.py` (new)

Declarative generator. The `WALKTHROUGHS` list at the top holds 13
walkthrough specs (path, MCP tool, oracle, args). Eight cell-template
builders cover the seven distinct MCP-tool flows plus a multi-oracle
template:

| Template | Walkthroughs covered |
|---|---|
| `_cells_analyze_variant_multilayer` | 5 (variant_analysis × 4, validation/SORT1_CEBP) |
| `_cells_chrombpnet_variant` | 1 (variant_analysis/SORT1_chrombpnet) |
| `_cells_discover_variant` | 2 (variant_analysis/SORT1_enformer, validation/TERT) |
| `_cells_discover_variant_cell_types` | 1 (discovery/SORT1_cell_type_screen) |
| `_cells_fine_map_causal_variant` | 1 (causal_prioritization/SORT1_locus) |
| `_cells_score_variant_batch` | 1 (batch_scoring) |
| `_cells_analyze_region_swap` | 1 (sequence_engineering/region_swap) |
| `_cells_simulate_integration` | 1 (sequence_engineering/integration_simulation) |
| `_cells_multioracle` | 1 (validation/SORT1_rs12740374_multioracle) |

Notable design choices:

- **`use_environment=True` everywhere** — notebooks run in the base
  `chorus` kernel; each oracle delegates its model load to its own
  mamba env via subprocess. No notebook-level env switching needed.
- **Causal-prioritization notebook inlines LD proxies** as a Python
  list of dicts. Avoids LDlink network dependency. Adds a
  commented-out `fetch_ld_variants(...)` block for users who want
  fresh proxies.
- **Multi-oracle notebook is the only one that runs three oracles
  end-to-end**, then calls `build_multi_oracle_report` to
  consolidate. 13 cells; same flow `regenerate_multioracle.py`
  delegates plus the `--consolidate` step.
- **AG track IDs** in spec lists use the real
  `OUTPUT_TYPE/<name>/<strand>` format (the v0.5.4 fix). No display
  names like `"DNASE:HepG2"`.

### `setup.py`

Added `nbformat>=5.0` to `extras_require["dev"]`. Already pulled in
transitively by `nbconvert`, but explicit is better.

### `examples/walkthroughs/README.md`

Added a one-paragraph note pointing users at `notebook.ipynb` per
walkthrough and the `jupyter nbconvert --execute` command.

### `CLAUDE.md`

Added `python scripts/generate_walkthrough_notebooks.py` to the
regeneration command block, with a note explaining it is codegen-only
and shouldn't be confused with the GPU-heavy notebook execution.

## Files changed

```
chorus/__init__.py                                                   version bump
setup.py                                                             version bump + nbformat dev-dep
CLAUDE.md                                                            regen instructions
examples/walkthroughs/README.md                                      notebook discoverability
scripts/generate_walkthrough_notebooks.py                            new — generator
audits/2026-05-15_v0.5.6_walkthrough_notebooks/                      report + pytest log
examples/walkthroughs/**/notebook.ipynb                              13 new notebooks
```

## Per-walkthrough notebook inventory

```
examples/walkthroughs/variant_analysis/SORT1_rs12740374/notebook.ipynb         10 cells
examples/walkthroughs/variant_analysis/SORT1_chrombpnet/notebook.ipynb         10 cells
examples/walkthroughs/variant_analysis/SORT1_enformer/notebook.ipynb            8 cells
examples/walkthroughs/variant_analysis/BCL11A_rs1427407/notebook.ipynb         10 cells
examples/walkthroughs/variant_analysis/FTO_rs1421085/notebook.ipynb            10 cells
examples/walkthroughs/validation/SORT1_rs12740374_with_CEBP/notebook.ipynb     10 cells
examples/walkthroughs/validation/SORT1_rs12740374_multioracle/notebook.ipynb   13 cells
examples/walkthroughs/validation/TERT_chr5_1295046/notebook.ipynb               8 cells
examples/walkthroughs/discovery/SORT1_cell_type_screen/notebook.ipynb           8 cells
examples/walkthroughs/causal_prioritization/SORT1_locus/notebook.ipynb         10 cells
examples/walkthroughs/sequence_engineering/region_swap/notebook.ipynb           8 cells
examples/walkthroughs/sequence_engineering/integration_simulation/notebook.ipynb  8 cells
examples/walkthroughs/batch_scoring/notebook.ipynb                              8 cells
```

## Verification

- ✅ Generator produces 13 notebooks, no errors.
- ✅ Static lint: every notebook has imports, oracle setup, predict
  call, save artifacts, and a title — see `pytest_log.txt`.
- ✅ Each notebook is a valid `nbformat` v4 document with the
  `"Python 3 (chorus)"` kernel metadata, matching the convention
  used by the existing `examples/notebooks/*.ipynb`.
- ✅ Full base regression: see `pytest_log.txt`.

## Out of scope (deliberate)

- **Auto-executing notebooks in CI.** Requires GPUs + per-oracle
  envs; current CI is CPU-only. Manual `jupyter nbconvert --execute`
  before each release, same policy as the existing showcase
  notebooks.
- **Per-walkthrough `.py` script alongside the notebook.** Doubles
  maintenance. `jupyter nbconvert --to script` is a one-liner if a
  user wants it.
- **Live LDlink fetch in the causal-prioritization notebook.**
  Inlined the 11 proxies; commented-out alternative path for
  network-on users.
- **Regenerating the existing `examples/notebooks/*.ipynb`** —
  those are multi-topic showcases, different artifact, unchanged.
