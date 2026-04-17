# Chorus First-User UX Audit v7

**Date**: 2026-04-17
**Platform**: macOS 15.7.4 / Apple M3 Ultra / 96 GB
**Branch**: `chorus-applications` @ `88d1a45`
**Auditor**: Claude Opus 4.7 (1M context) playing "first-time user"
**Audit branch**: `audit/2026-04-17-v7-first-user-ux`

## Scope

Adopt the perspective of a computational biology grad student who just
discovered Chorus on GitHub and is trying to use it. Walk through the
first ~30 minutes of their experience: reading the README, running the
Minimal Working Example, opening the first notebook, browsing the
example reports, skimming the MCP walkthrough, exploring CLI help.
Flag everything that would make them frustrated or confused.

This is not a correctness audit (v6 already covered that). It's a
polish / rough-edges audit.

## Executive summary

Four findings, all low-to-medium severity. The install path is solid,
the Minimal Example runs verbatim, and the MCP walkthrough conveys
intent well. The rough edges are:

1. **MEDIUM** — `chorus list` shows a phantom `base` env that's always
   "✗ Not installed" because `chorus-base.yml` exists in `environments/`
   but the actual base env is named `chorus`, not `chorus-base`. A new
   user who just finished `mamba env create -f environment.yml` sees
   a scary ✗ right after successful install.
2. **MEDIUM** — `docs/MCP_WALKTHROUGH.md:38` uses the wrong parameter
   name `alt_allele="T"` (singular string). The actual MCP tool
   signature is `alt_alleles: list[str]` (plural, list). Another
   example in the same file (line 70) uses the correct form, so a
   newcomer copy-pasting line 38 would hit a `TypeError`.
3. **LOW** — `examples/advanced_multi_oracle_analysis.ipynb:cell 1`
   has a stale copy-paste subtitle: "This notebook demonstrates all
   major features of the Chorus library **using the Enformer oracle**
   to analyze the GATA1 transcription start site (TSS) region." —
   but the title says "Enformer, ChromBPNet, and LegNet" and it IS
   a multi-oracle notebook. The subtitle is a leftover from
   `single_oracle_quickstart.ipynb`.
4. **LOW** — `examples/applications/variant_analysis/SORT1_rs12740374/README.md`
   has a "Key results" table claiming percentiles `99th, 98th, 95th,
   90th, 88th` across the 5 primary tracks. The actual `example_output.md`
   shows all five at `≥99th`. Numbers drifted after the v5 `_fmt_percentile`
   update but the summary table in the app's README was missed.

## Phase-by-phase findings

### Phase 1 — Reading the README

The README flows cleanly: Overview → Key terms → "Start here"
application examples → Prerequisites → Installation → Quick Start.
The Key Terms glossary added in commit `f935bf5` works well for a
first-time user — defines oracle / track / assay_id / effect percentile
/ log2FC before first use.

Nitpicks I noticed but didn't flag as bugs:
- "Prerequisites" mentions `mamba` without mentioning that `mamba`
  ships with Miniforge — the one-liner install of Miniforge does
  give you mamba, but a new user might not know that.
- The "Two env files, one source of truth" blockquote is clear — but
  the `environments/chorus-base.yml` file still exists in the repo
  and (as Finding #1 shows) leaks through `chorus list`.

### Phase 2 — Minimal Working Example (README lines 226-247)

**Runs verbatim.** Copy-pasted the 19-line code block into a fresh
file. First run on this machine (`enformer` env already set up from
v6) produced:

```
Mean signal: 0.47, Max: 15.04
```

No manual tweaks needed. `get_genome()`, `chorus.create_oracle()`,
`oracle.load_pretrained_model()`, and `oracle.predict()` all work
as documented.

### Phase 3 — Notebook first impressions

Opened all three notebooks and read the first few cells as a
newcomer would.

- `single_oracle_quickstart.ipynb` — clean narrative intro.
  Title: "Single Oracle Quick Start: GATA1 Regulatory Analysis with
  Enformer". Markdown cell explains why GATA1, then Installation
  cell, then imports. Good.

- `comprehensive_oracle_showcase.ipynb` — starts with a
  ⚠️ blockquote warning about 6 envs + several hours + 80 GB disk.
  That's exactly what a first-time user needs up-front.

- **`advanced_multi_oracle_analysis.ipynb`** — ⚠️ warning is there,
  title is correct ("Enformer, ChromBPNet, and LegNet"), but the
  second sentence of the first markdown cell says:

  > This notebook demonstrates all major features of the Chorus library
  > **using the Enformer oracle** to analyze the GATA1 transcription
  > start site (TSS) region.

  This is a copy-paste from `single_oracle_quickstart.ipynb`. Confusing
  because the title promises multi-oracle and the subtitle says
  "using the Enformer oracle". _See Finding #3._

### Phase 4 — Walking through one application example

Opened `examples/applications/variant_analysis/SORT1_rs12740374/`.
Starting from the README in that folder:

- README explains the biology cleanly (Musunuru 2010, C/EBP site creation).
- "Example prompt" gives a natural-language question a user can paste
  into Claude.
- "Key results" table shows what the analysis should produce.
- Output files listed with one-line descriptions.

**Finding**: the Key results table hard-codes percentiles `99th, 98th,
95th, 90th, 88th` across the 5 primary tracks. Opening
`example_output.md` in the same folder shows:

```
| DNASE:HepG2       | +0.449 | ≥99th | Strong opening        |
| CHIP:CEBPA:HepG2  | +0.376 | ≥99th | Strong binding gain   |
| CHIP:CEBPB:HepG2  | +0.274 | ≥99th | Moderate binding gain |
| CHIP:H3K27ac:HepG2| +0.178 | ≥99th | Moderate mark gain    |
```

All five tracks are at `≥99th` in the actual data. The README's
graduated (99 → 88th) table is stale. A new user who carefully
reads the README and then the MD will notice the discrepancy.
_See Finding #4._

### Phase 5 — MCP walkthrough + tool schemas

All 10 main MCP tools introspect cleanly:

| Tool | Params | Has `user_prompt`? | Docstring |
|------|--------|-------------------|-----------|
| `list_oracles` | 0 | n/a | ✓ |
| `list_tracks` | 2 | n/a | ✓ |
| `load_oracle` | 7 | n/a | ✓ |
| `analyze_variant_multilayer` | 9 | ✓ | ✓ |
| `discover_variant` | 8 | ✓ | ✓ |
| `discover_variant_cell_types` | 8 | ✓ | ✓ |
| `score_variant_batch` | 6 | ✓ | ✓ |
| `fine_map_causal_variant` | 9 | ✓ | ✓ |
| `analyze_region_swap` | 7 | ✓ | ✓ |
| `simulate_integration` | 7 | ✓ | ✓ |

**Finding**: the walkthrough's Example 1 (line 38) contains the wrong
kwarg:

```python
discover_variant(oracle_name="alphagenome", position="chr1:109274968",
    ref_allele="G", alt_allele="T", ...)          # ← wrong
```

The actual signature is `alt_alleles: list[str]`. Example 2 on line 70
uses `alt_alleles=["C"]` correctly. Only one walkthrough example is
affected. _See Finding #2._

### Phase 6 — Error messages

Exercised four common newcomer-error paths. Messages are
actionable and suggest what to do:

| Bad call | Error message |
|----------|--------------|
| `chorus.create_oracle('foo')` | `ValueError: Unknown oracle: foo. Available: ['enformer', 'borzoi', 'chrombpnet', 'sei', 'legnet', 'alphagenome']` |
| `list_tracks(oracle_name="foobarbaz")` | `{'error': "Unknown oracle: 'foobarbaz'. Valid names: enformer, borzoi, chrombpnet, sei, legnet, alphagenome"}` |
| `analyze_variant_multilayer(..., alt_allele="T")` | `TypeError: got an unexpected keyword argument 'alt_allele'` (would be more helpful if it said "did you mean `alt_alleles` (plural)?") |

### Phase 7 — CLI help

`chorus --help`, `chorus setup --help`, `chorus health --help`,
`chorus genome --help`, `chorus list`, `chorus genome list` all
render readable output.

`chorus list` output:

```
Available oracle environments:
--------------------------------------------------
alphagenome          chorus-alphagenome        ✓ Installed
base                 chorus-base               ✗ Not installed
borzoi               chorus-borzoi             ✓ Installed
chrombpnet           chorus-chrombpnet         ✓ Installed
enformer             chorus-enformer           ✓ Installed
legnet               chorus-legnet             ✓ Installed
sei                  chorus-sei                ✓ Installed
--------------------------------------------------
```

`base` showing as `✗ Not installed` is a false negative — the env
it's looking for (`chorus-base`) doesn't exist because the install
instructions use `chorus` instead (from the root `environment.yml`).
_See Finding #1._

## Proposed follow-up (separate PR)

### Finding #1 — hide `base` from `chorus list`

`chorus/core/environment/manager.py::list_available_oracles` scans
`environments/` for every `chorus-*.yml`. Filter out `base`:

```python
def list_available_oracles(self) -> List[str]:
    """List all oracles with environment definitions."""
    oracle_envs = []
    env_files = self.base_path.glob("*.yml")
    for env_file in env_files:
        if env_file.stem.startswith("chorus-") and env_file.stem != "chorus-base":
            oracle_name = env_file.stem.replace("chorus-", "")
            oracle_envs.append(oracle_name)
    return sorted(oracle_envs)
```

Also guard in `chorus setup --oracle base` so a user who runs it by
mistake gets a friendly "base is managed by `mamba env create -f
environment.yml`, not `chorus setup`" message.

### Finding #2 — fix the walkthrough kwarg typo

`docs/MCP_WALKTHROUGH.md:38`:

```diff
- ref_allele="G", alt_allele="T", gene_name="SORT1", user_prompt="Load AlphaGenome...")
+ ref_allele="G", alt_alleles=["T"], gene_name="SORT1", user_prompt="Load AlphaGenome...")
```

### Finding #3 — fix the NB3 subtitle

`examples/advanced_multi_oracle_analysis.ipynb` cell 1:

```diff
- This notebook demonstrates all major features of the Chorus library using the Enformer oracle to analyze the GATA1 transcription start site (TSS) region.
+ This notebook demonstrates a multi-oracle workflow — running Enformer,
+ ChromBPNet, and LegNet side by side on the same GATA1 TSS region so you can
+ compare their per-base resolution, TF-binding predictions, and MPRA activity.
```

### Finding #4 — refresh SORT1 README percentiles

`examples/applications/variant_analysis/SORT1_rs12740374/README.md`:

```diff
- | DNASE:HepG2 | +0.43 | 99th | Strong chromatin opening |
- | CEBPA ChIP | +0.37 | 98th | Strong TF binding gain |
- | CEBPB ChIP | +0.22 | 95th | Moderate TF binding gain |
- | H3K27ac | +0.18 | 90th | Moderate enhancer activation |
- | CAGE+ | +0.15 | 88th | Moderate transcription increase |
+ | DNASE:HepG2       | +0.45 | ≥99th | Strong opening        |
+ | CHIP:CEBPA:HepG2  | +0.38 | ≥99th | Strong binding gain   |
+ | CHIP:CEBPB:HepG2  | +0.27 | ≥99th | Moderate binding gain |
+ | CHIP:H3K27ac:HepG2| +0.18 | ≥99th | Moderate mark gain    |
+ | CAGE:HepG2        | +0.25 | ≥99th | Moderate increase     |
```

(Exact numbers sourced from the current `example_output.md`.)

### Optional nit — friendlier TypeError for wrong allele kwarg

The MCP `_safe_tool` wrapper (`chorus/mcp/server.py:137`) already
converts exceptions into `{"error": ...}` dicts. It could also
intercept `TypeError: got an unexpected keyword argument` and
suggest the closest valid parameter name. Low priority — the
current error is still diagnostic.

## Verdict

**PASS with 4 low-to-medium polish items.** The install + minimal
example + MCP plumbing all work correctly out of the box — this is
a product a first-time user can actually use within 30 minutes of
cloning the repo. The four findings are all documentation drift,
not runtime bugs; each is a one- or two-line fix.
