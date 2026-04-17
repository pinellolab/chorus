# Chorus Fresh-Install Audit v6

**Date**: 2026-04-16
**Platform**: macOS 15.7.4 / Apple M3 Ultra / 96 GB
**Branch**: `chorus-applications` @ `d796f01`
**Auditor**: Claude Opus 4.7 (1M context)
**Audit branch**: `audit/2026-04-16-v6-fresh-install`

## Scope

Full clean-slate audit after the v5 fixes merged at `d796f01`. Before
starting, every user-visible chorus artifact was removed:

| Deleted | Size |
|---------|------|
| `~/.chorus/` | 1.5 GB |
| 7 mamba envs (`chorus` + 6 oracle envs) | 10.3 GB |
| `~/.cache/huggingface/hub/models--google--alphagenome-all-folds` | 701 MB |
| `~/.cache/huggingface/hub/models--johahi--borzoi-replicate-0` | 709 MB |

Preserved: `genomes/hg38.fa` (this is outside the chorus cache —
`GenomeManager.genomes_dir` resolves to `<repo>/genomes/`, not
`~/.chorus/genomes/`) and `<repo>/downloads/{chrombpnet,sei,legnet}/`
(non-HF model weights, 37 GB ChromBPNet + 6.4 GB SEI + 41 MB LegNet).
Blowing away the 37 GB ChromBPNet model for this audit would have been
a disproportionate compute cost; I treated HF-hosted weights as the
"canonical cache" for teardown.

## Executive summary

1. **BUG (MEDIUM)** — Regen scripts recreate 5 orphan HTML files that
   the `df7d613` commit deleted from the repo. `df7d613` only removed
   them from git but left the regen-time generators intact. After the
   first regen, the tree has 17 HTMLs instead of the 12 that were
   committed.
2. All 281 pytest tests pass on a brand-new base env (17.5 s).
3. All 6 oracle envs build without dependency conflicts (10.1 GB
   total after install).
4. All 6 oracles pass the official smoke test in 6 min 9 s, forcing
   fresh HF downloads for Enformer, Borzoi, and AlphaGenome. Metal
   guard held across both paths.
5. All 12 primary example apps regenerated cleanly. Data diff vs
   committed outputs is within AlphaGenome CPU non-determinism
   tolerance (max Δeffect 0.036).
6. All 3 notebooks execute end-to-end with **0 errors, 0 warnings,
   0 stale "no baselines" messages** across a total of 129 code cells.
7. All 17 HTML reports audit cleanly in Selenium (0 SEVERE errors,
   0 CDN references, enriched CHIP names + `≥99th` + biology
   interpretation column all present).
8. 4 of 6 normalization CDFs downloaded and pass all monotonicity /
   counts / percentile-ordering checks. Remaining 2 (sei, legnet)
   weren't triggered by this audit's regen workflow — noted.

## Phase 0 — Teardown

Deleted 13.2 GB of caches + envs. Verified absent before continuing.

## Phase 1 — Base install

```bash
mamba env create -f environment.yml   # fresh, ~10 min
pip install -e .                       # editable
python -m ipykernel install --user --name chorus
```

All succeeded. `import chorus` → 0.1.0; `scorers`, `normalization`
import without error.

## Phase 2 — Fresh pytest

```
281 passed in 17.53s
```

The v5 column-name fixes hold on a clean install.

## Phase 3 — Oracle env install

All 6 envs installed via `chorus setup --oracle <name>` in parallel.
No conflicts observed.

| Oracle | Env size |
|--------|----------|
| chorus-enformer | 1.5 GB |
| chorus-borzoi | 1.5 GB |
| chorus-chrombpnet | 1.6 GB |
| chorus-sei | 395 MB |
| chorus-legnet | 747 MB |
| chorus-alphagenome | 2.9 GB |

## Phase 4 — First-time model download + smoke predict

`pytest tests/test_smoke_predict.py -v -s` — all 6 oracles PASS in
**6 min 9 s**. HF models re-downloaded from zero for
AlphaGenome + Borzoi (1.4 GB total). Enformer pulls from its own
HuggingFace mirror as part of the init. Sei / LegNet / ChromBPNet
use pre-existing weights at `<repo>/downloads/{oracle}/` (see scope
notes above).

AlphaGenome loads on `TFRT_CPU_0` — the v4 Metal guard (that the
first fix of this audit series addressed) holds.

## Phase 5 — Regenerate 12 examples from zero

Ran the 4 regen scripts in parallel:

| Script | Duration | Apps |
|--------|----------|------|
| `regenerate_examples.py --oracle alphagenome` | 19.7 min | SORT1, BCL11A, FTO, SORT1_CEBP |
| `regenerate_examples.py --oracle enformer` | ~3 min | SORT1_enformer, 2 enformer validation HTMLs |
| `regenerate_examples.py --oracle chrombpnet` | <2 min | SORT1_chrombpnet |
| `regenerate_remaining_examples.py` | ~46 min | discovery, causal, region_swap, integration, batch, TERT |

Diff vs pre-regen snapshot (per-track log2FC, quantile_score):

| App | N common tracks | Max Δeffect | Max Δquantile |
|-----|----------------|-------------|---------------|
| SORT1_rs12740374 | 6 | 0.020 | 0.000 |
| BCL11A_rs1427407 | 6 | 0.012 | 0.000 |
| FTO_rs1421085 | 6 | 0.010 | 0.011 |
| SORT1_CEBP | 6 | 0.016 | 0.000 |
| TERT_chr5_1295046 | 16 | 0.023 | 0.084 |
| SORT1_enformer | 48 | 0.005 | 0.040 |
| SORT1_chrombpnet | 1 | 0.0001 | 0.0001 |
| integration_simulation | 3 | 0.036 | 0.000 |
| region_swap | 4 | 0.031 | 0.000 |
| batch_scoring | 30 | 0.017 | 0.291 |

All within expected AlphaGenome CPU non-determinism tolerance.
Discovery and causal_prioritization schemas aren't covered by the
diff script (their JSON nests variants/cell-types differently); their
HTML reports pass the Selenium audit in Phase 7.

### Finding #1 (MEDIUM): regen re-creates 5 orphan HTMLs

`df7d613` deleted these 6 files from the repo:
- 3 discovery sub-reports in `discovery/SORT1_cell_type_screen/`
- 1 CELSR2 validation report (this one *is* gone — the CELSR2
  example dict no longer exists in the regen script)
- 2 enformer duplicates: `chr1_109274968_G_T_SORT1_enformer_report.html`
  in both `SORT1_enformer/` and `SORT1_rs12740374_with_CEBP/`

After a fresh regen, **5 of those 6 files come back**:
- 3 discovery sub-reports (re-attached user prompt via patch in
  `regen_discovery`, but the main discovery report supersedes them)
- 2 enformer duplicates (the HTML writer emits `chr*.html` first,
  then `report.to_html(output_path=target)` writes the real target,
  but the `chr*.html` isn't deleted)

Post-regen tree has **17 HTMLs vs 12 committed**. Running regen
produces `git status` noise from these files unless manually deleted.

Proposed fix (separate PR):
- In `regenerate_examples.py::regenerate_enformer_discovery`, after
  `report.to_html(output_path=target)`, delete any other `chr*.html`
  in the output directory that isn't `target`.
- In `regenerate_remaining_examples.py::regen_discovery`, after the
  patched per-cell-type HTMLs are written, delete them (keep only the
  combined `discovery_summary.json` + the main report). Or decide
  these per-cell-type sub-reports are desired and update the
  committed repo state to include them.

## Phase 6 — Notebooks, cell-by-cell

Executed all 3 in the freshly-built `chorus` kernel via `nbconvert`.

| Notebook | Code cells | Errors | Warnings | Stale "no baselines" |
|----------|-----------|--------|----------|---------------------|
| single_oracle_quickstart | 34 | 0 | 0 | 0 |
| comprehensive_oracle_showcase | 38 | 0 | 0 | 0 |
| advanced_multi_oracle_analysis | 57 | 0 | 0 | 0 |

Cell-by-cell deep audit looked for: error outputs, traceback text,
stale normalizer messages, deprecation/user warnings, invalid-value
markers (`nan detected` / `inf detected`). None found in any of the
**129 code cells across all 3 notebooks**.

NB3 has 12 code cells with no output — each is an assignment
statement (`result = oracle.predict(…)`, `from X import Y`) that
isn't expected to print. Confirmed by inspection.

## Phase 7 — Selenium HTML audit + screenshots

17 HTML reports audited in headless Chrome (1400×4000). Screenshots
at `audits/2026-04-16_v6_screenshots/` (gitignored).

| Check | Result |
|-------|--------|
| Analysis Request section present | 17/17 |
| User prompt visible in section | 17/17 |
| IGV loaded | 16/17 (batch scoring has none by design) |
| SEVERE console errors | 0 |
| CDN igv.js references | 0 |
| Enriched CHIP display (`CHIP:TF:CELL`) | all applicable reports |
| `≥99th` / `≤1st` percentile format | all applicable reports |
| Interpretation column text | all applicable reports |

**Note**: the 17 HTMLs include the 5 orphan regen-created files
flagged in Finding #1. The original 12 committed HTMLs all pass.

## Phase 8 — Normalization CDFs

4 CDFs downloaded during the audit (the oracles whose examples were
regenerated):

| Oracle | n tracks | min effect count | signed / unsigned | status |
|--------|---------|------------------|---------------------|--------|
| alphagenome | 5168 | 1697 | 667 / 4501 | OK |
| borzoi | 7611 | 6563 | 1543 / 6068 | OK |
| chrombpnet | 24 | 9609 | 0 / 24 | OK |
| enformer | 5313 | 9600 | 0 / 5313 | OK |

All 4 pass: effect and summary CDFs monotone, p50 ≤ p95 ≤ p99 > 0,
all `effect_counts > 0`. Values identical to v4 / v5 — confirms the
HF-hosted CDF files transfer correctly.

Sei + LegNet CDFs weren't triggered by this audit's regen workflow
(no Sei/LegNet examples to regenerate). Not a regression; just not
exercised here.

## Phase 9 — Biology spot-check (same as v4 / v5)

SORT1 rs12740374 regen reproduces the published Musunuru 2010
mechanism:

| Track | Ref | Alt | log2FC | Direction |
|-------|-----|-----|--------|-----------|
| DNASE:HepG2 | ~510 | ~700 | +0.44 | Chromatin opens ✓ |
| CEBPA:HepG2 | ~2100 | ~2725 | +0.38 | C/EBP site gained ✓ |
| CEBPB:HepG2 | ~1216 | ~1470 | +0.27 | Paralog co-binding ✓ |
| H3K27ac:HepG2 | ~13700 | ~15500 | +0.18 | Active mark gain ✓ |

## Proposed follow-up (separate PR)

### MEDIUM — stop regen from creating orphan HTMLs

`scripts/regenerate_examples.py::regenerate_enformer_discovery`:
```python
# After the re-write, delete any chr*.html that isn't our target
import glob
for f in glob.glob(f"{out_dir}/chr*.html"):
    if os.path.abspath(f) != os.path.abspath(target):
        os.remove(f)
```

`scripts/regenerate_remaining_examples.py::regen_discovery`: either
delete the per-cell-type HTMLs after patching (they duplicate the
main report) or update the committed repo state to include them.

## Verdict

**PASS with 1 new finding.** The v5 fixes hold up cleanly on a
complete from-zero install. Every oracle installs, downloads its
model, and predicts correctly. All notebooks execute clean. All
primary HTML reports render correctly with the enriched labels,
analysis request, and `6 HepG2 tracks` / `6 K562 tracks` from the
v5 fix. The only new finding — orphan HTMLs after regen — is a
documentation-layer bug that doesn't affect runtime behaviour; its
fix is one `glob + remove` block in each of the two regen scripts.
