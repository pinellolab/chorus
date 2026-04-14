# Chorus end-to-end audit prompt (paste into Claude Code on a fresh machine)

> **Before sending**: replace the two `REPLACE_*` token placeholders below
> with your real tokens. Do NOT commit this file with real tokens
> embedded — keep the placeholders in the repo copy.

---

You are going to do a **complete, end-to-end production audit** of the
Chorus genomics MCP library on a clean machine, starting from nothing
but the GitHub URL. **Skip nothing.** Test every script, every notebook,
every MCP tool, every application example, and verify that the
normalization backgrounds work correctly. Produce a final audit report
at the end describing what worked, what failed, and what a real new
user would experience.

## Credentials (already valid, ready to use)

```bash
export HF_TOKEN="REPLACE_WITH_YOUR_HF_READ_TOKEN"   # gated AlphaGenome model download
export LDLINK_TOKEN="REPLACE_WITH_YOUR_LDLINK_TOKEN" # causal auto-fetch
```

Persist them in `~/.bashrc` so MCP / subprocesses inherit them:

```bash
echo 'export HF_TOKEN="REPLACE_WITH_YOUR_HF_READ_TOKEN"' >> ~/.bashrc
echo 'export LDLINK_TOKEN="REPLACE_WITH_YOUR_LDLINK_TOKEN"' >> ~/.bashrc
source ~/.bashrc
```

## Working location

Clone the repo into your preferred dev directory and `cd` into it:

```bash
git clone https://github.com/pinellolab/chorus.git
cd chorus
git checkout chorus-applications   # the active branch with all the audit-pass changes
```

All subsequent commands run from this directory.

---

# Phase 1 — Installation (follow the README literally)

Open `README.md` and follow the **Fresh Install** section verbatim. Do
NOT improvise. If a command fails, log the exact error and continue —
note it in the audit report rather than working around it silently.

## 1.1 Base env + package
```bash
mamba env create -f environment.yml
mamba activate chorus
pip install -e .

# Verify entry points
python -c "import chorus; print(f'chorus {chorus.__version__}')"
which chorus
which chorus-mcp
```

## 1.2 Set up ALL six oracle environments
```bash
chorus setup --oracle enformer
chorus setup --oracle borzoi
chorus setup --oracle chrombpnet
chorus setup --oracle sei
chorus setup --oracle legnet
chorus setup --oracle alphagenome     # gated, needs HF_TOKEN
chorus list                            # confirm all six present
```

## 1.3 Download the reference genome
```bash
chorus genome download hg38
chorus genome info hg38
```

## 1.4 Health check (this also triggers first-use model downloads)
```bash
chorus health --timeout 600
```

Record total install time and total disk used (both are real new-user
concerns):

```bash
df -h ~/.cache/huggingface ~/.chorus genomes/ 2>/dev/null
du -sh ~/.cache/huggingface ~/.chorus genomes/ 2>/dev/null
```

---

# Phase 2 — Per-track background normalization

This is the system that turns raw log2FC into the percentiles every
report shows. **Verify it actually works**, not just that files exist.

## 2.1 Auto-download from HuggingFace (public dataset, no token needed)
```python
from chorus.analysis.normalization import (
    download_pertrack_backgrounds,
    get_pertrack_normalizer,
)

# Pre-download all six oracles' backgrounds
for o in ["alphagenome", "enformer", "borzoi", "chrombpnet", "sei", "legnet"]:
    n = download_pertrack_backgrounds(o)
    print(f"{o}: {n} files downloaded")

# Confirm cache landed in ~/.chorus/backgrounds/
import os
print(sorted(os.listdir(os.path.expanduser("~/.chorus/backgrounds/"))))
```

## 2.2 End-to-end percentile lookup for each oracle
For each oracle, do a real lookup and verify it returns a value in
`[0, 1]`:

```python
norm = get_pertrack_normalizer("alphagenome")
# Pick any track from norm._track_index for that oracle
# (track_id format is what the oracle returns from list_tracks)

# Effect percentile: how unusual is a +0.5 log2FC effect?
eff = norm.effect_percentile(
    "alphagenome",
    "DNASE/EFO:0001187 DNase-seq/.",   # HepG2 DNase
    raw_score=0.5,
    signed=False,
)
assert eff is not None and 0.0 <= eff <= 1.0, f"bad effect %ile: {eff}"

# Activity percentile: how active is a 500-unit signal?
act = norm.activity_percentile(
    "alphagenome",
    "DNASE/EFO:0001187 DNase-seq/.",
    raw_signal=500.0,
)
assert act is not None and 0.0 <= act <= 1.0, f"bad activity %ile: {act}"
print(f"alphagenome: effect=%ile {eff:.3f}, activity %ile {act:.3f}")
```

Run the equivalent for each of the six oracles (use a representative
track ID per oracle — list with `oracle.get_all_assay_ids()[:5]` if
unsure). **Flag any oracle where percentiles come back as `None`.**

---

# Phase 3 — Full test suite (NOTHING SKIPPED)

```bash
# Clear any cached state first
rm -rf .pytest_cache tests/__pycache__ chorus/**/__pycache__

# The full suite includes 6 smoke tests that load every oracle model
# and run a real predict() — expect ~25 min on a single GPU.
mamba run -n chorus pytest tests/ -v 2>&1 | tee /tmp/full_test.log
```

The expected outcome is **286 passed, 0 failed**. If anything fails,
include the test name + error in the final audit report.

---

# Phase 4 — Notebooks (run end-to-end with outputs)

```bash
mkdir -p /tmp/audit_nb
for nb in single_oracle_quickstart comprehensive_oracle_showcase advanced_multi_oracle_analysis; do
  echo "=== Running $nb ==="
  mamba run -n chorus jupyter nbconvert --execute --to notebook \
    --ExecutePreprocessor.timeout=900 \
    --output /tmp/audit_nb/${nb}_executed.ipynb \
    examples/$nb.ipynb 2>&1 | tee /tmp/audit_nb/${nb}.log
done
```

For each notebook, after execution count: total cells, cells with output,
cells with errors. A successful notebook has 0 error cells.

```python
import json
for nb in ["single_oracle_quickstart", "comprehensive_oracle_showcase",
            "advanced_multi_oracle_analysis"]:
    with open(f"/tmp/audit_nb/{nb}_executed.ipynb") as f:
        data = json.load(f)
    code = [c for c in data["cells"] if c.get("cell_type") == "code"]
    with_out = [c for c in code if c.get("outputs")]
    errs = [c for c in code if any(o.get("output_type") == "error"
                                    for o in c.get("outputs", []))]
    print(f"{nb}: {len(with_out)}/{len(code)} cells with output, {len(errs)} errors")
```

---

# Phase 5 — MCP server smoke test

## 5.1 Server starts
```bash
# In one shell:
mamba run -n chorus chorus-mcp &
MCP_PID=$!
sleep 5
ps -p $MCP_PID && echo "MCP started OK" || echo "MCP FAILED TO START"
kill $MCP_PID 2>/dev/null
```

## 5.2 Set up Claude Code MCP integration
```bash
claude mcp add chorus -- mamba run -n chorus chorus-mcp
```

Then in a Claude Code session (`claude` from the chorus repo dir),
verify the tools are visible:

> List the available Chorus MCP tools.

Expected: 24 tools including `analyze_variant_multilayer`,
`discover_variant_cell_types`, `score_variant_batch`,
`fine_map_causal_variant`, `analyze_region_swap`, `simulate_integration`.

## 5.3 Test EVERY application tool with a real prompt

Copy each prompt below into Claude Code in turn. For each, verify:
- The tool returns a non-error result
- A report file is written under the working directory
- The report has an `Analysis Request` block with the prompt at the top
- For analyses that use a normalizer, percentiles are populated (not None)

### Test prompts (one per application):

**(a) Variant analysis — should reproduce CEBPA binding gain in HepG2:**
> Load AlphaGenome and analyze rs12740374 (chr1:109274968 G>T) in HepG2
> liver cells. Use DNASE, CEBPA ChIP, CEBPB ChIP, H3K27ac, and CAGE
> tracks. Gene is SORT1.

Expected: report shows CEBPA + CEBPB with **strong binding gain**
(percentile ≈ 1.0). Save the produced HTML and confirm `igv` loads.

**(b) Discovery — find which cell types are affected:**
> Discover which cell types are most affected by rs12740374
> (chr1:109274968 G>T) using AlphaGenome.

Expected: ranking of top 5 cell types with effect sizes.

**(c) Batch scoring — five SORT1-locus SNPs across HepG2:**
> Score these 5 variants in HepG2 with AlphaGenome and rank by effect:
> rs12740374 chr1:109274968 G>T
> rs1626484 chr1:109275684 G>T
> rs660240 chr1:109275216 T>C
> rs4970836 chr1:109279175 G>A
> rs7528419 chr1:109274570 A>G
> Use DNASE, CEBPA, CEBPB, H3K27ac, and CAGE tracks. Gene is SORT1.

Expected: per-track table with one column per assay, percentiles in
each cell, rs12740374 as the top hit.

**(d) Causal prioritization with LD auto-fetch — confirms LDLINK_TOKEN works:**
> Fine-map the SORT1 LDL cholesterol GWAS locus. Lead variant is
> rs12740374. Auto-fetch LD proxies from LDlink (population CEU,
> r²≥0.85). Score each variant in HepG2 with DNASE, CEBPA, CEBPB,
> H3K27ac, and CAGE. Gene is SORT1.

Expected: ranked table where rs12740374 is the top causal variant
(composite ≈ 0.95). **If LDlink fails the fetch, the LDLINK_TOKEN env
var is not reaching the MCP subprocess** — flag it.

**(e) Region swap — sequence engineering:**
> Replace chr1:109274500-109275500 with a strong K562 promoter sequence
> and predict effects in K562 using DNASE, H3K27ac, H3K4me3, and CAGE.

Expected: report titled **"Region Swap Analysis Report"** (not "Multi-
Layer Variant Effect Report"), with a `Modified region: chr1:109,274,501-
109,275,500 (1,000 bp)` line and an IGV browser highlighting the
1000 bp region in red.

**(f) Integration simulation:**
> Simulate inserting a 366 bp CMV promoter construct at chr19:55115000
> and predict local disruption in K562.

Expected: report titled **"Integration Simulation Report"** with a
`Modification: Inserted N bp construct ...` line.

---

# Phase 6 — Application examples folder (every file)

The repo ships 13 worked examples. For each, verify:
- `example_output.md` exists and starts with an `## Analysis Request` block
- `example_output.json`, `example_output.tsv`, and at least one `*.html` exist
- The HTML's IGV browser loads (verify with selenium below)

```bash
for d in examples/applications/variant_analysis/*/ \
         examples/applications/validation/*/ \
         examples/applications/discovery/*/ \
         examples/applications/causal_prioritization/*/ \
         examples/applications/batch_scoring/ \
         examples/applications/sequence_engineering/*/ ; do
  [ -d "$d" ] || continue
  md="$d/example_output.md"
  has_md=$( [ -f "$md" ] && echo 1 || echo 0 )
  has_json=$( [ -f "$d/example_output.json" ] && echo 1 || echo 0 )
  has_tsv=$( [ -f "$d/example_output.tsv" ] && echo 1 || echo 0 )
  n_html=$(ls "$d"*.html 2>/dev/null | wc -l)
  has_prompt=$( [ -f "$md" ] && grep -c "^>" "$md" || echo 0 )
  echo "$d  md=$has_md json=$has_json tsv=$has_tsv html=$n_html prompt=$has_prompt"
done
```

## Selenium IGV verification

Verify every HTML report renders with a working IGV browser:

```bash
mamba run -n chorus python << 'PYEOF'
import os, glob, time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

results = []
for hp in sorted(glob.glob("examples/applications/**/*.html", recursive=True)):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1400,800")
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(f"file://{os.path.abspath(hp)}")
        time.sleep(8)
        igv = driver.execute_script("return typeof igv !== 'undefined'")
        ar = len(driver.find_elements(By.CSS_SELECTOR, ".analysis-request"))
        title = driver.find_element(By.TAG_NAME, "h1").text
        errs = [l for l in driver.get_log("browser") if l.get("level") == "SEVERE"]
        ok = "✅" if igv and ar and not errs else "⚠️"
        results.append((ok, hp, igv, ar, len(errs), title[:50]))
        print(f"{ok} {hp[-70:]:70s} igv={igv} AR={ar} errs={len(errs)} title='{title[:40]}'")
    finally:
        driver.quit()

# Save screenshots of three key examples for visual review
for label, hp in [
    ("sort1", "examples/applications/variant_analysis/SORT1_rs12740374/rs12740374_SORT1_alphagenome_report.html"),
    ("batch", "examples/applications/batch_scoring/batch_sort1_locus_scoring.html"),
    ("region_swap", "examples/applications/sequence_engineering/region_swap/region_swap_SORT1_K562_report.html"),
]:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1400,2000")
    driver = webdriver.Chrome(options=options)
    driver.get(f"file://{os.path.abspath(hp)}")
    time.sleep(12)
    driver.save_screenshot(f"/tmp/audit_{label}.png")
    driver.quit()
    print(f"saved /tmp/audit_{label}.png")
PYEOF
```

For each report, the row should show **igv=True AR=1 errs=0**. Anything
else is a bug — flag the file in the audit report.

---

# Phase 7 — Reproduce a background distribution from scratch (sanity check)

You don't have to wait the full 4–22 hours. Run a 10-position smoke
build for one oracle to confirm the build pipeline works:

```bash
# ChromBPNet is the fastest (per-model, ~25 min on full settings)
mamba run -n chorus-chrombpnet python scripts/build_backgrounds_chrombpnet.py \
    --part variants --gpu 0 --n-variants 10 --n-random 10 --n-ccre 10 \
    --n-tss 5 --n-gene-body 5 --reservoir-size 100 --n-cdf-points 100 2>&1 | tail -20
```

Verify a `*_interim_variants.npz` file shows up in `~/.chorus/backgrounds/`.

---

# Phase 8 — Documentation walkthrough as a new user

Open the README in a browser-like view (or read it sequentially). For
each link, follow it and verify:

- `examples/applications/` — the README there is helpful and the tool
  table accurately picks the right example for each user role
- `docs/MCP_WALKTHROUGH.md` — every prompt example is reproducible
- `docs/variant_analysis_framework.md` — the 5-layer scoring rules match
  the actual `score_track_effect` formulas
- `docs/API_DOCUMENTATION.md` — the application-layer section
  (`build_variant_report`, `score_variant_batch`,
  `prioritize_causal_variants`, `AnalysisRequest`) function signatures
  match the actual code

Test the **Further reading** table at the bottom of the main README — every
linked file should exist and be useful.

---

# Phase 9 — Final audit report

Write a markdown report at `/tmp/CHORUS_AUDIT_REPORT.md` with this
structure:

```markdown
# Chorus Audit Report — <date>

## Phase 1: Installation
- Time taken: <minutes>
- Disk used: <GB>
- Issues encountered: <list or "none">

## Phase 2: Backgrounds
- All 6 oracles auto-downloaded: <yes/no>
- All percentile lookups returned valid [0,1] values: <yes/no>
- Issues: <list>

## Phase 3: Test suite
- Result: <X passed, Y failed>
- Failed tests: <list with error summary>

## Phase 4: Notebooks
- single_oracle_quickstart: <pass/fail>, <N cells with output>, <M errors>
- comprehensive_oracle_showcase: <pass/fail>, ...
- advanced_multi_oracle_analysis: <pass/fail>, ...

## Phase 5: MCP tools (per prompt)
- (a) SORT1 variant analysis: <pass/fail> + key biology check
       (CEBPA strong binding gain present?)
- (b) Discovery: <pass/fail>
- (c) Batch scoring: <pass/fail> + per-track table verified?
- (d) Causal with LD auto-fetch: <pass/fail> + LDlink connectivity
- (e) Region swap: <pass/fail> + correct title + IGV region marker
- (f) Integration: <pass/fail> + correct title + sequence documented

## Phase 6: Application examples
- Total examples: 13
- HTML reports verified: <X/13 igv=True>
- Any with prompt missing in HTML: <list>

## Phase 7: Background reproduction
- Smoke build pipeline works: <yes/no>

## Phase 8: Documentation
- Broken links: <list or "none">
- Stale info: <list or "none">

## Top issues a real new user would hit (ranked)
1. <issue with severity + concrete fix>
2. ...

## Verdict
- Production ready: <yes/no/with-caveats>
- One-line summary
```

---

## Hard rules

- **Skip nothing.** If something looks like it works, prove it works.
- **Don't paper over failures.** If a step fails, log it, continue,
  and report it. Do not silently work around problems.
- **Use the exact commands and prompts in this document.** Don't
  paraphrase or improvise.
- **Report what you actually observed**, not what you expected. If
  CEBPA is *not* the top TF in the SORT1 analysis, say that and
  investigate whether it's a Chorus bug or a model limitation.
- **Don't push anything**. This is a read-only audit; do not commit or
  push to the repo.
- The audit report at `/tmp/CHORUS_AUDIT_REPORT.md` is the single
  deliverable. Make it complete enough that I can read it without ever
  looking at the test logs.

When everything is done, print the final report to stdout and stop.
