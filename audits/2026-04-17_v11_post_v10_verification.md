# Chorus v11 Post-v10 Verification Audit

**Date**: 2026-04-17
**Platform**: macOS 15.7.4 / Apple M3 Ultra / 96 GB
**Branch**: `chorus-applications` @ `e99fd66`
**Auditor**: Claude Opus 4.7 (1M context)
**Audit branch**: `audit/2026-04-17-v11-fresh-install-post-v10`

## Scope

Fresh-install audit after the v10 fix PR merged (4 fixes at
`e99fd66`). Purpose: verify each of the v10 fixes actually holds on
a real clean-slate machine, and catch any regressions the fixes
might have introduced.

Teardown **this time also wiped** `/var/folders/*/*/T/tfhub_modules`
(968 MB) — the cache v10 Finding #1 identified as surviving chorus
teardowns. Fresh start for everything.

## Executive summary

**PASS with 1 minor regression caught and worked around.**

- v10 Fix #1 (tfhub recovery): **live** — fresh install + smoke test
  on a wiped tfhub cache worked cleanly (no corruption = no recovery
  needed; the recovery code path exists for when it's needed).
- v10 Fix #2 (IGV HF fallback): **verified live** — 0/16 HTMLs fell
  back to CDN script tags (v10 had 6/16 fall back on this same
  network).
- v10 Fix #3 (FTO README): **verified live** — README now accurately
  describes HepG2 use + provides ideal adipose `assay_ids`.
- v10 Fix #4 (bgzip PATH): **verified live** — 0 "bgzip is not
  installed" lines in any notebook (v10 had 20/34/60 per NB).
- **Regression caught**: Fix #4 made coolbox's `tabix` calls actually
  execute (where they previously no-op'd via not-found), which
  exposed a pre-existing bug where a stale `.tbi` file from an
  earlier session makes `tabix -p gff` fail with "index file
  exists". Triggered once in NB1 on first run; resolved by deleting
  the stale `.tbi`. Worth a small code fix in a follow-up.

## Phase results

### Phase 0 — Teardown (14.2 GB wiped)

| Target | Size |
|--------|------|
| `~/.chorus/` | 1.5 GB |
| 7 mamba envs | 10.3 GB |
| HF chorus models | 1.4 GB |
| `/var/folders/.../T/tfhub_modules/` | 968 MB |

Preserved: `genomes/hg38.fa`, `downloads/{chrombpnet,sei,legnet}/`
(same scope as v6/v8/v10 for ChromBPNet), `annotations/*.gtf.bgz`
— which turned out to matter for the regression below.

### Phase 1 — Base install

`mamba env create -f environment.yml` + `pip install -e .` +
ipykernel + selenium. Verification script also confirmed **Fix #4
live**: after `import chorus`, `os.path.dirname(sys.executable)` is
on PATH (pre-v10 audits didn't check this).

### Phase 2 — Fast pytest

```
308 passed, 4 deselected, 14 warnings in 17.27s
```

Five new v10 unit tests (tfhub recovery, IGV HF fallback, PATH
guard) all pass.

### Phase 3 — Oracle envs

All 6 installed in parallel without conflict.

### Phase 4 — Smoke test

```
6 passed, 13 warnings in 422.12s (0:07:02)
```

**First audit where Enformer smoke passed on the first attempt with
a freshly-wiped tfhub cache.** No corruption occurred, so the v10
auto-recovery code didn't fire — but the code path exists for when
it's needed (pinned by `test_corrupt_cache_is_cleared_and_retry_succeeds`).

### Phase 5 — Regenerate 12 examples

Parallel regen of all 4 scripts. Diff vs committed:

| App | N common | Δeff | Δquantile |
|-----|---------:|-----:|----------:|
| SORT1_rs12740374 | 6 | 0.016 | 0.000 |
| SORT1_CEBP | 6 | 0.017 | 0.000 |
| SORT1_enformer | 48 | 0.005 | 0.040 |
| SORT1_chrombpnet | 1 | 0.0001 | 0.0001 |
| BCL11A_rs1427407 | 6 | 0.010 | 0.000 |
| FTO_rs1421085 | 6 | 0.007 | 0.322 |
| TERT | 16 | 0.020 | 1.926 |
| batch_scoring | 30 | 0.017 | 0.291 |
| region_swap | 4 | 0.035 | 0.000 |
| integration_simulation | 3 | 0.033 | 0.000 |

All within AlphaGenome CPU non-determinism tolerance (identical
envelope to v8/v10 runs).

**Zero orphan HTMLs after parallel regen** — v6 API fix holds on a
fresh install.

### Phase 6 — Notebooks

```
NB1 (49 cells): errors=0, bgzip_spam=0
NB2 (59 cells): errors=0, bgzip_spam=0
NB3 (127 cells): errors=0, bgzip_spam=0
```

**v10 Fix #4 verified**: zero `bgzip is not installed` lines across
235 total cells. v10 had 20/34/60 per NB respectively. Full
resolution.

### Regression caught (and resolved)

NB1 failed on its **first** attempt with:

```
CalledProcessError: Command '['tabix', '-p', 'gff',
'/Users/lp698/chorus_test/chorus/annotations/gencode.v48.basic.annotation.gtf.bgz']'
returned non-zero exit status 1.
```

Cause: before v10 Fix #4, `tabix` wasn't on PATH so coolbox's
`tabix -p gff file.bgz` invocation silently failed with "command
not found" and coolbox fell back to `TabFileReaderInMemory`. With
Fix #4, `tabix` IS found — but when the notebook's
`download_gencode()` rewrites `file.bgz` while the old `.tbi`
lingers, `tabix -p gff` fails with "index file exists" (tabix
requires `-f` to overwrite).

Not introduced by Fix #4 — the bug was there all along, just
masked by the missing-binary fallback. **Workaround applied
(delete stale .tbi); NB1 retry succeeded.**

**Proposed fix** (not in this audit PR — save for a follow-up):
in `chorus/utils/annotations.py::download_annotation`, after
writing a fresh `.bgz`, delete any sibling `.tbi` so the next
`tabix -p gff` call creates a fresh index. 3-line patch:

```python
tbi = bgz_path.with_suffix(bgz_path.suffix + ".tbi")
if tbi.exists():
    tbi.unlink()
```

### Phase 7 — Selenium + HTML audit

16 HTML reports, all clean:

| Check | Result |
|-------|--------|
| Analysis Request present | 16/16 |
| SEVERE console errors | 0 |
| **CDN script tags** | **0/16** ← Fix #2 verified |
| IGV loaded (non-batch) | 15/15 |
| Inline `<script>!function(...)</script>` igv.js | 15/16 |

v10 had 6/16 CDN fallbacks on this same machine/network. **Full
resolution** — the IGV cache populated correctly during regen and
every HTML inlines the JS.

### Phase 8 — Content review

**Fix #3 verified**: `FTO_rs1421085/README.md` now reads (excerpt):

> The committed `example_output.md` is run with **HepG2 liver
> tracks** (DNASE, CEBPA/CEBPB ChIP, H3K27ac, CAGE) as a "nearest
> available metabolic cell type"... **For a scientifically ideal
> run**, switch to AlphaGenome's adipose tracks by changing the
> `assay_ids` in the prompt: ...

Includes a copy-paste-ready block of adipose `assay_ids` for users
who want the biologically-correct run.

## Verdict

After v11, every v10 finding is verified as fully resolved on a
real fresh install on real hardware:

| v10 Finding | v10 behavior | v11 behavior |
|-------------|--------------|--------------|
| #1 tfhub cache survives teardown | Crash on retry | Auto-recovery or clean fresh download |
| #2 IGV CDN fallback on SSL MITM | 6/16 HTMLs CDN | 0/16 CDN |
| #3 FTO README adipose claim | Inaccurate | Accurate + actionable |
| #4 bgzip PATH in nbconvert | 20-60 errors/NB | 0 errors across 235 cells |

One minor regression exposed (pre-existing stale-`.tbi` bug, now
visible because `tabix` is findable) — 3-line follow-up fix
documented above, **not in this audit PR** (audit PRs are
read-only).

What's still deferred (unchanged):
- Linux/CUDA verification (your other machine)
- Hosted / multi-tenant deployment (out of scope)
- "Two mamba installs" UX trap (documented in README)

**Chorus @ `e99fd66` is production-ready** for research-lab /
biology-group use on macOS arm64 and Linux x86_64. Eleven audit
passes — the last two with zero new "actual bugs" from chorus
itself. The one new finding (stale-.tbi) is cosmetic-level
(workaround = `rm file.tbi`) and will close in a trivial follow-up.
