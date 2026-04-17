# V7 Scorched-Earth Fresh-Install Audit — 2026-04-17

**Machine**: Linux x86_64, 2× NVIDIA A100 80 GB PCIe, CUDA available
**Branch**: `chorus-applications` @ a383bbf (post-v7 PR merges)
**Methodology**: Full scorched-earth — every chorus* env deleted, `~/.chorus` + `~/.cache/huggingface` wiped, fresh clone to SSD (`/srv/local/lp698/chorus-audit-v7`), every README step followed verbatim. **No cached anything.**
**Duration**: ~8 hours (env creation dominates)

---

## Results: ALL GREEN

| Check | Result |
|-------|--------|
| Base env + pip install + kernel + HF auth + hg38 download | ✓ per README |
| 6 oracle envs via `chorus setup` (enformer/alphagenome/chrombpnet/borzoi/sei/legnet) | ✓ all built from scratch |
| **pytest: 301 code tests + 6 oracle smoke tests** | **307/307 pass** |
| 3 notebooks (quickstart + comprehensive + advanced) executed end-to-end | **0 errors across 235 cells** |
| All 13 application examples regenerated (alphagenome + enformer + chrombpnet) | ✓ |
| **Selenium sweep of all 16 HTML reports** | **16/16 CLEAN** (0 JS errors, 0 badge mismatches, 0 generic CHIP names, 0 orphan HTMLs) |

---

## Timeline

| Step | Time |
|------|------|
| Delete 7 envs + caches | 5 min |
| Clone + base env (mamba env create) | 30 min |
| pip install -e . + kernel + HF auth + hg38 download | 10 min |
| Merge 3 in-flight PRs (v6 discovery + v7 audit + v7 UX polish) | 2 min |
| 6 oracle envs (chained serial) | ~4 hours |
| Full pytest (301 code tests) | 53 min |
| Smoke tests retry on GPU 0 (GPU 1 was hijacked) | 7 min |
| Quickstart notebook | 30 min |
| Comprehensive + Advanced notebooks (parallel, one retry) | 50 min |
| All 13 examples regen (chained) | 60 min |
| Selenium 16 HTML sweep | 3 min |

---

## Merged during audit

Three in-flight PRs from the collaborator were all legitimate fixes and got merged:

- **PR #15** (v6 discovery orphan HTMLs): Made `discover_variant_effects`/`discover_and_report` accept `analysis_request`/`user_prompt` kwargs so the user prompt is baked into HTML on first write. Re-committed 3 per-cell-type discovery HTMLs + 1 enformer validation HTML that I had incorrectly deleted as "duplicates" in df7d613. These are the actual primary outputs of `discover_variant_cell_types`, not duplicates.
- **PR #16** (v7 first-user UX audit): Read-only audit report.
- **PR #17** (v7 UX polish): 4 drift fixes — `chorus list` phantom `base` oracle, wrong `alt_allele="T"` in walkthrough, stale notebook subtitle, stale SORT1 README percentile gradient.

---

## HTML report quality (16/16 CLEAN)

Every report verified to have:
- **Analysis Request** section with user prompt blockquote
- **Correct badge colors** matching interpretation labels (green "Minimal", yellow "Moderate", red "Very strong" — no inverted red "Minimal effect")
- **Enriched CHIP track names** (`CHIP:CEBPA:HepG2`, not generic `CHIP:HepG2`)
- **Self-contained igv.min.js** inlined (no CDN dependency, works on airgapped/corporate-proxy networks)
- **≥99th percentile display** for saturated values (no misleading "1.000" precision)
- **HTML `<title>`** with report_title + gene + position (e.g. "Multi-Layer Variant Report — SORT1 — chr1:109274968")
- **Per-track Ref/Alt/log2FC/Effect %ile** columns in batch scoring
- **Interpretation section** with clinical/biological narrative in flagship variant examples (SORT1, BCL11A, FTO)

---

## Transient issues encountered (not blockers)

1. **GPU 1 hijacked mid-audit**: Another user grabbed all 80 GB of GPU 1 during pytest. First smoke test run failed (4 oracles). Retried on GPU 0 → all 6/6 pass. Same thing later for the advanced notebook — retry on GPU 0 worked. Users on shared machines need to be aware that `--gpu` is not automatic across oracles.

---

## Verdict

**Production-ready on Linux CUDA.** The README install path works verbatim from a cold start, all 6 oracles train + predict correctly, tests pass, notebooks execute cleanly, every example reproduces with expected outputs, every HTML report renders without JS errors or visual bugs.

This is the 7th consecutive audit pass with no BLOCKS_USER findings. Nothing left to fix that I can see from a cold scorched-earth pass.
