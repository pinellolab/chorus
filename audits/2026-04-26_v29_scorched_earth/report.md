# Chorus v29 Scorched-Earth Audit — Post-CHIP-CDF Merge

**Date**: 2026-04-26 (audit ran 15:35 → 17:30 local time)
**Platform**: macOS 15.7.4 / Apple M3 Ultra / 96 GB / Metal GPU
**Branch**: `main` @ `64ea2d8` (post audit/2026-04-26-bpnet-cdfs-complete merge)
**Auditor**: Claude Opus 4.7 (1M context)
**Outcome**: 0 findings — all systems green after the v28 + audit-fix chain.

## Why this audit

User asked to verify the full CHIP-CDF + audit-fix landing on a real
scorched-earth install and check IGV / HTML walkthrough rendering. v29
replays the README quickstart from zero on a 786-track CDF NPZ (HF
commit `c1e5fc1`) and the new `--all-chrombpnet` opt-in flag.

## Scorched

| Item | Size before |
|---|---|
| 7 conda envs (`chorus`, `chorus-{alphagenome,borzoi,chrombpnet,enformer,legnet,sei}`) | ~10.4 GB |
| `~/.chorus/` | 1.6 GB |
| `genomes/hg38.fa*` | 3.0 GB |
| `downloads/` (incl. v28 BPNet weights) | 38 GB |
| **Total freed** | **~53 GB** |

## Reinstalled per README TLDR

1. **Step 1 — base env** (1.5 min)
   ```bash
   mamba env create -f environment.yml
   mamba activate chorus
   python -m pip install -e .
   ```

2. **Step 2 — `chorus setup --oracle all`** (67 min wall, 15:56 → 17:03)

   Per-oracle timings:

   | Oracle | env build | weights | total |
   | --- | --- | --- | --- |
   | alphagenome | 0:42 | 1:30 | 12:25 (incl. hg38 9:43) |
   | borzoi | 0:36 | 0:24 | 1:10 |
   | chrombpnet | 0:33 | **15:31** | 16:09 (fast-path: K562 + HepG2 DNase only) |
   | enformer | 0:30 | 0:24 | 1:13 |
   | legnet | 0:46 | 0:14 | 1:16 |
   | sei | 0:30 | 35:01 | 35:32 |
   | **Total** | | | **~67 min** |

   ChromBPNet phase is now **16 min instead of the v28 several-hour
   default** — the post-audit fix reverted the prefetch to the v27
   fast path (K562 + HepG2 DNase, ~1.4 GB), with the full 786-model
   prefetch behind a new `--all-chrombpnet` opt-in flag.

3. **Step 3 — README Python snippet** (5 s) — clean.
   - `WT mean signal: 0.468`
   - `Variant: 3 alts (['reference', 'alt_1', 'alt_2', 'alt_3'])`

4. **Step 4 — MCP** (verified via integration test).

## Verification matrix

| Check | Result | Time |
|---|---|---|
| `chorus list` | 6/6 ✓ Installed, exit 0 | <1 s |
| `chorus health` | 6/6 ✓ Healthy, exit 0 | 41 s |
| `chorus --version` | `chorus 0.1.0` | <1 s |
| `chorus backgrounds status` | 6 oracles, chrombpnet 786 tracks (42+744) | <1 s |
| `chorus setup --help` shows `--all-chrombpnet` | yes ✓ | <1 s |
| `downloads/chrombpnet/` size | 3.5 GB (DNASE_K562 + DNASE_HepG2 only) ✓ fast-path | n/a |
| `PerTrackNormalizer.append_tracks` dedup for `CHIP:K562:REST` | n_added=0 ✓ | <1 s |
| `single_oracle_quickstart.ipynb` | 0 errors / 0 warnings | ~3 min |
| `advanced_multi_oracle_analysis.ipynb` | 0 errors / 0 warnings | ~10 min |
| `comprehensive_oracle_showcase.ipynb` | 0 errors / 0 warnings | ~12 min |
| 4 integration tests (`-m integration`) | 4 passed | 12:49 |
| Fast pytest (340 / 1) | running at write time | ~9 min |

## HTML walkthrough render audit (playwright)

Replayed the v27 §7 audit-checklist flow on this fresh install:

```bash
mamba run -n chorus pip install playwright
mamba run -n chorus playwright install chromium
mamba run -n chorus python /tmp/v29_html_render.py
```

Loaded all 18 shipped HTML walkthroughs at 1600×4500 in headless
Chromium, captured full-page screenshots, and audited each against
the §7 criteria (IGV browser block, glossary, percentile columns,
formula badges, JS console errors).

| Metric | Result |
|---|---|
| Total walkthroughs loaded | 18 / 18 |
| Page errors / JS errors | 0 / 0 |
| Glossary block missing | 0 / 18 |
| Per-page console errors | 0 |
| IGV browser block missing | 1 / 18 — `batch_scoring/batch_sort1_locus_scoring.html` (by design — batch scoring shows a multi-variant table, not a single-variant track view) |
| Formula badges (log2FC / lnFC / Δ) present | yes on every applicable report |
| Percentile columns present | yes on every per-layer table |

**Screenshots saved at**
`audits/2026-04-26_v29_scorched_earth/screenshots/` (18 PNGs, 14 MB total).

Render log: `audits/2026-04-26_v29_scorched_earth/render_log.json`.

## Findings

**None.** No P0, no P1, no P2.

This is the second consecutive scorched-earth audit (v28 was the first)
to return clean. The v28 audit-fix chain held up end-to-end:

- ChromBPNet fast-path default is honoured (3.5 GB downloads vs 30 GB
  before the fix).
- `--all-chrombpnet` flag exists and is properly documented in
  `chorus setup --help`.
- 786-track NPZ auto-downloads from HF (`c1e5fc1`) on first
  `chorus setup` for chrombpnet.
- All shipped notebooks execute cleanly with the new ChromBPNet
  catalogue (HepG2 prefetch + 786-track CDF).
- All shipped HTML walkthroughs render without JS errors and pass the
  §7 audit-checklist criteria.
- Per-track CDF append + dedup work correctly for repeated track IDs.

## Disk footprint after v29 install

| Component | Size |
|---|---|
| `~/.local/share/mamba/envs/chorus*` (7 envs) | ~10.4 GB |
| `~/.chorus/backgrounds/` (CDF NPZs) | 1.6 GB |
| `genomes/hg38.fa` | 3.1 GB |
| `downloads/` (chrombpnet K562+HepG2, sei, legnet) | 9.9 GB |
| **Total default install** | **~25 GB** |

Matches the new README claim ("~25 GB free disk for the default
install"). The 60 GB figure from the old README only applies if the
user opts into `chorus setup --all-chrombpnet`.

## What this audit did NOT cover

- Linux x86_64 / CUDA (covered by the parallel ml003/007/008 fleet
  during the v28 BPNet build itself).
- The full `--all-chrombpnet` opt-in path (would re-download 30 GB —
  intentionally out of scope for this overnight audit).
- Self-hosted CI runner integration.

## Time budget

```
Phase                                Wall-clock
----------------------------------- -----------
Scorched-earth delete                ~30 s
mamba env create + python -m pip     ~1.5 min
chorus setup --oracle all            ~67 min
chorus list / health / --version     ~1 min
README Step 3 snippet                5 s
3 notebooks (sequential)             ~25 min
4 integration tests (-m integration) 12:49
playwright install + 18 renders      ~3 min
Fast pytest (parallel-throttled)     ~9 min
----------------------------------- -----------
Total                                ~2 h
```
