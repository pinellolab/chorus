# Overnight scorched-earth audit — 2026-04-29 → 2026-04-30

**Started**: 2026-04-29 22:57 EDT
**Operator**: Claude (overnight, autonomous, user explicitly said "don't skip anything")
**Mode**: TRUE scorched-earth — delete all 7 chorus oracle envs + the base `chorus` env + reinstall from zero following README, then exercise everything.

## What gets blown away

- All 7 oracle envs: `chorus-{enformer,borzoi,chrombpnet,sei,legnet,alphagenome,alphagenome_pt}`
- The base `chorus` env (after the oracle removes complete; my session needs it to issue the deletes, so it's destroyed last)
- **NOT touched**: `~/.cache/huggingface/hub/` (warm caches stay — install times reflect a returning user, not a literal first-time download. Documented as a caveat in the final report.)
- **NOT touched**: `~/.local/share/mamba/envs/<other>` (e.g. miniforge itself)
- **NOT touched**: `~/.chorus/backgrounds/` (CDFs auto-download on first use; clearing them only delays the new-user-flow probe)

## Safety rails

1. **Sequential env removal**: I delete env N, recreate env N, smoke-load env N. Only after N passes do I move to N+1. If env N fails to rebuild, I halt — user's other envs are still working.
2. **Base env destroyed last**: I issue all 7 oracle removes first (using base chorus CLI), then destroy base, then rebuild base from `environment.yml` per README, then re-issue `chorus setup` for each oracle.
3. **Logging at every step**: `audits/2026-04-30_overnight_full_audit/logs/<step>.log`.
4. **Halt-on-failure**: any step's exit code != 0 stops the chain. Manual cleanup is the user's morning task at that point — they still have the upstream HF mirrors ready.
5. **Backup of env names + sizes pre-deletion** to `phase_outputs/A/preflight.txt`.

## Phases (each phase writes to `phase_outputs/<phase>/`)

- **A — Static + repo consistency** (read-only): fast suite, integration imports, doc grep, git status. Already running in background.
- **B — Pre-deletion preflight**: snapshot every env's status, weight cache sizes, current commit. Rolling back from this is messy but at least documented.
- **C — Tear down** the 7 oracle envs + base.
- **D — Rebuild base** from `environment.yml` per README.
- **E — `chorus setup`**: rebuild all 7 oracles via the user-facing CLI flow (HF gate, prefetch, marker write).
- **F — `chorus health`** for every oracle.
- **G — Smoke tests**: per-oracle load + 1 prediction + finite-value check.
- **H — Notebooks**: quickstart (CPU), advanced + comprehensive (Metal where available).
- **I — Walkthroughs**: variant_analysis SORT1, cell-type screen, multi-oracle.
- **J — MCP server smoke**: list_oracles, list_tracks, recommend_alphagenome_backend, simple variant analysis.
- **K — README walk** as a new user: every code block in README's "Get running in one lunch break" section.
- **L — Final report**: `report.md` with status, P0/P1/P2 findings, sign-off for tomorrow's CDF upload.

## Time budget

Realistic: 6–10 h overnight. Most steps are I/O-bound; HF caches are warm so weight downloads will be fast.

## Status (live-updated)

- Phase A (static): in flight
- Phase B (preflight): pending
- Phase C (teardown): pending
- Phase D (rebuild base): pending
- Phase E (chorus setup): pending
- Phase F (health): pending
- Phase G (smoke): pending
- Phase H (notebooks): pending
- Phase I (walkthroughs): pending
- Phase J (MCP): pending
- Phase K (README walk): pending
- Phase L (report): pending
