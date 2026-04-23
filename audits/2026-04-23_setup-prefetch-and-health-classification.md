# 2026-04-23 — Setup prefetch + health classification + token flow

Author: Claude (session driven by Luca)
Scope: the consolidated change landing
`chorus/core/weights_probe.py`, `chorus/cli/_setup_prefetch.py`,
`chorus/cli/_setup_all.py`, `chorus/cli/_tokens.py`, plus
modifications to `chorus/cli/main.py`, `chorus/core/environment/runner.py`,
`chorus/__init__.py`, `chorus/utils/http.py`, `chorus/utils/ld.py`,
`chorus/oracles/legnet.py`, `chorus/oracles/sei.py`, and `README.md`.

## What was run

Sections of [`AUDIT_CHECKLIST.md`](AUDIT_CHECKLIST.md) that could be
affected by the change were executed in full; sections that are
orthogonal (§3 GPU, §4 CDF math, §6 notebooks, §7 HTML reports,
§8 MCP, §12/§13 reproducibility, §14 genomics edges, §15 offline,
§17 supply chain, §18 license) were **not** re-run — the change
doesn't touch those code paths.

## Summary

- 336 passed, 4 deselected (integration), 0 failed in the fast suite.
  `mamba run -n chorus python -m pytest tests/ --ignore=tests/test_smoke_predict.py -m "not integration" -q` → **pass**.
- `chorus health` on a machine with no setup markers: 6 oracles, 7.2 s
  total, each clearly reports "Not installed — run `chorus setup <oracle>`"
  with the exact missing artifacts. Previously Sei alone hung the
  120 s subprocess timeout.
- Health → Healthy transition verified end-to-end with a fabricated
  complete state (marker + artifacts).
- `chorus setup --oracle all` without an HF token + non-TTY stdin halts
  with rc=1 **before any env build** and emits the three token hints
  (`HF_TOKEN`, `--hf-token`, `huggingface-cli login`).
- `create_oracle('fakeOracle')` still raises `ValueError` naming the six
  valid options — the `kwargs.setdefault("use_environment", False)` change
  in `chorus/__init__.py` is a no-op for unknown oracles (the check
  runs first).
- Interactive `HF_TOKEN` prompt uses `getpass` (hidden); `LDLINK_TOKEN`
  prompt was switched from `input()` to `getpass` during the audit.

## Per-section findings

### §1 Installation & environment
- [x] **§1.3** `chorus --help` and every subcommand's `--help` render
      non-empty (setup: 20 lines, health: 8, genome: 12, etc).
- [x] **§1.6** Idempotency: `chorus setup --oracle enformer --no-weights
      --no-backgrounds --no-genome` on an already-present env returns
      exit 0 and does not rebuild.
- [x] **§1.9** `~/.chorus/backgrounds/` auto-download: verified by
      running `chorus setup --oracle enformer --no-weights --no-genome`
      which pulled `enformer_pertrack.npz` from HF in 21 s and wrote
      it to the canonical cache.
- [x] **New** Setup marker convention added: `downloads/<oracle>/.chorus_setup_v1`
      is the proof-of-install signal read by `chorus health` and
      written by `chorus setup` on success. Documented in
      `chorus/core/weights_probe.py` docstring.
- [x] **New P2, fixed during audit** `--force` now invalidates the
      stale marker up front so a mid-rebuild failure doesn't leave
      the oracle reporting Healthy (see `chorus/cli/main.py` and
      `chorus/cli/_setup_all.py`).

### §2 HuggingFace authentication
- [x] **§2.1** `HF_TOKEN` env path: verified — `whoami()` succeeds and
      we log the user name without exposing the token.
- [x] **§2.3** No-token, no-login path raises a single clear message
      that names `HF_TOKEN`, the exact gated repo URL
      (`huggingface.co/google/alphagenome-all-folds`), and the
      `huggingface-cli login` alternative. All three hints present in
      the AlphaGenome error and in the new `chorus setup` halt message.
- [x] **§2.4** Repo URL consistency: the string
      `huggingface.co/google/alphagenome-all-folds` appears in
      `chorus/oracles/alphagenome.py`, `chorus/oracles/alphagenome_source/templates/load_template.py`,
      `README.md` (three places including the new Tokens section), and
      the new `chorus/cli/_tokens.py`. No drift.

### §5 Python API sanity
- [x] **§5.1** `create_oracle('<name>', use_environment=False)` works
      for all 6 oracles (verified for legnet under `chorus-legnet` env).
      Invalid name raises `ValueError: Unknown oracle: fakeoracle.
      Available: ['enformer', 'borzoi', 'chrombpnet', 'sei', 'legnet',
      'alphagenome']`.
- [x] **New behaviour** `use_environment=False` now correctly
      propagates into the oracle instance via
      `kwargs.setdefault("use_environment", False)`. Previously the
      oracle would default to `use_environment=True` inside the
      "direct" branch, which made the `chorus setup` prefetch script
      re-spawn a subprocess back into the env it was already running
      in. Covered by a bespoke smoke test during the audit.
- [x] **§5.4** `predict_variant_effect` 1-based coordinate regression
      (`tests/test_prediction_methods.py::test_variant_position_is_1_based`)
      still passes (13/13 in test_prediction_methods.py).

### §9 Error messages
- [x] `create_oracle('fakeOracle')` names the six valid options.
- [x] AlphaGenome HF token missing → message contains `HF_TOKEN`,
      gated repo URL, and `huggingface-cli login`.
- [x] Network drop during `download_pertrack_backgrounds` returns 0
      and logs a warning (`tests/test_error_recovery.py::TestDownloadFailurePaths`
      2/2 pass).
- [x] `chorus setup --oracle all` halt message names `HF_TOKEN`,
      `--hf-token`, and `huggingface-cli login` — all three paths.
- [x] `test_missing_oracle_env_falls_back_gracefully` still passes.
- [x] `test_download_with_resume_leaves_partial_and_resumes_on_second_call`
      still passes after the tqdm integration.

### §10 Consistency of claims across the repo
- [x] Drift grep
      (`grep -rn '5,930\|5930\|196 kbp\|examples/applications/' --include='*.md' --include='*.py' .`
      excluding `audits/`) returns nothing.
- [x] No TODO/FIXME/WIP markers in any of the 12 changed/new files.
- [x] README "Tokens" section (new) names both tokens consistently
      with `chorus/cli/_tokens.py` resolution order. LDlink
      Troubleshooting block pre-existed and is now backed by the
      `LDLINK_TOKEN` env var + `~/.chorus/config.toml` fallback added
      to `chorus/utils/ld.py`.

### §11 Test suite
- [x] **Fast suite** `mamba run -n chorus python -m pytest tests/
      --ignore=tests/test_smoke_predict.py -m "not integration" -q`
      → 336 passed, 4 deselected, 0 failed, 72.7 s (threshold ≥334).
- [N/A] Integration suite not run (no release-host access in this
      session). Flagged for the release-host auditor.

### §16 Logging hygiene
- [x] `grep -rn 'hf_[a-zA-Z0-9]\{20,\}' chorus/ examples/ docs/ audits/`
      returns nothing — no real tokens committed.
- [x] All logger.info / logger.error calls in `chorus/cli/_tokens.py`
      log token **metadata** (source path, `whoami()` user name,
      success/rejection) but never the token value itself.
- [x] Interactive prompts use `getpass` (hidden stdin) for both HF and
      LDlink after a polish during the audit.

## Things deferred (not blocking this change)
- P1 §1.4 Running `chorus setup --oracle <X>` end-to-end on a fresh
  Linux/CUDA host and on macOS-arm64 — requires release hosts we don't
  have in this session.
- P1 §11 Integration-marked suite — same.
- P1 §8 MCP E2E for rs12740374 — no changes to MCP code; skipped.

## Files touched
```
 R  chorus/__init__.py                    (+5 lines)
 R  chorus/cli/main.py                    (+141 / -lines reorganized)
 R  chorus/core/environment/runner.py     (+19 lines, probe wire-up)
 R  chorus/oracles/legnet.py              (urlretrieve → download_with_resume)
 R  chorus/oracles/sei.py                 (lazy download + packaged-metadata fallback)
 R  chorus/utils/http.py                  (tqdm integration)
 R  chorus/utils/ld.py                    (LDLINK_TOKEN env + config fallback)
 R  README.md                             (+Tokens section)
 +  chorus/cli/_setup_all.py              (93 lines)
 +  chorus/cli/_setup_prefetch.py         (173 lines)
 +  chorus/cli/_tokens.py                 (205 lines)
 +  chorus/core/weights_probe.py          (140 lines)
```

## Verdict
**Green.** Safe to commit the 8 modifications + 4 new files, but
**do not `git add .` or `git add -A`** — `Untitled.ipynb` is a
pre-existing untracked stray (Apr 22, not part of this change) and
must be left out. Recommend a selective `git add` of the 12 files
listed above.
