# v19 post-checklist full pass — 2026-04-21

First audit run **against the new `AUDIT_CHECKLIST.md` runbook** (merged
in v18). Exercised the checklist end-to-end on a Linux/CUDA host to
validate the runbook itself and catch any late drift before ship.

## What was actually run

### §1 — Install smoke

`chorus --help` lists 6 subcommands: `setup`, `list`, `validate`,
`remove`, `health`, `genome`. All resolve.

### §4 — Per-track CDF sanity (all 6 oracles)

| oracle | n_tracks | effect_cdfs monotonic | summary_cdfs p50≤p95≤p99 | signed% | perbin_cdfs |
|---|---|---|---|---|---|
| enformer | 5,313 | ✓ | ✓ | 0% | yes |
| borzoi | 7,611 | ✓ | ✓ | 20% | yes |
| chrombpnet | 24 | ✓ | ✓ | 0% | yes |
| sei | 40 | ✓ | ✓ | 100% | no |
| legnet | 3 | ✓ | ✓ | 100% | no |
| alphagenome | 5,168 | ✓ | ✓ | 12% | yes |

Matches the expected catalog counts exactly. All monotonicity checks pass.

### §5 — `sequence_length` per oracle

All six `create_oracle(..., use_environment=False).sequence_length` values
match the README hardware matrix (Enformer 393,216 · Borzoi 524,288 ·
ChromBPNet 2,114 · Sei 4,096 · LegNet 200 · AlphaGenome 1,048,576).

### §10 — Repo-wide drift grep

Greps from the checklist (`5,930`, `5930`, `196 kbp`,
`examples/applications/`) on tracked files:

- **One real drift fixed**: `chorus/oracles/alphagenome.py:22` — class
  docstring said "AlphaGenome predicts 5,930 human functional genomic
  tracks"; corrected to **5,731** (matches v17 mcp spec + v18 metadata
  fix + README). The class docstring is what users see in `help(oracle)`.
- False-positive matches: bundled `igv.min.js` string literals, Sei /
  Borzoi numeric track IDs that happen to contain `5930`, and base64
  PNG payloads in committed notebook outputs. None are user-visible doc
  drift.

### §11 — Fast test suite

```
mamba run -n chorus python -m pytest tests/ --ignore=tests/test_smoke_predict.py -q
→ 334 passed, 1 skipped, 0 failed (668.92s)
```

Matches the checklist's `≥ 334 pass, ≤ 1 skip, 0 error` target exactly.

### §15 — Offline / air-gapped HTML rendering

`grep -oE '<script[^>]*src=…|<link[^>]*href=http…' examples/walkthroughs/**/*.html`
returned nothing. No report loads external scripts or stylesheets.
(The `cdn|googleapis` hits the checklist warned about turn out to be
string literals *inside* the vendored IGV library — IGV uses them for
its own GCS/Drive file-access features; they are never fetched at
report-load time.)

### §16 — Secrets in tracked files

`git ls-files | xargs grep -lE 'hf_[a-zA-Z0-9]{20,}'` returned nothing.
The top-level `AUDIT_PROMPT_WITH_TOKENS.md` does contain a real HF
token but is **gitignored** and not tracked.

### §17 — Dependency pinning

6 bare deps fixed in `environment.yml`: `jupyter`, `notebook`,
`ipykernel`, `samtools`, `htslib` now carry floor versions
(`>=1.0` / `>=6.4` / `>=6.0` / `>=1.15`). `pip` intentionally left
bare (conda's own packaging primitive).

### §18 — License / attribution

- `LICENSE` present (MIT, 2024 Pinello Lab) — ✓.
- Created **`docs/THIRD_PARTY.md`** enumerating every upstream oracle
  with paper DOI and license, the bundled IGV library with its
  upstream license URL, and the CDF dataset license. Linked from
  `README.md` in the "Further reading" table.
- Bundled `chorus/analysis/static/igv.min.js` does not carry an
  upstream MIT header in-line, but its upstream license is cited in
  `docs/THIRD_PARTY.md`. If strict header preservation matters for
  redistribution, wrap the min.js with its license block on the next
  IGV version bump (P2 follow-up, not a ship blocker).

## Fixed in this pass

1. **`chorus/oracles/alphagenome.py:22`** — "5,930" → "5,731" tracks in the
   `AlphaGenomeOracle` class docstring. Last remaining source-level
   drift after v18. **P1**
2. **`environment.yml`** — floor-pins on `jupyter`, `notebook`,
   `ipykernel`, `samtools`, `htslib`. Prevents a clean-install user
   from silently getting incompatible majors. **P1**
3. **`docs/THIRD_PARTY.md`** — new file; Oracle + IGV attribution table
   with papers + licenses. **P1**
4. **`README.md`** — added THIRD_PARTY.md link in "Further reading". **P2**
5. **`CLAUDE.md`** (new, root of repo) — points future Claude sessions
   at `audits/AUDIT_CHECKLIST.md` as the canonical ship-prep runbook,
   documents the env matrix and regen workflow.

## Deferred — P1 follow-ups not fixed here

- **§14 indel pre-validation** — `OracleBase.predict_variant_effect`
  does **not** check `len(ref) == len(alt)` before invoking the model.
  For SNV-only oracles this lets indels silently skate through and
  potentially crash inside the model wrapper. Each oracle has its own
  rule (AlphaGenome handles indels; Enformer/ChromBPNet/Sei/LegNet/Borzoi
  don't), so this needs a per-oracle capability flag + a shared guard
  in `core/base.py` with a clear "indels not supported by <oracle>"
  message. Not a one-line fix; spawning a focused PR is the right move.
- **§14 multi-allelic** — the predict path takes `alleles=['A','C','G','T']`
  today; the *report renderer* has not been exercised against > 1 alt
  in the current checklist run. Worth writing a regression test.
- **§17 `pip-audit`** — not run here (requires a fresh env create to be
  meaningful). Add to the release gate once CI runs the env create.
- **§3 CHORUS_DEVICE=cpu on a GPU host** — listed in checklist as P2;
  not exercised this pass.

## Bottom line

Checklist runbook works — every §4/§5/§10/§11/§15/§16 check produced a
clean binary result. One real drift found (`alphagenome.py` docstring,
5,930 → 5,731) and fixed. `environment.yml` tightened, `docs/THIRD_PARTY.md`
added, `CLAUDE.md` established. 334 tests green. No new P0s; three
P1 follow-ups filed for a later PR (§14 indel guard, §14 multi-allelic
report test, pip-audit in release gate).
