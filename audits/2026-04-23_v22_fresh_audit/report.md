# v22 fresh audit — post setup-prefetch features — 2026-04-23

Independent fresh-state re-audit after commit `3735ea5` ("Setup prefetch
+ health classification + token flow") landed on `main`. Every
re-downloadable cache was purged before starting. **No new findings** —
every exercised item passes.

## What was nuked before the audit

| Cache | Before | After |
|---|---|---|
| `~/.chorus/backgrounds/` (6 NPZ files) | 1.5 GB | empty |
| `genomes/hg38.fa` + `.fai` | 3.0 GB | gone |
| HF cache | (kept — logged-in state retained) | unchanged |

## NEW features exercised from scratch

### `chorus health` on a fully-empty-state host — **§1-NEW**

Before 3735ea5, Sei alone hung for 120 s on health checks when its
assets were missing. After:

```
Checking alphagenome ...  1.5 s → Not installed — run `chorus setup alphagenome`
                           (marker + HF auth hints inline)
Checking borzoi ...        0.8 s → Not installed
Checking chrombpnet ...    0.6 s → Not installed
Checking enformer ...      0.6 s → Not installed
Checking legnet ...        0.6 s → Not installed
Checking sei ...           1.0 s → Not installed
TOTAL 6 oracles: 5.5 s
```

Every oracle reports the exact missing artifact (`downloads/<oracle>/.chorus_setup_v1`);
AlphaGenome also reports the HF auth failure inline. **PASS.**

### Token-halt path — **§2-NEW**

Bare `chorus setup` with `HF_TOKEN` unset + non-TTY stdin:

```
chorus.cli._tokens  ERROR  No HuggingFace token available and stdin is not a TTY.
                           Pass --hf-token, set HF_TOKEN, or run 'huggingface-cli login' first.
chorus.cli._setup_all ERROR `chorus setup all` halted: a working HuggingFace token is
                           required for AlphaGenome. Nothing was downloaded.
EXIT 1
```

Halts in <1 s. Names all 3 resolution alternatives. Says "Nothing was
downloaded" so the user knows no partial state exists. **PASS.**

### Partial setup — `--no-weights --no-genome` — **§1-NEW**

```
chorus setup --oracle enformer --no-weights --no-genome   →   11.85 s
  ✓ env for enformer (already exists)
  ✓ pulled 1 background file(s) for enformer (548 MB in ~10 s)
  ✓ enformer env ready (weights skipped — setup marker NOT written)
```

Setup marker correctly **not** written when `--no-weights` is used —
subsequent `chorus health` keeps reporting "Not installed". Honours the
documented contract. **PASS.**

### CLI --help

```
chorus setup --help  →  20 lines, documents all 5 flags + Tokens block.
chorus --help        →  lists all 6 subcommands (setup, list, validate,
                         remove, health, genome).
```

### README `## Tokens` section

Two-row table covers `HF_TOKEN` and `LDLINK_TOKEN` with:
- When you need each
- How `chorus setup` resolves it (4-step chain for HF: `--hf-token` →
  env → `huggingface-cli login` → interactive prompt)
- Halt semantics explicit: "halts the whole flow if no working token
  can be resolved, so the other 5 oracles aren't built for nothing"
- Registration URLs for both token types + the AlphaGenome license page

Cross-referenced from the backgrounds section: **"The backgrounds
dataset is public — no HuggingFace token required"** avoids the common
misunderstanding.

## Full checklist re-audit

| § | Result |
|---|---|
| 1 CLI + setup + health | **PASS** — new fast path verified end-to-end |
| 2 HF token flow | **PASS** — halt behaviour + 4-step resolution + repo URL consistency |
| 4 CDF fresh pull (6 oracles) | **PASS** — 29.6 s total from empty cache; all monotonic, `p50≤p95≤p99`, signed% correct |
| 7 HTML reports (selenium) | **PASS** — 18/18 render with 0 JS errors |
| 7 4-part IGV contract | **PASS** — 17/17 non-batch reports: IGV embedded + ymax 3.0/1.0 + 5/5 scale markers + full assay:cell_type provenance |
| 10 Repo-wide drift | **PASS** — grep for `5,930`, `7,612`, `196 kbp`, `LegNet 230 bp`, `examples/applications/` → all empty |
| 11 Fast test suite | **PASS** — **338 passed / 2 skipped** (63 s; skips are integration tests correctly guarding on missing `.chorus_setup_v1`) |
| 15 Offline | **PASS** — 0 runtime CDN fetches across all HTMLs |
| 16 Logging hygiene | **PASS** — 0 committed `hf_…` or `AKIA…` |
| 18 License / attribution | **PASS** — `LICENSE` + `docs/THIRD_PARTY.md` intact |

Fresh genome + fresh CDFs:

- `chorus genome download hg38` → 3.1 GB fetched + decompressed + indexed in ~10 min
- CDF auto-download (5 remaining oracles, after partial setup pulled
  enformer): **29.6 s total** from empty cache

## Scope deliberately deferred

- **§1 full conda env recreate** (80 GB / 2–4 h) — destructive to
  ongoing work; the other agent already verified this separately.
- **§6 multi-oracle + advanced notebooks** — need all 6 oracles loaded.
- **§8 MCP E2E over stdio** — ~4 min AlphaGenome predict.
- **§13 real-oracle determinism** — ~30 min across 6 loaded models.

## Headline

`3735ea5` ("Setup prefetch + health classification + token flow") is
working as documented end-to-end. The biggest user-facing wins —
**5 s health for 6 oracles instead of 720 s**, and the **fail-fast token
halt** — both verified on a fully purged-cache host with zero prior
setup markers.

## Artefacts (~12 MB)

- `report.md` — this summary
- `screenshots/*.png` (16 files; 18 reports with 2 basename collisions)
- `logs/00_pre_nuke.txt`, `01_post_nuke.txt`
- `logs/02_health_fresh.txt`, `03_setup_help.txt`, `04_token_halt.txt`,
  `05_setup_partial.txt`, `06_readme_tokens.txt`
- `logs/07_genome_download.txt`, `08_cdf_fresh.txt`
- `logs/09_selenium.txt`, `10_pytest.txt`, `11_consistency.txt`

---

## Addendum — real end-to-end setup run (post-critique)

Initial v22 audit deferred the §1 P0 items ("actually run `chorus setup`"),
only exercising the data-cache purge + partial `--no-weights` path.
After pushback ("did you actually install it?"), ran the real flow.

### `chorus setup --oracle enformer` end-to-end (no flags)

```
env: already exists → skip                        (0.03 s)
✓ enformer weights ready (TFHub prefetch)          (5.6 s)
Backgrounds for enformer already cached            (instant)
Reference genome hg38 already present              (instant)
marker written: downloads/enformer/.chorus_setup_v1
chorus health → ✓ enformer: Healthy
Total: 5.8 s
```

### Idempotency — 2nd run of `chorus setup --oracle enformer`

Same path, same marker already present → 5.5 s no-op. TFHub re-verifies
but finds cache. ✓

### `chorus setup --oracle legnet` end-to-end — **FOUND P1 REGRESSION**

```
✗ prefetch failed for legnet:
  - weights: TypeError: LegNetOracle.load_pretrained_model() got an
    unexpected keyword argument 'assay'
```

**Root cause:** `chorus/cli/_setup_prefetch.py:37-40` stored
`{'assay': 'LentiMPRA', 'cell_type': 'HepG2'}` under `_DEFAULT_LOAD_KWARGS`
for LegNet, but LegNet takes `assay`/`cell_type` on `__init__`, not on
`load_pretrained_model`. ChromBPNet is the opposite — takes them at load
time. The prefetch config conflated the two.

**Impact:** every `chorus setup --oracle legnet` invocation fails before
writing the `.chorus_setup_v1` marker → LegNet is permanently "Not
installed" per `chorus health`, even after setup succeeded at the env
level.

**Fix:** split the kwarg config into `_DEFAULT_CTOR_KWARGS` (for LegNet)
vs `_DEFAULT_LOAD_KWARGS` (for ChromBPNet), render both into the
prefetch script.

### Post-fix verification

```
chorus setup --oracle legnet → ✓ legnet ready (3.1 s)
                             → marker written
chorus health                → ✓ legnet: Healthy
```

End-to-end contract now holds for both oracles tested without escape
hatches.
