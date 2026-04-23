# v23 scorched-earth install — 2026-04-23

True from-zero install test on macOS arm64. User authorization:
> "I want you do a scorched-to-the-earth install, remove all the chorus
> envs, then reinstall from scratch as a user following the docs"

## Pre-scorch state

| Asset | Size |
|---|---|
| 7 conda envs (`chorus`, `chorus-enformer`, `chorus-borzoi`, `chorus-chrombpnet`, `chorus-sei`, `chorus-legnet`, `chorus-alphagenome`) | (depends on env) |
| `~/.chorus/backgrounds/` | 1.5 GB |
| `genomes/hg38.*` | 3.1 GB |
| `downloads/*` | 43 GB |
| **Total reclaimed** | **~47 GB + all envs** |

## Post-scorch state

`mamba env list | grep chorus` → **empty**. `~/.chorus/` gone.
`genomes/` empty. `downloads/` empty.

## Install, exactly as README "Fresh Install" prescribes

```bash
# 1. mamba env create -f environment.yml              2m 30s
# 2. pip install -e .                                 ~3 s
# 3. chorus setup --oracle enformer                   10m 54s
#      (env build 36 s → weights 24 s → backgrounds 14 s → hg38 9m 26s)
```

### Timeline from the setup run

| t | step |
|---|---|
| 07:48:11 | Setting up environment for enformer... |
| 07:48:47 | ✓ env for enformer (36 s to build chorus-enformer with TF + Metal) |
| 07:48:47 | Pre-downloading enformer weights... |
| 07:49:11 | ✓ enformer weights ready (24 s TFHub) |
| 07:49:11 | Downloading enformer_pertrack.npz from HuggingFace... |
| 07:49:25 | ✓ 1 background file(s) for enformer (14 s, 523 MB) |
| 07:49:25 | Pre-downloading reference genome hg38... |
| 07:58:51 | Decompressing hg38... |
| 07:58:59 | Creating FASTA index... |
| 07:59:05 | ✓ hg38 ready |
| 07:59:05 | ✓ enformer ready |

**Total wall clock from `mamba env create` to "enformer ready": ~13 min 30 s.**

## Post-install verification

### `chorus health`

```
Checking enformer...
✓ enformer: Healthy    (6.5 s — was 720 s in the pre-3735ea5 world)
```

### Real end-to-end prediction

```python
import chorus
o = chorus.create_oracle('enformer',
                         use_environment=True,
                         reference_fasta='.../genomes/hg38.fa')
o.load_pretrained_model()
r = o.predict(('chrX', 48777634, 48790694), ['ENCFF413AHU', 'CNhs11250'])
```

Returns two 896-bin tracks:

- `ENCFF413AHU` (DNASE:K562): mean=0.4842, max=22.4620
- `CNhs11250` (CAGE:K562):    mean=0.5957, max=120.8120

Matches committed notebook values within CPU non-determinism
(mean=0.4841/0.5953, max=22.4569/120.7759). Full end-to-end library
function confirmed.

### Tests

Deferred to post-merge CI run. Results appended once pytest completes.

## Findings

**None.** The documented install path from a purged-cache + deleted-envs
state works exactly as the README claims, in ~14 min on macOS arm64
(dominated by the 9.5 min hg38 download from UCSC — that's the network,
not the code path).

The two v22 P1s (LegNet TypeError, multi-oracle scale block) already
merged via PRs #39 + #40 don't regress here — only enformer was
exercised in this scorched-earth run; legnet's fix is independently
verified.

## Scope notes

- **Only the `chorus` base env + `chorus-enformer` were rebuilt.** The
  other 5 per-oracle envs are also deleted but not re-installed —
  they'd each add 5-15 min of conda-build time and 1-5 GB of weights.
  The README install path is single-oracle by design; `chorus setup
  --oracle all` would cover the rest but takes 2-4 h.
- **HuggingFace auth**: this run didn't touch AlphaGenome (gated), so
  no HF_TOKEN was involved. The token-halt path was already exercised
  in v22.
- **No notebook re-execution** — this audit focused on the install
  flow, not the existing example outputs.

## Artefacts in `audits/2026-04-23_v23_scorched_earth/logs/`

- `00_pre_scorch.txt`   — env list + disk usage before deletion
- `01_post_scorch.txt`  — confirmed empty state
- `02_env_create.txt`   — mamba env create output
- `03_pip_install.txt`  — pip install -e . + chorus --help
- `04_setup_enformer.txt` — full chorus setup log (311 lines)
- `05_health.txt`       — chorus health output
- `06_prediction.txt`   — Python prediction on GATA1 TSS
- `07_pytest.txt`       — fast suite output

## Headline

Fresh install from 7-deleted-envs + 47 GB of purged caches reaches a
working Enformer prediction in **~14 minutes** on macOS arm64 following
the documented `mamba env create` → `pip install -e .` → `chorus setup
--oracle enformer` path. No surprises, no manual fix-ups, no missing
deps.

## Pytest update (post-report)

`pytest tests/ --ignore=tests/test_smoke_predict.py -q` in the
fresh-built `chorus` env: **338 passed, 2 skipped** in 45.67 s. The
2 skipped are integration tests correctly guarding on missing
`.chorus_setup_v1` markers for non-enformer oracles — only enformer
has a marker in this scorched-earth run.
