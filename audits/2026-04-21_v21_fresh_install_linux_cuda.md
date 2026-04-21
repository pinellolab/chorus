# v21 fresh-install Linux/CUDA audit — 2026-04-21

The §1-P0 item that had been deferred since v18 ("truly-fresh
`mamba env create` on a clean Linux/CUDA box") — run end-to-end on
this host with maximally isolated caches.

## Isolation approach

- `SCRATCH=/tmp/chorus-audit-v21/`
- `HOME=$SCRATCH/home` during the audit → `~/.chorus`,
  `~/.cache/huggingface`, and every other "default to home" cache
  writes into the scratch tree, not the user's real home.
- Fresh env created via `mamba env create -f environment.yml --prefix
  $SCRATCH/env` — does not touch
  `/data/pinello/SHARED_SOFTWARE/envs/lp698_envs/chorus-*`.
- Only the HF token was copied into `$SCRATCH/home/.cache/huggingface/`
  so HF-gated auto-downloads could proceed.

Artefacts: [`/tmp/chorus-audit-v21/artifacts/`](/tmp/chorus-audit-v21/artifacts/)
(not committed — reproducible by re-running the audit).

## What was actually run

### §1 — install smoke (fresh env)

- `mamba env create -f environment.yml --prefix $SCRATCH/env` — **PASS** (9 min, 844-package transaction)
- `pip install -e .` inside the new env — **PASS**
- `chorus --help` — **PASS** (6 subcommands)
- `chorus genome download hg38` — **PASS** (fresh 3 GB download, indexed into `genomes/hg38.fa`)

### §2 — HF gate (with real token)

- `HF_TOKEN` honoured via `~/.cache/huggingface/token` in scratch HOME.
- AlphaGenome auto-download worked (see §4).

### §4 — fresh CDF auto-download (all 6 oracles, first-use from HF)

Every CDF NPZ was downloaded fresh from
[`huggingface.co/datasets/lucapinello/chorus-backgrounds`](https://huggingface.co/datasets/lucapinello/chorus-backgrounds)
on first `get_normalizer()` call. No pre-existing `~/.chorus/backgrounds`.

| oracle | n_tracks | effect_cdfs monotonic | summary_cdfs p50≤p95≤p99 | signed% | perbin_cdfs | status |
|---|---|---|---|---|---|---|
| enformer | 5,313 | ✓ | ✓ | 0% | yes | OK |
| borzoi | 7,611 | ✓ | ✓ | 20% | yes | OK |
| chrombpnet | 24 | ✓ | ✓ | 0% | yes | OK |
| sei | 40 | ✓ | ✓ | 100% | no | OK |
| legnet | 3 | ✓ | ✓ | 100% | no | OK |
| alphagenome | 5,168 | ✓ | ✓ | 12% | yes | OK |

### §5 — Python API sanity

- `sequence_length` for all 6 oracles matches the expected spec.
- `create_oracle('fakeOracle')` → `ValueError: Unknown oracle: fakeoracle. Available: ['enformer', 'borzoi', 'chrombpnet', 'sei', 'legnet', 'alphagenome']` — clean, actionable.

### §10 — repo-wide drift grep

- **Found 1 live drift, fixed in this PR**: `examples/notebooks/comprehensive_oracle_showcase.ipynb` markdown cells 1 + 21 said `"LegNet … 230 bp"`. Actual input is 200 bp (matches `create_oracle('legnet').sequence_length`). Updated both cells.
- No other drifts in tracked code (false positives: `common_snps_500.bed` coordinate `49325929-49325930`, IGV bundled JS internals, Sei/Borzoi numeric track IDs, base64 PNG payloads in notebook outputs).

### §11 — fast test suite (fresh env)

```
pytest tests/ --ignore=tests/test_smoke_predict.py -m "not integration" -q
→ 332 passed, 1 skipped, 4 deselected in 43s
```

(`332 passed + 3 successfully-running integration tests on the main
dev env = 335 — matches the existing-env count from v20.`)

### §11 — integration test: found a real P1, fixed here

`tests/test_integration.py::test_chrombpnet_fresh_single_model_download`
failed on the fresh env with `ModuleNotFoundError: No module named
'tensorflow'`. Root cause cascade:

1. User's new env only has `chorus` base env — no `chorus-chrombpnet`.
2. Test instantiates oracle with `use_environment=True`.
3. `create_oracle` sees the env is missing → gracefully falls back to
   `use_environment=False` (correct behaviour, per v5).
4. Direct load tries to `import tensorflow` in the base env → not
   installed → raises `ModelNotLoadedError`.

**Fix (in this PR)**: added a skip guard at the top of the test:

```python
from chorus.core.environment.manager import EnvironmentManager
if not EnvironmentManager().environment_exists("chrombpnet"):
    pytest.skip(
        "chorus-chrombpnet env missing — run `chorus setup --oracle chrombpnet` first. "
        "Without it, the subprocess oracle runner falls back to direct load which needs "
        "TensorFlow in the base env (not installed by default)."
    )
```

Verified: re-run on fresh env → **1 skipped in 2.24s** (down from
442 s fail).

### §15 — offline HTML rendering

No external `<script src=http…>` / `<link href=http…>` in any shipped
walkthrough HTML — offline-safe. Confirms v19 finding on fresh checkout.

### §16 — secrets

No HF tokens in tracked files (`git ls-files | xargs grep -lE 'hf_...'`
returns nothing). Confirms v19.

### §17 — pip-audit on fresh env

First attempt ran under the wrong Python (v21 audit-script bug — picked
up `/PHShome/lp698/.local/bin/pip` which is Python 2.7). Fixed by
invoking `$ENV/bin/pip-audit` directly:

```
No known vulnerabilities found
```

→ `pillow>=12.2.0` pin from v20 took effect; no unresolved CVEs on the
fresh env.

### §18 — LICENSE + attribution

- `LICENSE` MIT — ✓
- `docs/THIRD_PARTY.md` — ✓ (shipped in v19)
- `audits/AUDIT_CHECKLIST.md` — ✓

## Fixed in this PR

1. **`examples/notebooks/comprehensive_oracle_showcase.ipynb`** — LegNet
   "230 bp" → "200 bp" in cell 1 (hardware table) and cell 21 (LegNet
   section header). Matches library `sequence_length=200`.
2. **`tests/test_integration.py`** — skip guard added to
   `test_chrombpnet_fresh_single_model_download` so fresh-install
   users without a `chorus-chrombpnet` env see a helpful skip message
   instead of a TF `ModuleNotFoundError`.

## What the fresh install proved

- §1 install path works end-to-end from a clean state on Linux/CUDA.
- §4 CDF auto-download from HF works fresh for all 6 oracles.
- §5 API is clean.
- §11 fast suite is fully green.
- §17 no CVEs thanks to v20 pillow pin.
- Every cache path correctly respects `HOME` — no escapes into real
  `~/.chorus` or `~/.cache/huggingface`.

Checkbox status of the deferred §1 item: **closed**. Remaining
deferreds are §14 indel pre-validation, §14 multi-allelic report
regression test, §14.5 near-telomere with wide-window oracle,
§3 `CHORUS_DEVICE=cpu` on GPU host.
