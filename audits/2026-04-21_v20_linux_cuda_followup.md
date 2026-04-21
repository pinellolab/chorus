# v20 Linux/CUDA follow-up — 2026-04-21

Closes the four open items flagged by the other auditor after v19:

1. **§18 attribution** — *already shipped in v19.* `docs/THIRD_PARTY.md` is in `37d11b9`; linked from README Further reading. No action.
2. **§17 pip-audit in CI** — wired up here.
3. **§1 fresh Linux/CUDA env probe** — run on this Linux-x86_64 + CUDA host.
4. **§14 genomics edge cases with a loaded oracle** — run on LegNet.

## §17 — pip-audit wired into CI

- `environment.yml`: floor-pinned `pillow>=12.2.0` — pip-audit flagged
  two CVEs in pillow 12.0.0 (CVE-2026-25990 PSD out-of-bounds write;
  CVE-2026-40192 FITS decompression bomb). Both are image-format bugs
  that Chorus doesn't exercise (matplotlib uses pillow only for PNG/JPEG
  plot output), but a clean-install user might use the env for other
  pipelines. Pinning costs nothing.
- `.github/workflows/tests.yml`: added a `pip-audit` step after the
  fast suite. `continue-on-error: true` so it's **advisory, not
  blocking** — summary lands in the CI job log for maintainers to
  inspect, the build stays green. `-l` limits to local packages so
  editable Chorus itself doesn't trip up the auditor.

Rationale for advisory-only: pinning every transitive dep's CVE fix
into `environment.yml` the moment pip-audit flags it would cause churn
every week. Keep CI green, surface the advisory, decide per-CVE.

## §1 — env matrix probe (Linux/CUDA host)

Not a *fresh* `mamba env create` (envs already exist on this
development host), but a canonical `EnvironmentManager.environment_exists`
check against the 6 oracle names:

```
enformer     env=chorus-enformer      exists=True
borzoi       env=chorus-borzoi        exists=True
chrombpnet   env=chorus-chrombpnet    exists=True
sei          env=chorus-sei           exists=True
legnet       env=chorus-legnet        exists=True
alphagenome  env=chorus-alphagenome   exists=True
```

All 6 envs detected via `mamba env list --json` — this validates the
code path CI would take. A truly **fresh** create is still needed for
release (separate 80 GB / 2–4 h task on a release runner); filed as
§1 P0 for the release gate, not fixable in a routine audit.

**API drift noticed**: `EnvironmentManager.environment_exists(arg)`
takes the *short oracle name* (`"enformer"`), not the full env name
(`"chorus-enformer"`). Passing the full name silently returns `False`
because internally it prefixes again → `"chorus-chorus-enformer"`.
Not a bug (this is how the API has always worked), but worth a
docstring clarification.

## §14 — genomics edge cases (LegNet)

Exercised on a loaded LegNet model (assay `LentiMPRA:HepG2`, 200 bp
window) against hg38:

| Case | Behaviour | Severity |
|---|---|---|
| **14.1** soft-masked (lowercase) ref — verified by code inspection at `chorus/core/base.py:337` where the ref-allele check applies `.upper()` to both sides. | ✅ handled | n/a |
| **14.2** insertion (`G→GT`) | silently ACCEPTED — treated as a 2-char alt; computes a Δ that is semantically nonsense. No error raised. | **P1** (already deferred in v19) |
| **14.3** chrM variant | silently ACCEPTED. LegNet runs happily since it only sees 200 bp of sequence, but wider-window oracles trained only on autosomes would return garbage. | **P2** — per-oracle `supported_chromosomes` list would be the clean fix |
| **14.4** bad chromosome (`chrZZ:100-300` via `predict()`) | **`KeyError: 'H'` — uncaught low-level error.** pyfaidx returns *something* (apparently containing an `H` character), which propagates to LegNet's one-hot encoder (`legnet_source/transforms.py:26`) before the user sees any useful message. | **P1 NEW** — `core/base.py` should validate `chrom in FASTA.keys()` before any encoding. |
| **14.5** near-telomere (`chr1:50`, 200 bp window) | ACCEPTED. Window fits inside chr1 (1–200). Not a real near-telomere stress test with LegNet's short window — need Enformer (393 kb) or AlphaGenome (1 Mb) for the actual pad/clamp check. | deferred — needs wider-window oracle run |
| **14.6** multi-allelic (`G→[A,C,T]`) | ACCEPTED — returns a dict with multiple alt columns. Report rendering not exercised here. | **P1** (already deferred in v19) |

### §14.4 is the new P1 from this pass

Reproducer:

```python
from chorus import create_oracle
o = create_oracle('legnet', use_environment=False)
o.load_pretrained_model()
o.predict('chrZZ:100-300', assay_ids=[o.assay_id])
# → KeyError: 'H'
```

**Expected**: `ValueError("chromosome 'chrZZ' not found in reference FASTA; known: chr1, chr2, …, chrM, chrY")`.

The fix goes in `core/base.py` — add a chromosome existence check in
`_parse_region` / `_parse_position` / the `predict` entry before
anything touches the FASTA. Must cover all 6 oracles, so it belongs
on the base class rather than per-oracle. Tagged for a focused PR.

## §14.1 soft-mask — verified by code inspection

```python
# chorus/core/base.py:337
if region_interval[real_pos].upper() != ref_allele.upper():
    ...
```

Both sides of the ref-allele comparison apply `.upper()`, so a
lowercase (soft-masked) base at the variant position does not trigger
a spurious mismatch warning. Matches the checklist's §14 guarantee.

## Fixed in this PR

- **`environment.yml`** — `pillow>=12.2.0` (pip-audit CVE fix).
- **`.github/workflows/tests.yml`** — advisory `pip-audit` step after fast suite.
- **`audits/2026-04-21_v20_linux_cuda_followup.md`** — this report.

## Still deferred

- §1 truly-fresh Linux/CUDA `mamba env create` → §1 P0 for release gate (~4h on a clean runner).
- §14.4 chrZZ → clean chromosome-existence guard in `core/base.py` — **NEW P1**, focused follow-up PR.
- §14.2 indel pre-validation — per-oracle capability flag + shared guard (from v19).
- §14.6 multi-allelic report regression test (from v19).
- §14.5 near-telomere with wider-window oracle (Enformer/AlphaGenome).
- §3 `CHORUS_DEVICE=cpu` on a GPU host — P2 from v18, not yet exercised.

Tests: 334 / 1 skip / 0 fail (from v19 run; no code changes that could regress the suite).
