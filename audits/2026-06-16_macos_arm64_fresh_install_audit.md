# Fresh-install audit (scorched-earth, nothing cached) — 2026-06-16

**Branch**: `main` HEAD `621ee0a` (v0.5.6, 7 oracles incl. EPInformer-seq)
**Host**: macOS 15.7.4 (Darwin 24.6.0), Apple Silicon **arm64**, `mamba` 2.4.0 from miniforge
**Auditor**: Claude Code (Opus 4.8), driven by @lucapinello
**Goal**: Validate the README's documented install path as a brand-new user, with
**everything wiped and no cached artifacts reused** — all chorus envs, weights,
backgrounds, genomes, and duplicate clones removed before starting. Catch the
first-run / cold-cache failures that warm pytest sweeps miss.

> **Status: COMPLETE.** Full from-scratch install audited end-to-end: env rebuild,
> dependency install, `chorus setup` (all 7 oracles + AlphaGenome PyTorch backend),
> `chorus health`, and a real Enformer prediction + variant-effect smoke test.
> **Final result: 8/8 oracle envs healthy, smoke test PASS** — after fixing the two
> install bugs below (#1 coolbox, #4 epinformerseq).

---

## TL;DR for developers — what's broken

| # | Severity | Problem | Who it hits |
|---|---|---|---|
| 1 | **P0 — blocker** | `mamba env create -f environment.yml` fails: `coolbox>=0.4.0` is **quarantined on PyPI** (zero installable files). Because pip installs the `pip:` block as one batch, the failure **also takes down `fastmcp`, `huggingface_hub`, and `oxbow`** — so the MCP server stack is missing after the documented Step 1. | **Every new user**, today, on the documented install path. |
| 4 | **P1 — broken oracle** | `chorus setup` **fails to build the `epinformerseq` env** (`Failed to create environment: None`). Root cause: `environments/chorus-epinformerseq.yml` declares the vestigial **`nvidia` conda channel**, and unlike every other oracle, epinformerseq is **not run through platform adaptation**, so the channel isn't stripped on macOS. The `nvidia` channel SSL-times-out → build dies. `nvidia` is unused (torch comes via pip). **Fix applied** (see below): removed the channel → env builds + weights + background all succeed. | Any non-CUDA host, or any host where conda.anaconda.org/nvidia is slow/blocked. The whole new EPInformer-seq oracle is unavailable. |
| 5 | P2 | After `git pull` upgrading chorus, the **already-running MCP server keeps serving stale in-memory code**. The live server listed only 6 oracles (no `epinformerseq`) even though the on-disk `ORACLE_SPECS` has all 8. **The MCP server must be restarted** (restart Claude Code) after an upgrade. | Anyone upgrading chorus while Claude Code / Desktop is running. |
| 2 | P3 | `chorus setup --help` claims `alphagenome_pt` is **skipped by default** and needs `--include-alternative-backends`. The **code already installs it by default** (`_SKIP_FROM_DEFAULT_SETUP` is empty), matching the README. So the *behavior* is correct (default-on); only the **CLI help text was stale** and misleading. | Anyone reading `--help` (it sent this audit down a needless `--include-alternative-backends` detour). |
| 3 | P3 | `genomes/` and `downloads/` resolve **relative to the current working directory**. Two clones on this host each accumulated a full data tree (~31 GB duplicated), and `chorus cleanup --all` only cleaned the tree it recorded, **orphaning 17 GB** in the other. No warning that multiple data roots exist. | Anyone with >1 chorus checkout, or who runs setup from different dirs. |

## Fixes applied — PR `fix/2026-06-16-fresh-install-coolbox-epinformerseq`

| # | Status | Change |
|---|---|---|
| 1 coolbox P0 | ✅ fixed | `environment.yml` now installs coolbox from the **official GitHub repo, SHA-pinned to the `0.4.0` tag** (`651b930…`) instead of the quarantined PyPI package. Verified on ARM64: installs, `import coolbox` (0.4.0), and all symbols chorus uses (`GTF`, `TabFileReaderInMemory`, `FMT2COLUMNS`) resolve. Comment tells maintainers to revert to PyPI once un-quarantined. |
| 4 epinformerseq P1 | ✅ fixed | Removed the dead `nvidia` channel from `environments/chorus-epinformerseq.yml`. Rebuilt clean (env + weights + background). |
| 4b diagnostics | ✅ fixed | `manager.py` logged `process.stderr` (always `None`, since stderr is merged into streamed stdout) → the useless `Failed to create environment: None`. Now captures the last 25 output lines + exit code in the error. |
| 2 help text P3 | ✅ fixed | Corrected the stale `--include-alternative-backends` help text to state it's a no-op and both AlphaGenome backends install by default. (Per maintainer: default-on is intended — behavior was already correct.) |
| 5 MCP restart | ✅ doc'd | Added a "restart Claude Code after upgrading" note to the README Upgrading section. |
| 3 cwd data paths | 📋 open | **Not auto-fixed** (behavioral/migration change). Documented below as a proposal for maintainer decision. |

---

## Pre-state (what "from scratch" had to tear down)

This host was **not** a clean slate — auditing it surfaced finding #3 before any
install even started:

- **8 conda envs** present (`chorus` base + 7 oracle envs) — but the install was
  **stale**: pre-v0.5.6, missing the `chorus-epinformerseq` env entirely.
- **Two full clones** with split data:
  - `/Users/lp698/chorus` — on `main` (latest), `downloads/` = **17 GB**, `genomes/hg38` 3.2 GB
  - `/Users/lp698/chorus_test/chorus` — on branch `poster/cshl-rs12740374-multioracle`
    (4 commits behind main, **clean tree, fully pushed, nothing unique** — verified
    safe to delete), `downloads/` = 6.4 GB, `genomes/hg38` 3.2 GB
- `~/.chorus/backgrounds` (shared) = 1.6 GB, **missing `epinformerseq_pertrack.npz`** (pre-v0.5.6).
- HF token cached at `~/.cache/huggingface/token` — validated live (`whoami` → `lucapinello`,
  gated `google/alphagenome-all-folds` accessible, gating accepted).

**Wipe performed** (~33 GB reclaimed): removed all 8 conda envs; `rm -rf
~/.chorus/backgrounds`, `/Users/lp698/chorus/downloads`, the canonical
`genomes/hg38.fa{,.fai}`, and the entire `/Users/lp698/chorus_test` clone.

---

## Step 1 — README "Install (5 minutes)"  →  **FAILS (P0)**

```bash
mamba env create -f environment.yml      # conda half OK; pip half FAILS
```

The conda dependencies resolve and link fine. The trailing `pip:` block fails:

```
Installing pip packages: coolbox>=0.4.0, oxbow>=0.4.0, fastmcp>=3.0, huggingface_hub>=0.20.0
ERROR: Could not find a version that satisfies the requirement coolbox>=0.4.0 (from versions: none)
ERROR: No matching distribution found for coolbox>=0.4.0
critical libmamba pip failed to install packages
```

### Root cause — coolbox is quarantined on PyPI

PyPI itself is reachable (`pip index versions oxbow` → lists fine). Only `coolbox`
returns nothing. The simple-index page is the proof:

```
GET https://pypi.org/simple/coolbox/   → 200, but ZERO <a> file links
    <meta name="pypi:project-status" content="quarantined">
GET https://pypi.org/pypi/coolbox/json → 404 ({"message": "Not Found"})
```

`quarantined` = PyPI admins have flagged the project (security review / malware
report / account issue), so **no release file is downloadable**. This is an
upstream PyPI state, not a local cache or network problem — which is exactly why
the "nothing cached" audit caught it and a warm machine with a cached wheel would not.

### Why this is P0, not cosmetic

`pip` installs the whole `pip:` list as **one invocation**, and one unresolvable
requirement aborts the entire batch. So after the documented Step 1, the env is
**also missing `fastmcp`, `huggingface_hub`, and `oxbow`** — i.e. the MCP server
cannot start and AlphaGenome auth helpers are absent. A new user following the
README has a hard-broken install with a confusing error pointed at coolbox only.

### coolbox usage — it's optional and lazily imported

coolbox is referenced only for **track-figure visualization** and every import is
**lazy (function-local)**, never module-level:

- `chorus/core/result.py:342` — `from coolbox.api import (...)` (inside a method)
- `chorus/utils/annotations.py:457-458` — `from coolbox.api import GTF`, `from coolbox.utilities.reader.tab import (...)`
- `chorus/analysis/_track_figure.py`, `chorus/__init__.py` (reference only)

Confirmed: with coolbox absent, `import chorus` and `from chorus.mcp import server`
both succeed. Only the coolbox plotting path is unavailable.

### Recommended fixes (for the team)

1. **Decouple the critical deps from coolbox** so a coolbox outage can't break the
   MCP install. Options, roughly in order:
   - Move `fastmcp`, `huggingface_hub`, `oxbow` out of the same failing pip batch
     (e.g. install them via conda where possible, or as a separate resilient step),
     **or** make coolbox a documented optional extra (`pip install chorus[viz]`).
   - Pin coolbox to a non-PyPI source while quarantined (git URL to the upstream
     repo `GangCaoLab/CoolBox`) or fall back to conda-forge `coolbox` (only 0.3.6
     available; note the env.yml comment that the conda build "lacks ARM64 support"
     — but 0.3.6 is `noarch py_0`, so re-verify whether that's still true).
   - Gate the coolbox import behind a clear "install coolbox for visualization" error.
2. **Track the PyPI quarantine** — open/expect an upstream issue; the package may be
   un-quarantined (false positive) or permanently removed.
3. Add a CI job that does a true cold `mamba env create -f environment.yml` so a
   dependency disappearing from PyPI fails CI, not users.

### Resolution (the fix shipped in this PR)

First bootstrapped without coolbox to verify the core is sound — installed the three
critical pip deps directly + editable chorus (`fastmcp 3.4.2`, `huggingface_hub 1.19.0`,
`oxbow 0.8.0`, `chorus 0.5.6`; `import chorus` ✓, `from chorus.mcp import server` ✓).

Then fixed it properly: **`environment.yml` now installs coolbox from the official
GitHub repo, SHA-pinned to the `0.4.0` tag**:

```yaml
    - coolbox @ git+https://github.com/GangCaoLab/CoolBox.git@651b930dbc59e3aa732f7ecc98e0af09e19e2719
```

The repo (`GangCaoLab/CoolBox`, 257★, GPL-3.0) is the legit, actively-maintained
upstream (last push 2026-06-08), the `0.4.0` tag satisfies the pin, and it's
pure-pip / `requires-python >=3.7` so it's cross-platform (the bioconda "no ARM64"
caveat is about the *conda* build, not this). Verified on ARM64: builds a wheel,
`import coolbox` → `0.4.0`, and the exact symbols chorus imports
(`GTF`, `TabFileReaderInMemory`, `FMT2COLUMNS`) all resolve.

> **Residual note:** the PyPI quarantine reason is not public. SHA-pinning to the
> upstream release commit is the most defensible mitigation; the comment in
> `environment.yml` tells maintainers to revert to `coolbox>=0.4.0` once PyPI
> un-quarantines the project.

---

## Finding #2 — stale `--help` text for `alphagenome_pt` (P3, docs only)

- **README** ("Disk usage breakdown" and "Two AlphaGenome backends") states
  `alphagenome_pt` is **"installed by default `chorus setup`"**.
- **`chorus setup --help`** stated the opposite: that it's *"skipped from the default
  install"* and needs `--include-alternative-backends`.

**The README is right.** The code's `_SKIP_FROM_DEFAULT_SETUP` set in
`chorus/cli/_setup_all.py` is **empty**, with a comment confirming both AlphaGenome
backends install by default for MPS access; `--include-alternative-backends` is a
documented no-op. So the *behavior* is correct (default-on) — only the **CLI help
string was stale**. (It cost this audit a needless `--include-alternative-backends`
invocation; a bare `chorus setup` would have installed `alphagenome_pt` anyway.)

**Fix:** corrected the help text to describe the flag as a no-op and state that both
backends install by default. No behavior change (per maintainer: default-on is intended).

---

## Finding #3 — cwd-relative data paths cause silent duplication (P2)

`genomes/` and `downloads/` are resolved relative to the **current working
directory** (verified: `GenomeManager().genomes_dir` returns
`<cwd>/genomes`). Consequences observed on this host:

- Running `chorus setup` from two different clones produced **two near-complete data
  trees** (~31 GB duplicated across `/Users/lp698/chorus` and
  `/Users/lp698/chorus_test/chorus`).
- `chorus cleanup --all --dry-run` reported it would clean only **one** tree
  (`chorus_test`, the last-recorded location) plus the shared backgrounds — i.e. it
  would have **left 17 GB orphaned** in the other clone, with no warning that a
  second data root existed.

**Fix ideas:** default data to a single user-level location (e.g. `~/.chorus/{genomes,downloads}`)
instead of cwd; or have `chorus cleanup`/`chorus health` detect and report all data
roots; or at minimum print the absolute resolved data path at setup time so users
notice duplication.

---

## Finding #4 — `epinformerseq` env build fails (P1, the new 7th oracle)

`chorus setup --include-alternative-backends` reported **`7/8 oracles ready`** —
the one failure was the new EPInformer-seq oracle:

```
=== Setting up epinformerseq ===
Creating environment chorus-epinformerseq from .../environments/chorus-epinformerseq.yml
chorus.core.environment.manager - ERROR - Failed to create environment: None
chorus.cli._setup_all - ERROR - ✗ Failed to build env for epinformerseq
```

### Root cause

- `environments/chorus-epinformerseq.yml` declared `channels: [conda-forge, bioconda, nvidia]`.
- The setup manager applies a **"Platform adaptation (macos_arm64)"** pass to every
  other oracle env — e.g. for `legnet` it logs *"removed CUDA, nvidia and pytorch
  channels (conda-forge has ARM builds)."* **epinformerseq is not registered for
  that adaptation**, so its `nvidia` channel was kept.
- `mamba` then tried to fetch `https://conda.anaconda.org/nvidia/{noarch,osx-arm64}/repodata.json`
  → **SSL connect error / timeout** (11 retries logged) → env create aborts with the
  unhelpful `Failed to create environment: None`.
- The `nvidia` channel is **vestigial**: epinformerseq's only torch dependency is
  installed via `pip: torch>=2.0`, not from the nvidia conda channel. Nothing in the
  env uses it.

### Fix applied (in this working tree, uncommitted)

Removed the dead `nvidia` channel from `environments/chorus-epinformerseq.yml`:

```diff
 channels:
   - conda-forge
   - bioconda
-  - nvidia
```

Re-ran `chorus setup --oracle epinformerseq` → **clean success**:

```
✓ env for epinformerseq
✓ epinformerseq weights ready
✓ pulled 1 background file(s) for epinformerseq   (epinformerseq_pertrack.npz, 2.3 MB)
✓ epinformerseq ready
Setup complete: 1/1 oracles ready.
```

### Recommendation for the team

- Drop the `nvidia` channel from `chorus-epinformerseq.yml` (correct on **all**
  platforms — it's unused), **and/or** add `epinformerseq` to the platform-adaptation
  registry so non-CUDA hosts strip CUDA/nvidia channels consistently with the other
  oracles. A grep for `nvidia` across `environments/*.yml` to confirm no other
  pip-torch oracle has the same latent issue would be worth it.
- Make `Failed to create environment: None` surface the underlying mamba stderr —
  the `None` gave no signal; the real cause was only visible in the raw libmamba log.

---

## Step 2 — `chorus setup` results (after fixes)

Launched with `CHORUS_NO_TIMEOUT=1`, empty stdin piped in to auto-skip the optional
LDlink prompt, HF token resolved from cache (non-interactive — no prompt). Run was
fully unattended.

| Oracle | env | weights | background | result |
|---|---|---|---|---|
| alphagenome (JAX) | ✓ | ✓ | ✓ | ready |
| alphagenome_pt (PyTorch) | ✓ | ✓ | ✓ (aliases alphagenome) | ready |
| borzoi | ✓ | ✓ | ✓ | ready |
| chrombpnet | ✓ | ✓ | ✓ | ready |
| enformer | ✓ | ✓ | ✓ | ready |
| legnet | ✓ | ✓ | ✓ | ready |
| sei | ✓ | ✓ | ✓ | ready |
| **epinformerseq** | ✗→**✓** | ✓ | ✓ | **ready after #4 fix** |

hg38 reference (3.2 GB) downloaded once and reused across all oracles.
**Final: 8/8 oracle envs present.** (`mamba env list` → `chorus` base + 8 oracle envs.)

## Step 3 — `chorus health` + smoke prediction  →  **PASS**

`chorus health --timeout 300` — **all 8 healthy**:

```
✓ alphagenome: Healthy        ✓ enformer: Healthy
✓ alphagenome_pt: Healthy     ✓ epinformerseq: Healthy
✓ borzoi: Healthy             ✓ legnet: Healthy
✓ chrombpnet: Healthy         ✓ sei: Healthy
```

End-to-end Python smoke test (README's β-globin Enformer example) — **PASS**:

```
chorus 0.5.6
WT mean DNase signal: 0.468                     # predict() at chr11:5,247,000-5,248,000
Variant scan OK: scored 3 alt alleles           # predict_variant_effect() C→A/G/T
SMOKE TEST PASS
```

Enformer loaded from the chorus HF mirror (`lucapinello/chorus-enformer`, 5313 tracks),
predicted, and scored a variant — oracle creation, model load, prediction, and
variant-effect scoring all work on a cold install.

## MCP server verification

- `claude mcp list` → `chorus: mamba run -n chorus chorus-mcp - ✔ Connected`
- Live MCP tool call `list_oracles()` returned results successfully (server serves
  tools, not just connects).
- **Caveat (finding #5):** the live server was started before this session's `git pull`,
  so it returned only 6 oracles (stale in-memory code). The on-disk `ORACLE_SPECS`
  has all 8 (`enformer, borzoi, chrombpnet, sei, legnet, epinformerseq, alphagenome,
  alphagenome_pt`). **Restart Claude Code** to pick up v0.5.6 (epinformerseq + current tools).

---

## Summary

A genuine scorched-earth install (every env/weight/background/genome wiped, plus a
duplicate clone removed, ~33 GB reclaimed; ~31 GB re-downloaded) surfaced **two
install-blocking bugs the warm CI/pytest path misses**:

1. **P0** — coolbox quarantined on PyPI breaks the documented `environment.yml`
   install *and* silently drops the MCP deps (`fastmcp`/`huggingface_hub`/`oxbow`).
2. **P1** — the new `epinformerseq` oracle env can't build on non-CUDA hosts because
   of a vestigial `nvidia` channel that platform-adaptation doesn't strip.

Both were worked around / fixed during this audit; with the fixes, **8/8 oracles
install, pass health, and a real prediction + variant scan succeed end-to-end.**
Two lower-severity doc/UX issues (#2 alphagenome_pt default, #3 cwd-relative data
paths) and one operational note (#5 restart MCP after upgrade) are documented above.
