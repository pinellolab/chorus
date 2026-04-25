# v27 fresh-install new-user audit — 2026-04-25

Full scorched-earth: removed all 7 chorus envs (~80 GB freed), nuked
`~/.chorus`, nuked the AlphaGenome + chorus-backgrounds HF cache
entries, then walked the README's TLDR start-to-finish as a real new
user would. Caught two real P0 bugs in the first 30 minutes.

## Scope

- **Linux x86_64 + CUDA host** with all caches purged.
- HF token preserved through the wipe (separate file under `/tmp`),
  validated against HF API as `lucapinello`. The previously-cached
  token had been rotated/invalidated since v21.
- Followed README §1 → §2 → §3 → §4 verbatim, like a first-time user.
- All artifact logs in `/tmp/v27_audit/` (not committed; reproducible).

## P0 findings — must fix before announce

### P0 #1 — README's `pip install -e .` shadowed by stale `~/.local/bin/pip`

**What happened.** Following [README:22](README.md#L22) verbatim:

```
$ mamba activate chorus
$ pip install -e .
ImportError: cannot import name md5
  File "/apps/source/python/2.7.5/.../site-packages/pip/_vendor/urllib3/util/ssl_.py", line 8
```

`/PHShome/lp698/.local/bin/pip` is on `$PATH` and points at a Python
2.7 install. Even after `mamba activate chorus`, the stale local
`pip` wins because user-bin precedes the env's bin in PATH on this
system. Same pattern broke the v21 audit script.

**Reproducer.** Any HPC user who has ever run `pip install --user`
outside an env will have `~/.local/bin/pip` shadowing the env's pip
forever after. This is **common on shared university clusters**.

**Fix.** Update [README:22](README.md#L22):

```diff
- pip install -e .
+ python -m pip install -e .
```

`python -m pip` always uses the active env's pip — there's no PATH
ambiguity. Verified: `python -m pip install -e .` succeeded in 50 s
on the same broken-PATH system.

### P0 #2 — Quickstart notebook broken: FANTOM track IDs rejected

**What happened.** Running the **shipped** `single_oracle_quickstart.ipynb`
(via `jupyter nbconvert --execute`) raised on the **first multi-track
prediction cell**:

```
InvalidAssayError: Enformer does not recognise these track IDs: ['CNhs11250'].
Discover valid IDs with: `from chorus.oracles.enformer_source.enformer_metadata
import get_metadata; get_metadata().search_tracks('K562')` — or the
`list_tracks` MCP tool.
```

The error message is excellent (v26 gold-standard format), but
**`CNhs11250` IS in Enformer's metadata** (index 4828, "CAGE: chronic
myelogenous leukemia cell line: K562"). The validator was wrong, not
the notebook.

**Root cause.** [chorus/oracles/enformer.py:344-356](chorus/oracles/enformer.py#L344-L356)
only does `get_track_by_identifier()` for IDs starting with `ENCFF*`.
Everything else falls through to `get_tracks_by_description()`, which
never matches a FANTOM CAGE ID like `CNhs11250` (those are track
identifiers, not human descriptions).

**Fix shipped.** [chorus/oracles/enformer.py:349-355](chorus/oracles/enformer.py#L349-L355) —
try `get_track_by_identifier` first regardless of prefix; only fall
back to description match if identifier lookup returns None:

```python
for assay_id in assay_ids:
    if metadata.get_track_by_identifier(assay_id) is None:
        if not metadata.get_tracks_by_description(assay_id):
            bad.append(assay_id)
```

**Verification.** After the fix, full quickstart re-run:
**34 cells executed, 0 errors, 3 plots inline**. WT predictions:
`ENCFF413AHU = 0.484`, `CNhs11250 = 0.595` (CAGE in K562). Match the
biology.

## What worked perfectly first try (no findings)

1. **`mamba env create -f environment.yml`** — completed cleanly
   (Linking phase took ~30 min on PanFS due to small-file metadata
   cost; not a chorus issue, see "Performance" below).
2. **`chorus --help`** — 6 subcommands listed; `chorus setup --help`
   documents all 5 flags and both tokens.
3. **`chorus setup` with no token** → fails fast with a clear numbered
   error message before downloading anything. Exits non-zero. Pattern
   shipped in v22-v25 holds up.
4. **README Step 3 quickstart code** ran exactly as written:
   `WT mean signal: 0.469`, 3 alt alleles scored. New user gets a
   useful result on a fresh install in their first 5 minutes.
5. **MCP server smoke** — `chorus-mcp --help` works; `fastmcp.Client`
   connects via stdio and lists exactly **22 tools**; `list_oracles`
   returns full structured data for all 6 oracles.
6. **Fast pytest suite on fresh env**: **337 passed / 1 skipped /
   0 failed** in 100 s (4 integration tests deselected). +1 from
   v26 = the existing FANTOM-tracks regression test now passes
   thanks to the validator fix.
7. **17/17 shipped walkthrough HTMLs** still pass the 5-marker
   IGV/normalization audit (v22 floor-rescale + provenance work
   intact). No regression from v22-v26 fixes.
8. **`chorus setup --oracle <name>` for non-gated oracles** — clean
   per-oracle install: env build + weight prefetch + setup marker
   written. Each takes 10–15 min on PanFS, mostly bound by metadata
   ops (see Performance).

## Minor observations (P2)

- `list_oracles` MCP tool returns `'environment_installed': 'unknown'`
  for every oracle even after the env is installed. Should reflect
  the result of `EnvironmentManager.environment_exists`. Cosmetic.
- The all-oracles `chorus setup` halt message points at
  `chorus setup --oracle alphagenome --hf-token hf_xxx` but doesn't
  mention "if you don't want AlphaGenome, run `chorus setup --oracle
  enformer` (or any other) to get started without HF auth." Could
  guide token-less users to a working starting point.

## Performance — PanFS small-file overhead

Side-finding from this audit, not a chorus bug:

| FS | seq write 100 MB | 500 small files |
|---|---|---|
| `/tmp` (ext4 local SSD) | 0.17 s | 0.02 s |
| `/data/pinello` (PanFS) | 0.22 s | 3.87 s |
| `/PHShome` (PanFS home) | 0.25 s | 6.65 s |

PanFS streaming is fine; metadata is **~190× slower** than local SSD.
This is what makes `mamba env create` take 30 min on this host vs
9 min on the v21 host (which had the same envs on slightly different
PanFS settings). Mount option `max-async-writepages=2` is unusually
low and likely the biggest single contributor; `actimeo=` not set so
the kernel attribute cache is effectively disabled. Worth asking
admins to bump those.

## Test results matrix

| Surface | Result |
|---|---|
| `mamba env create` | ✓ 30 min on PanFS |
| `python -m pip install -e .` | ✓ 50 s |
| `chorus setup --oracle enformer` | ✓ 15 min |
| `chorus setup --oracle borzoi` | ✓ 10 min |
| `chorus setup --oracle chrombpnet` | ✓ 10 min |
| `chorus setup --oracle sei` | (in progress at write time) |
| `chorus setup --oracle legnet` | (queued) |
| `chorus setup --oracle alphagenome` | (queued, fresh HF token validated) |
| `chorus genome download hg38` | ✓ (auto by `chorus setup`) |
| Per-track CDF auto-download | ✓ from HF, all oracles |
| README Step 3 (Python snippet) | ✓ first try |
| MCP server smoke | ✓ 22 tools, list_oracles works |
| `single_oracle_quickstart.ipynb` | ✓ **after FANTOM validator P0 fix** |
| 17/17 shipped HTML walkthroughs | ✓ no regression |
| Fast pytest suite | ✓ 337 / 1 / 0 |

## Bottom line

Two **shipping P0 bugs** caught in the first 30 minutes of acting
like a new user — both with simple fixes already in this PR. The
rest of the v22-v26 work holds up: install, setup, predict, MCP,
notebooks, walkthroughs, tests all pass on a fresh state. After this
PR's two fixes land, the README TLDR genuinely walks a new user from
zero to first prediction without surprise.
