# Installation-instructions audit — 2026-08-13

Every install claim checked against what the code and CLI actually do, on `main` at `9799a0c`
(v0.7.3 + #186). Five areas were audited by independent read-only agents — README install flow, env
files and names, undocumented prerequisites, environment variables documented vs read by the code,
and quantitative install claims — and each area's findings were then handed to a second agent told to
**refute** them. 59 problems survived that pass.

**Verdict: the installation instructions were not current, and one finding blocks installs.** No code
defect, no percentile movement; the fixes are documentation plus four user-facing strings in the CLI.

---

## The finding that matters

### The disk prerequisite was wrong by more than 2×

`README.md:20` asked for **~38 GB free disk**. Measured with `du -sh` on this Linux x86_64 + CUDA box
after a default `chorus setup`:

| bucket | documented | measured |
|---|---|---|
| 7 non-Cherimoya oracle envs | ~20 GB ("~3 GB each") | **53 GB** (6.1 GB smallest, 11 GB largest) |
| `alphagenome_pt` env | ~2.6 GB | **7.5 GB** |
| Cherimoya env | ~7 GB | 6.6 GB ✓ |
| base `chorus` env | *absent* | **2.7 GB** |
| default weight prefetch | *absent* | **~11 GB** (Sei alone 6.5 GB) |
| hg38 + index | ~3 GB | 3.1 GB ✓ |
| per-track CDF backgrounds | ~2 GB | 1.9 GB ✓ |
| **total** | **~38 GB** | **~85 GB** |

Two independent causes, and both are worth recording because each would have recurred:

**1. The per-env number was measured on the wrong platform.** "~3 GB each" is plausible for macOS
arm64, where `chorus/core/platform.py` strips the CUDA packages. On Linux every env carries its own
CUDA payload, arriving two different ways — pip `nvidia_*` wheels in the Enformer (2.9 GB), ChromBPNet
(2.9 GB), AlphaGenome (4.4 GB), AlphaGenome-PyTorch (2.7 GB), EPInformer-seq (2.7 GB) and Cherimoya
(2.7 GB) envs, and conda-side `libtorch_cuda`/`libcu*` in Borzoi (~1.9 GB), LegNet (~3.8 GB) and Sei
(~1.9 GB). `du -sc` across all nine — which counts a hardlinked file once — still gives 67 GB, so
there is no dedup credit hiding in it. The row is now labelled by platform rather than restated as a
single number.

*An adversarial pass corrected my own first fix here*: I wrote "~2.9 GB in *every* oracle env,
pip-installed so not shared", which is false for Borzoi, LegNet and Sei — they have no pip `nvidia`
directory at all. The mechanism differs per env even though the conclusion does not.

**2. The table had no row for weights at all.** `chorus setup` prefetches default weights for every
oracle (`chorus/cli/_setup_prefetch.py`), so a user adding the rows up by hand still landed ~11 GB
low. Sei is the extreme case: `sei.pth` is 3.32 GiB and the 3.1 GB `sei_model.tar.gz` it came from is
**kept beside it**, which is 3.1 GB of pure duplication a user can delete and now knows about.

Guarded by `tests/test_disk_claims_add_up.py`: the table's rows must sum to its own stated total
(1 GB tolerance for rounding) and the prerequisite bullet must not promise less than that total.
Mutation-tested against both the historical `~38 GB` and an over-correction.

---

## Broken commands and missing steps

| what | why it failed |
|---|---|
| `chorus setup all` | the subparser defines **only flags** — no positional — so this exits 2 with "unrecognized arguments: all". It was the documented prerequisite in `examples/walkthroughs/README.md`, and shipped in `chorus/utils/ld.py`'s hint and `_setup_all.py`'s own failure banner |
| `nohup chorus setup &` | the HF token resolves **before** any env is built and hard-fails when stdin is not a TTY, so the backgrounding the TLDR recommends aborted instantly with zero progress. `HF_TOKEN` / `--hf-token` / `--no-weights` now documented at that step |
| opening any shipped notebook | 16 of 19 declare kernel name `chorus`; nothing in the package registers it (`grep -rn "ipykernel\|kernelspec" chorus/` → 0 hits), yet the README said all three "work as soon as `chorus setup` finishes". `nbconvert` raises `NoSuchKernel` |
| `claude mcp add chorus -- …` | defaults to `--scope local`, so the recipe advertised as making chorus "available in every project" registered it for the clone only. Needs `-s user` |
| `chorus cleanup --all` | never touches the HuggingFace cache (0 hits for `hugging` in `_cleanup.py`) — ~20 GB, where most weights live — while the docs called it "Remove everything" |
| `$(chorus config data-dir)/genomes` | in this repo's own air-gap recipe; the command prints a multi-line human-readable block, not a path |
| a token-free `chorus setup` | `environments/README.md` said "no HuggingFace account" required, but the default flow resolves the token first and returns 1 **before building anything**, so the eight token-free oracles are not installed either |

## Wrong paths and stale numbers

* **`~/.chorus/backgrounds/` in every live place it appeared** (32 lines, 30 files) — README, four docs, two CLI `--help` strings, four
  `normalization.py` docstrings stating it as the `cache_dir` default, plus `scorers.py`,
  `result.py`, `_track_figure.py`, `build_backgrounds.py`, two builders and the audit runbook. The
  data directory has defaulted to the **installation tree** since 2026-08; `~/.chorus` is only the
  fallback for an unwritable install tree. Historical `audits/` entries and the "used to live in
  `$HOME`" narratives in `globals.py` and `CLAUDE.md` were deliberately left alone — they are correct
  as records.
* **`CHORUS_DATA_DIR` appeared in no user-facing doc** — the one switch that relocates all 85 GB
  existed only in `--help` output and `CLAUDE.md`. It now has its own README section, including the
  **backgrounds-only legacy rule** (`resolve_backgrounds_dir`) that returns `~/.chorus/backgrounds`
  when it already holds `*_pertrack.npz`. That rule is why `chorus config data-dir` can print a
  `backgrounds` path outside `data_dir`, which looked like a contradiction until it was traced.
* **"~1.4 GB" for the 2-model ChromBPNet default**, in three `_setup_prefetch.py` comments; the slim
  mirror serves 25 MB per model, so ~50 MB. The tell was internal: the same line put the full
  786-model catalogue at ~1.5 GB.
* **`alphagenome_pt` at "~1.7 GB env"** — measured 7.5 GB.
* **The backgrounds table omitted `cherimoya_ensemble_pertrack.npz`** — nine NPZs ship at the pinned
  revision, the table listed eight.
* **`environments/chorus-base.yml`** is documented as vestigial: nothing reads it (the manager
  *excludes* it and a test asserts that), it is not the subset the docs claimed in either direction,
  and it declares `name: chorus`, so installing it collides with the documented base env.
* **Three dead in-page anchors** — `#mcp-server` (the only link from the pitch into the 24-tool
  catalogue), plus two in `API_DOCUMENTATION.md` and one in `NORMALIZATION_GUIDE.md`. Now guarded by
  `tests/test_doc_links_resolve.py`, which encodes GitHub's slug rules — including that a space is
  *not* collapsed, so `A — B` is `a--b`. Getting that wrong gave the guard's first draft eight false
  positives.
* **`CONTRIBUTING.md` had no "running the tests" section at all**, and the browser suite added in this
  release (playwright + chromium) was documented nowhere outside CI. It also still said "All six core
  oracles" when there are eight.
* **`environment.yml` did not declare `pyyaml`**, which `chorus/core/environment/manager.py` imports;
  it worked only transitively.

## Verified correct — recorded so nobody "fixes" them

Measured and left alone: `~2 GB` backgrounds (nine NPZs sum to 1.907 GiB); `~3 GB` hg38 (3.1 GB);
`~50 MB` ChromBPNet slim fast-path (25 MB × 2); Cherimoya env `~7 GB` (6.6); EPInformer-seq weights
`~11 MB`; `python_requires>=3.10` matching `python=3.10`; `environment.yml` solves (`mamba env create
--dry-run`); `pip check` clean; jupyter/notebook/ipykernel genuinely *are* in `environment.yml`; and
the MCP one-liner `mamba run -n chorus chorus-mcp` answers a JSON-RPC `initialize`.

**One false positive worth naming.** An agent reported `| Sei | ~2.8 MB |` as a 1000× error against
Sei's 3.32 GiB checkpoint. That table is the **backgrounds NPZ** table ("Tracks covered"), not a
weights table, and every row matches the files on disk — `sei_pertrack.npz` really is 3 MB. The claim
was checked before being acted on, which is the only reason it was not "fixed" into being wrong.

## Verified by running it

The README's "3. Your first prediction" snippet was copied verbatim and executed on GPU 4:

```
WT mean signal: 0.468
Variant result: scored 3 alt alleles (['reference', 'alt_1', 'alt_2', 'alt_3'])
```

Both printed lines match what the README says they will.

## Deliberately not done

The audit surfaced 59 problems; the ones above are fixed. Left for a follow-up because they are
neither install-blocking nor doc-currency issues:

* `chorus cleanup --all` could *offer* to remove the HF cache rather than only being documented as not
  doing so — a CLI change, not a doc fix.
* `environments/README.md`'s GPU-support table and "Adding New Oracles" steps, and
  `docs/IMPLEMENTATION_GUIDE.md:293-311`, drifted from the current oracle set. Contributor-facing.
* `chorus/mcp/server.py`'s `--help` and `chorus/mcp/state.py:36` disagree with the README's MCP
  section on defaults.
* `_datadir.py`'s `_fmt_size` reported `hf_cache` as 41.5 GB where `du -shL` gives 20 GB — a
  double-count of hardlinked blobs in the size display only.

## Tests

Fast suite on the fixed tree: **1,792 passed, 29 skipped, 0 failed** (5 m 30 s), including the two
new guards (5 + 13 assertions, both mutation-tested).
