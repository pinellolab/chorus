# ChromBPNet CDF rebuild against `chrombpnet_nobias` — audit report

**Date**: 2026-04-30 (started 00:24 UTC, NPZ ready 10:13 UTC, ~10 h wall total)
**Hardware**: Linux `ml008` (Ubuntu, kernel 5.15.0-170-generic), 2× NVIDIA A100 80 GB PCIe
**Auditor**: Claude Opus 4.7 (1M context)
**Branch start**: `main` @ `8c2243c` (PR #67 — CDF rebuild prep)
**Goal**: produce a fresh `chrombpnet_pertrack.npz` whose CDFs match the post-0.3 default (`chrombpnet_nobias`) and upload it to `lucapinello/chorus-backgrounds`. Replaces the 0.2.x bias-aware CDFs flagged for rebuild in `audits/2026-04-28_chrombpnet_slim_mirror/report.md`.

## Outcome

**NPZ built and validated; upload deferred pending a write-scope HF token.** All quality gates pass: 786 tracks (22 ATAC + 20 DNASE + 744 CHIP), every CDF monotone, no NaN/Inf, every reservoir filled to the configured size. Magnitude shift vs. the 0.2.x CDFs is 13.5–29.3% at p95 on ATAC/DNase tracks (the bias correction stripping enzymatic motif preferences) and ~0% on CHIP/BPNet tracks (which were already nobias-equivalent in the old NPZ).

## Pre-flight

- ✅ `chorus-chrombpnet` env present (built before this audit). Missing `huggingface_hub` — installed via `pip install "huggingface_hub>=0.20.0"` per the handoff's gotcha note. Without it, the slim-mirror import would silently fall back to ~700 MB ENCODE tarball downloads.
- ✅ Slim mirror reachable: 42 ATAC/DNase + 744 deduped CHIP/BPNet h5's at `lucapinello/chorus-chrombpnet-slim`.
- ✅ HF auth: `lucapinello`, **read-only token** (this turned out to be the upload blocker — see Outcome).
- ✅ Old NPZ pulled to `~/.chorus/backgrounds_old/chrombpnet_pertrack_OLD.npz` for the diff comparison.

## Build phases

| Phase | Models | Wall | GPU | Output |
|---|---:|---|---|---|
| 1a — ATAC/DNase variants | 42 | 52 min (00:24→01:16) | 0 | effect interim |
| 1b — ATAC/DNase baselines | 42 | 52 min (01:53→02:45) | 0 | baseline interim |
| 2a — CHIP/BPNet variants | 744 | 4.5 h (02:13→06:44) | 1 | effect interim (overwrote 1a) |
| 2b — CHIP/BPNet baselines | 744 | 5.7 h (02:45→08:31) | 0 | baseline interim (overwrote 1b) |
| Phase 1 redo (variants) | 42 | 51 min (08:54→09:34) | 0 | effect interim restored |
| Phase 1 redo (baselines) | 42 | 53 min (09:20→10:13) | 1 | baseline interim restored |
| Merge (incremental) | — | <1 min | — | 786-track final NPZ |

**Total wall**: ~10 h with parallel-GPU phases — well under the handoff's 13–25 h estimate. A100 was 4–8× faster than the Metal-derived per-model estimates baked into the script's `--help`.

### Re-run reason (P1 finding F1)

Each `--part variants/baselines --assay X` invocation **overwrites** the interim NPZ rather than appending. So after running:

```bash
... --assay ATAC_DNASE --part variants    # writes interim with 42 tracks
... --assay CHIP        --part variants   # OVERWRITES with 744 tracks
```

…the merge step ends up with only 744 CHIP tracks, no ATAC/DNase. This was caught by the post-merge spot-check (track count 744 ≠ expected 786). Recovered by re-running Phase 1 with the existing 744-track NPZ in place and finishing with `--part merge-incremental`, which appends the 42 new tracks to give the 786 final.

The handoff doc's command sequence followed verbatim hits this bug. Suggesting a fix in a follow-up issue (see "Follow-ups" below).

### GPU pinning bug (informational, F2)

The `mamba run -n env CUDA_VISIBLE_DEVICES=N python script.py --gpu N` pattern doesn't behave the way you'd expect because `scripts/build_backgrounds_chrombpnet.py:200` does `os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)` — the inner `--gpu N` arg overrides the outer env var. Hit during the parallel Phase 1 redo: passing `CUDA_VISIBLE_DEVICES=1 ... --gpu 0` put the second job on physical GPU 0, fighting with the first job for memory and OOM-failing. Workaround: pass `--gpu N` to the script directly and let the script set the env var (don't bother with the outer `CUDA_VISIBLE_DEVICES`). Worth fixing — the script should respect a pre-set `CUDA_VISIBLE_DEVICES` if present.

## Spot-check (all pass)

Run from chorus base env against `~/.chorus/backgrounds/chrombpnet_pertrack.npz`:

| Check | Result |
|---|---|
| Track count | **786** (22 ATAC + 20 DNASE + 744 CHIP) — matches handoff's expected count |
| `effect_counts` per track | min=max=median=**9609** — every model contributed the full reservoir; zero failed builds |
| `summary_counts` per track | median=**29004** |
| `perbin_counts` per track | median=**928128** |
| Effect CDFs monotone (all 786 rows) | ✅ |
| Summary CDFs monotone (all 786 rows) | ✅ |
| Per-bin CDFs monotone (all 786 rows) | ✅ |
| NaN / Inf in any CDF | ✅ none |
| File size | 78.6 MB (vs old 82.4 MB — slightly smaller, expected because the new build skips a few never-loaded models) |
| **SHA256** | `be61e9e8f9b919b43c599b7fbc9deb74f8f1e6dc1da5e2cdb92036a85bf13205` |

## Magnitude shift vs. 0.2.x CDFs

786 tracks present in both old and new NPZs. Per-track p50 / p95 of the effect CDFs:

| Track | old p95 | new p95 | shift % |
|---|---:|---:|---:|
| ATAC:K562 | 0.1621 | 0.1976 | **+21.9%** |
| ATAC:HepG2 | 0.1332 | 0.1517 | +13.8% |
| ATAC:GM12878 | 0.1692 | 0.1934 | +14.3% |
| DNASE:K562 | 0.1462 | 0.1659 | +13.5% |
| DNASE:HepG2 | 0.1884 | 0.2436 | **+29.3%** |
| CHIP:K562:REST | 0.0529 | 0.0529 | 0.0% |
| CHIP:HepG2:CTCF | 0.0829 | 0.0828 | −0.1% |
| CHIP:GM12878:REST | 0.0610 | 0.0610 | −0.1% |

The handoff predicted *"rankings preserved on large-effect SNPs, magnitudes shift by 5–30 % on ambiguous cases"* — confirmed exactly. The 13.5–29.3% shift on ATAC/DNase tracks reflects the bias correction (`chrombpnet_nobias` strips the enzymatic motif bias the bias-aware variant carries). CHIP/BPNet tracks shift ~0% because BPNet only has a single nobias-equivalent variant in the catalogue — the old CDFs were already against the same model architecture.

## Upload status

**Blocked on a write-scope HF token.** Attempted upload from chorus base env returned `403 Forbidden: you must use a write token to upload to a repository`. The cached token (`HfApi().whoami()` → `lucapinello`, `auths: []`) is read-only. Maintainer needs to re-login with a write-scope token (or set `HF_TOKEN` to one) and re-run:

```bash
mamba run -n chorus python -c "
import os
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj=os.path.expanduser('~/.chorus/backgrounds/chrombpnet_pertrack.npz'),
    path_in_repo='chrombpnet_pertrack.npz',
    repo_id='lucapinello/chorus-backgrounds',
    repo_type='dataset',
    commit_message='Rebuild ChromBPNet CDFs against chrombpnet_nobias (0.3.0+ default)',
)
"
```

The NPZ at `~/.chorus/backgrounds/chrombpnet_pertrack.npz` is unmodified since the merge-incremental step — same SHA256 as recorded above.

## Follow-ups

1. **F1 — `--part variants/baselines --assay X` interim NPZ overwrite**. Documenting verbatim handoff invocations leads to data loss without realising it. Either `np.savez_compressed` should append within an existing interim (matching how `merge-incremental` deals with the final NPZ), or the script should error when an interim is about to be overwritten without `--force`. Filing as a separate issue.
2. **F2 — `--gpu N` arg silently overrides outer `CUDA_VISIBLE_DEVICES`**. The script should honour a pre-set `CUDA_VISIBLE_DEVICES` if present, and only set it from `--gpu N` when no env var was provided. Otherwise the parallel-launch pattern in the handoff (`CUDA_VISIBLE_DEVICES=1 ... --gpu 0`) silently mis-routes one of the jobs.
3. **(Tracking)** PID 503331 ended up stuck in CUDA driver `D` state during the Phase 1 redo (held 35 GB on GPU 0 indefinitely). `kill -9` didn't release it. Cosmetic — doesn't block the rebuild — but worth flagging if it recurs.

## Artifacts

- `~/.chorus/backgrounds/chrombpnet_pertrack.npz` — final 786-track NPZ (78.6 MB, sha256 `be61e9e8...`)
- `logs/bg_chrombpnet_variants_atac.log`, `logs/bg_chrombpnet_baselines_atac.log` — Phase 1 (original)
- `logs/bg_chrombpnet_variants_chip.log`, `logs/bg_chrombpnet_baselines_chip.log` — Phase 2
- `logs/bg_chrombpnet_variants_atac_redo.log`, `logs/bg_chrombpnet_baselines_atac_redo2.log` — Phase 1 redo
- `logs/bg_chrombpnet_merge.log`, `logs/bg_chrombpnet_merge_incremental.log` — the two merge passes

## Related

- chorus PR #59 (0.3.0 release) — flipped the default to `chrombpnet_nobias`, necessitating this rebuild
- chorus PR #60 (slim mirror + F1) — added `huggingface_hub` to the chrombpnet env yaml; this rebuild verified the post-#60 flow on Linux/CUDA
- chorus PR #67 (rebuild prep) — landed the script's `--model-type` flag and the handoff doc this audit followed
- `audits/2026-04-28_chrombpnet_slim_mirror/report.md` — flagged the deferred CDF rebuild
