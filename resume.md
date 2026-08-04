# Resume notes — chorus, updated 2026-08-03 (rebuild in flight)

Working notes for picking this up mid-stream. Deliberately weighted toward the
things a conversation summary loses: **retracted claims, environment traps,
measured numbers that were expensive to obtain, and decisions not to act.**

Not a changelog — `git log` and the PR bodies carry the narrative. This is what
you need so you don't repeat work or re-assert something already disproved.

---

## 1. Claims that were made and then REFUTED — do not re-assert these

These were all stated confidently before being checked. Each is wrong.

| retracted claim | what is actually true |
|---|---|
| "The IGV panel does not render in any shipped report" | **It renders fine.** That was a headless-Chromium artefact. Headless paints no IGV tracks even though canvas works, the container is sized and the JS is intact. A negative render result from headless proves nothing. |
| "Downloading just the HTML shows no panel; the folder is needed" | **False alarm**, reported then withdrawn by Luca. The HTML has zero relative `src`/`href` refs and a byte-identical solo copy behaves the same as in-folder. No folder dependency, no doc fix needed. |
| "LegNet predicts a single scalar, so its 131 MB payload is pure waste" | It produces one scalar **per 200 bp tiled window** — a real varying signal (3,633 distinct values). The payload was redundant because each value was expanded 200×, not because there was no signal. |
| "The `exp`→`expm1` fix explains the blog's +1.24 → +1.37 gap" | It accounts for **−0.12%**, in the *opposite* direction. |
| "Replace the flat IGV feature cap with a native-resolution rule" | **Would have been worse than the blunt cap.** It reads `track.resolution`, which was *fabricated as 1* for exactly the track that caused the incident — it would have reinstated the 131 MB report verbatim. The cap works *because* it ignores that field. |
| "#127's irreproducibility threatens #125's byte-identical gate" | No. The gate is byte-identical **sample counts** — integer tallies, deterministic. #127 affects predicted values. |
| "ChromBPNet run-to-run drift is ~0.26%" | That figure is **AlphaGenome (JAX)**. ChromBPNet (TensorFlow) is **bit-exact** — three runs, spread `0.000000e+00`. The two do not transfer. |

**Recurring lesson, four separate times:** AST hashes and greps overstate
divergence. `to_cdf_matrix` looked like 4 implementations and is numerically
identical in all 4; `add()` looked like 3 and is identical; Enformer's
`add_batch` looked divergent and is the same loop with a docstring. **Read the
bodies before concluding.**

---

## 2. Environment traps

**Python — never use bare `python`.** Per-oracle envs are isolated:

```
/home/nvidia/miniforge3/envs/chorus/bin/python              # base: analysis, tests, MCP
/home/nvidia/miniforge3/envs/chorus-alphagenome/bin/python  # JAX
/home/nvidia/miniforge3/envs/chorus-chrombpnet/bin/python   # TF
/home/nvidia/miniforge3/envs/chorus-legnet/bin/python       # PyTorch
...also -borzoi, -sei, -enformer, -cherimoya, -epinformerseq, -alphagenome_pt
```

`mamba` is **not** on PATH in the tool shell — use `/home/nvidia/miniforge3/bin/mamba`.

**Genome FASTA lives at `/home/nvidia/chorus/genomes/hg38.fa`**, NOT
`~/.chorus/genomes/`. Several scripts assume the latter and fail.

**Playwright + headless Chromium works, with one non-obvious step.**
`playwright install-deps` needs root and is *not* required:

```bash
/home/nvidia/miniforge3/envs/chorus/bin/pip install playwright
/home/nvidia/miniforge3/envs/chorus/bin/playwright install chromium
export LD_LIBRARY_PATH=/home/nvidia/miniforge3/envs/chorus-browsertest/lib:/home/nvidia/miniforge3/envs/chorus/lib
```

`libnspr4.so` was **already** in the `chorus` env and only needed to be on
`LD_LIBRARY_PATH`. The other five (`libatk-1.0`, `libatk-bridge-2.0`,
`libXrandr`, `libgbm`, `libatspi`) came from the purpose-made
`chorus-browsertest` conda env. `ldd` then reports 0 missing.
**But see §1 — headless is not a valid oracle for whether IGV renders.**

**Credentials.** Do not route raw tokens through Bash — the classifier blocks
it, and it has cost several attempts. Have Luca run `hf auth login --force`
himself. LDlink token belongs in `~/.chorus/config.toml` under
`[tokens] ldlink = "..."`; `chorus/utils/ld.py` resolves arg → env → that file.
The HF token was rotated on 2026-08-01, which **revoked the stored one and broke
gated AlphaGenome downloads** until re-login — expect that if model loads start
failing with "Invalid user token".

**GPUs:** 8×H100, all free most of the time. Pin `CUDA_VISIBLE_DEVICES` per job;
nothing serialises them and parallel setup used to collide.

---

## 2b. 2026-08-03 overnight session — the rebuild is RUNNING

**Six GPU jobs launched ~03:00, all under `XLA_FLAGS=--xla_gpu_deterministic_ops=true`.**
Do not kill them; do not `git checkout` a branch that changes
`scripts/build_backgrounds_*.py` in the worktree they were launched from (they have
already imported, but the `--part merge` step is a NEW process and must see current
code).

| job | GPU | progress at 03:30 | ETA |
|---|---|---|---|
| alphagenome variants (6,000 pos) | 0 | 385/5949 | ~539 min |
| alphagenome baselines (10,500) | 1 | 1410/10500 | ~243 min |
| borzoi variants | 2 | 2100/5949 | ~39 min |
| borzoi baselines (31,500) | 3 | 2800/31500 | ~219 min |
| enformer variants | 4 | 1950/5949 | ~43 min |
| enformer baselines (31,500) | 5 | 2700/31500 | ~223 min |

AlphaGenome variants is the long pole (~9 h → done ~12:10). Logs: alphagenome writes
`logs/bg_alphagenome_{part}.log`; **borzoi and enformer append `_gpu{N}` to the
filename**, and the live stream is `/data/chorus_data/rebuild_{oracle}_{part}.log`.

**AFTER they finish, `--part merge` still has to run** — that is what writes the
final NPZ. It has not been run and the numbers have NOT been validated. Nothing has
been published to HuggingFace; the published files are still the old ones.

### What changed in the builders, and why the counts will look different

* effect positions are now **gene-anchored**, not uniform (#83). Measured: median
  distance to nearest TSS 102,333 bp → **9,430 bp**; within 1 kb 1.4 % → **21.3 %**;
  within 100 bp of a splice junction 2.3 % → **37.4 %**.
* RNA emits one sample per **(gene, track)**, not one per track over a
  chromosome-pooled mask (#144 inst. 3). At SORT1 the old mask was **128,663 bins**
  against a per-gene median of **4,123** — a 31x mismatch.
* So `effect_counts` will show **two separated clusters** (non-RNA vs RNA). That is
  correct. A **consecutive run** like enformer's old 9600-9606 is the bug (#123).
  Enformer's smoke run now gives `1 distinct, range 9-9`; borzoi `2 distinct, 9-44`.

### Corrections made to my own earlier claims

* **"Borzoi's track_ids are opaque FANTOM accessions with no other way to tell CAGE
  from RNA" — WRONG.** Luca corrected this. All **7,611/7,611** resolve against the
  vendored `borzoi_source/borzoi_metadata.py` (`description` carries `CAGE:`/`RNA:`).
  So #124 never gated the Borzoi rebuild. Provenance is still worth having, for a
  weaker reason: the join binds a row to whatever version of that file is on disk.
* **"the 1 Mb window contains a gene" was the wrong metric** for justifying the
  region set — 94 % vs 85 % barely discriminates. Distance from the *variant* to the
  anchor is what matters, since CAGE is scored in a 501 bp window centred on it.
* AlphaGenome is **bit-exact within one process** with no flags. #127 was
  per-process compilation; two processes on the *same* GPU diverged as much as on
  different ones. `--xla_gpu_deterministic_ops=true` fixes it for +0.6 s/pass;
  `--xla_gpu_autotune_level=0` also works but costs **180x** and is not used.

### Still open before publishing

1. run `--part merge` for all three oracles;
2. verify: no consecutive-integer runs in any `*_counts`; grid invariant passes
   (`tests/test_background_grid_integrity.py -m integration`);
3. regen the 13 walkthroughs — this turns the **integration-marked staleness guard**
   in `tests/test_window_span_parity.py` green. It is currently RED on purpose:
   258 of 1,090 windowed example rows moved (#148) and examples have not been
   regenerated;
4. CHANGELOG with a per-change attribution column, then three HF revisions.

**Luca authorised the HF upload after the rebuild**, but the before/after numbers
have not been shown to him yet — do that first.

## 3. State as of now

`main` is green: **646 passed, 14 skipped, 0 failed** (`pytest -m "not integration"`).
CI runs `pytest tests/ --ignore=tests/test_smoke_predict.py -m "not integration"` = 505 of 526 collected.

**Open PRs — merge #140 before #141** (they both touch #125; #141 deliberately
excludes chrombpnet so they don't conflict):

- **#140** chrombpnet builder onto the shared primitives (61 lines deleted)
- **#141** six more builders onto the shared `ReservoirSampler` (302 deleted)

**Merged 2026-07-31/08-01:** 113, 114, 117, 118, 119, 120, 121, 130, 131, 132,
134, 136, 137, 138.

**HuggingFace:** `lucapinello/chorus-backgrounds` `chrombpnet_pertrack.npz` is
**753 rows**, sha256 `76f267dc862edc86052f2b25a2a8520e960dd193ad39c4ecd19e32b8a8546553`.
Verified by fresh *unauthenticated* download. 9 human ATAC/DNASE + 744 CHIP.

---

## 4. Open issues, and what is actually true about each

| # | one-line status |
|---|---|
| **#83** | The big one. Percentiles are over-optimistic on **every** oracle, not just AlphaGenome. Plan: derive each track's floor from its own background CDF at one global quantile `q` (recommend 0.90) rather than hand-setting ~25 constants. **Awaiting Luca's `q` decision.** |
| **#122** | AlphaGenome builder passes `description` (no mark name) to `classify_chip_layer`, so 0 of 2,733 CHIP tracks classified as histone → **1,075 tracks compare a 2001 bp statistic to a 501 bp null.** One-line fix; correcting the rows needs a rebuild. |
| **#123** | Enformer `effect_counts` takes 7 values (9,600–9,606) — one per-variant `try/except` wraps the per-track loop, so its tracks aren't ranked against the same variant set. |
| **#124** | Species/genome consistency is enforced *by accident*. Also `AlphaGenomeOracle(organism="mouse")` is accepted, stored, and **never read**. |
| **#125** | In progress — see §5. |
| **#127** | AlphaGenome only: two identical runs differ on 454 fields with **36 sign flips**; for CAGE the noise *exceeds* the effect. **`AUDIT_CHECKLIST.md:247`'s advice to "compare percentiles rather than raw values" is measurably wrong** — quantiles changed in 80 of 100 rows, worse than raw. |
| **#128** | A ref/genome mismatch only *warns*, then scores a synthetic sequence. 1 of 4 examples was wrong (BCL11A). |
| **#133** | **Fixed 2026-08-04.** The path now points at `examples/walkthroughs/`, but only because it ships with a refuse-rather-than-degrade guard: the rehydrated HTML is rendered to a temp file and compared against the artefact on disk, and the write is refused below 50% of the incumbent size. Measured on the current tree, all 14 rehydratable reports come back at 0.3–1.0%, so every one is refused and nothing is written — `--check` exits 1 having touched no file. `from_dict` still drops the per-bin IGV arrays; the guard makes that loud instead of silent. `--force` overrides. |
| **#135** | Reframed. Don't add a report-level feature cap — **encode the payload compactly**: 58 of 60.8 bytes per feature are `{"chr":…,"start":…,"end":…}` scaffolding and all 60 wig tracks are perfect regular grids, so values-only + a `(chr,start,step)` header is **12.8× smaller with zero loss** (13.99 MB → 1.09 MB). |
| **#139** | Only one verified finding survives: `genome:"hg38"` triggers **six** external fetches, so the "viewable offline" claim in `_ensure_igv_local` is false. A locus report needs chrom bounds + optionally the genes in the window (already in `annotations/gencode.v48.basic.annotation.gtf`), not the 3 GB sequence. |

---

## 5. #125 migration — where it stands and the traps

`chorus/analysis/background_sampling.py` is the shared module (merged, #137).
Its equivalence tests extract each builder's class **by AST and exec it**, so
they compare against live source, not a snapshot.

Per-builder state (after #140 and #141 merge):

```
ReservoirSampler   7 of 8 migrated; alphagenome still local
one_hot_encode     chrombpnet only
compute_effect     chrombpnet only (as a 2-arg wrapper)
get_sequence       none migrated
score_window_sum   none migrated
```

**Only four genuine behavioural divergences exist across ~30 duplicated
definitions.** Each is a *parameter* in the shared module, not a compromise —
unifying any silently would move a shipped background:

| divergence | parameter |
|---|---|
| AlphaGenome capacity **20,000** (others 50,000) | `ReservoirSampler(capacity=)` |
| LegNet N-threshold **0.3** (others 0.5) | `get_sequence(max_n_fraction=)` |
| log2fc / logfc / diff | `compute_effect(formula=)` |
| EPInformer-seq **`(4, L)`** | `one_hot_encode(channels_first=)` |

Whether LegNet's 0.3 is deliberate for a 200 bp window is **its owner's call** —
don't settle it by taking the majority value.

**Traps found while migrating ChromBPNet — check for these in every builder:**

- `get_sequence` takes a **pysam** handle (`ref.fetch`, `ref.get_reference_length`);
  the shared one takes pyfaidx and slices. Unifying changes which positions are
  accepted as background samples.
- `score_window_sum` spans `2*(WINDOW_BP//2)+1` bins — **odd, inclusive** —
  vs the shared `window_bp // resolution`. For `WINDOW_BP=1000` that's
  **1001 vs 1000**, which would shift every ChromBPNet activity value.

**AlphaGenome is deferred on purpose.** Its 37-line vectorised `add_batch` is
**proven equivalent** to the 3-line loop, so this is performance not
correctness — but it has the largest build (5,168 tracks). Port the fast path
*into* the shared module under the existing equivalence test rather than
dropping it.

**Two anti-patterns already fixed here — don't reintroduce them:**

1. A migration deletes the evidence that justified it. Both comparison tests now
   **assert the shared import** when a builder has no local class, instead of
   skipping. Verified by breaking the import and watching it fail. Watch the
   skip count: it jumped 3 → 12 on #141, which is what exposed the second one.
2. The pre-migration CDF is pinned as **golden values** in
   `tests/test_background_sampling.py` (counts `[150,150]`, two 8-point rows to
   1e-9), captured from chrombpnet's copy immediately before deletion. That keeps
   the numerics checked after the last local copy is gone. If it fails, every
   shipped background now differs from the one on HuggingFace.

---

## 6. Blog post — `audits/2026-006_chorus_blogpost.revised.md`

Luca's decision: **fix the in-repo revised draft only**, not the live site. The
**published post at genomicsxai.github.io diverged from that file** — it shipped
the pre-review Enformer window (196,608; correct is **393,216**) while picking up
the post-review tool count. Reconcile against the file, not the published text.

The draft now carries 15 numbered corrections. Both worked examples are
reproducible from committed artefacts:

- `validation/SORT1_rs12740374_with_CEBP` — 11 tracks, all four C/EBP figures
  reproduce within rounding (CEBPB +3.044 vs blog's +3.1, CEBPA +2.764/+2.8,
  CEBPG +2.269/+2.3, CEBPD +1.818/+1.9). CEBPG and CEBPD were in **no** example
  before.
- `causal_prioritization/CDYL_rs9504151` — rs9504151 ranks **#1 of 56**,
  composite **0.991** vs the audit's recorded 0.995, effect **−1.362** vs −1.363.
  Reproduces with **no LDlink token** (`ld_proxies.tsv` is committed).
  **Resolves the audit's 54-vs-56 discrepancy:** `snvs_only=True` → 56
  (1 sentinel + 55 proxies), `False` → 64.
  Its HTML is **not committed** — 25.70 MB, over the ceiling; set
  `CHORUS_WRITE_LARGE_HTML=1` to write it locally.

Still wrong in the published post: the oracle count ("seven" → **eight models /
nine registered**, with Cherimoya absent entirely), and the "floating-point
drift" explanation (see §1).

---

## 7. Numbers worth not re-deriving

- ChromBPNet SORT1 `DNASE:HepG2` effect: **+1.375621**, bit-exact across runs.
  At `4ad7be7` it was **+0.317674** (the 1 bp auto-region bug). With `exp`
  instead of `expm1`: **+1.373940**.
- Committed examples pre-refresh had a median `|percentile|` of **exactly
  1.0000**; saturation at exactly 1.0 was **76–100%** across 11 of 13 examples,
  now **0–41%**. The guard threshold is 60% with ~20 points of margin.
  A *median* threshold was rejected — `SORT1_chrombpnet` scores one track at
  0.9995 off a genuine +1.376 effect.
- FTO's committed quantile `0.38082765845992667` is **exactly 727/1909** — the
  numerical fingerprint of the pre-#119 denominator (5.24× inflation).
- IGV reports gzip to **12–17%** of on-disk size. Artefact ceiling is **50 MiB**
  (`tests/test_committed_examples.py`); GitHub's hard wall is 100 MiB.
- LegNet after the resolution fix: **5,243** features (= `ceil(1048576/200)`),
  **3,633** distinct, range −1.1333..+1.1317. Before: 3,987 / 2,344 /
  −0.9473..+1.1317 — the cap was hiding 1,289 values *and* clipping the minimum.
- Per-oracle effect background sizes: AlphaGenome **1,697–1,909**, Enformer
  9,600–9,606, Borzoi 6,563–9,609, Sei/LegNet 9,609, EPInformer-seq 9,608,
  Cherimoya/ChromBPNet 18,672, ChromBPNet CHIP 37,344.
- Canonical track counts live in `audits/AUDIT_CHECKLIST.md:205` and it says
  **"No doc may disagree"** — update it whenever a count changes.

---

## 8. Outstanding for Luca

- **#83's `q`** — the one global quantile that derives every per-track floor.
  Recommend 0.90. Everything else in Stage 0 is derived from it.
- The **HF token is in this transcript in plaintext** (the rotated one too). If
  the transcript is ever shared, rotate again.
- Whether **LegNet's 0.3 N-threshold** is deliberate (§5).
- **Mouse support tier** — currently dropped entirely; re-adding needs an mm10
  FASTA *and* an mm10 region set (SCREEN has mm10 cCREs; the DHS vocabulary does
  not).
