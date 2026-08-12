# Chorus — notes for Claude sessions

## Audit discipline

Before any ship-prep, release, or "is this ready?" review, run the
audit checklist:

- **[`audits/AUDIT_CHECKLIST.md`](audits/AUDIT_CHECKLIST.md)** — 18-section
  reusable runbook (Install → HF gate → GPU → CDFs → Python API →
  Notebooks → HTML reports → MCP → Error paths → Repo consistency →
  Tests → Reproducibility → Determinism → Edge cases → Offline →
  Logging → Dependencies → License). Every check has an exact command
  and a P0/P1/P2 severity.

When an audit uncovers findings, write a dated report in
`audits/YYYY-MM-DD_<short-name>.md` following the format used by
`2026-04-21_v18_fresh_full_audit.md` (what was run, what was fixed,
what was deferred, tests-pass summary).

## Environments

Oracle envs are isolated — their deps don't coexist. Always run per-oracle
work through the matching mamba env:

```bash
mamba run -n chorus              # base (MCP, analysis, reports)
mamba run -n chorus-alphagenome  # JAX
mamba run -n chorus-enformer     # TF
mamba run -n chorus-chrombpnet   # TF
mamba run -n chorus-borzoi       # PyTorch
mamba run -n chorus-cherimoya    # PyTorch + Triton (CUDA or CPU)
mamba run -n chorus-sei          # PyTorch
mamba run -n chorus-legnet       # PyTorch
```

The list above is **incomplete** — `conda env list` shows more, and two of the missing
ones are needed:

```bash
mamba run -n chorus-cherimoya      # PyTorch 2.13 + the `cherimoya` package
mamba run -n chorus-epinformerseq  # PyTorch
```

Both builders `import torch`, and cherimoya additionally imports the `cherimoya`
package itself, which exists **only** in `chorus-cherimoya`. Running it elsewhere does
not fail fast: it logs `Failed to load <track>` once per track and carries on, so a
1,518-track run spent 75 minutes loading nothing before dying at the provenance step.
Always check `conda env list` rather than trusting this section.

Pass `--gpu N` to the five builders that accept it rather than relying on
`CUDA_VISIBLE_DEVICES` alone. They used to *overwrite* the env var with the `--gpu`
default of 0, so two processes launched with different env values both landed on GPU 0 —
the first took 78 GB and the second failed every forward pass with `Attempting to
perform BLAS operation using StreamExecutor without BLAS support`, silently dropping all
5,968 positions. An explicit env var now wins, but `--gpu` is unambiguous. `legnet` and
`epinformerseq` have only `--device`, so they do need the env var.

`--part` differs: most take `both`, but **epinformerseq takes `all`**.

`conda run` **buffers stdout**, so a long build's log stays 0 bytes until the process
exits — during the 2026-08-06 rebuild that meant no progress visibility on a 14-hour
job. Use `conda run --no-capture-output ... python -u` when you need to watch a build,
and key failure detection off the exit code rather than off log contents.

`CUDA_VISIBLE_DEVICES=0|1` respected across all envs. Per-track CDFs
auto-download from
`huggingface.co/datasets/lucapinello/chorus-backgrounds` on first use.

## Where downloaded data goes

Bulk data defaults to the **chorus installation directory**, not `$HOME`.
Backgrounds used to land in `~/.chorus/backgrounds/` (7.8 GB) and model
weights in `~/.cache/huggingface/` (12 GB, because nothing set `HF_HOME`);
both now follow one switch:

```bash
export CHORUS_DATA_DIR=/data/chorus_data          # per-shell, highest priority
chorus config data-dir --set /data/chorus_data    # persist for this install
chorus setup --data-dir /data/chorus_data         # choose at install time
chorus config data-dir                            # show what resolved, and why
chorus config data-dir --set PATH --migrate       # move existing backgrounds
```

Resolution order: `CHORUS_DATA_DIR` > `<install>/chorus_data_dir.txt` >
the install dir > `~/.chorus` (only if the install tree is not writable,
e.g. a pip install into system site-packages).

Two things deliberately do NOT follow it: **credentials**
(`~/.chorus/config.toml`, the HF token) stay with the user, because a shared
data dir is the wrong place for a personal token; and **conda environments**
stay with the installation.

## Background nulls

Every percentile is a rank against a per-track background null. Before changing a region
set, a sampling rule, a retention policy or adding an oracle, read:

- **[`docs/BACKGROUND_NULL_PROTOCOL.md`](docs/BACKGROUND_NULL_PROTOCOL.md)** — which
  regions and why, how they are sampled, which SNPs, how the CDFs are computed
  (stratified or not), the guard inventory, a step-by-step for adding a new oracle, and a
  dated decision log with the measurement behind each call.

It is a LIVING document: update it in the same commit as any change it describes. Two
rules it exists to enforce — the effect and baseline nulls are different reference classes
and must not be unified, and **no composition change ships without a two-arm measurement**
(every unmeasured composition guess in this project was wrong).

## Regeneration

**Which script, which oracle, which env.** Getting this wrong is the single most common way a
regeneration silently does nothing: the wrong env raises `ModuleNotFoundError` per oracle and the
script carries on, and an invalid `--oracle` is an argparse error that scrolls past in a tail'd
log. Established by trial 2026-08-12:

| artefact | script | valid `--oracle` | env |
|---|---|---|---|
| walkthroughs | `regenerate_examples.py` | `alphagenome`, `enformer`, `chrombpnet`, `all` | `chorus` (these use `use_environment=True`) |
| per-oracle multioracle | `regenerate_multioracle.py --oracle X` | `chrombpnet`, `cherimoya`, `legnet`, `alphagenome` — **no enformer** | `chorus-X` (each uses `use_environment=False`) |
| unified IGV panel | `regenerate_multioracle.py --consolidate` | — | `chorus` (reads cached per-oracle reports) |
| discovery, causal, region_swap, integration, batch, TERT | `regenerate_remaining_examples.py --only all` | — | `chorus-alphagenome` (`use_environment=False`) |

Enformer has no `--oracle enformer` in the multioracle script; its single-oracle report comes
from `regenerate_examples.py`.

Do **not** write `mamba run -n X --no-capture-output ...` — the flag after `-n` makes the wrapper
die with `exec: --: invalid option` before the script starts, which reads as a completed stage in
a tail'd log. Either put the flag first or call the env's python directly
(`/home/nvidia/miniforge3/envs/chorus-X/bin/python`).


After any correctness fix (e.g. the ref-allele off-by-one) every
committed example output drifts. Regenerate with:

```bash
python scripts/regenerate_examples.py             # walkthroughs
python scripts/regenerate_multioracle.py --oracle <name>  # per-oracle
python scripts/regenerate_multioracle.py --consolidate    # unified IGV
python scripts/generate_walkthrough_notebooks.py  # per-walkthrough .ipynb (codegen)
jupyter nbconvert --to notebook --execute --inplace examples/notebooks/*.ipynb
```

The `generate_walkthrough_notebooks.py` step is codegen-only (writes
the `.ipynb` files declared in its `WALKTHROUGHS` list); re-run it
after editing per-walkthrough args (assay_ids, positions, alleles).
Executing every walkthrough notebook end-to-end requires GPU + the
per-oracle envs, so do it manually before a release rather than in
the regen sweep.

Notebooks must be re-executed on GPU (advanced + comprehensive pull in
multiple oracles; quickstart is CPU-safe).

## Branch flow

Ship branch is `main` — that's what users see. Other agents may open
audit branches as `audit/YYYY-MM-DD-v<N>-<slug>` and fix branches as
`fix/YYYY-MM-DD-<slug>`; review then merge into `main`, and don't
rebase published audit branches. (Earlier guidance named
`chorus-applications` as the ship branch — that was incorrect.)
