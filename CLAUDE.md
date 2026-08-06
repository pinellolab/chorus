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
mamba run -n chorus-sei          # PyTorch
mamba run -n chorus-legnet       # PyTorch
```

Two oracles have **no env of their own** and are missing from the list above, which
cost two failed builds in the 2026-08-06 fleet rebuild: **cherimoya** and
**epinformerseq** both `import torch` inside their builders, and the base `chorus` env
has no torch. Run both under `chorus-borzoi` (torch 2.12.1 + CUDA; both import cleanly).

Pass `--gpu N` to the five builders that accept it rather than relying on
`CUDA_VISIBLE_DEVICES` alone. They used to *overwrite* the env var with the `--gpu`
default of 0, so two processes launched with different env values both landed on GPU 0 —
the first took 78 GB and the second failed every forward pass with `Attempting to
perform BLAS operation using StreamExecutor without BLAS support`, silently dropping all
5,968 positions. An explicit env var now wins, but `--gpu` is unambiguous. `legnet` and
`epinformerseq` have only `--device`, so they do need the env var.

`--part` differs: most take `both`, but **epinformerseq takes `all`**.

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

## Regeneration

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
