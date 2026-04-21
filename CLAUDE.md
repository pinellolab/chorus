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

`CUDA_VISIBLE_DEVICES=0|1` respected across all envs. Per-track CDFs
auto-download from
`huggingface.co/datasets/lucapinello/chorus-backgrounds` on first use.

## Regeneration

After any correctness fix (e.g. the ref-allele off-by-one) every
committed example output drifts. Regenerate with:

```bash
python scripts/regenerate_examples.py             # walkthroughs
python scripts/regenerate_multioracle.py --oracle <name>  # per-oracle
python scripts/regenerate_multioracle.py --consolidate    # unified IGV
jupyter nbconvert --to notebook --execute --inplace examples/notebooks/*.ipynb
```

Notebooks must be re-executed on GPU (advanced + comprehensive pull in
multiple oracles; quickstart is CPU-safe).

## Branch flow

Ship branch is `chorus-applications`. Other agents open audit
branches as `audit/YYYY-MM-DD-v<N>-<slug>` and fix branches as
`fix/YYYY-MM-DD-<slug>`. Review then merge into `chorus-applications`;
don't rebase published audit branches.
