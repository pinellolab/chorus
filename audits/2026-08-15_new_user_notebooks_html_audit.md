# New-user audit: notebooks and HTML reports — 2026-08-15

Scope: what a first-time user meets, with the two areas named explicitly — the executed notebooks and
the committed HTML. Run against `5fc2d35` (= `v0.7.3`) plus the fixes in this PR.

## Verified working

| check | result |
|---|---|
| `chorus health` | exit 0, every installed oracle Healthy |
| `chorus list` | exit 0 |
| every `chorus <cmd>` cited in README | all 8 present in `--help` (`backgrounds cleanup config genome health list remove setup`) |
| TLDR snippet, end to end | exit 0 — **64 s** idle H100, **59 s** CPU-only |
| 6 library notebooks | all executed, **157 cells with outputs, 0 error outputs** |
| `single_oracle_quickstart.ipynb` re-executed today | **exit 0, 32/34 cells with output, 0 errors**, ~40 min on CPU |
| 20 committed HTML reports | all 20 browser-tested (40 parametrized cases); **56 passed** |
| runtime network per report | only `hgdownload.soe.ucsc.edu`, harness-instrumented |
| notebook codegen | idempotent, 0 dirty files after re-run |
| full gate | fast **2,033**, browser **56**, integration **158 passed** |

The quickstart re-execution is the strongest single result here: a notebook committed earlier still
runs clean against today's code, on CPU, using the `chorus` kernel that `chorus setup` now registers.

## Findings, fixed in this PR

**F1 (medium) — three of six library notebooks did not point at the `chorus` kernel.**

| notebook | declared |
|---|---|
| `cherimoya_quickstart.ipynb` | `name="python3"`, `display="Python 3"` |
| `epinformerseq_testing.ipynb` | `name="python3"`, `display="Python 3 (chorus)"` |
| `klf1_validated_enhancer_profiles.ipynb` | `name="python3"`, `display="chorus"` |

`name: "python3"` resolves to whatever the reader's default Jupyter kernel is. On this host that
*accidentally* is the chorus env — `jupyter kernelspec list` shows `python3` pointing at
`/home/nvidia/miniforge3/envs/chorus/bin/python`, so `import chorus` succeeds and the problem is
invisible. For anyone running JupyterLab from a base or system Python it resolves to their
interpreter and the first import fails.

`klf1` was the worst of the three: `display_name: "chorus"` with `name: "python3"`, so JupyterLab
shows the reader the word "chorus" while running a different interpreter. A wrong kernel that
announces itself as the right one is harder to diagnose than a missing one.

Fixed: all six now declare `chorus` / `Python 3 (chorus)`, metadata only — 5 changed lines across 3
files, and all 157 output cells preserved. Guarded by
`test_every_library_notebook_declares_the_chorus_kernel`, which asserts *every* one rather than most.

**F2 (low) — "pre-run" advertised more than it delivered.**

`examples/walkthroughs/README.md` is titled *"pre-run, MCP-driven worked examples"*. True of the
reports: every directory ships its HTML, JSON and TSV already generated. **Not** true of the
`notebook.ipynb` beside them — all 12 carry **zero** outputs, by design, because they are code-generated
and executing them needs the matching oracle env and usually a GPU.

Nothing said so. A reader who opens the notebook expecting the pre-run results the title promises
finds an empty file and reasonably concludes something is broken. Now stated directly under that
title, pointing at `examples/notebooks/` for notebooks that do come with their results.

## Accepted, not changed

- **No staleness guard on library-notebook outputs.** Nothing checks that committed outputs match what
  current code would produce; re-execution needs a GPU and every per-oracle env, so `CLAUDE.md` keeps
  it as a manual pre-release step. The quickstart run above is that check done by hand for one of six.
- **The 12 walkthrough notebooks ship unexecuted** — deliberate, now documented rather than fixed.
- **Reports need network on first open.** Documented, and `CHORUS_IGV_BUNDLE_SEQUENCE=1` closes it
  (0/0 canvases in 45 s → 100/100 in 1.6 s offline). Opt-in, so the committed reports are unchanged.

## Two false positives worth recording

Both cost time and both looked like real defects:

1. **"Two HTML reports are not browser-tested."** `_committed_reports()` uses `git ls-files`, so it
   covers all 20. Two report *filenames* appear in two directories, and pytest disambiguates duplicate
   parametrize ids by appending `0`/`1` — my extraction regex required `.html]` and silently dropped
   them. 40 parametrized cases = 20 reports × 2 tests, all present.
2. **"Reports contact accounts.google.com, drive.google.com, feross.org."** Those strings live *inside*
   the bundled igv.js (Google Drive OAuth support, and a library author's URL). They are never
   fetched: the harness instruments actual requests and sees only UCSC.

The lesson for the next audit is the same in both cases — a grep over a 6 MiB bundled artefact
measures what is *present*, not what is *executed*. Prefer the instrumented render.
