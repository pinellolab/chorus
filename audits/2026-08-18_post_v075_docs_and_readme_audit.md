# Post-0.7.5 audit: notebooks, examples, documentation, nulls — and a README that had grown clutter

Scope as requested: notebooks, examples, documentation, the null model, and whether the README is simple
to follow. Run against `main` at `399d631` / `v0.7.5`.

## Verified clean

**Null artefacts.** All 9 shipped NPZs carry `schema_version 4`, one build campaign
(`2026-08-06 unified rebuild`), `build_config` and `layers_per_row`. Track counts match the documented
figures exactly (5,168 / 7,611 / 1,518 / 1,518 / 753 / 5,313 / 33 / 3 / 21,947) and every CDF is
10,000 points. Signed flags sum to **24,160** — borzoi 1,543 + alphagenome 667 + sei 21,947 + legnet 3 —
which is exactly what `BACKGROUND_NULL_PROTOCOL.md` claims. The protocol's own numbers reproduce from the
artefacts.

**Notebooks.** Six library notebooks: zero error outputs, zero tracebacks, zero `WARNING` lines, all on
kernel `chorus`. The two beyond the documented three (`epinformerseq_testing`,
`klf1_validated_enhancer_profiles`) *are* documented — in `examples/notebooks/README.md`, which I had
missed before checking.

**Examples / reports.** All six linked walkthroughs ship Markdown + JSON + TSV + HTML with IGV in the
HTML, as claimed. Browser rendering is covered by CI on every PR.

**README facts.** Spot-checked against code and artefacts: the coordinate-convention table (`CATCA` /
`ATCA`), Cherimoya's 1,149 DNase + 369 ATAC, the 4 K562 ATAC experiments, `score_ism`'s 25 bp default, all
five sequence lengths, 24 MCP tools, and AlphaGenome's 1 bp vs 128 bp bin split. All exact.

## Six findings

| # | finding | where |
|---|---|---|
| 1 | Sei listed as **"40 sequence classes"** — its pre-0.7.4 scope — in the table the README calls a "full side-by-side comparison", which also covered only **6 of 8** oracles | `examples/walkthroughs/README.md` |
| 2 | The AlphaGenome backend-equivalence claim (**"1–2 % per-track fp32 noise"**) survived here after being corrected in the README: understated, and false for the `SPLICE_SITES` assay before 0.7.5 | `docs/variant_analysis_framework.md` |
| 3 | **`describe_tracks()` appears zero times in `docs/`** — including the "Full Python API reference" | `docs/API_DOCUMENTATION.md` and three others |
| 4 | The null protocol had **no record that `alphagenome_pt` has no null of its own** and ranks against `alphagenome`'s — a protocol-level decision, and one that was silently violated for 738 tracks | `docs/BACKGROUND_NULL_PROTOCOL.md` |
| 5 | No decision-log entry for the 0.7.5 change, though `CLAUDE.md` requires the protocol to be updated in the same commit as any change it describes | same |
| 6 | Library notebooks ship **partially executed** (16 blank code cells across three) while their README says each "produces … outputs inline" | `examples/notebooks/README.md` |

Findings 2 and 5 are mine: I corrected the equivalence claim in the README without grepping for it
elsewhere, and I changed which values get ranked against the alphagenome null without logging it. Finding 3
is the same shape as the README gap fixed in 0.7.5 — the uniform API shipped without reaching the reference
documentation.

All six fixed. §8b of the null protocol now states the alias, the premise it rests on, the test that
enforces it, and the instruction to extend that test before aliasing another oracle.

## The README had grown clutter, and I had added to it

Measured before touching anything: **24 blockquote asides totalling 12,229 characters**, the longest
1,692. Two of the five worst were ones I wrote earlier the same night. The pattern is per-release detail
accumulating in the reading path — my version of the "which revision should you install?" aside had grown
a sentence per release, which does not converge.

| | before | after |
|---|---|---|
| aside text | 12,229 chars | **6,793** (−44%) |
| longest aside | 1,692 chars | **704** |
| free-disk bullet | 940 chars | ~450 |

What changed structurally rather than just shorter:

* **One `Caveats` section** now holds the four things that can change a number (Enformer cross-process
  variation, the pre-0.7.5 `alphagenome_pt` splice history, Cherimoya-vs-ChromBPNet magnitude, hg38-only).
  Inline asides became two-line pointers to it. Caveats are findable in one place instead of ambushing the
  reader mid-flow.
* **Token plumbing** moved out of the quick start into the installation section, where someone hits it.
* **Per-release detail delegated to the CHANGELOG.** The README states the rule ("install a tag"); the
  CHANGELOG carries which release moved which number. That stops the README growing by a line per release.
* The enformer install itemisation moved from a TLDR parenthetical into the disk-usage section.

One real inconsistency fell out of the restructure: the free-disk bullet still said a single-oracle
install is "~13 GiB, **not 85**" after the total had moved to 87.

## Guard notes

Three guards caught my own edits during this pass, which is the system working:

* `test_no_weights_is_not_sold_as_a_getting_started_option` — my compression dropped the "cannot predict"
  phrasing it exists to protect.
* `test_live_docs_do_not_claim_a_stale_track_count` — my phrase "took it from 40 tracks to 21,947" read as
  a current claim of 40 tracks for Sei.
* `test_the_tldr_install_size_agrees_with_the_disk_table` — my own guard, keyed to the exact sentence
  "The install itself is ~N GiB", broke the first time that sentence was reworded. Re-anchored on the
  prerequisite bullet: a guard that fails on rewording rather than on drift is noise.

## False positives worth recording

Each looked like a defect and was not:

* **The internal anchors.** My slug function collapsed whitespace, so 8 anchors looked broken — including
  pre-existing ones. GitHub emits one hyphen per space, so removed punctuation leaves a double hyphen
  (`#cherimoya--catv1`). All 21 resolve.
* **NPZ provenance "missing".** I read a `provenance` key; it lives inside `build_config` as JSON.
* **Two undocumented notebooks.** Documented in `examples/notebooks/README.md`.
* **`describe_tracks` "missing" from `docs/`** was real — but I nearly filed the *walkthroughs* Sei row as
  a stale count when the notebook's "21,907 profiles and 40 sequence classes" is correct as components.

## Gates

Fast suite **2,146 passed / 34 skipped**. Integration **159 passed / 6 skipped / 0 failed** (run alone on
GPU 6). Browser: CI.
