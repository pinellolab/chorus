# Audit records

Dated reports from every audit of this project. Each is a snapshot of what was measured on a
specific tree — read them as records, not as descriptions of the current code. Several name commit
shas that are no longer the release; resolve the current one with `git rev-list -n1 v0.7.3`.

`AUDIT_CHECKLIST.md` is different: it is the **live** 19-section runbook, and `CLAUDE.md` names it
as the thing to run before any ship-prep or release. Keep it current in the same commit as anything
it describes.

## The raw artefacts are archived outside the repo

These directories used to also carry the raw output each audit produced — headless-Chromium
screenshots, per-probe logs, executed notebooks, prediction `.npz` dumps. That was **347 files and
122 MB**, against **0.9 MB** for the 79 reports that hold the actual reasoning, and it made the
repo's file tree read as an audit dump rather than a library: `audits/` was 426 of 869 tracked
files, nearly 3× the package itself.

Removed from the tree as of v0.7.3. The reports stay, because they are the record; the artefacts
were evidence for conclusions the reports already state.

**Nothing was lost.** They remain in this repository's git history, and a full copy was taken
before removal — a `.tar.gz` plus a browsable tree, checked to contain every one of the 426 files.
If you need one, ask a maintainer rather than assuming it is gone.

> **A caveat worth stating plainly:** removing files from the tree changes what a visitor *browses*
> and what a checkout *contains*. It does **not** shrink `git clone`, because history still carries
> the blobs. Making the clone smaller would require rewriting history, which invalidates every
> existing clone and every published sha — deliberately not done.
