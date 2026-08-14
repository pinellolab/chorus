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

> **This paragraph used to end "deliberately not done". It was then done.** Removing files from the
> tree changes what a visitor *browses* and what a checkout *contains*, but not what `git clone`
> costs, because history still carried the blobs. So the history **was** rewritten for v0.7.3:
> `git-filter-repo` purged 338 paths / 142.6 MB, the pack went 266.5 MiB → 218.5 MiB, and the
> resulting tree hash was byte-identical to the one before it (`b3069df3…`) — the rewrite removed
> history, not content. Every sha changed; see the "⚠ Git history was rewritten" section at the top
> of [`CHANGELOG.md`](../CHANGELOG.md) for what a clone-holder has to do.

## The merged branches were deleted too (2026-08-14)

The rewrite alone did not shrink what GitHub reports, because 27 stale remote branches still pointed
at pre-rewrite commits and kept every one of those objects reachable. 18 of them had a **merged PR**
(#70, #78, #80, #87–#89, #157, #158, #164, #169–#177, `release/v0.7.2`), so their work is in `main` by
construction and the refs were redundant; those were deleted. The 8 with no merged PR were
deliberately left, on the principle that deleting a branch someone may still want is worse than
carrying it.

Two notes for anyone auditing this later. GitHub recomputes repository size on its **own GC
schedule**, so the reported figure does not drop at the moment of deletion. And because this project
squash-merges, `git branch --merged` can never confirm a branch landed — ancestry does not show it.
The authoritative check is the PR record:

```bash
gh pr list --state all --limit 400 --json number,headRefName,state,mergedAt
```
