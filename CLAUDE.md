# CLAUDE.md — project-specific guidance for Claude Code sessions

## Audit / release checklist

Before any "ship-ready" or "is this consistent?" pass on this repo, read
[`audits/AUDIT_CHECKLIST.md`](audits/AUDIT_CHECKLIST.md) first. It's an
18-section runbook covering install, docs, notebooks, HTML reports
(incl. IGV selenium render recipe), CDF/normalization, GPU/device,
HuggingFace auth, MCP, error paths, scientific determinism, genomics
edge cases, offline/air-gapped, logging hygiene, dependency supply
chain, license/attribution, and test suite. Every item has an exact
command or grep pattern, plus P0/P1/P2 severity.

Walk the checklist top-to-bottom for a fresh audit. Leave artefacts
(findings report, screenshots, fresh notebook outputs, CDF and device
probe logs) in `audits/YYYY-MM-DD_vNN_<label>/` per the appendix.

## Prior audit reports

`audits/*.md` — chronological snapshots of what was audited, found,
and fixed. Don't re-run an old audit; build on the most recent one and
cite the item number from the checklist.
