# Fine-mapping rs9504151 (CDYL locus, FEV1/FVC lung function)

The second worked example from the Chorus blog post, made reproducible.

## Why this directory exists

Every quantitative claim about this locus previously traced to a single
markdown table in
[`audits/2026-06-16_blogpost_reproduction_report.md`](../../../../audits/2026-06-16_blogpost_reproduction_report.md),
describing a GPU run whose **inputs and outputs were never saved**. Across the
whole git history only 9 rsIDs from this locus appear anywhere — not the
56-variant proxy set the numbers were computed from — and the AlphaGenome track
list behind "rank #1, composite 0.995" was recorded nowhere at all. The
fine-map path had not been exercised since 2026-06-17.

Both inputs are now committed, so the claim is checkable:

| file | what it is |
|---|---|
| `ld_proxies.tsv` | the LDlink LDproxy response, 56 rows, with the exact query in its header |
| `assay_ids.txt` | the 21 AlphaGenome lung-fibroblast tracks (`CL:0002553`) |

This is the same approach
[`SORT1_locus`](../SORT1_locus/) already uses — it inlines its 11 proxies as
`LDVariant` literals, which is precisely why SORT1 regenerates with no LDlink
token and this example could not. 56 literals would be unreadable inline, so
this is a data file plus a loader.

## Reproduce

```bash
mamba run -n chorus-alphagenome python scripts/regenerate_remaining_examples.py \
    --only cdyl --gpu 0
```

**No LDlink token needed** — the proxy set is committed. AlphaGenome is a gated
HuggingFace model, so that part does need `hf auth login`. Runs in ~2 minutes on
one H100.

## Result

rs9504151 ranks **#1 of 56**, composite **0.999**, largest effect **−1.368** on
`DNASE:fibroblast of lung`. The sentinel is also the top candidate, which is the
interesting outcome for a credible set where 27 of the 56 variants sit at
r² = 1.00 and so cannot be separated by LD at all.

| Rank | Variant | r² | Composite | Largest effect |
|---|---|---|---|---|
| 1 | **rs9504151** ★ | 1.00 | 0.999 | −1.368 · DNASE |
| 2 | rs658325 | 1.00 | 0.686 | −1.109 · DNASE |
| 3 | rs386522231 | 0.87 | 0.652 | −1.035 · DNASE |
| 4 | rs62384944 | 0.93 | 0.441 | +0.293 · DNASE |
| 5 | rs9504169 | 0.83 | 0.421 | +0.005 · RNA |

It reproduces the audit's recorded AlphaGenome figures closely — composite
0.995 → **0.999**, alt effect −1.363 → **−1.368** — so those numbers were right;
they simply could not be checked by anyone until now.

## Reading it carefully

- **The ranking is a composite of raw values, not percentiles.**
  `causal.py:756-788` weights `max_effect` 0.35, `n_layers` 0.25,
  `convergence` 0.20 and `ref_activity` 0.20, all from raw scores. So the
  known percentile problems ([#83](https://github.com/pinellolab/chorus/issues/83))
  do not enter the ranking — but neither does any magnitude calibration.
- **rs62384944 is not uniquely at r² ≈ 0.93.** The blog singles it out, but it
  sits in a cluster of **11** variants at essentially that same r², so its rank
  is more arbitrary than the prose implies.
- **Chorus does not compute posterior inclusion probabilities.** The PIP = 0.51
  the blog quotes is from the Sniff preprint, not from this tool
  (`grep -rni "posterior_inclusion\|susie\|polyfun" chorus/` → zero hits).
- **LegNet cannot adjudicate here.** Its panel is K562/HepG2/WTC11 — no lung
  fibroblast — so a LegNet ranking at this locus is cell-type mismatched. The
  repo's own corrected draft says so explicitly, and the published post's
  "LegNet ranked second" claim has no supporting run anywhere.
- The **ChromBPNet IMR-90** figures the audit recorded (composite 0.896, effect
  −0.985) are **not** re-run here and should be treated as stale: four
  corrections landed afterwards (`expm1` counts, CDF regen, percentile
  denominator, BPNet CHIP), and both `max_effect` and `ref_activity` read off
  the corrected counts.

## The committed HTML is the largest in the repo, deliberately

`rs9504151_CDYL_locus_causal_report.html` is **25.7 MiB** — 21 lung-fibroblast
tracks × 2 alleles, ~42× a single-track panel. It used to be the one example with
no browser panel, skipped behind `CHORUS_WRITE_LARGE_HTML=1` because it exceeded a
20 MiB ceiling that had been picked as "headroom over today's largest artefact"
rather than derived from a failure.

That ceiling is 50 MiB now and the question is answered by loading the report
rather than weighing it —
[`tests/test_committed_reports_render_in_a_browser.py`](../../../../tests/test_committed_reports_render_in_a_browser.py)
opens every committed report in headless Chromium and asserts every canvas paints.
Measured for this file: **134 canvases, all painted, 11.3 s**, against 8.8 s for the
smallest 1.3 MiB report. 20× the bytes costs 1.3× the load, because the seconds go
on genome-resource round-trips rather than on the payload
([#139](https://github.com/pinellolab/chorus/issues/139)).

Adding it moved no numbers: regenerating this example produced a JSON differing from
the committed one in **1 of 4,001 leaf values**, and that one is the timestamp.
