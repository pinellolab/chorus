# Can a committed report render without network? — feasibility, 2026-08-15

**Question.** All 19 committed HTML reports fetch their reference sequence from
`hgdownload.soe.ucsc.edu`. Opened offline, the IGV panel paints nothing. v0.7.3 documented that as a
limitation and made CI tolerant of the host being down ([#206](https://github.com/pinellolab/chorus/pull/206),
[#210](https://github.com/pinellolab/chorus/pull/210)). This asks whether the user-facing half can
actually be fixed, before any code is written.

**Answer: yes, and more cheaply than expected.** The blocker is one JSON key.

## What is already inlined, and what is not

Measured on `examples/walkthroughs/variant_analysis/SORT1_enformer/rs12740374_SORT1_enformer_report.html`
(6.4 MiB). Its `"reference"` object is 41,732 characters and contains:

| key | state |
|---|---|
| `cytobandURL` | **inlined** as `data:text/plain;base64,…` |
| `chromSizesURL` | **inlined** as `data:text/plain;base64,…` |
| `twoBitURL` | **remote** — `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit` |

So one key is the entire remaining dependency. Everything else igv.js needs already travels with the
file.

## Route 1 — just omit the sequence: **worse than the status quo**

Three arms, external requests blocked in all of them, rendered through `tests/browser_harness.py`:

| arm | result |
|---|---|
| A — as committed | `canvases 0/0 painted (NOT converged)`, 45 s |
| B — `twoBitURL` deleted | `canvases 0/0 painted (NOT converged)`, 45 s, **no external requests** |
| C — `twoBitURL` and any explicit sequence track deleted | `canvases 0/0 painted (NOT converged)` |

`0/0` means igv.js allocated **no canvases at all** — it never finished initialising, rather than
initialising and drawing nothing.

The confound was checked before drawing a conclusion: the edited `"reference"` object still parses as
JSON (keys `['chromSizesURL', 'cytobandURL', 'id', 'name']`), so arm B failed for a real reason.

That reason is visible in the bundled igv.js:

```js
fastaURL||t.twobitURL||(i=t.id), i){let t=Qu.KNOW…
```

With neither `fastaURL` nor `twoBitURL`, igv.js falls back to resolving `id: "hg38"` against its
**hosted genome registry** — the remote-catalogue lookup that [#139](https://github.com/pinellolab/chorus/issues/139)
removed by inlining in the first place. Omitting the sequence does not degrade gracefully; it
reintroduces a worse network dependency.

## Route 2 — inline a sequence: **works**

igv.js accepts a sequence as a `data:` URL. A self-consistent miniature genome (one 2 kb contig,
`indexed: false`) created a browser and 6 canvases.

The obvious objection is coordinates: an unindexed FASTA implies its own contig length, while these
reports display loci tens of megabases in. That objection **does not hold**, which is the key finding:

> With `fastaURL` carrying a **2 kb stub** labelled `chr1`, and `chromSizesURL` declaring chr1's true
> length of 248,956,422, `igv.createBrowser` succeeded at locus **chr1:109,274,000-109,276,000** —
> 109 Mb beyond the end of the provided sequence — producing **8 canvases, 8 measured, 2 painted**,
> with no console errors and no uncaught exceptions.

igv.js takes chromosome lengths from `chromSizesURL`, not from the FASTA. The sequence is required for
*initialisation*, not for laying out or drawing data tracks. The dependency is structural, not
functional — which is why a stub is enough to make everything else paint.

## What this implies for an implementation

Swap the remote `twoBitURL` for an inlined `fastaURL` + `indexed: false`, keeping `chromSizesURL` as
it already is.

**Inline the real sequence for the displayed window, not a stub.** A stub of repeating `ACGT` would
let a reader zoom in and read fabricated bases, which is worse than a blank track — chorus already
requires hg38 to generate a report, so extracting the window costs nothing. Sizes: a 114,688 bp
Enformer window is ~112 KiB raw, ~150 KiB base64, against a 6.4 MiB report — about 2 %.

Open questions an implementation must answer, not settled here:

1. **Panning outside the inlined window.** igv.js will request sequence it does not have. It must
   degrade to a blank sequence track rather than throwing; untested.
2. **Which window to inline.** The report's initial locus is the obvious choice, but discovery reports
   with several loci would need each, or a union.
3. **Re-rendering all 19 committed reports**, with the usual consequence that every committed
   artefact's bytes change and the guard suite must be re-run.
4. **The sequence track's visibility.** At 114 kb igv.js does not draw bases anyway, so for most
   reports the inlined sequence is insurance rather than something a reader sees.

## Recommendation

Worth doing, and small: one key in the generator plus a re-render. It removes the last network
dependency from a shipped artefact, which is the difference between "works on a plane" and "works if
UCSC is up".

Not urgent: the CI half that was actively costing time is already fixed, and the limitation is now
documented honestly. Nothing here changes a number chorus reports.

## How to reproduce

The two experiment scripts are `/tmp/feasibility_no_sequence.py` and the stub-sequence probe recorded
in this session's transcript; both drive `tests/browser_harness.py` with `block_external=True`. They
are deliberately not committed — see [`audits/README.md`](README.md) on keeping artefacts out of the
tree. The measurements above are the part worth keeping.
