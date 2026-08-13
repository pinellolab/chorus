# Third-party attribution

Chorus wraps eight deep-learning oracles and one genome-browser library.
Each ships under its own license, and model weights are **not**
redistributed in this repo — they are fetched from the original authors'
hosts at first-use time.

## Deep-learning oracles

| Oracle | Authors | Paper | Weights / code license |
|---|---|---|---|
| **Enformer** | Avsec et al., DeepMind | [Effective gene expression prediction from sequence by integrating long-range interactions (Nature Methods 2021)](https://www.nature.com/articles/s41592-021-01252-x) | Apache-2.0 (code); weights on TensorFlow Hub |
| **Borzoi** | Linder et al., Calico Labs | [Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation (Nature Genetics 2025)](https://www.nature.com/articles/s41588-024-02053-6) | Apache-2.0 (code); weights on Zenodo |
| **ChromBPNet** | Pampari et al., Kundaje Lab (Stanford) | [ChromBPNet: bias factorized, base-resolution deep learning models of chromatin accessibility (bioRxiv 2024)](https://www.biorxiv.org/content/10.1101/2024.12.25.630221v1) | MIT (code); weights on ENCODE |
| **Cherimoya / CATv1** | Schreiber | Preprint forthcoming — cite the repository: [github.com/jmschrei/cherimoya](https://github.com/jmschrei/cherimoya) | MIT (code); CATv1 weights **CC-BY-4.0**, fetched lazily per experiment from [`programmable-genomics/CATv1`](https://huggingface.co/programmable-genomics/CATv1) |
| **Sei** | Chen et al., Troyanskaya Lab (Princeton) | [A sequence-based global map of regulatory activity for deciphering human genetics (Nature Genetics 2022)](https://www.nature.com/articles/s41588-022-01102-2) | BSD-3-Clause (code + weights) |
| **LegNet** | Penzar et al., Vaishnav Lab (Broad) | [LegNet: a best-in-class deep learning model for short DNA regulatory regions (Bioinformatics 2023)](https://academic.oup.com/bioinformatics/article/39/8/btad457/7220619) | MIT (code); weights bundled with source |
| **AlphaGenome** | Avsec et al., Google DeepMind | [AlphaGenome: advancing regulatory variant effect prediction with a unified DNA sequence model (Nature 2026)](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/) | Gated on HuggingFace (`google/alphagenome-all-folds`); accept the license to download weights |

Chorus does not modify the upstream model code beyond the adapter
layer in `chorus/oracles/<name>.py`. Each oracle's predict / score
semantics are those of the original publication.

**EPInformer-seq is deliberately absent from that table: it is not third-party.** Its weights
are trained for chorus and served from this project's own HuggingFace repo
([`lucapinello/chorus-epinformerseq-v2`](https://huggingface.co/lucapinello/chorus-epinformerseq-v2)),
there is no vendored upstream code under `chorus/oracles/epinformerseq_source/`, and the
architecture is described in `chorus/oracles/epinformerseq.py`. What it *does* borrow is the
ChromBPNet-style frozen bias net that subtracts Tn5/MNase sequence preference in logit space —
credited to the Kundaje lab in the row above. Recorded here because an audit reasonably read
"7 of 8 oracles attributed" as a missing attribution rather than as a first-party model
(2026-08-12 audit, F5). If the *name* is meant to credit upstream EPInformer work, add that
citation here — this note deliberately does not invent one.

## Bundled third-party JavaScript

- **IGV.js** (Integrative Genomics Viewer, Robinson et al., Broad/UCSD) —
  [igv.org](https://igv.org/), [github.com/igvteam/igv.js](https://github.com/igvteam/igv.js),
  MIT license. Shipped as `chorus/analysis/static/igv.min.js` (1.3 MB, inlined) so a
  report needs no CDN for the library itself. Source license at
  [github.com/igvteam/igv.js/blob/master/LICENSE](https://github.com/igvteam/igv.js/blob/master/LICENSE).

  A report no longer resolves its genome through igv.org's hosted registry
  ([#139](https://github.com/pinellolab/chorus/issues/139)): chromosome lengths, the
  ideogram and the gene track are bundled, which took one report from 14 requests across
  two hosts to **9 across one**, and — on one report, the SORT1 Cherimoya panel — from 9.6 s
  to **2.2 s** to paint (across the whole 19-report corpus the range moved 8.6–10.8 s to
  2.2–4.4 s, a 3.9x mean). **One resource is
  still remote — the reference sequence** (`hg38.2bit` from UCSC), because every igv.js
  version requires a sequence source and hg38 is 3 GB. Point `CHORUS_IGV_SEQUENCE_URL` at
  a self-hosted copy and a report needs no internet at all; serving it same-origin with the
  report measured 0 external requests and 0.8 s.

- **Bundled *inside* `igv.min.js`** — the igv.js UMD bundle vendors third-party libraries of
  its own, which chorus therefore ships too. Their licence banners survive inside the file
  (`grep -o '@license[^*]*' chorus/analysis/static/igv.min.js`):

  | library | notice in the bundle | licence |
  |---|---|---|
  | [DOMPurify](https://github.com/cure53/DOMPurify) | `@license DOMPurify 3.2.1 \| (c) Cure53 and other contributors` | Apache-2.0 OR MPL-2.0 |
  | [pako](https://github.com/nodeca/pako) | `@license (MIT AND Zlib)` | MIT AND Zlib |

  Noted because the audit asked whether the bundle "carries its upstream license header": the
  UMD bundle opens with code rather than a banner, so the licences are present but not at the
  top of the file, and these two were credited nowhere outside it (2026-08-12 audit, F6).

- **UCSC hg38 cytoband table** (`cytoBandIdeo`, primary chromosomes only) — shipped as
  `chorus/analysis/static/cytoBandIdeo_hg38.txt.gz`, 6.1 kB, from
  [hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cytoBandIdeo.txt.gz](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cytoBandIdeo.txt.gz).
  UCSC genome-annotation data is free to use and redistribute
  ([genome.ucsc.edu/license](https://genome.ucsc.edu/license/)). igv.js draws the ideogram
  from it, and the per-chromosome maximum band end supplies the chromosome lengths, so this
  one file replaces both the `cytoBandIdeo` and the `hg38.chrom.sizes` fetch.

- **GENCODE v48 basic annotation** supplies the inline gene track, scoped to the drawn
  window (~59 genes and 5.5 kB for a 1 Mb locus). It replaces the UCSC `ncbiRefSeq` track
  the registry used to attach, and has the advantage of being the same annotation chorus
  uses for every gene lookup, so the panel agrees with the numbers printed beside it.
  Already credited under the annotation section below; downloaded, not vendored.

## Per-track background CDFs

The NPZ CDFs under `<data-dir>/backgrounds/` are derived from the oracle
authors' published predictions on a reference set of genomic loci.
They are computed by Chorus and distributed at
[`huggingface.co/datasets/lucapinello/chorus-backgrounds`](https://huggingface.co/datasets/lucapinello/chorus-backgrounds)
under CC-BY-4.0 — attribute the original oracle publications above
when citing numbers derived from them.

## Chorus itself

MIT-licensed (see [`LICENSE`](../LICENSE)). Cite as:

> Pinello Lab. *Chorus: unified interface for genomic deep-learning oracles.* 2026.
