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

- **IGV.js 3.8.4** (Integrative Genomics Viewer, Robinson et al., Broad/UCSD) —
  [igv.org](https://igv.org/), [github.com/igvteam/igv.js](https://github.com/igvteam/igv.js),
  MIT license. Shipped as `chorus/analysis/static/igv.min.js` (1.5 MB, inlined) so a
  report needs no CDN for the library itself. The file is upstream's `dist/igv.min.js` for
  that release, unmodified: the same bytes as the `igv@3.8.4` URL in `_IGV_CDN`
  (`chorus/analysis/_igv_report.py`), sha256
  `91bda9aa8268cbd9642713bad4a48e9089cb8cfab9b5b4aaf7e648a3a759a6fc`. It replaced the 3.1.1
  bundle in [#246](https://github.com/pinellolab/chorus/pull/246). The minified file has no
  banner, so its version is only in the code
  (`grep -oE 'version:"[0-9.]+"' chorus/analysis/static/igv.min.js`), and igv.js's own MIT
  notice is not in it at all. That notice, *Copyright (c) 2014-2025 Broad Institute of MIT
  and Harvard, Regents of the University of California* plus the MIT permission text, is
  igv.js's [`LICENSE` at the v3.8.4 tag](https://github.com/igvteam/igv.js/blob/v3.8.4/LICENSE),
  the version that matches the bundled copy.

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
  its own, which chorus therefore ships too. Their banners survive inside the file
  (`grep -oE '(@license|Copyright|Licensed under|license information)[^*]*' chorus/analysis/static/igv.min.js`):

  | library | notice in the bundle | licence |
  |---|---|---|
  | [DOMPurify](https://github.com/cure53/DOMPurify) | `@license DOMPurify 3.4.11 \| (c) Cure53 and other contributors` | Apache-2.0 OR MPL-2.0 |
  | [pako](https://github.com/nodeca/pako) | `pako 2.0.4 … @license (MIT AND Zlib)`, and a second copy, 2.1.0 | MIT AND Zlib |
  | [vanilla-picker](https://github.com/Sphinxxxx/vanilla-picker) | `vanilla-picker v2.12.3 … Copyright 2017-2024 Andreas Borgen …, Adam Brooks …`, and a second copy, v2.12.1 | ISC |
  | [Canvas 2 Svg](https://github.com/gliffy/canvas2svg) | `Canvas 2 Svg v1.0.19 … Copyright (c) 2014 Gliffy Inc.` | MIT |
  | [twobit-js](https://github.com/GMOD/twobit-js), in part | `Some portions of this code adapated from the GMOD two-bit.js project, @Copyright (c) 2017 Robert Buels` | MIT (upstream's; the banner names none) |
  | [cram-js](https://github.com/GMOD/cram-js), as a webpack bundle | `For license information please see cram-bundle.js.LICENSE.txt` | MIT (upstream's). That file is in igv.js's source tree ([v3.8.4](https://github.com/igvteam/igv.js/blob/v3.8.4/js/cram/cram-bundle.js.LICENSE.txt)), not in its npm package or in chorus: xzwasm, walloc, buffer and is-buffer (MIT), xz-embedded (public domain), ieee754 (BSD-3-Clause) |

  Noted because the audit asked whether the bundle "carries its upstream license header": the
  UMD bundle opens with code rather than a banner, so the licences are present but not at the
  top of the file, and these two were credited nowhere outside it (2026-08-12 audit, F6). That
  pass listed only the two `@license` banners; the other four rows, present in the 3.1.1 bundle
  too, were added when #246's 3.8.4 bundle was checked (2026-11-15 audit, P2-30).

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
  Downloaded, not vendored; its source and citation are under
  [Annotation and conservation data](#annotation-and-conservation-data) below.

## Annotation and conservation data

None of this is vendored. Each source is downloaded on first use into `<data-dir>/`, from the
host named below: `annotations/` for the GTF, `downloads/` for the bigwigs. The three
conservation sources feed no chorus score; they are an overlay on `show_conservation=True`
reports, and readable directly through `chorus.analysis.conservation`. Recorded here because
[#246](https://github.com/pinellolab/chorus/pull/246) started downloading GPN-Star, phyloP and
phastCons without crediting them anywhere reachable from the README, and the GENCODE line
above pointed at this section before it existed (2026-11-15 audit, P1-3).

- **GPN-Star** (Ye et al., Song Lab, UC Berkeley) —
  [Predicting genome-wide functional constraints with GPN-Star (Nature 2026)](https://www.nature.com/articles/s41586-026-11005-5).
  Chorus downloads the hg38 100-way vertebrate model's scores,
  `bigwig/gpn-star-hg38-v100-200m/entropy.bw` and `llr_{A,C,G,T}.bw`, from
  [`songlab/gpn-star-scores`](https://huggingface.co/datasets/songlab/gpn-star-scores) at the
  revision it pins, `a7b13bbf0d2338d74a7e5f0f8466e41ac0722f50` (`GPN_STAR_HF_REVISION` in
  `chorus/analysis/conservation.py`). **Apache-2.0**, per the dataset card at that revision.
  The dataset's current card cites the Nature paper above; the card at the pinned revision
  cites its bioRxiv preprint
  ([doi:10.1101/2025.09.21.677619](https://doi.org/10.1101/2025.09.21.677619)).

- **phyloP 100-way** and **phastCons 100-way** (UCSC Genome Browser) — per-base scores
  computed with the PHAST package on UCSC's
  [`multiz100way`](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/) alignment
  of 99 vertebrate genomes to hg38. From `hgdownload.soe.ucsc.edu` chorus downloads
  [`hg38.phyloP100way.bw`](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/) and
  [`hg38.phastCons100way.bw`](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/),
  under UCSC's data-use terms: each directory's `README.txt` says "All the files in this
  directory are freely available for public use", restrictions on the underlying assemblies
  are listed at [genome.ucsc.edu/goldenPath/credits.html](https://genome.ucsc.edu/goldenPath/credits.html),
  and UCSC asks that the data contributor be credited
  ([genome.ucsc.edu/conditions.html](https://genome.ucsc.edu/conditions.html)). Both READMEs
  give these references for the methods (the phyloP entry there omits Rosenbloom; this is the
  published author list):
  - **phyloP**: Pollard KS, Hubisz MJ, Rosenbloom KR, Siepel A.
    [Detection of nonneutral substitution rates on mammalian phylogenies](https://doi.org/10.1101/gr.097857.109).
    *Genome Res* 2010;20(1):110–121.
  - **phastCons**: Siepel A, Bejerano G, Pedersen JS, *et al.*
    [Evolutionarily conserved elements in vertebrate, insect, worm, and yeast genomes](https://doi.org/10.1101/gr.3715005).
    *Genome Res* 2005;15(8):1034–1050.

- **GENCODE v48** (GENCODE consortium; release 48 is Ensembl 114, on GRCh38.p14) — the basic
  gene annotation, `gencode.v48.basic.annotation.gtf.gz`, from
  [`ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/`](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/).
  It is `gencode_v48_basic` in `chorus/utils/annotations.py`: the default for every gene
  lookup, and the source of the IGV reports' gene track. `gencode_v48_comprehensive` and
  `gencode_v47_basic` are registered beside it and fetched only when named. GENCODE data is
  [open access](https://www.gencodegenes.org/pages/data_access.html). Cite GENCODE's current
  reference publication: Mudge JM *et al.*
  [GENCODE 2025: reference gene annotation for human and mouse](https://doi.org/10.1093/nar/gkae1078).
  *Nucleic Acids Res* 2025;53(D1):D966–D975.

## Per-track background CDFs

The NPZ CDFs under `<data-dir>/backgrounds/` are derived from the oracle
authors' published predictions on a reference set of genomic loci.
They are computed by Chorus and distributed at
[`huggingface.co/datasets/lucapinello/chorus-backgrounds`](https://huggingface.co/datasets/lucapinello/chorus-backgrounds)
under CC-BY-4.0 — attribute the original oracle publications above
when citing numbers derived from them.

## Chorus itself

MIT-licensed (see [`LICENSE`](../LICENSE)). For the citation, use the BibTeX entry in
[the README](../README.md#citation) — it is the canonical one. This file previously carried a second,
differently-worded citation with a different title and author, which left a reader to guess which was
authoritative.
