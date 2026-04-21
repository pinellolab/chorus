# Third-party attribution

Chorus wraps six deep-learning oracles and one genome-browser library.
Each ships under its own license, and model weights are **not**
redistributed in this repo — they are fetched from the original authors'
hosts at first-use time.

## Deep-learning oracles

| Oracle | Authors | Paper | Weights / code license |
|---|---|---|---|
| **Enformer** | Avsec et al., DeepMind | [Effective gene expression prediction from sequence by integrating long-range interactions (Nature Methods 2021)](https://www.nature.com/articles/s41592-021-01252-x) | Apache-2.0 (code); weights on TensorFlow Hub |
| **Borzoi** | Linder et al., Calico Labs | [Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation (Nature Genetics 2025)](https://www.nature.com/articles/s41588-024-02053-6) | Apache-2.0 (code); weights on Zenodo |
| **ChromBPNet** | Pampari et al., Kundaje Lab (Stanford) | [ChromBPNet: bias factorized, base-resolution deep learning models of chromatin accessibility (bioRxiv 2024)](https://www.biorxiv.org/content/10.1101/2024.12.25.630221v1) | MIT (code); weights on ENCODE |
| **Sei** | Chen et al., Troyanskaya Lab (Princeton) | [A sequence-based global map of regulatory activity for deciphering human genetics (Nature Genetics 2022)](https://www.nature.com/articles/s41588-022-01102-2) | BSD-3-Clause (code + weights) |
| **LegNet** | Penzar et al., Vaishnav Lab (Broad) | [LegNet: a best-in-class deep learning model for short DNA regulatory regions (Bioinformatics 2023)](https://academic.oup.com/bioinformatics/article/39/8/btad457/7220619) | MIT (code); weights bundled with source |
| **AlphaGenome** | Avsec et al., Google DeepMind | [AlphaGenome: advancing regulatory variant effect prediction with a unified DNA sequence model (Nature 2026)](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/) | Gated on HuggingFace (`google/alphagenome-all-folds`); accept the license to download weights |

Chorus does not modify the upstream model code beyond the adapter
layer in `chorus/oracles/<name>.py`. Each oracle's predict / score
semantics are those of the original publication.

## Bundled third-party JavaScript

- **IGV.js** (Integrative Genomics Viewer, Robinson et al., Broad/UCSD) —
  [igv.org](https://igv.org/), [github.com/igvteam/igv.js](https://github.com/igvteam/igv.js),
  MIT license. Shipped as `chorus/analysis/static/igv.min.js` so HTML
  reports render offline. Source license at
  [github.com/igvteam/igv.js/blob/master/LICENSE](https://github.com/igvteam/igv.js/blob/master/LICENSE).

## Per-track background CDFs

The NPZ CDFs under `~/.chorus/backgrounds/` are derived from the oracle
authors' published predictions on a reference set of genomic loci.
They are computed by Chorus and distributed at
[`huggingface.co/datasets/lucapinello/chorus-backgrounds`](https://huggingface.co/datasets/lucapinello/chorus-backgrounds)
under CC-BY-4.0 — attribute the original oracle publications above
when citing numbers derived from them.

## Chorus itself

MIT-licensed (see [`LICENSE`](../LICENSE)). Cite as:

> Pinello Lab. *Chorus: unified interface for genomic deep-learning oracles.* 2026.
