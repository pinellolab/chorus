# Chorus Notebooks — Python library walkthroughs

End-to-end Jupyter notebooks that exercise the Chorus Python API
directly (no Claude / MCP required). Each one runs top-to-bottom from a
fresh kernel and produces plots, numeric outputs, and example HTML
reports inline.

> **Looking for pre-run MCP walkthroughs?** See
> [`../walkthroughs/`](../walkthroughs/) — those are concrete worked
> examples with their outputs already committed, driven by Chorus's
> MCP server from Claude in natural language.

## Which one should I open first?

| Notebook | For whom | What you learn | Typical time |
|---|---|---|---|
| **[single_oracle_quickstart.ipynb](single_oracle_quickstart.ipynb)** | First-time users · bench biologists who can read Python | Load one oracle (Enformer), predict at a locus, score a variant's effect, interpret results with effect percentiles. Includes a gene-expression example. | 15 min |
| **[advanced_multi_oracle_analysis.ipynb](advanced_multi_oracle_analysis.ipynb)** | Intermediate · want to compare oracles | Score the same variant with multiple oracles (ChromBPNet, Enformer, Borzoi, Sei, LegNet, AlphaGenome), plot cross-oracle track comparisons with gene annotations, understand where each oracle is strong. | 45 min |
| **[cherimoya_quickstart.ipynb](cherimoya_quickstart.ipynb)** | Anyone who needs a specific cell type or tissue | Cherimoya/CATv1 across 1,518 ENCODE DNase/ATAC experiments: search the atlas, pick the right experiment when a biosample has several, predict, score a variant, and compare accessibility across biosamples with activity percentiles. | 20 min |
| **[comprehensive_oracle_showcase.ipynb](comprehensive_oracle_showcase.ipynb)** | Power users · need every feature in one place | All six (pre-EPI) oracles, all prediction modes (wild-type, variant, region swap, sequence insertion, discovery), the full visualization + normalization stack. | 60 min |

### Topic-focused notebooks

| Notebook | What it shows |
|---|---|
| **[klf1_validated_enhancer_profiles.ipynb](klf1_validated_enhancer_profiles.ipynb)** | Five-oracle (EPInformer-seq, ChromBPNet, Borzoi, Enformer, AlphaGenome) profile comparison at the 3 CRISPR-validated KLF1 enhancers — cell-specificity (K562 vs GM12878), per-bp DNase + H3K27ac tracks, and overlap with the validated CRE windows. |
| **[epinformerseq_testing.ipynb](epinformerseq_testing.ipynb)** | Smoke test for the EPInformer-seq oracle: loads each per-cell model, runs a 1024-bp prediction, and walks through the assay-id format + variant-effect API. |

## Prerequisites

Before opening any notebook, from the repo root:

```bash
# 1. Activate the chorus base env
mamba activate chorus

# 2. Install at least one oracle (Enformer is the lightest, runs on CPU)
chorus setup --oracle enformer

# 3. Download the reference genome
chorus genome download hg38

# 4. Nothing — `chorus setup` registers the `chorus` Jupyter kernel for you.
```

All 18 shipped notebooks declare kernel name `chorus`, and until recently nothing created it:
`jupyter nbconvert --execute` raised `NoSuchKernel: No such kernel named chorus` and JupyterLab
silently prompted you to pick one, which looks like a broken notebook rather than a missing step.
`chorus setup` now does it.

If you passed `--no-jupyter-kernel`, installed before this change, or the registration warned and
carried on (it never fails setup), do it by hand from the env chorus is installed in:

```bash
python -m ipykernel install --user --name chorus --display-name "Python 3 (chorus)"
```

Then `jupyter lab` (or `jupyter notebook`) and pick a notebook from this
folder. The first time you run any cell, select the **"Python 3 (chorus)"**
kernel from the Kernel menu.

## Scaling up

- `single_oracle_quickstart` runs fully on CPU with 8 GB RAM (Enformer).
- `cherimoya_quickstart` needs a CUDA GPU for fast execution, but not to
  run at all — it completes on CPU, just slowly, because it predicts across
  many biosamples and that is where CPU falls 45–150× behind (the
  `chorus-cherimoya` env is Linux/CUDA; Apple Silicon is CPU-only).
- `advanced_multi_oracle_analysis` needs **all six oracle envs**
  installed (see the matrix in
  [`../../README.md#setting-up-oracle-environments-one-by-one`](../../README.md#setting-up-oracle-environments-one-by-one)).
  Each oracle loads via subprocess isolation so there's no dependency
  conflict between them. A GPU is recommended but not required; the
  AlphaGenome cells will fall back to CPU if CUDA isn't available.
- `comprehensive_oracle_showcase` has the same requirements as the
  advanced notebook, plus it exercises LegNet and Sei, which are small
  models that run comfortably on CPU.

## If you hit a problem

- **`KeyError: 'attributes'` in a `frame.plot(...)` cell** — you're
  running from an older chorus install. The fix (`make_gene_track`) is
  in commits on or after `f07ec53`. Re-run `pip install -e .` from the
  repo root.
- **Subprocess oracle load timeout (AlphaGenome)** — the first-time
  checkpoint restore can take 2–3 minutes on a cold
  `~/.cache/huggingface`. Give it another run; cached it should take
  ~30 s.
- **Notebook cells show `<Figure ... >` but no image** — check your
  matplotlib backend; `%matplotlib inline` should be in cell 1.

## Want to add one?

See [CONTRIBUTING.md § Contributing an example or walkthrough](../../CONTRIBUTING.md#contributing-an-example-or-walkthrough).
Note the `notebook.ipynb` inside each `examples/walkthroughs/*/` directory is **code-generated** by
`scripts/generate_walkthrough_notebooks.py` — the hand-written tutorials are the ones in this
directory.
