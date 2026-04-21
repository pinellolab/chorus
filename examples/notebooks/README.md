# Chorus Notebooks — Python library walkthroughs

Three end-to-end Jupyter notebooks that exercise the Chorus Python API
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
| **[comprehensive_oracle_showcase.ipynb](comprehensive_oracle_showcase.ipynb)** | Power users · need every feature in one place | All six oracles, all prediction modes (wild-type, variant, region swap, sequence insertion, discovery), the full visualization + normalization stack. | 60 min |

## Prerequisites

Before opening any notebook, from the repo root:

```bash
# 1. Activate the chorus base env
mamba activate chorus

# 2. Install at least one oracle (Enformer is the lightest, runs on CPU)
chorus setup --oracle enformer

# 3. Download the reference genome
chorus genome download hg38

# 4. Register the chorus env as a Jupyter kernel so the notebooks pick
#    up chorus on first run (they ship with kernel name `python3`).
python -m ipykernel install --user --name chorus \
    --display-name "Python 3 (chorus)"
```

Then `jupyter lab` (or `jupyter notebook`) and pick a notebook from this
folder. The first time you run any cell, select the **"Python 3 (chorus)"**
kernel from the Kernel menu.

## Scaling up

- `single_oracle_quickstart` runs fully on CPU with 8 GB RAM (Enformer).
- `advanced_multi_oracle_analysis` needs **all six oracle envs**
  installed (see the matrix in
  [`../../README.md#setting-up-oracle-environments`](../../README.md#setting-up-oracle-environments)).
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
