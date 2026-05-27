"""Constants for the EPInformer-seq per-cell oracle."""

# Native input length of the PerCellProfileNet model.
EPINFORMERSEQ_WINDOW: int = 1024

# Default step for sliding-window inference across larger intervals.
EPINFORMERSEQ_DEFAULT_STEP: int = 64

# 11 Roadmap cells, each with its own PerCellProfileNet main + frozen BiasNet.
# Models trained on per-rep ENCODE BAMs (recommended single rep per assay).
EPINFORMERSEQ_AVAILABLE_CELLTYPES = [
    "K562",
    "GM12878",
    "HepG2",
    "A549",
    "HeLa",
    "HMEC",
    "HSMM",
    "HUVEC",
    "NHEK",
    "NHLF",
    "H1",
]

# Two predicted channels (per-bp profile) + one merged-enhancer summary.
# - "Enhancer_DNase": per-bp DNase cut-site signal at K562/GM12878/...
# - "Enhancer_H3K27ac": per-bp H3K27ac cut-site signal
# - "Enhancer_H3K27ac_DNase": geometric mean (matches the legacy EPInformer-seq
#   single-assay output, useful for backward compatibility in chorus pipelines)
EPINFORMERSEQ_DEFAULT_ASSAY: str = "Enhancer_H3K27ac_DNase"
EPINFORMERSEQ_AVAILABLE_ASSAYS = [
    "Enhancer_H3K27ac_DNase",
    "Enhancer_DNase",
    "Enhancer_H3K27ac",
]
