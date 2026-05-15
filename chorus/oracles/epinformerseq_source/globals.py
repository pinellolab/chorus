"""Constants for the EPInformer-seq Combined oracle."""

# Native input length of the CellCondProfileNet model.
EPINFORMERSEQ_WINDOW: int = 1024

# Default step for sliding-window inference across larger intervals.
EPINFORMERSEQ_DEFAULT_STEP: int = 64

# Same 11 Roadmap cells as the legacy EPInformer-seq oracle, jointly trained.
# Each cell has its own frozen bias model, plus a shared multi-cell main model
# (CellCondProfileNet) that consumes a 32-d cell embedding via FiLM modulation.
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
EPINFORMERSEQ_CELL2IDX = {c: i for i, c in enumerate(EPINFORMERSEQ_AVAILABLE_CELLTYPES)}

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
