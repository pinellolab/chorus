"""Constants for the EPInformer-seq per-cell oracle.

One architecture: ``PerCellProfileNetWide`` -- a 2114-bp input is run through
the dilated body, then the central 1024 bp is cropped for the heads
(ChromBPNet-style "valid" geometry, so every output base has a full
real-sequence receptive field). Trained on 5' DNase cut-sites, so the model is
DNase-only.
"""

# Profile output / aggregation length (the central crop the heads see).
EPINFORMERSEQ_WINDOW: int = 1024

# Model input length: 2114 bp in, central 1024 bp cropped for the heads.
EPINFORMERSEQ_WIDE_WINDOW: int = 2114
EPINFORMERSEQ_INPUT_WINDOW: int = EPINFORMERSEQ_WIDE_WINDOW

# Default step for sliding-window inference across larger intervals.
EPINFORMERSEQ_DEFAULT_STEP: int = 64

# 11 Roadmap cells, each with its own PerCellProfileNetWide main + frozen BiasNet.
# Models trained on 5' DNase cut-sites from ENCODE BAMs (DNase-only).
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

# DNase-only model (trained on 5' DNase cut-sites). The scalar is the per-bp
# peak max of the DNase channel over the central 256 bp of the 1024-bp output.
EPINFORMERSEQ_DEFAULT_ASSAY: str = "Enhancer_DNase"
EPINFORMERSEQ_AVAILABLE_ASSAYS = [
    "Enhancer_DNase",
]
