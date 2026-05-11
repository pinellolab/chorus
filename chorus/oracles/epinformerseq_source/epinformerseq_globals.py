"""Constants for the EPInformer-seq oracle."""

# Native input length of the enhancer_predictor_256bp model.
EPINFORMERSEQ_WINDOW: int = 256

# Default step (= window, since this is a single-window scalar model).
EPINFORMERSEQ_DEFAULT_STEP: int = 256

# 11 Roadmap cell lines (10 immortalized + H1 hESC) with H3K27ac-summit
# checkpoints (fold 2 best from 12-fold leave-chrom-out CV — chr3+chr13 held
# out for evaluation; mean R 0.717 across cells, the highest-scoring fold).
# Trained on log2(0.1 + sqrt(DNase × H3K27ac)) anchored at ENCODE H3K27ac
# narrowPeak summits, with DNase and H3K27ac RPM averaged across all
# biological replicate BAMs (legacy-style mean-of-replicates).
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

# Single combined-activity assay (geometric mean of DNase + H3K27ac signal).
# Track values are returned in linear sqrt(DNase × H3K27ac) RPM-space (the
# model emits log2(0.1+·) and the helper un-transforms before returning).
EPINFORMERSEQ_DEFAULT_ASSAY: str = "Enhancer_H3K27ac_DNase"
EPINFORMERSEQ_AVAILABLE_ASSAYS = ["Enhancer_H3K27ac_DNase"]
