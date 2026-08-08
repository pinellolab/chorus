"""Constants for the Cherimoya / CATv1 oracle.

CATv1 (the Cherimoya Accessibility aTlas, v1) is a family of 1,518
per-experiment chromatin accessibility models — 1,149 DNase-seq and 369
ATAC-seq ENCODE experiments, each trained across 5 chromosome-held-out
folds.  Architecturally it sits in the BPNet / ChromBPNet family and
shares ChromBPNet's exact input/output geometry, so the offset
arithmetic in ``chorus/oracles/chrombpnet.py`` transfers unchanged.

Weights live on HuggingFace under a CC-BY-4.0 licence and are fetched
lazily per experiment; nothing is mirrored into chorus.
"""

# HuggingFace repo holding the CATv1 checkpoints (CC-BY-4.0).
CATV1_HF_REPO = "programmable-genomics/CATv1"
CATV1_HF_REPO_TYPE = "model"

# Checkpoint path template within that repo.
CATV1_CHECKPOINT_TEMPLATE = "models/{encode_id}/cherimoya.fold_{fold}.torch"

# Model geometry.  Identical to ChromBPNet, which is what makes the
# ported offset arithmetic and the shared 501 bp scoring window valid.
CATV1_INPUT_LENGTH = 2114
CATV1_OUTPUT_LENGTH = 1000
CATV1_BIN_SIZE = 1

# ``(2114 - 1000) // 2`` — bases trimmed from each end of the input to
# get the output window.  Cherimoya exposes the same value as
# ``model.trimming``; asserted equal at load time.
CATV1_TRIMMING = (CATV1_INPUT_LENGTH - CATV1_OUTPUT_LENGTH) // 2

# Number of cross-validation folds per experiment.  Fold 0 is the
# default: Chorus's ChromBPNet oracle also defaults to fold 0 and its
# background CDFs were built at fold 0, so fold-0-to-fold-0 is the
# directly comparable configuration.  CATv1 uses the same chromosome
# partition as the ENCODE ChromBPNet annotations, so the correspondence
# is exact rather than approximate.
CATV1_N_FOLDS = 5

# Sentinel for `fold=`: average the expected-counts predictions of all five
# folds. CATv1's model card offers both usages -- "use a single fold (e.g.
# fold_0), or average the predictions of all five folds for a more robust
# estimate" -- and the ensemble is what chorus ships, because a single fold is a
# sample rather than the model. Measured at rs12740374, ENCSR149XIL: the five
# folds give linear ratios 3.469 / 2.393 / 2.716 / 2.765 / 2.768 and absolute
# reference counts spanning 2.49x for the identical sequence, so which fold you
# pick moves the answer more than most things chorus guards against.
#
# NOTE the mean is over the expected-counts PROFILES, not over the two raw heads
# and not over per-fold log2FCs -- see scoring.heads_equivalent_to_profile. The
# three give different answers (1.4588 / -- / 1.4849 log2FC at rs12740374) and
# only the first is "average the predictions".
CATV1_ENSEMBLE = "ensemble"
# The shipped default. Must match what the background CDFs were built with:
# a query scored one way against a null built the other way is not a
# percentile of anything. build_backgrounds_cherimoya.py --fold defaults here
# too, and tests/test_cherimoya_ensemble.py pins the two together.
CATV1_DEFAULT_FOLD = CATV1_ENSEMBLE

# CATv1 is human-only.  Unlike ChromBPNet (which also ships mouse
# developmental models), every CATv1 experiment is GRCh38.
CATV1_ASSEMBLY = "GRCh38"

# ENCODE ``assay_term_name`` -> the assay string chorus uses.  The
# right-hand values are load-bearing: ``OraclePredictionTrack.create``
# dispatches on them to pick DNaseOraclePredictionTrack /
# ATACOraclePredictionTrack, and ``classify_track_layer`` maps them to
# the ``chromatin_accessibility`` layer.  Changing them silently drops
# tracks into the "other" layer with no background normalization.
ASSAY_TERM_TO_CHORUS = {"DNase-seq": "DNASE", "ATAC-seq": "ATAC"}
CATV1_ASSAY_TYPES = ["ATAC", "DNASE"]

# Central scoring window, matching LAYER_CONFIGS['chromatin_accessibility']
# in chorus/analysis/scorers.py and WINDOW_BP in the ChromBPNet
# background builder.
CATV1_SCORING_WINDOW_BP = 501


def catv1_track_id(assay: str, encode_id: str) -> str:
    """Build the canonical track id for a CATv1 experiment.

    The ENCODE experiment accession is already unique on its own — 1,518
    distinct accessions for 1,518 experiments, none spanning two assays
    — so the ``ASSAY:`` prefix is redundant for uniqueness.  It is kept
    because it makes the assay legible at a glance (these ids end up as
    plot axis labels and in LLM-facing tool output) and matches the
    ``ASSAY:``-prefixed shape the other oracles use.

    Deliberately *not* including the biosample: ``(assay, biosample)`` is
    ambiguous for 1,188 of the 1,518 experiments, and a scheme that
    appended the accession only when needed would flip a biosample's id
    format the moment a new experiment was added for it — orphaning the
    committed background CDF row.  The biosample is available on the
    track object (``cell_type``) and via ``CATv1Metadata.describe``.

    Args:
        assay: ``'ATAC'`` or ``'DNASE'``.
        encode_id: ENCODE experiment accession, e.g. ``'ENCSR000EOT'``.

    Returns:
        e.g. ``'DNASE:ENCSR000EOT'``.
    """
    return f"{assay}:{encode_id}"
