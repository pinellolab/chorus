"""Constants for the BATv1 atlas, the ChIP-seq half of the Cherimoya oracle.

BATv1 (the Cherimoya Binding ATlas, v1) is a family of 3,429 per-experiment
transcription-factor and DNA-associated-protein binding models, one per ENCODE
ChIP-seq experiment, covering 1,139 targets in 183 biosamples, each trained
across the same 5 chromosome-held-out folds as CATv1.  It shares CATv1's
geometry exactly (2,114 bp in, 1,000 bp out, trimming 557), so every offset in
``chorus/oracles/cherimoya.py`` and the 501 bp scoring window transfer unchanged.
Two things differ, and both are handled where the model is run rather than here:

* **The models take a control track.**  Each was trained with the experiment's
  two control bigWigs (plus and minus strand) as a second input.  chorus has no
  control signal to give them, so they are run through cherimoya's
  ``ControlWrapper`` with an all-zero control -- the setting the BATv1 model card
  recommends for predictions of the sequence's own contribution.  The resulting
  counts are lower than the observed track; percentiles are unaffected because
  each track is ranked against a null built the same way.
* **The profile head is two-stranded.**  One softmax is taken jointly over both
  strands and all positions, and the count head predicts ``log(total + 1)`` over
  both strands -- the same ``log1p`` convention as CATv1, *not* BPNet CHIP's
  pooled ``logsumexp`` one (see ``chorus/core/count_head.py``).  chorus sums the
  two strands into one track, so a BATv1 track has the same shape as a CATv1
  track and one background row.

Weights live on HuggingFace under a CC-BY-4.0 licence and are fetched lazily
per experiment; nothing is mirrored into chorus.
"""

# HuggingFace repo holding the BATv1 checkpoints (CC-BY-4.0).
BATV1_HF_REPO = "programmable-genomics/BATv1"
BATV1_HF_REPO_TYPE = "model"

# The repository commit the vendored metadata and the background nulls were
# built from.  Checkpoints are fetched at the repo head; the release's
# ``manifest.csv`` carries an md5 per checkpoint for anyone who needs to check.
BATV1_HF_REVISION = "c631ebd86b46d78c2e239ee06f3fce5ff15f4ad9"

# Checkpoint path template within that repo; the same layout as CATv1.
BATV1_CHECKPOINT_TEMPLATE = "models/{encode_id}/cherimoya.fold_{fold}.torch"

# ENCODE ``assay_term_name`` -> the assay string chorus uses.  ``CHIP`` is the
# value ``OraclePredictionTrack.create`` dispatches to CHIPOraclePredictionTrack
# and ``classify_track_layer`` routes through ``classify_chip_layer``, which
# lands every BATv1 target in the ``tf_binding`` layer: no BATv1 target name
# matches a histone-mark pattern (checked at vendoring time; the atlas holds
# transcription factors, chromatin remodelers, polymerase subunits, cofactors
# and similar, no histone marks).
BATV1_ASSAY_TERM_TO_CHORUS = {"ChIP-seq": "CHIP"}
BATV1_ASSAY_TYPES = ["CHIP"]

# What a BATv1 checkpoint declares, asserted at load time alongside the shared
# trimming check: two control tracks (plus and minus strand) and one signal
# group of two strands.
BATV1_N_CONTROL_TRACKS = 2
BATV1_SIGNAL_GROUPS = [2]
