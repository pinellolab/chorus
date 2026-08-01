# ENCODE ChromBPNet model registry.
#
# Sourced from the ENCODE Portal search:
#   https://www.encodeproject.org/search/?type=Annotation&annotation_type=ChromBPNet-model
# Last sync: 2026-04-25.
#
# Schema:
#   - Top-level key: assay (ATAC / DNASE / CHIP).
#   - Sub-key: ``cell_type`` string the user passes to
#     ``load_pretrained_model(assay=..., cell_type=...)``.
#   - Value: ENCFF accession of the model tar on the ENCODE Portal.
#
# HUMAN (hg38) ONLY — see 2026-08-01.
#
# ENCODE also publishes a mouse developmental atlas of ChromBPNet models
# (embryonic forebrain/midbrain/hindbrain/limb/liver/heart/neural tube/
# facial prominence, stages E11.5-E14.5). Those 33 entries used to live
# here and have been REMOVED, because every code path around them
# assumes hg38:
#
#   - ``scripts/build_backgrounds_chrombpnet.py`` opens ``genomes/hg38.fa``
#     and draws its DHS-anchored positions from the hg38 DHS vocabulary,
#     so the per-track CDFs shipped for the mouse rows were built by
#     pushing *human* sequence through *mouse* models. Those 33 rows are
#     dropped from ``chrombpnet_pertrack.npz`` in the same change.
#   - Nothing in the registry recorded an organism, so there was no field
#     to filter or assert on. That is what let the mismatch ship.
#
# Re-adding mouse needs an mm10/mm39 FASTA in the genome manager *and* an
# mm10 region set for background construction (the cCRE Registry has an
# mm10 build; the DHS vocabulary does not), so it is tracked as its own
# piece of work rather than a registry edit.
#
# CAUTION when filtering this registry by biosample name: the mouse
# tissue names collided with *human* ENCODE CHIP biosamples — "liver",
# "heart", "brain" and "forebrain" all appear as human CHIP contexts in
# ``chrombpnet_JASPAR_metadata.tsv``. Any species filter must key on
# ``(assay, cell_type)`` or on the ENCFF accession, never on the bare
# tissue name.
#
# Note:
#   - CHIP / BPNet models live in ``chrombpnet_JASPAR_metadata.tsv``
#     (1259 TF×cell_type entries, JASPAR_DeepLearning 2026 release), all
#     human. They're loaded via the BPNetMetadata path, not this dict.

CHROMBPNET_MODELS_DICT: dict[str, dict[str, str]] = {
    "ATAC": {
        # ── Human cell lines (ENCODE 4) ──
        "K562": "ENCFF984RAF",                       # ENCSR467RSV
        "HepG2": "ENCFF137WCM",                      # ENCSR380YGX
        "GM12878": "ENCFF142IOR",                    # ENCSR389HIH
        "IMR-90": "ENCFF113GSV",                     # ENCSR978WIX
    },
    "DNASE": {
        # ── Human cell lines ──
        "HepG2":   "ENCFF615AKY",                    # ENCSR006CUK
        "IMR-90":  "ENCFF515HBV",                    # ENCSR137OLC
        "GM12878": "ENCFF673TIN",                    # ENCSR003WJE
        "K562":    "ENCFF574YLK",                    # ENCSR296UHQ
        "H1":      "ENCFF138PJQ",                    # ENCSR085MTT (NEW in v28)
    },
    "CHIP": {},
}


def iter_unique_models():
    """Iterate ``(assay, cell_type, encff_id)`` once per distinct ENCFF.

    The registry above intentionally has aliases (e.g. ``"limb"``,
    ``"limb_E12.5"``) that point to the same ENCFF tar. Callers that
    iterate over every model — `discover_variant_effects`,
    `scripts/build_backgrounds_chrombpnet.py` — should use this helper
    to avoid loading the same weights N times. Returns the canonical
    bare-biosample alias when one exists, otherwise the stage-suffixed
    key.
    """
    seen: dict[str, tuple[str, str]] = {}
    for assay in ("ATAC", "DNASE"):
        for cell_type, encff in CHROMBPNET_MODELS_DICT.get(assay, {}).items():
            key = f"{assay}:{encff}"
            existing = seen.get(key)
            # Prefer the shorter/bare name when an alias collision exists,
            # so e.g. "limb" wins over "limb_E12.5" for the canonical row.
            if existing is None or len(cell_type) < len(existing[1]):
                seen[key] = (assay, cell_type)
    for key, (assay, cell_type) in seen.items():
        encff = key.split(":", 1)[1]
        yield assay, cell_type, encff


def iter_unique_bpnet_models():
    """Iterate ``(cell_type, tf, model_url, identifier)`` for every
    BPNet/CHIP model in the JASPAR_DeepLearning 2026 release.

    Reads from ``chrombpnet_JASPAR_metadata.tsv``. Each entry is unique
    per ``(TF, cell_line)``; the TSV may have multiple replicates per
    pair, in which case we yield only the first (matches
    ``BPNetMetadata.get_weights_by_cell_and_tf`` behaviour).

    Used by:
    - ``scripts/build_backgrounds_chrombpnet.py --assay CHIP`` to score
      per-track CDFs over all 1259 TF binding models.
    - Future MCP / discovery code that wants to enumerate the BPNet
      catalogue without the heavyweight ``BPNetMetadata`` class.
    """
    import os
    import pandas as pd

    metadata_path = os.path.join(
        os.path.dirname(__file__), "chrombpnet_JASPAR_metadata.tsv",
    )
    df = pd.read_csv(metadata_path, sep="\t")
    # Drop dups so we only yield one model per (TF, CELL_LINE).
    df = df.drop_duplicates(subset=["TF_NAME", "CELL_LINE"], keep="first")
    for _, row in df.iterrows():
        identifier = f"{row['BASE_ID']}.{row['VERSION']}"
        yield row["CELL_LINE"], row["TF_NAME"], row["MODEL_URL"], identifier
