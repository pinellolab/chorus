# ENCODE ChromBPNet model registry.
#
# Sourced from the ENCODE Portal search:
#   https://www.encodeproject.org/search/?type=Annotation&annotation_type=ChromBPNet-model
# Last sync: 2026-04-25 — 42 of 43 published annotations have a model
# tar (ENCSR498NUZ retina DNase-seq has no model file yet).
#
# Schema:
#   - Top-level key: assay (ATAC / DNASE / CHIP).
#   - Sub-key: ``cell_type`` string the user passes to
#     ``load_pretrained_model(assay=..., cell_type=...)``.
#   - Value: ENCFF accession of the model tar on the ENCODE Portal.
#
# Disambiguation: where ENCODE published more than one model for the
# same biosample (mouse developmental atlas — multiple embryonic
# stages, sometimes multiple replicates per stage), the registry
# exposes both:
#   1. A bare biosample alias (e.g. ``"limb"``) → kept identical to
#      the v27 chorus default so existing scripts keep working.
#   2. Stage-suffixed entries (``"limb_E11.5"`` / ``"limb_E12.5"`` /
#      ``"limb_E14.5"``) → reach the other ENCODE-published variants.
#
# Known gaps (v28 audit):
#   - Per-track CDF backgrounds (``~/.chorus/backgrounds/chrombpnet_pertrack.npz``)
#     were built against the original 24 (assay, biosample) keys only.
#     Stage-suffixed variants will predict normally but won't have
#     percentile-normalised scoring until the CDFs are rebuilt by
#     ``scripts/build_backgrounds_chrombpnet.py`` on a GPU host.
#   - CHIP / BPNet models live in ``chrombpnet_JASPAR_metadata.tsv``
#     (1259 TF×cell_type entries, JASPAR_DeepLearning 2026 release).
#     They're loaded via the BPNetMetadata path, not this dict.

CHROMBPNET_MODELS_DICT: dict[str, dict[str, str]] = {
    "ATAC": {
        # ── Human cell lines (ENCODE 4) ──
        "K562": "ENCFF984RAF",                       # ENCSR467RSV
        "HepG2": "ENCFF137WCM",                      # ENCSR380YGX
        "GM12878": "ENCFF142IOR",                    # ENCSR389HIH
        "IMR-90": "ENCFF113GSV",                     # ENCSR978WIX

        # ── Mouse developmental atlas — bare-biosample defaults
        # (kept identical to chorus v27 values; the corresponding
        # _E* alias points to the same ENCFF). ──
        "neural_tube": "ENCFF031NTE",                # ENCSR226MTN E11.5
        "heart":       "ENCFF621FME",                # ENCSR970BZN E14.5
        "hindbrain":   "ENCFF620TIL",                # ENCSR638HWZ E11.5
        "liver":       "ENCFF302KPO",                # ENCSR533ULZ E14.5
        "limb":        "ENCFF606BNG",                # ENCSR303KKP E12.5
        "embryonic_facial_prominence": "ENCFF894OSV",  # ENCSR443PIH E11.5
        "forebrain":   "ENCFF501TIK",                # ENCSR465POI E11.5
        "midbrain":    "ENCFF503GTJ",                # ENCSR843KLE E11.5

        # ── Stage-suffixed mouse variants (new in v28) ──
        "limb_E11.5":         "ENCFF236MMP",         # ENCSR495VMZ
        "limb_E12.5":         "ENCFF606BNG",         # ENCSR303KKP (same as "limb")
        "limb_E14.5":         "ENCFF905ILM",         # ENCSR281ETP
        "heart_E11.5":        "ENCFF750NGU",         # ENCSR345GMN
        "heart_E12.5":        "ENCFF054CFO",         # ENCSR226NMV
        "heart_E14.5":        "ENCFF621FME",         # ENCSR970BZN (same as "heart")
        "liver_E11.5":        "ENCFF596GBF",         # ENCSR229OZL
        "liver_E12.5":        "ENCFF129BPA",         # ENCSR566EUK
        "liver_E14.5":        "ENCFF302KPO",         # ENCSR533ULZ (same as "liver")
        "hindbrain_E11.5":    "ENCFF620TIL",         # ENCSR638HWZ (same as "hindbrain")
        "hindbrain_E14.5":    "ENCFF528XQO",         # ENCSR173UYQ
        "embryonic_facial_prominence_E11.5": "ENCFF894OSV",  # ENCSR443PIH (same as bare)
        "embryonic_facial_prominence_E12.5": "ENCFF927BHN",  # ENCSR310TZS
        "embryonic_facial_prominence_E14.5": "ENCFF817FFY",  # ENCSR574HAS
        "forebrain_E11.5":    "ENCFF501TIK",         # ENCSR465POI (same as "forebrain")
        "midbrain_E11.5":     "ENCFF503GTJ",         # ENCSR843KLE (same as "midbrain")
        "neural_tube_E11.5":  "ENCFF031NTE",         # ENCSR226MTN (same as "neural_tube")
        "neural_tube_unknown_stage": "ENCFF941SIQ",  # ENCSR414ZDH (timepoint not specified)
    },
    "DNASE": {
        # ── Human cell lines ──
        "HepG2":   "ENCFF615AKY",                    # ENCSR006CUK
        "IMR-90":  "ENCFF515HBV",                    # ENCSR137OLC
        "GM12878": "ENCFF673TIN",                    # ENCSR003WJE
        "K562":    "ENCFF574YLK",                    # ENCSR296UHQ
        "H1":      "ENCFF138PJQ",                    # ENCSR085MTT (NEW in v28)

        # ── Mouse developmental atlas — bare-biosample defaults ──
        "forelimb_bud": "ENCFF244FWC",               # ENCSR187FBX E11.5 (only stage)
        "hindlimb_bud": "ENCFF676WUE",               # ENCSR293UHA E11.5 (NEW in v28)
        "forebrain":    "ENCFF271AKE",               # ENCSR566UGU E11.5
        "liver":        "ENCFF053FQE",               # ENCSR632SIB E14.5
        "limb":         "ENCFF067XGA",               # ENCSR393JKG E11.5
        "neural_tube":  "ENCFF541EGW",               # ENCSR432KJZ E11.5
        "embryonic_facial_prominence": "ENCFF715EMI",  # ENCSR352WOM E11.5
        "midbrain":     "ENCFF431PYL",               # ENCSR962UZB E14.5
        "hindbrain":    "ENCFF186EHP",               # ENCSR797RNF E11.5

        # ── Stage-suffixed mouse variants (new in v28) ──
        "limb_E11.5":         "ENCFF067XGA",         # ENCSR393JKG (same as "limb")
        "limb_E14.5":         "ENCFF275WOE",         # ENCSR551DHM
        "liver_E11.5":        "ENCFF044WHP",         # ENCSR117PTY
        "liver_E14.5":        "ENCFF053FQE",         # ENCSR632SIB (same as "liver")
        "forebrain_E11.5":    "ENCFF271AKE",         # ENCSR566UGU (same as "forebrain")
        "forebrain_E14.5":    "ENCFF251GBK",         # ENCSR491OUP
        "midbrain_E11.5":     "ENCFF858GDY",         # ENCSR031VGX
        "midbrain_E14.5":     "ENCFF431PYL",         # ENCSR962UZB (same as "midbrain")
        "hindbrain_E11.5":    "ENCFF186EHP",         # ENCSR797RNF (same as "hindbrain")
        "hindbrain_E14.5":    "ENCFF911DCI",         # ENCSR125VQW
        "embryonic_facial_prominence_E11.5": "ENCFF715EMI",  # ENCSR352WOM (same as bare)
        "embryonic_facial_prominence_E14.5": "ENCFF424ENU",  # ENCSR158PDD
        "neural_tube_E11.5":  "ENCFF541EGW",         # ENCSR432KJZ (same as "neural_tube")
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
