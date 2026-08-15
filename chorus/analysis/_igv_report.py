"""Generate an IGV.js-based interactive genome browser for HTML reports.

Embeds signal tracks as inline feature arrays in a single HTML file, which the reader can
zoom and pan. Gene annotations are **inlined too**, from chorus's own GENCODE v48 scoped to
the drawn window -- not fetched from IGV's built-in genome, which is what #139 replaced.

The one thing still fetched at view time is the reference **sequence** (`hg38.2bit` from
UCSC), because every igv.js version requires a sequence source and hg38 is 3 GB. So a report
is *nearly* self-contained, and this docstring deliberately does not say "self-contained
offline" -- the previous wording did, and it was false for three months. See
`igv_reference_config` below for what is bundled and `CHORUS_IGV_SEQUENCE_URL` for closing
the last gap.

Track data is downsampled to keep the HTML file size manageable while
preserving the shape of peaks and effects.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from chorus.core.globals import CHORUS_LIB_DIR

logger = logging.getLogger(__name__)

# IGV.js is bundled as a package resource at
# ``chorus/analysis/static/igv.min.js`` so every install has an offline-
# usable copy without any network round-trip. The legacy CDN + HF
# fallback paths remain as secondary options in case a downstream
# consumer stripped the static file from the wheel.
#
# Inlining the JS removes the *library* from the report's network dependencies.
# It does NOT make the report self-contained, which this comment used to claim:
# the genome is a separate dependency, and see igv_reference_config below for
# what is now bundled and the one resource that cannot be. The CDN <script>
# fallback is the last resort and only triggers when both the bundled copy and
# both network paths fail.
_IGV_CDN = "https://cdn.jsdelivr.net/npm/igv@3.1.1/dist/igv.min.js"
_IGV_LOCAL = CHORUS_LIB_DIR / "igv.min.js"
_IGV_BUNDLED = Path(__file__).parent / "static" / "igv.min.js"
# HuggingFace mirror — tertiary fallback for unusual installs where the
# bundled resource is missing (e.g. stripped by a packer) and stdlib
# urllib is blocked by a MITM proxy.
_IGV_HF_REPO = "lucapinello/chorus-backgrounds"
_IGV_HF_FILENAME = "igv.min.js"

# ----------------------------------------------------------------------
# The genome reference (#139)
# ----------------------------------------------------------------------
# `genome: "hg38"` is a *registry lookup*, not a genome. igv.js resolves the
# string against its hosted catalogue, which means every shipped report used to
# open six remote resources across two hosts:
#
#     igv.org/genomes/genomes.json                     the catalogue
#     igv.org/genomes/data/hg38/hg38_alias.tab         chromosome aliases
#     hgdownload.soe.ucsc.edu .../hg38.chrom.sizes     chromosome lengths
#     hgdownload.soe.ucsc.edu .../cytoBandIdeo.txt.gz  the ideogram
#     hgdownload.soe.ucsc.edu .../ncbiRefSeq.txt.gz    the gene track
#     hgdownload.soe.ucsc.edu .../hg38.2bit            the sequence (ranged)
#
# Measured: 14 requests per report, and the catalogue fetch is FATAL — with the
# network blocked the panel does not degrade, it never appears, dying on
# `Error accessing resource: https://igv.org/genomes/genomes.json`. So the
# docstring above claiming air-gapped viewability was false in the strongest way.
#
# Four of those six are replaced by inline data, the fifth (the gene track) by
# features read from chorus's own GENCODE annotation, and the catalogue lookup
# disappears with `loadDefaultGenomes: false`. Measured effect on one report:
# 14 requests across 2 hosts -> 9 across 1, and 10.5 s to paint -> 2.3 s.
#
# THE SEQUENCE CANNOT BE BUNDLED, and that is a property of igv.js rather than a
# decision. Every version requires a sequence source: omit it and 3.1.1 dies in
# `Ec.loadAll` on `undefined.startsWith`, while 3.8.5 dies on "url must be either
# a 'File', 'string', 'function', or 'Promise'". A data: URI does not work
# either -- igv decodes data URIs inline and treats them as a NON-indexed FASTA,
# taking chromosome lengths from the body, so a stub declaring real lengths in its
# .fai renders an ideogram and ruler that look perfect while every feature track
# silently draws nothing (measured: 3 of 5 canvases painted against 5 of 5 with a
# real reference). hg38 is 3 GB, so inlining the real thing is not an option.
#
# A site that needs true offline use must therefore serve the 2bit itself and
# point CHORUS_IGV_SEQUENCE_URL at it; that is the only remaining fetch.
_UCSC_HG38_2BIT = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit"

#: UCSC's hg38 ideogram table, primary chromosomes only, vendored at 6.1 kB.
#:
#: Serves double duty: igv.js draws the ideogram from it, and the per-chromosome
#: maximum band end IS the chromosome length (verified against the FASTA index:
#: chr1 248,956,422 both ways), so one asset replaces both the cytoband and the
#: chrom.sizes fetch.
_CYTOBAND_BUNDLED = Path(__file__).parent / "static" / "cytoBandIdeo_hg38.txt.gz"


def _data_uri(text: str) -> str:
    """A ``data:`` URL igv.js can load without touching the network."""
    import base64

    return "data:text/plain;base64," + base64.b64encode(text.encode()).decode("ascii")


def _cytoband_table() -> "tuple[str, str] | None":
    """``(cytoband text, chrom.sizes text)`` from the vendored table, or None.

    Returns None rather than raising if the asset was stripped from the wheel, so a
    report still renders — falling back to the registry lookup, which is worse but
    working. The failure this protects against is a packaging accident, not a bug.
    """
    if not _CYTOBAND_BUNDLED.exists():
        logger.warning(
            "%s is missing, so the report will resolve its genome through igv.org's "
            "catalogue instead of the bundled tables (six remote fetches, fatal with "
            "no network). Reinstall chorus to restore it.", _CYTOBAND_BUNDLED,
        )
        return None
    import gzip

    with gzip.open(_CYTOBAND_BUNDLED, "rt") as fh:
        cyto = fh.read()
    ends: dict[str, int] = {}
    for line in cyto.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            ends[parts[0]] = max(ends.get(parts[0], 0), int(parts[2]))
    sizes = "".join(f"{c}\t{n}\n" for c, n in ends.items())
    return cyto, sizes


#: A one-contig placeholder FASTA. igv.js needs *a* sequence source to build a browser at all --
#: `fastaURL||t.twobitURL||(i=t.id)` in the bundled igv.js, where the `id` branch resolves hg38
#: against igv.org's hosted registry, which is the remote catalogue fetch #139 removed. So with no
#: sequence the panel does not degrade, it never appears: measured `canvases 0/0 painted` with
#: external requests blocked.
#:
#: What it is NOT: the real sequence for the displayed window. An unindexed FASTA positions its bases
#: from offset 0 of the contig, and these reports display loci tens of megabases in -- placing the
#: real window correctly would need that many leading bases. Inlining real bases at the wrong
#: coordinates is worse than showing none, because a reader can zoom in and read them.
#:
#: So it is deliberately `N`. Chromosome lengths still come from `chromSizesURL`, which igv.js
#: prefers over the FASTA -- verified: with a 2 kb placeholder and chr1's true 248,956,422 declared,
#: `createBrowser` succeeded at chr1:109,274,000 (109 Mb past the placeholder) and painted its data
#: tracks. The sequence track has nothing to draw; every other track renders.
_PLACEHOLDER_CONTIG = "chr1"
_PLACEHOLDER_BASES = 1000


def _placeholder_sequence_uri() -> str:
    """A `data:` FASTA holding `N` x _PLACEHOLDER_BASES, enough for igv.js to initialise."""
    fasta = f">{_PLACEHOLDER_CONTIG}\n" + ("N" * _PLACEHOLDER_BASES) + "\n"
    return _data_uri(fasta)


def igv_reference_config(
    sequence_url: str | None = None, *, bundle_sequence: bool = False
) -> dict | None:
    """An explicit inline reference, or None to fall back to the registry lookup.

    Everything igv.js needs about hg38 except the sequence itself: chromosome
    names and lengths, and the ideogram, both from one vendored 6.1 kB table and
    both passed as ``data:`` URLs so they cost no round-trip. See the block comment
    above for what this replaces, what it measured, and why the sequence is the one
    piece that has to stay remote.

    *sequence_url* overrides where the sequence comes from;
    ``CHORUS_IGV_SEQUENCE_URL`` does the same by environment. That is the hook for
    an air-gapped site, and it accepts a **FASTA** as well as a 2bit precisely
    because every chorus install already downloads ``hg38.fa`` — serve the genomes
    directory over HTTP and reports need no internet at all:

        python -m http.server -d "$(chorus config data-dir --show)/genomes" 8000
        export CHORUS_IGV_SEQUENCE_URL=http://localhost:8000/hg38.fa

    A ``file://`` path does **not** work, and that is a browser rule rather than a
    chorus limitation: a page opened from disk may not read sibling files, so the
    sequence has to be served.
    """
    import os

    tables = _cytoband_table()
    if tables is None:
        return None
    cyto, sizes = tables
    reference = {
        "id": "hg38",
        "name": "Human (GRCh38/hg38)",
        "cytobandURL": _data_uri(cyto),
        "chromSizesURL": _data_uri(sizes),
    }

    explicit = sequence_url or os.environ.get("CHORUS_IGV_SEQUENCE_URL")
    if explicit is None and (bundle_sequence or os.environ.get("CHORUS_IGV_BUNDLE_SEQUENCE")):
        # Bundle a placeholder sequence so igv.js can initialise with no network at all. See
        # _placeholder_sequence_uri for why this is a placeholder and not the real bases.
        reference["fastaURL"] = _placeholder_sequence_uri()
        reference["indexed"] = False
        return reference

    url = explicit or _UCSC_HG38_2BIT
    # The only remote resource left, and igv.js requires one; see above. A FASTA
    # needs its index named too, since igv reads lengths and offsets from the .fai
    # rather than parsing the whole file.
    if url.endswith(".2bit"):
        reference["twoBitURL"] = url
    else:
        reference["fastaURL"] = url
        reference["indexURL"] = url + ".fai"
    return reference


def igv_browser_config(
    locus: str,
    tracks: list,
    roi: list | None = None,
    *,
    genome: str = "hg38",
    gene_track: bool = True,
    bundle_sequence: bool | None = None,
) -> dict:
    """Assemble the ``igv.createBrowser`` config, one place for all three reports.

    ``_igv_report``, ``multi_oracle_report`` and ``causal`` each built this dict
    themselves, which is how the pooling defect shipped: three copies of the same
    thing, one of them patched. Anything genome-related belongs here now.

    Falls back to the bare ``genome`` string only when the bundled tables are
    unavailable, so a stripped install degrades to the old behaviour instead of
    rendering nothing.
    """
    import os

    if bundle_sequence is None:
        bundle_sequence = bool(os.environ.get("CHORUS_IGV_BUNDLE_SEQUENCE"))
    reference = (
        igv_reference_config(bundle_sequence=bundle_sequence) if genome == "hg38" else None
    )
    config: dict = {
        "locus": locus,
        "showRuler": True,
        "showNavigation": True,
        "showCenterGuide": True,
        "roi": roi or [],
        "tracks": list(tracks),
    }
    if reference is None:
        config["genome"] = genome
        return config
    config["reference"] = reference
    # Skip igv.org's catalogue fetch entirely. Without this the reference above is
    # still honoured but the catalogue is loaded first regardless -- and that fetch
    # is the one that makes an offline load fatal.
    config["loadDefaultGenomes"] = False
    if gene_track:
        track = _gencode_track_for_window(locus)
        if track is not None:
            config["tracks"].append(track)
    return config


def _gencode_track_for_window(locus: str) -> dict | None:
    """An inline gene track for *locus*, from chorus's own GENCODE annotation.

    The registry-supplied genome came with UCSC's ``ncbiRefSeq`` track, so dropping
    the registry drops the gene track with it -- and a locus panel without genes is
    a real loss, not a cosmetic one. This replaces it with GENCODE v48 (what chorus
    already uses for every gene lookup, so the report agrees with the numbers beside
    it), scoped to the drawn window and inlined as features.

    Returns None when the annotation is not on disk. Report generation must not fail
    for want of a gene track; it warns and ships the signal tracks.
    """
    try:
        chrom, span = locus.split(":")
        start, end = (int(x.replace(",", "")) for x in span.split("-"))
    except (ValueError, AttributeError):
        return None
    try:
        from chorus.utils.annotations import get_genes_in_region

        genes = get_genes_in_region(chrom, start, end)
    except Exception as exc:                       # annotation absent or unreadable
        logger.info("No inline gene track (%s: %s); the panel will show signal only.",
                    type(exc).__name__, exc)
        return None
    if genes is None or len(genes) == 0:
        return None

    features = []
    for _, g in genes.iterrows():
        name = g.get("gene_name") or g.get("gene_id")
        if not name:
            continue
        features.append({
            "chr": chrom,
            "start": int(g["start"]) - 1,          # GTF is 1-based inclusive
            "end": int(g["end"]),
            "name": str(name),
            "strand": str(g.get("strand") or "."),
        })
    if not features:
        return None
    logger.info("Inlined %d GENCODE genes for %s", len(features), locus)
    return {
        "name": "Genes (GENCODE v48)",
        "type": "annotation",
        "format": "annotation",
        "displayMode": "COLLAPSED",
        "color": "rgb(0, 0, 150)",
        "height": 60,
        "order": 10_000,                           # keep it under the signal tracks
        "features": features,
    }


def _ensure_igv_local() -> Path | None:
    """Return a path to ``igv.min.js`` that callers can read + inline.

    Resolution order:
      1. ``chorus/analysis/static/igv.min.js`` — bundled with the
         package. Always present in a standard install; no network
         touched.
      2. ``~/.chorus/lib/igv.min.js`` — legacy cache from earlier chorus
         versions. Kept for continuity.
      3. CDN via stdlib ``urllib`` (``download_with_resume``).
      4. HuggingFace dataset mirror via ``huggingface_hub``.

    Returns the local path when the file is available, ``None`` if all
    four sources failed (callers then fall back to a CDN ``<script>``
    tag in the rendered HTML — reports remain viewable online).
    """
    # 1. Bundled package resource (fast path — no I/O beyond the stat).
    if _IGV_BUNDLED.exists() and _IGV_BUNDLED.stat().st_size > 0:
        return _IGV_BUNDLED

    # 2. Legacy user cache from pre-v13 installs.
    if _IGV_LOCAL.exists() and _IGV_LOCAL.stat().st_size > 0:
        return _IGV_LOCAL

    # Bundled file missing (stripped by a packer?) and no legacy cache.
    # Fall back to the download paths to stay functional.
    _IGV_LOCAL.parent.mkdir(parents=True, exist_ok=True)

    # 3. CDN via stdlib urllib.
    try:
        from chorus.utils.http import download_with_resume
        download_with_resume(_IGV_CDN, _IGV_LOCAL, label="igv.min.js")
        if _IGV_LOCAL.exists() and _IGV_LOCAL.stat().st_size > 0:
            logger.info("Cached igv.min.js from CDN to %s.", _IGV_LOCAL)
            return _IGV_LOCAL
    except Exception as exc:
        logger.debug("CDN fetch of igv.min.js failed (%s); trying HF mirror.", exc)

    # 4. HuggingFace mirror (works through SSL-MITM proxies where stdlib
    # urllib fails — huggingface_hub uses httpx + certifi).
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            _IGV_HF_REPO,
            filename=_IGV_HF_FILENAME,
            repo_type="dataset",
            local_dir=str(_IGV_LOCAL.parent),
        )
        dp = Path(downloaded)
        if dp != _IGV_LOCAL and dp.exists():
            dp.replace(_IGV_LOCAL)
        if _IGV_LOCAL.exists() and _IGV_LOCAL.stat().st_size > 0:
            logger.info("Cached igv.min.js from HuggingFace mirror to %s.", _IGV_LOCAL)
            return _IGV_LOCAL
    except Exception as exc:
        logger.warning(
            "igv.min.js unavailable: bundled resource missing, CDN and HF "
            "mirror both failed (%s); reports will reference %s at view time.",
            exc, _IGV_CDN,
        )
    return None

# Vivid alt colours that contrast strongly with the grey ref
_LAYER_COLORS = {
    "chromatin_accessibility": "0,100,220",    # bright blue (DNASE/ATAC)
    "tf_binding":              "220,30,30",     # bright red (ChIP-TF)
    "histone_marks":           "200,50,160",    # magenta (ChIP-Histone)
    "tss_activity":            "230,120,0",     # bright orange (CAGE)
    "gene_expression":         "120,50,200",    # purple (RNA)
    "promoter_activity":       "230,120,0",     # orange (LentiMPRA)
    "splicing":                "140,86,75",     # brown
    "regulatory_classification": "0,170,190",   # teal (Sei)
}

_REF_COLOR = "180,180,180"  # light grey — strong contrast with vivid alt

# Layer-aware CDF percentile thresholds for IGV visualization.
# floor_pctile = noise threshold (anything below maps to 0).
# peak_pctile  = "1.0" reference point.
#
# Sharp signals (CAGE, TF, DNASE) use floor=p95 / peak=p99 — captures
# all real peaks while suppressing model noise.  Broad histone marks
# use floor=p90 / peak=p99 to preserve their domain shape.
_LAYER_FLOOR_PCTILE = {
    "tss_activity":              0.95,  # CAGE/PRO-CAP — sharp TSS peaks
    "tf_binding":                0.95,  # ChIP-TF — sharp binding peaks
    "chromatin_accessibility":   0.90,  # DNASE/ATAC — lowered from 0.95 so the peak base/shoulder displays alongside the top
    "splicing":                  0.95,  # SPLICE — sharp signals
    "histone_marks":             0.90,  # ChIP-Histone — broad domains
    "gene_expression":           0.90,  # RNA-seq — broad coverage
    "promoter_activity":         0.85,  # LentiMPRA via LegNet — predictions are even sparser than chromatin (most of genome is not a strong promoter); floor at p85 keeps moderately-active promoters visible.  Note: LegNet's summary_cdfs is signed, so repressive values still clip to 0; lowering the floor expands only the positive half.
    "regulatory_classification": 0.95,
}
_PEAK_PCTILE = 0.99
_DEFAULT_FLOOR_PCTILE = 0.95
# Display max: tall enough to show strong peaks (>>p99) without
# saturating most bins.  1.0 = p99 (top 1% threshold), so 3.0 captures
# 3x stronger than the genome-wide top 1%.  Bins above 3.0 clip but
# this is rare for real biology.
#: Top of the display axis, in units where 1.0 is the track's genome-wide p99.
#:
#: 4.0, raised from 3.0. This does NOT rescale anything: the band is still
#: ``(v - floor) / (peak - floor)``, so 1.0 still means p99 and every value below 3.0 is the
#: number it always was. It only moves where clipping starts, which reveals the bins that were
#: being flattened onto the old ceiling. Measured share of each track's clipped mass that a
#: ceiling of 4.0 recovers:
#:
#:   chrombpnet DNASE   43%      alphagenome H3K27ac  33%
#:   cherimoya  DNASE   41%      alphagenome DNASE    24%
#:                               alphagenome CAGE     25%
#:
#: The cost is uniform: a value of 3.0 now occupies 75% of the axis instead of 100%, so every
#: track looks correspondingly shorter. Uniform is the important part -- raising the ceiling for
#: everything preserves cross-oracle comparability exactly, whereas a per-track ceiling would
#: destroy the one property that makes stacked panels readable together.
#:
#: Going much higher does not help. To clip nothing at all a track needs its genome-wide
#: maximum on the axis, and that is ~196 for chrombpnet DNase and ~160 for AlphaGenome DNase in
#: band units; at that ceiling a p99-level peak occupies 0.5% of the height and every panel
#: reads as flat. Above 4.0 the remaining clipped mass is a long thin tail, so the trade turns
#: bad quickly.
_DISPLAY_MAX = 4.0
_HIGH_RES_ORACLES = ["chrombpnet", "legnet"] # for visualization mean vs max pooling

#: igv.js re-reduces the emitted features to pixels with this function, and it is ``max`` for
#: every track.
#:
#: This is the SECOND pooling stage and it wants the opposite default from the first, because
#: the collapse factors differ by two orders of magnitude. The feature stage reduces ~349 native
#: bins into one display bin, where max lifted AlphaGenome DNase's floor to 0.707 and so has to
#: be decided per track (see :func:`choose_aggregation`). igv.js then collapses only 2-3 of those
#: already-pooled features per pixel on a 1 Mb panel, where max has almost no opportunity to
#: promote background but mean still dilutes a sharp peak.
#:
#: Measured on the committed SORT1 multi-oracle panel, collapsing 3:1 as the browser does:
#:
#:                       peak lost by mean      floor under max      saturation under max
#:   legnet LentiMPRA         2.33x             -0.645 -> -0.500      0.000 -> 0.000
#:   alphagenome DNASE        1.56x              0.015 ->  0.028      0.000 -> 0.000
#:   chrombpnet DNASE         1.38x              0.000 ->  0.000      0.003 -> 0.008
#:   alphagenome CAGE         1.31x              0.000 ->  0.000      0.013 -> 0.032
#:   cherimoya  DNASE         1.14x              0.000 ->  0.074      0.010 -> 0.025
#:   alphagenome H3K27ac      1.00x              0.000 ->  0.000      0.005 -> 0.007
#:
#: So max costs one small floor lift (Cherimoya, 0.074 of a 0-3 axis) and roughly doubles
#: saturation while staying well under the 0.075 readability limit, and it loses no peak
#: anywhere. Mean was worse in two ways: it cost up to 2.33x of peak height, and it cost
#: *unequally* -- 1.38x for ChromBPNet against 1.14x for Cherimoya introduces a 1.2x relative
#: distortion between the two tracks a cross-oracle panel exists to compare. It also cancels
#: signed tracks against themselves, which is why LegNet is the worst case.
#:
#: Mirroring the feature stage's per-track choice was considered and is also wrong here: it
#: would send AlphaGenome DNase and the ChIP tracks to mean, costing 1.1-1.6x of peak for floor
#: protection that a 3:1 collapse does not need.
_IGV_WINDOW_FUNCTION = "max"


def browser_window_function(used_log: bool) -> str:
    """Which function igv.js should use to reduce features to pixels for this track.

    ``max`` for everything except a log-scaled track, which gets ``mean``.

    WHY max IS THE DEFAULT. This is the second pooling stage and it wants the opposite answer
    from the first, because the collapse factors differ by two orders of magnitude. The feature
    stage reduces ~349 native bins per display bin, where max lifted AlphaGenome DNase's floor
    to 0.707 and so has to be measured per track (:func:`choose_aggregation`). igv.js then
    collapses only 2-3 of those already-pooled features per pixel, where max has almost no
    opportunity to promote background but mean still dilutes a sharp peak. Measured on the
    SORT1 panel at 3:1, the peak height mean costs: legnet 2.33x, alphagenome DNase 1.56x,
    chrombpnet 1.38x, CAGE 1.31x, cherimoya 1.14x. It costs *unequally*, which is the
    disqualifying part -- 1.38x against 1.14x is a 1.2x relative distortion between the two
    tracks a cross-oracle panel exists to compare -- and it cancels signed tracks against
    themselves, which is why LegNet is worst.

    WHY A LOG-SCALED TRACK IS THE EXCEPTION. The log band compresses the top of the range, so
    many more bins sit just under the ceiling; max then promotes them over it, and clipped flat
    tops read as coverage rather than as peaks. Measured on AlphaGenome CAGE, the only
    log-scaled track: saturation 0.003 under mean against 0.023 under max at 2:1, a 7.7x rise
    that made the track look like RNA-seq gene-body signal instead of TSS spikes. Ink barely
    moved (0.186 -> 0.192), so it is the ceiling and not the density that does the damage. Mean
    costs that track 1.31x of peak height, which is the cheaper of the two harms.

    Raising ``_DISPLAY_MAX`` instead was considered and rejected: eliminating clipping on the
    linear tracks needs a ceiling near 200 (chrombpnet DNase reaches 196.5 in band units), at
    which point a p99-level peak occupies 0.5% of the axis, and a per-track ceiling would break
    the shared 0-3 axis that makes heights comparable across oracles in the first place.
    """
    return "mean" if used_log else _IGV_WINDOW_FUNCTION



# Hard ceiling on the JSON features one IGV track may emit, enforced in
# _calculate_track_bin_size for every oracle. IGV cannot usefully draw more
# than a few thousand features across a browser window, and the shipped HTML
# embeds them inline — so an unbounded track is both useless and unpublishable.
# 4,000 sits just above the widest legitimate need (borzoi 3,277 at 524 kb,
# alphagenome 3,005 at 1 Mb) with headroom, and far enough below GitHub's
# 100 MiB file limit that no realistic report can approach it. See issue #129.
_MAX_FEATURES_PER_TRACK = 4_000

#: FALLBACK pooling preference, used only when a track has no CDF and therefore cannot be
#: measured. When the values ARE display-scaled, `choose_aggregation` decides from the data
#: and overrides these -- see its docstring for why five attempts at a predictor failed.
#:
#: Kept because a raw, un-rescaled track has no display scale to ask "did the floor rise"
#: against, so something has to be assumed. The historical reason these lists exist at all:
#: Cherimoya, a BPNet-family 1 bp model, was absent from the max-pooled list and rendered at
#: 0.547 instead of 3.000 on the same 0-3 axis as ChromBPNet -- a 5.5x display-only dilution
#: in a report whose entire purpose is cross-oracle comparison.
#:
#: These are preferences, not decisions. Do not add measurement notes here; the measured
#: behaviour lives with `choose_aggregation` and in
#: tests/test_display_scale_is_measured_not_declared.py.
_POINT_PROFILE_ORACLES = frozenset({
    "chrombpnet", "cherimoya", "alphagenome", "alphagenome_pt",
})

#: Oracles whose fallback is mean: pre-binned coverage (Enformer 128 bp, Borzoi 32 bp) or a
#: window statistic rather than a profile (Sei, EPInformer-seq). Recorded explicitly so the
#: guard test can tell "decided" from "never considered".
_COVERAGE_ORACLES = frozenset({"enformer", "borzoi", "sei", "epinformerseq"})


#: Displayed-floor above which max-pooling is judged to have cost more than it bought.
#:
#: Derived from measurement. The displayed floor is the MEDIAN of the max-pooled display
#: values, read off the committed panels at the bin size each report actually uses:
#:
#:   keeps max                              flips to mean
#:     chrombpnet DNASE:HepG2      0.0000     alphagenome DNASE:K562 (BCL11A)  0.1990
#:     cherimoya  DNASE:HepG2      0.0000     alphagenome DNASE:HepG2 (SORT1)  0.7072
#:     alphagenome CAGE:K562       0.0000     alphagenome ATAC:HepG2  (SORT1)  0.9056
#:     alphagenome CAGE:HepG2      0.0229
#:     alphagenome CAGE:HepG2      0.0644
#:
#: Measured gap 0.064 to 0.199, and 0.15 sits inside it. Note the margin is NOT symmetric:
#: 2.3x above the highest track that keeps max, but only 1.33x below the lowest that flips.
#: An earlier revision of this table claimed a 5x lower margin and recorded ChromBPNet's floor
#: as 0.013 -- that number is Cherimoya's SATURATION, transcribed into the wrong column; the
#: pooled median for ChromBPNet is 0.0000, the same as Cherimoya's, so the gap has no lower
#: edge in the oracles that matter and the constant is bounded from above only.
#:
#: NOTE the statistic is the MEDIAN, and two alternatives were tried and are wrong. An "ink
#: fraction" flips Cherimoya and ChromBPNet to mean -- Cherimoya inks 41% of its display bins
#: and still reads well, so ink cannot distinguish "many real peaks" from "inflated floor".
#: Saturation is what makes a panel unreadable, but saturation is fixed by the display SCALE
#: (see :func:`escalate_scale_if_saturated` below), not by the pooling operator. Keep the two
#: concerns separate: pooling protects the floor, the scale protects the peaks.
#:
#: This limit is NOT applied to signed tracks -- "does max lift the floor" is meaningless for a
#: track with no floor at zero; see the call sites.
_MAX_POOL_FLOOR_LIMIT = 0.15


def choose_aggregation(display_values, bins_per, *, limit=_MAX_POOL_FLOOR_LIMIT):
    """Decide mean vs max from the data, rather than from the oracle's name.

    Max-pooling can never lose a peak (it keeps the largest value in the bin by
    construction) and mean-pooling can never lift a floor (it cannot exceed the bin's own
    mean). So the poolings fail asymmetrically, and there is exactly one question worth
    asking: *does max-pooling lift THIS track's floor into the signal band?* If not, max is
    free and strictly better; if so, it has traded the floor away for a peak the display
    was going to clip anyway.

    That question was previously answered by a hardcoded list of oracle names, and five
    attempts to replace it with a predictor all failed -- resolution, per-bin ``max/p99``
    from the artefact, the artefact's signal mass above p99, profile density, and
    density x collapse factor. Each got the sign wrong on at least one oracle. The two
    clearest counterexamples: AlphaGenome and Cherimoya both emit DNase at 1 bp and both
    collapse 349 native bins per display bin, yet max lifts AlphaGenome's floor to 0.707
    and Cherimoya's to 0.000 -- and *Cherimoya* is the denser of the two by every density
    measure tried. And AlphaGenome needs opposite answers for its own 1 bp and 128 bp
    tracks, which no per-oracle rule can express at all.

    So this measures instead of predicting. It costs one extra reduce over an array already
    in memory, and it decides per track and per window, so a new oracle is correct without
    anyone remembering to add it to a list.

    Requires *display-scaled* values (1.0 = genome-wide p99). On a raw, un-rescaled track
    there is no scale to compare a floor against, so callers keep their static preference.
    """
    import numpy as np

    if bins_per <= 1:
        return "max"          # nothing is being collapsed; max == mean == identity
    v = np.asarray(display_values, dtype=float)
    n = (len(v) // bins_per) * bins_per
    if n == 0:
        return "max"
    pooled = v[:n].reshape(-1, bins_per).max(1)
    if pooled.size < 100:
        # Too few display bins for a summary statistic to mean anything -- the trap that made
        # an earlier measurement report a 0.343 floor for a track whose real floor is 0.000,
        # off a 2,114 bp profile that yielded three bins.
        return "max"
    # The median, deliberately. An "ink fraction" (share of display bins above ~0) was tried
    # and is WRONG: it cannot tell "this track legitimately has hundreds of real peaks across
    # 1 Mb" from "max-pooling inflated the floor", so it flipped Cherimoya and ChromBPNet to
    # mean and re-broke the very defect this rule exists to fix. The median asks the narrower
    # question this rule is actually for -- has the typical bin left the floor.
    return "mean" if float(np.median(pooled)) > limit else "max"


def rescale_for_display(
    values,
    layer: str,
    normalizer=None,
    oracle_name: str | None = None,
    assay_id: str | None = None,
    log_scale: bool = False,
):
    """Single-track display rescale.  Canonical helper used by every
    track-rendering path (IGV WIG, matplotlib PNG, CoolBox, notebooks)
    so they share one source of truth for normalization semantics.

    Returns ``(out_values, cfg)`` where ``cfg`` is a dict with:

    - ``rescaled`` (bool): True iff CDF-based rescale was applied.
      False means the values were returned unchanged and the caller
      should autoscale per-track.
    - ``signed`` (bool): True iff the layer is signed (Borzoi RNA, Sei,
      LentiMPRA).  Signed tracks use symmetric ``[-DISPLAY_MAX, +DISPLAY_MAX]``;
      unsigned use ``[0, DISPLAY_MAX]``.
    - ``ymin`` / ``ymax`` (float): suggested y-axis limits.  Renderers
      can use these to set IGV ``min``/``max``, matplotlib ``set_ylim``,
      or CoolBox ``MinValue``/``MaxValue``.
    - ``floor_pctile`` / ``peak_pctile`` / ``display_max`` (float): the
      thresholds used (informational; same for every rendering path).

    All semantics:
      - 1.0 (unsigned) or ±1.0 (signed) = genome-wide p99 of |signal|
      - DISPLAY_MAX = 3.0 = 3× p99 above the floor (cap)
      - 0.0 (unsigned) = below the layer floor (genome-wide noise)

    Pass ``normalizer=None`` to opt out (returns values unchanged with
    ``rescaled=False``, ``ymin/ymax`` set to data min/max for autoscale).
    """
    import numpy as np

    if normalizer is None or oracle_name is None or assay_id is None:
        v = np.asarray(values)
        return values, {
            "rescaled": False, "signed": False,
            "ymin": float(v.min()) if v.size else 0.0,
            "ymax": float(v.max()) if v.size else 1.0,
            "floor_pctile": None, "peak_pctile": None,
            "display_max": _DISPLAY_MAX,
        }

    from .normalization import PerTrackNormalizer
    if not isinstance(normalizer, PerTrackNormalizer):
        v = np.asarray(values)
        return values, {
            "rescaled": False, "signed": False,
            "ymin": float(v.min()) if v.size else 0.0,
            "ymax": float(v.max()) if v.size else 1.0,
            "floor_pctile": None, "peak_pctile": None,
            "display_max": _DISPLAY_MAX,
        }

    signed = normalizer.is_signed(oracle_name, assay_id)
    if signed:
        out = normalizer.signed_floor_rescale_batch(
            oracle_name, assay_id, values,
            peak_pctile=_PEAK_PCTILE, max_value=_DISPLAY_MAX,
        )
        if out is None:
            v = np.asarray(values)
            return values, {
                "rescaled": False, "signed": True,
                "ymin": float(v.min()) if v.size else -1.0,
                "ymax": float(v.max()) if v.size else 1.0,
                "floor_pctile": None, "peak_pctile": _PEAK_PCTILE,
                "display_max": _DISPLAY_MAX,
            }
        return out, {
            "rescaled": True, "signed": True,
            "ymin": -_DISPLAY_MAX, "ymax": _DISPLAY_MAX,
            "floor_pctile": None, "peak_pctile": _PEAK_PCTILE,
            "display_max": _DISPLAY_MAX,
        }

    floor_p = _LAYER_FLOOR_PCTILE.get(layer, _DEFAULT_FLOOR_PCTILE)
    peak_p = _PEAK_PCTILE
    if log_scale:
        floor_p, peak_p = _LOG_FLOOR_PCTILE, _LOG_PEAK_PCTILE
    out = normalizer.perbin_floor_rescale_batch(
        oracle_name, assay_id, values,
        floor_pctile=floor_p, peak_pctile=peak_p, max_value=_DISPLAY_MAX,
        log_scale=log_scale,
    )
    if out is None:
        v = np.asarray(values)
        return values, {
            "rescaled": False, "signed": False,
            "ymin": float(v.min()) if v.size else 0.0,
            "ymax": float(v.max()) if v.size else 1.0,
            "floor_pctile": floor_p, "peak_pctile": peak_p,
            "display_max": _DISPLAY_MAX,
        }
    return out, {
        "rescaled": True, "signed": False,
        "ymin": 0.0, "ymax": _DISPLAY_MAX,
        "floor_pctile": floor_p, "peak_pctile": peak_p,
        "log_scale": log_scale,
        "display_max": _DISPLAY_MAX,
    }


def apply_floor_rescale(
    normalizer,
    oracle_name: str | None,
    assay_id: str,
    layer: str,
    ref_vals,
    alt_vals,
    log_scale: bool = False,
):
    """Floor-subtract + rescale a ref/alt value pair using the normalizer.

    Returns ``(rescaled, ref_out, alt_out, signed)``.

    - ``rescaled=True, signed=False``: unsigned floor-rescale, values map
      to ``[0, _DISPLAY_MAX]`` with layer-aware thresholds (p95/p99 for
      sharp signals, p90/p99 for broad domains).  1.0 = genome-wide p99
      peak.  IGV scale_cfg should be ``{min: 0, max: _DISPLAY_MAX}``.
    - ``rescaled=True, signed=True``: signed symmetric rescale, values
      map to ``[-_DISPLAY_MAX, +_DISPLAY_MAX]`` using ``p99(|cdf|)`` as
      the unit.  ±1.0 = genome-wide top-1% absolute effect.  IGV
      scale_cfg should be ``{min: -_DISPLAY_MAX, max: +_DISPLAY_MAX}``.
    - ``rescaled=False``: no normalizer / no CDF / lookup miss.  Caller
      should fall back to raw autoscale.

    Used by every IGV-rendering path so panels share the same semantics.
    """

    # Delegate to the unified single-track rescaler (rescale_for_display)
    # so IGV, matplotlib, CoolBox and notebook callers all share the same
    # semantics — only the wrapper differs (this one returns a 4-tuple
    # for the ref/alt pair instead of (values, cfg)).
    ref_out, cfg_ref = rescale_for_display(
        ref_vals, layer, normalizer=normalizer,
        oracle_name=oracle_name, assay_id=assay_id, log_scale=log_scale,
    )
    alt_out, cfg_alt = rescale_for_display(
        alt_vals, layer, normalizer=normalizer,
        oracle_name=oracle_name, assay_id=assay_id, log_scale=log_scale,
    )
    # Both ref/alt should have identical scale_cfg (same track, same CDF).
    # If either failed to rescale, fall back to passthrough.
    if not (cfg_ref["rescaled"] and cfg_alt["rescaled"]):
        return False, ref_vals, alt_vals, cfg_ref["signed"]
    return True, ref_out, alt_out, cfg_ref["signed"]

#: Log-band anchors, used only when the linear band is measured to clip too much.
#:
#: Chosen by measuring the rendered panel, not the CDF: p99.5/p99.9 with log1p takes
#: AlphaGenome CAGE to saturation 0.013 with its peak still at 3.00 -- the same regime as
#: Cherimoya. p99.9/p99.99 looked right from the CDF alone and is wrong: it drops the peak
#: to 1.24 and erases the track.
#: Appended to a track's label when it was re-rendered on the log band, so the reader knows
#: this panel's 1.0 is genome-wide p99.9 rather than p99.
_LOG_SCALE_LABEL = " (log scale)"

_LOG_FLOOR_PCTILE = 0.995
_LOG_PEAK_PCTILE = 0.999

#: Fraction of DISPLAYED bins allowed to sit at the ceiling before a track is re-rendered
#: on the log band.
#:
#: Saturation -- not ink -- is what makes a panel unreadable. Cherimoya *inks* 41% of its bins
#: and looks right, which is why an ink criterion was tried and failed.
#:
#: CALIBRATED ON THE CORPUS, NOT ON ONE PANEL, AND RE-VERIFIED WHENEVER THE AXIS MOVES.
#: An earlier value of 0.04 came from the geometric midpoint of a single panel's gap. Measured
#: instead across all 346 subtracks of the committed IGV panels, 0.04 cuts through the middle of
#: the population and would escalate Enformer CAGE tracks that render acceptably.
#:
#: Re-derived on 2026-08-12 after :data:`_DISPLAY_MAX` moved 3.0 -> 4.0, which lowers every
#: saturation figure and so invalidated the previous calibration. Same method, new axis:
#:
#:   must escalate      alphagenome CAGE, linear band as drawn      0.1085 - 0.1308
#:   ------------------------- gap, 2.49x -------------------------
#:   must NOT move      enformer CAGE substantia nigra (the top)    0.0435
#:                      corpus median 0.0022, p90 0.0101
#:
#: 0.075 sits inside that gap, so the constant is unchanged and no escalation decision flips:
#: CAGE still escalates, Enformer's CAGE tracks still do not. Note CAGE's linear saturation
#: barely moved with the ceiling (0.131 -> 0.1308) because its band is p95=0.0050 to p99=0.0405
#: against a maximum of 852 -- almost all real TSS signal is far above 4x the band, so a taller
#: axis does not unclip it. That is the whole reason the log band exists for this layer.
#:
#: The committed panels show CAGE POST-escalation (log band), so its trigger value cannot be read
#: off them; it has to be measured by rescaling a fresh prediction on the linear band. Every
#: other track in the corpus did not escalate, so those panel values are the linear values.
_MAX_DISPLAY_SATURATION = 0.075


def _display_saturation(values, bins_per: int, aggregation: str) -> tuple[float, float]:
    """Saturated fraction and peak of the values as they will be DRAWN.

    Not as they are computed: a display bin covers ``bins_per`` native bins, and pooling is
    what turns a 1.2% native clip rate into a 13.1% displayed one -- max-pooling gives each
    display bin 349 chances to inherit a clipped value. Measured natively, CAGE (0.005-0.014)
    is indistinguishable from the ChIP tracks (0.001-0.008) that must not move; measured as
    drawn, it separates from every one of them by 10x. So the trigger has to be applied here,
    after pooling is known.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0, 0.0
    if bins_per > 1:
        m = (v.size // bins_per) * bins_per
        if m >= bins_per:
            r = v[:m].reshape(-1, bins_per)
            v = r.mean(axis=1) if aggregation == "mean" else r.max(axis=1)
    return float((v >= _DISPLAY_MAX - 1e-3).mean()), float(v.max())


def escalate_scale_if_saturated(
    normalizer,
    oracle_name: str | None,
    assay_id: str,
    layer: str,
    raw_ref,
    raw_alt,
    disp_ref,
    disp_alt,
    bins_per: int,
    aggregation: str,
):
    """Re-render a track on the log band when the linear band clips too much of the panel.

    Returns ``(ref_out, alt_out, used_log)``, leaving the inputs untouched when the linear
    band is fine or when the log band does not actually help.

    WHY THIS IS MEASURED HERE rather than predicted from the CDF. The linear
    ``floor=p95, peak=p99`` convention assumes signal decays smoothly out of the background.
    That holds for accessibility and fails for base-resolution TSS/splice assays: AlphaGenome
    CAGE has p95=0.0050 and p99=0.0405 against a maximum of 852, so every real TSS from
    strength 1 to 3000 rendered at exactly 3.00, with 13.1% of the panel's bins pinned at the
    ceiling. Four genome-wide CDF statistics were tried as a proxy for that and every one of
    them overlaps between the tracks that need the log band and the tracks that must not move:

        max/p99.9        must-log p5 697, down to 172; must-stay p95 20.5, max 4212 (cbp ChIP)
        p99.9/p99        must-log p5 5.7;  must-stay p95 15.6
        p99/p95          must-log p5 3.0;  must-stay p95 10.0
        predicted clip   must-log p5 0.0028; must-stay p95 0.0045

    ``max/p99.9`` looked clean at 41x separation until ChromBPNet's ChIP tracks were included
    in the protected set; on a 10,000-point grid ``p99.99`` IS the maximum, so that statistic
    is a ratio to a single extreme order statistic -- the exact thing the null protocol warns
    against. There is no threshold on it that fixes CAGE without also log-scaling 130
    other tracks: 102 ChromBPNet ChIP, 10 Enformer and 8 Borzoi CAGE, 7 AlphaGenome TF-ChIP,
    2 ChromBPNet DNase and 1 Cherimoya DNase -- AlphaGenome's own ChIP tracks included.

    ACCEPTANCE IS TWO-SIDED, so a wrong trigger cannot damage a track: the log band is kept
    only if it leaves the strongest feature at or above 1.0 (genome-wide p99) AND either
    clears the saturation limit or at least halves the clipping. The peak half is what an
    earlier attempt lacked -- p99.9/p99.99 anchors dropped CAGE's peak to 1.24 of 3.0,
    "fixing" saturation by erasing the signal. The halving half is what stops an epsilon
    improvement from counting as a fix, and a degenerate band that collapses the track to a
    two-level barcode is rejected outright.
    """
    sat, _ = _display_saturation(disp_ref, bins_per, aggregation)
    if sat <= _MAX_DISPLAY_SATURATION:
        return disp_ref, disp_alt, False

    ok, log_ref, log_alt, _signed = apply_floor_rescale(
        normalizer, oracle_name, assay_id, layer, raw_ref, raw_alt, log_scale=True,
    )
    if not ok:
        return disp_ref, disp_alt, False

    log_sat, log_peak = _display_saturation(log_ref, bins_per, aggregation)

    # A log band whose anchors collapsed renders a two-level barcode -- every value a hair
    # above the floor at exactly 3.0, everything else at exactly 0.0 -- and it would pass
    # both tests below, because clipping guarantees peak 3.0. Reachable from real data:
    # chrombpnet CHIP:HEK293:ZNF24 has p99.5 = -7.4e-07 and p99.9 = -3.3e-10, which the
    # log path's ``max(x, 0.0)`` maps to the same 0.0, leaving denom pinned at 1e-9.
    if np.unique(np.asarray(log_ref, dtype=np.float64)).size < 3:
        return disp_ref, disp_alt, False

    # Acceptance is two-sided AND the improvement has to be real. ``log_sat < sat`` alone is
    # satisfied by an epsilon: a track going 0.550 -> 0.500 would be re-rendered, relabelled,
    # and still ship with half the panel pinned -- having paid the full cost of the log band
    # (compressed peaks, floor moved from p95 to p99.5) for five percentage points. So the
    # band must either clear the limit outright or at least halve the clipping.
    if log_peak >= 1.0 and (log_sat <= _MAX_DISPLAY_SATURATION or log_sat <= 0.5 * sat):
        return log_ref, log_alt, True
    return disp_ref, disp_alt, False


def _calculate_track_bin_size(
    resolution: int,
    window_bp: int,
    source_oracle: str,
) -> tuple[int, str]:
    """Calculate appropriate bin size and aggregation method.
    
    Returns:
        (bin_size, aggregation_method) where aggregation is "mean" or "max"
    """

    # Pooling is chosen by OUTPUT GEOMETRY, not by oracle name.
    #
    # A base-resolution model emits a profile: narrow, tall peaks one base wide.
    # Mean-pooling such a track over a display bin dilutes a single sharp peak
    # with hundreds of near-zero neighbours, and the floor-rescale then pushes
    # the diluted value toward the floor -- the panel ends up nearly empty.
    # A coarse-resolution model (AlphaGenome/Enformer at 128 bp, Borzoi at 32 bp)
    # has already integrated over its bin, so mean is the faithful summary and
    # max would overstate a single bin.
    #
    # This used to be a hardcoded list of two oracle NAMES, and that is exactly
    # how it broke: Cherimoya is a BPNet-family model with the same 1 bp output
    # geometry as ChromBPNet, but it was not in the list, so it fell through to
    # mean-pooling. Measured on the SORT1 multi-oracle panel (1,048,396 bp window,
    # 349 bp bins, DNASE:ENCSR149XIL): the ensemble 1 bp profile peaks at 11.10,
    # which max-pools to a rendered 3.000 -- the same ceiling ChromBPNet reaches --
    # but mean-pooled to a rendered 0.547. A 5.5x display-only dilution, in a
    # report whose entire purpose is cross-oracle comparison, with both panels
    # drawn on the same 0-3 axis. The scores were never affected: the 501 bp
    # window sum is linear and identical either way (log2FC 1.4576 vs
    # ChromBPNet's 1.3756).
    #
    # The criterion is base resolution, and the membership is declared rather than
    # predicated, because no predicate got it right. `resolution <= 1` catches the
    # right set today only by accident of which oracles exist; "spikiness" read off
    # the artefact points the wrong way (perbin max/p99 is 22 for Cherimoya against
    # 65 for AlphaGenome). Both were measured before being rejected.
    #
    # So membership is a per-oracle decision, written as one. The protection against
    # it drifting is not a cleverer predicate -- it is
    # tests/test_igv_pooling_is_declared_per_oracle.py, which enumerates the oracle
    # registry and fails when an oracle is neither listed as a base-resolution track
    # nor explicitly recorded as coverage. A silent fall-through is what broke
    # Cherimoya, which rendered at a fifth of its height for exactly that reason.
    if source_oracle in _POINT_PROFILE_ORACLES:
        # BPNet-family: base-resolution point profiles, sparse and one base wide.
        # ChromBPNet keeps its deliberate 20 bp preference (PR #79); the budget
        # bound below widens it on a wide window without changing aggregation.
        preferred = 20 if source_oracle == "chrombpnet" else window_bp // 3_000
        bin_size, aggregation = preferred, "max"
    elif source_oracle == "legnet":
        bin_size, aggregation = resolution, "max"
    else:
        bin_size, aggregation = window_bp // 3_000, "mean"

    # Then bound it, for every oracle including the two above.
    #
    # A preferred bin size is a rendering choice; the budget is a hard limit.
    # Returning bare `resolution` (LegNet) makes
    # `bins_per = max(1, bin_size // resolution)` equal 1 in
    # `_downsample_to_features` — one JSON feature per input bin, no
    # downsampling at all. That is how the LegNet report reached 131 MB from
    # 1.29 MB and became impossible to commit (GitHub rejects above 100 MiB).
    # ChromBPNet's fixed 20 bp has the same shape of problem on a wide window
    # (10,000 features at 200 kb), it just never got large enough to notice.
    #
    # The cap only binds when the window is wide: at ChromBPNet's real 2,114 bp
    # input the floor is 1 bp, so its deliberate 20 bp survives untouched.
    # Aggregation is never changed — widening a bin must not silently turn
    # max-pooling into mean-pooling.
    # See issue #129 and tests/test_igv_feature_budget.py.
    min_bin_for_budget = -(-window_bp // _MAX_FEATURES_PER_TRACK)  # ceil div
    bin_size = max(bin_size, min_bin_for_budget, resolution)
    return bin_size, aggregation

def build_igv_html(
    ref_pred,
    alt_pred,
    variant_chrom: str,
    variant_pos: int,
    ref_allele: str = "",
    alt_allele: str = "",
    gene_name: Optional[str] = None,
    genome: str = "hg38",
    bin_size: int = 0,
    normalizer=None,
    oracle_name: Optional[str] = None,
    modification_region: Optional[tuple[int, int]] = None,
) -> str:
    """Build the IGV.js browser configuration as an HTML fragment.

    Args:
        ref_pred: Reference OraclePrediction.
        alt_pred: Alternate OraclePrediction.
        variant_chrom: Chromosome.
        variant_pos: Variant position.
        ref_allele: Reference allele string.
        alt_allele: Alternate allele string.
        gene_name: Gene to mention in the header.
        genome: IGV genome identifier (default hg38).
        bin_size: Downsample bin size in bp.  0 = auto-detect.
        normalizer: Optional QuantileNormalizer with baseline backgrounds.
            When provided, signal values are mapped to genome-wide activity
            percentiles [0, 1], making all tracks directly comparable.
        oracle_name: Oracle name for baseline lookup (required if normalizer given).

    Returns:
        HTML string containing the IGV.js browser div + script.
    """
    from .scorers import classify_track_layer

    assay_ids = list(ref_pred.keys())
    if not assay_ids:
        return ""

    # Determine prediction window
    first = ref_pred[assay_ids[0]]
    pred_start = first.prediction_interval.reference.start
    pred_end = first.prediction_interval.reference.end
    window_bp = pred_end - pred_start

    # Auto bin size: target ~3000 features per track
    if bin_size <= 0:
        bin_size = max(1, window_bp // 3000)

    # Build tracks
    tracks = []

    # Variant / modification annotation track.
    # For region swaps and insertions, highlight the full affected region.
    # For point variants, highlight the single nucleotide position.
    if modification_region is not None:
        marker_start, marker_end = modification_region
        marker_label = f"{variant_chrom}:{marker_start+1:,}-{marker_end:,} ({ref_allele}>{alt_allele})"
    else:
        marker_start = variant_pos - 1
        marker_end = variant_pos + max(len(ref_allele), 1)
        marker_label = f"{variant_chrom}:{variant_pos:,} {ref_allele}>{alt_allele}"

    tracks.append({
        "name": f"Modification: {ref_allele}>{alt_allele}",
        "type": "annotation",
        "displayMode": "EXPANDED",
        "height": 25,
        "color": "red",
        "features": [{
            "chr": variant_chrom,
            "start": marker_start,
            "end": marker_end,
            "name": marker_label,
        }],
    })

    # When a PerTrackNormalizer is available, rescale raw bin values
    # using CDF-derived noise floor (p95) and peak threshold (p99).
    # This preserves peak shape (linear transform) while making tracks
    # comparable across cell types: 1.0 = top 1% of bins genome-wide.
    # Falls back to raw autoscale when no normalizer is available.
    use_floor = normalizer is not None and oracle_name is not None

    for assay_id in assay_ids:
        ref_track = ref_pred[assay_id]
        alt_track = alt_pred[assay_id]

        layer = classify_track_layer(ref_track)
        rgb = _LAYER_COLORS.get(layer, "70,130,180")

        t_res = ref_track.resolution
        actual_bp_in_array = len(ref_track.values) * t_res
        t_start = variant_pos - (actual_bp_in_array // 2)

        raw_ref, raw_alt = ref_track.values, alt_track.values
        ref_vals = ref_track.values
        alt_vals = alt_track.values

        # Apply layer-aware floor-subtract + rescale when available
        floor_ok = False
        signed_track = False
        used_log = False
        if use_floor:
            floor_ok, ref_vals, alt_vals, signed_track = apply_floor_rescale(
                normalizer, oracle_name, assay_id, layer, ref_vals, alt_vals,
            )

        track_bin_size, agg_method = _calculate_track_bin_size(
            t_res, window_bp, first.source_model,
        )
        if floor_ok:
            # Values are display-scaled by now, so the choice can be measured rather than
            # assumed. Un-rescaled tracks keep the static preference above: without a
            # display scale, "does the floor rise" has no reference to be asked against.
            bins_per = max(1, track_bin_size // t_res)
            # Signed tracks are excluded from BOTH measured decisions, deliberately.
            # ``choose_aggregation`` asks whether max-pooling lifts the floor, which has no
            # meaning for a track with no floor at zero: max over a bin holding a strong
            # repression and a weak activation returns the activation, so the repressive
            # half of the panel simply disappears. Measured on borzoi ENCFF734OLC+ (signed,
            # 32 bp, 11 native bins per display bin) the measured choice flips mean -> max
            # and takes displayed saturation 0.000 -> 0.138. 2,253 tracks are signed
            # (borzoi 1,543, alphagenome 667, sei 40, legnet 3), so they keep the static
            # geometry-based choice, which is what shipped and works.
            if not signed_track:
                agg_method = choose_aggregation(ref_vals, bins_per)
                ref_vals, alt_vals, used_log = escalate_scale_if_saturated(
                    normalizer, oracle_name, assay_id, layer, raw_ref, raw_alt,
                    ref_vals, alt_vals, bins_per, agg_method,
                )

        # Signed tracks have negative values that ``skip_zeros`` would
        # incorrectly count as background — disable the threshold drop
        # so the repressive half stays in the wig features.
        ref_features = _downsample_to_features(
            ref_vals, variant_chrom, t_start, t_res, track_bin_size,
            skip_zeros=not (floor_ok or signed_track),
            aggregation_method=agg_method
        )
        alt_features = _downsample_to_features(
            alt_vals, variant_chrom, t_start, t_res, track_bin_size,
            skip_zeros=not (floor_ok or signed_track),
            aggregation_method=agg_method
        )

        group_id = assay_id.replace(":", "_").replace(" ", "_")
        if floor_ok and signed_track:
            # Symmetric signed scale: ±1.0 = genome-wide top-1% |effect|.
            scale_cfg = {"min": -_DISPLAY_MAX, "max": _DISPLAY_MAX, "autoscale": False}
            name_suffix = ""
        elif floor_ok:
            scale_cfg = {"min": 0, "max": _DISPLAY_MAX, "autoscale": False}
            # Disclose the transform. 1.0 means genome-wide p99 on a linear track and
            # p99.9 on a log one, so two same-assay panels in one report can legitimately
            # sit on different bands -- BCL11A's two CAGE:K562 tracks measured 0.053 and
            # 0.036, and only the first escalated. The axis was always per-track (1.0 is
            # *this* track's percentile, not a shared raw value), so mixing is not new;
            # leaving it unlabelled would be. Follows the ``(per-track norm)`` precedent.
            name_suffix = _LOG_SCALE_LABEL if used_log else ""
        else:
            scale_cfg = {"autoscale": True, "autoscaleGroup": group_id}
            name_suffix = ""

        # Build a human-readable display name from track metadata.
        # Use _track_description from variant_report for enriched CHIP names
        # (e.g. "CHIP:CEBPA:HepG2" instead of generic "CHIP:HepG2").
        from chorus.analysis.variant_report import _track_description
        display_name = _track_description(ref_track) or assay_id
        if display_name == assay_id:
            meta = getattr(ref_track, "metadata", None)
            if meta and isinstance(meta, dict) and meta.get("description"):
                display_name = meta["description"]
            elif hasattr(ref_track, "assay_type") and hasattr(ref_track, "cell_type"):
                display_name = f"{ref_track.assay_type}:{ref_track.cell_type}"

        # Merged overlay: ref (grey) + alt (coloured) on same panel
        source_model = first.source_model
        tracks.append({
            "name": f"{display_name}{name_suffix}",
            "type": "merged",
            "height": 80,
            "tracks": [
                {
                    "type": "wig",
                    "name": f"{display_name} ref",
                    "color": f"rgb({_REF_COLOR})",
                    "windowFunction": browser_window_function(used_log),
                    **scale_cfg,
                    "features": ref_features,
                },
                {
                    "type": "wig",
                    "name": f"{display_name} alt",
                    "color": f"rgb({rgb})",
                    "windowFunction": browser_window_function(used_log),
                    **scale_cfg,
                    "features": alt_features,
                },
            ],
        })

    # ROI: red stripe across all tracks highlighting the modification
    roi = [{
        "name": "Modification",
        "color": "rgba(255, 0, 0, 0.12)",
        "features": [{
            "chr": variant_chrom,
            "start": marker_start,
            "end": marker_end,
        }],
    }]

    # Initial locus: full prediction window
    locus = f"{variant_chrom}:{pred_start}-{pred_end}"

    igv_options = igv_browser_config(locus, tracks, roi, genome=genome)

    # Build HTML fragment
    options_json = json.dumps(igv_options, separators=(",", ":"))

    # Inline IGV.js from local cache (no network needed) or fall back to CDN.
    local = _ensure_igv_local()
    if local is not None:
        igv_js = local.read_text()
        igv_script_tag = f"<script>{igv_js}</script>"
    else:
        igv_script_tag = f'<script src="{_IGV_CDN}"></script>'

    html = f"""
<div id="igv-div" style="margin: 1rem 0; min-height: 400px;"></div>
{igv_script_tag}
<script>
(async function() {{
    try {{
        const browser = await igv.createBrowser(
            document.getElementById("igv-div"),
            {options_json}
        );
        console.log("IGV browser created successfully");
    }} catch(e) {{
        console.error("IGV error:", e);
        document.getElementById("igv-div").innerHTML =
            '<p style="color:red;padding:1rem">Error loading IGV browser: ' + e.message + '</p>';
    }}
}})();
</script>
"""
    return html


def _downsample_to_features(
    values: np.ndarray,
    chrom: str,
    start: int,
    resolution: int,
    bin_size: int,
    skip_zeros: bool = True,
    aggregation_method: str = "mean"
) -> list[dict]:
    """Downsample a signal array into IGV wig features.

    Aggregates bins by taking the mean over each output bin.
    When *skip_zeros* is True (default for raw data), bins with near-zero
    signal are omitted to reduce JSON size.  Set to False for
    percentile-normalized data to avoid gaps.
    """
    n = len(values)
    vals = values.astype(np.float64)

    # Number of original bins per output bin
    bins_per = max(1, bin_size // resolution)

    features = []
    if skip_zeros:
        threshold = float(np.percentile(np.abs(vals[vals != 0]), 5)) if np.any(vals != 0) else 0
    else:
        threshold = -1  # never skip

    for i in range(0, n, bins_per):
        chunk = vals[i:i + bins_per]

        if aggregation_method == "mean":
            v = float(np.mean(chunk))
        else:
            v = float(np.max(chunk))

        # Skip near-zero bins to reduce JSON size (only for raw data)
        if skip_zeros and abs(v) < threshold * 0.1:
            continue

        feat_start = start + i * resolution
        feat_end = start + min(i + bins_per, n) * resolution

        features.append({
            "chr": chrom,
            "start": feat_start,
            "end": feat_end,
            "value": round(v, 4),
        })

    return features
