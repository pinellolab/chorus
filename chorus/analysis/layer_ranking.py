"""Rank tissues (or body systems) within each molecular layer from per-track variant effects.

This is the analysis behind the "tissue rank within each molecular layer" figure. It is deliberately
split into small steps so an agent can run them one at a time:

1. **Discovery** (:func:`chorus.analysis.discovery.discover_variant_effects`, MCP ``discover_variant``)
   scores every track of one oracle and writes two tables next to the HTML report:

   * ``*_track_scores.csv``    one row per unique track (the source of truth);
   * ``*_prediction_matrix.csv`` one row per cell type, per model x layer: every track's percentile
     plus Fisher's combined p (a human-readable view of the same data).

2. **Ranking** (:func:`rank_variant_tissues`, MCP ``rank_variant_tissues``) reads one track table per
   oracle, groups cell types into tissues or body systems with ``cell_type_annotation.csv``,
   aggregates each group's track p-values with Fisher or ACAT, ranks the groups within each layer and
   draws the rank table.

No oracle is needed for step 2, so the same discovery output can be re-ranked with another method or
grouping without a model run.

Statistics
----------
Each track's percentile ``|pctile|`` (effect magnitude against that track's own background null) becomes a
one-sided p-value ``p = max(1 - |pctile|, 1/(N+1))`` with ``N = 10_000`` points in the stored null, so
a track beyond the whole null still has a finite p (about 1e-4).

* **Fisher**: ``X2 = -2 sum ln p_i ~ chi2(2k)``. Asks whether the group is affected *broadly*; assumes
  independent tracks, which tracks of one tissue are not, so p is optimistic and grows with ``k``.
* **ACAT** (Cauchy combination, equal weights): dominated by the smallest p, valid under arbitrary
  dependence. Asks whether the group contains *a strong hit*; penalises large groups roughly like
  Bonferroni.

Ties (common at the 1/(N+1) floor) are broken by the largest ``|raw_score|`` in the group.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

logger = logging.getLogger(__name__)

#: Points of the stored compact background CDFs (``BackgroundDistribution.to_compact_cdf``): the p floor is 1/(N+1).
N_NULL = 10_000

METHODS = ("fisher", "acat")

#: Rows of the figure. ``tissue`` -> ``tissue_organ`` column of the annotation, ``system`` -> ``tissue_system``.
GROUP_LEVELS = {
    "tissue": "tissue_organ", "organ": "tissue_organ", "tissue_organ": "tissue_organ",
    "system": "tissue_system", "tissue_system": "tissue_system",
}
_UNITS = {"tissue_organ": ("tissue", "tissues"), "tissue_system": ("system", "systems")}

#: Layers shown by default, in column order. PRO-CAP is left out: it has 6 cell types / 12 tracks.
DEFAULT_LAYERS = ("DNase", "ATAC", "ChIP-TF", "ChIP-histone", "RNA-seq", "CAGE")

#: The oracle that supplies each layer in the published SORT1 figure. A layer whose designated oracle is absent
#: from the input falls back to the oracle with the most tracks in that layer (and says so in ``notes``).
DEFAULT_LAYER_ORACLES = {
    "DNase": "Cherimoya", "ATAC": "Cherimoya", "ChIP-TF": "BPNet",
    "ChIP-histone": "AlphaGenome", "RNA-seq": "AlphaGenome", "CAGE": "AlphaGenome",
}

#: Registry oracle name -> display label (also the ``model`` column of the track table).
MODEL_LABELS = {
    "alphagenome": "AlphaGenome", "alphagenome_pt": "AlphaGenomePT", "cherimoya": "Cherimoya",
    "chrombpnet": "ChromBPNet", "enformer": "Enformer", "borzoi": "Borzoi", "sei": "Sei",
    "legnet": "LegNet", "epinformerseq": "EPInformerSeq",
}

TRACK_TABLE_COLUMNS = ["model", "molecular_layer", "track_id", "cell_type_key", "cell_type", "percentile",
                       "raw_score"]

_STRAND_SUFFIX = re.compile(r":[+-]$")


# ---------------------------------------------------------------------------
# Track tables (written by discovery, read by the ranking)
# ---------------------------------------------------------------------------

def display_layer(assay: str | None, layer: str | None) -> str:
    """Name of the figure column a track belongs to, from its assay and scoring layer."""
    a = (assay or "").upper()
    if a == "CHIP":
        return "ChIP-histone" if layer == "histone_marks" else "ChIP-TF"
    mapped = {"DNASE": "DNase", "ATAC": "ATAC", "CAGE": "CAGE", "PRO_CAP": "PRO-CAP", "RNA": "RNA-seq"}
    if a in mapped:
        return mapped[a]
    # No assay (BPNet heads carry none when the oracle catalogue is unavailable): the scoring layer still
    # decides the ChIP split and RNA. Accessibility / TSS layers are ambiguous without an assay, so they
    # keep their layer name rather than being guessed into DNase vs ATAC.
    by_layer = {"tf_binding": "ChIP-TF", "histone_marks": "ChIP-histone", "gene_expression": "RNA-seq"}
    return by_layer.get(layer) or assay or str(layer)


def model_label(oracle_name: str, molecular_layer: str) -> str:
    """Display model of a track. ChromBPNet's TF-ChIP heads are BPNet; its accessibility models stay ChromBPNet."""
    n = oracle_name.lower()
    if n == "chrombpnet":
        return "BPNet" if molecular_layer == "ChIP-TF" else "ChromBPNet"
    return MODEL_LABELS.get(n, oracle_name)


def track_table_from_effects(effects: Iterable, oracle_name: str, catalog: dict | None = None) -> pd.DataFrame:
    """One row per unique track from discovery's :class:`TrackEffect` list.

    ``catalog`` maps a (strand-stripped) ``track_id`` to ``{"cell_type": ..., "assay": ...}`` taken from the
    oracle's ``describe_tracks()``; where present it overrides what discovery parsed from the track
    description, which is wrong for BPNet ids. Tracks without a percentile (no background null) are dropped.
    Strands of one track (``CHIP:cell:TF:+`` / ``:-``) are not independent evidence, so only the stronger is kept.
    """
    catalog = catalog or {}
    rows = []
    for e in effects:
        if e.effect_pctile is None or not np.isfinite(e.effect_pctile):
            continue
        tid = _STRAND_SUFFIX.sub("", str(e.assay_id))
        cat = catalog.get(tid, {})
        cell = cat.get("cell_type") or e.cell_type
        if not isinstance(cell, str) or not cell.strip():
            continue
        layer = display_layer(cat.get("assay") or getattr(e, "assay", ""), e.layer)
        rows.append({"model": model_label(oracle_name, layer), "molecular_layer": layer, "track_id": tid,
                     "cell_type_key": cell.strip().lower(), "cell_type": cell,
                     "percentile": abs(float(e.effect_pctile)), "raw_score": float(e.raw_score)})
    if not rows:
        return pd.DataFrame(columns=TRACK_TABLE_COLUMNS)
    t = pd.DataFrame(rows)
    t["_abs_raw"] = t.raw_score.abs()
    t = t.sort_values(["percentile", "_abs_raw"], ascending=False).drop_duplicates(
        ["molecular_layer", "model", "track_id"])
    t = t.sort_values(["molecular_layer", "model", "cell_type_key", "percentile"],
                      ascending=[True, True, True, False])
    return t[TRACK_TABLE_COLUMNS].reset_index(drop=True)


def _slug(model: str, layer: str) -> str:
    return f"{model}_{layer.lower().replace('-', '_').replace(' ', '')}".lower()


def to_p(v, n_null: int = N_NULL):
    """Empirical p of a track: ``1 - |percentile|``, floored at 1/(N+1) (percentile 1.0 only means 'beyond the null')."""
    return np.clip(1.0 - np.asarray(v, float), 1.0 / (n_null + 1), 1.0)


def fisher_logp(v, n_null: int = N_NULL) -> float:
    """ln of Fisher's combined p, finite however many tracks sit at the floor.

    ``X2 = -2 sum ln p_i ~ chi2(2k)`` and ``X2/2 ~ Gamma(k)``; for integer ``k`` its survival function is the
    Poisson sum ``Q = exp(-x) * sum_{j<k} x^j / j!`` with ``x = X2/2``, evaluated here with ``logsumexp``.
    ``scipy.stats.chi2.logsf`` returns ``-inf`` once a group has ~120+ tracks at the floor (ln p below about
    -700), and every such group would then tie. This agrees with it to 1e-15 where it is finite.
    """
    p = to_p(v, n_null)
    x = float(-np.log(p).sum())
    if x <= 0.0:
        return 0.0
    j = np.arange(len(p))
    return float(-x + logsumexp(j * np.log(x) - gammaln(j + 1)))


def acat_p(v, n_null: int = N_NULL) -> float:
    """Cauchy combination test (equal weights); valid under arbitrary dependence."""
    p = to_p(v, n_null)
    t = np.tan((0.5 - p) * np.pi).mean()
    # the 1/t form keeps precision for tiny p
    return float(np.arctan(1.0 / t) / np.pi) if t > 0 else float(0.5 - np.arctan(t) / np.pi)


def prediction_matrix(table: pd.DataFrame) -> pd.DataFrame:
    """One row per cell type; per model x layer ``<m>_<l>_all`` (every track, strongest first) and ``<m>_<l>_agg``.

    ``_agg`` is Fisher's combined p over the cell type's tracks in that model x layer (smaller = stronger).
    """
    rows: dict[str, dict] = {}
    for (model, layer), d in table.groupby(["model", "molecular_layer"]):
        base = _slug(model, layer)
        for key, x in d.groupby("cell_type_key"):
            x = x.sort_values("percentile", ascending=False)
            rows.setdefault(key, {})[f"{base}_all"] = ", ".join(
                f"{p:.4f} ({tid})" for p, tid in zip(x.percentile, x.track_id))
            rows[key][f"{base}_agg"] = f"{np.exp(fisher_logp(x.percentile.values)):.3e}"
    labels = table.groupby("cell_type_key").cell_type.agg(lambda s: s.value_counts().index[0])
    if not rows:
        return pd.DataFrame(columns=["cell_type_key", "cell_type_label"])
    m = pd.DataFrame.from_dict(rows, orient="index")
    m = m[sorted(m.columns, key=lambda c: (c.rsplit("_", 1)[0], c.endswith("_agg")))]
    m.index.name = "cell_type_key"
    m.insert(0, "cell_type_label", labels.reindex(m.index))
    return m.sort_index().reset_index()


def _header(variant_label: str, what: str) -> list[str]:
    return [f"# {variant_label}",
            "# Sign convention: raw_score = alt - ref (positive = the alt allele raises the predicted signal).",
            "# percentile = |effect percentile| against the track's own background null, i.e. a MAGNITUDE in [0,1] "
            "with no sign;",
            "# the direction of a track is the sign of raw_score in the *_track_scores.csv table.",
            f"# {what} Lines starting with '#' come first; read with pandas.read_csv(path, skiprows=<their count>)."]


def _write_commented(df: pd.DataFrame, path: Path, header: list[str]) -> None:
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        df.to_csv(f, index=False, na_rep="")


def write_discovery_tables(table: pd.DataFrame, out_dir: str | Path, stem: str, variant_label: str) -> dict:
    """Write ``<stem>_track_scores.csv`` and ``<stem>_prediction_matrix.csv``; return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tracks_path = out / f"{stem}_track_scores.csv"
    matrix_path = out / f"{stem}_prediction_matrix.csv"
    _write_commented(table, tracks_path, _header(variant_label, "One row per unique track."))
    _write_commented(prediction_matrix(table), matrix_path,
                     _header(variant_label, "One row per cell type; <model>_<layer>_all lists every track, "
                                            "<model>_<layer>_agg is Fisher's combined p."))
    return {"track_scores_csv": str(tracks_path), "prediction_matrix_csv": str(matrix_path)}


# ---------------------------------------------------------------------------
# RNA-seq: one score per (gene, track), kept apart from the per-track table
# ---------------------------------------------------------------------------
# An RNA effect is a property of a track AND a gene (the mean over the gene's exons), so a locus with 30 genes
# has 30 x ~670 rows for AlphaGenome alone. They live in their own wide tables, genes as rows and tracks as
# columns, so the per-track table stays small and the user chooses the gene at ranking time.

RNA_LAYER = "RNA-seq"
RNA_META_COLS = ["gene", "distance_bp", "strand", "n_tracks", "n_sig_0.95", "median_abs_raw"]
_RNA_STRAND = re.compile(r"/[+-]$")
RNA_STRAND_MODES = ("collapse", "matched")


def _track_strand(track_id: str) -> str:
    """``+`` / ``-`` from an id ending in ``/+`` or ``/-``; ``.`` (unstranded) otherwise."""
    m = _RNA_STRAND.search(str(track_id))
    return m.group(0)[1] if m else "."


def _strand_matched(gene_strand: pd.Series, track_strand: pd.Series) -> pd.Series:
    """A gene is read on an unstranded track, or on the stranded track of its own strand (or any, if its strand is unknown)."""
    return (track_strand == ".") | (gene_strand.isin(["?", "", None])) | (gene_strand == track_strand)


def rna_rows_to_tables(rows: Iterable[dict], oracle_name: str, thr: float = 0.95):
    """Pivot per-(gene, track) rows into the three RNA tables.

    ``rows`` are dicts with ``gene``, ``distance_bp``, ``strand`` (the gene's, optional), ``track_id``,
    ``cell_type``, ``percentile`` (signed, None when the effect is inside the noise floor) and ``raw_score``.

    Strand-specific tracks are kept as separate columns: which strand a gene may be read on is a decision for
    the consumer (:func:`load_rna_tables` ``strand_mode``), and collapsing here would make it irreversible. The
    published ranking keeps the stronger strand; the Liver panel reads a gene only on its own strand, and the two
    pick a different track in 17% of SORT1's liver rows. The gene summary (``n_tracks``, ``n_sig``,
    ``median_abs_raw``) counts only strand-matched reads, the ones that mean something.

    Returns ``(percentiles, raw_scores, tracks)``: genes x tracks matrices (percentiles keep their sign; the
    leading columns are gene metadata) and one row per track naming its cell type, model and strand.
    """
    r = pd.DataFrame(list(rows))
    if r.empty:
        return None
    if "strand" not in r:
        r["strand"] = "?"
    r["strand"] = r.strand.fillna("?").astype(str)
    # None (noise floor) becomes NaN; an all-None column would otherwise stay object dtype and break .abs()
    r["percentile"] = pd.to_numeric(r.percentile, errors="coerce")
    r["raw_score"] = pd.to_numeric(r.raw_score, errors="coerce")
    r["_a"] = r.percentile.abs().fillna(-1.0)
    r["_r"] = r.raw_score.abs().fillna(-1.0)
    r = r.sort_values(["_a", "_r"], ascending=False).drop_duplicates(["gene", "track_id"])
    tracks = r.groupby("track_id").agg(cell_type=("cell_type", "first")).reset_index()
    tracks["track_key"] = tracks.track_id.map(lambda t: _RNA_STRAND.sub("", str(t)))
    tracks["strand"] = tracks.track_id.map(_track_strand)
    tracks["cell_type_key"] = tracks.cell_type.str.strip().str.lower()
    tracks["model"] = model_label(oracle_name, RNA_LAYER)
    tracks = tracks.sort_values(["cell_type_key", "track_id"])[
        ["track_id", "track_key", "strand", "cell_type_key", "cell_type", "model"]]
    genes = r.groupby("gene").agg(distance_bp=("distance_bp", "first"), strand=("strand", "first")).sort_values(
        "distance_bp")
    pct = r.pivot(index="gene", columns="track_id", values="percentile").reindex(index=genes.index,
                                                                              columns=tracks.track_id)
    raw = r.pivot(index="gene", columns="track_id", values="raw_score").reindex(index=genes.index,
                                                                             columns=tracks.track_id)
    ok = pd.DataFrame({t: _strand_matched(genes.strand, pd.Series(st, index=genes.index))
                       for t, st in zip(tracks.track_id, tracks.strand)})
    mp, mr = pct.where(ok), raw.where(ok)
    # Percentiles saturate for RNA (84% of SORT1 gene x track scores are at 1.0 and the top genes all have
    # ~390 'significant' tracks), so n_sig barely separates genes; the median |log fold change| does
    # (PSRC1 0.047, CELSR2 0.021, MYBPHL 0.014, SORT1 0.008) and is what a user choosing a gene should look at.
    meta = pd.DataFrame({"distance_bp": genes.distance_bp.astype(int), "strand": genes.strand,
                         "n_tracks": mp.notna().sum(axis=1), f"n_sig_{thr:g}": (mp.abs() >= thr).sum(axis=1),
                         "median_abs_raw": mr.abs().median(axis=1).round(5)})
    pct = pd.concat([meta, pct], axis=1)
    pct.index.name = raw.index.name = "gene"
    return pct.reset_index(), raw.reset_index(), tracks


def write_rna_tables(rows: Iterable[dict], oracle_name: str, out_dir: str | Path, stem: str,
                     variant_label: str) -> dict:
    """Write ``<stem>_rna_gene_percentiles.csv`` (+ ``_raw_scores.csv`` and ``_tracks.csv``); return paths and genes.

    The raw-score matrix is not optional: 202 of 390 SORT1 RNA tracks sit at the percentile floor, and the
    larger raw effect is what breaks ties between tissues there (7-10 of 49 tissue ranks move without it).
    """
    built = rna_rows_to_tables(rows, oracle_name)
    if built is None:
        return {}
    pct, raw, tracks = built
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pct_p, raw_p, trk_p = (out / f"{stem}_rna_gene_{k}.csv" for k in ("percentiles", "raw_scores", "tracks"))
    head = _header(variant_label, "RNA-seq: one row per gene, one column per track (signed effect percentile).")
    head.append(f"# companions: {raw_p.name} | {trk_p.name}")
    _write_commented(pct, pct_p, head)
    _write_commented(raw, raw_p, _header(variant_label, "RNA-seq: raw_score (signed log fold change) per gene x track."))
    _write_commented(tracks, trk_p, _header(variant_label, "RNA-seq tracks: id, cell type and model of each column."))
    sig_col = next(c for c in pct.columns if c.startswith("n_sig_"))
    genes = pct[["gene", "distance_bp", "strand", "n_tracks", sig_col, "median_abs_raw"]].rename(
        columns={sig_col: "n_sig_0.95"})
    return {"rna_gene_percentiles_csv": str(pct_p), "rna_gene_raw_scores_csv": str(raw_p),
            "rna_tracks_csv": str(trk_p), "rna_genes": genes.to_dict("records")}


def _is_rna_matrix(path: str | Path) -> bool:
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                return line.startswith("gene,distance_bp")
    return False


def load_rna_tables(path: str | Path, strand_mode: str = "collapse"):
    """Long RNA rows (with a ``gene`` column), the per-gene summary and the variant label from a
    ``*_rna_gene_percentiles.csv``.

    ``strand_mode`` decides how strand-specific tracks become one track per (gene, ``track_key``):

    * ``"collapse"`` (the ranking's rule, and what the committed SORT1 figure used): the two strands of a track
      are one piece of evidence and the stronger |percentile| wins (ties: larger |raw_score|), whatever the gene's
      strand.
    * ``"matched"``: a gene is read only on unstranded tracks and on the stranded track of its own strand.
    """
    if strand_mode not in RNA_STRAND_MODES:
        raise ValueError(f"strand_mode must be one of {RNA_STRAND_MODES}, got {strand_mode!r}")
    pct, header = _read_commented_csv(path)
    comp = next((h for h in header if h.startswith("# companions:")), None)
    if comp is None:
        raise ValueError(f"{path} lists no companion files; pass the *_rna_gene_percentiles.csv written by discovery")
    raw_name, trk_name = (x.strip() for x in comp.split(":", 1)[1].split("|"))
    raw, _ = _read_commented_csv(Path(path).parent / raw_name)
    tracks, _ = _read_commented_csv(Path(path).parent / trk_name)
    if "track_key" not in tracks:                                  # tables written before strands were kept
        tracks["track_key"] = tracks.track_id.map(lambda t: _RNA_STRAND.sub("", str(t)))
        tracks["strand"] = tracks.track_id.map(_track_strand)
    if "strand" not in pct:
        pct.insert(2, "strand", "?")
    meta_cols = [c for c in pct.columns if c in RNA_META_COLS or c.startswith("n_sig_")]
    summary = pct[meta_cols].copy()
    long = (pct.drop(columns=[c for c in meta_cols if c != "gene"]).melt(id_vars="gene", var_name="track_id",
                                                                          value_name="percentile")
            .merge(raw.melt(id_vars="gene", var_name="track_id", value_name="raw_score"), on=["gene", "track_id"])
            .merge(tracks.rename(columns={"strand": "track_strand"}), on="track_id", how="left")
            .merge(summary[["gene", "strand"]].rename(columns={"strand": "gene_strand"}), on="gene"))
    if strand_mode == "matched":
        long = long[_strand_matched(long.gene_strand.astype(str), long.track_strand)]
    long = long[long.percentile.notna()] if strand_mode == "collapse" else long[long.raw_score.notna()]
    long["track_id"] = long.track_key
    long["molecular_layer"] = RNA_LAYER
    long["signed_percentile"] = long.percentile
    long["percentile"] = long.percentile.abs()
    long["_a"] = long.percentile.fillna(-1.0)
    long["_r"] = long.raw_score.abs().fillna(-1.0)
    long = long.sort_values(["_a", "_r"], ascending=False).drop_duplicates(["gene", "track_id"])
    cols = ["gene"] + TRACK_TABLE_COLUMNS
    extra = ["signed_percentile", "track_strand", "gene_strand"]
    return long[cols + extra].reset_index(drop=True), summary, (header[0].lstrip("# ").strip() if header else None)

def _read_commented_csv(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Read a CSV with a leading block of ``#`` lines.

    Not ``comment='#'``: Sei track ids contain ``#`` (``TA#...``), and pandas would cut those rows short.
    """
    header = []
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            header.append(line.rstrip("\n"))
    return pd.read_csv(path, skiprows=len(header)), header


def _clean_table(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate the columns of a track table, take |percentile|, keep the strongest row per track."""
    missing = {"model", "molecular_layer", "track_id", "cell_type_key", "percentile"} - set(df.columns)
    if missing:
        raise ValueError(f"{source} is not a track table (missing columns {sorted(missing)}). Pass the "
                         f"*_track_scores.csv file written by discover_variant, not the prediction matrix.")
    df = df.copy()
    if "raw_score" not in df.columns:
        df["raw_score"] = np.nan
    if "cell_type" not in df.columns:
        df["cell_type"] = df.cell_type_key
    df = df[TRACK_TABLE_COLUMNS]
    df["percentile"] = df.percentile.abs()
    df["_abs_raw"] = df.raw_score.abs()
    df = df.sort_values(["percentile", "_abs_raw"], ascending=False).drop_duplicates(
        ["molecular_layer", "model", "track_id"]).drop(columns="_abs_raw")
    return df.reset_index(drop=True)


def load_track_tables(paths: str | Path | Sequence[str | Path]) -> tuple[pd.DataFrame, str | None]:
    """Concatenate one or more ``*_track_scores.csv`` files; also return the variant label from the first header."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames, label = [], None
    for p in paths:
        df, header = _read_commented_csv(p)
        frames.append(_clean_table(df, str(p)))
        if label is None and header:
            label = header[0].lstrip("# ").strip() or None
    return _clean_table(pd.concat(frames, ignore_index=True), "the track tables"), label


def _add_rna_gene(table, label, rna_paths, rna_gene, notes):
    """Append the chosen gene's RNA-seq rows to the per-track table.

    Returns ``(table, label, selected_gene_or_None, genes_available)``. Never picks a gene silently when the
    choice matters: with several genes and no ``rna_gene`` the RNA-seq column is omitted and the notes say why.
    """
    longs, sums = [], []
    for q in rna_paths:
        long, summary, lab = load_rna_tables(q)
        longs.append(long)
        sums.append(summary)
        label = label or lab
    long = pd.concat(longs, ignore_index=True)
    summary = pd.concat(sums, ignore_index=True).drop_duplicates("gene")
    sig = next(c for c in summary.columns if c.startswith("n_sig_"))
    summary = summary.sort_values(["median_abs_raw", "distance_bp"], ascending=[False, True])
    available = summary.gene.tolist()
    by_lower = {g.lower(): g for g in available}
    if rna_gene:
        if rna_gene.lower() not in by_lower:
            raise ValueError(f"rna_gene {rna_gene!r} is not among the {len(available)} genes scored: {available}")
        gene = by_lower[rna_gene.lower()]
    elif len(available) == 1:
        gene = available[0]
    else:
        top = ", ".join(f"{r.gene} (median |logFC| {r.median_abs_raw:.3g}, {int(r.distance_bp):,} bp)"
                        for _, r in summary.head(8).iterrows())
        notes.append(f"RNA-seq: {len(available)} genes were scored and none was chosen, so the column is omitted; "
                     f"pass rna_gene. Largest effect first: {top}")
        return table, label, None, available
    if (table.molecular_layer == RNA_LAYER).any():
        notes.append("RNA-seq rows already in the track table were replaced by the chosen gene's")
        table = table[table.molecular_layer != RNA_LAYER]
    rows = long[long.gene == gene].drop(columns="gene")
    return _clean_table(pd.concat([table, rows], ignore_index=True), "the track tables"), label, gene, available


def default_annotation_path() -> Path:
    return Path(__file__).parent / "data" / "cell_type_annotation.csv"


def load_annotation(path: str | Path | None = None) -> pd.DataFrame:
    """``cell_type_annotation.csv``: cell type -> ``tissue_organ`` / ``tissue_system`` (UBERON / CL ancestry)."""
    df, _ = _read_commented_csv(path or default_annotation_path())
    missing = {"cell_type_key", "tissue_organ", "tissue_system"} - set(df.columns)
    if missing:
        raise ValueError(f"annotation lacks columns {sorted(missing)}")
    return df


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def resolve_layer_oracles(table: pd.DataFrame, layers: Sequence[str],
                          layer_oracles: dict | None = None) -> tuple[dict, list[str]]:
    """Pick one oracle per layer: explicit choice, else the published figure's, else the one with most tracks."""
    chosen, notes = {}, []
    layer_oracles = layer_oracles or {}
    for layer in layers:
        d = table[table.molecular_layer == layer]
        if d.empty:
            notes.append(f"layer {layer}: no tracks in the input, column omitted")
            continue
        counts = d.groupby("model").size().sort_index()
        by_lower = {m.lower(): m for m in counts.index}
        explicit = layer_oracles.get(layer)
        if explicit is not None:
            if explicit.lower() not in by_lower:
                raise ValueError(f"layer {layer}: oracle {explicit!r} requested but the input only has "
                                 f"{sorted(counts.index)} for it")
            chosen[layer] = by_lower[explicit.lower()]
            continue
        want = DEFAULT_LAYER_ORACLES.get(layer)
        if want and want.lower() in by_lower:
            chosen[layer] = by_lower[want.lower()]
            continue
        chosen[layer] = counts.idxmax()
        notes.append(f"layer {layer}: designated oracle {want or '-'} not in the input; using {chosen[layer]} "
                     f"({int(counts.max())} tracks)")
    return chosen, notes


def rank_groups(table: pd.DataFrame, group_map: pd.Series, layer_oracles: dict, method: str, thr: float = 0.95,
                k: int = 3, min_tracks: int = 1, min_layers_for_global: int = 4):
    """Rank groups within each layer. Returns ``(ranks, scores, ntracks, nsigs, extra)``.

    ``scores`` hold ln(combined p) of ``method``; ``extra`` hold ACAT p, Fisher ln p and the top-k mean of every
    group so the methods can be compared without a second pass. ``Global`` orders groups by median rank/N over
    the layers they are ranked in (needs ``min(min_layers_for_global, n_layers)`` of them; ties by mean rank/N).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    cols, score, ntr, nsig, acat, topk, fish = {}, {}, {}, {}, {}, {}, {}
    for layer, oracle in layer_oracles.items():
        d = table[(table.molecular_layer == layer) & (table.model == oracle)].copy()
        d["group"] = d.cell_type_key.map(group_map)
        d["abs_raw"] = d.raw_score.abs()
        d = d[d.group.notna()]
        d = d.sort_values("percentile", ascending=False).drop_duplicates(["group", "track_id"])
        g = d.groupby("group")["percentile"]
        tk = g.apply(lambda v: np.sort(v.values)[::-1][:k].mean())
        fl = g.apply(lambda v: fisher_logp(v.values))
        ac = g.apply(lambda v: np.log(acat_p(v.values)))
        best = pd.DataFrame({"score": {"fisher": fl, "acat": ac}[method],
                             "tie": d.groupby("group")["abs_raw"].max().fillna(0.0), "n": g.size(),
                             "nsig": g.apply(lambda v: (v >= thr).sum())})
        best = best[best.n >= min_tracks].sort_index()
        best = best.sort_values(["score", "tie"], ascending=[True, False], kind="stable")
        best["rank"] = np.arange(1, len(best) + 1)
        cols[layer], score[layer], ntr[layer], nsig[layer] = best["rank"], best["score"], best["n"], best["nsig"]
        acat[layer] = g.apply(lambda v: acat_p(v.values)).reindex(best.index)
        topk[layer], fish[layer] = tk.reindex(best.index), fl.reindex(best.index)
        logger.info("%-13s %-12s unique tracks=%5d groups=%4d significant=%d", layer, oracle, len(d), len(best),
                    int(best.nsig.sum()))
    ix = list(layer_oracles)
    ranks, scores, ntracks, nsigs = (pd.DataFrame(x).reindex(columns=ix)
                                     for x in (cols, score, ntr, nsig))
    extra = {"acat_p": pd.DataFrame(acat).reindex(columns=ix), "topk_mean": pd.DataFrame(topk).reindex(columns=ix),
             "fisher_logp": pd.DataFrame(fish).reindex(columns=ix)}
    for x in (ranks, scores, ntracks, nsigs):
        x.index.name = "group"
    frac = ranks / ranks.max()
    ok = ranks.notna().sum(axis=1) >= min(min_layers_for_global, len(ix))
    key = pd.DataFrame({"med": frac[ok].median(axis=1), "mean": frac[ok].mean(axis=1)}).sort_values(["med", "mean"])
    ranks["Global"] = pd.Series(np.arange(1, len(key) + 1), index=key.index)
    return ranks, scores, ntracks, nsigs, extra


def method_comparison(extra: dict, method: str, layers: Iterable[str]) -> pd.DataFrame:
    """Spearman between the chosen ranking and the alternatives, per layer (over groups present in both)."""
    rows = []
    for layer in layers:
        strength = {"fisher": -extra["fisher_logp"][layer], "acat": -extra["acat_p"][layer],
                    "top3": extra["topk_mean"][layer]}
        for name, v in strength.items():
            if name == method:
                continue
            ok = strength[method].notna() & v.notna()
            rows.append(dict(layer=layer, vs=name, n_groups=int(ok.sum()),
                             spearman=float(strength[method][ok].rank().corr(v[ok].rank())) if ok.sum() > 2
                             else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_INK, _MUTED, _NA_FILL = "#1b2430", "#6b7280", "#eef0f3"
_TIERS = [(0.05, "#8a3a00", "top 5%"), (0.10, "#d2691e", "5–10%"), (0.25, "#f39c4a", "10–25%"),
          (0.50, "#fbd2a3", "25–50%"), (1.01, "#fdf1e3", "bottom half")]
_WRAP_OVER = 27   # row labels longer than this go on two lines so they stay inside the left margin


def _wrap2(label: str, minlen: int = _WRAP_OVER) -> str:
    """Two-line version of a long label, split at the space nearest the middle."""
    if len(label) <= minlen or " " not in label:
        return label
    if " (" in label:
        return label.replace(" (", "\n(", 1)
    cuts = [i for i, ch in enumerate(label) if ch == " "]
    i = min(cuts, key=lambda c: abs(c - len(label) / 2))
    return label[:i] + "\n" + label[i + 1:]


def _tier_colour(frac: float):
    from matplotlib.colors import to_rgb
    for hi, col, _ in _TIERS:
        if frac <= hi:
            return to_rgb(col)


def _text_colour(rgb) -> str:
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb[:3]]
    return _INK if 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] > 0.30 else "white"


def draw_rank_table(ranks: pd.DataFrame, ntracks: pd.DataFrame, nsigs: pd.DataFrame, layer_oracles: dict, *,
                    level: str, method: str, thr: float, top: int, variant_label: str, out_stem: str | Path,
                    formats: Sequence[str] = ("png", "pdf"),
                    layer_notes: Optional[dict] = None) -> list[str]:
    """Discrete-tier rank table: big number = rank in the layer, corner = n_sig/n, colour = rank/N tier.

    Rows are the top ``top`` groups by Global order; one column per layer plus Global.
    """
    import matplotlib as mpl
    mpl.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    singular, plural = _UNITS[level]
    cols = list(layer_oracles) + ["Global"]
    n = ranks[cols].max()
    order = ranks["Global"].dropna().sort_values().head(top).index
    if len(order) == 0:
        raise ValueError("no group is ranked in enough layers to get a Global order; lower min_tracks or supply "
                         "more layers")
    R, F = ranks.loc[order, cols], (ranks[cols] / n).loc[order]
    nr, nc = len(order), len(cols)
    S = 1.45                                   # scale of the published figure
    W, rowh, left, top_in, bot_in = 10.0, 0.40, 0.27, 1.95, 0.95
    H = rowh * nr + top_in + bot_in

    rc = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
          "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 7}
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([left, bot_in / H, 0.985 - left, rowh * nr / H])
        ax.set_xlim(-0.5, nc - 0.5 + 0.25)
        ax.set_ylim(nr - 0.5, -0.5)
        ax.axis("off")
        for i, g in enumerate(order):
            for j, c in enumerate(cols):
                x = j + (0.25 if c == "Global" else 0)
                v, f = R.loc[g, c], F.loc[g, c]
                if np.isnan(v):
                    ax.add_patch(Rectangle((x - .5, i - .5), 1, 1, fc=_NA_FILL, ec="#d5d9df", lw=0, hatch="////"))
                    ax.add_patch(Rectangle((x - .5, i - .5), 1, 1, fc=_NA_FILL, ec="none", alpha=0.75))
                    ax.text(x, i, "NA", ha="center", va="center", fontsize=6 * S, color=_MUTED)
                    continue
                col = _tier_colour(f)
                ax.add_patch(Rectangle((x - .5, i - .5), 1, 1, fc=col, ec="white", lw=1.2))
                ax.text(x, i - 0.04, f"{int(v)}", ha="center", va="center", fontsize=7.5 * S, color=_text_colour(col))
                if c != "Global":
                    ax.text(x + 0.47, i + 0.45, f"{int(nsigs.loc[g, c])}/{int(ntracks.loc[g, c])}", ha="right",
                            va="bottom", fontsize=5 * S, color=_text_colour(col))
        for j, c in enumerate(cols):
            x = j + (0.25 if c == "Global" else 0)
            note = f"{layer_notes[c]} · " if layer_notes and c in layer_notes else ""
            sub = "median rank/N\nover layers" if c == "Global" else f"{layer_oracles[c]}\n{note}{int(n[c])} {plural}"
            ax.text(x, -0.58, sub, ha="center", va="bottom", fontsize=8, color=_MUTED, linespacing=1.15)
            ax.text(x, -1.32, c, ha="center", va="bottom", fontsize=7.5 * 1.2, fontweight="bold", color=_INK,
                    linespacing=1.0)
        for i, g in enumerate(order):
            ax.text(-0.56, i, _wrap2(str(g)), ha="right", va="center", fontsize=7 * S, color=_INK, linespacing=1.05)
        lg = fig.add_axes([left, 0.03, 0.71, 0.06])
        lg.axis("off")
        lg.legend(handles=[Patch(fc=col, ec="none", label=lab) for _, col, lab in _TIERS], ncol=5, frameon=False,
                  loc="center", fontsize=6 * S, handlelength=1.2, columnspacing=1.2,
                  title="rank / N within layer", title_fontsize=6 * S)
        score_txt = {"fisher": "Fisher's combined p over all tracks (p = 1 − |percentile|)",
                     "acat": "ACAT (Cauchy) combined p over all tracks (p = 1 − |percentile|)"}[method]
        fig.text(0.02, 1 - 0.12 / H, f"{variant_label} — {singular} rank within each molecular layer",
                 fontsize=8.5 * S, fontweight="bold", color=_INK, va="top")
        fig.text(0.02, 1 - 0.50 / H,
                 f"Top {nr} {plural} by Global order; large number = rank (1 = strongest);\n"
                 f"corner = tracks with |percentile| ≥ {thr:g} / all tracks; score = {score_txt}",
                 fontsize=5.8 * S, color=_MUTED, va="top", linespacing=1.3)
        fig.text(0.98, 0.012, "Discrete rank tiers (orange)", fontsize=5.5 * S, color=_MUTED, ha="right")
        paths = []
        for fmt in formats:
            p = f"{out_stem}.{fmt}"
            fig.savefig(p, dpi=300 if fmt == "png" else None)
            paths.append(p)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def rank_variant_tissues(
    track_tables: str | Path | Sequence[str | Path] | pd.DataFrame,
    output_dir: str | Path,
    *,
    method: str = "fisher",
    group_level: str = "tissue",
    top_n: int = 15,
    min_tracks: int = 1,
    layers: Optional[Sequence[str]] = None,
    layer_oracles: Optional[dict] = None,
    annotation: str | Path | pd.DataFrame | None = None,
    threshold: float = 0.95,
    variant_label: Optional[str] = None,
    rna_gene: Optional[str] = None,
    formats: Sequence[str] = ("png", "pdf"),
) -> dict:
    """Rank tissues or body systems within each molecular layer and draw the rank table.

    Args:
        track_tables: ``*_track_scores.csv`` file(s) written by discovery (one per oracle), or a DataFrame
            with the same columns.
        output_dir: Where the figure and CSVs are written (created if missing).
        method: ``"fisher"`` (breadth: is the group broadly affected?) or ``"acat"`` (peak: does it contain a
            strong hit?). See the module docstring.
        group_level: ``"tissue"`` (organ / cell-lineage, ~60 groups) or ``"system"`` (body system, ~19 groups).
        top_n: Rows of the figure (top groups by Global order). The CSVs always hold every group.
        min_tracks: Smallest number of tracks a group needs in a layer to be ranked there (default 1 = all).
        layers: Layers to show, default ``DNase, ATAC, ChIP-TF, ChIP-histone, RNA-seq, CAGE`` that have data.
        layer_oracles: ``{layer: oracle}`` overrides; see :func:`resolve_layer_oracles` for the defaults.
        annotation: Path or DataFrame mapping ``cell_type_key`` to ``tissue_organ`` / ``tissue_system``.
            Default is the copy shipped with chorus.
        threshold: ``|percentile|`` cutoff for the ``n_sig/n`` corner label.
        variant_label: Figure title prefix; default read from the track table header.
        rna_gene: Gene whose RNA-seq effects fill the RNA-seq column. Needed when the input holds the
            ``*_rna_gene_percentiles.csv`` of several genes (pass it in ``track_tables`` next to the
            track tables); with one gene it is used automatically, with several and no choice the
            RNA-seq column is left out and ``notes`` lists the genes.

    Returns:
        Dict with ``files`` (paths), ``layers`` (layer -> oracle, n groups ranked), ``top`` (the figure's rows),
        ``unmapped_cell_types`` and ``notes``.
    """
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if group_level not in GROUP_LEVELS:
        raise ValueError(f"group_level must be 'tissue' or 'system', got {group_level!r}")
    level = GROUP_LEVELS[group_level]

    rna_paths: list = []
    if isinstance(track_tables, pd.DataFrame):
        table, label = _clean_table(track_tables, "track_tables"), None
    else:
        paths = [track_tables] if isinstance(track_tables, (str, Path)) else list(track_tables)
        rna_paths = [q for q in paths if _is_rna_matrix(q)]
        plain = [q for q in paths if q not in rna_paths]
        table, label = load_track_tables(plain) if plain else (pd.DataFrame(columns=TRACK_TABLE_COLUMNS), None)
    notes: list[str] = []
    rna_selected, rna_available = None, []
    if rna_paths:
        table, label, rna_selected, rna_available = _add_rna_gene(table, label, rna_paths, rna_gene, notes)
    variant_label = variant_label or label or "Variant"

    ann = annotation if isinstance(annotation, pd.DataFrame) else load_annotation(annotation)
    group_map = ann.drop_duplicates("cell_type_key").set_index("cell_type_key")[level]
    group_map = group_map[group_map.notna() & (group_map.astype(str).str.strip() != "")]

    layers = list(layers) if layers else list(DEFAULT_LAYERS)   # absent layers are dropped with a note, not silently
    chosen, layer_notes = resolve_layer_oracles(table, layers, layer_oracles)
    notes += layer_notes
    if not chosen:
        raise ValueError(f"none of the layers {layers} has tracks in the input (found "
                         f"{sorted(table.molecular_layer.unique())})")

    used = table[[(m, l) in {(o, ly) for ly, o in chosen.items()} for m, l in zip(table.model, table.molecular_layer)]]
    unmapped = (used[~used.cell_type_key.isin(group_map.index)].groupby("cell_type_key")
                .agg(cell_type=("cell_type", "first"), n_tracks=("track_id", "size"),
                     layers=("molecular_layer", lambda s: ";".join(sorted(set(s))))).reset_index())
    if len(unmapped):
        notes.append(f"{len(unmapped)} cell types ({int(unmapped.n_tracks.sum())} tracks) are not in the "
                     f"annotation and were left out of the ranking; see unmapped_cell_types.csv")

    ranks, scores, ntracks, nsigs, extra = rank_groups(table, group_map, chosen, method, threshold,
                                                       min_tracks=min_tracks)
    ranks = ranks.sort_values(["Global"] + list(chosen), na_position="last")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}

    def save(name, df, **kw):
        p = out / name
        df.to_csv(p, na_rep="NA", **kw)
        files[p.stem] = str(p)

    save("layer_rank_table.csv", ranks, float_format="%.0f")
    save("layer_rank_over_N_table.csv", ranks / ranks.max(), float_format="%.4f")
    sc = scores.reindex(ranks.index)
    save("layer_score_neglog10p_table.csv", -sc / np.log(10), float_format="%.3f")
    save("layer_score_table.csv", np.exp(sc), float_format="%.3e")
    save("layer_ntracks_table.csv", ntracks.reindex(ranks.index), float_format="%.0f")
    save("layer_nsig_table.csv", nsigs.reindex(ranks.index), float_format="%.0f")
    save("method_comparison.csv", method_comparison(extra, method, chosen), index=False, float_format="%.3f")
    if len(unmapped):
        save("unmapped_cell_types.csv", unmapped, index=False)

    files["figure"] = draw_rank_table(ranks, ntracks, nsigs, chosen, level=level, method=method, thr=threshold,
                                      top=top_n, variant_label=variant_label, out_stem=out / "layer_rank_table",
                                      formats=formats,
                                      layer_notes={RNA_LAYER: rna_selected} if rna_selected else None)
    top_groups = ranks["Global"].dropna().sort_values().head(top_n).index
    summary = {
        "method": method, "group_level": level, "top_n": top_n, "min_tracks": min_tracks, "threshold": threshold,
        "variant_label": variant_label,
        "layers": {layer: {"oracle": o, "n_groups_ranked": int(ranks[layer].max()) if ranks[layer].notna().any() else 0}
                   for layer, o in chosen.items()},
        "n_tracks_used": int(len(used)), "n_unmapped_cell_types": int(len(unmapped)), "notes": notes,
        "rna_gene": rna_selected, "rna_genes_available": rna_available,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    files["run_summary"] = str(out / "run_summary.json")
    return {
        **summary, "files": files,
        "top": [{"group": g, "global_rank": int(ranks.loc[g, "Global"]),
                 **{layer: (None if np.isnan(ranks.loc[g, layer]) else int(ranks.loc[g, layer])) for layer in chosen}}
                for g in top_groups],
        "unmapped_cell_types": unmapped.cell_type.tolist(),
    }
