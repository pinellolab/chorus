"""Every percentile a committed example reports must be the one the pinned null gives.

A percentile is a rank against a per-track background null, so a committed one is a function
of two things that are both on disk: the raw effect and reference value in the example's JSON,
and the null in ``<data-dir>/backgrounds``. Checking it needs no model, no genome and no GPU,
and until now no test did.

``variant_analysis/SORT1_chrombpnet`` (regenerated in 3d0b6b1) reports DNASE:HepG2, +1.376, as
effect "≥99th (1.18× null max)" with activity 0.885. The pinned ``chrombpnet_pertrack.npz``
gives 0.9995, inside the null (whose largest effect is 1.606), and 0.906. The two earlier
published ChromBPNet nulls give the same, and v0.7.5 committed exactly those numbers. The
conservation walkthrough added in #246 carried the same clamp and built its headline claim on
it ("exceeds every one of the ~18,000 background variants"), and was pulled from v0.8.0 for it
(release audit P1-1). The JSON records no background revision, so nothing else could notice.

Per scored track this compares ``quantile_score`` (the effect percentile), ``effect_exceedance``
(how far past the null's maximum, set when the percentile is clamped) and
``ref_signal_percentile`` (the activity percentile). Every shipped grid is 10,000 points wide,
so a percentile moves in steps of 1e-4 (2e-4 when signed). It is compared to 4 decimal places,
so a change of even one rank fails.

Covered: every ``example_output.json`` shape (single-oracle variant reports, including region
swap and integration; discovery's per-cell-type reports; batch ``track_scores``; causal
``per_layer_top_track``; the multi-oracle consensus), plus the multi-oracle directory's
``*_variant_report.json``, which carry the activity percentiles its consensus omits. Not
covered: values derived from these (batch ``max_quantile``, causal ``composite``), their
``.md``/``.tsv``/``.html`` renderings, and rows that carry no percentile. Cherimoya is ranked
against the fold-0 null, because the JSON records no fold and fold 0 is the default.

Skipped where the backgrounds are not downloaded, as in CI. It trusts that the local files are
the pinned revision (``test_the_pinned_revision_matches_the_artefacts_on_this_machine`` compares
their sizes against it), and a failure prints the sha256 of each file it ranked against.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from chorus.analysis.normalization import (
    _HF_REVISION,
    PerTrackNormalizer,
    normalization_key,
)
from chorus.analysis.scorers import LAYER_CONFIGS

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"

OUTPUTS = sorted(EXAMPLES.rglob("example_output.json")) + sorted(
    EXAMPLES.rglob("*_variant_report.json"))

_PERCENTILE_KEYS = ("quantile_score", "effect_exceedance", "ref_signal_percentile")

#: Half of one rank on a 10,000-point grid: "equal to 4 decimal places".
_TOL = 0.5e-4

#: CausalResult.to_dict rounds raw_score (and quantile_score) to 4 dp.
_CAUSAL_RAW_DECIMALS = 4


def _ids(path: Path) -> str:
    return str(path.relative_to(EXAMPLES))


def _scored_rows(doc: dict):
    """``(where, oracle, layer, row, raw_decimals)`` for every scored track in one output.

    One branch per shape, because each keeps the oracle and the layer somewhere different.
    ``raw_decimals`` is how far ``raw_score`` was rounded before it was written, or None for
    full precision.
    """
    def variant_report(rep, oracle, prefix=""):
        for allele, payload in (rep.get("alleles") or {}).items():
            for row in (payload or {}).get("all_scores") or []:
                yield f"{prefix}{allele}", oracle, row.get("layer"), row, None

    if "alleles" in doc:  # variant report, region swap, integration
        yield from variant_report(doc, doc.get("oracle"))
    for cell_type, rep in (doc.get("reports") or {}).items():  # discovery
        yield from variant_report(rep, rep.get("oracle") or doc.get("oracle"),
                                  prefix=f"reports[{cell_type}].")
    if isinstance(doc.get("scores"), list):  # batch: the oracle is only in analysis_request
        oracle = (doc.get("analysis_request") or {}).get("oracle_name")
        for s in doc["scores"]:
            for row in (s.get("track_scores") or {}).values():
                yield f"scores[{s.get('variant_id')}]", oracle, row.get("layer"), row, None
    for r in doc.get("rankings") or []:  # causal: the layer is the key
        for layer, row in (r.get("per_layer_top_track") or {}).items():
            yield (f"rankings[{r.get('variant_id')}]", doc.get("oracle"), layer, row,
                   _CAUSAL_RAW_DECIMALS)
    for c in doc.get("consensus") or []:  # multi-oracle: the oracle is the key
        for oracle, row in (c.get("oracles") or {}).items():
            if row:
                yield "consensus", oracle, c.get("layer"), row, None


def _count_percentile_rows(node) -> int:
    """Dicts anywhere in *node* that carry a percentile, not counting ``scores_by_layer``.

    ``scores_by_layer`` repeats every ``all_scores`` row grouped by layer, so it is skipped
    rather than counted twice.
    """
    if isinstance(node, list):
        return sum(_count_percentile_rows(v) for v in node)
    if not isinstance(node, dict):
        return 0
    return int(any(k in node for k in _PERCENTILE_KEYS)) + sum(
        _count_percentile_rows(v) for k, v in node.items() if k != "scores_by_layer")


def _effect_percentile_range(norm, key, track, value, half_width, signed):
    """The effect percentiles any raw score within ``±half_width`` of *value* could give.

    The causal report rounds ``raw_score`` to 4 dp, and that alone moves the rank: recomputed
    from the rounded value, 221 of CDYL_rs9504151's 274 committed percentiles disagree, by up
    to 0.006 on an RNA-seq row, and every one lies inside this range. The range is read off
    the CDF row itself, because inside a tied run the rank is a hashed draw and only the run's
    ends bound it.
    """
    entry = norm._ensure_loaded(key)
    idx = None if entry is None else norm._resolve_row(track, entry)
    if idx is None or not norm._has_samples(entry, "effect_cdfs", idx):
        return None
    row = np.asarray(entry["effect_cdfs"][idx])
    denom = norm._get_denominator(entry, "effect_cdfs", idx)
    lo = value - half_width if signed else max(value - half_width, 0.0)
    a = min(int(np.searchsorted(row, lo, side="left")) / denom, 1.0)
    b = min(int(np.searchsorted(row, value + half_width, side="right")) / denom, 1.0)
    return (2.0 * a - 1.0, 2.0 * b - 1.0) if signed else (a, b)


def _mismatches(norm, where, oracle, layer, row, raw_decimals) -> tuple[int, list[str]]:
    """(values compared, one line per committed value the pinned null does not give)."""
    key = normalization_key(oracle)
    cfg = LAYER_CONFIGS.get(layer)
    signed = cfg.signed if cfg else True  # the default _apply_normalization uses
    track = row["assay_id"]
    raw, ref = row.get("raw_score"), row.get("ref_value")
    label = f"{where} {oracle}:{track}" + (
        f" [{row['region_label']}]" if row.get("region_label") else "")

    def same(got, want):
        return got is not None and abs(got - want) <= _TOL

    n, bad = 0, []
    q = row.get("quantile_score")
    if q is not None:
        n += 1
        if raw is None:
            bad.append(f"{label}: effect percentile {q} has no raw_score to rank")
        elif raw_decimals is None:
            got = norm.effect_percentile(key, track, raw if signed else abs(raw), signed=signed)
            if not same(got, q):
                bad.append(f"{label}: effect percentile committed {q}, pinned null {got} "
                           f"(raw_score {raw})")
        else:
            span = _effect_percentile_range(norm, key, track, raw if signed else abs(raw),
                                            0.5 * 10 ** -raw_decimals, signed)
            if span is None or not span[0] - _TOL <= q <= span[1] + _TOL:
                bad.append(f"{label}: effect percentile committed {q}, pinned null "
                           f"{span} for raw_score {raw} ± its rounding")

    exc = row.get("effect_exceedance")
    if exc is not None and raw is not None and raw_decimals is None:
        n += 1
        got = norm.effect_exceedance(key, track, raw if signed else abs(raw), signed=signed)
        if not same(got, exc):
            bad.append(f"{label}: effect_exceedance committed {exc}, pinned null {got} "
                       f"(null support {norm.effect_null_support(key, track)})")

    act = row.get("ref_signal_percentile")
    if act is not None and ref is not None:
        n += 1
        got = norm.activity_percentile(key, track, ref)
        if not same(got, act):
            bad.append(f"{label}: activity percentile committed {act}, pinned null {got} "
                       f"(ref_value {ref})")
    return n, bad


def _npz_on_disk(norm, key) -> Path:
    """The file a key is ranked against; alphagenome_pt reads alphagenome's."""
    path = norm.npz_path(key)
    return path if path.exists() else norm.npz_path(norm._CDF_ALIASES.get(key, key))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def norm():
    return PerTrackNormalizer()


def test_there_are_committed_outputs_to_check():
    """Guard the guard: a bad glob silently turns every check below into a no-op."""
    n = sum(p.name == "example_output.json" for p in OUTPUTS)
    assert n >= 13, f"found only {n} example_output.json under {EXAMPLES}"


@pytest.mark.parametrize("path", OUTPUTS, ids=_ids)
def test_this_file_reads_every_committed_percentile(path: Path):
    """Hermetic, so it runs in CI where the comparison below skips.

    An output shape `_scored_rows` does not know would otherwise be checked by nothing while
    the comparison passed, so every dict that carries a percentile must be one it yields, with
    the oracle, layer and track id the comparison needs.
    """
    doc = json.loads(path.read_text())
    rows = [r for r in _scored_rows(doc) if any(k in r[3] for k in _PERCENTILE_KEYS)]
    expected = _count_percentile_rows(doc)
    assert rows and len(rows) == expected, (
        f"{_ids(path)} carries {expected} rows with a percentile but _scored_rows reads "
        f"{len(rows)}; a new output shape needs a branch there")
    incomplete = [f"{where}: oracle={oracle!r} layer={layer!r} assay_id={row.get('assay_id')!r}"
                  for where, oracle, layer, row, _ in rows
                  if not (isinstance(oracle, str) and isinstance(layer, str)
                          and isinstance(row.get("assay_id"), str))]
    assert not incomplete, "rows the comparison cannot rank:\n  " + "\n  ".join(incomplete[:10])


@pytest.mark.parametrize("path", OUTPUTS, ids=_ids)
def test_committed_percentiles_match_the_pinned_null(path: Path, norm):
    """Recompute each committed percentile from its own raw values and the local null."""
    rows = list(_scored_rows(json.loads(path.read_text())))
    keys = sorted({normalization_key(oracle) for _, oracle, *_ in rows}, key=str)
    missing = [k for k in keys if norm.n_tracks(k) == 0]
    if missing:
        pytest.skip(f"no downloaded background for {missing} under {norm.cache_dir}")

    n_checked, bad = 0, []
    for where, oracle, layer, row, raw_decimals in rows:
        n, lines = _mismatches(norm, where, oracle, layer, row, raw_decimals)
        n_checked += n
        bad += lines
    assert n_checked, f"{_ids(path)}: no committed percentile was compared"
    if bad:
        nulls = "\n".join(f"  {k}: {p} (sha256 {_sha256(p)[:12]})"
                          for k, p in ((k, _npz_on_disk(norm, k)) for k in keys))
        pytest.fail(
            f"{_ids(path)}: {len(bad)} of {n_checked} committed values are not what the pinned "
            f"null gives:\n  " + "\n  ".join(bad[:25])
            + (f"\n  ... and {len(bad) - 25} more" if len(bad) > 25 else "")
            + f"\nRanked against (this chorus pins {_HF_REVISION}):\n{nulls}\n"
            "Regenerate the example against these backgrounds, or, if they are not that "
            "revision, refetch them first.",
            pytrace=False)
