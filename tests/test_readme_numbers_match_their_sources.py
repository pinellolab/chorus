"""The README's published numbers must come from, and agree with, their sources.

Five wrong numbers shipped in one day's work and none of them was caught by a test, because nothing
checked a documented figure against the thing it measures. Each has its own guard below.

* **The single-oracle install size double-counted the weights.** The itemisation charged 1.87 GiB for
  Enformer weights because the measuring script globbed `*enformer*` over the HF cache and this host
  happens to hold **two** mirrors — `lucapinello/chorus-enformer` (the one
  `chorus/oracles/enformer.py:121` actually loads) and an unrelated `EleutherAI/enformer-official-rough`.
  The README's own per-asset table said ~960 MB two hundred lines further down, so the file
  contradicted itself.
* **The itemisation did not add up.** Components were rounded to 1 dp while the total came from the
  unrounded values, so `2.4 + 5.9 + 3.1 + 1.9 + 0.5` displayed as summing to 13.7 when it sums to 13.8.
  A reader who checks the arithmetic finds it wrong.
* **The activity-null floor was a per-oracle figure mistaken for the global minimum.** The README said
  ~29,000 positions; 29,002 is LegNet's value, and AlphaGenome's own floor is **19,504**.
* **The strata table asserted a uniformity the artefacts contradict.** One row claimed all eight
  oracles share one mixture. `build_config.reference_sets.activity_derivation` in the shipped NPZs
  shows LegNet drops `gene_body`, and ChromBPNet and Cherimoya drop `gene_body` *and* add a
  5,000-position `dhs` stratum.
* **The worked example's activity percentile was wrong in one of the two places it appears** (0.81 vs a
  measured 0.9625).

The first two guards need nothing but the README and are in the fast suite. The rest need the shipped
artefacts and are integration-marked.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"


# ── the install itemisation ──────────────────────────────────────────────────────

def _itemisation() -> tuple[list[float], float, str]:
    """(components, stated total, raw text) from the `measured for enformer: …` parenthetical."""
    text = README.read_text()
    m = re.search(r"measured for `enformer`:([^)]*?)=\s*([\d.]+)\s*GiB", text)
    assert m, "the README no longer itemises the single-oracle install size"
    parts = [float(x) for x in re.findall(r"([\d.]+)\s*GiB", m.group(1))]
    return parts, float(m.group(2)), m.group(0)


def test_the_itemised_install_size_adds_up():
    """The exact defect: components displayed at a precision where the sum does not close."""
    parts, total, raw = _itemisation()
    assert len(parts) >= 4, f"expected several components, parsed {parts} from {raw!r}"
    assert abs(sum(parts) - total) < 0.005, (
        f"the itemisation lists {' + '.join(str(p) for p in parts)} = {sum(parts):.2f} GiB but states "
        f"{total} GiB. Anyone verifying a number we publish should be able to add it up; if the "
        f"components are rounded, round the total from the same rounded values."
    )


def test_the_weights_component_agrees_with_the_per_asset_table():
    """A figure repeated in two places in one file must not contradict itself.

    The itemisation said 1.87 GiB for Enformer weights while the per-asset table said ~960 MB. The
    cause was double-counting two mirrors of the same model; the symptom was visible without any
    measurement at all, just by reading both numbers.
    """
    text = README.read_text()
    parts, _, raw = _itemisation()

    m = re.search(r"\|\s*Enformer weights\s*\|\s*~?([\d.]+)\s*(MB|GB|GiB|MiB)\s*\|", text)
    assert m, "the per-asset disk table no longer lists Enformer weights"
    value, unit = float(m.group(1)), m.group(2)
    as_gib = {"MB": value * 1e6 / 1024**3, "MiB": value / 1024,
              "GB": value * 1e9 / 1024**3, "GiB": value}[unit]

    # the weights component is the one closest to the table's figure; assert it is *close*, not equal,
    # since the table rounds and the itemisation measures
    weights = min(parts, key=lambda p: abs(p - as_gib))
    assert abs(weights - as_gib) < 0.15, (
        f"the install itemisation charges {weights} GiB for weights while the per-asset table says "
        f"{m.group(1)} {unit} (= {as_gib:.2f} GiB). One of them is wrong — the first version of this "
        f"number summed two different HF mirrors of the same model.\nitemisation: {raw!r}"
    )


# ── the artefact-derived numbers ─────────────────────────────────────────────────

def _npz_dir() -> Path | None:
    from chorus.analysis.normalization import CHORUS_BACKGROUNDS_DIR

    d = Path(CHORUS_BACKGROUNDS_DIR)
    return d if d.is_dir() and any(d.glob("*_pertrack.npz")) else None


def _shipped_npzs() -> list[Path]:
    d = _npz_dir()
    if d is None:
        return []
    return [f for f in sorted(d.glob("*_pertrack.npz"))
            if not any(s in f.name for s in (".pre", ".prepad"))]


@pytest.mark.integration
@pytest.mark.parametrize("kind,key,pattern", [
    ("effect", "effect_retained", r"reference population \(~([\d,]+)–([\d,]+) variants\)"),
    ("activity", "summary_retained", r"and ~([\d,]+)–([\d,]+) genome-wide positions"),
])
def test_the_documented_null_sizes_bracket_the_shipped_artefacts(kind, key, pattern):
    """The stated range must actually contain every shipped oracle's value.

    The activity floor was ~29,000 while AlphaGenome ships 19,504 — a per-oracle number mistaken for
    the global minimum. A range that excludes a shipped oracle misdescribes the method.
    """
    import numpy as np

    files = _shipped_npzs()
    if not files:
        pytest.skip("no shipped *_pertrack.npz available on this host")

    lo_obs, hi_obs = None, None
    for f in files:
        z = np.load(f, allow_pickle=True)
        if key not in z.files:
            continue
        a = z[key]
        if not a.size:
            continue
        lo, hi = int(a.min()), int(a.max())
        lo_obs = lo if lo_obs is None else min(lo_obs, lo)
        hi_obs = hi if hi_obs is None else max(hi_obs, hi)
    if lo_obs is None:
        pytest.skip(f"no {key} present in the shipped artefacts")

    m = re.search(pattern, README.read_text())
    assert m, f"the README no longer states a {kind} reference-population range"
    lo_doc = int(m.group(1).replace(",", ""))
    hi_doc = int(m.group(2).replace(",", ""))

    # 5 % slack: the README rounds ("~17,800", "~19,500") and should not need editing for noise
    assert lo_doc <= lo_obs * 1.05, (
        f"README's {kind} floor is {lo_doc:,} but a shipped oracle goes down to {lo_obs:,}. A stated "
        f"range that excludes a shipped artefact misdescribes the reference population."
    )
    assert hi_doc >= hi_obs * 0.95, (
        f"README's {kind} ceiling is {hi_doc:,} but a shipped oracle reaches {hi_obs:,}"
    )


@pytest.mark.integration
def test_every_oracles_strata_deviation_is_documented():
    """The strata table must match `build_config.reference_sets.activity_derivation`.

    One row once claimed all eight oracles share one mixture. Three do not, and the deviations are
    recorded in the artefacts themselves — so this is checkable rather than a matter of memory.
    """
    import numpy as np

    files = _shipped_npzs()
    if not files:
        pytest.skip("no shipped *_pertrack.npz available on this host")

    text = README.read_text()
    i = text.index("| oracle | effect reference population |")
    table = text[i:text.index("\n\n", i)]

    undocumented = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if "build_config" not in z.files:
            continue
        try:
            cfg = json.loads(str(z["build_config"][0]))
        except (ValueError, TypeError):
            continue
        deriv = (cfg.get("reference_sets") or {}).get("activity_derivation") or {}
        dropped = list(deriv.get("drop_strata") or [])
        added = list((deriv.get("add_strata") or {}).keys())
        if not dropped and not added:
            continue  # base mixture, covered by the first row

        name = f.name.replace("_pertrack.npz", "").replace("_ensemble", "")
        for stratum in dropped:
            if stratum not in table:
                undocumented.append(f"{name} drops `{stratum}` and the table does not say so")
        for stratum in added:
            if stratum not in table:
                undocumented.append(f"{name} adds `{stratum}` and the table does not say so")

    assert not undocumented, (
        "the README strata table does not describe deviations that the shipped artefacts record:\n  "
        + "\n  ".join(sorted(set(undocumented)))
        + "\nThis table is the methods description for every percentile chorus publishes."
    )


@pytest.mark.integration
def test_the_worked_example_percentiles_agree_wherever_they_appear():
    """`+0.45 log₂FC` is quoted in the intro and again in the backgrounds section.

    They disagreed: 0.81 in one place against a measured 0.9625 in the other.
    """
    from chorus.analysis.normalization import get_pertrack_normalizer

    norm = get_pertrack_normalizer("alphagenome")
    if norm is None:
        pytest.skip("alphagenome backgrounds not available on this host")

    track = "DNASE/EFO:0001187 DNase-seq/."
    act = norm.activity_percentile("alphagenome", track, 512.0)
    if act is None:
        pytest.skip(f"track {track!r} carries no activity CDF on this host")

    quoted = {m for m in re.findall(r"`0\.\d\d effect %ile, (0\.\d\d) activity %ile`",
                                   README.read_text())}
    assert quoted, "the intro no longer quotes the worked example's percentile pair"
    for q in quoted:
        assert abs(float(q) - act) < 0.01, (
            f"the README quotes {q} as the worked example's activity percentile; the shipped CDFs "
            f"give {act:.4f}. The same example is quoted in two sections and they must agree with "
            f"each other and with the artefact."
        )


# ── the disk-usage breakdown ─────────────────────────────────────────────────────

# `tests/test_disk_claims_add_up.py` already sums this table against its stated total, so that is not
# repeated here. Summing alone could not catch what drifted, though: the backgrounds bucket understated
# reality by ~1.5 GB *and* the total was consistent with it, so the table added up while being wrong.
# What follows ties the figures to the artefacts and to the other place the README states the same
# number. The drift: the 0.7.4 Sei rebuild took Sei's NPZ from 40 tracks (~3 MB) to 21,947
# (1.5 GB), so the "Per-oracle CDF backgrounds" bucket understated reality by ~1.5 GB and the stated
# total stayed at ~85 GB. Nothing added the column up, so nothing noticed.

def _disk_table_rows() -> list[tuple[str, float]]:
    """(label, size in decimal GB) for each bucket row, excluding the total."""
    text = README.read_text()
    start = text.index("#### Disk usage breakdown")
    end = text.index("#### Where chorus puts large files")
    rows = []
    for label, num, unit in re.findall(
        r"^\|\s*(?!\*\*Total)([^|]+?)\s*\|\s*~?([\d.]+)\s*(GB|MB)\s*\|", text[start:end], re.M
    ):
        rows.append((label.strip(), float(num) / (1000.0 if unit == "MB" else 1.0)))
    return rows


def _disk_table_total() -> float:
    text = README.read_text()
    m = re.search(r"\|\s*\*\*Total default\*\*\s*\|\s*\*\*~?([\d.]+)\s*GB\*\*\s*\|", text)
    assert m, "the disk-usage table no longer states a total"
    return float(m.group(1))


def test_the_tldr_install_size_agrees_with_the_disk_table():
    """The same quantity is stated twice — in the TLDR (GiB) and the table (GB, loosely)."""
    text = README.read_text()
    m = re.search(r"The install itself\s*\n?\s*is ~([\d.]+) GiB", text)
    assert m, "the TLDR no longer states the install size"
    tldr, table = float(m.group(1)), _disk_table_total()
    assert abs(tldr - table) <= 1.0, (
        f"the TLDR says ~{tldr} GiB and the disk table says ~{table} GB for the same install. "
        f"They drifted apart once already when the Sei background grew; keep them together."
    )


@pytest.mark.integration
def test_the_backgrounds_bucket_matches_the_shipped_npzs():
    """The bucket that drifted, tied to the artefacts it describes."""
    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR  # noqa: PLC0415

    d = Path(CHORUS_BACKGROUNDS_DIR)
    npzs = sorted(d.glob("*_pertrack.npz"))
    if not npzs:
        pytest.skip(f"no shipped backgrounds in {d}")
    actual_gib = sum(p.stat().st_size for p in npzs) / 1024**3
    label, stated = next(
        (lab, size) for lab, size in _disk_table_rows() if "backgrounds" in lab.lower()
    )
    assert abs(actual_gib - stated) <= 0.6, (
        f"README's backgrounds bucket says ~{stated} GB but {len(npzs)} shipped NPZs in {d} total "
        f"{actual_gib:.2f} GiB. This is exactly how the Sei rebuild slipped past the docs."
    )
