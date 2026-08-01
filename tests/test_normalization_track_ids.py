"""The percentile path and the display path must resolve a track id the same way.

`PerTrackNormalizer` has one matcher, `_match_track_id`, whose docstring names
the LegNet case explicitly: the oracle emits ``"LentiMPRA:HepG2"`` while the
background row is keyed ``"HepG2"``. But only the *display* helpers
(`perbin_floor_rescale_batch`, `is_signed`, `signed_floor_rescale_batch`) ever
called it. `_lookup` / `_lookup_batch` did exact-match plus a strand-suffix
strip, so `effect_percentile` and `activity_percentile` returned ``None``.

The visible symptom was a whole oracle with no percentile: LegNet's three
shipped CDF rows were unreachable, and the report rendered an em dash that
read like deliberate suppression rather than a lookup miss. The invisible
symptom was worse — the IGV panel rescaled against row *i* while the table
column resolved to nothing, so the two halves of one report disagreed about
which background row the track even had.

See https://github.com/pinellolab/chorus/issues/126.
"""

import numpy as np
import pytest

from chorus.analysis.normalization import PerTrackNormalizer

WIDTH = 10_000


def _normalizer(oracle: str, keys: list[str], signed: bool = False):
    """A normalizer over synthetic rows keyed exactly as the real NPZ is.

    Row *i* is ``arange(WIDTH) + i * WIDTH`` so a wrong-row match is
    detectable: each row covers a disjoint value range.
    """
    rows = np.stack([
        np.arange(WIDTH, dtype=np.float64) + i * WIDTH for i in range(len(keys))
    ])
    nz = PerTrackNormalizer()
    nz._loaded[oracle] = {
        "track_ids": list(keys),
        "track_index": {k: i for i, k in enumerate(keys)},
        "effect_cdfs": rows,
        "summary_cdfs": rows.copy(),
        "effect_counts": np.full(len(keys), WIDTH, dtype=np.int64),
        "summary_counts": np.full(len(keys), WIDTH, dtype=np.int64),
        "signed_flags": np.array([signed] * len(keys)),
    }
    return nz


# The real shipped naming schemes, oracle by oracle. Left = what the oracle
# emits as `assay_id`/`track_id`; right = how the background row is keyed.
# Verified against the shipped NPZs and each oracle's assay_id construction.
RESOLUTION_CASES = [
    # LegNet: build_backgrounds_legnet.py writes bare cell types
    # (`track_ids=np.array(cell_types)`), the oracle asks with the assay
    # prefix (`legnet.py:50`: f"{self.assay}:{self.cell_type}").
    ("legnet", ["K562", "HepG2", "WTC11"], "LentiMPRA:HepG2", "HepG2"),
    ("legnet", ["K562", "HepG2", "WTC11"], "LentiMPRA:K562", "K562"),
    # ChromBPNet accessibility: keyed exactly as emitted.
    ("chrombpnet", ["ATAC:K562", "DNASE:HepG2"], "DNASE:HepG2", "DNASE:HepG2"),
    # BPNet/CHIP: emitted per strand, stored strand-merged.
    ("chrombpnet", ["CHIP:HepG2:CEBPA"], "CHIP:HepG2:CEBPA:+", "CHIP:HepG2:CEBPA"),
    ("chrombpnet", ["CHIP:HepG2:CEBPA"], "CHIP:HepG2:CEBPA:-", "CHIP:HepG2:CEBPA"),
    # AlphaGenome: full catalogue identifier, stored verbatim.
    ("alphagenome", ["DNASE/EFO:0001187 DNase-seq/."],
     "DNASE/EFO:0001187 DNase-seq/.", "DNASE/EFO:0001187 DNase-seq/."),
    # Enformer / Borzoi: bare accessions.
    ("enformer", ["ENCFF430NNH", "CNhs13807"], "CNhs13807", "CNhs13807"),
]


@pytest.mark.parametrize("oracle,keys,queried,expected_key", RESOLUTION_CASES)
def test_effect_percentile_resolves_every_shipped_naming_scheme(
    oracle, keys, queried, expected_key
):
    """`effect_percentile` must resolve what the oracle actually emits."""
    nz = _normalizer(oracle, keys)
    expected_idx = keys.index(expected_key)
    # A value inside that row's disjoint range, at a known rank.
    probe = float(expected_idx * WIDTH + 2_500)

    got = nz.effect_percentile(oracle, queried, probe)

    assert got is not None, (
        f"{oracle}: effect_percentile({queried!r}) returned None; the "
        f"background row is keyed {expected_key!r} and _match_track_id "
        f"resolves it, so the percentile path must too"
    )
    # Rows are disjoint, so the value pins which row answered.
    assert got == pytest.approx(0.2501, abs=1e-4), (
        f"{oracle}: {queried!r} resolved to the wrong row — got {got}"
    )


@pytest.mark.parametrize("oracle,keys,queried,expected_key", RESOLUTION_CASES)
def test_activity_percentile_resolves_the_same_ids(
    oracle, keys, queried, expected_key
):
    """`activity_percentile` reads `summary_cdfs` and must agree with effect."""
    nz = _normalizer(oracle, keys)
    probe = float(keys.index(expected_key) * WIDTH + 2_500)
    assert nz.activity_percentile(oracle, queried, probe) is not None


@pytest.mark.parametrize("oracle,keys,queried,expected_key", RESOLUTION_CASES)
def test_lookup_agrees_with_match_track_id(oracle, keys, queried, expected_key):
    """The percentile path must pick the row `_match_track_id` names.

    This is the invariant the defect broke: one matcher, one answer. If these
    two ever diverge again, the IGV panel and the table will silently describe
    different background rows.
    """
    nz = _normalizer(oracle, keys)
    entry = nz._loaded[oracle]

    matched = nz._match_track_id(queried, entry["track_index"])
    assert matched == expected_key

    probe = float(keys.index(expected_key) * WIDTH + 2_500)
    via_lookup = nz._lookup(oracle, queried, "effect_cdfs", probe)
    via_matched = nz._lookup(oracle, matched, "effect_cdfs", probe)
    assert via_lookup == via_matched


def test_batch_and_scalar_lookups_agree():
    """`_lookup_batch` must resolve ids identically to `_lookup`."""
    nz = _normalizer("legnet", ["K562", "HepG2", "WTC11"])
    probe = 1_2500.0  # inside HepG2's row (index 1)

    scalar = nz._lookup("legnet", "LentiMPRA:HepG2", "effect_cdfs", probe)
    batch = nz._lookup_batch(
        "legnet", "LentiMPRA:HepG2", "effect_cdfs", np.array([probe])
    )

    assert scalar is not None
    assert batch is not None
    assert batch[0] == pytest.approx(scalar)


def test_an_unknown_track_still_returns_none():
    """Widening the matcher must not turn a genuine miss into a wrong answer."""
    nz = _normalizer("legnet", ["K562", "HepG2", "WTC11"])
    assert nz.effect_percentile("legnet", "LentiMPRA:NOT_A_CELL", 1.0) is None
    assert nz.activity_percentile("legnet", "totally:unrelated", 1.0) is None
