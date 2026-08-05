"""Every builder must name layers in ONE vocabulary. A second one matched zero rows.

Backgrounds carry a per-row ``layers_per_row`` array, and downstream code keys on it.
The builders did not agree on what to put there:

* AlphaGenome and Borzoi wrote canonical :data:`LAYER_CONFIGS` names —
  ``chromatin_accessibility``, ``histone_marks``, ``tf_binding``, ``tss_activity``,
  ``gene_expression``, ``splicing``.
* Enformer wrote its own internal ``spec_key`` — ``DNASE``, ``ATAC``, ``CHIP_TF``,
  ``CHIP_HIST``, ``CAGE``.

An effect-null composition keyed on ``chromatin_accessibility`` therefore matched
**472 of AlphaGenome's 5,168 rows and 0 of Enformer's 5,313** — silently, for the one
oracle where the change had actually been measured to help (50 % of its accessibility
rows saturated, dropping to 0 %). No exception, no warning: the operation simply did
nothing for half the fleet.

That is the same defect class as #122 (builder scored 501 bp, query scored 2001) and
#144 (builder and query summed different bin spans): two producers, two conventions,
and nothing comparing them. The fix follows the same shape as those — one shared
definition, called by everyone, that *raises* on anything it does not recognise
rather than falling back to a default.

Why raising matters here: a ``.get(key, key)``-style fallback would have let ``DNASE``
flow straight through into a shipped artefact, which is exactly how this happened.
An unrecognised assay group is a build-time error worth stopping a ten-hour job for.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
BUILDERS = ("enformer", "borzoi", "alphagenome")
BACKGROUNDS = Path.home() / ".chorus" / "backgrounds"


def _src(oracle: str) -> str:
    return (REPO / "scripts" / f"build_backgrounds_{oracle}.py").read_text()


# ---------------------------------------------------------------------------
# The mapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key,expected", [
    ("DNASE", "chromatin_accessibility"),
    ("ATAC", "chromatin_accessibility"),
    ("CHIP_TF", "tf_binding"),
    ("CHIP_HIST", "histone_marks"),
    ("CHIP_HISTONE", "histone_marks"),
    ("CAGE", "tss_activity"),
    ("PRO_CAP", "tss_activity"),
    ("RNA", "gene_expression"),
    ("RNA_SEQ", "gene_expression"),
    ("SPLICE_SITE_USAGE", "splicing"),
])
def test_every_builder_assay_key_maps_to_a_canonical_layer(key, expected):
    from chorus.analysis.scorers import canonical_layer

    assert canonical_layer(key) == expected


def test_canonical_names_are_idempotent():
    """AlphaGenome and Borzoi already write canonical names; passing one through
    must be a no-op, or routing them through the mapper would corrupt them."""
    from chorus.analysis.scorers import LAYER_CONFIGS, canonical_layer

    for name in LAYER_CONFIGS:
        assert canonical_layer(name) == name


def test_every_mapping_target_is_a_real_layer():
    """A typo in the mapping table would reintroduce the bug pointing the other way."""
    from chorus.analysis.scorers import _ASSAY_KEY_TO_LAYER, LAYER_CONFIGS

    unknown = {v for v in _ASSAY_KEY_TO_LAYER.values() if v not in LAYER_CONFIGS}
    assert not unknown, f"mapping targets absent from LAYER_CONFIGS: {sorted(unknown)}"


def test_an_unrecognised_key_raises_rather_than_passing_through():
    """The load-bearing behaviour. A silent fallback is how ``DNASE`` reached a
    shipped background in the first place."""
    from chorus.analysis.scorers import canonical_layer

    with pytest.raises(KeyError, match="no canonical layer"):
        canonical_layer("SOME_NEW_ASSAY")
    with pytest.raises(KeyError):
        canonical_layer("")


# ---------------------------------------------------------------------------
# The builders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("oracle", BUILDERS)
def test_builder_routes_its_per_row_layer_through_the_mapper(oracle):
    src = _src(oracle)
    assert "canonical_layer(t['layer'])" in src, (
        f"{oracle} must build layers_per_row via canonical_layer so a "
        f"non-canonical name cannot reach the artefact"
    )
    assert re.search(r"from chorus\.analysis\.scorers import [^\n]*canonical_layer", src)


@pytest.mark.parametrize("oracle", BUILDERS)
def test_builder_does_not_write_spec_key_as_the_layer(oracle):
    """The specific regression. Enformer's ``spec_key`` is a window-selection key,
    not a layer, and the two vocabularies are not interchangeable."""
    src = _src(oracle)
    assert "t.get('spec_key') or 'unknown'" not in src
    assert "[str(t.get('spec_key')" not in src


# ---------------------------------------------------------------------------
# The artefacts on disk
# ---------------------------------------------------------------------------


def _interims():
    if not BACKGROUNDS.is_dir():
        return []
    return sorted(p for p in BACKGROUNDS.glob("*_effect_cdfs_interim*.npz"))


@pytest.mark.parametrize("path", _interims() or [Path("none")],
                         ids=lambda p: p.name)
def test_any_stored_layers_per_row_is_canonical(path: Path):
    """Whatever is on disk must already be in the canonical vocabulary.

    Files written before this fix legitimately have no ``layers_per_row`` at all and
    are skipped; what must never happen is a file that HAS the field and fills it
    with something downstream cannot key on.
    """
    if not path.exists():
        pytest.skip("no interims on disk")
    from chorus.analysis.scorers import LAYER_CONFIGS

    with np.load(path, allow_pickle=False) as data:
        if "layers_per_row" not in data.files:
            pytest.skip("predates the per-row layer field")
        values = {str(x) for x in data["layers_per_row"]}

    bad = sorted(v for v in values if v not in LAYER_CONFIGS)
    assert not bad, (
        f"{path.name} stores non-canonical layer name(s) {bad}. Downstream code keys "
        f"on this field against LAYER_CONFIGS, so these rows would match nothing — "
        f"silently, which is how Enformer's 5,313 rows were skipped while "
        f"AlphaGenome's 472 were swapped."
    )
