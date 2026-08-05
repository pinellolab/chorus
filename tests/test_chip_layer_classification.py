"""Builder and query must agree on whether a CHIP track is histone or TF.

#122, which is instance 1 of the #144 class. The two sides classified the same
track from different fields:

* the **builder** passed ``info['description']`` to its own
  ``classify_chip_layer`` (``build_backgrounds_alphagenome.py:249``). AlphaGenome's
  descriptions read ``"CHIP:<cell type>"`` and carry **no mark name**, so no
  pattern ever matched and **0 of 2,733** CHIP tracks were built as histone —
  every one got the 501 bp ``tf_binding`` window.
* the **query** searched ``assay_id``, which *does* carry the mark
  (``scorers.py:185-195``), and returned ``histone_marks`` for 1,075 of them —
  scored at 2001 bp.

So those tracks compared a 2001 bp statistic against a 501 bp null.

Measured on the shipped ``alphagenome_pertrack.npz`` (5,168 rows):

===================================================  =====
CHIP_HISTONE tracks                                  1,116
CHIP_TF tracks                                       1,617
of the 1,116, matched by the 15-mark pattern list     1,075
**missed** by it (all acetylation)                       41
CHIP_TF tracks the patterns wrongly match                 0
===================================================  =====

The 41 missed are H2AK5ac, H2BK5ac, H2BK12ac, H2BK120ac, H3K18ac and friends —
real histone marks absent from a hand-maintained list. So the identifier prefix
is not merely a cheaper fix than extending that list, it is a *more correct* one:
AlphaGenome already states the distinction, and the prefix classifies all 1,116.

Extending the shared 15-mark list instead would move ~105 enformer and ~105
borzoi tracks from 501 to 2001 bp and stale two more backgrounds, so that stays
out of scope and is filed separately.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chorus.analysis.scorers import classify_chip_layer

_NPZ_CANDIDATES = (
    Path("/data/chorus_data/backgrounds/alphagenome_pertrack.npz"),
    Path.home() / ".chorus" / "backgrounds" / "alphagenome_pertrack.npz",
)

# The exact ids the 15-mark pattern list cannot see.
ACETYLATION_THE_PATTERNS_MISS = (
    "CHIP_HISTONE/CL:0000047 Histone ChIP-seq H2AK5ac/.",
    "CHIP_HISTONE/CL:0000047 Histone ChIP-seq H2BK5ac/.",
    "CHIP_HISTONE/CL:0000134 Histone ChIP-seq H2BK120ac/.",
    "CHIP_HISTONE/CL:0000134 Histone ChIP-seq H2BK12ac/.",
    "CHIP_HISTONE/CL:0000047 Histone ChIP-seq H3K18ac/.",
)


def _shipped_track_ids() -> list[str] | None:
    for path in _NPZ_CANDIDATES:
        if path.exists():
            with np.load(path, allow_pickle=True) as data:
                return [str(t) for t in data["track_ids"]]
    return None


# ---------------------------------------------------------------------------
# The prefix is authoritative
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("assay_id", ACETYLATION_THE_PATTERNS_MISS)
def test_acetylation_marks_the_pattern_list_misses(assay_id):
    """These are the 41. A pattern-based classifier calls them tf_binding."""
    assert classify_chip_layer(assay_id) == "histone_marks"


def test_prefix_wins_when_the_description_lacks_the_mark():
    """The actual #122 trigger: AlphaGenome descriptions are ``CHIP:<cell type>``.

    The builder saw only that string, so nothing matched. The prefix has to be
    consulted, not the description.
    """
    assert classify_chip_layer(
        "CHIP_HISTONE/EFO:0001187 Histone ChIP-seq H3K27ac/.",
        description="CHIP:hepatocyte",
    ) == "histone_marks"
    assert classify_chip_layer(
        "CHIP_TF/EFO:0001187 TF ChIP-seq CEBPA/.", description="CHIP:hepatocyte",
    ) == "tf_binding"


def test_prefix_beats_a_misleading_mark_substring():
    """A TF track must not be reclassified because its cell type reads like a mark."""
    assert classify_chip_layer("CHIP_TF/... TF ChIP-seq H3K27ac-like/.") == "tf_binding"


# ---------------------------------------------------------------------------
# Fallback for oracles with no prefix (enformer, borzoi, chrombpnet)
# ---------------------------------------------------------------------------


def test_pattern_fallback_still_works_without_a_prefix():
    assert classify_chip_layer("ENCFF123ABC H3K27ac") == "histone_marks"
    assert classify_chip_layer("ENCFF123ABC CTCF") == "tf_binding"


def test_fallback_reads_the_description_too():
    assert classify_chip_layer("ENCFF123ABC", description="H3K4me3 ChIP") == "histone_marks"


def test_unknown_defaults_to_tf_binding():
    """Unchanged default: the 501 bp window is the conservative choice."""
    assert classify_chip_layer("something opaque") == "tf_binding"


# ---------------------------------------------------------------------------
# Against the shipped artefact
# ---------------------------------------------------------------------------


def test_every_shipped_chip_track_classifies_by_its_prefix():
    ids = _shipped_track_ids()
    if ids is None:
        pytest.skip("no downloaded alphagenome background")

    histone = [t for t in ids if t.startswith("CHIP_HISTONE/")]
    tf = [t for t in ids if t.startswith("CHIP_TF/")]
    assert len(histone) == 1116, f"expected 1,116 CHIP_HISTONE, got {len(histone)}"
    assert len(tf) == 1617, f"expected 1,617 CHIP_TF, got {len(tf)}"

    wrong_h = [t for t in histone if classify_chip_layer(t) != "histone_marks"]
    wrong_t = [t for t in tf if classify_chip_layer(t) != "tf_binding"]
    assert not wrong_h, f"{len(wrong_h)} CHIP_HISTONE misclassified, e.g. {wrong_h[:3]}"
    assert not wrong_t, f"{len(wrong_t)} CHIP_TF misclassified, e.g. {wrong_t[:3]}"


def test_the_prefix_classifies_strictly_more_than_the_pattern_list():
    """Pins the 1,075 vs 1,116 gap, so a regression to patterns is visible."""
    from chorus.analysis.scorers import _HISTONE_PATTERNS

    ids = _shipped_track_ids()
    if ids is None:
        pytest.skip("no downloaded alphagenome background")

    histone = [t for t in ids if t.startswith("CHIP_HISTONE/")]
    patterns = {p.upper() for p in _HISTONE_PATTERNS}
    by_pattern = sum(any(p in t.upper() for p in patterns) for t in histone)
    assert by_pattern == 1075, f"pattern list matched {by_pattern}, expected 1,075"
    assert len(histone) - by_pattern == 41


# ---------------------------------------------------------------------------
# The builder must not keep its own copy (#144)
# ---------------------------------------------------------------------------


def test_builder_imports_the_shared_classifier_and_has_no_local_copy():
    """Source-text assertion, the ``tests/test_cherimoya.py:609`` pattern.

    The builder carried a private ``classify_chip_layer`` *and* a private
    15-mark ``HISTONE_PATTERNS`` — a fifth copy of arithmetic that already
    existed elsewhere, which is exactly how #122 became possible.
    """
    src = Path("scripts/build_backgrounds_alphagenome.py").read_text()
    # Assert the NAME is imported from that module, not an exact import line. The
    # line-literal version broke the moment a second name was added to the same
    # import (canonical_layer), even though the property held -- a guard that fails
    # on formatting trains people to edit the guard.
    import ast
    imported = {
        alias.asname or alias.name
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.ImportFrom) and node.module == "chorus.analysis.scorers"
        for alias in node.names
    }
    assert "classify_chip_layer" in imported, \
        f"builder must import the shared classifier; imports {sorted(imported)}"
    assert "def classify_chip_layer" not in src, \
        "builder still defines its own classify_chip_layer"
    assert "HISTONE_PATTERNS = frozenset" not in src, \
        "builder still carries its own histone pattern list"


def test_builder_passes_the_identifier_not_the_description():
    """#122 in one line: it passed the field that does not carry the mark."""
    src = Path("scripts/build_backgrounds_alphagenome.py").read_text()
    assert "classify_chip_layer(desc)" not in src, \
        "builder still classifies from the description, which has no mark name"
