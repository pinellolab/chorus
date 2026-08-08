"""A shipped background row that the query path cannot reach is a silent dead oracle.

Sei shipped 40 built, verified, non-degenerate background rows that **no query could
ever use**. `oracles/sei.py` set `assay_type = info.name` for its sequence classes
(e.g. `"Polycomb-repressed"`), `classify_track_layer` has no branch for that and
returned `"other"`, `LAYER_CONFIGS.get("other")` is `None`, so `score_track_effect`
returned `None` and every Sei track scored `raw_score=None`. Nothing raised. Sei simply
never appeared in any committed example output, and the absence read as "we did not
include Sei in the examples" rather than "Sei cannot be scored".

Two independent things have to hold for a shipped row to be usable, and this module
checks both, because the failure above needed only one of them to break:

1. the track's ``assay_type`` must classify to a real ``LAYER_CONFIGS`` layer;
2. the track's ``assay_id`` must resolve to a background row.

Check 2 is deliberately run against **every** shipped row of **every** oracle, since it
is pure CPU and catches the id-drift class (LegNet ships rows keyed by bare cell name,
``'K562'``, while its assay_id is ``'LentiMPRA:K562'`` -- only the fuzzy matcher bridges
that, and this pins that the bridge keeps working).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent

ORACLES = ["alphagenome", "borzoi", "enformer", "chrombpnet",
           "cherimoya", "sei", "legnet", "epinformerseq"]


def _track_ids(oracle: str) -> list[str]:
    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR

    path = CHORUS_BACKGROUNDS_DIR / f"{oracle}_pertrack.npz"
    if not path.exists():
        return []
    with np.load(path, allow_pickle=True) as d:
        return [str(x) for x in d["track_ids"]]


# ---------------------------------------------------------------------------
# 2. every shipped row resolves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("oracle", ORACLES)
def test_every_shipped_background_row_resolves(oracle):
    """A row nobody can look up is a row nobody can use."""
    from chorus.analysis.normalization import PerTrackNormalizer

    ids = _track_ids(oracle)
    if not ids:
        pytest.skip(f"no downloaded background for {oracle}")

    norm = PerTrackNormalizer()
    entry = norm._ensure_loaded(oracle)
    assert entry is not None, f"{oracle}: background did not load"

    unresolved = [t for t in ids if norm._resolve_row(t, entry) is None]
    assert not unresolved, (
        f"{oracle}: {len(unresolved)} of {len(ids)} shipped track_ids do not resolve "
        f"to a row, e.g. {unresolved[:5]}"
    )


@pytest.mark.parametrize("oracle", ORACLES)
def test_resolution_is_one_to_one(oracle):
    """Two track ids resolving to the SAME row would silently share a null.

    Worse than an unresolved id, because it returns a plausible number computed
    against the wrong track's background.
    """
    from chorus.analysis.normalization import PerTrackNormalizer

    ids = _track_ids(oracle)
    if not ids:
        pytest.skip(f"no downloaded background for {oracle}")
    norm = PerTrackNormalizer()
    entry = norm._ensure_loaded(oracle)

    seen: dict[int, str] = {}
    collisions = []
    for t in ids:
        row = norm._resolve_row(t, entry)
        if row is None:
            continue
        if row in seen:
            collisions.append((seen[row], t, row))
        else:
            seen[row] = t
    assert not collisions, (
        f"{oracle}: distinct track_ids resolving to the same background row: "
        f"{collisions[:5]}"
    )


# ---------------------------------------------------------------------------
# 1. the layer half -- the actual Sei regression
# ---------------------------------------------------------------------------


def test_sei_sequence_classes_are_one_layer_not_forty_assay_types():
    """The exact regression, pinned without needing a GPU.

    Sei has 40 sequence classes. They are one regulatory layer. If the oracle labels
    tracks with the class NAME instead of the literal ``"sequence-class"``,
    ``classify_track_layer`` returns ``"other"`` and every score becomes None.
    """
    from chorus.analysis.scorers import LAYER_CONFIGS, classify_track_layer

    class _T:
        def __init__(self, assay_type, assay_id=""):
            self.assay_type, self.assay_id = assay_type, assay_id

    layer = classify_track_layer(_T("sequence-class"))
    assert layer == "regulatory_classification"
    assert LAYER_CONFIGS.get(layer) is not None, (
        "regulatory_classification must have a LAYER_CONFIGS entry, or scoring "
        "returns None even with the right layer name"
    )

    # And the failure mode itself: a class name must NOT be used as the assay_type.
    # Names are parsed from the SHIPPED track ids with the oracle's own parser, so
    # this is tied to the rows that actually exist rather than to a constant.
    from chorus.oracles.sei_source.annotations import SeiClass

    ids = _track_ids("sei")
    if not ids:
        pytest.skip("no downloaded background for sei")
    names = [SeiClass.from_str(i).name for i in ids if SeiClass.is_id(i)]
    assert len(names) == 40, f"expected Sei's 40 sequence classes, parsed {len(names)}"
    bad = [n for n in names if classify_track_layer(_T(n)) != "other"]
    assert not bad, (
        f"a Sei class NAME classifies to a real layer ({bad[:3]}), which would hide "
        f"the regression this test exists to catch"
    )
    # The point: every one of them lands in 'other', so labelling tracks with the
    # name (as sei.py did) silently disables all 40.
    assert all(classify_track_layer(_T(n)) == "other" for n in names)


def test_the_sei_oracle_assigns_the_literal_sequence_class_label():
    """Pinned against the source, because the assignment is inline in ``predict``.

    Parsed rather than string-matched: an earlier guard in this repo asserted an exact
    line of source text and passed only because the replacement it was verifying had
    introduced that very line.
    """
    import ast
    import inspect

    from chorus.oracles import sei as sei_mod

    tree = ast.parse(inspect.getsource(sei_mod))
    assigned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "assay_type":
                    if isinstance(node.value, ast.Constant):
                        assigned.add(node.value.value)
                    elif isinstance(node.value, ast.Attribute):
                        assigned.add(f"<attr:{node.value.attr}>")
    assert "sequence-class" in assigned, (
        f"sei.py never assigns the literal 'sequence-class' to assay_type; "
        f"assignments found: {sorted(assigned)}. Without it all 40 sequence-class "
        f"tracks classify as 'other' and score None."
    )
    assert "<attr:name>" not in assigned, (
        "sei.py assigns `info.name` to assay_type -- that is the regression: the "
        "class name is not an assay type, and it makes classify_track_layer return "
        "'other' for all 40 tracks"
    )


@pytest.mark.integration
def test_sei_actually_produces_scores_end_to_end():
    """The only check that would have caught this: run it and look.

    Runs through ``conda run -n chorus-sei`` rather than importing Sei directly,
    because Sei's deps do not coexist with the base env and ``chorus-sei`` has no
    pytest -- so a test written to import Sei in-process could never execute here,
    which is the "guard that protects nothing" failure this repo has hit repeatedly.

    Every unit-level guard above was satisfiable without Sei working. This one is not.
    """
    import json
    import os
    import subprocess

    ids = _track_ids("sei")
    if not ids:
        pytest.skip("no downloaded background for sei")

    conda = os.path.expanduser("~/miniforge3/bin/conda")
    if not os.path.exists(conda):
        pytest.skip("conda not found; cannot reach the chorus-sei env")

    code = r"""
import json, sys
sys.path.insert(0, %r)
from chorus.analysis.scorers import score_variant_multilayer
from chorus.core.globals import CHORUS_BACKGROUNDS_DIR, CHORUS_DATA_DIR
from chorus.oracles.sei import SeiOracle
import numpy as np
with np.load(CHORUS_BACKGROUNDS_DIR / "sei_pertrack.npz", allow_pickle=True) as d:
    ids = [str(x) for x in d["track_ids"]]
o = SeiOracle(); o.load_pretrained_model()
pos = "chr1:109274968"
res = o.predict_variant_effect(
    genomic_region=f"{pos}-109274969", variant_position=pos, alleles=["G", "T"],
    assay_ids=ids, genome=str(CHORUS_DATA_DIR / "genomes" / "hg38.fa"))
sc = score_variant_multilayer(res)
out = {a: {"n": len(tr),
           "n_scored": sum(1 for f in tr.values() if f.get("raw_score") is not None),
           "layers": sorted({f.get("layer") for f in tr.values()})}
       for a, tr in sc.items()}
print("@@@" + json.dumps(out))
""" % str(REPO)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    proc = subprocess.run([conda, "run", "-n", "chorus-sei", "python", "-c", code],
                          capture_output=True, text=True, timeout=1800, env=env)
    marker = [l for l in proc.stdout.splitlines() if l.startswith("@@@")]
    assert marker, (
        f"the sei subprocess produced no result.\nstdout:\n{proc.stdout[-2000:]}"
        f"\nstderr:\n{proc.stderr[-2000:]}"
    )
    got = json.loads(marker[-1][3:])
    assert got, "score_variant_multilayer returned nothing"
    for allele, info in got.items():
        assert info["n"] == len(ids), f"{allele}: scored {info['n']} of {len(ids)}"
        assert info["n_scored"] == len(ids), (
            f"{allele}: only {info['n_scored']} of {len(ids)} Sei tracks produced a "
            f"raw_score; before the fix this was 0 for every track"
        )
        assert info["layers"] == ["regulatory_classification"], info["layers"]
