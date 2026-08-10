"""ChromBPNet's registry must stay human-only, and its two model families both built.

Two independent ways this oracle has gone wrong, both of which produce a background that
looks entirely normal:

**Mouse models scored with human sequence.** 33 ENCODE mouse ChromBPNet models were
removed on 2026-08-01 because the builder opens ``hg38.fa`` unconditionally — their CDFs
had been built by pushing human sequence through a mouse model. Nothing about the
resulting file looks wrong: the rows are non-degenerate, monotone, correctly shaped, and
meaningless. Only the registry says so.

**Half the oracle silently omitted.** ``--assay`` defaults to ``ATAC_DNASE``, so a
rebuild launched without it enumerates the 9 ChromBPNet accessibility models and drops
all 744 BPNet CHIP models. That happened during the 2026-08-06 rebuild: it exited 0,
scored 9 of 9 tracks, reported 100% yield, exact retention and a perbin tail with 400
exact grid slots, and wrote a 1.0 MB file to replace an 80 MB one. A flawless build of
1.2% of the job.

The 753 tracks are TWO architectures under one oracle name, which is the root of the
confusion worth pinning here:

    ChromBPNet   9 models    ATAC/DNASE accessibility, 5 cell lines
    BPNet      744 models    CHIP-seq TF, 240 distinct TFs

(NB: the taxonomy id 9606 collides visually with Enformer's 9,606-point CDF grid, which
appears all over ``tests/test_sampling_all_or_nothing.py``. They are unrelated.)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
META = REPO / "chorus" / "oracles" / "chrombpnet_source" / "chrombpnet_JASPAR_metadata.tsv"
HUMAN = "9606"


def _rows():
    with open(META) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# ---------------------------------------------------------------------------
# Human only
# ---------------------------------------------------------------------------


def test_every_bpnet_model_in_the_registry_is_human():
    """The builder opens hg38.fa, so a non-human model is scored on the wrong genome."""
    rows = _rows()
    assert rows, f"{META} is empty or unreadable"
    non_human = [(r["BASE_ID"], r["TF_NAME"], r["CELL_LINE"], r["TAX_ID"])
                 for r in rows if r["TAX_ID"] != HUMAN]
    assert not non_human, (
        f"{len(non_human)} non-human model(s) in the registry, e.g. {non_human[:5]}. "
        f"The builder feeds hg38 sequence to whatever it loads, so these would produce "
        f"a well-formed, monotone, non-degenerate and meaningless null."
    )
    assert len({r["TAX_ID"] for r in rows}) == 1


def test_no_shipped_chrombpnet_row_is_mouse():
    """Checked against the artefact, not just the registry.

    Uses the model registry as the source of truth for which cell types are human,
    rather than substring-matching names: "Mus" matches "muscle", and the human tissues
    `psoas muscle`, `cardiac muscle cell`, `smooth muscle cell`, `skeletal muscle
    myoblast` and `esophagus muscularis mucosa` all false-positive on a naive check.
    """
    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR

    path = CHORUS_BACKGROUNDS_DIR / "chrombpnet_pertrack.npz"
    if not path.exists():
        pytest.skip("no downloaded background for chrombpnet")
    with np.load(path, allow_pickle=True) as d:
        ids = [str(x) for x in d["track_ids"]]

    human_cells = {r["CELL_LINE"] for r in _rows() if r["TAX_ID"] == HUMAN}
    from chorus.oracles.chrombpnet_source.chrombpnet_globals import iter_unique_models
    human_cells |= {c for _a, c, _e in iter_unique_models()}

    # chrombpnet ids are ATAC:CELL / DNASE:CELL / CHIP:CELL:TF -- cell is index 1.
    unknown = sorted({t.split(":")[1] for t in ids if t.count(":") >= 1
                      and t.split(":")[1] not in human_cells})
    assert not unknown, (
        f"shipped rows reference cell types absent from the human registry: {unknown[:8]}"
    )


# ---------------------------------------------------------------------------
# Both families, complete
# ---------------------------------------------------------------------------


def test_the_two_families_are_what_the_shipped_background_contains():
    from chorus.oracles.chrombpnet_source.chrombpnet_globals import (
        iter_unique_bpnet_models, iter_unique_models,
    )
    acc = list(iter_unique_models())
    chip = list(iter_unique_bpnet_models())
    assert len(acc) == 9, f"ChromBPNet accessibility family is {len(acc)}, expected 9"
    assert len(chip) == 744, f"BPNet CHIP family is {len(chip)}, expected 744"
    assert len(acc) + len(chip) == 753

    from chorus.core.globals import CHORUS_BACKGROUNDS_DIR
    path = CHORUS_BACKGROUNDS_DIR / "chrombpnet_pertrack.npz"
    if not path.exists():
        pytest.skip("no downloaded background for chrombpnet")
    with np.load(path, allow_pickle=True) as d:
        ids = [str(x) for x in d["track_ids"]]
    assert len(ids) == 753, (
        f"the shipped background has {len(ids)} tracks against 753 enumerable models. "
        f"A build launched without --assay all covers only the 9 accessibility models."
    )
    n_chip = sum(1 for t in ids if t.startswith("CHIP:"))
    n_acc = sum(1 for t in ids if t.startswith(("ATAC:", "DNASE:")))
    assert (n_acc, n_chip) == (9, 744), (n_acc, n_chip)


def test_the_default_assay_does_not_cover_the_oracle():
    """Pinned deliberately: the default is a partial build, so it must be refused.

    Not a bug in the default — scoring 744 BPNet models takes hours and the flag exists
    for a reason. The bug was that a partial build could reach a shipped file. That is
    what scope_violations now prevents.
    """
    from chorus.analysis.background_sampling import scope_violations

    assert scope_violations(9, label="chrombpnet", n_shipped=753), (
        "a 9-of-753 build must be refused; every other guard passes it"
    )
    assert scope_violations(753, label="chrombpnet", n_shipped=753) == []
