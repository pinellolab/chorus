"""An outside contributor must be able to develop an oracle without our HuggingFace dataset.

The capability existed and the documentation did not mention it. `normalization.py` carries an
explicit comment — *"Override with the CHORUS_BACKGROUNDS_REPO environment variable so contributors
can self-host their own backgrounds (e.g. while developing a new oracle)"* — and `cache_dir` is
checked before any download. Yet `CHORUS_BACKGROUNDS_REPO` appeared in **none** of README.md,
CONTRIBUTING.md, `docs/BACKGROUND_NULL_PROTOCOL.md` or `docs/NORMALIZATION_GUIDE.md`.

That gap is the single largest deterrent to an outside model contribution. Read as written, §8 says
percentiles require a per-track null, the canonical nulls live in a dataset you cannot write to, and
building one means scoring ~18,000 positions through your model — so a contributor concludes they
need the lab's infrastructure and stops. The escape hatch turns that wall into a config line.

These tests pin both the mechanism and its documentation, because a feature nobody can find is
indistinguishable from one that does not exist.
"""
from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def test_the_repo_override_is_read_from_the_environment():
    """`CHORUS_BACKGROUNDS_REPO` must actually redirect the dataset lookup."""
    import chorus.analysis.normalization as norm

    saved = os.environ.get("CHORUS_BACKGROUNDS_REPO")
    try:
        os.environ["CHORUS_BACKGROUNDS_REPO"] = "someuser/my-own-backgrounds"
        importlib.reload(norm)
        assert norm._HF_REPO == "someuser/my-own-backgrounds", (
            f"the override had no effect; _HF_REPO is {norm._HF_REPO!r}. Without it a contributor "
            f"cannot host their own background while developing an oracle."
        )
    finally:
        if saved is None:
            os.environ.pop("CHORUS_BACKGROUNDS_REPO", None)
        else:
            os.environ["CHORUS_BACKGROUNDS_REPO"] = saved
        importlib.reload(norm)
        assert norm._HF_REPO == "lucapinello/chorus-backgrounds", "default repo not restored"


@pytest.mark.integration
def test_a_local_npz_serves_an_oracle_name_the_dataset_never_heard_of():
    """`cache_dir` is checked before any download — the fully-offline contributor path.

    Integration-marked only because it copies a real shipped NPZ to get a valid file; the mechanism
    under test is local-only and touches no network.
    """
    from chorus.analysis.normalization import CHORUS_BACKGROUNDS_DIR, get_pertrack_normalizer

    donor = Path(CHORUS_BACKGROUNDS_DIR) / "legnet_pertrack.npz"
    if not donor.is_file():
        pytest.skip("no shipped NPZ available to use as a stand-in")

    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "chorus_selfhost_probe"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy(donor, tmp / "brandnewmodel_pertrack.npz")
        nz = get_pertrack_normalizer("brandnewmodel", cache_dir=str(tmp))
        assert nz is not None, (
            "a local <name>_pertrack.npz did not satisfy get_pertrack_normalizer, so the documented "
            "offline contributor path does not work"
        )
        assert len(nz._loaded["brandnewmodel"]["track_ids"]) > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.parametrize("doc", [
    "CONTRIBUTING.md",
    "docs/BACKGROUND_NULL_PROTOCOL.md",
])
def test_the_contributor_facing_docs_say_how(doc: str):
    """A capability nobody can find is indistinguishable from one that does not exist."""
    text = (REPO / doc).read_text()
    assert "CHORUS_BACKGROUNDS_REPO" in text, (
        f"{doc} does not mention CHORUS_BACKGROUNDS_REPO. Without it, §8 reads as 'you need our "
        f"HuggingFace dataset', which is the biggest deterrent to an outside model contribution."
    )
    assert "cache_dir" in text, (
        f"{doc} does not mention the local `cache_dir` path, which is the fully-offline option"
    )


def test_contributing_says_a_partial_background_is_acceptable():
    """Willingness matters as much as capability: say what we will take.

    A contributor who believes only a full 18,000-position null is acceptable will not open the PR.
    """
    text = (REPO / "CONTRIBUTING.md").read_text()
    lowered = text.lower()
    assert "small" in lowered and "block your contribution" in lowered, (
        "CONTRIBUTING should state plainly that a small background, or none with percentiles "
        "labelled as unavailable, is still a useful PR — otherwise step 10 reads as a wall"
    )
