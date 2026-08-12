"""A chorus version must name the artefact revision it was verified against.

A percentile is a function of **(code, artefacts)**, and the artefacts live in a separate
HuggingFace dataset whose ``main`` moves independently. Until 0.7.0 nothing pinned it, so
the same chorus commit produced different numbers depending on when the user happened to
download — the exact thing a version tag is supposed to rule out.

That is demonstrated rather than hypothetical: on 2026-08-10 every file in the dataset was
replaced in place by the unified rebuild, which silently changed the behaviour of every
already-released chorus version that fetched afterwards. The old artefacts were not lost —
a dataset repo is a git repo — so both states are now tagged and the pairing is explicit:

    chorus <= 0.6.0   backgrounds-2026-08-01-preunified   schema < 4, thinned ceilings
    chorus 0.7.0      backgrounds-2026-08-06-schema4      exact effect/summary retention

These tests are deliberately split by cost. The first three are offline and always run:
they check that the pin exists, that it is a tag rather than a floating branch, and that
every download site honours it. The last two reach the network and are marked
``integration``: they check the pin still resolves and that what it resolves to is what the
local artefacts actually are.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NORMALIZATION = REPO / "chorus" / "analysis" / "normalization.py"


def test_the_revision_is_pinned_and_is_not_a_moving_branch():
    from chorus.analysis.normalization import _HF_REVISION

    assert _HF_REVISION, "no artefact revision pinned"
    assert _HF_REVISION not in {"main", "master", "HEAD"}, (
        f"the artefact revision is pinned to {_HF_REVISION!r}, which moves. A release that "
        f"tracks the dataset's head is not reproducible: the 2026-08-10 rebuild replaced "
        f"every file in place and changed the numbers of every version that fetched after "
        f"it. Pin a tag."
    )
    assert _HF_REVISION.startswith("backgrounds-"), (
        f"expected a dataset tag of the form backgrounds-<date>-<name>, got "
        f"{_HF_REVISION!r}; the naming is what makes the code/artefact pairing legible"
    )


def test_every_download_site_honours_the_pin():
    """One unpinned call is enough to reintroduce the drift.

    The legacy per-layer path also *lists* the repo, and listing ``main`` while fetching a
    tag can ask for a file that does not exist at that revision — so the listing counts as
    a site too.
    """
    src = NORMALIZATION.read_text()
    downloads = [m.start() for m in re.finditer(r"hf_hub_download\(", src)]
    assert downloads, "no hf_hub_download call found -- has the download path moved?"
    unpinned = []
    for pos in downloads:
        call = src[pos:pos + 400]
        end = call.find(")")
        if "revision=" not in call[:end if end > 0 else len(call)]:
            line = src[:pos].count("\n") + 1
            unpinned.append(line)
    assert not unpinned, (
        f"hf_hub_download at line(s) {unpinned} does not pass revision=, so it fetches the "
        f"dataset's head regardless of the pin"
    )

    listings = [m.start() for m in re.finditer(r"list_repo_files\(", src)]
    unpinned_lists = []
    for pos in listings:
        call = src[pos:pos + 300]
        end = call.find(")")
        if "revision=" not in call[:end if end > 0 else len(call)]:
            unpinned_lists.append(src[:pos].count("\n") + 1)
    assert not unpinned_lists, (
        f"list_repo_files at line(s) {unpinned_lists} lists the dataset head while the "
        f"download is pinned; the two must agree or the fetch can 404"
    )


def test_the_override_is_documented_because_it_defeats_reproducibility():
    """`CHORUS_BACKGROUNDS_REVISION=main` is a legitimate development escape hatch.

    It is also a foot-gun for anyone reporting numbers, so the reason has to be written
    next to it rather than discovered.
    """
    src = NORMALIZATION.read_text()
    assert "CHORUS_BACKGROUNDS_REVISION" in src
    window_start = src.index("CHORUS_BACKGROUNDS_REVISION")
    context = src[max(0, window_start - 1400):window_start + 600]
    assert "reproduc" in context.lower(), (
        "the CHORUS_BACKGROUNDS_REVISION override has no note explaining that overriding "
        "it costs reproducibility; that is the whole reason the pin exists"
    )


@pytest.mark.integration
def test_the_pinned_revision_still_resolves():
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    from chorus.analysis.normalization import _HF_REPO, _HF_REVISION

    api = HfApi()
    try:
        files = api.list_repo_files(
            _HF_REPO, repo_type="dataset", revision=_HF_REVISION,
        )
    except HfHubHTTPError as exc:  # pragma: no cover - network/gating dependent
        pytest.skip(f"cannot reach {_HF_REPO}: {exc}")
    npz = sorted(f for f in files if f.endswith("_pertrack.npz"))
    # Nine, not eight: Cherimoya ships one null per fold mode since v0.7.2 --
    # cherimoya_pertrack.npz (fold 0, the default) and cherimoya_ensemble_pertrack.npz.
    # A percentile is a rank against a null, so each mode needs a null built by the same
    # model; the folds disagree by 2.02x on the same sequence. See
    # tests/test_fold_selects_its_own_null.py.
    assert len(npz) == 9, (
        f"the pinned revision {_HF_REVISION} holds {len(npz)} per-track files, expected 9 "
        f"(eight oracles, plus a second Cherimoya null for the ensemble fold mode): {npz}"
    )


@pytest.mark.integration
def test_the_pinned_revision_matches_the_artefacts_on_this_machine():
    """The pin must name the artefacts we actually tested against, not merely a valid tag.

    Compares byte size rather than downloading 1.9 GB. A size match is not proof of
    identity, but a size *mismatch* is proof the pin and the local files disagree, which is
    the failure this catches.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    from chorus.core.globals import CHORUS_DATA_DIR, resolve_backgrounds_dir

    from chorus.analysis.normalization import _HF_REPO, _HF_REVISION

    bg = Path(resolve_backgrounds_dir(Path(CHORUS_DATA_DIR)))
    local = {p.name: p.stat().st_size for p in bg.glob("*_pertrack.npz")}
    if not local:
        pytest.skip(f"no per-track artefacts under {bg}")

    api = HfApi()
    try:
        remote = {
            s.path: getattr(s, "size", 0)
            for s in api.list_repo_tree(
                _HF_REPO, repo_type="dataset", revision=_HF_REVISION, recursive=True,
            )
        }
    except HfHubHTTPError as exc:  # pragma: no cover - network dependent
        pytest.skip(f"cannot reach {_HF_REPO}: {exc}")

    mismatched = {
        name: (size, remote.get(name))
        for name, size in local.items()
        if name in remote and remote[name] != size
    }
    assert not mismatched, (
        f"the pinned revision {_HF_REVISION} disagrees with the artefacts on disk "
        f"(name: local, remote): {mismatched}. Either the pin names the wrong revision or "
        f"the local files are from a different build -- in both cases the percentiles this "
        f"machine produces are not the ones the pin claims."
    )
