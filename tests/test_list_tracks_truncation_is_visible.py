"""``list_tracks`` must never hand back a silent subsample.

All four search branches (borzoi, enformer, cherimoya, alphagenome) cap
``tracks`` at 200 rows. That cap is fine — 1,504 RNA records is not a useful
payload for an MCP response. What was not fine is that the cap was invisible:
the response carried ``num_results`` (true count) and ``tracks`` (200 rows) with
nothing tying them together, so a caller reading the field *named after the
thing it asked for* got 200 of 1,504 and could not tell.

That is the same failure mode as the reservoir thinning this release fixes: a
uniform subsample presented as the whole population, detectable only by someone
who already suspected it. The reservoir version cost an 8.3x-wrong tail. This
version costs an agent concluding a track does not exist.

So these tests assert the contract, not the implementation:

  - ``showing`` and ``truncated`` are ALWAYS present, on every branch, whether
    or not anything was dropped. A flag that appears only when set is a flag you
    have to know to look for.
  - ``showing == len(tracks)``, so the two can't drift.
  - ``truncated`` is true iff rows were actually dropped.
  - when truncated, ``note`` tells the caller what to do about it.

The oracle list is parametrised over the real metadata, so a fifth search branch
added later without going through ``_track_page`` fails here rather than
shipping.
"""
from __future__ import annotations

import pytest

from chorus.mcp.server import _TRACK_RESULT_CAP, list_tracks

# (oracle, query) pairs chosen so the first four exceed the cap and the last
# does not — the un-truncated path needs the same guarantees, and asserting
# only on the truncated one would let `showing`/`truncated` become conditional.
OVER_CAP = [
    ("alphagenome", "RNA"),
    ("enformer", "HepG2"),
    ("borzoi", "K562"),
    ("cherimoya", "ATAC"),
]
UNDER_CAP = [
    ("enformer", "DNASE:HepG2"),
    ("cherimoya", "DNASE:ENCSR000EOT"),
]


def _search(oracle: str, query: str) -> dict:
    r = list_tracks(oracle, query=query)
    if "error" in r:
        pytest.skip(f"{oracle} metadata unavailable: {r['error']}")
    return r


@pytest.mark.parametrize("oracle,query", OVER_CAP + UNDER_CAP)
def test_pagination_fields_are_always_present(oracle: str, query: str) -> None:
    """Present unconditionally — not only when something was dropped."""
    r = _search(oracle, query)
    for field in ("num_results", "showing", "truncated", "tracks"):
        assert field in r, f"{oracle}/{query} response is missing {field!r}: {sorted(r)}"
    assert isinstance(r["truncated"], bool)


@pytest.mark.parametrize("oracle,query", OVER_CAP + UNDER_CAP)
def test_showing_equals_the_rows_actually_returned(oracle: str, query: str) -> None:
    """The count and the list cannot disagree, which is how the bug hid."""
    r = _search(oracle, query)
    assert r["showing"] == len(r["tracks"])
    assert r["showing"] <= _TRACK_RESULT_CAP
    assert r["showing"] <= r["num_results"]


@pytest.mark.parametrize("oracle,query", OVER_CAP + UNDER_CAP)
def test_truncated_is_true_exactly_when_rows_were_dropped(oracle: str, query: str) -> None:
    r = _search(oracle, query)
    assert r["truncated"] is (r["num_results"] > r["showing"])


@pytest.mark.parametrize("oracle,query", OVER_CAP)
def test_a_truncated_response_says_so_in_prose(oracle: str, query: str) -> None:
    """A machine-readable flag plus a human/model-readable next step.

    The consumer here is usually an LLM, which will read ``note`` even if it
    ignores ``truncated``.
    """
    r = _search(oracle, query)
    assert r["truncated"] is True, f"{oracle}/{query} no longer exceeds the cap; pick another query"
    assert "note" in r
    note = r["note"]
    assert str(r["num_results"]) in note
    assert str(r["showing"]) in note
    assert "num_results" in note, "the note should name the field carrying the true count"


@pytest.mark.parametrize("oracle,query", UNDER_CAP)
def test_an_untruncated_response_carries_no_apology(oracle: str, query: str) -> None:
    """No spurious ``note`` when the caller already has everything."""
    r = _search(oracle, query)
    assert r["truncated"] is False
    assert r["num_results"] == r["showing"]
    assert "note" not in r


def test_every_search_branch_goes_through_the_helper() -> None:
    """Guard the pattern, not just today's four call sites.

    A new oracle branch that hand-rolls ``{"tracks": results[:200]}`` would pass
    every test above (it is never exercised) while reintroducing exactly the
    defect. So assert the raw slice appears nowhere in the module body outside
    the helper's own docstring.
    """
    import inspect
    import re

    import chorus.mcp.server as server

    src = inspect.getsource(server)
    # strip the helper (its docstring quotes the old pattern deliberately)
    helper = inspect.getsource(server._track_page)
    src = src.replace(helper, "")
    offenders = re.findall(r"results\[:\s*\d+\s*\]", src)
    assert not offenders, (
        f"hand-rolled track truncation found outside _track_page: {offenders}. "
        "Return _track_page(oracle_name, query, results) so the cap stays visible."
    )
