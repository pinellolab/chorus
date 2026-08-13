"""``extract_sequence_with_padding`` must return exactly what it was asked for (audit F2).

The function had two branches and only one handled the chromosome boundary. When the requested
interval was already at least ``total_length`` wide it called ``fasta.fetch`` with no bounds
check — and pysam **silently clamps**, so a caller near a telomere got fewer bases than it asked
for, with no error, and with the returned metadata reporting ``leftN: 0, rightN: 0``, i.e. that
no padding had been needed.

Measured on chr1 (248,956,422 bp) before the fix, asking for 2,114 bp:

    interval start      returned
    L - 2114            2,114     (in bounds, fine)
    L - 1000            1,000
    L -   40               40

Silently returning the wrong length is worse than raising, which is what the raw
``extract_sequence`` does at a boundary. It matters because ``chorus/oracles/sei.py`` calls this
with ``total_length=SEI_WINDOW``, so a short one-hot could reach the model, and because the
other branch (interval narrower than ``total_length``) always padded correctly — so the two
branches disagreed about the same question.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _genome() -> str:
    for candidate in ("/data/chorus_data/genomes/hg38.fa", REPO / "genomes" / "hg38.fa"):
        p = Path(candidate)
        if p.exists() and Path(str(p) + ".fai").exists():
            return str(p)
    pytest.skip("no indexed hg38 available")


@pytest.fixture(scope="module")
def hg38():
    return _genome()


@pytest.fixture(scope="module")
def chr1_length(hg38):
    import pysam

    with pysam.FastaFile(hg38) as fa:
        return fa.get_reference_length("chr1")


# The offsets that used to fail. L-2114 is in bounds and was always correct; it is kept as the
# control, because a "fix" that pads unconditionally would break it.
@pytest.mark.parametrize("offset_from_end", [2114, 2000, 1000, 40, 1])
def test_the_full_length_is_returned_at_the_three_prime_end(hg38, chr1_length, offset_from_end):
    from chorus.utils.sequence import extract_sequence_with_padding

    start = chr1_length - offset_from_end
    seq, meta = extract_sequence_with_padding(
        hg38, "chr1", start, start + 2114, 2114, return_meta=True)

    assert len(seq) == 2114, (
        f"asked for 2,114 bp starting {offset_from_end} bp from chr1's end and got "
        f"{len(seq)}; pysam clamps silently, so this is the defect that reached Sei"
    )
    # The metadata has to agree with what was actually done, which is the half of this bug
    # that would have hidden it: it used to hardcode zeros.
    assert meta["rightN"] == max(0, 2114 - offset_from_end)
    assert meta["leftN"] == 0
    assert seq[-meta["rightN"]:] == "N" * meta["rightN"] if meta["rightN"] else True


def test_the_full_length_is_returned_at_the_five_prime_end(hg38):
    from chorus.utils.sequence import extract_sequence_with_padding

    seq, meta = extract_sequence_with_padding(
        hg38, "chr1", -500, 1614, 2114, return_meta=True)
    assert len(seq) == 2114
    assert meta["leftN"] == 500 and meta["rightN"] == 0
    assert seq[:500] == "N" * 500


def test_a_mid_chromosome_window_is_untouched(hg38):
    """The control: no padding, no N, and the metadata says so."""
    from chorus.utils.sequence import extract_sequence_with_padding

    seq, meta = extract_sequence_with_padding(
        hg38, "chr1", 100_000_000, 100_002_114, 2114, return_meta=True)
    assert len(seq) == 2114
    assert "N" not in seq
    assert meta == {"start_change": 0, "end_change": 0, "leftN": 0, "rightN": 0}


@pytest.mark.parametrize("offset_from_end", [2114, 1000, 40])
def test_both_branches_agree_at_a_boundary(hg38, chr1_length, offset_from_end):
    """The narrow-interval branch always padded correctly; the wide one now matches it.

    Two implementations of one question is how this defect existed at all, so pin that they
    give the same answer where they overlap: both must return exactly ``total_length``.
    """
    from chorus.utils.sequence import extract_sequence_with_padding

    start = chr1_length - offset_from_end
    wide = extract_sequence_with_padding(hg38, "chr1", start, start + 2114, 2114)
    narrow = extract_sequence_with_padding(hg38, "chr1", start, start + 100, 2114)
    assert len(wide) == len(narrow) == 2114


def test_sei_would_have_been_the_victim():
    """Records why this is not theoretical: the caller that passes a model's window size."""
    src = (REPO / "chorus" / "oracles" / "sei.py").read_text()
    assert "extract_sequence_with_padding" in src, (
        "sei.py no longer calls the padding helper -- if the caller set moved, re-derive "
        "which oracles this bug could reach"
    )
