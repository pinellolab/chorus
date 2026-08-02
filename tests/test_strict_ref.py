"""A ref-allele/genome mismatch must raise, not warn and score something else (#128).

Chorus used to log "does not match the genome" and then **substitute the supplied
allele into the prediction interval** — scoring a synthetic, non-reference
sequence and reporting the result as though nothing were wrong.

That is not hypothetical. A committed BCL11A example carried ``ref="G"`` where
hg38 has ``T``, shipped that way for months, and the warning fired on **every
single run**. A warning that always fires is not a safety net.

So ``strict_ref`` defaults to **True**. This is a breaking change: callers with
slightly-off coordinates now get an exception where they previously got a
plausible-looking number. That is the point. ``strict_ref=False`` or
``CHORUS_ALLOW_REF_MISMATCH=1`` restores the old behaviour for anyone who wants
to score a synthetic reference deliberately.

The offline half is ``tests/fixtures/example_ref_alleles.tsv``, which pins every
variant used by a committed example against hg38 so a bad coordinate is caught at
test time rather than at scoring time.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from chorus.core.base import OracleBase
from chorus.core.exceptions import ChorusError, ReferenceAlleleMismatchError

FIXTURE = Path("tests/fixtures/example_ref_alleles.tsv")
_HG38 = Path("genomes/hg38.fa")


class _Bare(OracleBase):
    """Minimal concrete oracle: only ``__init__`` behaviour is under test here.

    Every abstract method is stubbed rather than reaching for a real oracle, so
    these stay hermetic — no weights, no FASTA, no GPU.
    """

    def load_pretrained_model(self, weights: str = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def _predict(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def _get_bin_size(self):  # pragma: no cover
        return 128

    def _get_context_size(self):  # pragma: no cover
        return 393_216

    def _get_sequence_length_bounds(self):  # pragma: no cover
        return (1, 393_216)

    def fine_tune(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def list_assay_types(self):  # pragma: no cover
        return []

    def list_cell_types(self):  # pragma: no cover
        return []


# ---------------------------------------------------------------------------
# The default and its escape hatches
# ---------------------------------------------------------------------------


def test_strict_ref_defaults_to_true():
    """The breaking change, asserted explicitly so it cannot be reverted quietly."""
    with mock.patch.dict(os.environ, {}, clear=True):
        assert _Bare(use_environment=False).strict_ref is True


def test_env_var_opts_out():
    with mock.patch.dict(os.environ, {"CHORUS_ALLOW_REF_MISMATCH": "1"}, clear=True):
        assert _Bare(use_environment=False).strict_ref is False


def test_explicit_argument_beats_the_env_var():
    """An explicit True must win, so a stray env var cannot weaken a caller."""
    with mock.patch.dict(os.environ, {"CHORUS_ALLOW_REF_MISMATCH": "1"}, clear=True):
        assert _Bare(use_environment=False, strict_ref=True).strict_ref is True
    with mock.patch.dict(os.environ, {}, clear=True):
        assert _Bare(use_environment=False, strict_ref=False).strict_ref is False


def test_the_error_is_catchable_both_ways():
    """``ValueError`` for callers with that contract, ``ChorusError`` for the rest —
    the same dual inheritance ``InvalidRegionError`` uses."""
    assert issubclass(ReferenceAlleleMismatchError, ValueError)
    assert issubclass(ReferenceAlleleMismatchError, ChorusError)


def test_the_message_says_what_to_do():
    """An error a user cannot act on is barely better than the warning was."""
    err = ReferenceAlleleMismatchError(
        "reference allele 'G' does not match chr2:60490908, where the genome has "
        "'T'. Check the coordinates and the genome build. To do that deliberately, "
        "pass strict_ref=False or set CHORUS_ALLOW_REF_MISMATCH=1."
    )
    text = str(err)
    assert "chr2:60490908" in text          # where
    assert "'G'" in text and "'T'" in text  # what was expected vs found
    assert "strict_ref=False" in text       # how to override


# ---------------------------------------------------------------------------
# The committed fixture
# ---------------------------------------------------------------------------


def _fixture_rows() -> list[tuple[str, int, str, str]]:
    rows = []
    for line in FIXTURE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        chrom, pos, ref, example = line.split("\t")[:4]
        rows.append((chrom, int(pos), ref, example))
    return rows


def test_fixture_is_well_formed():
    rows = _fixture_rows()
    assert len(rows) == 4, f"expected 4 pinned variants, got {len(rows)}"
    for chrom, pos, ref, example in rows:
        assert chrom.startswith("chr")
        assert pos > 0
        assert set(ref) <= set("ACGT"), f"{example}: {ref!r} is not a nucleotide"
        assert example


def test_fixture_covers_every_real_variant_in_the_examples():
    """If an example gains a variant, it must be pinned here too.

    Skips ``ref_allele`` values that are display sentinels rather than
    nucleotides: region_swap and integration_simulation record ``"wt"``, and both
    go through predict_region_replacement, never the ref-allele check.
    """
    import glob
    import json

    pinned = {(c, p) for c, p, _, _ in _fixture_rows()}
    found: set[tuple[str, int]] = set()
    sentinels: set[str] = set()

    for path in glob.glob("examples/**/example_output.json", recursive=True):
        with open(path) as fh:
            data = json.load(fh)

        def walk(obj):
            if isinstance(obj, dict):
                chrom, pos, ref = (obj.get("chrom"), obj.get("position"),
                                   obj.get("ref_allele"))
                if isinstance(chrom, str) and isinstance(pos, int) and isinstance(ref, str) and ref:
                    if set(ref.upper()) <= set("ACGT"):
                        found.add((chrom, pos))
                    else:
                        sentinels.add(ref)
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, list):
                for value in obj:
                    walk(value)

        walk(data)

    assert found <= pinned, f"unpinned example variants: {sorted(found - pinned)}"
    # documents the sentinel rather than silently dropping it
    assert sentinels <= {"wt"}, f"unexpected non-nucleotide ref alleles: {sentinels}"


@pytest.mark.integration
def test_pinned_alleles_match_hg38():
    """The check that would have caught BCL11A. Needs the 3 GB FASTA."""
    if not _HG38.exists():
        pytest.skip("genomes/hg38.fa not available")
    from pyfaidx import Fasta

    fasta = Fasta(str(_HG38), as_raw=True, sequence_always_upper=True)
    wrong = []
    for chrom, pos, ref, example in _fixture_rows():
        actual = str(fasta[chrom][pos - 1:pos - 1 + len(ref)])
        if actual != ref:
            wrong.append(f"{example} {chrom}:{pos} says {ref!r}, hg38 has {actual!r}")
    assert not wrong, "\n".join(wrong)
