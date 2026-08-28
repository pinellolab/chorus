"""Tests for the bigwig-flavored sibling of chorus.utils.genome.require_assembly.

FASTA files get their assembly fingerprinted via a ``.fai``/pysam chr1 length;
bigwigs carry no such index but do expose chromosome lengths directly in their
header (``bw.chroms()``), so these functions fingerprint from that dict instead,
reusing the same ``ASSEMBLY_CHR1_LENGTH`` table and the same
raise-on-confident-mismatch / warn-on-unrecognized asymmetry as ``require_assembly``.
"""
from __future__ import annotations

import logging

import pytest

from chorus.core.exceptions import GenomeAssemblyMismatchError
from chorus.utils.genome import (
    ASSEMBLY_CHR1_LENGTH,
    chr1_length_from_chrom_sizes,
    detect_assembly_from_chrom_sizes,
    require_assembly_for_bigwig,
    require_assembly_from_chrom_sizes,
)


def _write_fixture_bigwig(path, chrom="chr1", chrom_size=1000, values=None):
    import pyBigWig

    if values is None:
        values = [1.0] * 20
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([(chrom, chrom_size)])
    bw.addEntries(chrom, list(range(len(values))), values=values, span=1, step=1)
    bw.close()


def test_chr1_length_from_chrom_sizes_accepts_chr1_and_bare_1():
    assert chr1_length_from_chrom_sizes({"chr1": 248_956_422, "chr2": 1}) == 248_956_422
    assert chr1_length_from_chrom_sizes({"1": 248_956_422, "2": 1}) == 248_956_422
    assert chr1_length_from_chrom_sizes({"chr2": 1}) is None


def test_detect_assembly_from_chrom_sizes_matches_known_builds():
    for build, length in ASSEMBLY_CHR1_LENGTH.items():
        assert detect_assembly_from_chrom_sizes({"chr1": length}) == build


def test_detect_assembly_from_chrom_sizes_unrecognized_length_returns_none():
    assert detect_assembly_from_chrom_sizes({"chr1": 12345}) is None


def test_require_assembly_from_chrom_sizes_raises_on_confident_mismatch():
    mm10_chr1 = ASSEMBLY_CHR1_LENGTH["mm10"]
    with pytest.raises(GenomeAssemblyMismatchError, match="mm10"):
        require_assembly_from_chrom_sizes({"chr1": mm10_chr1}, "hg38")


def test_require_assembly_from_chrom_sizes_warns_and_returns_none_when_unrecognized(caplog):
    with caplog.at_level(logging.WARNING):
        result = require_assembly_from_chrom_sizes({"chr1": 999}, "hg38")
    assert result is None
    assert "unverified" in caplog.text.lower() or "could not identify" in caplog.text.lower()


def test_require_assembly_from_chrom_sizes_typo_in_expected_raises_value_error():
    with pytest.raises(ValueError, match="not an assembly"):
        require_assembly_from_chrom_sizes({"chr1": ASSEMBLY_CHR1_LENGTH["hg38"]}, "GRCh38")


def test_require_assembly_from_chrom_sizes_returns_found_on_match():
    hg38_chr1 = ASSEMBLY_CHR1_LENGTH["hg38"]
    assert require_assembly_from_chrom_sizes({"chr1": hg38_chr1}, "hg38") == "hg38"


def test_require_assembly_for_bigwig_matches_declared_build(tmp_path):
    bw_path = tmp_path / "matching.bw"
    _write_fixture_bigwig(bw_path, chrom_size=ASSEMBLY_CHR1_LENGTH["hg38"])
    assert require_assembly_for_bigwig(bw_path, "hg38") == "hg38"


def test_require_assembly_for_bigwig_raises_on_mismatch(tmp_path):
    bw_path = tmp_path / "mismatched.bw"
    _write_fixture_bigwig(bw_path, chrom_size=ASSEMBLY_CHR1_LENGTH["mm10"])
    with pytest.raises(GenomeAssemblyMismatchError):
        require_assembly_for_bigwig(bw_path, "hg38")
