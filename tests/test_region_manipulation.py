"""
Unit tests for predict_region_replacement and predict_region_insertion_at.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from chorus.core.base import OracleBase
from chorus.core.result import OraclePrediction, OraclePredictionTrack
from chorus.core.interval import Interval, Sequence
from chorus.core.exceptions import (
    InvalidRegionError,
    InvalidSequenceError,
    ModelNotLoadedError,
)


class MockOracle(OracleBase):
    """Mock oracle for testing region manipulation methods."""

    def __init__(self, reference_fasta=None):
        self.oracle_name = "test"
        super().__init__(use_environment=False)
        self.loaded = True
        self._context_size = 393216
        self._output_size = 114688
        self._bin_size = 128
        self.reference_fasta = reference_fasta

    def _get_context_size(self):
        return self._context_size

    def _get_bin_size(self):
        return self._bin_size

    def _get_sequence_length_bounds(self):
        return (1000, self._context_size)

    def load_pretrained_model(self, weights=None):
        self.loaded = True

    def list_assay_types(self):
        return ["DNase", "RNA-seq", "ChIP-seq", "ATAC-seq"]

    def list_cell_types(self):
        return ["K562", "HepG2", "GM12878"]

    def _predict(self, seq, assay_ids=None):
        """Return OraclePrediction with deterministic data based on input."""
        num_bins = self._output_size // self._bin_size  # 896

        if assay_ids is None:
            assay_ids = ["DNase:K562"]

        # Build deterministic seed from input
        if isinstance(seq, Interval):
            raw = seq.sequence
        elif isinstance(seq, str):
            raw = seq
        else:
            raw = "A" * 1000

        seed = sum(ord(c) for c in raw[:1000]) % (2**31)
        np.random.seed(seed)

        # Build query interval
        if isinstance(seq, Interval):
            query_interval = seq
        else:
            query_interval = Interval.make(Sequence(sequence=raw))

        prediction = OraclePrediction()
        for i, assay_id in enumerate(assay_ids):
            parts = assay_id.split(":")
            assay_type = parts[0] if parts else "UNKNOWN"
            cell_type = parts[1] if len(parts) > 1 else "UNKNOWN"

            values = np.random.rand(num_bins).astype(np.float32)

            track = OraclePredictionTrack.create(
                source_model="mock",
                assay_id=assay_id,
                track_id=i,
                assay_type=assay_type,
                cell_type=cell_type,
                query_interval=query_interval,
                prediction_interval=query_interval,
                input_interval=query_interval,
                resolution=self._bin_size,
                values=values,
            )
            prediction.add(assay_id, track)

        return prediction

    def fine_tune(self, tracks, track_names, **kwargs):
        pass

    @property
    def output_size(self):
        return self._output_size


class TestRegionReplacement:
    """Test predict_region_replacement functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.fasta_path = Path(self.temp_dir) / "test_genome.fa"

        with open(self.fasta_path, "w") as f:
            f.write(">chr1\n")
            f.write("A" * 1000000 + "\n")  # 1Mb chromosome
            f.write(">chr8\n")
            f.write("T" * 1000000 + "\n")

        import pysam
        pysam.faidx(str(self.fasta_path))

        self.oracle = MockOracle(reference_fasta=str(self.fasta_path))

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_region_replacement_basic(self):
        """Test basic region replacement with string input."""
        region = "chr1:499000-501000"
        new_seq = "ACGT" * 500  # 2kb

        results = self.oracle.predict_region_replacement(
            genomic_region=region,
            seq=new_seq,
            assay_ids=["DNase:K562", "RNA-seq:HepG2"],
        )

        assert "raw_predictions" in results
        assert "normalized_scores" in results

        raw = results["raw_predictions"]
        assert isinstance(raw, OraclePrediction)
        assert raw["DNase:K562"].values.shape == (896,)
        assert raw["RNA-seq:HepG2"].values.shape == (896,)

    def test_region_replacement_with_dataframe(self):
        """Test region replacement with DataFrame input."""
        region_df = pd.DataFrame({
            "chrom": ["chr8"],
            "start": [400000],
            "end": [402000],
        })

        new_seq = "TGCA" * 500  # 2kb

        results = self.oracle.predict_region_replacement(
            genomic_region=region_df,
            seq=new_seq,
            assay_ids=["DNase:K562"],
        )

        assert results["raw_predictions"]["DNase:K562"].values.shape == (896,)

    def test_region_replacement_invalid_region(self):
        """Test error handling for invalid region format."""
        with pytest.raises(Exception):
            self.oracle.predict_region_replacement(
                genomic_region="invalid_region_format",
                seq="ACGT" * 1000,
                assay_ids=["DNase:K562"],
            )

    def test_region_replacement_different_sequences_different_results(self):
        """Test that different sequences produce different predictions."""
        region = "chr1:400000-402000"

        seq1 = "A" * 2000
        seq2 = "T" * 2000

        results1 = self.oracle.predict_region_replacement(
            genomic_region=region,
            seq=seq1,
            assay_ids=["DNase:K562"],
        )

        results2 = self.oracle.predict_region_replacement(
            genomic_region=region,
            seq=seq2,
            assay_ids=["DNase:K562"],
        )

        # Different replacement sequences should give different predictions
        assert not np.array_equal(
            results1["raw_predictions"]["DNase:K562"].values,
            results2["raw_predictions"]["DNase:K562"].values,
        )

    def test_region_replacement_no_genome(self):
        """Test error when no reference genome is provided."""
        oracle_no_ref = MockOracle()

        with pytest.raises(ValueError, match="No reference genome"):
            oracle_no_ref.predict_region_replacement(
                genomic_region="chr1:1000-2000",
                seq="ACGT" * 250,
                assay_ids=["DNase:K562"],
            )


class TestSequenceInsertion:
    """Test predict_region_insertion_at functionality."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fasta_path = Path(self.temp_dir) / "test_genome.fa"

        with open(self.fasta_path, "w") as f:
            f.write(">chr1\n")
            f.write("A" * 1000000 + "\n")
            f.write(">chr8\n")
            f.write("T" * 1000000 + "\n")

        import pysam
        pysam.faidx(str(self.fasta_path))

        self.oracle = MockOracle(reference_fasta=str(self.fasta_path))

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_insertion_basic(self):
        """Test basic sequence insertion."""
        insert_seq = "ACGT" * 250  # 1kb
        position = "chr1:500000"

        results = self.oracle.predict_region_insertion_at(
            genomic_position=position,
            seq=insert_seq,
            assay_ids=["DNase:K562", "ChIP-seq:GM12878"],
        )

        assert "raw_predictions" in results
        raw = results["raw_predictions"]
        assert isinstance(raw, OraclePrediction)
        assert raw["DNase:K562"].values.shape == (896,)
        assert raw["ChIP-seq:GM12878"].values.shape == (896,)

    def test_insertion_empty_sequence(self):
        """Test insertion of empty sequence."""
        with pytest.raises(InvalidSequenceError):
            self.oracle.predict_region_insertion_at(
                genomic_position="chr1:500000",
                seq="",
                assay_ids=["DNase:K562"],
            )

    def test_insertion_affects_predictions(self):
        """Test that different insertions produce different results."""
        position = "chr1:500000"

        seq1 = "A" * 1000
        seq2 = "CACGTG" * 167  # E-box motifs

        results1 = self.oracle.predict_region_insertion_at(
            genomic_position=position,
            seq=seq1,
            assay_ids=["DNase:K562"],
        )

        results2 = self.oracle.predict_region_insertion_at(
            genomic_position=position,
            seq=seq2,
            assay_ids=["DNase:K562"],
        )

        assert not np.array_equal(
            results1["raw_predictions"]["DNase:K562"].values,
            results2["raw_predictions"]["DNase:K562"].values,
        )


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fasta_path = Path(self.temp_dir) / "test_genome.fa"

        with open(self.fasta_path, "w") as f:
            f.write(">chr1\n")
            f.write("A" * 500000 + "\n")

        import pysam
        pysam.faidx(str(self.fasta_path))

        self.oracle = MockOracle(reference_fasta=str(self.fasta_path))

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_model_not_loaded(self):
        """Test error when model not loaded."""
        self.oracle.loaded = False

        with pytest.raises(ModelNotLoadedError):
            self.oracle.predict_region_replacement(
                genomic_region="chr1:100000-102000",
                seq="ACGT" * 500,
                assay_ids=["DNase:K562"],
            )

    def test_invalid_region_format(self):
        """Test error with invalid region format."""
        with pytest.raises(InvalidRegionError):
            self.oracle.predict_region_replacement(
                genomic_region="bad_format",
                seq="ACGT" * 500,
                assay_ids=["DNase:K562"],
            )

    def test_invalid_dna_characters(self):
        """Test error with invalid DNA characters."""
        with pytest.raises(InvalidSequenceError):
            self.oracle.predict_region_replacement(
                genomic_region="chr1:100000-102000",
                seq="XYZQ" * 500,
                assay_ids=["DNase:K562"],
            )


if __name__ == "__main__":
    pytest.main([__file__])
