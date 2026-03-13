"""Tests for the three main prediction methods in Chorus."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from chorus.core.base import OracleBase
from chorus.core.result import OraclePrediction, OraclePredictionTrack
from chorus.core.interval import Interval, Sequence
from chorus.core.exceptions import ModelNotLoadedError


class MockOracle(OracleBase):
    """Mock oracle for testing base prediction methods.

    Returns deterministic OraclePrediction objects from _predict().
    """

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
        return ["DNase", "RNA-seq", "ChIP-seq"]

    def list_cell_types(self):
        return ["K562", "HepG2", "GM12878"]

    def _predict(self, seq, assay_ids=None):
        """Return OraclePrediction with random tracks."""
        num_bins = self._output_size // self._bin_size  # 896

        if assay_ids is None:
            assay_ids = ["DNase:K562"]

        # Build a seed from the input for reproducibility
        if isinstance(seq, str):
            seed = len(seq) % (2**31)
            query_interval = Interval.make(Sequence(sequence=seq))
        elif isinstance(seq, Interval):
            seed = len(seq.sequence) % (2**31)
            query_interval = seq
        else:
            seed = 42
            query_interval = Interval.make(Sequence(sequence="A" * 1000))

        np.random.seed(seed)

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


class TestPredictionMethods:
    """Test suite for prediction methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.fasta_path = Path(self.temp_dir) / "test_genome.fa"

        # Write a simple test genome
        with open(self.fasta_path, "w") as f:
            f.write(">chr1\n")
            f.write("A" * 500000 + "\n")
            f.write(">chr2\n")
            f.write("T" * 300000 + "\n")

        import pysam
        pysam.faidx(str(self.fasta_path))

        self.oracle = MockOracle(reference_fasta=str(self.fasta_path))

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_predict_with_sequence(self):
        """Test prediction with a raw sequence string."""
        test_seq = "ACGT" * 1000  # 4kb

        results = self.oracle.predict(
            input_data=test_seq,
            assay_ids=["DNase:K562", "RNA-seq:HepG2"],
        )

        # predict() returns OraclePrediction
        assert isinstance(results, OraclePrediction)
        assert results["DNase:K562"].values.shape == (896,)
        assert results["RNA-seq:HepG2"].values.shape == (896,)

    def test_predict_region_replacement(self):
        """Test region replacement."""
        region = "chr1:100000-102000"
        new_seq = "ACGT" * 500  # 2kb replacement

        results = self.oracle.predict_region_replacement(
            genomic_region=region,
            seq=new_seq,
            assay_ids=["DNase:K562"],
        )

        assert "raw_predictions" in results
        assert "normalized_scores" in results
        assert isinstance(results["raw_predictions"], OraclePrediction)
        assert results["raw_predictions"]["DNase:K562"].values.shape == (896,)

    def test_predict_region_insertion_at(self):
        """Test sequence insertion at position."""
        insert_seq = "ACGT" * 250  # 1kb

        results = self.oracle.predict_region_insertion_at(
            genomic_position="chr1:250000",
            seq=insert_seq,
            assay_ids=["DNase:K562", "RNA-seq:HepG2"],
        )

        assert "raw_predictions" in results
        assert "normalized_scores" in results
        assert isinstance(results["raw_predictions"], OraclePrediction)
        assert results["raw_predictions"]["DNase:K562"].values.shape == (896,)
        assert results["raw_predictions"]["RNA-seq:HepG2"].values.shape == (896,)

    def test_predict_variant_effect_snp(self):
        """Test variant effect prediction for SNP."""
        results = self.oracle.predict_variant_effect(
            genomic_region="chr1:200000-300000",
            variant_position="chr1:250000",
            alleles=["A", "C"],
            assay_ids=["DNase:K562"],
        )

        assert "predictions" in results
        assert "effect_sizes" in results
        assert "variant_info" in results
        assert "reference" in results["predictions"]
        assert "alt_1" in results["predictions"]
        assert "alt_1" in results["effect_sizes"]

    def test_predict_variant_effect_multiallelic(self):
        """Test variant effect prediction for multi-allelic variant."""
        results = self.oracle.predict_variant_effect(
            genomic_region="chr1:200000-300000",
            variant_position="chr1:250000",
            alleles=["A", "C", "G", "T"],
            assay_ids=["DNase:K562"],
        )

        assert "reference" in results["predictions"]
        assert "alt_1" in results["predictions"]
        assert "alt_2" in results["predictions"]
        assert "alt_3" in results["predictions"]
        assert len(results["effect_sizes"]) == 3

    def test_error_handling_model_not_loaded(self):
        """Test error when model not loaded."""
        unloaded_oracle = MockOracle(reference_fasta=str(self.fasta_path))
        unloaded_oracle.loaded = False

        with pytest.raises(ModelNotLoadedError):
            unloaded_oracle.predict(
                input_data="ACGT" * 1000,
                assay_ids=["DNase:K562"],
            )

    def test_error_handling_no_genome(self):
        """Test error when reference genome not set."""
        oracle_no_ref = MockOracle()  # no reference_fasta

        with pytest.raises(ValueError, match="No reference genome"):
            oracle_no_ref.predict_region_replacement(
                genomic_region="chr1:1000-2000",
                seq="ACGT" * 250,
                assay_ids=["DNase:K562"],
            )


if __name__ == "__main__":
    pytest.main([__file__])
