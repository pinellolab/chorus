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

    def test_variant_position_is_1_based(self, caplog):
        """Ref-allele check must treat `variant_position='chrN:P'` as 1-based.

        Builds a chr1 genome where position 100,000 (1-based) is 'G' and the
        two neighbouring positions are 'A'. The oracle must read 'G' at
        chr1:100000 and not fire the "does not match the genome" warning.
        Regression for the off-by-one that returned the base at 1-based
        P+1 (so rs12740374 returned 'T' instead of 'G').
        """
        import logging
        # Build a custom genome: 'A' everywhere except a 'G' anchor
        tmp = Path(tempfile.mkdtemp())
        fa = tmp / "anchor.fa"
        # 1-based position 100000 = 0-based 99999.
        # Pad with A, put G at 99999, A elsewhere, over 200 kb total.
        seq = ['A'] * 200_000
        seq[99_999] = 'G'  # 1-based pos 100000
        with open(fa, "w") as fh:
            fh.write(">chr1\n" + "".join(seq) + "\n")
        import pysam
        pysam.faidx(str(fa))

        oracle = MockOracle(reference_fasta=str(fa))

        # Provide ref='G' matching the genome — warning must NOT fire.
        with caplog.at_level(logging.WARNING, logger="chorus.core.base"):
            oracle.predict_variant_effect(
                genomic_region="chr1:50000-150000",
                variant_position="chr1:100000",
                alleles=["G", "A"],
                assay_ids=["DNase:K562"],
            )
        matching = [r for r in caplog.records if "does not match the genome" in r.getMessage()]
        assert not matching, (
            f"Warning fired unexpectedly — ref-allele check is off-by-one again. "
            f"Messages: {[r.getMessage() for r in matching]}"
        )

        # And with the WRONG ref — warning MUST fire (proves the check still works).
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="chorus.core.base"):
            oracle.predict_variant_effect(
                genomic_region="chr1:50000-150000",
                variant_position="chr1:100000",
                alleles=["T", "A"],  # genome has G, user says T → mismatch
                assay_ids=["DNase:K562"],
            )
        matching = [r for r in caplog.records if "does not match the genome" in r.getMessage()]
        assert matching, "Warning should fire when user's ref really doesn't match genome"

        shutil.rmtree(tmp)

    def test_bad_chromosome_gives_actionable_error(self):
        """A chromosome not in the reference FASTA must fail with a
        message that names the bad chrom and the FASTA path — not a
        low-level pysam.KeyError or a downstream one-hot-encoder
        KeyError('H').

        Regression for v20 §14.4 finding:
            oracle.predict(('chrZZ', 100, 300), [...]) used to crash
            deep in LegNet's transforms with KeyError: 'H'.

        MockOracle._predict shortcircuits the input to random data so
        we exercise the chokepoint (GenomeRef.slop → pysam) directly
        plus the predict_variant_effect path which does go through
        real region_interval[...] indexing.
        """
        from chorus.core.interval import GenomeRef, IntervalException
        from chorus.core.exceptions import InvalidRegionError

        # Path A: GenomeRef.slop — the actual crash site before the fix
        gr = GenomeRef(chrom="chrZZ", start=100, end=300,
                       fasta=str(self.fasta_path))
        with pytest.raises(IntervalException, match="Chromosome 'chrZZ' not found"):
            gr.slop(extension_needed=1000, how="both")

        # Path B: predict_variant_effect(string) — goes through
        # extract_sequence → raises InvalidRegionError
        with pytest.raises(InvalidRegionError, match="[Cc]hromosome.*chrZZ.*not found"):
            self.oracle.predict_variant_effect(
                genomic_region="chrZZ:100-300",
                variant_position="chrZZ:150",
                alleles=["A", "C"],
                assay_ids=["DNase:K562"],
            )

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
