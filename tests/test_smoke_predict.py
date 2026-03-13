"""Smoke tests: instantiate each oracle, load a model, and run predict().

These tests require:
  - All 5 conda environments set up (chorus setup --oracle <name>)
  - Reference genome at genomes/hg38.fa
  - Pretrained model weights downloaded for chrombpnet, sei, and legnet

Run with: pytest tests/test_smoke_predict.py -v -s
(Use -s to see live output from environment subprocesses.)
"""

import pytest
import numpy as np
import chorus


REFERENCE_FASTA = "genomes/hg38.fa"

# Genomic regions on chr1 sized to each oracle's input window
REGIONS = {
    "chrombpnet":   ("chr1", 1_000_000, 1_002_114),
    "enformer":     ("chr1", 1_000_000, 1_393_216),
    "borzoi":       ("chr1", 1_000_000, 1_524_288),
    "sei":          ("chr1", 1_000_000, 1_004_096),
    "legnet":       ("chr1", 1_000_000, 1_002_048),
    "alphagenome":  ("chr1", 1_000_000, 2_048_576),
}


@pytest.fixture(scope="module")
def chrombpnet_oracle():
    oracle = chorus.create_oracle(
        "chrombpnet", use_environment=True, reference_fasta=REFERENCE_FASTA
    )
    oracle.load_pretrained_model(assay="ATAC", cell_type="K562")
    return oracle


@pytest.fixture(scope="module")
def enformer_oracle():
    oracle = chorus.create_oracle(
        "enformer", use_environment=True, reference_fasta=REFERENCE_FASTA
    )
    oracle.load_pretrained_model()
    return oracle


@pytest.fixture(scope="module")
def borzoi_oracle():
    oracle = chorus.create_oracle(
        "borzoi", use_environment=True, reference_fasta=REFERENCE_FASTA
    )
    oracle.load_pretrained_model()
    return oracle


@pytest.fixture(scope="module")
def sei_oracle():
    oracle = chorus.create_oracle(
        "sei", use_environment=True, reference_fasta=REFERENCE_FASTA
    )
    oracle.load_pretrained_model()
    return oracle


@pytest.fixture(scope="module")
def legnet_oracle():
    oracle = chorus.create_oracle(
        "legnet", use_environment=True, reference_fasta=REFERENCE_FASTA,
        cell_type="K562",
    )
    oracle.load_pretrained_model()
    return oracle


class TestSmokeChrombpnet:
    def test_predict(self, chrombpnet_oracle):
        result = chrombpnet_oracle.predict(REGIONS["chrombpnet"])
        tracks = dict(result.items())
        assert len(tracks) > 0, "No tracks returned"
        for name, track in tracks.items():
            assert track.values.shape[0] > 0, f"Empty values for {name}"
            assert np.isfinite(track.values).all(), f"Non-finite values in {name}"


class TestSmokeEnformer:
    def test_predict(self, enformer_oracle):
        result = enformer_oracle.predict(
            REGIONS["enformer"], assay_ids=["ENCFF413AHU"]
        )
        tracks = dict(result.items())
        assert len(tracks) == 1
        track = tracks["ENCFF413AHU"]
        assert track.values.shape == (896,)
        assert np.isfinite(track.values).all()


class TestSmokeBorzoi:
    def test_predict(self, borzoi_oracle):
        result = borzoi_oracle.predict(
            REGIONS["borzoi"], assay_ids=["ENCFF413AHU"]
        )
        tracks = dict(result.items())
        assert len(tracks) == 1
        track = tracks["ENCFF413AHU"]
        assert track.values.shape == (6144,)
        assert np.isfinite(track.values).all()


class TestSmokeSei:
    def test_predict(self, sei_oracle):
        result = sei_oracle.predict(
            REGIONS["sei"],
            assay_ids=["TA#HeLa_Epithelium_Cervix@BTAF1@ID:1"],
        )
        tracks = dict(result.items())
        assert len(tracks) == 1
        for name, track in tracks.items():
            assert track.values.shape[0] > 0, f"Empty values for {name}"
            assert np.isfinite(track.values).all(), f"Non-finite values in {name}"


class TestSmokeLegnet:
    def test_predict(self, legnet_oracle):
        result = legnet_oracle.predict(REGIONS["legnet"])
        tracks = dict(result.items())
        assert len(tracks) > 0, "No tracks returned"
        for name, track in tracks.items():
            assert track.values.shape[0] > 0, f"Empty values for {name}"
            assert np.isfinite(track.values).all(), f"Non-finite values in {name}"


@pytest.fixture(scope="module")
def alphagenome_oracle():
    oracle = chorus.create_oracle(
        "alphagenome", use_environment=True, reference_fasta=REFERENCE_FASTA
    )
    oracle.load_pretrained_model()
    return oracle


class TestSmokeAlphagenome:
    def test_predict(self, alphagenome_oracle):
        # Use the first available assay id from metadata
        from chorus.oracles.alphagenome_source.alphagenome_metadata import get_metadata
        metadata = get_metadata()
        assay_ids = list(metadata._track_index_map.keys())[:1]
        if not assay_ids:
            pytest.skip("No AlphaGenome track metadata available")
        result = alphagenome_oracle.predict(
            REGIONS["alphagenome"], assay_ids=assay_ids
        )
        tracks = dict(result.items())
        assert len(tracks) == 1
        for name, track in tracks.items():
            assert track.values.shape[0] > 0, f"Empty values for {name}"
            assert np.isfinite(track.values).all(), f"Non-finite values in {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
