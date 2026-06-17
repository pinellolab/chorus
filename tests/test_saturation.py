"""Smoke tests for in-silico saturation mutagenesis (chorus.analysis.saturation).

Model-free: a tiny FASTA + a mock oracle whose effect depends on the substituted
base, so the importance profile and matrix shape are deterministic and assertable
in CI (no GPU / no model weights).
"""
import pytest

from chorus.analysis.saturation import saturation_mutagenesis

BASES = "ACGT"


def test_saturation_mutagenesis_shapes_and_importance(tmp_path, monkeypatch):
    # Tiny reference genome
    fa = tmp_path / "mini.fa"
    seq = "ACGT" * 10  # 40 bp
    fa.write_text(">chr1\n" + seq + "\n")

    class MockOracle:
        def predict_variant_effect(self, region, position, alleles, assay_ids=None, genome=None):
            # Carry the alt base through so the patched scorer can use it.
            return {"predictions": {"reference": {}, "alternate_1": {}},
                    "variant_info": {"position": position}, "_alt": alleles[1]}

    import chorus.analysis.discovery as disc
    from chorus.analysis.discovery import TrackEffect

    def fake_score(variant_result, oracle_name, **kw):
        # Every substitution disrupts the site equally (-1.0), so importance
        # (= -mean over the three substitutions) is +1.0 at every position.
        return [TrackEffect(assay_id="x", layer="tf_binding", cell_type="c",
                            description="t", raw_score=-1.0, abs_score=1.0,
                            ref_value=1.0, alt_value=1.0)]

    monkeypatch.setattr(disc, "_score_all_tracks", fake_score)

    res = saturation_mutagenesis(MockOracle(), "mock", "chr1:20", ["x"],
                                 genome=str(fa), window=11)

    assert len(res["ref_seq"]) == 11
    assert len(res["scores"]) == 11 and all(len(r) == 4 for r in res["scores"])
    assert len(res["importance"]) == 11
    assert res["window"] == 11
    # The reference base at each position is left at 0 (we only score substitutions).
    for i, b in enumerate(res["ref_seq"]):
        assert res["scores"][i][BASES.index(b)] == 0.0
    # importance = -mean(effect over the three substitutions) = +1.0 everywhere.
    assert all(v == pytest.approx(1.0) for v in res["importance"])
