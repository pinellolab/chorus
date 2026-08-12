"""Human-only must be a check, not a coincidence (#124).

Chorus ships nine hg38 oracles and no mouse ones, and until this file existed that was
true **by accident**. Measured across the shipped backgrounds:

    Enformer      1,643 upstream mouse tracks excluded by enformer_human_targets.txt
    Borzoi        2,608                        excluded by borzoi_human_targets.txt
    AlphaGenome   Organism.MUS_MUSCULUS        excluded by a hardcoded HOMO_SAPIENS
    ChromBPNet    the mouse developmental atlas -- excluded by NOTHING

The first three are human because somebody picked a human-only metadata file. A file
choice is not an assertion: nothing connected it to ``genomes/hg38.fa``, which every
builder opens, so nothing would have caught a future ``*_mouse_targets.txt``. ChromBPNet
is the case where it went wrong -- a hand-written accession dict with no organism field,
so 33 mm10 models were scored against hg38 sequence using the hg38 DHS vocabulary, and
#121 removed them.

The reason this is a hard failure and not a warning: mm10 chr1:1,000,000 exists in hg38
too. Every coordinate resolves, every prediction returns, every percentile lands in
[0, 1]. There is no symptom. The answer is simply about a different piece of DNA.

Four mechanisms, each pinned below:

  * oracles **declare** ``training_genome`` on their own class, so there is something to
    check a FASTA against, and a new oracle cannot inherit "hg38" by saying nothing;
  * builders **check** it against the reference they open, before a 14-hour job starts;
  * the stamp scripts **observe** the assembly instead of restating a filename, so the
    ``build_config.genome`` field is evidence rather than an assumption;
  * the loader **refuses** an artefact that declares a genome chorus does not rank
    against, on the ``BackgroundArtefactMismatch`` contract -- wrong numbers are not a
    degradation to fall back from.

The enumeration tests are the load-bearing ones. Wiring a guard into seven builders and
missing the eighth is exactly how the three duplicated IGV render paths shipped with only
one patched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
BUILDERS = sorted(SCRIPTS.glob("build_backgrounds_*.py"))


# ──────────────────────────────────────────────────────────────────────
# The declaration
# ──────────────────────────────────────────────────────────────────────

def _oracle_classes():
    """Every concrete oracle class, imported from the base env."""
    import importlib

    from chorus.core.base import OracleBase

    found = []
    for path in sorted((REPO / "chorus" / "oracles").glob("*.py")):
        if path.name.startswith("_"):
            continue
        mod = importlib.import_module(f"chorus.oracles.{path.stem}")
        for name in dir(mod):
            obj = getattr(mod, name)
            if (isinstance(obj, type) and issubclass(obj, OracleBase)
                    and obj is not OracleBase and obj.__module__ == mod.__name__):
                found.append(obj)
    assert len(found) >= 9, f"expected at least the 9 shipped oracles, found {found}"
    return found


def test_every_oracle_declares_its_training_genome_on_its_own_class():
    """Inherited would defeat the purpose: a mouse oracle must not inherit "hg38"."""
    from chorus.utils.genome import ASSEMBLY_CHR1_LENGTH

    for cls in _oracle_classes():
        assert "training_genome" in cls.__dict__, (
            f"{cls.__name__} does not declare training_genome on its own class. "
            f"Inheriting the base default would reproduce #124 exactly -- chorus is "
            f"human-only because of a metadata file choice, and an inherited 'hg38' is "
            f"another such choice wearing an assertion's clothes."
        )
        declared = cls.__dict__["training_genome"]
        assert declared in ASSEMBLY_CHR1_LENGTH, (
            f"{cls.__name__}.training_genome={declared!r} is not an assembly chorus can "
            f"identify from a FASTA, so require_reference_assembly could not check it"
        )


def test_the_base_class_default_is_none_so_silence_is_not_an_answer():
    from chorus.core.base import OracleBase

    assert OracleBase.training_genome is None


# ──────────────────────────────────────────────────────────────────────
# The check, and that every builder performs it
# ──────────────────────────────────────────────────────────────────────

def test_every_builder_checks_the_reference_before_opening_it():
    """The enumeration guard. Eight builders, ten FASTA opens, no exceptions."""
    assert len(BUILDERS) == 8, f"builder set changed: {[p.name for p in BUILDERS]}"
    offenders = []
    n_sites = 0
    for path in BUILDERS:
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            if "pysam.FastaFile(" not in line:
                continue
            n_sites += 1
            window = "\n".join(lines[max(0, i - 12):i])
            if "require_reference_assembly(" not in window:
                offenders.append(f"{path.name}:{i + 1}  {line.strip()}")
    assert n_sites >= 10, f"only found {n_sites} FASTA opens; did the builders change?"
    assert not offenders, (
        "these builders open a reference FASTA without checking it is the assembly their "
        f"model was trained on: {offenders}. Call "
        "chorus.analysis.background_sampling.require_reference_assembly(path, Oracle, "
        "label=...) first -- a 14-hour build should fail in its first second, and mm10 "
        "coordinates resolve against hg38 without complaint."
    )


def test_a_recognised_but_wrong_assembly_raises():
    from chorus.core.exceptions import GenomeAssemblyMismatchError
    from chorus.utils.genome import require_assembly

    fasta = REPO / "genomes" / "hg38.fa"
    if not fasta.exists():
        pytest.skip("hg38 reference not present")

    assert require_assembly(fasta, "hg38") == "hg38"
    with pytest.raises(GenomeAssemblyMismatchError) as exc:
        require_assembly(fasta, "mm10", context="unit")
    msg = str(exc.value)
    assert "hg38" in msg and "mm10" in msg
    assert "195,471,971" in msg, "the message should show the lengths that disagree"


def test_an_unknown_expected_assembly_raises_rather_than_disabling_the_check():
    """``expected="GRCh38"`` is a typo, and a typo must not silently pass everything."""
    from chorus.utils.genome import require_assembly

    with pytest.raises(ValueError, match="not an assembly chorus can identify"):
        require_assembly(REPO / "genomes" / "hg38.fa", "GRCh38")


def test_an_unidentifiable_reference_warns_rather_than_blocking(tmp_path, caplog):
    """No chr1 means no claim -- refusing would break legitimate custom references."""
    from chorus.utils.genome import detect_assembly, require_assembly

    fake = tmp_path / "custom.fa"
    fake.write_text(">contig_1\nACGT\n")
    assert detect_assembly(fake) is None
    with caplog.at_level("WARNING"):
        assert require_assembly(fake, "hg38", context="unit") is None
    assert "Could not identify" in caplog.text


def test_an_oracle_that_declares_nothing_is_refused():
    """"Nobody said" is the state #124 is about, so the guard must not tolerate it."""
    from chorus.analysis.background_sampling import require_reference_assembly

    class Undeclared:
        training_genome = None

    with pytest.raises(ValueError, match="does not declare training_genome"):
        require_reference_assembly(REPO / "genomes" / "hg38.fa", Undeclared,
                                   label="unit")


def test_hg38_and_hg19_are_distinguishable_at_all():
    """The whole approach rests on chr1 lengths differing, so state the margins."""
    from chorus.utils.genome import ASSEMBLY_CHR1_LENGTH

    lengths = ASSEMBLY_CHR1_LENGTH
    assert len(set(lengths.values())) == len(lengths), "two assemblies share a chr1 length"
    assert lengths["hg19"] - lengths["hg38"] == 294_199
    assert lengths["mm10"] - lengths["mm39"] == 317_692


# ──────────────────────────────────────────────────────────────────────
# The stamp: observation, not restatement
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("script", ["stamp_provenance_v4.py",
                                    "stamp_background_provenance.py"])
def test_the_stamper_measures_the_genome_instead_of_asserting_it(script):
    """A hardcoded ``genome`` value would make the loader compare two constants.

    Parsed rather than grepped: this file and the stampers both *discuss* the old literal,
    and a text search cannot tell prose from code.
    """
    import ast

    src = (SCRIPTS / script).read_text()
    hardcoded = [
        f"line {node.lineno}: {ast.unparse(value)}"
        for node in ast.walk(ast.parse(src)) if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "genome"
        and isinstance(value, ast.Constant) and isinstance(value.value, str)
    ]
    assert not hardcoded, (
        f"{script} states the assembly as a literal ({hardcoded}). The loader now refuses "
        f"an artefact whose declared genome is not the one chorus ranks against, and that "
        f"check is worthless if the declaration is this script's assumption rather than "
        f"something read off the FASTA. Use chorus.utils.genome.detect_assembly()."
    )
    assert "detect_assembly" in src


# ──────────────────────────────────────────────────────────────────────
# The refusal, at load
# ──────────────────────────────────────────────────────────────────────

def _write_npz(path: Path, config: dict | None) -> None:
    arrays = {
        "track_ids": np.array(["t0", "t1"]),
        "effect_cdfs": np.zeros((2, 8), dtype=np.float32),
        "effect_counts": np.array([8, 8]),
    }
    if config is not None:
        arrays["build_config"] = np.array([json.dumps(config)])
    np.savez(path, **arrays)


def test_an_artefact_from_another_genome_is_refused_not_ranked_against(tmp_path):
    from chorus.analysis.normalization import (
        BackgroundGenomeMismatch,
        PerTrackNormalizer,
        RANKING_GENOME,
    )

    _write_npz(tmp_path / "mouseoracle_pertrack.npz",
               {"schema_version": 4, "oracle": "mouseoracle", "genome": "mm10"})
    with pytest.raises(BackgroundGenomeMismatch) as exc:
        PerTrackNormalizer(cache_dir=str(tmp_path))._ensure_loaded("mouseoracle")
    msg = str(exc.value)
    assert "mm10" in msg and RANKING_GENOME in msg


def test_a_refused_artefact_is_not_left_cached_as_usable(tmp_path):
    from chorus.analysis.normalization import BackgroundGenomeMismatch, PerTrackNormalizer

    _write_npz(tmp_path / "mouseoracle_pertrack.npz",
               {"schema_version": 4, "genome": "mm10"})
    norm = PerTrackNormalizer(cache_dir=str(tmp_path))
    with pytest.raises(BackgroundGenomeMismatch):
        norm._ensure_loaded("mouseoracle")
    assert "mouseoracle" not in norm._loaded, (
        "the entry was cached before the guard ran, so the second call would succeed "
        "and rank against it"
    )


def test_an_unstamped_artefact_makes_no_claim_and_still_loads(tmp_path):
    """Every pre-provenance background has no genome key; refusing them helps nobody."""
    from chorus.analysis.normalization import PerTrackNormalizer

    _write_npz(tmp_path / "oldoracle_pertrack.npz", None)
    entry = PerTrackNormalizer(cache_dir=str(tmp_path))._ensure_loaded("oldoracle")
    assert entry is not None and entry["build_config"] is None


def test_the_genome_refusal_is_not_swallowed_by_the_legacy_fallback(tmp_path):
    """The lesson BackgroundFoldMismatch was added with, inherited by the base class.

    ``get_normalizer`` absorbs load failures and drops to the legacy ``.npy`` scan, which
    is right for "no percentiles available" and catastrophic for "the percentiles would be
    wrong": the caller gets numbers, from a different reference class, with no error.
    """
    from chorus.analysis.normalization import (
        BackgroundArtefactMismatch,
        BackgroundFoldMismatch,
        BackgroundGenomeMismatch,
        get_normalizer,
    )

    assert issubclass(BackgroundGenomeMismatch, BackgroundArtefactMismatch)
    assert issubclass(BackgroundFoldMismatch, BackgroundArtefactMismatch)

    _write_npz(tmp_path / "mouseoracle_pertrack.npz",
               {"schema_version": 4, "genome": "mm39"})
    with pytest.raises(BackgroundArtefactMismatch):
        get_normalizer("mouseoracle", cache_dir=str(tmp_path))


def test_get_normalizer_reraises_by_base_class_not_by_listing_subclasses():
    """Source check: the next guard in the family must inherit the contract for free."""
    src = (REPO / "chorus" / "analysis" / "normalization.py").read_text()
    body = src[src.index("def get_normalizer("):]
    body = body[:body.index("\n# ") if "\n# " in body else len(body)]
    assert "except BackgroundArtefactMismatch:" in body, (
        "get_normalizer should re-raise the whole BackgroundArtefactMismatch family. "
        "Catching individual subclasses means the next one added is silently swallowed, "
        "and the symptom is a plausible percentile from the wrong null."
    )


def test_every_shipped_artefact_declares_the_genome_it_was_built_on():
    from chorus.analysis.normalization import (
        CHORUS_BACKGROUNDS_DIR,
        PerTrackNormalizer,
        RANKING_GENOME,
    )

    paths = sorted(Path(CHORUS_BACKGROUNDS_DIR).glob("*_pertrack.npz"))
    if not paths:
        pytest.skip("no backgrounds downloaded")
    for path in paths:
        with np.load(str(path), allow_pickle=True) as z:
            assert "build_config" in z.files, f"{path.name} has no provenance at all"
            config = PerTrackNormalizer._read_build_config(z["build_config"], path.name)
        assert config is not None, f"{path.name} build_config would not parse"
        assert config.get("genome") == RANKING_GENOME, (
            f"{path.name} declares genome {config.get('genome')!r}; chorus ranks against "
            f"{RANKING_GENOME!r} and the loader will refuse it"
        )


# ──────────────────────────────────────────────────────────────────────
# The inert switch
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module,cls_name", [
    ("chorus.oracles.alphagenome", "AlphaGenomeOracle"),
    ("chorus.oracles.alphagenome_pt", "AlphaGenomePTOracle"),
])
def test_a_non_human_organism_raises_rather_than_being_stored_and_ignored(module, cls_name):
    """It used to return human predictions under a mouse label.

    ``organism`` was assigned to ``self.organism`` and read by nothing: the metadata loader
    hardcodes ``Organism.HOMO_SAPIENS`` and the PyTorch forward pass passes
    ``organism_index=0``. Of make-it-work / remove-it / raise, only raising is both honest
    and affordable -- mouse needs an mm10 reference, an mm10 reference class for the null,
    and a background pass over ~4,300 tracks.
    """
    import importlib

    cls = getattr(importlib.import_module(module), cls_name)
    assert cls(use_environment=False).organism == "human"
    for spelling in ("Human", "homo_sapiens"):
        assert cls(use_environment=False, organism=spelling).organism == "human"
    for bad in ("mouse", "Mus musculus", "mm10"):
        with pytest.raises(NotImplementedError) as exc:
            cls(use_environment=False, organism=bad)
        assert "#124" in str(exc.value), "point the caller at the reason it is unsupported"
