"""Custom exceptions for the Chorus library."""


class ChorusError(Exception):
    """Base exception class for Chorus library."""
    pass


class ModelNotLoadedError(ChorusError):
    """Raised when trying to use a model that hasn't been loaded."""
    pass


class InvalidSequenceError(ChorusError, ValueError):
    """Raised when an invalid DNA sequence is provided.

    Inherits from ``ValueError`` as well so legacy ``except ValueError``
    handlers still catch it — v26 P2 #19.
    """
    pass


class InvalidAssayError(ChorusError, ValueError):
    """Raised when an invalid assay type is requested.

    Inherits from ``ValueError`` as well — v26 P2 #19.
    """
    pass


class InvalidRegionError(ChorusError, ValueError):
    """Raised when an invalid genomic region is specified.

    Inherits from ``ValueError`` as well so the MCP helpers
    ``_parse_region`` / ``_parse_position`` keep their
    ``except ValueError`` contract while also being catchable as
    ``ChorusError`` — v26 P2 #19.
    """
    pass


class ReferenceAlleleMismatchError(ChorusError, ValueError):
    """The supplied reference allele is not what the genome has at that position.

    Almost always wrong coordinates or the wrong genome build. Chorus used to
    only *warn* and then substitute the supplied allele into the prediction
    interval, which means it scored a **synthetic, non-reference sequence** and
    reported the result as if nothing were amiss.

    That is not hypothetical: a committed BCL11A example carried ``ref="G"``
    where hg38 has ``T`` and shipped that way for months, with the warning firing
    on every single run. 1 of 4 examples checked was wrong.

    Inherits ``ValueError`` as well, matching ``InvalidRegionError``, so callers
    with an ``except ValueError`` contract keep working.
    """
    pass


class GenomeAssemblyMismatchError(ChorusError, ValueError):
    """A reference FASTA is not the assembly the model was trained on.

    Every shipped oracle is human hg38, and that is currently enforced *by
    accident*: Enformer and Borzoi are human-only because someone selected
    ``*_human_targets.txt``, and AlphaGenome because ``Organism.HOMO_SAPIENS``
    is hardcoded in its metadata loader. Nothing connected any of those choices
    to the FASTA the builders open, so nothing would have caught a future
    ``*_mouse_targets.txt`` — and the one registry with no organism field at all
    (ChromBPNet's hand-written accession dict) is exactly where it did go wrong:
    33 mm10 models were scored against hg38 sequence using the hg38 DHS
    vocabulary before #121 removed them.

    Raised rather than warned because the output of scoring the wrong assembly
    is not an error, it is a plausible number: mm10 chr1:1,000,000 exists in
    hg38 too, so every coordinate resolves, every prediction returns, and the
    only symptom is that the answer is about a different piece of DNA.

    Inherits ``ValueError`` as well, matching :class:`InvalidRegionError`.
    """
    pass


class FileFormatError(ChorusError):
    """Raised when a file format is invalid or unsupported."""
    pass


class EmptyPredictionsError(ChorusError):
    """Raised when a report builder receives an empty predictions dict.

    Common cause: passing an ``assay_ids`` list to ``predict_variant_effect``
    that didn't match any tracks on the oracle, so the predict loop
    returned an empty dict. The message includes a hint to check
    ``oracle.get_all_assay_ids()``.
    """
    pass


class EnvironmentNotReadyError(ChorusError):
    """Raised when an oracle's conda env setup failed and a later API
    call (predict / load_pretrained_model / etc.) would otherwise run
    against a half-initialized oracle.

    Previously :meth:`OracleBase._setup_environment` would log a warning
    on failure, flip ``use_environment`` to False, and return — letting
    the user hit confusing downstream errors ("No module named
    'tensorflow'" inside the base chorus env). v26 P1 #11 wants the
    oracle to remember the failure and raise this on next use with an
    actionable pointer to ``chorus setup`` or ``chorus health``.
    """
    pass