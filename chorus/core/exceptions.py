"""Custom exceptions for the Chorus library."""


class ChorusError(Exception):
    """Base exception class for Chorus library."""
    pass


class ModelNotLoadedError(ChorusError):
    """Raised when trying to use a model that hasn't been loaded."""
    pass


class InvalidSequenceError(ChorusError):
    """Raised when an invalid DNA sequence is provided."""
    pass


class InvalidAssayError(ChorusError):
    """Raised when an invalid assay type is requested."""
    pass


class InvalidRegionError(ChorusError):
    """Raised when an invalid genomic region is specified."""
    pass


class FileFormatError(ChorusError):
    """Raised when a file format is invalid or unsupported."""
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