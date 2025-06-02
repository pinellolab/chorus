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