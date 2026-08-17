"""Domain exceptions.

Each carries the API_SPEC error code and HTTP status so the API layer can map
domain errors to responses without a translation table (ARCHITECTURE.md §1:
the API layer holds no business logic, only "error mapping").
"""

from __future__ import annotations


class PyVecError(Exception):
    """Base class. Maps to ``500 INTERNAL_ERROR`` unless overridden."""

    code = "INTERNAL_ERROR"
    status = 500

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.__class__.__name__)
        self.message = message or self.__class__.__name__


class InvalidDimensionError(PyVecError):
    code = "INVALID_DIMENSION"
    status = 400


class InvalidMetricError(PyVecError):
    code = "INVALID_METRIC"
    status = 400


class InvalidIndexTypeError(PyVecError):
    code = "INVALID_INDEX_TYPE"
    status = 400


class InvalidRequestError(PyVecError):
    """Well-formed JSON, but semantically wrong (e.g. empty batch)."""

    code = "INVALID_REQUEST"
    status = 400


class CollectionNotFoundError(PyVecError):
    code = "COLLECTION_NOT_FOUND"
    status = 404


class CollectionExistsError(PyVecError):
    code = "COLLECTION_EXISTS"
    status = 409


class IdNotFoundError(PyVecError):
    code = "ID_NOT_FOUND"
    status = 404


class IdExistsError(PyVecError):
    code = "ID_EXISTS"
    status = 409


class PayloadTooLargeError(PyVecError):
    code = "PAYLOAD_TOO_LARGE"
    status = 413


class NoTextFieldError(PyVecError):
    """Text or hybrid query against a collection with no ``text_field``."""

    code = "NO_TEXT_FIELD"
    status = 400


class CorruptDataError(PyVecError):
    """On-disk state failed a structural check that recovery cannot repair."""

    code = "CORRUPT_DATA"
    status = 500
