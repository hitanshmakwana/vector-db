"""PyVec — a single-node vector database built from scratch in Python.

Implements HNSW and IVF-Flat approximate nearest neighbour indexes, BM25 sparse
retrieval, and Reciprocal Rank Fusion for hybrid search over a mmap-backed
store with an append-only write-ahead log.

Quick start::

    from pyvec import Collection

    c = Collection.create("docs", root="./data", dimension=4, metric="cosine",
                          index_type="hnsw", text_field="content")
    c.insert([{"id": "d1", "vector": [1, 0, 0, 0], "metadata": {"content": "hi"}}])
    c.search([1, 0, 0, 0], k=1)
"""

from pyvec.core.collection import Collection
from pyvec.core.collection_manager import CollectionManager
from pyvec.core.errors import (
    IdExistsError,
    IdNotFoundError,
    InvalidDimensionError,
    InvalidIndexTypeError,
    InvalidMetricError,
    NoTextFieldError,
    PayloadTooLargeError,
    PyVecError,
)
from pyvec.core.types import IndexType, Metric

__version__ = "0.1.0"

__all__ = [
    "Collection",
    "CollectionManager",
    "IndexType",
    "Metric",
    "PyVecError",
    "InvalidDimensionError",
    "InvalidMetricError",
    "InvalidIndexTypeError",
    "IdExistsError",
    "IdNotFoundError",
    "NoTextFieldError",
    "PayloadTooLargeError",
    "__version__",
]
