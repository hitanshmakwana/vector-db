"""Index implementations: flat (brute force), HNSW, IVF-Flat, BM25.

ADR-007: all of these are implemented from scratch. FAISS/hnswlib appear only in
``benchmarks/`` as baselines.
"""

from pyvec.indexes.bm25 import BM25Index
from pyvec.indexes.flat import FlatIndex
from pyvec.indexes.hnsw import HNSWIndex
from pyvec.indexes.ivf import IVFFlatIndex

__all__ = ["FlatIndex", "HNSWIndex", "IVFFlatIndex", "BM25Index"]


def build_dense_index(index_type, dim, metric, source, params=None):
    """Factory: construct the dense index named by ``index_type``."""
    from pyvec.core.types import IndexType

    index_type = IndexType.parse(index_type)
    params = dict(params or {})
    if index_type is IndexType.FLAT:
        return FlatIndex(dim=dim, metric=metric, source=source, **params)
    if index_type is IndexType.HNSW:
        return HNSWIndex(dim=dim, metric=metric, source=source, **params)
    return IVFFlatIndex(dim=dim, metric=metric, source=source, **params)
