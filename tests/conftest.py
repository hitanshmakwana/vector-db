"""Shared fixtures.

Every random draw in the suite comes from a seeded generator. PRD NF5 asks for
index agreement with brute force "modulo floating-point ties", and a test that
draws fresh random data each run cannot distinguish a real recall regression from
an unlucky sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyvec.core.collection import Collection
from pyvec.core.collection_manager import CollectionManager
from pyvec.core.types import ArrayVectorSource

SEED = 42


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    return root


@pytest.fixture
def manager(data_root: Path):
    m = CollectionManager(data_root)
    yield m
    m.close()


@pytest.fixture
def make_collection(data_root: Path):
    """Factory for collections that are always closed at teardown.

    Closing matters on Windows: an open memmap holds a file handle, and pytest's
    tmp_path cleanup fails if anything is still mapped.
    """
    created: list[Collection] = []

    def _make(name: str = "test", dimension: int = 8, **kwargs) -> Collection:
        c = Collection.create(name, data_root, dimension=dimension, **kwargs)
        created.append(c)
        return c

    yield _make
    for c in created:
        try:
            c.close()
        except Exception:
            pass


@pytest.fixture
def vectors(rng: np.random.Generator) -> np.ndarray:
    """1000 x 32 Gaussian vectors.

    Gaussian rather than uniform-in-a-cube on purpose: it is the harder case for
    both indexes. High-dimensional Gaussian data has no cluster structure for IVF
    to exploit and weak "hubness" for HNSW's long edges to use, so recall
    measured here is a floor, not a flattering number.
    """
    return rng.normal(size=(1000, 32)).astype(np.float32)


@pytest.fixture
def source(vectors: np.ndarray) -> ArrayVectorSource:
    return ArrayVectorSource(vectors)


@pytest.fixture
def queries(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(50, 32)).astype(np.float32)


def recall_at_k(
    truth: list[tuple[int, float]] | list[int],
    got: list[tuple[int, float]] | list[int],
    k: int,
) -> float:
    """``|top_k_index ∩ top_k_true| / k`` — the definition from BENCHMARKS.md."""

    def ids(seq):
        return {x[0] if isinstance(x, tuple) else x for x in seq[:k]}

    return len(ids(truth) & ids(got)) / k


TEXT_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown dog outpaces a quick quick fox",
    "lorem ipsum dolor sit amet consectetur adipiscing",
    "the lazy dog sleeps all day long in the sun",
    "machine learning models embed text into dense vectors",
    "vector databases index embeddings for similarity search",
    "the fox and the hound became unlikely friends",
    "dense retrieval and sparse retrieval have complementary strengths",
]
