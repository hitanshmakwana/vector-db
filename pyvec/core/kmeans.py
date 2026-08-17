"""Mini k-means with k-means++ initialisation.

Used only by the IVF-Flat index to learn its ``nlist`` centroids. PROJECT_PLAN
week 3 budgets ~50 LOC for this; the extra length here is empty-cluster
handling and the sampling path, both of which matter in practice.

k-means++ initialisation (Arthur & Vassilvitskii 2007) is not optional:
LEARNING.md layer 2 flags it, and with uniform-random seeds on clustered data
you reliably get a few centroids owning most of the dataset, which destroys
IVF recall at low ``nprobe``.

Everything is seeded. BENCHMARKS.md: "a benchmark that gives different numbers
on different runs is worthless."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyvec.core.distance import pairwise_distance
from pyvec.core.types import VECTOR_DTYPE, Metric

__all__ = ["KMeansResult", "kmeans", "assign"]

#: Points sampled for training when the dataset is larger. PROJECT_PLAN week 3:
#: "train from a sample if collection is large (sample 100k for k-means)".
DEFAULT_TRAIN_SAMPLE = 100_000

#: A cluster count below this fraction of the sample is pointless — with fewer
#: than a handful of points per cluster, centroids just memorise noise.
MIN_POINTS_PER_CENTROID = 1


@dataclass(slots=True)
class KMeansResult:
    centroids: np.ndarray  # (k, dim) float32
    assignments: np.ndarray  # (n,) int32 — for the *training* sample only
    inertia: float  # sum of ordering distances to the assigned centroid
    iterations: int
    converged: bool


def _kmeanspp_init(
    x: np.ndarray, k: int, metric: Metric, rng: np.random.Generator
) -> np.ndarray:
    """Choose ``k`` seeds, each new one biased toward far-from-existing points.

    Standard D^2 sampling. ``pairwise_distance`` returns the ordering distance,
    which for cosine/dot can be negative; we shift to a non-negative weight
    before sampling because probabilities cannot be negative.
    """
    n = x.shape[0]
    centroids = np.empty((k, x.shape[1]), dtype=VECTOR_DTYPE)
    first = int(rng.integers(n))
    centroids[0] = x[first]

    # closest[i] = ordering distance from point i to the nearest chosen centroid
    closest = pairwise_distance(metric, x, centroids[0:1]).reshape(-1)

    for c in range(1, k):
        weights = closest - closest.min()
        total = float(weights.sum())
        if total <= 0.0:
            # All remaining points coincide with a centroid. Fill with random
            # distinct picks; duplicates would produce empty clusters we then
            # have to repair anyway.
            centroids[c] = x[int(rng.integers(n))]
        else:
            probs = weights / total
            centroids[c] = x[int(rng.choice(n, p=probs))]
        d_new = pairwise_distance(metric, x, centroids[c : c + 1]).reshape(-1)
        np.minimum(closest, d_new, out=closest)

    return centroids


def assign(
    x: np.ndarray, centroids: np.ndarray, metric: Metric, block: int = 8192
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each row of ``x`` to its nearest centroid.

    Returns ``(labels, distances)``. Processed in blocks so the ``(n, k)``
    distance matrix never has to exist all at once — at n=1M, k=4096 that
    matrix would be 16GB in float32.
    """
    n = x.shape[0]
    labels = np.empty(n, dtype=np.int32)
    dists = np.empty(n, dtype=np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        d = pairwise_distance(metric, x[start:stop], centroids)
        idx = np.argmin(d, axis=1)
        labels[start:stop] = idx.astype(np.int32)
        dists[start:stop] = d[np.arange(stop - start), idx]
    return labels, dists


def kmeans(
    x: np.ndarray,
    k: int,
    *,
    metric: Metric = Metric.L2,
    max_iter: int = 25,
    tol: float = 1e-4,
    seed: int = 42,
    sample: int | None = DEFAULT_TRAIN_SAMPLE,
) -> KMeansResult:
    """Lloyd's algorithm with k-means++ seeding.

    Args:
        x: ``(n, dim)`` training data.
        k: number of centroids (``nlist``). Clamped to ``n``.
        metric: the collection's metric, so cluster geometry matches the
            geometry search will use.
        max_iter: Lloyd iterations. 25 is plenty for IVF — centroid quality
            past that point does not move recall measurably, and build time is
            dominated by this loop (LEARNING.md layer 2).
        tol: stop when the relative inertia improvement drops below this.
        seed: RNG seed. Fixed by default.
        sample: cap on training rows; ``None`` trains on everything.
    """
    x = np.ascontiguousarray(x, dtype=VECTOR_DTYPE)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {x.shape}")
    n = x.shape[0]
    if n == 0:
        raise ValueError("cannot run k-means on an empty array")

    rng = np.random.default_rng(seed)

    if sample is not None and n > sample:
        idx = rng.choice(n, size=sample, replace=False)
        idx.sort()  # sequential access is much kinder to an mmap-backed source
        x = np.ascontiguousarray(x[idx])
        n = x.shape[0]

    k = max(1, min(k, n // MIN_POINTS_PER_CENTROID, n))

    centroids = _kmeanspp_init(x, k, metric, rng)
    labels = np.zeros(n, dtype=np.int32)
    prev_inertia = np.inf
    inertia = np.inf
    iterations = 0
    converged = False

    for iterations in range(1, max_iter + 1):
        labels, dists = assign(x, centroids, metric)
        inertia = float(dists.sum())

        # Recompute centroids as the mean of their members. np.add.at would be
        # correct but is slow; bincount over each dimension is the vectorised
        # form of the same scatter-add.
        counts = np.bincount(labels, minlength=k)
        sums = np.zeros((k, x.shape[1]), dtype=np.float64)
        np.add.at(sums, labels, x)

        empty = np.flatnonzero(counts == 0)
        if empty.size:
            # Repair empty clusters by stealing the points that are currently
            # worst-served. Left alone, an empty cluster stays empty forever and
            # we silently end up with fewer than nlist buckets.
            worst = np.argsort(-dists)[: empty.size]
            for cluster, point in zip(empty, worst):
                sums[cluster] = x[point]
                counts[cluster] = 1
                # Remove the stolen point from its old cluster's accumulation.
                old = labels[point]
                if counts[old] > 1:
                    sums[old] -= x[point]
                    counts[old] -= 1
                labels[point] = cluster

        centroids = (sums / counts[:, None]).astype(VECTOR_DTYPE)

        if metric.normalize_on_insert:
            # A mean of unit vectors is not a unit vector. For cosine, centroids
            # must live on the same sphere as the data or the ordering distance
            # (which assumes unit norms) is wrong. This is spherical k-means.
            from pyvec.core.distance import normalize

            centroids = normalize(centroids)

        if prev_inertia < np.inf:
            improvement = (prev_inertia - inertia) / max(abs(prev_inertia), 1e-12)
            if improvement < tol:
                converged = True
                break
        prev_inertia = inertia

    return KMeansResult(
        centroids=centroids,
        assignments=labels,
        inertia=inertia,
        iterations=iterations,
        converged=converged,
    )
