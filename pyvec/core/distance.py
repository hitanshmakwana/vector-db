"""Distance kernels.

Everything here is NumPy-vectorised: one query against an ``(n, dim)`` block of
candidates in a single BLAS call. ARCHITECTURE.md §3 — "a naive Python loop over
1M vectors is ~1000x slower than the NumPy version". No Python-level loop over
vectors is allowed in this module.

Two vocabularies, kept deliberately separate:

**Ordering distance** (:func:`distance`) — always *lower is better*, for every
metric. This is what the indexes sort by, so index code never branches on the
metric. It is also allowed to be monotonically transformed (we use *squared* L2,
skipping the sqrt) because ordering is all the indexes need.

**Score** (:func:`score_from_distance`) — the user-facing value in the API
response. Cosine/dot return a similarity (higher better), L2 returns a true
Euclidean distance (lower better), matching API_SPEC.

The mapping, for a query ``q`` and candidate ``x``:

===========  ==========================  ============================
metric       ordering distance           score
===========  ==========================  ============================
``cosine``   ``1 - cos(q, x)``           ``1 - d``  (the cosine)
``dot``      ``-dot(q, x)``              ``-d``     (the dot product)
``l2``       ``||q - x||^2``             ``sqrt(d)``
===========  ==========================  ============================

Cosine assumes both sides are unit-normalised, which the collection guarantees
by normalising on insert (ADR-009 / LEARNING.md layer 0). That makes the cosine
hot path a plain dot product.
"""

from __future__ import annotations

import numpy as np

from pyvec.core.types import VECTOR_DTYPE, Metric

__all__ = [
    "cosine_similarity",
    "l2_distance",
    "dot_product",
    "distance",
    "pairwise_distance",
    "squared_norms",
    "score_from_distance",
    "normalize",
    "as_vector",
    "as_matrix",
]

# Guard against 1/0 when a caller normalises an all-zero vector.
_EPS = 1e-12


def as_vector(v: np.ndarray | list[float], dim: int | None = None) -> np.ndarray:
    """Coerce to a contiguous 1-D float32 vector."""
    a = np.ascontiguousarray(v, dtype=VECTOR_DTYPE).reshape(-1)
    if dim is not None and a.shape[0] != dim:
        from pyvec.core.errors import InvalidDimensionError

        raise InvalidDimensionError(
            f"expected dimension {dim}, got {a.shape[0]}"
        )
    return a


def as_matrix(x: np.ndarray | list, dim: int | None = None) -> np.ndarray:
    """Coerce to a contiguous 2-D ``(n, dim)`` float32 matrix."""
    a = np.ascontiguousarray(x, dtype=VECTOR_DTYPE)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {a.shape}")
    if dim is not None and a.shape[1] != dim:
        from pyvec.core.errors import InvalidDimensionError

        raise InvalidDimensionError(
            f"expected dimension {dim}, got {a.shape[1]}"
        )
    return a


def normalize(x: np.ndarray) -> np.ndarray:
    """Unit-normalise along the last axis. Zero vectors are left as zeros."""
    a = np.asarray(x, dtype=VECTOR_DTYPE)
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    return (a / np.maximum(norms, _EPS)).astype(VECTOR_DTYPE, copy=False)


# --------------------------------------------------------------------------- #
# Raw metrics — the textbook formulas, one query vs. many candidates.
# --------------------------------------------------------------------------- #


def dot_product(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``sum(q[i] * x[i])``. Higher is more similar."""
    return np.asarray(x, dtype=VECTOR_DTYPE) @ np.asarray(q, dtype=VECTOR_DTYPE)


def cosine_similarity(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``dot(q, x) / (|q| * |x|)``. Range ``[-1, 1]``, higher is more similar.

    This normalises explicitly, so it is correct for un-normalised input. The
    index hot path does *not* call this — it calls :func:`distance`, which
    assumes pre-normalised vectors.
    """
    q = np.asarray(q, dtype=VECTOR_DTYPE)
    x = np.asarray(x, dtype=VECTOR_DTYPE)
    qn = max(float(np.linalg.norm(q)), _EPS)
    xn = np.maximum(np.linalg.norm(x, axis=-1), _EPS)
    return (x @ q) / (xn * qn)


def l2_distance(q: np.ndarray, x: np.ndarray, squared: bool = False) -> np.ndarray:
    """Euclidean distance. Lower is more similar.

    Expanded as ``|x|^2 - 2*x.q + |q|^2`` so the heavy term is a single matmul
    rather than an ``(n, dim)`` temporary from broadcasting a subtraction.

    **Accuracy.** The expansion suffers cancellation when the true distance is
    small relative to the vector norms. The three terms are near-equal in that
    regime, so the subtraction loses most of the significant digits: absolute
    error in the squared value is roughly ``eps * ||x||^2``, and the square root
    turns that into roughly ``||x|| * sqrt(eps)`` — about 3e-4 relative to the
    vector norm in float32. A vector's distance to *itself* therefore comes out
    around 1e-4 rather than exactly 0.

    The reduction terms are accumulated in float64 here, which removes the part
    of that error contributed by the summations, but the ``x @ q`` matmul is
    still float32 (upcasting a whole candidate block would double the memory
    traffic on the benchmark ground-truth path) so the bound above still applies.

    This is the standard trade and FAISS makes it too: the alternative, computing
    ``((x - q)**2).sum(axis=1)`` directly, is numerically stable but materialises
    an ``(n, dim)`` temporary and runs several times slower. Ordering is unaffected
    at this error scale, which is all the indexes need — see :func:`distance`.
    """
    q = np.asarray(q, dtype=VECTOR_DTYPE)
    x = np.asarray(x, dtype=VECTOR_DTYPE)
    xx = np.einsum("ij,ij->i", x, x, dtype=np.float64)
    d2 = xx - 2.0 * (x @ q).astype(np.float64) + float(np.dot(q, q))
    # Cancellation can still push a near-zero value slightly negative.
    np.maximum(d2, 0.0, out=d2)
    return d2 if squared else np.sqrt(d2)


# --------------------------------------------------------------------------- #
# Ordering distance — what the indexes actually use.
# --------------------------------------------------------------------------- #


def distance(
    metric: Metric,
    q: np.ndarray,
    x: np.ndarray,
    x_sqnorms: np.ndarray | None = None,
) -> np.ndarray:
    """Ordering distance of ``q`` against each row of ``x``. Lower is better.

    ``x`` may be 1-D (single candidate) or 2-D ``(n, dim)``; the return is
    always a 1-D array of length ``n``.

    This is the hottest function in the project — HNSW calls it once per
    expanded graph node, ~145 times per insert. The L2 branch is written out
    inline rather than delegating to :func:`l2_distance` specifically to avoid a
    second round of array coercion per call, which profiling showed cost about
    as much as the arithmetic.

    Args:
        x_sqnorms: precomputed ``||x_i||^2``, one per row of ``x``. L2 only.
            Supplying it replaces the ``einsum`` over the candidate block with
            an array lookup; HNSW keeps such a cache (see
            ``HNSWIndex._sqnorms``). Ignored for cosine and dot, which need no
            norms at all.
    """
    x = np.asarray(x, dtype=VECTOR_DTYPE)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    q = np.asarray(q, dtype=VECTOR_DTYPE).reshape(-1)

    ip = x @ q
    if metric is Metric.L2:
        xx = (
            np.asarray(x_sqnorms, dtype=np.float32)
            if x_sqnorms is not None
            else np.einsum("ij,ij->i", x, x)
        )
        d = xx - 2.0 * ip
        d += float(q @ q)
        # Cancellation can push near-zero values slightly negative.
        np.maximum(d, 0.0, out=d)
        return d
    # cosine (pre-normalised) and dot both reduce to a negated inner product;
    # the 1.0 offset for cosine keeps the value in [0, 2] for readability and
    # does not affect ordering.
    if metric is Metric.COSINE:
        return 1.0 - ip
    return -ip


def pairwise_distance(metric: Metric, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ordering distances between every row of ``a`` and every row of ``b``.

    Returns shape ``(len(a), len(b))``. Used by HNSW's neighbour-selection
    heuristic (candidate-to-candidate distances) and by k-means assignment.
    """
    a = as_matrix(a)
    b = as_matrix(b)
    if metric is Metric.L2:
        d2 = (
            np.einsum("ij,ij->i", a, a)[:, None]
            - 2.0 * (a @ b.T)
            + np.einsum("ij,ij->i", b, b)[None, :]
        )
        np.maximum(d2, 0.0, out=d2)
        return d2
    ip = a @ b.T
    if metric is Metric.COSINE:
        return 1.0 - ip
    return -ip


def squared_norms(x: np.ndarray) -> np.ndarray:
    """``||x_i||^2`` per row, for the L2 fast path of :func:`distance`."""
    x = np.asarray(x, dtype=VECTOR_DTYPE)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.einsum("ij,ij->i", x, x).astype(np.float32, copy=False)


def score_from_distance(metric: Metric, d: float) -> float:
    """Convert an ordering distance back to the API-visible ``score``."""
    if metric is Metric.L2:
        return float(np.sqrt(max(d, 0.0)))
    if metric is Metric.COSINE:
        return float(1.0 - d)
    return float(-d)


def distance_from_score(metric: Metric, score: float) -> float:
    """Inverse of :func:`score_from_distance`. Used by tests and the CLI."""
    if metric is Metric.L2:
        return float(score) ** 2
    if metric is Metric.COSINE:
        return 1.0 - float(score)
    return -float(score)
