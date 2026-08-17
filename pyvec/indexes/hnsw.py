"""HNSW — Hierarchical Navigable Small World graph index.

Implemented from Malkov & Yashunin (2016), arXiv:1603.09320. Algorithm numbers
in the comments below refer to that paper:

* Algorithm 1 — INSERT
* Algorithm 2 — SEARCH-LAYER
* Algorithm 3 — SELECT-NEIGHBORS-SIMPLE  (**not used**, see below)
* Algorithm 4 — SELECT-NEIGHBORS-HEURISTIC
* Algorithm 5 — K-NN-SEARCH

Mental model (LEARNING.md layer 1): a stack of proximity graphs. Layer 0 holds
every vector with dense local connectivity; each layer above holds an
exponentially thinning random subset with longer edges. A search enters at the
top, greedily walks toward the query, drops a layer, repeats — a skip list for
graphs. Long top edges cross the space cheaply; dense bottom edges refine.

Three things this file gets right that a naive implementation gets wrong, all
called out as traps in LEARNING.md and PROJECT_PLAN.md:

1. **Neighbour selection uses the heuristic (algorithm 4), not top-M.** Taking
   the M nearest candidates produces clusters of mutually-close neighbours and
   leaves whole regions unreachable. The heuristic keeps a candidate only if it
   is closer to the new node than to anything already selected, which preserves
   long-range edges. Getting this wrong pins recall around 50%.
2. **Level assignment is exponential**, ``floor(-ln(U(0,1)) * mL)`` with
   ``mL = 1/ln(M)``. A uniform or off-by-one distribution gives a graph that
   looks fine and caps recall around 80%.
3. **The entry point is a single global node.** First insert has to create it;
   deletion must not invalidate it (we tombstone, so the node stays in the graph
   as a routing waypoint — ADR-010).

Performance note: the graph walk is the part of PyVec that is most Python-bound
(one dict lookup and one heap op per hop). The single most effective mitigation
is batching — every candidate frontier is scored with **one** vectorised distance
call over a gathered block, never one call per neighbour. See
:meth:`HNSWIndex._search_layer`.
"""

from __future__ import annotations

import heapq
import math
import os
import random
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from pyvec.core.distance import (
    as_vector,
    distance,
    pairwise_distance,
    squared_norms,
)
from pyvec.core.errors import CorruptDataError
from pyvec.core.types import InternalId, Metric, VectorSource

__all__ = ["HNSWIndex"]

_MAGIC = b"PYVECHNS"
_FORMAT_VERSION = 1
# magic, version, dim, M, M0, ef_construction, ef_search, entry_point,
# max_level, num_layers, num_deleted, num_draws, seed
_HEADER = struct.Struct("<8sIIIIIIqiIIQQ")

#: LEARNING.md layer 1 "key parameters you'll actually tune".
DEFAULT_M = 16
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 64

#: Bound on the retry loop that widens ``ef`` when tombstones crowd out results.
_MAX_TOMBSTONE_RETRIES = 3


class HNSWIndex:
    """A multi-layer proximity graph supporting incremental insert and k-NN.

    Args:
        dim: vector dimension.
        metric: the collection's metric. The graph is built *for* this metric;
            a graph built for L2 does not answer cosine queries correctly, which
            is precisely why ADR-009 fixes the metric per collection.
        source: where to read vectors from (the collection's mmap store).
        M: max connections per node on layers >= 1. Layer 0 uses ``M0``
            (default ``2M``). Higher M = better recall, more memory, slower
            build.
        ef_construction: beam width while inserting. Higher = better graph,
            slower build. Not used at query time.
        ef_search: default beam width at query time. Overridable per query,
            which is the knob users trade recall against latency with.
        seed: RNG seed for level assignment.
    """

    name = "hnsw"

    def __init__(
        self,
        dim: int,
        metric: Metric,
        source: VectorSource,
        *,
        M: int = DEFAULT_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
        M0: int | None = None,
        seed: int = 42,
        **_ignored: Any,
    ) -> None:
        if M < 2:
            raise ValueError("M must be >= 2 (mL = 1/ln(M) is undefined at M=1)")
        self.dim = int(dim)
        self.metric = Metric.parse(metric)
        self.source = source
        self.M = int(M)
        self.M0 = int(M0) if M0 is not None else 2 * int(M)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)

        #: ``mL`` normalises the level distribution so the expected number of
        #: layers is ~log_M(N). The paper shows 1/ln(M) minimises overlap
        #: between layers, which is what makes the greedy descent work.
        self.mL = 1.0 / math.log(self.M)

        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._draws = 0  # replayed on load, so restarts stay deterministic

        #: layer -> node -> neighbours. ARCHITECTURE.md's
        #: ``list[dict[int, list[int]]]``. ``layers[0]`` contains every node.
        self.layers: list[dict[InternalId, list[InternalId]]] = []
        self.node_levels: dict[InternalId, int] = {}
        self.entry_point: InternalId | None = None
        self.max_level: int = -1
        self._deleted: set[InternalId] = set()

        #: Squared-norm cache for the L2 hot path, indexed by internal id and
        #: grown geometrically. Profiling the build showed the per-frontier
        #: ``einsum`` over candidate blocks was the single largest cost; with
        #: this cache it becomes a fancy-index lookup. Cosine and dot need no
        #: norms, so they get no cache and no bookkeeping.
        self._needs_sqnorms = self.metric is Metric.L2
        self._sqnorms: np.ndarray | None = (
            np.zeros(1024, dtype=np.float32) if self._needs_sqnorms else None
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.node_levels) - len(self._deleted)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "num_vectors": len(self),
            "num_deleted": len(self._deleted),
            "max_level": self.max_level,
            "entry_point": self.entry_point,
            "level_histogram": self.level_histogram(),
            "memory_bytes": self.memory_bytes(),
            **self.params,
        }

    def level_histogram(self) -> list[int]:
        """Nodes present per layer.

        Debug aid for trap #2: this should decay by roughly a factor of ``M``
        per level. If it does not, level assignment is broken and recall will
        sit around 80% no matter what else you fix (PROJECT_PLAN week 2,
        "common failure modes").
        """
        return [len(layer) for layer in self.layers]

    def memory_bytes(self) -> int:
        """Approximate graph footprint, excluding the vectors themselves."""
        edges = sum(len(n) for layer in self.layers for n in layer.values())
        # A CPython list of small ints costs ~8 bytes per slot plus overhead;
        # count the dict entry too. This is an estimate, labelled as such.
        return edges * 8 + len(self.node_levels) * 100

    # ------------------------------------------------------------------ #
    # Squared-norm cache (L2 only)
    # ------------------------------------------------------------------ #

    def _note_norm(self, node: InternalId, vec: np.ndarray) -> None:
        if not self._needs_sqnorms:
            return
        assert self._sqnorms is not None
        if node >= self._sqnorms.shape[0]:
            grown = np.zeros(max(node + 1, self._sqnorms.shape[0] * 2), np.float32)
            grown[: self._sqnorms.shape[0]] = self._sqnorms
            self._sqnorms = grown
        # Must go through the same kernel as _rebuild_norm_cache. `float(vec @ vec)`
        # would be the obvious thing to write here, but np.dot and einsum sum in
        # different orders, so the two paths disagree in the last bits — and a
        # reloaded index would then report very slightly different distances than
        # the one that wrote it. Reproducibility across restarts is worth more
        # than saving one reshape.
        self._sqnorms[node] = squared_norms(vec.reshape(1, -1))[0]

    def _norms_for(self, nodes: Sequence[InternalId]) -> np.ndarray | None:
        if not self._needs_sqnorms:
            return None
        assert self._sqnorms is not None
        return self._sqnorms[np.asarray(nodes, dtype=np.int64)]

    def _rebuild_norm_cache(self) -> None:
        """Recompute the whole cache from the source, e.g. after :meth:`load`."""
        if not self._needs_sqnorms or not self.node_levels:
            return
        nodes = np.fromiter(
            self.node_levels.keys(), dtype=np.int64, count=len(self.node_levels)
        )
        self._sqnorms = np.zeros(int(nodes.max()) + 1, dtype=np.float32)
        # Chunked so a 1M-vector reload never materialises the whole store.
        for start in range(0, nodes.shape[0], 65_536):
            chunk = nodes[start : start + 65_536]
            self._sqnorms[chunk] = squared_norms(self.source.gather(chunk))

    # ------------------------------------------------------------------ #
    # Level assignment
    # ------------------------------------------------------------------ #

    def _random_level(self) -> int:
        """``floor(-ln(U(0,1)) * mL)`` — trap #2.

        ``random()`` returns ``[0, 1)``, so guard the zero case: ``ln(0)`` is
        ``-inf`` and would put a node on an infinitely tall tower.
        """
        self._draws += 1
        u = self._rng.random()
        if u <= 0.0:
            u = 1e-12
        return int(math.floor(-math.log(u) * self.mL))

    # ------------------------------------------------------------------ #
    # Algorithm 2 — SEARCH-LAYER
    # ------------------------------------------------------------------ #

    def _search_layer(
        self,
        q: np.ndarray,
        entry_points: list[tuple[float, InternalId]],
        ef: int,
        layer: int,
    ) -> list[tuple[float, InternalId]]:
        """Greedy best-first walk on one layer with a beam of width ``ef``.

        Args:
            q: query vector.
            entry_points: ``(distance, node)`` pairs to start from, distances
                already computed against ``q``.
            ef: beam width. ``1`` gives plain greedy descent.
            layer: which graph to walk.

        Returns:
            Up to ``ef`` ``(distance, node)`` pairs as a **max-heap keyed by
            negated distance** — i.e. ``result[0]`` is the *furthest* kept
            candidate, which is what the caller needs to test the stop
            condition cheaply.
        """
        graph = self.layers[layer]
        visited: set[InternalId] = {node for _, node in entry_points}

        # candidates: min-heap by distance — the frontier still to expand.
        candidates: list[tuple[float, InternalId]] = list(entry_points)
        heapq.heapify(candidates)
        # found: max-heap by distance (negated) — the current best ef.
        found: list[tuple[float, InternalId]] = [(-d, n) for d, n in entry_points]
        heapq.heapify(found)
        while len(found) > ef:
            heapq.heappop(found)

        gather = self.source.gather
        metric = self.metric

        while candidates:
            d, c = heapq.heappop(candidates)
            furthest = -found[0][0]
            # Paper's stop condition: the nearest unexpanded candidate is
            # already worse than the worst thing we're keeping, so no
            # descendant can improve the beam. The `len(found) >= ef` guard
            # keeps us going while the beam is still underfull.
            if d > furthest and len(found) >= ef:
                break

            neighbours = graph.get(c)
            if not neighbours:
                continue
            fresh = [n for n in neighbours if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)

            # One gather + one distance call for the whole frontier. Doing this
            # per-neighbour instead costs ~5x on the query path.
            dists = distance(metric, q, gather(fresh), self._norms_for(fresh))

            for n, dn in zip(fresh, dists):
                dn = float(dn)
                if len(found) < ef:
                    heapq.heappush(candidates, (dn, n))
                    heapq.heappush(found, (-dn, n))
                elif dn < -found[0][0]:
                    heapq.heappush(candidates, (dn, n))
                    heapq.heapreplace(found, (-dn, n))

        return found

    # ------------------------------------------------------------------ #
    # Algorithm 4 — SELECT-NEIGHBORS-HEURISTIC
    # ------------------------------------------------------------------ #

    def _select_neighbours(
        self,
        candidates: list[tuple[float, InternalId]],
        m: int,
        *,
        keep_pruned: bool = True,
    ) -> list[InternalId]:
        """Pick <= ``m`` diverse neighbours from ``candidates``. Trap #1.

        A candidate ``e`` is accepted only if it is closer to the query point
        than to every already-accepted neighbour. That rejects candidates which
        sit "behind" an accepted one from the query's point of view, so the
        retained edge set spans directions rather than piling into one cluster.

        Args:
            candidates: ``(distance_to_q, node)`` pairs. Order irrelevant.
            m: maximum neighbours to return.
            keep_pruned: paper's ``keepPrunedConnections``. Backfill from the
                rejected pile if we ended up with fewer than ``m``, so node
                degree stays near the budget. Keeping this on measurably helps
                recall at low ``M``.
        """
        if len(candidates) <= m:
            return [n for _, n in sorted(candidates)]

        ordered = sorted(candidates)  # nearest to q first
        ids = [n for _, n in ordered]
        ds = [d for d, _ in ordered]

        # Gather once, then all candidate-to-candidate distances come out of one
        # (n, n) matrix. n <= ef_construction, so this is a few hundred rows.
        vecs = self.source.gather(ids)
        # .tolist() looks wasteful but is a large win: the acceptance test below
        # touches a handful of cells per candidate, and NumPy's per-call overhead
        # on a 1-element fancy index dwarfs the arithmetic. Profiling showed this
        # loop spending ~40% of build time inside `ndarray.min` before the
        # conversion.
        cross = pairwise_distance(self.metric, vecs, vecs).tolist()

        selected: list[int] = []  # positions into `ordered`
        discarded: list[int] = []

        for i in range(len(ordered)):
            if len(selected) >= m:
                break
            if not selected:
                selected.append(i)
                continue
            # Closer to q than to anything already selected? Short-circuits on
            # the first violation, which is the common case for a rejection.
            row = cross[i]
            di = ds[i]
            if all(row[j] > di for j in selected):
                selected.append(i)
            else:
                discarded.append(i)

        if keep_pruned:
            for i in discarded:
                if len(selected) >= m:
                    break
                selected.append(i)

        return [ids[i] for i in selected]

    def _prune_connections(self, node: InternalId, layer: int, m: int) -> None:
        """Re-select ``node``'s neighbours down to ``m`` using the heuristic.

        Called when a back-link pushes an existing node over its degree budget.
        Truncating to the nearest ``m`` instead would reintroduce trap #1 through
        the back door — the diversity property has to be maintained on *both*
        ends of every edge.
        """
        conns = self.layers[layer][node]
        if len(conns) <= m:
            return
        vec = self.source.get(node)
        ds = distance(
            self.metric, vec, self.source.gather(conns), self._norms_for(conns)
        )
        self.layers[layer][node] = self._select_neighbours(
            [(float(d), n) for d, n in zip(ds, conns)], m
        )

    # ------------------------------------------------------------------ #
    # Algorithm 1 — INSERT
    # ------------------------------------------------------------------ #

    def add(
        self, ids: Sequence[InternalId], vectors: np.ndarray | None = None
    ) -> None:
        """Insert nodes one at a time.

        Inserts are inherently sequential: each new node is routed through the
        graph the previous ones built. There is no batch shortcut that preserves
        the graph's properties.
        """
        if vectors is not None:
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            if vectors.shape[0] != len(ids):
                raise ValueError(
                    f"got {len(ids)} ids but {vectors.shape[0]} vectors"
                )
        for pos, node in enumerate(ids):
            vec = (
                vectors[pos]
                if vectors is not None
                else self.source.get(int(node))
            )
            self._insert(int(node), vec)

    def _insert(self, node: InternalId, vec: np.ndarray) -> None:
        if node in self.node_levels:
            # Re-inserting an id would leave stale edges pointing at the old
            # position. Upsert is delete-then-insert at the collection level.
            self._deleted.discard(node)
            return

        level = self._random_level()
        self.node_levels[node] = level
        self._note_norm(node, vec)

        while len(self.layers) <= level:
            self.layers.append({})
        for lc in range(level + 1):
            self.layers[lc].setdefault(node, [])

        # Trap #3: the very first node becomes the global entry point and has
        # nothing to connect to.
        if self.entry_point is None:
            self.entry_point = node
            self.max_level = level
            return

        ep_node = self.entry_point
        ep_dist = float(distance(self.metric, vec, self.source.get(ep_node))[0])
        entry: list[tuple[float, InternalId]] = [(ep_dist, ep_node)]

        # Phase 1: greedy descent (ef=1) through the layers above `level`. We
        # are not linking here, only finding a good entry point for phase 2.
        for lc in range(self.max_level, level, -1):
            found = self._search_layer(vec, entry, 1, lc)
            best = max(found)  # max of (-d, n) == smallest distance
            entry = [(-best[0], best[1])]

        # Phase 2: from min(level, max_level) down to 0, find ef_construction
        # candidates and wire up bidirectional edges.
        for lc in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(vec, entry, self.ef_construction, lc)
            candidates = [(-d, n) for d, n in found]

            m_max = self.M0 if lc == 0 else self.M
            neighbours = self._select_neighbours(candidates, self.M)

            self.layers[lc][node] = list(neighbours)
            for n in neighbours:
                conns = self.layers[lc].setdefault(n, [])
                conns.append(node)
                if len(conns) > m_max:
                    self._prune_connections(n, lc, m_max)

            # The whole beam seeds the next layer down, not just the best node —
            # this is what keeps the descent from getting stuck in a local
            # minimum on the way to layer 0.
            entry = candidates

        if level > self.max_level:
            self.max_level = level
            self.entry_point = node

    # ------------------------------------------------------------------ #
    # Deletion — ADR-010
    # ------------------------------------------------------------------ #

    def remove(self, ids: Sequence[InternalId]) -> None:
        """Tombstone. Nodes stay in the graph as routing waypoints.

        True HNSW deletion means re-wiring every in-edge of the removed node,
        which is an open research problem (see FreshDiskANN). Soft delete is
        O(1) and correct; recall degrades in proportion to
        ``deleted_count / total_count`` until ``optimize()`` rebuilds. Milvus
        and Qdrant make the same trade.
        """
        for i in ids:
            i = int(i)
            if i in self.node_levels:
                self._deleted.add(i)

    # ------------------------------------------------------------------ #
    # Algorithm 5 — K-NN-SEARCH
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        ef_search: int | None = None,
        exclude: set[InternalId] | None = None,
        **_params: Any,
    ) -> list[tuple[InternalId, float]]:
        """Top-``k`` nearest live nodes as ``(internal_id, ordering_distance)``.

        Args:
            ef_search: beam width override. ``max(ef_search, k)`` is enforced —
                a beam narrower than ``k`` cannot return ``k`` results.
            exclude: extra ids to omit (the collection's tombstone set, which
                may be ahead of the index's own after a WAL replay).
        """
        if self.entry_point is None or k <= 0:
            return []

        q = as_vector(query, self.dim)
        dead = self._deleted if not exclude else self._deleted | set(exclude)
        ef = max(int(ef_search or self.ef_search), k)

        for attempt in range(_MAX_TOMBSTONE_RETRIES + 1):
            results = self._knn(q, ef, dead)
            # Widen the beam only if tombstones are what starved the result set.
            if len(results) >= k or not dead or attempt == _MAX_TOMBSTONE_RETRIES:
                break
            ef *= 2

        return results[:k]

    def _knn(
        self, q: np.ndarray, ef: int, dead: set[InternalId]
    ) -> list[tuple[InternalId, float]]:
        ep_node = self.entry_point
        assert ep_node is not None
        ep_dist = float(distance(self.metric, q, self.source.get(ep_node))[0])
        entry: list[tuple[float, InternalId]] = [(ep_dist, ep_node)]

        # Descend greedily with ef=1 down to layer 1...
        for lc in range(self.max_level, 0, -1):
            found = self._search_layer(q, entry, 1, lc)
            best = max(found)
            entry = [(-best[0], best[1])]

        # ...then one wide beam on layer 0, where every vector lives.
        found = self._search_layer(q, entry, ef, 0)
        out = sorted((-d, n) for d, n in found)
        return [(n, d) for d, n in out if n not in dead]

    # ------------------------------------------------------------------ #
    # Persistence — custom binary CSR
    # ------------------------------------------------------------------ #
    # ARCHITECTURE.md allows "pickle or a custom binary format". Pickle of a
    # million-entry dict-of-lists takes tens of seconds to load, which blows
    # PRD NF3 (1M vectors ready in <30s) on its own. This format stores each
    # layer as CSR arrays read with a single np.fromfile, then rebuilds the
    # dicts — dominated by dict construction rather than unpickling.

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")

        with open(tmp, "wb") as f:
            f.write(
                _HEADER.pack(
                    _MAGIC,
                    _FORMAT_VERSION,
                    self.dim,
                    self.M,
                    self.M0,
                    self.ef_construction,
                    self.ef_search,
                    -1 if self.entry_point is None else int(self.entry_point),
                    self.max_level,
                    len(self.layers),
                    len(self._deleted),
                    self._draws,
                    self.seed,
                )
            )
            np.asarray(sorted(self._deleted), dtype=np.int64).tofile(f)

            for layer in self.layers:
                nodes = np.fromiter(layer.keys(), dtype=np.int32, count=len(layer))
                degrees = np.fromiter(
                    (len(layer[int(n)]) for n in nodes),
                    dtype=np.int32,
                    count=len(layer),
                )
                indptr = np.zeros(len(nodes) + 1, dtype=np.int64)
                np.cumsum(degrees, out=indptr[1:])
                flat = np.fromiter(
                    (n for node in nodes for n in layer[int(node)]),
                    dtype=np.int32,
                    count=int(indptr[-1]),
                )
                f.write(struct.pack("<I", len(nodes)))
                nodes.tofile(f)
                indptr.tofile(f)
                flat.tofile(f)
            f.flush()
            os.fsync(f.fileno())

        tmp.replace(path)

    def load(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        with open(path, "rb") as f:
            raw = f.read(_HEADER.size)
            if len(raw) < _HEADER.size:
                raise CorruptDataError(f"{path}: truncated HNSW header")
            (
                magic,
                version,
                dim,
                M,
                M0,
                ef_construction,
                ef_search,
                entry_point,
                max_level,
                num_layers,
                num_deleted,
                draws,
                seed,
            ) = _HEADER.unpack(raw)
            if magic != _MAGIC:
                raise CorruptDataError(f"{path}: not an HNSW index file")
            if version != _FORMAT_VERSION:
                raise CorruptDataError(
                    f"{path}: HNSW format version {version}, expected "
                    f"{_FORMAT_VERSION}"
                )

            self.dim = int(dim)
            self.M = int(M)
            self.M0 = int(M0)
            self.ef_construction = int(ef_construction)
            self.ef_search = int(ef_search)
            self.mL = 1.0 / math.log(self.M)
            self.entry_point = None if entry_point < 0 else int(entry_point)
            self.max_level = int(max_level)
            self.seed = int(seed)

            self._deleted = {
                int(i) for i in np.fromfile(f, dtype=np.int64, count=num_deleted)
            }

            self.layers = []
            self.node_levels = {}
            for lc in range(num_layers):
                (count,) = struct.unpack("<I", f.read(4))
                nodes = np.fromfile(f, dtype=np.int32, count=count)
                indptr = np.fromfile(f, dtype=np.int64, count=count + 1)
                flat = np.fromfile(f, dtype=np.int32, count=int(indptr[-1]))
                layer: dict[InternalId, list[InternalId]] = {}
                for pos in range(count):
                    node = int(nodes[pos])
                    layer[node] = flat[indptr[pos] : indptr[pos + 1]].tolist()
                    # A node's level is the highest layer it appears on.
                    if self.node_levels.get(node, -1) < lc:
                        self.node_levels[node] = lc
                self.layers.append(layer)

        # Replay the level-assignment RNG so inserts after a restart follow the
        # same sequence they would have without one. BENCHMARKS.md demands
        # reproducibility; a fresh RNG after reload would quietly break it.
        self._rng = random.Random(self.seed)
        self._draws = 0
        for _ in range(int(draws)):
            self._random_level()

        # Norms are derived state, not persisted — cheaper to recompute from the
        # mmap than to store and validate a second copy.
        self._rebuild_norm_cache()

    # ------------------------------------------------------------------ #
    # Consistency check — used by tests
    # ------------------------------------------------------------------ #

    def validate(self) -> list[str]:
        """Structural invariants. Returns a list of violations (empty = good)."""
        problems: list[str] = []
        if self.entry_point is not None:
            if self.node_levels.get(self.entry_point, -1) != self.max_level:
                problems.append("entry point is not on the top layer")
        for lc, layer in enumerate(self.layers):
            m_max = self.M0 if lc == 0 else self.M
            for node, conns in layer.items():
                if len(conns) > m_max:
                    problems.append(
                        f"layer {lc} node {node}: degree {len(conns)} > {m_max}"
                    )
                if node in conns:
                    problems.append(f"layer {lc} node {node}: self-loop")
                if len(set(conns)) != len(conns):
                    problems.append(f"layer {lc} node {node}: duplicate edges")
                for n in conns:
                    if n not in layer:
                        problems.append(
                            f"layer {lc} node {node}: edge to absent node {n}"
                        )
        for node, level in self.node_levels.items():
            for lc in range(level + 1):
                if node not in self.layers[lc]:
                    problems.append(f"node {node} missing from layer {lc}")
        return problems

    def iter_nodes(self) -> Iterable[InternalId]:
        return self.node_levels.keys()
