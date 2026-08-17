"""BM25 sparse retrieval over an inverted index.

The scoring function (Robertson & Walker, ~1994) that is still a strong baseline
three decades later, and still beats naive dense retrieval on plenty of
benchmarks. For a query ``Q`` and document ``D``::

    BM25(Q, D) = sum_i  IDF(q_i) * ( tf(q_i, D) * (k1 + 1) )
                        / ( tf(q_i, D) + k1 * (1 - b + b * |D| / avgdl) )

    IDF(q_i)   = ln( (N - df(q_i) + 0.5) / (df(q_i) + 0.5) + 1 )

Reading the formula as three ideas:

* **IDF** — rare terms carry more signal than common ones.
* **k1 saturation** — the tenth occurrence of a word says much less than the
  first. Term frequency contributes with diminishing returns.
* **b length normalisation** — long documents are penalised so they cannot win
  simply by containing more words.

Versus TF-IDF: both have IDF; only BM25 has saturation and length
normalisation. That is the whole difference, and it is a standard interview
question (RESUME.md).

Data structures follow ARCHITECTURE.md §4 exactly:

* ``postings: dict[str, list[tuple[doc_id, term_freq]]]``
* ``doc_lens: dict[doc_id, int]``, ``avg_doc_len``, ``num_docs``
* ``idf_cache: dict[str, float]`` — computed lazily, invalidated when ``N``
  changes because IDF depends on the corpus size.

``k1 = 1.5``, ``b = 0.75`` are the well-known defaults; both are exposed.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from pyvec.core.tokenize import tokenize
from pyvec.core.types import InternalId

__all__ = ["BM25Index", "DEFAULT_K1", "DEFAULT_B"]

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class BM25Index:
    """Inverted index plus BM25 scorer.

    Supports both batch build and online single-document add, because the
    collection updates dense and sparse indexes together on every insert
    (ADR-011: one collection, one insert, both indexes).
    """

    name = "bm25"

    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.postings: dict[str, list[tuple[InternalId, int]]] = {}
        self.doc_lens: dict[InternalId, int] = {}
        self.total_len = 0
        self._idf_cache: dict[str, float] = {}
        self._deleted: set[InternalId] = set()

        #: Derived NumPy view of ``postings``, built lazily and dropped on any
        #: mutation. ``postings`` stays the canonical structure (it is what
        #: ARCHITECTURE.md specifies and what save/load round-trips); this is
        #: purely a scoring accelerator. See :meth:`search`.
        self._np_postings: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        self._np_doclen: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self.num_docs

    @property
    def num_docs(self) -> int:
        """Live documents. Tombstoned docs are excluded from ``N`` for IDF."""
        return len(self.doc_lens) - len(self._deleted)

    @property
    def avg_doc_len(self) -> float:
        n = self.num_docs
        if n <= 0:
            return 0.0
        deleted_len = sum(self.doc_lens.get(d, 0) for d in self._deleted)
        return (self.total_len - deleted_len) / n

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    @property
    def params(self) -> dict[str, Any]:
        return {"k1": self.k1, "b": self.b}

    def stats(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "num_docs": self.num_docs,
            "num_deleted": len(self._deleted),
            "vocabulary_size": self.vocabulary_size,
            "avg_doc_len": self.avg_doc_len,
            "num_postings": sum(len(p) for p in self.postings.values()),
            "memory_bytes": self.memory_bytes(),
            **self.params,
        }

    def memory_bytes(self) -> int:
        """Rough footprint: posting tuples plus term strings plus doc lengths."""
        postings = sum(len(p) for p in self.postings.values()) * 60
        terms = sum(len(t) for t in self.postings) * 2 + len(self.postings) * 60
        return postings + terms + len(self.doc_lens) * 80

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def add(self, doc_id: InternalId, text: str) -> None:
        """Index one document. Re-adding the same id replaces its postings."""
        doc_id = int(doc_id)
        if doc_id in self.doc_lens:
            self.remove([doc_id], purge=True)

        tokens = tokenize(text)
        self.doc_lens[doc_id] = len(tokens)
        self.total_len += len(tokens)
        self._deleted.discard(doc_id)

        if tokens:
            freqs: dict[str, int] = defaultdict(int)
            for t in tokens:
                freqs[t] += 1
            for term, tf in freqs.items():
                self.postings.setdefault(term, []).append((doc_id, tf))

        # N changed, so every cached IDF is stale.
        self._idf_cache.clear()
        self._invalidate_np()

    def add_batch(self, docs: Iterable[tuple[InternalId, str]]) -> None:
        """Bulk build. Same result as repeated :meth:`add`, one IDF flush."""
        for doc_id, text in docs:
            self.add(doc_id, text)

    def remove(self, ids: Sequence[InternalId], purge: bool = False) -> None:
        """Tombstone documents, or physically drop them when ``purge``.

        Tombstoning is O(1) and consistent with ADR-010; physical removal has to
        walk the whole posting table, so it happens only on ``optimize()`` and on
        the re-add path (where leaving stale postings would double-count).
        """
        ids = [int(i) for i in ids]
        if not purge:
            self._deleted.update(i for i in ids if i in self.doc_lens)
            self._idf_cache.clear()
            self._invalidate_np()
            return

        drop = {i for i in ids if i in self.doc_lens}
        if not drop:
            return
        for i in drop:
            self.total_len -= self.doc_lens.pop(i, 0)
            self._deleted.discard(i)
        empty_terms = []
        for term, plist in self.postings.items():
            filtered = [(d, tf) for d, tf in plist if d not in drop]
            if len(filtered) != len(plist):
                if filtered:
                    self.postings[term] = filtered
                else:
                    empty_terms.append(term)
        for term in empty_terms:
            del self.postings[term]
        self._idf_cache.clear()
        self._invalidate_np()

    def optimize(self) -> dict[str, Any]:
        """Physically remove tombstoned docs, reclaiming posting-list space."""
        removed = len(self._deleted)
        self.remove(sorted(self._deleted), purge=True)
        self._deleted.clear()
        return {"purged_docs": removed, "vocabulary_size": self.vocabulary_size}

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def idf(self, term: str) -> float:
        """``ln((N - df + 0.5) / (df + 0.5) + 1)``.

        The ``+1`` inside the log is the Lucene/Elasticsearch variant. It keeps
        IDF non-negative: without it, a term appearing in more than half the
        corpus gets a negative weight, and a document can be *penalised* for
        matching a query term, which surprises everyone who hits it.
        """
        cached = self._idf_cache.get(term)
        if cached is not None:
            return cached

        plist = self.postings.get(term)
        if not plist:
            self._idf_cache[term] = 0.0
            return 0.0

        df = sum(1 for d, _ in plist if d not in self._deleted)
        n = self.num_docs
        if df == 0 or n == 0:
            self._idf_cache[term] = 0.0
            return 0.0
        value = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        self._idf_cache[term] = value
        return value

    def search(
        self,
        text: str,
        k: int = 10,
        *,
        exclude: set[InternalId] | None = None,
        **_params: Any,
    ) -> list[tuple[InternalId, float]]:
        """Top-``k`` documents by BM25 score. Higher score is better.

        The accumulator from LEARNING.md layer 3 — for each query term, walk its
        posting list and add that term's contribution to every document's running
        score — but vectorised. Only documents containing at least one query term
        are ever touched, which is what makes sparse retrieval cheap regardless of
        vocabulary size.

        **Why vectorised.** Cost is proportional to the total length of the posting
        lists the query terms touch, and real text is Zipfian: a head term's
        posting list covers most of the corpus. The obvious Python loop over
        ``(doc_id, tf)`` tuples measured **32 ms per query** on 20k documents with a
        Zipf-distributed vocabulary, i.e. ~30 QPS, and the profile was all
        interpreter overhead rather than arithmetic. Restructuring the same
        arithmetic as array ops took it to **3.7 ms (8.7x)** with identical
        rankings.

        Production engines go further with skip lists and block-max WAND to avoid
        touching full posting lists at all; that is the next step, not something
        this does.
        """
        if k <= 0:
            return []
        terms = tokenize(text)
        if not terms or not self.doc_lens:
            return []

        dead = self._deleted if not exclude else self._deleted | set(exclude)
        avgdl = self.avg_doc_len
        if avgdl <= 0:
            return []

        postings, doclen = self._numpy_view()
        k1 = self.k1
        b = self.b

        id_blocks: list[np.ndarray] = []
        contribution_blocks: list[np.ndarray] = []

        # Repeated query terms legitimately count twice in the sum, so iterate the
        # multiset rather than deduplicating.
        for term in terms:
            entry = postings.get(term)
            if entry is None:
                continue
            idf = self.idf(term)
            if idf == 0.0:
                continue
            ids, tfs = entry
            # float64 throughout: matching the scalar path bit-for-bit keeps the
            # hand-checked test values and the deterministic tie order valid.
            norm = k1 * (1.0 - b + b * (doclen[ids] / avgdl))
            id_blocks.append(ids)
            contribution_blocks.append(idf * (tfs * (k1 + 1.0)) / (tfs + norm))

        if not id_blocks:
            return []

        all_ids = np.concatenate(id_blocks)
        all_contributions = np.concatenate(contribution_blocks)

        # Group contributions by document. `unique` returns candidates in
        # ascending id order, which combines with the stable sort below to give
        # exactly the documented (-score, doc_id) tie-break.
        candidates, inverse = np.unique(all_ids, return_inverse=True)
        scores = np.bincount(
            inverse, weights=all_contributions, minlength=candidates.shape[0]
        )

        order = np.argsort(-scores, kind="stable")
        out: list[tuple[InternalId, float]] = []
        for position in order:
            doc_id = int(candidates[position])
            if doc_id in dead:
                continue
            out.append((doc_id, float(scores[position])))
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------------ #
    # Derived NumPy view of the posting table
    # ------------------------------------------------------------------ #

    def _invalidate_np(self) -> None:
        self._np_postings = None
        self._np_doclen = None

    def _numpy_view(self) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """Build (and cache) the array form of ``postings`` and ``doc_lens``.

        Rebuilt from scratch after any mutation. That is the right trade for a
        search-heavy workload — the collection rebuilds this once per batch of
        inserts and then amortises it over every subsequent query — but it does
        make a strictly alternating insert/query workload pay for the rebuild each
        time. Fixing that would mean incrementally maintaining the arrays, which
        is a real amount of bookkeeping for a case no benchmark here exercises.
        """
        if self._np_postings is not None and self._np_doclen is not None:
            return self._np_postings, self._np_doclen

        max_id = max(self.doc_lens) + 1 if self.doc_lens else 1
        doclen = np.zeros(max_id, dtype=np.float64)
        for doc_id, length in self.doc_lens.items():
            doclen[doc_id] = length

        postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, plist in self.postings.items():
            count = len(plist)
            ids = np.fromiter((d for d, _ in plist), dtype=np.int64, count=count)
            tfs = np.fromiter((f for _, f in plist), dtype=np.float64, count=count)
            postings[term] = (ids, tfs)

        self._np_postings = postings
        self._np_doclen = doclen
        return postings, doclen

    def explain(self, text: str, doc_id: InternalId) -> dict[str, Any]:
        """Per-term score breakdown for one document.

        Exists because hand-checking BM25 arithmetic against a tiny corpus is
        the only practical way to be sure the implementation is right, and
        ``tests/test_bm25.py`` does exactly that.
        """
        doc_id = int(doc_id)
        avgdl = self.avg_doc_len
        norm = (
            self.k1 * (1.0 - self.b + self.b * (self.doc_lens.get(doc_id, 0) / avgdl))
            if avgdl > 0
            else 0.0
        )
        parts = []
        total = 0.0
        for term in tokenize(text):
            plist = self.postings.get(term, [])
            tf = next((f for d, f in plist if d == doc_id), 0)
            idf = self.idf(term)
            contribution = (
                idf * (tf * (self.k1 + 1.0)) / (tf + norm) if tf else 0.0
            )
            total += contribution
            parts.append(
                {
                    "term": term,
                    "tf": tf,
                    "df": sum(1 for d, _ in plist if d not in self._deleted),
                    "idf": idf,
                    "contribution": contribution,
                }
            )
        return {
            "doc_id": doc_id,
            "doc_len": self.doc_lens.get(doc_id, 0),
            "avg_doc_len": avgdl,
            "num_docs": self.num_docs,
            "terms": parts,
            "score": total,
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    # ARCHITECTURE.md §4 startup path says the BM25 index can simply be rebuilt
    # from metadata on load ("fast, ~seconds for 1M docs"), and Collection does
    # exactly that. These methods exist for the cases where re-tokenising the
    # whole corpus is not wanted — tests, and standalone use of the index.

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "total_len": self.total_len,
            "doc_lens": {str(d): n for d, n in self.doc_lens.items()},
            "deleted": sorted(self._deleted),
            # Flattened to two parallel arrays per term: JSON arrays of pairs
            # would triple the file size in brackets alone.
            "postings": {
                term: [[d for d, _ in plist], [tf for _, tf in plist]]
                for term, plist in self.postings.items()
            },
        }
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

    def load(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        self.k1 = float(payload.get("k1", DEFAULT_K1))
        self.b = float(payload.get("b", DEFAULT_B))
        self.total_len = int(payload.get("total_len", 0))
        self.doc_lens = {int(d): int(n) for d, n in payload["doc_lens"].items()}
        self._deleted = {int(i) for i in payload.get("deleted", [])}
        self.postings = {
            term: list(zip((int(d) for d in ids), (int(f) for f in freqs)))
            for term, (ids, freqs) in payload["postings"].items()
        }
        self._idf_cache.clear()
        self._invalidate_np()
