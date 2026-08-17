"""Dataset loading for the benchmarks.

BENCHMARKS.md pins the datasets and their sources:

* **SIFT-1M** — 1,000,000 x 128, L2, with 10,000 queries and precomputed 100-NN
  ground truth. ``sift-128-euclidean.hdf5`` from ANN-Benchmarks.
* **GloVe-100** — 1,183,514 x 100, cosine. ``glove-100-angular.hdf5``, same source.
* **MS MARCO passages** — a 100k-passage subset, embedded with
  ``all-MiniLM-L6-v2`` (ADR-008: sentence-transformers is a demo/benchmark-only
  dependency and never imported from inside ``pyvec/``).

URLs are pinned, downloads are cached under ``benchmarks/datasets/``, and every
loader can produce a **seeded synthetic stand-in** instead. The stand-in exists so
that ``python -m benchmarks.sift_1m --synthetic`` runs on a laptop with no network
and no 500MB download — useful for checking the harness end to end. It is clearly
labelled in the output so a synthetic number can never be mistaken for a real one.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Dataset", "load_sift", "load_glove", "load_msmarco", "DATASETS_DIR"]

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

#: Pinned URLs. ANN-Benchmarks hosts these HDF5 files and has for years; pinning
#: the exact filename is what makes a run reproducible months later. HTTPS because
#: the plain-HTTP endpoint redirects anyway.
SIFT_URL = "https://ann-benchmarks.com/sift-128-euclidean.hdf5"
GLOVE_URL = "https://ann-benchmarks.com/glove-100-angular.hdf5"

#: Hugging Face dataset id for the hybrid evaluation corpus.
MSMARCO_DATASET = "microsoft/ms_marco"

#: The host rejects urllib's default ``Python-urllib/3.x`` agent with HTTP 403, so
#: send something it will serve. Not cleverness — just the price of fetching from
#: a CDN that filters obvious scripts.
_USER_AGENT = (
    "Mozilla/5.0 (compatible; pyvec-benchmarks/0.1; "
    "+https://github.com/pyvec/pyvec)"
)


@dataclass(slots=True)
class Dataset:
    """A benchmark dataset in the shape every benchmark script wants."""

    name: str
    train: np.ndarray  # (n, dim) float32 — the vectors to index
    test: np.ndarray  # (n_queries, dim) float32
    metric: str  # "l2" | "cosine" | "dot"
    neighbours: np.ndarray | None = None  # (n_queries, k) int64 ground truth
    synthetic: bool = False

    @property
    def dim(self) -> int:
        return int(self.train.shape[1])

    @property
    def size(self) -> int:
        return int(self.train.shape[0])

    def describe(self) -> str:
        tag = "  [SYNTHETIC STAND-IN, not the real dataset]" if self.synthetic else ""
        truth = "precomputed" if self.neighbours is not None else "none"
        return (
            f"{self.name}: {self.size:,} x {self.dim} vectors, "
            f"{self.test.shape[0]:,} queries, metric={self.metric}, "
            f"ground truth={truth}{tag}"
        )

    def subset(self, n: int | None, n_queries: int | None = None) -> "Dataset":
        """Take the first ``n`` vectors and ``n_queries`` queries.

        Ground truth is dropped rather than truncated: neighbour ids computed over
        1M vectors are simply wrong for a 100k prefix, and silently keeping them
        would produce recall numbers that look plausible and mean nothing.
        """
        if n is None and n_queries is None:
            return self
        train = self.train if n is None else self.train[:n]
        test = self.test if n_queries is None else self.test[:n_queries]
        shrunk = n is not None and n < self.size

        neighbours = self.neighbours
        if shrunk:
            neighbours = None
        elif neighbours is not None and n_queries is not None:
            # Ground truth is one row per query; keep the rows aligned with the
            # queries that survive. Leaving 10k rows against 1k queries happens to
            # work because zip() stops at the shorter side, but it is the kind of
            # accidental correctness that breaks the moment someone indexes
            # directly instead of zipping.
            neighbours = neighbours[:n_queries]

        return Dataset(
            # Only rename when the vector set actually shrank. Taking a query
            # subset does not change *which* dataset this is, and renaming on
            # every call produced names like "sift-1m[:100000][:100000]".
            name=f"{self.name}-{train.shape[0]}" if shrunk else self.name,
            train=train,
            test=test,
            metric=self.metric,
            neighbours=neighbours,
            synthetic=self.synthetic,
        )


# --------------------------------------------------------------------------- #
# Download + HDF5
# --------------------------------------------------------------------------- #


def _download(url: str, dest: Path) -> Path:
    """Fetch to a ``.part`` file, then rename — so an interrupted download is
    never mistaken for a complete one on the next run."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}\n       -> {dest}", file=sys.stderr)
    tmp = dest.with_suffix(dest.suffix + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_report = [0.0]
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                f.write(block)
                done += len(block)
                # Throttle progress output: this runs to a log file as often as a
                # terminal, and 500 lines of progress is noise either way.
                now = time.monotonic()
                if now - last_report[0] > 2.0 or done == total:
                    last_report[0] = now
                    pct = f"{100 * done / total:5.1f}%" if total else "  ?  "
                    print(
                        f"  {pct}  {done >> 20:,} / {total >> 20:,} MiB",
                        file=sys.stderr, flush=True,
                    )
    if total and done != total:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{url}: expected {total} bytes, got {done}. Download incomplete."
        )
    tmp.replace(dest)
    return dest


def _load_ann_benchmarks_hdf5(url: str, name: str, metric: str) -> Dataset:
    try:
        import h5py
    except ImportError:
        raise RuntimeError(
            f"reading {name} needs h5py. Install the benchmark extras:\n"
            f"    pip install -e '.[demo]'\n"
            f"or run with --synthetic to exercise the harness without the "
            f"real dataset."
        ) from None

    path = _download(url, DATASETS_DIR / Path(url).name)
    with h5py.File(path, "r") as f:
        train = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        test = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        neighbours = (
            np.ascontiguousarray(f["neighbors"][:], dtype=np.int64)
            if "neighbors" in f
            else None
        )
    return Dataset(
        name=name, train=train, test=test, metric=metric, neighbours=neighbours
    )


# --------------------------------------------------------------------------- #
# Synthetic stand-ins
# --------------------------------------------------------------------------- #


def _synthetic(
    name: str,
    n: int,
    dim: int,
    metric: str,
    n_queries: int = 1000,
    clusters: int = 100,
    seed: int = 42,
) -> Dataset:
    """Clustered Gaussian data, seeded.

    Clustered rather than uniform because real embedding datasets cluster, and a
    uniform stand-in would make IVF look far worse than it does on anything real
    (LEARNING.md layer 2). Queries are drawn from the same distribution, as they
    are in SIFT and GloVe.

    Cluster separation is deliberately mild — comparable to the within-cluster
    noise — so the clusters overlap and neighbours are genuinely contested. With
    well-separated blobs every index scores recall 1.0 and the sweep tells you
    nothing.

    Even so: **ANN difficulty scales with n.** At a few thousand vectors the true
    neighbours are so isolated that any index finds them all, and recall saturates
    at 1.0 regardless of parameters. The synthetic mode is for checking that the
    harness runs end to end, not for judging index quality — the real dataset is
    what produces meaningful recall curves.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=1.0, size=(clusters, dim)).astype(np.float32)
    assign = rng.integers(0, clusters, size=n)
    train = (centres[assign] + rng.normal(size=(n, dim))).astype(np.float32)
    q_assign = rng.integers(0, clusters, size=n_queries)
    test = (centres[q_assign] + rng.normal(size=(n_queries, dim))).astype(np.float32)
    if metric == "cosine":
        train /= np.linalg.norm(train, axis=1, keepdims=True)
        test /= np.linalg.norm(test, axis=1, keepdims=True)
    return Dataset(
        name=f"{name}-synthetic", train=train, test=test, metric=metric,
        neighbours=None, synthetic=True,
    )


# --------------------------------------------------------------------------- #
# Public loaders
# --------------------------------------------------------------------------- #


def load_sift(synthetic: bool = False, n: int | None = None) -> Dataset:
    """SIFT-1M: 1M x 128, L2. The primary benchmark dataset."""
    if synthetic:
        return _synthetic("sift", n or 100_000, 128, "l2")
    return _load_ann_benchmarks_hdf5(SIFT_URL, "sift-1m", "l2").subset(n)


def load_glove(synthetic: bool = False, n: int | None = None) -> Dataset:
    """GloVe-100: 1.18M x 100, cosine. The "text embedding" domain."""
    if synthetic:
        return _synthetic("glove", n or 100_000, 100, "cosine")
    return _load_ann_benchmarks_hdf5(GLOVE_URL, "glove-100", "cosine").subset(n)


def load_msmarco(
    n_passages: int = 100_000,
    synthetic: bool = False,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """MS MARCO passages for the hybrid evaluation.

    Returns ``(passages, queries)`` where a passage is
    ``{"id", "text", "vector"}`` and a query is
    ``{"id", "text", "vector", "relevant": {passage_id: gain}}``.

    The real path needs ``datasets`` and ``sentence-transformers`` (ADR-008,
    ``pip install -e '.[demo]'``). The synthetic path builds a corpus with known
    relevance judgements, which is enough to verify the evaluation pipeline and
    to demonstrate the hybrid-over-dense effect, but is *not* an MS MARCO number
    and is labelled as such wherever it is reported.
    """
    if synthetic:
        return _synthetic_text_corpus(n_passages, seed=seed)

    # Embedding 81k passages takes ~35 minutes on CPU. Cache the result: without
    # this, any question about the *evaluation* (does rrf_k matter? are the
    # candidate counts too small?) costs another half hour before you can even
    # look, which in practice means the question does not get asked.
    cache = DATASETS_DIR / f"msmarco_{n_passages}_{Path(model).name}.npz"
    if cache.exists():
        print(f"loading cached MS MARCO embeddings from {cache.name}", file=sys.stderr)
        with np.load(cache, allow_pickle=True) as z:
            passages = [
                {"id": str(i), "text": str(t), "vector": v}
                for i, t, v in zip(z["p_ids"], z["p_texts"], z["p_vecs"])
            ]
            queries = [
                {"id": str(i), "text": str(t), "vector": v,
                 "relevant": json.loads(str(r))}
                for i, t, v, r in zip(
                    z["q_ids"], z["q_texts"], z["q_vecs"], z["q_rel"]
                )
            ]
        return passages, queries

    try:
        from datasets import load_dataset  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        raise RuntimeError(
            "the MS MARCO evaluation needs the demo extras:\n"
            "    pip install -e '.[demo]' datasets\n"
            "or run with --synthetic to exercise the evaluation pipeline."
        ) from None

    print(f"loading MS MARCO v1.1 and embedding with {model}", file=sys.stderr)
    encoder = SentenceTransformer(model)
    # Fully-qualified repo id: `huggingface_hub` now rejects a bare `ms_marco`
    # ("Repository id must be 'namespace/name'"), so the old short form fails at
    # resolve time rather than falling back.
    raw = load_dataset(MSMARCO_DATASET, "v1.1", split="validation")

    passages: list[dict] = []
    queries: list[dict] = []
    seen: dict[str, int] = {}
    for row in raw:
        if len(passages) >= n_passages:
            break
        relevant: dict[str, float] = {}
        texts = row["passages"]["passage_text"]
        selected = row["passages"]["is_selected"]
        for text, is_selected in zip(texts, selected):
            pid = seen.get(text)
            if pid is None:
                pid = len(passages)
                seen[text] = pid
                passages.append({"id": f"p{pid}", "text": text})
            if is_selected:
                relevant[f"p{pid}"] = 1.0
        if relevant:
            queries.append(
                {"id": f"q{len(queries)}", "text": row["query"], "relevant": relevant}
            )

    print(f"  embedding {len(passages):,} passages", file=sys.stderr)
    p_vecs = encoder.encode(
        [p["text"] for p in passages], batch_size=256,
        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True,
    )
    q_vecs = encoder.encode(
        [q["text"] for q in queries], batch_size=256,
        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True,
    )
    for p, v in zip(passages, p_vecs):
        p["vector"] = np.asarray(v, dtype=np.float32)
    for q, v in zip(queries, q_vecs):
        q["vector"] = np.asarray(v, dtype=np.float32)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        p_ids=np.array([p["id"] for p in passages]),
        p_texts=np.array([p["text"] for p in passages]),
        p_vecs=np.stack([p["vector"] for p in passages]),
        q_ids=np.array([q["id"] for q in queries]),
        q_texts=np.array([q["text"] for q in queries]),
        q_vecs=np.stack([q["vector"] for q in queries]),
        q_rel=np.array([json.dumps(q["relevant"]) for q in queries]),
    )
    print(f"cached embeddings to {cache.name}", file=sys.stderr)
    return passages, queries


#: Vocabulary for the synthetic text corpus.
#:
#: The corpus is built so that **neither retriever alone can answer the query**,
#: which is the only configuration that actually tests fusion:
#:
#: * ``_FILLER`` is shared by every passage regardless of topic. It therefore has
#:   near-zero IDF and gives BM25 no way to tell topics apart. The *topic* lives
#:   only in the embedding.
#: * ``_RARE`` terms are assigned independently of topic. A rare term has high IDF,
#:   so BM25 can find every passage carrying it — but those passages are spread
#:   across all topics, so BM25 cannot tell which one is on-topic.
#:
#: A query names one topic (via its vector) and one rare term (via its text), and
#: only passages matching *both* are relevant. Dense retrieval narrows to the
#: right topic and cannot see the term; BM25 narrows to the right term and cannot
#: see the topic; RRF, which rewards agreement, ranks the intersection first.
#:
#: An earlier version gave each topic its own distinctive words, and BM25 scored a
#: perfect 1.0 — the lexical signal alone identified the answer, so the benchmark
#: measured nothing about fusion.
_FILLER = (
    "the a of and to in is for with on at by from as that this it be are was "
    "system data value result process method example case"
).split()

_TOPIC_COUNT = 8

_RARE = [
    "zygomorphic", "quixotic", "brobdingnagian", "sesquipedalian",
    "antidisestablishmentarian", "borborygmus", "flibbertigibbet", "grandiloquent",
]


def _synthetic_text_corpus(
    n_passages: int, seed: int = 42, dim: int = 64, rare_fraction: float = 0.05
) -> tuple[list[dict], list[dict]]:
    """A corpus where dense and sparse retrieval each hold half the answer.

    See the ``_FILLER`` / ``_RARE`` notes above for why it is built this way.
    Returns ``(passages, queries)`` with one query per (topic, rare term) pair
    that has at least one matching passage.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(_TOPIC_COUNT, dim)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    passages: list[dict] = []
    # (topic, rare term) -> passage ids matching both. These are the answers.
    needles: dict[tuple[int, str], list[str]] = {}

    for i in range(n_passages):
        topic = int(rng.integers(_TOPIC_COUNT))
        text_words = list(rng.choice(_FILLER, size=12, replace=True))
        rare = None
        if rng.random() < rare_fraction:
            # Chosen independently of the topic, so the term identifies a set of
            # passages spanning every topic.
            rare = _RARE[int(rng.integers(len(_RARE)))]
            text_words.insert(int(rng.integers(len(text_words))), rare)
            needles.setdefault((topic, rare), []).append(f"p{i}")

        vec = centres[topic] + rng.normal(scale=0.30, size=dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        passages.append(
            {
                "id": f"p{i}",
                "text": " ".join(text_words),
                "vector": vec.astype(np.float32),
                "topic": topic,
                "rare": rare,
            }
        )

    queries: list[dict] = []
    for (topic, rare), pids in sorted(needles.items()):
        # A query is only interesting when the answer set is small; a rare term
        # matching half a topic would be findable by BM25 alone.
        if len(pids) > 3:
            continue
        vec = centres[topic] + rng.normal(scale=0.05, size=dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        queries.append(
            {
                "id": f"q{topic}-{rare}",
                # Filler words plus the rare term: the text alone cannot express
                # which topic is wanted, exactly as a short real query often
                # cannot.
                "text": " ".join(list(rng.choice(_FILLER, size=4)) + [rare]),
                "vector": vec.astype(np.float32),
                "relevant": {pid: 1.0 for pid in pids},
            }
        )
    return passages, queries
