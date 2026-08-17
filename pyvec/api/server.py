"""FastAPI application.

Deliberately thin (ARCHITECTURE.md §1): validate, delegate, serialise, map
errors. Every endpoint in API_SPEC.md is implemented here and nothing else is.

Two notes on the async story. Handlers are declared ``def``, not ``async def``,
on purpose: the collection layer is synchronous and blocking (mmap reads, BLAS
calls, lock acquisition), and FastAPI runs plain ``def`` handlers in a thread
pool. Declaring them ``async`` would run that blocking work directly on the
event loop and stall every other request. The thread pool also means concurrent
searches genuinely overlap — NumPy releases the GIL inside BLAS calls, so the
read side of the RWLock earns its keep.

Run it::

    uvicorn pyvec.api.server:app --port 8080
    PYVEC_DATA_DIR=/var/lib/pyvec uvicorn pyvec.api.server:app --port 8080
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from pyvec import __version__
from pyvec.api.schemas import (
    CollectionDetail,
    CollectionSummary,
    CreateCollectionRequest,
    CreateCollectionResponse,
    HealthResponse,
    HybridQueryRequest,
    HybridResponse,
    HybridResult,
    InsertRequest,
    InsertResponse,
    ListCollectionsResponse,
    OptimizeResponse,
    QueryRequest,
    QueryResponse,
    QueryResult,
    SnapshotResponse,
    TextQueryRequest,
    VectorResponse,
)
from pyvec.core.collection_manager import CollectionManager
from pyvec.core.errors import PyVecError

__all__ = ["app", "create_app", "get_manager"]

#: Where collections live. Environment-driven so the same image can be pointed at
#: a mounted volume without a config file.
DATA_DIR_ENV = "PYVEC_DATA_DIR"
DEFAULT_DATA_DIR = "./data"

_START_TIME = time.monotonic()


def create_app(
    data_dir: str | Path | None = None,
    *,
    manager: CollectionManager | None = None,
) -> FastAPI:
    """Build an app instance.

    Args:
        data_dir: data root. Defaults to ``$PYVEC_DATA_DIR`` then ``./data``.
        manager: inject a prebuilt manager. Tests use this to point each test
            at its own tmpdir without touching the environment.
    """
    root = Path(
        data_dir or os.environ.get(DATA_DIR_ENV) or DEFAULT_DATA_DIR
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        # Startup: open every collection on disk, replaying WALs as needed
        # (ARCHITECTURE.md startup/recovery path).
        #
        # `is not None`, not `or`: CollectionManager defines __len__, so a manager
        # holding zero collections is falsy. With `or` an injected empty manager
        # was silently discarded and replaced by one pointing at the default data
        # directory — which is exactly the state every test injects.
        application.state.manager = (
            manager if manager is not None else CollectionManager(root)
        )
        application.state.started_at = time.monotonic()
        yield
        # Shutdown: checkpoint everything so the next start has nothing to
        # replay. A hard kill skips this, which is what the WAL is for.
        application.state.manager.close()

    application = FastAPI(
        title="PyVec",
        version=__version__,
        summary=(
            "A single-node vector database built from scratch in Python: "
            "HNSW, IVF-Flat, BM25 and RRF hybrid search."
        ),
        lifespan=lifespan,
    )
    _register_error_handlers(application)
    _register_routes(application)
    return application


def get_manager(request: Request) -> CollectionManager:
    return request.app.state.manager


# --------------------------------------------------------------------------- #
# Error mapping
# --------------------------------------------------------------------------- #


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(PyVecError)
    async def _domain_error(_request: Request, exc: PyVecError) -> JSONResponse:
        # Every domain exception carries its own code and status, so this is the
        # entire translation layer (see pyvec/core/errors.py).
        return JSONResponse(
            status_code=exc.status,
            content={"error": exc.message, "code": exc.code},
        )

    @app.exception_handler(ValueError)
    async def _value_error(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400, content={"error": str(exc), "code": "INVALID_REQUEST"}
        )

    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Pydantic rejects an oversized batch before the handler runs; surface it
        # as the documented 413 rather than a generic 422.
        detail = exc.errors()
        for err in detail:
            if err.get("type") == "too_long":
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "batch exceeds the 1000-item limit",
                        "code": "PAYLOAD_TOO_LARGE",
                    },
                )
        return JSONResponse(
            status_code=400,
            content={
                "error": "request validation failed",
                "code": "INVALID_REQUEST",
                "detail": _jsonable(detail),
            },
        )


def _jsonable(value: Any) -> Any:
    """Pydantic error payloads can embed exception objects, which JSON hates."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# --------------------------------------------------------------------------- #
# Timing helper
# --------------------------------------------------------------------------- #


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run ``fn`` and return ``(result, elapsed_ms)`` for the ``took_ms`` field."""
    start = time.perf_counter()
    result = fn()
    return result, round((time.perf_counter() - start) * 1000, 3)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


def _register_routes(app: FastAPI) -> None:
    # ---------------------------- collections --------------------------- #

    @app.post("/collections", status_code=201, response_model=CreateCollectionResponse)
    def create_collection(
        body: CreateCollectionRequest, request: Request
    ) -> CreateCollectionResponse:
        collection = get_manager(request).create(
            name=body.name,
            dimension=body.dimension,
            metric=body.metric,
            index_type=body.index.type,
            index_params=body.index.params,
            text_field=body.text_field,
            capacity=body.capacity,
            bm25_params=body.bm25_params,
        )
        return CreateCollectionResponse(
            name=collection.name, created_at=collection.created_at
        )

    @app.get("/collections", response_model=ListCollectionsResponse)
    def list_collections(request: Request) -> ListCollectionsResponse:
        return ListCollectionsResponse(
            collections=[
                CollectionSummary(**row) for row in get_manager(request).list()
            ]
        )

    @app.get("/collections/{name}", response_model=CollectionDetail)
    def get_collection(name: str, request: Request) -> CollectionDetail:
        return CollectionDetail(**get_manager(request).get(name).stats())

    @app.delete("/collections/{name}", status_code=204)
    def drop_collection(name: str, request: Request) -> Response:
        get_manager(request).drop(name)
        return Response(status_code=204)

    # ------------------------------ vectors ----------------------------- #

    @app.post(
        "/collections/{name}/insert", status_code=201, response_model=InsertResponse
    )
    def insert(name: str, body: InsertRequest, request: Request) -> InsertResponse:
        collection = get_manager(request).get(name)
        result = collection.insert(
            [item.model_dump() for item in body.items], upsert=body.upsert
        )
        return InsertResponse(**result)

    @app.get("/collections/{name}/vectors/{vector_id}", response_model=VectorResponse)
    def get_vector(name: str, vector_id: str, request: Request) -> VectorResponse:
        return VectorResponse(**get_manager(request).get(name).get(vector_id))

    @app.delete("/collections/{name}/vectors/{vector_id}", status_code=204)
    def delete_vector(name: str, vector_id: str, request: Request) -> Response:
        get_manager(request).get(name).delete(vector_id)
        return Response(status_code=204)

    # ------------------------------ queries ----------------------------- #

    @app.post("/collections/{name}/query", response_model=QueryResponse)
    def query(name: str, body: QueryRequest, request: Request) -> QueryResponse:
        collection = get_manager(request).get(name)
        hits, took = _timed(
            lambda: collection.search(
                body.vector, k=body.k, filter=body.filter, params=body.params
            )
        )
        return QueryResponse(
            results=[
                QueryResult(id=h.id, score=h.score, metadata=h.metadata)
                for h in hits
            ],
            took_ms=took,
        )

    @app.post("/collections/{name}/query/text", response_model=QueryResponse)
    def query_text(
        name: str, body: TextQueryRequest, request: Request
    ) -> QueryResponse:
        collection = get_manager(request).get(name)
        hits, took = _timed(
            lambda: collection.search_text(body.text, k=body.k, filter=body.filter)
        )
        return QueryResponse(
            results=[
                QueryResult(id=h.id, score=h.score, metadata=h.metadata)
                for h in hits
            ],
            took_ms=took,
        )

    @app.post("/collections/{name}/query/hybrid", response_model=HybridResponse)
    def query_hybrid(
        name: str, body: HybridQueryRequest, request: Request
    ) -> HybridResponse:
        collection = get_manager(request).get(name)
        hits, took = _timed(
            lambda: collection.search_hybrid(
                body.vector,
                body.text,
                k=body.k,
                filter=body.filter,
                params=body.params,
            )
        )
        return HybridResponse(
            results=[
                HybridResult(
                    id=h.id,
                    rrf_score=h.score,
                    dense_rank=h.ranks.get("dense"),
                    sparse_rank=h.ranks.get("sparse"),
                    metadata=h.metadata,
                )
                for h in hits
            ],
            took_ms=took,
        )

    # ---------------------------- maintenance --------------------------- #

    @app.post(
        "/collections/{name}/optimize", status_code=202, response_model=OptimizeResponse
    )
    def optimize(name: str, request: Request) -> OptimizeResponse:
        # API_SPEC: "For a learning project this can be synchronous and just
        # block." It does — the job_id is returned for interface compatibility
        # with the async version, and status is already terminal.
        result = get_manager(request).get(name).optimize()
        return OptimizeResponse(
            job_id=f"opt-{uuid.uuid4().hex[:8]}",
            status="completed",
            compacted=result["compacted"],
            num_vectors=result["num_vectors"],
            took_ms=result["took_ms"],
        )

    @app.post(
        "/collections/{name}/snapshot", status_code=201, response_model=SnapshotResponse
    )
    def snapshot(name: str, request: Request) -> SnapshotResponse:
        return SnapshotResponse(**get_manager(request).get(name).snapshot())

    # -------------------------- health and meta ------------------------- #

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            uptime_s=round(time.monotonic() - request.app.state.started_at, 3),
            version=__version__,
            collections=len(get_manager(request)),
        )

    @app.get("/metrics")
    def metrics(request: Request) -> Response:
        """Prometheus text exposition format.

        Hand-rolled rather than pulling in ``prometheus_client``: it is a
        well-specified text format and this is a handful of lines. Counters are
        per-collection, which is the dimension anyone debugging would slice by.
        """
        manager = get_manager(request)
        lines: list[str] = [
            "# HELP pyvec_uptime_seconds Process uptime.",
            "# TYPE pyvec_uptime_seconds gauge",
            f"pyvec_uptime_seconds {time.monotonic() - request.app.state.started_at:.3f}",
            "# HELP pyvec_collections Number of open collections.",
            "# TYPE pyvec_collections gauge",
            f"pyvec_collections {len(manager)}",
            "# HELP pyvec_vectors Live vectors per collection.",
            "# TYPE pyvec_vectors gauge",
            "# HELP pyvec_deleted_vectors Tombstoned vectors per collection.",
            "# TYPE pyvec_deleted_vectors gauge",
            "# HELP pyvec_operations_total Operations served per collection.",
            "# TYPE pyvec_operations_total counter",
        ]
        for row in manager.list():
            collection = manager.get(row["name"])
            label = f'collection="{_escape(collection.name)}"'
            lines.append(f"pyvec_vectors{{{label}}} {len(collection)}")
            lines.append(
                f"pyvec_deleted_vectors{{{label}}} {collection.num_deleted}"
            )
            for op, count in collection.stats_counters.items():
                lines.append(
                    f'pyvec_operations_total{{{label},op="{op}"}} {count}'
                )
        return Response(
            content="\n".join(lines) + "\n",
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "name": "pyvec",
            "version": __version__,
            "docs": "/docs",
            "endpoints": [
                "POST   /collections",
                "GET    /collections",
                "GET    /collections/{name}",
                "DELETE /collections/{name}",
                "POST   /collections/{name}/insert",
                "POST   /collections/{name}/query",
                "POST   /collections/{name}/query/text",
                "POST   /collections/{name}/query/hybrid",
                "GET    /collections/{name}/vectors/{id}",
                "DELETE /collections/{name}/vectors/{id}",
                "POST   /collections/{name}/optimize",
                "POST   /collections/{name}/snapshot",
                "GET    /health",
                "GET    /metrics",
            ],
        }


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "")


#: Module-level app so ``uvicorn pyvec.api.server:app`` works as documented.
app = create_app()
