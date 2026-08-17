"""Thin HTTP client, per the sketch in API_SPEC.md.

Built on ``urllib`` from the standard library rather than ``requests``/``httpx``:
a client SDK that drags in a dependency tree is annoying to adopt, and the entire
surface here is "POST some JSON, parse the reply".

    from pyvec.client import PyVecClient

    c = PyVecClient("http://localhost:8080")
    c.create_collection("docs", dimension=384, metric="cosine",
                        index="hnsw", text_field="content")
    c.insert("docs", items=[{"id": "d1", "vector": vec,
                             "metadata": {"content": "..."}}])
    results = c.hybrid("docs", vector=q_vec, text="quick fox", k=10)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["PyVecClient", "PyVecHTTPError"]

DEFAULT_TIMEOUT = 30.0
#: Server-side cap (API_SPEC). Larger inserts are chunked client-side.
MAX_BATCH = 1000


class PyVecHTTPError(RuntimeError):
    """Non-2xx response. Carries the server's structured error payload."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"HTTP {status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


class PyVecClient:
    """Synchronous client for the PyVec HTTP API."""

    def __init__(
        self, base_url: str = "http://localhost:8080", timeout: float = DEFAULT_TIMEOUT
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #

    def _request(
        self, method: str, path: str, payload: Any | None = None
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
                if not body or resp.status == 204:
                    return None
                return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            code, message = "HTTP_ERROR", exc.reason or "request failed"
            try:
                parsed = json.loads(raw.decode("utf-8"))
                code = parsed.get("code", code)
                message = parsed.get("error", message)
            except (ValueError, UnicodeDecodeError):
                if raw:
                    message = raw.decode("utf-8", "replace")[:500]
            raise PyVecHTTPError(exc.code, code, message) from None
        except urllib.error.URLError as exc:
            raise PyVecHTTPError(0, "CONNECTION_FAILED", str(exc.reason)) from None

    # ------------------------------------------------------------------ #
    # Collections
    # ------------------------------------------------------------------ #

    def create_collection(
        self,
        name: str,
        dimension: int,
        *,
        metric: str = "cosine",
        index: str = "hnsw",
        index_params: Mapping[str, Any] | None = None,
        text_field: str | None = None,
        capacity: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "index": {"type": index, "params": dict(index_params or {})},
        }
        if text_field is not None:
            payload["text_field"] = text_field
        if capacity is not None:
            payload["capacity"] = capacity
        return self._request("POST", "/collections", payload)

    def list_collections(self) -> list[dict[str, Any]]:
        return self._request("GET", "/collections")["collections"]

    def describe(self, name: str) -> dict[str, Any]:
        return self._request("GET", f"/collections/{name}")

    def drop_collection(self, name: str) -> None:
        self._request("DELETE", f"/collections/{name}")

    # ------------------------------------------------------------------ #
    # Vectors
    # ------------------------------------------------------------------ #

    def insert(
        self,
        name: str,
        items: Sequence[Mapping[str, Any]],
        *,
        upsert: bool = False,
        batch_size: int = MAX_BATCH,
    ) -> dict[str, int]:
        """Insert records, chunking automatically past the server's batch limit."""
        batch_size = max(1, min(batch_size, MAX_BATCH))
        totals = {"inserted": 0, "duplicates_skipped": 0}
        for chunk in _chunks(items, batch_size):
            result = self._request(
                "POST",
                f"/collections/{name}/insert",
                {"items": [_normalise_item(i) for i in chunk], "upsert": upsert},
            )
            totals["inserted"] += result.get("inserted", 0)
            totals["duplicates_skipped"] += result.get("duplicates_skipped", 0)
        return totals

    def get(self, name: str, vector_id: str) -> dict[str, Any]:
        return self._request("GET", f"/collections/{name}/vectors/{vector_id}")

    def delete(self, name: str, vector_id: str) -> None:
        self._request("DELETE", f"/collections/{name}/vectors/{vector_id}")

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def query(
        self,
        name: str,
        vector: Sequence[float],
        k: int = 10,
        *,
        filter: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "vector": _as_floats(vector),
            "k": k,
            "params": dict(params or {}),
        }
        if filter:
            payload["filter"] = dict(filter)
        return self._request("POST", f"/collections/{name}/query", payload)["results"]

    def query_text(
        self,
        name: str,
        text: str,
        k: int = 10,
        *,
        filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"text": text, "k": k}
        if filter:
            payload["filter"] = dict(filter)
        return self._request(
            "POST", f"/collections/{name}/query/text", payload
        )["results"]

    def hybrid(
        self,
        name: str,
        vector: Sequence[float],
        text: str,
        k: int = 10,
        *,
        filter: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "vector": _as_floats(vector),
            "text": text,
            "k": k,
            "params": dict(params or {}),
        }
        if filter:
            payload["filter"] = dict(filter)
        return self._request(
            "POST", f"/collections/{name}/query/hybrid", payload
        )["results"]

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #

    def optimize(self, name: str) -> dict[str, Any]:
        return self._request("POST", f"/collections/{name}/optimize")

    def snapshot(self, name: str) -> dict[str, Any]:
        return self._request("POST", f"/collections/{name}/snapshot")

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def metrics(self) -> str:
        url = f"{self.base_url}/metrics"
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            return resp.read().decode("utf-8")


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _as_floats(vector: Sequence[float]) -> list[float]:
    """Accept NumPy arrays without making NumPy a client dependency."""
    tolist = getattr(vector, "tolist", None)
    if callable(tolist):
        return [float(v) for v in tolist()]
    return [float(v) for v in vector]


def _normalise_item(item: Mapping[str, Any]) -> dict[str, Any]:
    out = {
        "id": str(item["id"]),
        "vector": _as_floats(item["vector"]),
    }
    if item.get("metadata"):
        out["metadata"] = dict(item["metadata"])
    return out
