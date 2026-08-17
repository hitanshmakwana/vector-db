"""Command line interface.

PRD F15/G7: a small CLI for common operations. Two families of subcommand:

* ``serve`` runs the HTTP server locally.
* everything else is a wrapper over :class:`~pyvec.client.PyVecClient`, so the
  CLI exercises exactly the same API surface an external integrator would.

    pyvec serve --port 8080 --data-dir ./data
    pyvec create docs --dimension 384 --metric cosine --index hnsw --text-field content
    pyvec insert docs --file vectors.jsonl
    pyvec query docs --vector "0.1,0.2,..." -k 5
    pyvec hybrid docs --vector-file q.json --text "quick fox"
    pyvec ls
    pyvec describe docs
    pyvec rm docs --yes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from pyvec import __version__
from pyvec.client import PyVecClient, PyVecHTTPError

__all__ = ["main", "build_parser"]

DEFAULT_URL = "http://localhost:8080"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyvec",
        # ASCII only in help text: argparse writes it to stdout, and a Windows
        # console under a legacy code page cannot encode an em-dash.
        description="PyVec - a vector database built from scratch in Python.",
    )
    parser.add_argument("--version", action="version", version=f"pyvec {__version__}")
    parser.add_argument(
        "--url", default=DEFAULT_URL, help=f"server base URL (default {DEFAULT_URL})"
    )
    parser.add_argument(
        "--json", action="store_true", help="emit raw JSON instead of a table"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------- serve ------------------------------ #
    p = sub.add_parser("serve", help="run the HTTP server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--data-dir", default=None, help="collection storage root")
    p.add_argument("--reload", action="store_true", help="auto-reload on edit")

    # ---------------------------- collections --------------------------- #
    p = sub.add_parser("create", help="create a collection")
    p.add_argument("name")
    p.add_argument("--dimension", type=int, required=True)
    p.add_argument("--metric", default="cosine", choices=["cosine", "l2", "dot"])
    p.add_argument("--index", default="hnsw", choices=["hnsw", "ivf", "flat"])
    p.add_argument(
        "--index-params",
        default=None,
        help='JSON, e.g. \'{"M": 16, "ef_construction": 200}\'',
    )
    p.add_argument("--text-field", default=None, help="metadata field to BM25-index")
    p.add_argument("--capacity", type=int, default=None)

    sub.add_parser("ls", help="list collections")

    p = sub.add_parser("describe", help="show collection stats")
    p.add_argument("name")

    p = sub.add_parser("rm", help="drop a collection and its files")
    p.add_argument("name")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    # ------------------------------ vectors ----------------------------- #
    p = sub.add_parser("insert", help="insert vectors from a JSON/JSONL file")
    p.add_argument("name")
    p.add_argument(
        "--file",
        required=True,
        help="JSONL of {id, vector, metadata} records, or a JSON array of them; "
        "'-' reads stdin",
    )
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--upsert", action="store_true")

    p = sub.add_parser("get", help="fetch one vector by id")
    p.add_argument("name")
    p.add_argument("id")

    p = sub.add_parser("delete", help="delete one vector by id")
    p.add_argument("name")
    p.add_argument("id")

    # ------------------------------ queries ----------------------------- #
    for cmd, helptext in (
        ("query", "dense vector search"),
        ("hybrid", "dense + BM25 search fused with RRF"),
    ):
        p = sub.add_parser(cmd, help=helptext)
        p.add_argument("name")
        p.add_argument("--vector", default=None, help="comma-separated floats")
        p.add_argument("--vector-file", default=None, help="JSON array of floats")
        if cmd == "hybrid":
            p.add_argument("--text", required=True)
        p.add_argument("-k", type=int, default=10)
        p.add_argument("--filter", default=None, help="JSON metadata filter")
        p.add_argument("--params", default=None, help="JSON index params")

    p = sub.add_parser("search-text", help="BM25 search")
    p.add_argument("name")
    p.add_argument("text")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--filter", default=None, help="JSON metadata filter")

    # ---------------------------- maintenance --------------------------- #
    p = sub.add_parser("optimize", help="compact tombstones, rebuild the index")
    p.add_argument("name")

    p = sub.add_parser("snapshot", help="write an on-disk snapshot")
    p.add_argument("name")

    sub.add_parser("health", help="server health")
    sub.add_parser("metrics", help="Prometheus metrics")

    return parser


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #


def _load_json_arg(raw: str | None, what: str) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: --{what} is not valid JSON: {exc}") from None


def _read_vector(args: argparse.Namespace) -> list[float]:
    if args.vector and args.vector_file:
        raise SystemExit("error: pass --vector or --vector-file, not both")
    if args.vector:
        try:
            return [float(x) for x in args.vector.replace(" ", "").split(",") if x]
        except ValueError as exc:
            raise SystemExit(f"error: --vector must be comma-separated floats: {exc}")
    if args.vector_file:
        text = sys.stdin.read() if args.vector_file == "-" else Path(
            args.vector_file
        ).read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, dict) and "vector" in data:
            data = data["vector"]
        return [float(x) for x in data]
    raise SystemExit("error: one of --vector or --vector-file is required")


def _read_items(path: str) -> list[dict[str, Any]]:
    """Accept a JSON array or JSONL; both are common for vector dumps."""
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise SystemExit("error: expected a JSON array of records")
        return data
    items = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"error: {path}:{lineno} is not valid JSON: {exc}")
    if not items:
        raise SystemExit(f"error: no records found in {path}")
    return items


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #


def _emit(value: Any, as_json: bool) -> None:
    if as_json or value is None:
        print(json.dumps(value, indent=2, default=str))
        return
    if isinstance(value, list) and value and isinstance(value[0], dict):
        _print_table(value)
    elif isinstance(value, dict):
        width = max((len(str(k)) for k in value), default=0)
        for key, val in value.items():
            if isinstance(val, (dict, list)):
                val = json.dumps(val, default=str)
            print(f"{str(key).ljust(width)}  {val}")
    else:
        print(value)


def _print_table(rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns and key != "metadata":
                columns.append(key)
    widths = {
        c: max(len(c), max(len(_cell(r.get(c))) for r in rows)) for c in columns
    }
    print("  ".join(c.ljust(widths[c]) for c in columns))
    print("  ".join("-" * widths[c] for c in columns))
    for row in rows:
        print("  ".join(_cell(row.get(c)).ljust(widths[c]) for c in columns))


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return str(value)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "serve":
        return _serve(args)

    client = PyVecClient(args.url)
    try:
        return _dispatch(client, args)
    except PyVecHTTPError as exc:
        if exc.status == 0:
            print(
                f"error: cannot reach a PyVec server at {args.url}\n"
                f"       ({exc.message})\n"
                f"       start one with: pyvec serve",
                file=sys.stderr,
            )
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    if args.data_dir:
        import os

        os.environ["PYVEC_DATA_DIR"] = str(args.data_dir)
    print(
        f"PyVec {__version__} serving on http://{args.host}:{args.port}  "
        f"(data: {args.data_dir or 'PYVEC_DATA_DIR or ./data'})"
    )
    uvicorn.run(
        "pyvec.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _dispatch(client: PyVecClient, args: argparse.Namespace) -> int:
    cmd = args.command
    as_json = args.json

    if cmd == "create":
        _emit(
            client.create_collection(
                args.name,
                dimension=args.dimension,
                metric=args.metric,
                index=args.index,
                index_params=_load_json_arg(args.index_params, "index-params"),
                text_field=args.text_field,
                capacity=args.capacity,
            ),
            as_json,
        )
    elif cmd == "ls":
        collections = client.list_collections()
        if not collections:
            print("no collections") if not as_json else _emit([], True)
        else:
            _emit(collections, as_json)
    elif cmd == "describe":
        _emit(client.describe(args.name), as_json)
    elif cmd == "rm":
        if not args.yes:
            answer = input(
                f"drop collection {args.name!r} and delete its files? [y/N] "
            )
            if answer.strip().lower() not in {"y", "yes"}:
                print("aborted")
                return 1
        client.drop_collection(args.name)
        print(f"dropped {args.name}")
    elif cmd == "insert":
        items = _read_items(args.file)
        result = client.insert(
            args.name, items, upsert=args.upsert, batch_size=args.batch_size
        )
        _emit(result, as_json)
    elif cmd == "get":
        _emit(client.get(args.name, args.id), as_json)
    elif cmd == "delete":
        client.delete(args.name, args.id)
        print(f"deleted {args.id}")
    elif cmd == "query":
        _emit(
            client.query(
                args.name,
                _read_vector(args),
                k=args.k,
                filter=_load_json_arg(args.filter, "filter"),
                params=_load_json_arg(args.params, "params"),
            ),
            as_json,
        )
    elif cmd == "search-text":
        _emit(
            client.query_text(
                args.name,
                args.text,
                k=args.k,
                filter=_load_json_arg(args.filter, "filter"),
            ),
            as_json,
        )
    elif cmd == "hybrid":
        _emit(
            client.hybrid(
                args.name,
                _read_vector(args),
                args.text,
                k=args.k,
                filter=_load_json_arg(args.filter, "filter"),
                params=_load_json_arg(args.params, "params"),
            ),
            as_json,
        )
    elif cmd == "optimize":
        _emit(client.optimize(args.name), as_json)
    elif cmd == "snapshot":
        _emit(client.snapshot(args.name), as_json)
    elif cmd == "health":
        _emit(client.health(), as_json)
    elif cmd == "metrics":
        print(client.metrics(), end="")
    else:  # pragma: no cover — argparse rejects unknown commands first
        raise SystemExit(f"unknown command {cmd!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
