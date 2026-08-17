"""A reader-writer lock.

ARCHITECTURE.md §"Concurrency model": single writer, many readers per
collection. Rolled by hand on top of ``threading.Condition`` rather than taking
the ``readerwriterlock`` dependency — it is 40 lines and the semantics matter
enough to be explicit about them.

The policy is **writer-preferring**: a waiting writer blocks new readers from
acquiring. Reader-preferring locks starve writers under sustained read load,
which for a search engine (reads vastly outnumber writes) means inserts could
hang indefinitely.

Reentrancy is not supported. Do not acquire the read lock while holding it.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

__all__ = ["RWLock"]


class RWLock:
    """Writer-preferring reader-writer lock."""

    __slots__ = ("_cond", "_readers", "_writer", "_waiting_writers")

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    # -- raw acquire/release ------------------------------------------------ #

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer or self._waiting_writers > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
            finally:
                self._waiting_writers -= 1
            self._writer = True

    def release_write(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()

    # -- context managers -------------------------------------------------- #

    @contextmanager
    def read(self) -> Iterator[None]:
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write(self) -> Iterator[None]:
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()

    # -- introspection, for tests ------------------------------------------ #

    @property
    def readers(self) -> int:
        return self._readers

    @property
    def write_locked(self) -> bool:
        return self._writer
