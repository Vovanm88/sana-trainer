from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")
R = TypeVar("R")


class BoundedProducer(Iterator[R]):
    _END = object()

    def __init__(self, source: Iterable[T], function: Callable[[T], R], maxsize: int):
        self.queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.source = source
        self.function = function
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            for item in self.source:
                self.queue.put((True, self.function(item)))
        except BaseException as error:
            self.queue.put((False, error))
        finally:
            self.queue.put((True, self._END))

    def __iter__(self) -> "BoundedProducer[R]":
        return self

    def __next__(self) -> R:
        success, value = self.queue.get()
        if not success:
            raise value
        if value is self._END:
            raise StopIteration
        return value
