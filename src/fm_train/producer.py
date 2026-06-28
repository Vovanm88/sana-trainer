from __future__ import annotations

import queue
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")
R = TypeVar("R")


class BoundedProducer(Iterator[R]):
    _END = object()

    def __init__(
        self,
        source: Iterable[T],
        function: Callable[[T], R],
        maxsize: int,
        workers: int = 1,
    ):
        if workers < 1:
            raise ValueError("BoundedProducer workers must be positive")
        self.queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.source = source
        self.function = function
        self.workers = workers
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            if self.workers == 1:
                for item in self.source:
                    self.queue.put((True, self.function(item)))
            else:
                self._run_parallel()
        except BaseException as error:
            self.queue.put((False, error))
        finally:
            self.queue.put((True, self._END))

    def _run_parallel(self) -> None:
        source = iter(self.source)
        in_flight: dict[Future[R], int] = {}
        completed: dict[int, R] = {}
        submitted = 0
        next_output = 0
        max_in_flight = max(self.workers, self.queue.maxsize)
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            exhausted = False
            while in_flight or not exhausted:
                while not exhausted and len(in_flight) < max_in_flight:
                    try:
                        item = next(source)
                    except StopIteration:
                        exhausted = True
                    else:
                        future = executor.submit(self.function, item)
                        in_flight[future] = submitted
                        submitted += 1
                if not in_flight:
                    continue
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    completed[in_flight.pop(future)] = future.result()
                while next_output in completed:
                    self.queue.put((True, completed.pop(next_output)))
                    next_output += 1

    def __iter__(self) -> "BoundedProducer[R]":
        return self

    def __next__(self) -> R:
        success, value = self.queue.get()
        if not success:
            raise value
        if value is self._END:
            raise StopIteration
        return value
