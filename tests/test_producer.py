import time

import pytest

from fm_train.producer import BoundedProducer


def test_bounded_producer_rejects_invalid_worker_count():
    with pytest.raises(ValueError, match="workers"):
        BoundedProducer([1], lambda value: value, maxsize=1, workers=0)


def test_bounded_producer_parallel_workers_preserve_order():
    def slow_square(value: int) -> int:
        time.sleep((8 - value) * 0.005)
        return value * value

    started = time.perf_counter()
    result = list(BoundedProducer(range(8), slow_square, maxsize=4, workers=4))
    elapsed = time.perf_counter() - started

    assert result == [0, 1, 4, 9, 16, 25, 36, 49]
    assert elapsed < 0.16
