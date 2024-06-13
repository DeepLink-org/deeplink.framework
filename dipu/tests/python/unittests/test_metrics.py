# Copyright (c) 2023, DeepLink.
import random
import torch
import torch_dipu
from collections.abc import Iterable
from torch_dipu.testing._internal.common_utils import TestCase, run_tests
from typing import Protocol


class MetricsGroup(Protocol):
    name: str
    type: str
    help: str
    values: list[
        #          labels,       value |        histogram ( thresholds,   buckets, sum)
        (list[(str, str)], int | float | tuple[list[int] | list[float], list[int], int])
    ]


def lookup(groups: Iterable[MetricsGroup], name: str, labels: list[(str, str)]):
    item = next(x for x in groups if x.name == name)
    keys = set(labels)
    value = next(v for k, v in item.values if keys.issubset(k))
    return value


def allocate_tensor(count: int) -> int:
    total = 0
    for _ in range(0, count):
        nbytes = random.randrange(0, 100000)
        total += nbytes
        tensor = torch.empty(size=(nbytes,), dtype=torch.uint8, device="dipu")
    return total


class TestMetrics(TestCase):
    def test_allocator_metrics(self):
        name = "allocator_size"
        labels = [("type", "caching"), ("device", "0"), ("method", "allocate")]

        _, last_bucket, last_size = lookup(torch_dipu._C.metrics(), name, labels)

        expected_count = 100
        expected_total = allocate_tensor(expected_count)

        _, next_bucket, next_size = lookup(torch_dipu._C.metrics(), name, labels)
        count = sum(next_bucket) - sum(last_bucket)
        total = next_size - last_size

        self.assertEqual(expected_count, count, msg=f"{expected_count} == {count}")
        self.assertLessEqual(expected_total, total, msg=f"{expected_total} <= {total}")


if __name__ == "__main__":
    run_tests()
