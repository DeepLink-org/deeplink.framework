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


class TestMetrics(TestCase):
    def test_allocator_metrics(self):

        total = 0
        count = 100
        for i in range(0, count):
            nbytes = random.randrange(0, 100000)
            total += nbytes
            tensor = torch.empty(size=(nbytes,), dtype=torch.uint8, device="dipu")
        value = torch_dipu._C.metrics()
        labels = [("type", "caching"), ("device", "0"), ("method", "allocate")]
        output = lookup(value, "allocator_size", labels)

        self.assertEqual(count, sum(output[1]))
        self.assertLessEqual(total, output[2])


if __name__ == "__main__":
    run_tests()
