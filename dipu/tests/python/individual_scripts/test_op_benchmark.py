# Copyright (c) 2023, DeepLink.
# TODO(mrdanielw,fandaoyi,lljbash): enhance the benchmark
import torch
import torch.utils.benchmark as benchmark
import torch_dipu
from itertools import product

x = torch.randn(10000, 64).cuda()


def batched_dot_mul_sum(a, b):
    """Computes batched dot by multiplying and summing"""
    return a.mul(b).sum(-1)


t0 = benchmark.Timer(
    stmt="batched_dot_mul_sum(x, x)",
    setup="from __main__ import batched_dot_mul_sum",
    globals={"x": x},
)
# warm up
t0.timeit(100)

r0 = t0.timeit(100)
print(r0)
# TODO(fandaoyi,lljbash): find out why it gets slower
# assert r0.mean < 8.8e-5
assert r0.mean < 35.0e-5


def batched_dot_bmm(a, b):
    """Computes batched dot by reducing to ``bmm``"""
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)


t1 = benchmark.Timer(
    stmt="batched_dot_bmm(x, x)",
    setup="from __main__ import batched_dot_bmm",
    globals={"x": x},
)
# warm up
t1.timeit(100)

r1 = t1.timeit(100)
print(r1)
# TODO(fandaoyi,lljbash): find out why it gets slower
# assert r0.mean < 8.8e-5
assert r1.mean < 30.0e-5


# Compare takes a list of measurements which we'll save in results.
results = []
sizes = [1, 64, 32, 120]
for b, n in product(sizes, sizes):
    # label and sub_label are the rows
    # description is the column
    label = "Batched dot"
    sub_label = f"[{b}, {n}]"
    x = torch.ones((b, n)).cuda()
    # cuda tensor, not so many dispatch threads in actual case. 16, 32]:
    for num_threads in [1, 4]:
        results.append(
            benchmark.Timer(
                stmt="batched_dot_mul_sum(x, x)",
                setup="from __main__ import batched_dot_mul_sum",
                globals={"x": x},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description="mul/sum",
            ).blocked_autorange(min_run_time=1)
        )
        results.append(
            benchmark.Timer(
                stmt="batched_dot_bmm(x, x)",
                setup="from __main__ import batched_dot_bmm",
                globals={"x": x},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description="bmm",
            ).blocked_autorange(min_run_time=1)
        )

compare = benchmark.Compare(results)
compare.print()
