import os
from multiprocessing import Process


def test_mem_stats(algorithm, log_mask):
    os.environ["DIPU_DEVICE_MEMCACHING_ALGORITHM"] = algorithm
    os.environ["DIPU_DEBUG_ALLOCATOR"] = str(log_mask)
    print("allocator algorithm:", algorithm)
    import torch
    import torch_dipu
    import random

    ins = []
    pin_ins = []
    real_allocated = 0
    for i in range(100):
        numel = random.randint(0, 1 << 20)
        x = torch.randn(numel).to(torch.device("cuda:0"))
        y = torch.randn(numel).pin_memory()
        ins.append(x)
        pin_ins.append(y)
        allocated = torch.cuda.memory_allocated(x.device)
        allocated_default = torch.cuda.memory_allocated()
        allocated_by_index = torch.cuda.memory_allocated(0)
        pin_allocated = torch.cuda.memory_allocated(y.device)
        reserved = torch.cuda.memory_reserved(x.device)
        real_allocated += ((numel * 4 - 1) | 511) + 1
        print(
            f"numel:{numel}, allocated:{allocated}, reserved:{reserved}, real_allocated:{real_allocated}"
        )
        assert allocated == real_allocated
        assert allocated == allocated_default
        assert allocated == allocated_by_index
        assert pin_allocated == real_allocated
        assert reserved >= allocated
        del x, y

    real_max_allocate = real_allocated

    for i in range(len(ins)):
        numel = ins[0].numel()
        real_allocated -= ((numel * 4 - 1) | 511) + 1
        ins.pop(0)
        assert torch.cuda.memory_allocated() == real_allocated

    del pin_ins

    allocated_default = torch.cuda.memory_allocated()
    assert allocated_default == 0
    if algorithm == "RAW":
        assert torch.cuda.memory_reserved() == 0
    else:
        assert torch.cuda.memory_reserved() > 0

    torch.cuda.empty_cache()

    assert torch.cuda.memory_reserved() == 0
    assert torch.cuda.memory_allocated() == 0
    assert torch.cuda.max_memory_allocated() == real_max_allocate
    assert torch.cuda.max_memory_reserved() > 0


if __name__ == "__main__":
    p1 = Process(
        target=test_mem_stats,
        args=("BF", 0),
    )
    p1.start()
    p1.join()

    p2 = Process(
        target=test_mem_stats,
        args=("BS", 0),
    )
    p2.start()
    p2.join()

    p3 = Process(target=test_mem_stats, args=("RAW", 0))
    p3.start()
    p3.join()

    assert p1.exitcode == 0
    assert p2.exitcode == 0
    assert p3.exitcode == 0
