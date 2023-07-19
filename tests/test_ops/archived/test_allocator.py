import os
from multiprocessing import Process

def test_allocator(max_allocate, algorithm, log_mask):
    os.environ['DIPU_DEVICE_MEMCACHING_ALGORITHM'] = algorithm
    os.environ['DIPU_DEBUG_ALLOCATOR'] = str(log_mask)
    #os.environ['DIPU_MEM_CHECK'] = '1' # TODO(zhaoguochun): support memcheck
    import torch
    import torch_dipu
    for nbytes in range(0, max_allocate):
        if nbytes % 1024 == 0:
            print(f"allocate {nbytes} nbytes")
        raw = torch.empty(size = (nbytes,), dtype = torch.uint8)
        pin = raw.pin_memory()
        if nbytes > 0:
            assert pin.is_pinned() == True
            assert raw.is_pinned() == False

        assert raw.numel() == nbytes
        assert pin.numel() == nbytes

        index = 0
        torch.cuda.set_device(index)
        x1 = raw.to("dipu")
        assert x1.device.index == index
        assert x1.numel() == nbytes
    print(f"allocate [0, {max_allocate}) bytes use {algorithm} success")


if __name__=='__main__':
    max_allocate = 1 << 20
    p1 = Process(target = test_allocator, args = (max_allocate, "BF", 3),)
    p2 = Process(target = test_allocator, args = (max_allocate, "BS", 0),)
    p3 = Process(target = test_allocator, args = (max_allocate, "RAW", 0))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()

    assert p1.exitcode == 0
    assert p2.exitcode == 0
    assert p3.exitcode == 0