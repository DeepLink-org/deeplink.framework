import torch
import torch_dipu
ins = []
pin_ins = []
for i in range(100):
    x = torch.randn(512).to(torch.device('cuda:0'))
    y = torch.randn(512).pin_memory()
    ins.append(x)
    pin_ins.append(y)
    allocated = torch.cuda.memory_allocated(x.device)
    allocated_default = torch.cuda.memory_allocated()
    pin_allocated = torch.cuda.memory_allocated(y.device)
    reserved = torch.cuda.memory_reserved(x.device)
    assert allocated == len(ins) * 512 * 4
    assert allocated == allocated_default
    assert pin_allocated == len(pin_ins) * 512 * 4
    assert reserved >= allocated
    print(f"allocated:{allocated} , reserved:{reserved}")
    del x, y

del ins
del pin_ins
allocated_default = torch.cuda.memory_allocated()
assert allocated_default == 0
assert torch.cuda.memory_reserved() > 0

torch.cuda.empty_cache()
assert torch.cuda.memory_reserved() == 0
assert torch.cuda.memory_allocated() == 0