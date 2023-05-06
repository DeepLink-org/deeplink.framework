import torch
import torch._dynamo as dynamo

# decomp table will be added into config file later
from torch._inductor.decomposition import decompositions
aten = torch.ops.aten
del decompositions[aten._native_batch_norm_legit_functional.default]
del decompositions[aten.native_batch_norm_backward.default]
del decompositions[aten.convolution_backward.default]

def bar(a):
    x = torch.abs(a)
    x = torch.rsqrt(x)
    return x

torch._dynamo.reset()
# after dicp python setup or pip install, topsgraph backends should be found
opt_bar = dynamo.optimize(backend='topsgraph', dynamic=False)(bar)

inp1 = torch.randn(3, 4)
print(opt_bar(inp1))
