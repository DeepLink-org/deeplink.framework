import torch
import torch._dynamo as dynamo
from typing import List

def bar(a):
    x = torch.abs(a)
    x = torch.rsqrt(x)
    return x

torch._dynamo.reset()
# after dicp python setup or pip install, topsgraph backends should be found
opt_bar = dynamo.optimize(backend='topsgraph', dynamic=False)(bar)

inp1 = torch.randn(3, 4)
print(opt_bar(inp1))
