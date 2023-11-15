import torch


from dicp.vendor.AscendGraph.ascend_op import *
from collections import namedtuple 
Node= namedtuple("variables", ['meta'])

an = Node(meta={'val':torch.randn(2, 3)}) 
bn = Node(meta={'val':torch.randn(2, 3)}) 
cn = Node(meta={'val':torch.randn(3, 2)}) 
a=an.meta['val']
b=bn.meta['val']
c=cn.meta['val']

ADD=Add()
SUB=Sub()
MM=MatMul()

RAND_T=torch.rand



def wrap_tensor(t:torch.Tensor):
    return Node(meta={'val':t}) 
def unwrap_tensor(d):
    return d.meta['val']


def infer_model(a,b,c):
    
    
    t1= wrap_tensor(RAND_T(ADD.infer_result(a,b).shape))
    t2= wrap_tensor(RAND_T(ADD.infer_result(a,t1).shape))


    return unwrap_tensor(t2)


def fn(a, b, c):

    t1 = torch.add(a, b)
    t2 = torch.add(a, t1, alpha=1.51)

    return t2
    t3 = torch.add(t2, 8.9)
    t4 = torch.add(t3, 9.9, alpha=1.61)
    x = torch.sub(t4, b)
    y = torch.mm(x, c)
    return torch.sub(y, 2.12)
    

# opt_model = torch.compile(fn, backend='ascendgraph')
#opt_model = torch.compile(fn, backend='inductor')

y = infer_model(an, bn, cn)
y2 = fn(a, b, c)
print('*** y shape: ',y.shape)
print(y2,' shape: ',y2.shape)
assert y.dtype == y2.dtype,"dtype mismatch"
# assert torch.allclose(y.to('cpu'), y2)