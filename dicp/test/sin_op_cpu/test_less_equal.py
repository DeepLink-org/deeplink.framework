import torch


from dicp.vendor.AscendGraph.ascend_op import *
from collections import namedtuple 
Node= namedtuple("variables", ['meta'])
def wrap_tensor(t:torch.Tensor):
    return Node(meta={'val':t}) 
def unwrap_tensor(d):
    return d.meta['val']


EQUAL=Equal()
LESS=Less()
LESSEQUAL=LessEqual()

an = Node(meta={'val':torch.randn(2,5,3,4,6)}) 
bn = Node(meta={'val':torch.randn(2,5,3,4,6)}) 


def test_infer():
    a=an.meta['val']
    b=bn.meta['val']


    tn,t=[],[]

    t.append( torch.eq(a,b))
    t.append( torch.less(a,b))
    t.append( torch.le(a,b))

  
    tn.append( EQUAL.infer_result(an,bn))
    tn.append( LESS.infer_result(an,bn))
    tn.append( LESSEQUAL.infer_result(an,bn))

    

    
    for cnt,t in enumerate(zip(tn,t)):
        i,j=t
        # print(i," ",j)
        print(cnt," ",j.shape,j.dtype," *** ",torch.randn(i.shape).shape,i.dtype)
        assert list(i.shape) == list(j.shape)," shape mismatch!"
        assert i.dtype == j.dtype," dtype mismatch!"


test_infer()


# 

    


# y = infer_model(an, bn, cn)
# y2 = fn(a, b, c)
# print('*** y shape: ',y.shape)
# print(y2,' shape: ',y2.shape)

