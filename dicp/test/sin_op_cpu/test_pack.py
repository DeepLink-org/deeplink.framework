import torch


from dicp.vendor.AscendGraph.ascend_op import *
from collections import namedtuple 
Node= namedtuple("variables", ['meta'])
def wrap_tensor(t:torch.Tensor):
    return Node(meta={'val':t}) 
def unwrap_tensor(d):
    return d.meta['val']


PACK=Pack()
T1 = torch.tensor([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
T2 = torch.tensor([[10, 20, 30],
                 [40, 50, 60],
                 [70, 80, 90]])


an = Node(meta={'val':(T1,T2)}) 




def test_infer():
    a=an.meta['val']

    tn,t=[],[]

    t.append( torch.stack(a,dim=0))
    t.append( torch.stack(a,dim=1))
    t.append( torch.stack(a,dim=-1))

  
    tn.append( PACK.infer_result(an,dim=0))
    tn.append( PACK.infer_result(an,dim=1))
    tn.append( PACK.infer_result(an,dim=-1))


    
    for cnt,tup in enumerate(zip(tn,t)):
        i,j=tup
        # print(i," ",j)
        print(cnt," ",j.shape,j.dtype," *** ",torch.randn(i.shape).shape,i.dtype)
        assert list(i.shape) == list(j.shape)," shape mismatch!"
        assert i.dtype == j.dtype," dtype mismatch!"

    # tn.append( PACK.infer_result(an,dim=3))
    # t.append( torch.stack(a,dim=3))
    


test_infer()