import torch


from dicp.vendor.AscendGraph.ascend_op import *
from collections import namedtuple 
Node= namedtuple("variables", ['meta'])
def wrap_tensor(t:torch.Tensor):
    return Node(meta={'val':t}) 
def unwrap_tensor(d):
    return d.meta['val']

SORT=Sort()
TOPK=TopK()

an = Node(meta={'val':torch.randn(5,1,2,3,4)}) 
bn = Node(meta={'val':torch.randn(2,5,3,4,6)}) 
cn = Node(meta={'val':torch.randn((1,1,1,10,12))}) 
dn = Node(meta={'val':torch.randn(2,5,5,7,8)}) 


def test_infer():
    a=an.meta['val']
    b=bn.meta['val']
    c=cn.meta['val']
    d=dn.meta['val']


    tn,t=[],[]
    t+= list(torch.sort(a))
    t+= list(torch.sort(b))
    t+= list(torch.topk(c,0,-2))
    t+= list(torch.topk(d,7))

  

    tn+= list(SORT.infer_result(an))
    tn+= list(SORT.infer_result(bn))
    tn+= list(TOPK.infer_result(cn,0,dim=-2))
    tn+= list(TOPK.infer_result(dn,7))

    

    
    for cnt,t in enumerate(zip(tn,t)):
        i,j=t
        print(cnt," ",j.shape,j.dtype," *** ",torch.randn(i.shape).shape,i.dtype)
        assert list(i.shape) == list(j.shape)," shape mismatch!"
        assert i.dtype == j.dtype," dtype mismatch!"

    tn+=  list(TOPK.infer_result(cn,11,dim=-2)) # torch will raise error"selected index k out of range"

test_infer()


# 

    


# y = infer_model(an, bn, cn)
# y2 = fn(a, b, c)
# print('*** y shape: ',y.shape)
# print(y2,' shape: ',y2.shape)

