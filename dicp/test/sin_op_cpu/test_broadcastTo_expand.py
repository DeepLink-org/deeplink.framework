import torch


from dicp.vendor.AscendGraph.ascend_op import *
from collections import namedtuple 
Node= namedtuple("variables", ['meta'])

an = Node(meta={'val':torch.randn(3 ,1,1)}) 
bn = Node(meta={'val':torch.randn(5,7, 3)}) 
cn = Node(meta={'val':torch.randn((3,))}) 
dn = Node(meta={'val':torch.empty((0,))}) 


ADD=Add()
SUB=Sub()
MM=MatMul()

RAND_T=torch.rand
BROADCASTTO=BroadcastTo()
EXPAND=Expand()

def wrap_tensor(t:torch.Tensor):
    return Node(meta={'val':t}) 
def unwrap_tensor(d):
    return d.meta['val']


def test_infer_broadcastTo(an,bn,cn,dn):
    a=an.meta['val']
    b=bn.meta['val']
    c=cn.meta['val']
    d=dn.meta['val']

    tn,t=[],[]

    t.append(torch.broadcast_to(a,(5,3,4,1)))
    t.append(a.expand([5,-1,4,1]))
    t.append(torch.broadcast_to(b,[5,7,3]))
    t.append(a.expand([3,3,3]))
    

    tn.append( BROADCASTTO.infer_result(an,[5,3,4,1]))
    tn.append( EXPAND.infer_result(an,[5,-1,4,1]))
    tn.append( BROADCASTTO.infer_result(bn,[5,7,3]))
    tn.append( EXPAND.infer_result(cn,[3,3,3]))
    


    for cnt,t in enumerate(zip(tn,t)):
        i,j=t
        print(cnt," ",j.shape,j.dtype," *** ",torch.randn(i.shape).shape,i.dtype)
        assert list(i.shape) == list(j.shape)," shape mismatch!"
        assert i.dtype == j.dtype," dtype mismatch!"

    # tn+= BROADCASTTO.infer_result(dn,[3,3,3])
    # t+= torch.broadcast_to(d,[3,3,3])


test_infer_broadcastTo(an,bn,cn,dn)




    


# y = infer_model(an, bn, cn)
# y2 = fn(a, b, c)
# print('*** y shape: ',y.shape)
# print(y2,' shape: ',y2.shape)

