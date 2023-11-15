import torch


from dicp.vendor.AscendGraph.ascend_op import *
from collections import namedtuple 
import torch.nn.functional as F
Node= namedtuple("variables", ['meta'])

w1n = Node(meta={'val':torch.ones((16 ,3,5,5),dtype=torch.int16)}) 
x1n = Node(meta={'val':torch.ones((1,3,28,28),dtype=torch.int16)}) 
w2n = Node(meta={'val':torch.ones((8,4,3,3),dtype=torch.int64)}) 
x2n = Node(meta={'val':torch.ones((1,4,5,5),dtype=torch.int64)}) 


CONV=Conv2D()


def wrap_tensor(t:torch.Tensor):
    return Node(meta={'val':t}) 
def unwrap_tensor(d):
    return d.meta['val']


def test_infer_broadcastTo(w1n,x1n,w2n,x2n):
    w1=w1n.meta['val']
    x1=x1n.meta['val']
    w2=w2n.meta['val']
    x2=x2n.meta['val']

    tn,t=[],[]

    t.append(F.conv2d(x1,w1))
    t.append(F.conv2d(x1,w1,stride=2,padding=2))
    t.append(F.conv2d(x2,w2))
    t.append(F.conv2d(x2,w2,stride=(2,1),padding=(2,4)))
    
    # print(x1.shape," ",w1.shape)
    tn.append( CONV.infer_result(x1n,w1n))
    tn.append( CONV.infer_result(x1n,w1n,stride=2,padding=2))
    tn.append( CONV.infer_result(x2n,w2n))
    tn.append( CONV.infer_result(x2n,w2n,stride=(2,1),padding=(2,4)))
    


    for cnt,t in enumerate(zip(tn,t)):
        i,j=t
        # print(i.shape)
        print(cnt," ",j.shape,j.dtype," *** ",torch.randn(i.shape).shape,i.dtype)
        assert list(i.shape) == list(j.shape)," shape mismatch!"
        assert i.dtype == j.dtype," dtype mismatch!"

    # tn+= BROADCASTTO.infer_result(dn,[3,3,3])
    # t+= torch.broadcast_to(d,[3,3,3])


test_infer_broadcastTo(w1n,x1n,w2n,x2n)




    


# y = infer_model(an, bn, cn)
# y2 = fn(a, b, c)
# print('*** y shape: ',y.shape)
# print(y2,' shape: ',y2.shape)

