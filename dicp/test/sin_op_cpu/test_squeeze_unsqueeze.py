import torch


from dicp.vendor.AscendGraph.ascend_op import *
from collections import namedtuple 
Node= namedtuple("variables", ['meta'])
def wrap_tensor(t:torch.Tensor):
    return Node(meta={'val':t}) 
def unwrap_tensor(d):
    return d.meta['val']

SQUEEZE=Squeeze()
UNSQUEEZE=Unsqueeze()

 
an = Node(meta={'val':torch.randn((1,1,1,1,12))}) 
bn = Node(meta={'val':torch.randn(1,5,1,7,1)})
cn = Node(meta={'val':torch.randn(5,1,2,3,4)}) 
dn = Node(meta={'val':torch.randn(2,5,3,4,6)}) 


def test_infer():
    a=an.meta['val']
    b=bn.meta['val']
    c=cn.meta['val']
    d=dn.meta['val']


    tn,t=[],[]
    t.append( torch.squeeze  (a,[0,-3]))
    t.append( torch.squeeze  (b,2))
    t.append( torch.unsqueeze(c,0))
    t.append( torch.unsqueeze(d,-1))

  

    tn.append( SQUEEZE.infer_result  (an,[0,-3]))
    tn.append( SQUEEZE.infer_result  (bn,2))
    tn.append( UNSQUEEZE.infer_result(cn,0))
    tn.append( UNSQUEEZE.infer_result(dn,-1))

    

    
    for cnt,t in enumerate(zip(tn,t)):
        i,j=t

        print(cnt," ",j.shape,j.dtype," *** ",torch.randn(i.shape).shape,i.dtype)
        assert list(i.shape) == list(j.shape)," shape mismatch!"
        assert i.dtype == j.dtype," dtype mismatch!"

test_infer()


# 

    


# y = infer_model(an, bn, cn)
# y2 = fn(a, b, c)
# print('*** y shape: ',y.shape)
# print(y2,' shape: ',y2.shape)

