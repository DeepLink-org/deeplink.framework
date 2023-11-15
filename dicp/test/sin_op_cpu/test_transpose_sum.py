import torch


from dicp.vendor.AscendGraph.ascend_op import *
from collections import namedtuple 
Node= namedtuple("variables", ['meta'])
def wrap_tensor(t:torch.Tensor):
    return Node(meta={'val':t}) 
def unwrap_tensor(d):
    return d.meta['val']

TRANSPOSE=Transpose()
REDUCESUM=ReduceSum()

an = Node(meta={'val':torch.randn(5,1,2,3,4)}) 
bn = Node(meta={'val':torch.randn(2,5,3,4,6)}) 
cn = Node(meta={'val':torch.randn((1,1,1,1,12))}) 
dn = Node(meta={'val':torch.randn(2,5,5,7,8)}) 
en = Node(meta={'val':torch.randn(2,5,5,7,8)}) 

def test_infer():
    a=an.meta['val']
    b=bn.meta['val']
    c=cn.meta['val']
    d=dn.meta['val']
    e=en.meta['val']

    tn,t=[],[]
    t.append( torch.permute(a,[3,4,2,0,1]))
    t.append( torch.permute(b,[0,-1,3,2,1]))
    t.append( torch.sum(c))
    t.append( torch.sum(d,dim=[0,2,-1]))
    t.append( torch.sum(e,dim=[0,2,4],keepdim=True))
  

    tn.append( TRANSPOSE.infer_result(an,[3,4,2,0,1]))
    tn.append( TRANSPOSE.infer_result(bn,[0,-1,3,2,1]))
    tn.append( REDUCESUM.infer_result(cn))
    tn.append( REDUCESUM.infer_result(dn,axes=[0,2,-1]))
    tn.append( REDUCESUM.infer_result(en,axes=[0,2,4],keep_dims=True))
    

    
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

