import torch


from dicp.vendor.AscendGraph.ascend_op import *
from collections import namedtuple 
Node= namedtuple("variables", ['meta'])
def wrap_tensor(t:torch.Tensor):
    return Node(meta={'val':t}) 
def unwrap_tensor(d):
    return d.meta['val']

REDUCEVARMEAN=ReduceStdV2Update()
REDUCEMAX=ReduceMax()
REDUCEMEAN=ReduceMean()

an = Node(meta={'val':torch.randn(5,1,2,3,4)}) 
bn = Node(meta={'val':torch.randn(2,5,3,4,6)}) 
cn = Node(meta={'val':torch.randn((1,1,1,1,12))})
dn = Node(meta={'val':torch.randn(2,5,5,7,8)})  


def test_infer():
    a=an.meta['val']
    b=bn.meta['val']
    c=cn.meta['val']
    d=dn.meta['val']


    tn,t=[],[]
    t+= list(torch.var_mean(a,dim=-3))
    t+= list(torch.var_mean(b,dim=[2,3],keepdim=True))
    t.append( torch.amax(c))
    t.append( torch.amax(d,dim=[0,2,-1]))
    t.append( torch.amax(d,dim=[0,2,4],keepdim=True))
    t.append( torch.mean(c))
    t.append( torch.mean(d,dim=[0,2,-1]))
    t.append( torch.mean(d,dim=[0,2,4],keepdim=True))

  

    tn+= list(REDUCEVARMEAN.infer_result(an,dim=-3))
    tn+= list(REDUCEVARMEAN.infer_result(bn,dim=[2,3],keepdim=True))
    tn.append( REDUCEMAX.infer_result(cn))
    tn.append( REDUCEMAX.infer_result(dn,dim=[0,2,-1]))
    tn.append( REDUCEMAX.infer_result(dn,dim=[0,2,4],keepdim=True))
    tn.append( REDUCEMEAN.infer_result(cn))
    tn.append( REDUCEMEAN.infer_result(dn,dim=[0,2,-1]))
    tn.append( REDUCEMEAN.infer_result(dn,dim=[0,2,4],keepdim=True))

    

    
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

