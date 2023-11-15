import torch


from dicp.vendor.AscendGraph.ascend_op import *
from collections import namedtuple 
Node= namedtuple("variables", ['meta'])
def wrap_tensor(t:torch.Tensor):
    return Node(meta={'val':t}) 
def unwrap_tensor(d):
    return d.meta['val']

MM=MatMul()
BMM=BatchMatMul()

an = Node(meta={'val':torch.randn(5 ,4)}) 
bn = Node(meta={'val':torch.randn((7,1))}) 
cn = Node(meta={'val':torch.randn(2,5,3)}) 
dn = Node(meta={'val':torch.empty(2,5,5)}) 


def test_infer():
    a=an.meta['val']
    b=bn.meta['val']
    c=cn.meta['val']
    d=dn.meta['val']

    tn,t=[],[]
    t.append( torch.mm(a,torch.t(a)))
    t.append( torch.mm(b,torch.t(b)))
    t.append( torch.bmm(torch.transpose(c,1,2),d))
    t.append( torch.bmm(torch.transpose(d,1,2),torch.transpose(d,1,2)))

  

    tn.append( MM.infer_result( an,an,trans_b=True,change_input=True))
    tn.append( MM.infer_result( bn,bn,trans_b=True))
    tn.append( BMM.infer_result(cn,dn,adj_x1=True))
    tn.append( BMM.infer_result(dn,dn,adj_x1=True,adj_x2=True))

    
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

