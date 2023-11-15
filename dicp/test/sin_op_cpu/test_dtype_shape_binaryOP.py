import torch


from dicp.vendor.AscendGraph.ascend_op import *
from collections import namedtuple 
Node= namedtuple("variables", ['meta'])
def wrap_tensor(t:torch.Tensor):
    return Node(meta={'val':t}) 
def unwrap_tensor(d):
    return d.meta['val']

ADD=Add()
SUB=Sub()
DIV=Div()
MUL=Mul()
SELECT=Select()
POW=Pow()
EQ=Equal()
LE=LessEqual()
LT=Less()
GE=GreaterEqual()
GT=Greater()


an = Node(meta={'val':torch.ones((5,1,4,1),dtype=torch.float32)}) 
bn = Node(meta={'val':torch.ones((  3,1,1),dtype=torch.int32)}) 
cn = Node(meta={'val':torch.ones((1,1),dtype=torch.complex64)}) 
dn = Node(meta={'val':torch.ones((1,1,7),dtype=torch.bool)}) 
en = Node(meta={'val':torch.ones((1,7,1),dtype=torch.bool)}) 

def test_infer():
    a=an.meta['val']
    b=bn.meta['val']
    c=cn.meta['val']
    d=dn.meta['val']
    e=en.meta['val']

    tn,t=[],[]
    
    t.append( torch.add(a,b))
    t.append( torch.sub(c,b))
    t.append( torch.div(a,b))
    t.append( torch.mul(c,b))
    t.append( torch.where(d,a,b))
    t.append( torch.pow(b,c))
    t.append( torch.eq(a,b))
    t.append( torch.le(b,a))
    t.append( torch.lt(a,b))
    t.append( torch.ge(b,a))
    t.append( torch.gt(a,b))
    
  

    tn.append( ADD.infer_result(an,bn))
    tn.append( SUB.infer_result(cn,bn))
    tn.append( DIV.infer_result(an,bn))
    tn.append( MUL.infer_result(cn,bn))
    tn.append( SELECT.infer_result(dn,an,bn))
    tn.append( POW.infer_result(bn,cn))
    tn.append( EQ.infer_result(an,bn))
    tn.append( LE.infer_result(bn,an))
    tn.append( LT.infer_result(an,bn))
    tn.append( GE.infer_result(bn,an))
    tn.append( GT.infer_result(an,bn))
    
    

    
    for cnt,tup in enumerate(zip(tn,t)):
        i,j=tup
        print(cnt," ",j.shape,j.dtype," *** ",torch.randn(i.shape).shape,i.dtype)
        assert list(i.shape) == list(j.shape)," shape mismatch!"
        assert i.dtype == j.dtype," dtype mismatch!"

    
    # print(torch.gt(a,e).shape)
    # t.append(torch.gt(a,e))
    # tn.append( GT.infer_result(an,en)) 

test_infer()


# 

    


# y = infer_model(an, bn, cn)
# y2 = fn(a, b, c)
# print('*** y shape: ',y.shape)
# print(y2,' shape: ',y2.shape)

