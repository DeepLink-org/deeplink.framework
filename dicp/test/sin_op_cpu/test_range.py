import torch


from dicp.vendor.AscendGraph.ascend_op import *


ADD=Add()
SUB=Sub()
MM=MatMul()
RAND_T=torch.rand
RANGE=Range()



def test_infer():
    

    tn,t=[],[]

    t.append( torch.arange(3,18,3))
    t.append( torch.arange(3,1,-0.5))
    t.append( torch.arange(3,1,-0.6))
    t.append( torch.arange(start=0,end=5))


    tn.append( RANGE.infer_result(3,18,3))
    tn.append( RANGE.infer_result(3,1,-0.5))
    tn.append( RANGE.infer_result(3,1,-0.6))
    tn.append( RANGE.infer_result(start=5))

    
    for cnt,t in enumerate(zip(tn,t)):
        i,j=t
        print(cnt," ",j.shape,j.dtype," *** ",torch.randn(i.shape).shape,i.dtype)
        assert list(i.shape) == list(j.shape)," shape mismatch!"
        assert i.dtype == j.dtype," dtype mismatch!"

test_infer()




    


# y = infer_model(an, bn, cn)
# y2 = fn(a, b, c)
# print('*** y shape: ',y.shape)
# print(y2,' shape: ',y2.shape)

