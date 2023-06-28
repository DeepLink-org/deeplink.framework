import torch
import acl


def check_ret(message, ret):
    if ret != 0:
        raise Exception("{} failed ret={}"
                        .format(message, ret))

ret = acl.init()
check_ret("acl.init", ret)

ret = acl.rt.set_device(0)
check_ret("acl.rt.set_device", ret)

    
mat1 = torch.randn(1, 32, 32, 32)
mat2 = torch.randn(1, 32, 32, 128)

print('mat1.shape:', mat1.size())
print('mat2.shape:', mat2.size())

def fn(x, y):
    z = torch.matmul(x, y)
    return z

opt_model1 = torch.compile(fn, backend='ascendgraph')
#opt_model1 = torch.compile(fn, backend='inductor')

a1 = opt_model1(mat1.to(torch.float), mat2.to(torch.float))


ret = acl.rt.reset_device(0)
check_ret("acl.rt.reset_device", ret)
ret = acl.finalize()
check_ret("acl.finalize", ret)
print('Resources released successfully.')
