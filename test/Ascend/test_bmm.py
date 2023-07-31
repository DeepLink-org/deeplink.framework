import torch
import acl

def check_ret(message, ret):
    if ret != 0:
        raise Exception("{} failed ret={}"
                        .format(message, ret))

# ret = acl.init()
# check_ret("acl.init", ret)

# ret = acl.rt.set_device(0)
# check_ret("acl.rt.set_device", ret)
    
input = torch.randn(3, 3, 4)
mat2 = torch.randn(3, 4, 5)

def fn(input, mat2):
    y = torch.bmm(input, mat2)
    return y

opt_model = torch.compile(fn, backend='ascendgraph')
#opt_model = torch.compile(fn, backend='inductor')

y = opt_model(input, mat2)
print(y)
print(y.shape)


# ret = acl.rt.reset_device(0)
# check_ret("acl.rt.reset_device", ret)
# ret = acl.finalize()
# check_ret("acl.finalize", ret)
print('Resources released successfully.')
