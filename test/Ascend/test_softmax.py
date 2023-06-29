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

softmax = torch.randn(2, 3)

def fn(x):
    y = torch.nn.functional.softmax(x, -1)
    return y

opt_model = torch.compile(fn, backend='ascendgraph')
#opt_model = torch.compile(fn, backend='inductor')

y = opt_model(softmax)
print(y)
print(y.shape)


ret = acl.rt.reset_device(0)
check_ret("acl.rt.reset_device", ret)
ret = acl.finalize()
check_ret("acl.finalize", ret)
print('Resources released successfully.')
