import torch


def check_ret(message, ret):
    if ret != 0:
        raise Exception("{} failed ret={}"
                        .format(message, ret))

import acl
ret = acl.init()
check_ret("acl.init", ret)

ret = acl.rt.set_device(0)
check_ret("acl.rt.set_device", ret)

import pickle
# with open('transpose.pkl', 'wb') as f:
#     a = torch.randn(1, 2, 3, 4)
#     pickle.dump(a, f)
with open('transpose.pkl', 'rb') as f:
    a = pickle.load(f)

def fn(x):
    y = torch.transpose(x, 1, 2).contiguous()
    return y

#opt_model = torch.compile(fn, backend='ascendgraph')
opt_model = torch.compile(fn, backend='inductor')

y = opt_model(a)
print(y)
print(y.shape)


ret = acl.rt.reset_device(0)
check_ret("acl.rt.reset_device", ret)
ret = acl.finalize()
check_ret("acl.finalize", ret)
print('Resources released successfully.')
