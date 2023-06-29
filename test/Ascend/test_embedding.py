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

embedding = torch.nn.Embedding(10, 3)

def fn(input):
    y = embedding(input)
    return y

opt_model = torch.compile(fn, backend='ascendgraph')
#opt_model = torch.compile(fn, backend='inductor')

torch.manual_seed(1)

input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
print(input)
y = opt_model(input)
print(y)
print(y.shape)


ret = acl.rt.reset_device(0)
check_ret("acl.rt.reset_device", ret)
ret = acl.finalize()
check_ret("acl.finalize", ret)
print('Resources released successfully.')
