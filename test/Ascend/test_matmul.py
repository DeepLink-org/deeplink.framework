import torch
import torch._dynamo as dynamo
import logging

# dynamo.config.verbose = True
# dynamo.config.log_level  = logging.INFO
# dynamo.config.output_code  = True

#torch._logging.set_logs(dynamo=logging.DEBUG)


def check_ret(message, ret):
    if ret != 0:
        raise Exception("{} failed ret={}"
                        .format(message, ret))

import acl
ret = acl.init()
check_ret("acl.init", ret)

ret = acl.rt.set_device(0)
check_ret("acl.rt.set_device", ret)

aa = torch.rand(1, 2, 3)

import pickle
# with open('matmul1.pkl', 'wb') as f:
#     mat1 = torch.randn(1, 32, 32, 32)
#     pickle.dump(mat1, f)
# with open('matmul1.pkl', 'rb') as f:
#     mat1 = pickle.load(f)
with open('scores.pkl', 'rb') as f:
    mat1 = pickle.load(f)
    
# with open('matmul2.pkl', 'wb') as f:
#     mat2 = torch.randn(1, 32, 32, 128)
#     pickle.dump(mat2, f)
# with open('matmul2.pkl', 'rb') as f:
#     mat2 = pickle.load(f)
with open('values.pkl', 'rb') as f:
    mat2 = pickle.load(f)
    
with open('output.pkl', 'rb') as f:
    output = pickle.load(f)

output = output.to(torch.float)

print('mat1.shape:', mat1.size())
print('mat2.shape:', mat2.size())

def fn(x, y):
    z = torch.matmul(x, y)
    return z

opt_model1 = torch.compile(fn, backend='ascendgraph')
#opt_model1 = torch.compile(fn, backend='inductor')

a1 = opt_model1(mat1.to(torch.float), mat2.to(torch.float))
print(a1)
print(a1.sum())
print(a1.shape)

# a2 = opt_model2(mat1.to(torch.float), mat2.to(torch.float))
# print(a2.sum())
# print(a2.shape)

# print(output.sum())

# diff = output - a1
# abs_diff = torch.abs(diff)
# print("差异的绝对值:", abs_diff)


ret = acl.rt.reset_device(0)
check_ret("acl.rt.reset_device", ret)
ret = acl.finalize()
check_ret("acl.finalize", ret)
print('Resources released successfully.')
