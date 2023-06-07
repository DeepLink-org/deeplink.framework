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

import pickle
# with open('bmm1.pkl', 'wb') as f:
#     input = torch.randn(3, 3, 4)
#     pickle.dump(input, f)
with open('bmm1.pkl', 'rb') as f:
    input = pickle.load(f)
    
# with open('bmm2.pkl', 'wb') as f:
#     mat2 = torch.randn(3, 4, 5)
#     pickle.dump(mat2, f)
with open('bmm2.pkl', 'rb') as f:
    mat2 = pickle.load(f)

def fn(input, mat2):
    y = torch.bmm(input, mat2)
    return y

opt_model = torch.compile(fn, backend='ascendgraph')
#opt_model = torch.compile(fn, backend='inductor')

y = opt_model(input, mat2)
print(y)
print(y.shape)


ret = acl.rt.reset_device(0)
check_ret("acl.rt.reset_device", ret)
ret = acl.finalize()
check_ret("acl.finalize", ret)
print('Resources released successfully.')
