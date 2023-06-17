import torch
import torch._dynamo as dynamo
import logging

# dynamo.config.verbose = True
# dynamo.config.log_level  = logging.INFO
# dynamo.config.output_code  = True

torch._logging.set_logs(dynamo=logging.DEBUG)


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
# with open('mm1.pkl', 'wb') as f:
#     mm1 = torch.randn(2, 3)
#     pickle.dump(mm1, f)
with open('mm1.pkl', 'rb') as f:
    mm1 = pickle.load(f)
    
# with open('mm2.pkl', 'wb') as f:
#     mm2 = torch.randn(3, 3)
#     pickle.dump(mm2, f)
with open('mm2.pkl', 'rb') as f:
    mm2 = pickle.load(f)

def fn(mm1, mm2):
    y = torch.mm(mm1, mm2)
    return y

opt_model = torch.compile(fn, backend='ascendgraph')
#opt_model = torch.compile(fn, backend='inductor')

y = opt_model(mm1, mm2)
print(y)
print(y.shape)


ret = acl.rt.reset_device(0)
check_ret("acl.rt.reset_device", ret)
ret = acl.finalize()
check_ret("acl.finalize", ret)
print('Resources released successfully.')
