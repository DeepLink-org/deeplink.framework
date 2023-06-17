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
# with open('softmax.pkl', 'wb') as f:
#     softmax = torch.randn(2, 3)
#     pickle.dump(softmax, f)
with open('softmax.pkl', 'rb') as f:
    softmax = pickle.load(f)

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
