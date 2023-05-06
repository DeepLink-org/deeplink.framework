import os

import torch
import mmcls
from mmcv import Config
from mmcls.models import build_classifier

import torch
import torch.fx

# config
mmcls_root = os.path.dirname(mmcls.__file__)
mim_dir = os.path.join(mmcls_root, ".mim")
if os.path.exists(mim_dir):
    config_parent = mim_dir
else:
    config_parent = os.path.dirname(mmcls_root)
config_root = os.path.join(config_parent, 'configs')
resnet101_cfg = Config.fromfile(
    os.path.join(config_root, '_base_/models/resnet101.py')
)
resnet50_cfg = Config.fromfile(
    os.path.join(config_root, '_base_/models/resnet50.py')
)
resnet18_cfg = Config.fromfile(
    os.path.join(config_root, '_base_/models/resnet18.py')
)

def check_ret(message, ret):
    if ret != 0:
        raise Exception("{} failed ret={}"
                        .format(message, ret))

import acl
device_id = 0
ret = acl.init()
check_ret("acl.init", ret)

ret = acl.rt.set_device(device_id)
check_ret("acl.rt.set_device", ret)

# data
inputs = torch.randn(4, 3, 224, 224)
gt = torch.randint(100, (4,), dtype=torch.int64) # 100 means [0, 100)

print("resnet18:")
model = build_classifier(resnet18_cfg.model)
model.train()
compiled_model = torch.compile(model.forward_train, backend='ascendgraph')
loss = compiled_model(inputs, gt)
loss['loss'].backward()

ret = acl.rt.reset_device(device_id)
check_ret("acl.rt.reset_device", ret)

ret = acl.finalize()
check_ret("acl.finalize", ret)
print('Resources released successfully.')
