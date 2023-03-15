import os

import torch
import mmcls
from mmcv import Config
from mmcls.models import build_classifier

from op_collector import InnerCompilerOpCollectorContext

import torch
import torch.fx
from opset_convert import topsgraph_opset_convert
from torch.tops.operator import *
import time

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


# data
inputs = torch.randn(4, 3, 224, 224)
gt = torch.randint(100, (4,), dtype=torch.int64) # 100 means [0, 100)

# model resnet18
with InnerCompilerOpCollectorContext(collector_name="resnet18_1_2.0.0a0+git004c3f5", write_file=True) as ctx:
    print("resnet18:")
    model = build_classifier(resnet18_cfg.model)
    model.train()
    #print(type(model))
    compiled_model = torch.compile(model.forward_train)
    # loss = compiled_model(inputs, gt)
    loss = compiled_model(inputs, gt)
    for k, v in ctx.cached_gm_inputs_dict.items():
        #print(v[0])
        #transformed = topsgraph_opset_convert(v[0])
        print(type(v[0]))
        m = v[0]
        traced = torch.fx.symbolic_trace(m)
        print(traced.graph)
        transformed = topsgraph_opset_convert(traced)
        #print(f"*****trans: \n{transformed.graph}")
        newG = f'****origin graph{transformed.graph}'
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filen = "/home/test/"+timestr+".py"
        nf = open(filen, 'a')
        nf.write(newG)
        nf.close()




        #print(transformed.graph)

    # loss['loss'].backward()

        
