# (cpu) pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# pip install openmim
# mim install mmcv-full mmcls

import os
import argparse

import torch
import mmcls
from mmcv import Config
from mmcls.models import build_classifier
from op_collector import InnerCompilerOpCollectorContext
from torch._inductor.decomposition import decompositions
aten = torch.ops.aten
del decompositions[aten._native_batch_norm_legit_functional.default]
del decompositions[aten.native_batch_norm_backward.default]
del decompositions[aten.convolution_backward.default]
from dicp.TopsGraph.opset_transform import topsgraph_opset_transform
from dicp.common.compile_fx import *

def parse_config():
    # parse args
    args_parser = argparse.ArgumentParser("ResNet Op")
    args_parser.add_argument("--depth", "-d", choices=[18, 50, 101], type=int, default=18)
    args_parser.add_argument("--backward", "-b", action="store_true")
    args, _ = args_parser.parse_known_args()

    # config
    mmcls_root = os.path.dirname(mmcls.__file__)
    mim_dir = os.path.join(mmcls_root, ".mim")
    if os.path.exists(mim_dir):
        config_parent = mim_dir
    else:
        config_parent = os.path.dirname(mmcls_root)
    config_root = os.path.join(config_parent, 'configs')
    resnet_cfg = Config.fromfile(
        os.path.join(config_root, f'_base_/models/resnet{args.depth}.py')
    )
    return resnet_cfg, args.depth, args.backward

def run(cfg, depth, backward):
    # data
    inputs = torch.randn(4, 3, 224, 224)
    # 100 means [0, 100)
    gt = torch.randint(100, (4,), dtype=torch.int64)

    # model resnet
    collector_name = f"resnet{depth}_train"
    if backward:
        collector_name += "_with_backward"
    with InnerCompilerOpCollectorContext(collector_name=collector_name, write_file=True) as ctx:
        # print(f"resnet{depth} op:")
        model = build_classifier(cfg.model)
        # print(model)
        model.train()
        # compiled_model = torch.compile(model.forward_train, backend="topsgraph")
        compiled_model = torch.compile(model.forward_train, backend="inductor")

        loss = compiled_model(inputs, gt)
        if backward:
            loss['loss'].backward()
        for k, v in ctx.cached_gm_inputs_dict.items():
            m = v[0]
            traced = torch.fx.symbolic_trace(m)
            oG = f'****origin graph {traced.print_readable(print_output=False)}'
            print(oG)
            transformed = topsgraph_opset_transform(traced)
            nG = f'****new graph {transformed.print_readable(print_output=False)}'
            print(nG)

def main():
    params = parse_config()
    run(*params)

if __name__ == "__main__":
    main()
