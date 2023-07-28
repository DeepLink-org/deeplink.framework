# (cpu) pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cpu 
# pip install openmim
# mim install mmcv-full mmcls

import os
import time
import argparse

import faulthandler
faulthandler.enable()

import torch
import mmcls
from mmcv import Config
from mmcls.models import build_classifier

def parse_config():
    # parse args
    args_parser =argparse.ArgumentParser("ResNet Op")
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
    # raise ValueError(resnet_cfg)
    return resnet_cfg, args.depth, args.backward

def run(cfg, depth, backward):
    # data
    inputs = torch.randn(4, 3, 226, 226)
    gt = torch.randint(100, (4,), dtype=torch.int64)

    model = build_classifier(cfg.model)
    model.train()
    compiled_model = torch.compile(model.forward_train, backend='topsgraph')
    print(f"warm up", flush=True)
    for i in range(0, 5):
        l = compiled_model(inputs, gt)
        l['loss'].backward()
    print(f"finish warm up", flush=True)
    
    loss = compiled_model(inputs, gt)
    
    if backward:
        loss['loss'].backward()
        
def main():
    params = parse_config()
    run(*params)

if __name__ == "__main__":
    main()
    print("Success!")
    