import os
import argparse
import torch_dipu
import torch
import torch._dynamo as dynamo
import pytest
os.environ.setdefault("DICP_TOPS_DIPU", "True")


class CompiledModel():
    def __init__(self, model, dynamic):
        self.model = model
        self.dynamic = dynamic


def update_dynamo_config(dynamic=False):
    if dynamic:
        dynamo.config.dynamic_shapes = True
        dynamo.config.assume_static_by_default = False
    else:
        dynamo.config.dynamic_shapes = False
        dynamo.config.assume_static_by_default = True

def get_device():
    return torch_dipu.dipu.device.__dipu__

def compile_model(model, backend, need_dynamic=False):
    static_model = torch.compile(model, backend=backend, dynamic=False)
    if need_dynamic:
        dynamic_model = torch.compile(model, backend=backend, dynamic=True)
        return [CompiledModel(static_model, False), CompiledModel(dynamic_model, True)]
    return [CompiledModel(static_model, False)]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--need_dynamic", type=bool, default=False)
    parser.add_argument("--backend", type=str, default=None)
    args, _ = parser.parse_known_args()
    assert args.backend is not None
    return args
