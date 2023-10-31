import os
import argparse
import torch
import operator
import random
import torch._dynamo as dynamo

os.environ.setdefault("DIPU_MOCK_CUDA", "false")
os.environ.setdefault("DICP_TOPS_DIPU", "True")
torch.manual_seed(1)
import torch_dipu
import pytest


class CompiledModel():
    def __init__(self, model, dynamic):
        self.model = model
        self.dynamic = dynamic


class Size():
    def __init__(self, static_size, dynamic_size):
        self.static = static_size
        self.dynamic = dynamic_size


def update_dynamo_config(dynamic=False):
    if dynamic:
        dynamo.config.dynamic_shapes = True
        dynamo.config.assume_static_by_default = False
    else:
        dynamo.config.dynamic_shapes = False
        dynamo.config.assume_static_by_default = True


def get_device():
    device_name = torch_dipu.dipu.device.__dipu__
    device_index = "0"
    device = f"{device_name}:{device_index}"
    return device


def compile_model(model, backend, dynamic=False):
    if dynamic:
        dynamic_model = torch.compile(model, backend=backend, dynamic=True)
        return [CompiledModel(dynamic_model, True)]
    static_model = torch.compile(model, backend=backend, dynamic=False)
    return [CompiledModel(static_model, False)]


def parse_bool_arg(arg):
    if isinstance(arg, bool):
        return arg
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError('Bool value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic", type=parse_bool_arg, default=False)
    parser.add_argument("--backend", type=str, default=None)
    args, _ = parser.parse_known_args()
    # assert args.backend is not None
    return args
