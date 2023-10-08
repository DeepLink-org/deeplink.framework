import os
import argparse
import torch
import torch._dynamo as dynamo

os.environ.setdefault("DIPU_MOCK_CUDA", "false")
os.environ.setdefault("DICP_TOPS_DIPU", "True")
import torch_dipu


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
    device_name = torch_dipu.dipu.device.__dipu__
    device_index = "0"
    device = f"{device_name}:{device_index}"
    return device

def compile_model(model, backend, need_dynamic=False):
    static_model = torch.compile(model, backend=backend, dynamic=False)
    if need_dynamic:
        dynamic_model = torch.compile(model, backend=backend, dynamic=True)
        return [CompiledModel(static_model, False), CompiledModel(dynamic_model, True)]
    return [CompiledModel(static_model, False)]

def parse_bool_arg(arg):
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Bool value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--need_dynamic", type=parse_bool_arg, default=False)
    parser.add_argument("--backend", type=str, default=None)
    args, _ = parser.parse_known_args()
    # assert args.backend is not None
    return args
