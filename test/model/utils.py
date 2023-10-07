import os
os.environ.setdefault("DIPU_MOCK_CUDA", "false")
os.environ.setdefault("DICP_TOPS_DIPU", "True")
import argparse
import torch
import torch._dynamo as dynamo
import torch_dipu


def update_dynamo_config(dynamic=True):
    if dynamic:
        dynamo.config.dynamic_shapes = True
        dynamo.config.assume_static_by_default = False
    else:
        dynamo.config.dynamic_shapes = False
        dynamo.config.assume_static_by_default = True

def get_device():
    device_name = torch_dipu.dipu.device.__dipu__
    device_index = "7"
    return f"{device_name}:{device_index}"

def parse_bool_arg(arg):
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Bool value expected.')
