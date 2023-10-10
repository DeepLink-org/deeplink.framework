import torch
from torch.types import _device
from typing import Union, Optional
from .device import __diputype__ as diputype
from .device import _get_device_index
from torch_dipu import _C


class Generator:

    def __new__(cls, device: Union[_device, str, None] = None) -> None:
        if device is None:
            device = torch.device('cpu')
            generator = torch._C.Generator(device)
            return generator

        device = torch.device(device)
        if device.type == 'cpu':
            generator = torch._C.Generator(device)
            return generator
        elif device.type == diputype:
            dipu_generator = _C._create_dipu_generator(_get_device_index(device.index, True))
            return dipu_generator
        else:
            raise Exception(f"unsupport device type {device.type}")

def apply_generator_patch():
    torch.Generator = Generator