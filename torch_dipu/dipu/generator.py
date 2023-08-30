import torch
from torch.types import _device
from typing import Union, Optional
from .device import __diputype__ as diputype
from .device import _get_device_index
from torch_dipu import _C


class Generator:
    generator: Optional[torch._C.Generator] = None
    dipu_generator: Optional[torch._C.Generator] = None
    device: _device

    def __new__(self, device: Union[_device, str, None] = None) -> None:
        if device is None:
            self.device = torch.device('cpu')
            self.generator = torch._C.Generator(device)
            return self.generator

        self.device = torch.device(device)
        if self.device.type == 'cpu':
            self.generator = torch._C.Generator(device)
            return self.generator
        elif self.device.type == diputype:
            self.dipu_generator = _C._create_dipu_generator(_get_device_index(self.device.index, True))
            return self.dipu_generator
        else:
            raise Exception(f"unsupport device type {self.device.type}")

    def get_state(self):
        if self.generator is not None:
            return self.generator.get_state()
        return self.dipu_generator.get_state()

    def set_state(self, new_state):
        if self.generator is not None:
            self.generator.set_state(new_state)
            return
        self.dipu_generator.set_state(new_state)

    def manual_seed(self, seed):
        if self.generator is not None:
            return self.generator.manual_seed(seed)
        return self.dipu_generator.manual_seed(seed)

    def seed(self):
        if self.generator is not None:
            return self.generator.seed()
        return self.dipu_generator.seed()

    def initial_seed(self):
        if self.generator is not None:
            return self.generator.initial_seed()
        return self.dipu_generator.initial_seed()


def apply_generator_patch():
    torch.Generator = Generator