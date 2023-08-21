import torch
from torch.types import _device
from typing import Union

class Generator(torch.Generator):
    def __init__(self, device: Union[_device, str, None] = None):
        super().__init__(device)


def apply_generator_patch():
    torch.Generator = Generator