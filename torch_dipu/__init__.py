import os

import torch

# use env to control?
mockcuda = True
from torch_dipu import dipu

from torch_dipu.dipu import diputype
from torch_dipu.dipu  import vendor_type