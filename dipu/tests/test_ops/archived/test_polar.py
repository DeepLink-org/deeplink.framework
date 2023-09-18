import torch
import torch_dipu

import numpy as np
abs = torch.tensor([1, 2], dtype=torch.float64)
angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
assert torch.allclose(torch.polar(abs, angle), torch.polar(abs.cuda(), angle.cuda()).cpu())