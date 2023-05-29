# Copyright (c) 2023, DeepLink.
import random
import numpy as np

import torch
import torch_dipu

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("dipu")
input = torch.randn(2, 3, 4).to(device)
t = input.view(3, -1)
m = torch.mean(t, 1)
v = torch.var(t, 1)

result = torch.nn.functional.batch_norm(input, m, v, training=True)
print(result)

result = torch.nn.functional.batch_norm(input.cpu(), m.cpu(), v.cpu(), training=True)
print(result)
