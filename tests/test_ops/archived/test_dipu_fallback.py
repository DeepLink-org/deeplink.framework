import os
os.environ['DIPU_FORCE_FALLBACK_OPS_LIST'] = 'add.out,sub.out'
#echo "add.out,sub.out" >> .dipu_force_fallback_op_list.config

import torch
import torch_dipu

x = torch.randn(3,4).cuda()

y = x + x
z = x - x