# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu

torch.seed()
torch.manual_seed(1)
assert torch.cuda.initial_seed() == 1
assert torch.initial_seed() == 1
for i in range(torch.cuda.device_count()):
    torch.cuda.manual_seed(i)

state = torch.cuda.get_rng_state(0)
print("first state = ", state)
new_state = torch.ones_like(state)
torch.cuda.set_rng_state(new_state, 0)
current_state = torch.cuda.get_rng_state(0)
print("second state = ", current_state)

t1 = torch.arange(0, 100, dtype=torch.float32).cuda()
t2 = t1.clone()
torch.manual_seed(1)
t1.uniform_()
torch.manual_seed(1)
t2.uniform_()
print(f"t1 = {t1}")
print(f"t2 = {t2}")
assert torch.allclose(t1, t2)
print("allclose success")