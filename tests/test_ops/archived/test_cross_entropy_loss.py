import torch
import torch.nn.functional as F
import torch_dipu

def test_cross_entropy_loss(input, target, devicestr : str):
    device = torch.device(devicestr)
    input = input.to(device)
    input.requires_grad_(True)
    target = target.to(device)
    loss = F.cross_entropy(input, target)
    print(f"loss = {loss}")

    loss.backward()
    print(f"input.grad = {input.grad}")


input = torch.randn(3, 5)
# target with class indices
target = torch.randint(5, (3,), dtype = torch.int64)
test_cross_entropy_loss(input, target, "dipu")
test_cross_entropy_loss(input, target, "cpu")

# target with class probabilities
input = torch.randn(3, 5)
target = torch.randn(3, 5).softmax(dim = 1)
test_cross_entropy_loss(input, target, "dipu")
test_cross_entropy_loss(input, target, "cpu")
