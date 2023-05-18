import  torch
import torch_dipu

input = torch.tensor([1, 1, 1, 0, 1])

print(torch.nonzero(input))
print(torch.nonzero(input.cuda()))

torch.tensor([[0.6, 0.0, 0.0, 0.0],
              [0.0, 0.4, 0.0, 0.0],
              [0.0, 0.0, 1.2, 0.0],
              [0.0, 0.0, 0.0,-0.4]])

print(torch.nonzero(input))
print(torch.nonzero(input.cuda()))

input = torch.tensor([1, 1, 1, 0, 1])

print(torch.nonzero(input, as_tuple=True))
print(torch.nonzero(input.cuda(), as_tuple=True))

input = torch.tensor([[0.6, 0.0, 0.0, 0.0],
                            [0.0, 0.4, 0.0, 0.0],
                            [0.0, 0.0, 1.2, 0.0],
                            [0.0, 0.0, 0.0,-0.4]])

print(torch.nonzero(input, as_tuple=True))
print(torch.nonzero(input.cuda(), as_tuple=True))


print(torch.nonzero(torch.tensor(5), as_tuple=True))
print(torch.nonzero(torch.tensor(5).cuda(), as_tuple=True))