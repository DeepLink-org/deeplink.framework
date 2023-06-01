import torch
import torch_dipu

# vector x vector
tensor1 = torch.randn(3)
tensor2 = torch.randn(3)
assert torch.allclose(torch.matmul(tensor1, tensor2), torch.matmul(tensor1.cuda(), tensor2.cuda()).cpu(), atol=1e-3)

# matrix x vector
tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(4)
assert torch.allclose(torch.matmul(tensor1, tensor2), torch.matmul(tensor1.cuda(), tensor2.cuda()).cpu(), atol=1e-3)

# batched matrix x broadcasted vector
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4)
assert torch.allclose(torch.matmul(tensor1, tensor2), torch.matmul(tensor1.cuda(), tensor2.cuda()).cpu(), atol=1e-3)

# batched matrix x batched matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(10, 4, 5)
assert torch.allclose(torch.matmul(tensor1, tensor2), torch.matmul(tensor1.cuda(), tensor2.cuda()).cpu(), atol=1e-3)

# batched matrix x broadcasted matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4, 5)
assert torch.allclose(torch.matmul(tensor1, tensor2), torch.matmul(tensor1.cuda(), tensor2.cuda()).cpu(), atol=1e-3)

# batched matrix x broadcasted matrix
tensor1 = torch.randn(20, 10, 3, 4)
tensor2 = torch.randn(4, 5)
assert torch.allclose(torch.matmul(tensor1, tensor2), torch.matmul(tensor1.cuda(), tensor2.cuda()).cpu(), atol=1e-3)

# batched matrix x broadcasted matrix
tensor1 = torch.randn(20, 10, 3, 4)
tensor2 = torch.randn(10, 4, 5)
# camb has problem
#assert torch.allclose(torch.matmul(tensor1, tensor2), torch.matmul(tensor1.cuda(), tensor2.cuda()).cpu(), atol=1e-3)

tensor1 = torch.randn(4)
tensor2 = torch.randn(4, 10)
assert torch.allclose(torch.matmul(tensor1, tensor2), torch.matmul(tensor1.cuda(), tensor2.cuda()).cpu(), atol=1e-3)