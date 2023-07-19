import torch
import torch_dipu


def test_linalg_qr(A_list):
    for A in A_list:
        Q1, R1 = torch.linalg.qr(A)
        Q2, R2 = torch.linalg.qr(A.cuda())
        assert torch.allclose(Q1, Q2.cpu(), atol=1e-4)
        assert torch.allclose(R1, R2.cpu(), atol=1e-4)

        Q1, R1 = torch.linalg.qr(A, mode='r')
        Q2, R2 = torch.linalg.qr(A.cuda(), mode='r')
        assert torch.allclose(Q1, Q2.cpu(), atol=1e-4)
        assert torch.allclose(R1, R2.cpu(), atol=1e-4)

        Q1, R1 = torch.linalg.qr(A, mode='complete')
        Q2, R2 = torch.linalg.qr(A.cuda(), mode='complete')
        assert torch.allclose(Q1, Q2.cpu(), atol=1e-4)
        assert torch.allclose(R1, R2.cpu(), atol=1e-4)

A_list = []
shapeList = [(1024, 384), (384, 1024), (64, 1, 128), (128, 64, 32, 3), (2, 32, 130, 100), (2, 32, 100, 150), (1024, 1024), (4, 284, 384), (3, 64, 64)]
for shape in shapeList:
    A = torch.randn(shape)
    A_list.append(A)

test_linalg_qr(A_list)
