import torch
import torch_dipu

a = torch.tensor((1, 2))
b = torch.tensor((0, 1))


def test_rsub(a, b, alpha=1):
    r1 = torch.rsub(a.cpu(), b.cpu(), alpha=alpha)
    r2 = torch.rsub(a.cuda(), b.cuda(), alpha=alpha).cpu()
    print(r1)
    print(r2)
    assert torch.allclose(r1, r2)

def test_rsub_scalar(a, b, alpha=1):
    r1 = torch.rsub(a.cpu(), b, alpha=alpha)
    r2 = torch.rsub(a.cuda(), b, alpha=alpha).cpu()
    print(r1)
    print(r2)
    assert torch.allclose(r1, r2)


test_rsub(a, b)
test_rsub(torch.ones(4,5), torch.ones(4,5) * 10)
test_rsub(torch.ones(4,5) * 1.1, torch.ones(4,5) * 5, alpha=4)
test_rsub_scalar(torch.ones(4,5), 10)
test_rsub_scalar(torch.ones(4,5), 10, 2.5)