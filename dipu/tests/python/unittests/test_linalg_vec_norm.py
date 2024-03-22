# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests, onlyOn


@onlyOn("NPU")
class TestLinalgVectorNorm(TestCase):
    def test_linalg_vector_norm_keepdim(self):
        t1 = torch.randn((64, 128, 256)).to(torch.float16) - 4
        n1_cpu = torch.linalg.vector_norm(t1, keepdim=True)
        n1_dev = torch.linalg.vector_norm(t1.cuda(), keepdim=True)
        self.assertEqual(n1_cpu, n1_dev.cpu(), prec=1e-3)

        t2 = torch.randn((64, 128, 256)).to(torch.float32) - 4
        n2_cpu = torch.linalg.vector_norm(t2, dim=[1, 2], keepdim=True)
        n2_dev = torch.linalg.vector_norm(t2.cuda(), dim=[1, 2], keepdim=True)
        self.assertEqual(n2_cpu, n2_dev.cpu(), prec=1e-3)

        t3 = torch.randn((64, 128, 128)).to(torch.bfloat16) - 4
        n3_cpu = torch.linalg.vector_norm(t3, keepdim=True)
        n3_dev = torch.linalg.vector_norm(t3.cuda(), keepdim=True)
        self.assertEqual(n3_cpu, n3_dev.cpu(), prec=1e-3)

    def test_linalg_vector_norm_unkeepdim(self):
        t1 = torch.randn((64, 128, 256)).to(torch.float16) - 4
        n1_cpu = torch.linalg.vector_norm(t1)
        n1_dev = torch.linalg.vector_norm(t1.cuda())
        self.assertEqual(n1_cpu, n1_dev.cpu(), prec=1e-3)

        t2 = torch.randn((64, 128, 256)).to(torch.float32) - 4
        n2_cpu = torch.linalg.vector_norm(t2, dim=[1, 2])
        n2_dev = torch.linalg.vector_norm(t2.cuda(), dim=[1, 2])
        self.assertEqual(n2_cpu, n2_dev.cpu(), prec=1e-3)

        t3 = torch.randn((128, 128, 256)).to(torch.bfloat16) - 4
        n3_cpu = torch.linalg.vector_norm(t3, dim=[1, 2])
        n3_dev = torch.linalg.vector_norm(t3.cuda(), dim=[1, 2])
        self.assertEqual(n3_cpu, n3_dev.cpu(), prec=1e-3)


if __name__ == "__main__":
    run_tests()
