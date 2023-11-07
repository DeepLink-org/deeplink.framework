# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestLinalgQr(TestCase):
    def _test_linalg_qr(self, A_list):
        for A in A_list:
            Q1, R1 = torch.linalg.qr(A)
            Q2, R2 = torch.linalg.qr(A.cuda())
            self.assertEqual(Q1, Q2.cpu(), prec=1e-4)
            self.assertEqual(R1, R2.cpu(), prec=1e-3)

            Q1, R1 = torch.linalg.qr(A, mode="r")
            Q2, R2 = torch.linalg.qr(A.cuda(), mode="r")
            self.assertEqual(Q1, Q2.cpu(), prec=1e-3)
            self.assertEqual(R1, R2.cpu(), prec=1e-3)

            Q1, R1 = torch.linalg.qr(A, mode="complete")
            Q2, R2 = torch.linalg.qr(A.cuda(), mode="complete")
            self.assertEqual(Q1, Q2.cpu(), prec=1e-3)
            self.assertEqual(R1, R2.cpu(), prec=1e-3)

    def test_linalg_qr(self):
        A_list = []
        shapeList = [
            (1024, 384),
            (384, 1024),
            (64, 1, 128),
            (128, 64, 32, 3),
            (2, 32, 130, 100),
            (2, 32, 100, 150),
            (1024, 1024),
            (4, 284, 384),
            (3, 64, 64),
        ]
        for shape in shapeList:
            A = torch.randn(shape)
            A_list.append(A)

        self._test_linalg_qr(A_list)


if __name__ == "__main__":
    run_tests()
