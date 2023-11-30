# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestPinMemory(TestCase):
    def test_pin_memory(self):
        a = torch.tensor([1, 2, 3])
        self.assertFalse(a.is_pinned())

        b = a.pin_memory()
        self.assertTrue(b.is_pinned())
        self.assertEqual(a, b)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())

        c = b.pin_memory()
        self.assertTrue(c.is_pinned())
        self.assertEqual(a, c)
        self.assertEqual(b.data_ptr(), c.data_ptr())

        x = torch.randn(3, 4, pin_memory=True)
        self.assertTrue(x.is_pinned())

        x = torch.randn(3, 4, pin_memory=False)
        self.assertFalse(x.is_pinned())

        x = torch.empty(3, 4, pin_memory=True)
        self.assertTrue(x.is_pinned())

        x = torch.empty(3, 4, pin_memory=False)
        self.assertFalse(x.is_pinned())


if __name__ == "__main__":
    run_tests()
