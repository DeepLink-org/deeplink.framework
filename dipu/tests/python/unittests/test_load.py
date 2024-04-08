import tempfile
import os
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestLoad(TestCase):
    def test_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_p = os.path.join(tmpdir, "test_load.pth")
            
            t = torch.randn(3, 3).cuda()
            torch.save(t, save_p) 
            t_load = torch.load(save_p)

            self.assertEqual(t, t_load, prec=0)


if __name__ == "__main__":
    run_tests() 
