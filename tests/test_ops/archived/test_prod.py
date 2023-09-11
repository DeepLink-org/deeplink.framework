import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestProd(TestCase):
    def _test_prod_with_args(self, input_tensor, *args, **kwargs):
        out = torch.prod(input_tensor, *args, **kwargs)
        out_cuda = torch.prod(input_tensor.cuda(), *args, **kwargs)
        self.assertRtolEqual(out, out_cuda.cpu())

    def test_prod(self):
        input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self._test_prod_with_args(input_tensor)

    def test_prod_dim_int(self):
        input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self._test_prod_with_args(input_tensor, 0)
        self._test_prod_with_args(input_tensor, 1)
        self._test_prod_with_args(input_tensor, 0, True)
        self._test_prod_with_args(input_tensor, 1, True)

    def test_prod_bool(self):
        input_arrays = [[True, True], [True, False], [False, False]]
        for input_array in input_arrays:
            input_tensor = torch.tensor(input_array)
            out = torch.prod(input_tensor).item()
            out_cuda = torch.prod(input_tensor.cuda()).item()
            self.assertTrue(out == out_cuda)


if __name__ == "__main__":
    run_tests()
