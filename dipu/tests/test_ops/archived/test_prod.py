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
            self.assertEqual(out, out_cuda)

    def test_prod_dtype(self):
        test_dtypes = [torch.float16, torch.float32]
        for input_dtype in test_dtypes:
            input_tensor = torch.tensor(
                [[1, 2, 3], [4, 5, 6]], dtype=input_dtype, device='dipu')
            for output_dtype in test_dtypes:
                expected_output = torch.tensor(
                    720, dtype=output_dtype, device='dipu')
                out = torch.prod(input_tensor, dtype=output_dtype)
                self.assertEqual(out, expected_output, exact_dtype=True)

                expected_output = torch.tensor(
                    [6, 120], dtype=output_dtype, device='dipu')
                out = torch.prod(input_tensor, 1, dtype=output_dtype)
                self.assertEqual(out, expected_output, exact_dtype=True)


if __name__ == "__main__":
    run_tests()
