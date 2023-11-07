import torch_dipu
import torch

from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestSchema(TestCase):
    def _test_im2col_col2im(self, input, kernel_size, dilation, padding, stride):
        im2col = torch._C._nn.im2col
        col2im = torch._C._nn.col2im

        y_cpu = im2col(
            input,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        y_device = im2col(
            input.cuda(),
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        self.assertTrue(torch.allclose(y_cpu, y_device.cpu()))

        im_cpu = col2im(
            y_cpu,
            input.shape[1:],
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        im_device = col2im(
            y_device,
            input.shape[1:],
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        self.assertTrue(torch.allclose(im_cpu, im_device.cpu(), atol=1e-5, rtol=1e-5))

    def test_im2col_col2im(self):
        # input_dim=3
        for shape in ([1, 7, 8],):
            numel = 1
            for i in range(len(shape)):
                numel = numel * shape[i]
            im = torch.arange(numel).reshape(shape).float()
            self._test_im2col_col2im(
                im, kernel_size=(1, 1), dilation=(1, 1), padding=(0, 0), stride=(1, 1)
            )
            self._test_im2col_col2im(
                im, kernel_size=(2, 3), dilation=(1, 1), padding=(0, 0), stride=(1, 1)
            )
            self._test_im2col_col2im(
                im, kernel_size=(3, 3), dilation=(1, 1), padding=(0, 0), stride=(1, 1)
            )
            self._test_im2col_col2im(
                im, kernel_size=(3, 3), dilation=(1, 1), padding=(0, 0), stride=(1, 1)
            )
            self._test_im2col_col2im(
                im, kernel_size=(3, 3), dilation=(2, 3), padding=(0, 0), stride=(1, 1)
            )
            self._test_im2col_col2im(
                im, kernel_size=(3, 3), dilation=(1, 1), padding=(4, 5), stride=(1, 1)
            )
            self._test_im2col_col2im(
                im, kernel_size=(3, 3), dilation=(2, 2), padding=(3, 3), stride=(2, 2)
            )

    def test_fold_unfold(self):
        # which use im2col and col2im and input_dim=4
        # demo from pytorch
        unfold_cuda = torch.nn.Unfold(kernel_size=(2, 3)).cuda()
        input_cuda = torch.randn(2, 5, 3, 4).cuda()

        input_cpu = input_cuda.cpu()
        unfold_cpu = unfold_cuda.cpu()

        output_cuda = unfold_cuda(input_cuda)
        output_cpu = unfold_cpu(input_cpu)

        self.assertTrue(torch.allclose(output_cuda.cpu(), output_cpu))

        inp_cuda = torch.randn(1, 3, 10, 12).cuda()
        w_cuda = torch.randn(2, 3, 4, 5).cuda()
        inp_unf_cuda = torch.nn.functional.unfold(inp_cuda, (4, 5))
        out_unf_cuda = (
            inp_unf_cuda.transpose(1, 2)
            .matmul(w_cuda.view(w_cuda.size(0), -1).t())
            .transpose(1, 2)
        )
        out_cuda = torch.nn.functional.fold(out_unf_cuda, (7, 8), (1, 1))

        inp = inp_cuda.cpu()
        w = w_cuda.cpu()
        inp_unf = torch.nn.functional.unfold(inp, (4, 5))
        out_unf = (
            inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        )
        out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))

        self.assertTrue(torch.allclose(out_cuda.cpu(), out, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    run_tests()
