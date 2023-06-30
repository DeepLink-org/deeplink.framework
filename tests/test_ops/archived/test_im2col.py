import torch
import torch_dipu


im2col = torch._C._nn.im2col
col2im = torch._C._nn.col2im


def _test_im2col_col2im(input, kernel_size, dilation, padding, stride):
    y_cpu = im2col(input, kernel_size = kernel_size, dilation = dilation, padding = padding, stride = stride)
    y_device = im2col(input.cuda(), kernel_size = kernel_size, dilation = dilation, padding = padding, stride = stride)
    assert torch.allclose(y_cpu, y_device.cpu())

    print(y_device.size())
    im_cpu = col2im(y_cpu, input.shape[1:], kernel_size = kernel_size,  dilation = dilation, padding = padding, stride = stride)
    im_device = col2im(y_device, input.shape[1:], kernel_size = kernel_size,  dilation = dilation, padding = padding, stride = stride)
    assert torch.allclose(im_cpu, im_device.cpu(), atol = 1e-5, rtol = 1e-5)

for shape in ([1, 7, 8],):
    numel = 1
    for i in range(len(shape)):
        numel = numel * shape[i]
    im = torch.arange(numel).reshape(shape).float()
    print(im.size())
    _test_im2col_col2im(im, kernel_size = (1, 1), dilation = (1, 1), padding = (0, 0), stride = (1, 1))
    _test_im2col_col2im(im, kernel_size = (2, 3), dilation = (1, 1), padding = (0, 0), stride = (1, 1))
    _test_im2col_col2im(im, kernel_size = (3, 3), dilation = (1, 1), padding = (0, 0), stride = (1, 1))
    _test_im2col_col2im(im, kernel_size = (3, 3), dilation = (1, 1), padding = (0, 0), stride = (1, 1))
    _test_im2col_col2im(im, kernel_size = (3, 3), dilation = (2, 3), padding = (0, 0), stride = (1, 1))
    _test_im2col_col2im(im, kernel_size = (3, 3), dilation = (1, 1), padding = (4, 5), stride = (1, 1))
    _test_im2col_col2im(im, kernel_size = (3, 3), dilation = (2, 2), padding = (3, 3), stride = (2, 2))