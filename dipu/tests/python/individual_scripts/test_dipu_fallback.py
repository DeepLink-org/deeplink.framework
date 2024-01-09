# Copyright (c) 2023, DeepLink.
import io
import torch
from stdout_redirector import stdout_redirector
from local_eviron import local_eviron
from multiprocessing import Process, set_start_method


def test_fallback(
    op_name: list, diopi_proto: list, test_fn: callable, other_check_item: list = None
) -> str:
    captured = io.BytesIO()
    with stdout_redirector(captured):
        with local_eviron(
            {
                "DIPU_FORCE_FALLBACK_OPS_LIST": ",".join(op_name),
                "DIPU_DUMP_OP_ARGS": "1",
                "DIPU_LOG_FALLBACK_INFO": "1",
            }
        ):
            import torch_dipu

            test_fn()
    output = captured.getvalue().decode()
    print(output)
    assert all(
        f"force fallback has been set, {name} will be fallback to cpu" in output
        for name in op_name
    )
    assert all(item not in output for item in diopi_proto)
    if other_check_item is not None:
        assert all(item in output for item in other_check_item)


def _test_dipu_fallback():
    def fn():
        x = torch.randn(3, 4).cuda()
        _ = x + x
        _ = x - x

    test_fallback(
        ["add.out", "sub.out"], ["diopiAdd", "diopiSub"], fn, ["dipu_fallback"]
    )


def _test_cpu_fallback():
    def fn():
        device = "cuda"
        m = torch.nn.BatchNorm2d(100, affine=False).to(device)
        input = torch.randn(20, 100, 35, 45).to(device)
        m(input)

    test_fallback(
        ["native_batch_norm"],
        ["diopiBatchNorm"],
        fn,
        ["cpu_fallback:\taten::native_batch_norm", "dipu_fallback"],
    )


def _test_dipu_index_put_impl_fallback():
    def fn():
        dipu_tensor = torch.tensor([1, 2, 3, 4, 5]).cuda()
        indices = torch.tensor([1, 3]).cuda()
        values = torch.tensor([10, 40]).cuda()
        torch._index_put_impl_(dipu_tensor, (indices,), values, accumulate=False)

        tensor = dipu_tensor.cpu()
        indices = indices.cpu()
        values = values.cpu()
        torch._index_put_impl_(tensor, (indices,), values, accumulate=False)

        assert torch.allclose(tensor, dipu_tensor.cpu())

    test_fallback(
        ["_index_put_impl_"],
        ["diopiIndexPut"],
        fn,
        ["custom fallback to cpu, name=_index_put_impl_"],
    )


def _test_dipu_copy_fallback_():
    def fn():
        source_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        target_dipu = torch.zeros_like(source_tensor).cuda()
        target_dipu.copy_(source_tensor)

        source_tensor = source_tensor.cpu()
        target_tensor = torch.zeros_like(source_tensor)
        target_tensor.copy_(source_tensor)

        assert torch.allclose(target_tensor, target_dipu.cpu())

    test_fallback(
        ["copy_"],
        ["diopiCopyInp"],
        fn,
        ["custom fallback to dipu copy, name=copy_"],
    )


def _test_dipu_convolution_backward_overrideable_fallback():
    def fn():
        torch.manual_seed(42)
        device = torch.device("dipu")
        m = torch.nn.Conv2d(2, 3, 3, stride=2).to(device)
        m.weight = torch.nn.Parameter(torch.ones_like(m.weight))
        m.bias = torch.nn.Parameter(torch.ones_like(m.bias))
        input_dipu = torch.randn(2, 2, 5, 5).to(device).requires_grad_(True)
        output_dipu = m(input_dipu)
        output_dipu.backward(torch.ones_like(output_dipu))

        torch.manual_seed(42)
        m = torch.nn.Conv2d(2, 3, 3, stride=2)
        m.weight = torch.nn.Parameter(torch.ones_like(m.weight))
        m.bias = torch.nn.Parameter(torch.ones_like(m.bias))
        input_cpu = torch.randn(2, 2, 5, 5, requires_grad=True)
        output_cpu = m(input_cpu)
        output_cpu.backward(torch.ones_like(output_cpu))

        assert torch.allclose(output_dipu.cpu(), output_cpu)
        assert torch.allclose(input_dipu.grad.cpu(), input_cpu.grad)

    test_fallback(
        ["convolution_backward_overrideable"],
        ["diopiConvolution2dBackward"],
        fn,
        ["custom fallback to cpu, name=convolution_backward_overrideable"],
    )


def _test_dipu_convolution_overrideable_fallback():
    def fn():
        m = torch.nn.Conv2d(2, 3, 3, stride=2).cuda()
        m.weight = torch.nn.Parameter(torch.ones_like(m.weight))
        m.bias = torch.nn.Parameter(torch.ones_like(m.bias))
        input_dipu = torch.randn(2, 2, 5, 5).cuda()
        output_dipu = m(input_dipu)

        m = m.cpu()
        m.weight = torch.nn.Parameter(torch.ones_like(m.weight))
        m.bias = torch.nn.Parameter(torch.ones_like(m.bias))
        input_cpu = input_dipu.cpu()
        output_cpu = m(input_cpu)

        assert torch.allclose(output_dipu.cpu(), output_cpu)

    test_fallback(
        ["convolution_overrideable"],
        ["diopiConvolution2d"],
        fn,
        ["custom fallback to cpu, name=convolution_overrideable"],
    )

def _test_dipu_silu_fallback():
    def fn():
        m = torch.nn.SiLU().cuda()
        input_dipu = torch.tensor([1, 2, 3, 4]).cuda()
        out_dipu = m(input_dipu)

        m = m.cpu()
        input_cpu = input_dipu.cpu()
        out_cpu = m(input_cpu)

        assert torch.allclose(out_dipu.cpu(), out_cpu)

    test_fallback(
        ["silu.out"],
        ["diopiSilu"],
        fn,
        ["custom fallback to cpu, name=silu.out"],
    )


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    p1 = Process(target=_test_dipu_fallback)
    p1.start()
    p1.join()

    p2 = Process(target=_test_cpu_fallback)
    p2.start()
    p2.join()

    p3 = Process(target=_test_dipu_index_put_impl_fallback)
    p3.start()
    p3.join()

    p4 = Process(target=_test_dipu_copy_fallback_)
    p4.start()
    p4.join()

    p5 = Process(target=_test_dipu_convolution_backward_overrideable_fallback)
    p5.start()
    p5.join()

    p6 = Process(target=_test_dipu_convolution_overrideable_fallback)
    p6.start()
    p6.join()
    
    p7 = Process(target=_test_dipu_silu_fallback)
    p7.start()
    p7.join()
