# Copyright (c) 2023, DeepLink.
import tempfile
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests, onlyOn
import torch._dynamo as dynamo
import subprocess
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity
from utils.local_eviron import local_eviron


def check_string_in_directory(directory, search_string):
    grep_process = subprocess.Popen(
        ["grep", "-r", search_string, directory],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output, _ = grep_process.communicate()
    if output:
        return True
    else:
        return False


class TestProfiler(TestCase):
    @onlyOn("NPU")
    def test_aot_profiler(self):
        x = torch.randn(3, 4).cuda()
        y = torch.randn(3, 4).cuda()
        path = "./results/aot/"
        with torch_dipu.profiler.NativeProfile(path, True):
            x.add_(y)

        self.assertTrue(check_string_in_directory(path, "test_profiler_vendor.py"))
        self.assertTrue(check_string_in_directory(path, "aten::add_"))
        self.assertTrue(check_string_in_directory(path, "Add"))

    @onlyOn("NPU")
    def test_dicp_profiler(self):
        def fn(x):
            y = torch.nn.functional.softmax(x, -1)
            y = y * 5
            y = torch.relu(y)
            return y

        opt_model = torch.compile(fn, backend="ascendgraph")
        input = torch.randn(2, 3).cuda()
        # warmup
        for _ in range(5):
            opt_model(input)
        path = "./results/dicp/"
        with torch_dipu.profiler.NativeProfile(path, True):
            y = opt_model(input)
            z = y + y

        self.assertTrue(check_string_in_directory(path, "test_profiler_vendor.py"))
        self.assertTrue(check_string_in_directory(path, "aten::add"))
        self.assertTrue(check_string_in_directory(path, "mulrelu"))
        self.assertTrue(check_string_in_directory(path, "softmax"))

    @onlyOn("CUDA")
    def test_profiler_cuda(self):
        model = models.resnet18().cuda()
        inputs = torch.randn(5, 3, 224, 224).cuda()

        with local_eviron({"KINETO_LOG_LEVEL": "999"}):  # suppress profiler logs
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                record_shapes=True,
                with_modules=True,
                with_stack=True,
                experimental_config=torch._C._profiler._ExperimentalConfig(
                    verbose=True
                ),
            ) as prof:
                output = model(inputs)
                output.sum().backward()

        profile_output = prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=1000
        )
        self.assertNotIn("diopiConvolution2dBackward", profile_output)
        self.assertNotIn("dipu_convolution_", profile_output)
        self.assertNotIn("LaunchKernel_dipu", profile_output)
        self.assertNotIn("LaunchKernel_diopi", profile_output)
        self.assertIn("aten::cudnn_convolution", profile_output)
        self.assertIn("aten::add", profile_output)
        self.assertIn("vectorized_elementwise_kernel", profile_output)
        self.assertIn("Self CPU time total", profile_output)
        self.assertIn("Self CUDA time total", profile_output)
        self.assertIn("5, 3, 224, 224", profile_output)

        profile_stack_output = prof.key_averages(group_by_stack_n=15).table(
            sort_by="cuda_time_total", row_limit=1000
        )
        self.assertIn("Source Location", profile_stack_output)
        self.assertIn("resnet.py", profile_stack_output)

        profile_memory_output = prof.key_averages().table(
            sort_by="self_cuda_memory_usage", row_limit=1000
        )
        self.assertIn("Self CPU Mem", profile_memory_output)
        self.assertIn("Self CUDA Mem", profile_memory_output)
        self.assertIn("Mb", profile_memory_output)
        self.assertIn("Kb", profile_memory_output)

        with tempfile.TemporaryDirectory() as tmpdir:
            prof.export_chrome_trace(f"{tmpdir}/resnet18_profiler_cuda.json")

    @onlyOn("MLU")
    def test_profiler_mlu(self):
        model = models.resnet18().cuda()
        inputs = torch.randn(5, 3, 224, 224).cuda()

        with local_eviron({"KINETO_LOG_LEVEL": "999"}):  # suppress profiler logs
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                record_shapes=True,
                with_modules=True,
                with_stack=True,
                experimental_config=torch._C._profiler._ExperimentalConfig(
                    verbose=True
                ),
            ) as prof:
                output = model(inputs)
                output.sum().backward()

        profile_output = prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=1000
        )
        self.assertNotIn("diopiConvolution2dBackward", profile_output)
        self.assertNotIn("dipu_convolution_", profile_output)
        self.assertNotIn("LaunchKernel_dipu", profile_output)
        self.assertNotIn("LaunchKernel_diopi", profile_output)
        self.assertIn("cnInvokeKernel", profile_output)
        self.assertIn("cnnlConvolutionForward", profile_output)
        self.assertIn("aten::add", profile_output)
        self.assertIn("cnnlOpTensor", profile_output)
        self.assertIn("Self CPU time total", profile_output)
        self.assertIn("Self CUDA time total", profile_output)
        self.assertIn("5, 3, 224, 224", profile_output)

        profile_stack_output = prof.key_averages(group_by_stack_n=15).table(
            sort_by="cuda_time_total", row_limit=1000
        )
        self.assertIn("Source Location", profile_stack_output)
        self.assertIn("resnet.py", profile_stack_output)

        profile_memory_output = prof.key_averages().table(
            sort_by="self_cuda_memory_usage", row_limit=1000
        )
        self.assertIn("Self CPU Mem", profile_memory_output)
        self.assertIn("Self CUDA Mem", profile_memory_output)
        self.assertIn("Mb", profile_memory_output)
        self.assertIn("Kb", profile_memory_output)

        with tempfile.TemporaryDirectory() as tmpdir:
            prof.export_chrome_trace(f"{tmpdir}/resnet18_profiler_mlu.json")


if __name__ == "__main__":
    run_tests()
