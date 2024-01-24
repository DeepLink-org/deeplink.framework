# Copyright (c) 2023, DeepLink.
import os

os.environ["FORCE_USE_DIPU_PROFILER"] = "True"

import tempfile
import torch
import torch_dipu
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity
from torch_dipu.testing._internal.common_utils import TestCase, run_tests, onlyOn
from utils.local_eviron import local_eviron


class TestProfiler(TestCase):
    def test_profiler(self):
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
        self.assertIn("diopiConvolution2dBackward", profile_output)
        self.assertIn("dipu_convolution_", profile_output)
        self.assertIn("LaunchKernel_dipu", profile_output)
        self.assertIn("LaunchKernel_diopi", profile_output)
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
            prof.export_chrome_trace(f"{tmpdir}/dipu_resnet18_profiler.json")


if __name__ == "__main__":
    run_tests()
