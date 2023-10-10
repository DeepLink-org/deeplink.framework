# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity
from torch_dipu.testing._internal.common_utils import TestCase, run_tests
from tests.utils.local_eviron import local_eviron


class TestProfiler(TestCase):
    def test_profiler(self):
        model = models.resnet18().cuda()
        inputs = torch.randn(5, 3, 224, 224).cuda()

        with local_eviron({"KINETO_LOG_LEVEL": "999"}):  # suppress profiler logs
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            ) as prof:
                output = model(inputs)
                output.sum().backward()

        profile_output = prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=1000
        )

        self.assertIn("::diopiConvolution2dBackward", profile_output)
        self.assertIn("dipu_convolution_", profile_output)
        self.assertIn("Self CPU time total", profile_output)
        self.assertIn("Self CUDA time total", profile_output)


if __name__ == "__main__":
    run_tests()
