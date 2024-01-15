# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests, onlyOn
import torch._dynamo as dynamo
import subprocess

def check_string_in_directory(directory, search_string):
    grep_process = subprocess.Popen(["grep", "-r", search_string, directory], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

        self.assertTrue(check_string_in_directory(path, "test_profiler.py"))
        self.assertTrue(check_string_in_directory(path, "aten::add_"))
        self.assertTrue(check_string_in_directory(path, "Add"))


    @onlyOn("NPU")
    def test_dicp_profiler(self):
        def fn(x):
            y = torch.nn.functional.softmax(x, -1)
            y = y * 5
            y = torch.relu(y)
            return y

        opt_model = torch.compile(fn, backend='ascendgraph')
        input = torch.randn(2, 3).cuda()
        # warmup
        for _ in range(5):
            opt_model(input)
        path = "./results/dicp/"
        with torch_dipu.profiler.NativeProfile(path, True):
            y = opt_model(input)
            z = y + y

        self.assertTrue(check_string_in_directory(path, "test_profiler.py"))
        self.assertTrue(check_string_in_directory(path, "aten::add"))
        self.assertTrue(check_string_in_directory(path, "mulrelu"))
        self.assertTrue(check_string_in_directory(path, "softmax"))


if __name__ == "__main__":
    run_tests()
