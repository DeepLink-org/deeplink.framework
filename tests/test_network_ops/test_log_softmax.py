import torch
import numpy as np

from torch_dipu.testing._internal.testcase import TestCase, run_tests
from torch_dipu.testing._internal.common_utils import create_common_tensor


class TestLogSoftmax(TestCase):
    def cpu_op_exec(self, input1, dim):
        output = torch.nn.functional.log_softmax(input1, dim)
        output = output.numpy()
        return output

    def dipu_op_exec_new(self, input1, dim):
        output = torch.nn.functional.log_softmax(input1, dim)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def logsoftmax_result(self, shape_format):
        for item in shape_format:
            cpu_input1, dipu_input1 = create_common_tensor(item, 0, 10)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
           
            cpu_output = self.cpu_op_exec(cpu_input1, 0)
            dipu_output = self.dipu_op_exec_new(dipu_input1, 0)
            cpu_output = cpu_output.astype(dipu_output.dtype)
            self.assertRtolEqual(cpu_output, dipu_output)

    def test_logsoftmax_shape_format_fp16_2d(self):
        format_list = [0, 3]
        shape_format = [
            [np.float16, i, [256, 1000]] for i in format_list 
        ]
        self.logsoftmax_result(shape_format)
        
    def test_logsoftmax_shape_format_fp32_2d(self):
        format_list = [0, 3]
        shape_format = [
            [np.float32, i, [256, 1000]] for i in format_list 
        ]
        self.logsoftmax_result(shape_format)
        
    def test_logsoftmax_shape_format_fp16_3d(self):
        format_list = [0, 3]
        shape_format = [
            [np.float16, i, [32, 48, 64]] for i in format_list 
        ]
        self.logsoftmax_result(shape_format)
        
    def test_logsoftmax_shape_format_fp32_3d(self):
        format_list = [0, 3]
        shape_format = [
            [np.float32, i, [32, 48, 1024]] for i in format_list
        ]
        self.logsoftmax_result(shape_format)
        
    def test_logsoftmax_shape_format_fp16_4d(self):
        format_list = [0, 3]
        shape_format = [
            [np.float16, i, [32, 24, 18, 1000]] for i in format_list
        ]
        self.logsoftmax_result(shape_format)

    def test_logsoftmax_shape_format_fp32_4d(self):
        format_list = [0, 3]
        shape_format = [
            [np.float32, i, [32, 24, 18, 1000]] for i in format_list
        ]
        self.logsoftmax_result(shape_format)
            

if __name__ == "__main__":
    run_tests()
