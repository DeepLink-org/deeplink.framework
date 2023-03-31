import torch
import numpy as np

from torch_dipu.testing._internal.common_utils import create_common_tensor, TestCase, run_tests


class TestLogSoftmaxBackward(TestCase):
    def cpu_op_exec(self, input1, input2, n, dtype):
        output = torch._log_softmax_backward_data(input1, input2, n, dtype)
        output = output.numpy()
        return output

    def dipu_op_exec_new(self, input1, input2, n, dtype):
        output = torch._log_softmax_backward_data(input1, input2, n, dtype)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def logsoftmax_backward_result(self, shape_format, min_lmt, max_lmt, dtype):
        for item in shape_format:
            dim = np.random.randint(0, len(item[2]))
            cpu_input1, dipu_input1 = create_common_tensor(item, min_lmt, max_lmt)
            cpu_input2, dipu_input2 = create_common_tensor(item, min_lmt, max_lmt)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, dim, torch.float32)
            dipu_output = self.dipu_op_exec_new(dipu_input1, dipu_input2, dim, dtype)
            cpu_output = cpu_output.astype(dipu_output.dtype)
            self.assertRtolEqual(cpu_output, dipu_output)

    # TODO:Fix me    
    # def test_logsoftmax_backward_shape_format_fp16_1d(self):
    #     format_list = [0, 3]
    #     shape_format = [
    #         [np.float16, i, [18]] for i in format_list 
    #     ]
    #     self.logsoftmax_backward_result(shape_format, 0, 2, torch.float16)
        
    def test_logsoftmax_backward_shape_format_fp32_1d(self):
        format_list = [0, 3]
        shape_format = [
            [np.float32, i, [18]] for i in format_list 
        ]
        self.logsoftmax_backward_result(shape_format, 0, 50, torch.float32)
        
    # TODO:Fix me   
    # def test_logsoftmax_backward_shape_format_fp16_2d(self):
    #     format_list = [0, 3, 29]
    #     shape_format = [
    #         [np.float16, i, [256, 1000]] for i in format_list 
    #     ]
    #     self.logsoftmax_backward_result(shape_format, 0, 2, torch.float16)
        
    def test_logsoftmax_backward_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        shape_format = [
            [np.float32, i, [256, 1000]] for i in format_list 
        ]
        self.logsoftmax_backward_result(shape_format, 0, 50, torch.float32)
        
    # TODO:Fix me   
    # def test_logsoftmax_backward_shape_format_fp16_3d(self):
    #     format_list = [0, 3, 29]
    #     shape_format = [
    #         [np.float16, i, [32, 48, 64]] for i in format_list 
    #     ]
    #     self.logsoftmax_backward_result(shape_format, 0, 2, torch.float16)
        
    def test_logsoftmax_backward_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        shape_format = [
            [np.float32, i, [32, 48, 64]] for i in format_list 
        ]
        self.logsoftmax_backward_result(shape_format, 0, 50, torch.float32)
        
    # TODO:Fix me   
    # def test_logsoftmax_backward_shape_format_fp16_4d(self):
    #     format_list = [0, 3]
    #     shape_format = [
    #         [np.float16, i, [32, 24, 18, 18]] for i in format_list 
    #     ]
    #     self.logsoftmax_backward_result(shape_format, 0, 2, torch.float16)
        
    def test_logsoftmax_backward_shape_format_fp32_4d(self):
        format_list = [0, 3]
        shape_format = [
            [np.float32, i, [32, 24, 18, 18]] for i in format_list 
        ]
        self.logsoftmax_backward_result(shape_format, 0, 50, torch.float32)
            

if __name__ == "__main__":
    run_tests()
