import sys
sys.path.append("../..")

import torch
import numpy as np

from torch_dipu.testing._internal.testcase import TestCase, run_tests
from torch_dipu.testing._internal.common_utils import cpu, dipu, create_common_tensor


class TestAddmm(TestCase):

    def generate_scalar(self, dtype, min_x, max_x):
        if dtype == "float32" or "float16":
            scalar = np.random.uniform(min_x, max_x)
        if dtype == "int32":
            scalar = np.random.randint(min_x, max_x)
        return scalar

    def cpu_op_exec(self, input1, input2, input3, scalar1, scalar2):
        output = torch.addmm(
            input1,
            input2,
            input3,
            beta=scalar1,
            alpha=scalar2)
        output = output.numpy()
        return output

    def dipu_op_exec(self, input1, input2, input3, scalar1, scalar2):
        input1 = input1.to(dipu)
        input2 = input2.to(dipu)
        input3 = input3.to(dipu)
        output = torch.addmm(
            input1,
            input2,
            input3,
            beta=scalar1,
            alpha=scalar2)
        output = output.to(cpu)
        output = output.numpy()
        return output

    def dipu_op_exec_out(
            self,
            input1,
            input2,
            input3,
            scalar1,
            scalar2,
            input4):
        input1 = input1.to(dipu)
        input2 = input2.to(dipu)
        input3 = input3.to(dipu)
        output = input4.to(dipu)
        torch.addmm(
            input1,
            input2,
            input3,
            beta=scalar1,
            alpha=scalar2,
            out=output)
        output = output.to(cpu)
        output = output.numpy()
        return output

    def dipu_op_exec_inplace(self, input1, input2, input3, scalar1, scalar2):
        input1 = input1.to(dipu)
        input2 = input2.to(dipu)
        input3 = input3.to(dipu)
        input1.addmm_(input2, input3, beta=scalar1, alpha=scalar2)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_transpose_exec(self, input1, input2, input3, scalar1, scalar2):
        input3_t = input3.t()
        output = torch.addmm(
            input1,
            input2,
            input3_t,
            beta=scalar1,
            alpha=scalar2)
        output = output.numpy()
        return output

    def dipu_op_transpose_exec(self, input1, input2, input3, scalar1, scalar2):
        input1 = input1.to(dipu)
        input2 = input2.to(dipu)
        input3 = input3.to(dipu)
        input3_t = input3.t()
        output = torch.addmm(
            input1,
            input2,
            input3_t,
            beta=scalar1,
            alpha=scalar2)
        output = output.to(cpu)
        output = output.numpy()
        return output

    def test_addmm_shape_format_int(self):
        format_list = [0]
        shape_list = [(3, 3), (3, 5), (5, 3)]
        shape_format1 = [
            [np.int32, i, shape_list[0]] for i in format_list
        ]
        shape_format2 = [
            [np.int32, i, shape_list[1]] for i in format_list
        ]
        shape_format3 = [
            [np.int32, i, shape_list[2]] for i in format_list
        ]
        shape_format = [[i, j, k, "int32"]
                        for i in shape_format1 for j in shape_format2 for k in shape_format3]
        for item in shape_format:
            cpu_input1, dipu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, dipu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_input3, dipu_input3 = create_common_tensor(item[2], 0, 100)
            cpu_input4, dipu_input4 = create_common_tensor(item[0], 0, 100)

            scalar1 = self.generate_scalar(item[3], 0, 10)
            scalar2 = self.generate_scalar(item[3], 0, 10)

            cpu_output = self.cpu_op_exec(
                cpu_input1, cpu_input2, cpu_input3, scalar1, scalar2)
            dipu_output = self.dipu_op_exec(
                dipu_input1, dipu_input2, dipu_input3, scalar1, scalar2)

            dipu_output1 = self.dipu_op_exec_out(
                dipu_input1, dipu_input2, dipu_input3, scalar1, scalar2, dipu_input4)
            dipu_output2 = self.dipu_op_exec_inplace(
                dipu_input1, dipu_input2, dipu_input3, scalar1, scalar2)

            self.assertRtolEqual(cpu_output, dipu_output)
            self.assertRtolEqual(cpu_output, dipu_output1)
            self.assertRtolEqual(cpu_output, dipu_output2)

    def test_addmm_shape_format_fp32(self):
        format_list = [0]
        shape_list = [(3, 3), (3, 5), (5, 3)]
        shape_format1 = [
            [np.float16, i, shape_list[0]] for i in format_list
        ]
        shape_format2 = [
            [np.float16, i, shape_list[1]] for i in format_list
        ]
        shape_format3 = [
            [np.float16, i, shape_list[2]] for i in format_list
        ]
        shape_format = [[i, j, k, "float32"]
                        for i in shape_format1 for j in shape_format2 for k in shape_format3]
        for item in shape_format:
            cpu_input1, dipu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, dipu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_input3, dipu_input3 = create_common_tensor(item[2], 0, 100)
            cpu_input4, dipu_input4 = create_common_tensor(item[0], 0, 100)

            scalar1 = self.generate_scalar(item[3], 0, 10)
            scalar2 = self.generate_scalar(item[3], 0, 10)

            cpu_output = self.cpu_op_exec(
                cpu_input1, cpu_input2, cpu_input3, scalar1, scalar2)
            dipu_output = self.dipu_op_exec(
                dipu_input1, dipu_input2, dipu_input3, scalar1, scalar2)

            dipu_output1 = self.dipu_op_exec_out(
                dipu_input1, dipu_input2, dipu_input3, scalar1, scalar2, dipu_input4)
            dipu_output2 = self.dipu_op_exec_inplace(
                dipu_input1, dipu_input2, dipu_input3, scalar1, scalar2)

            self.assertRtolEqual(cpu_output, dipu_output)
            self.assertRtolEqual(cpu_output, dipu_output1)
            self.assertRtolEqual(cpu_output, dipu_output2)

    def test_addmm_shape_format_fp16(self, device=dipu):
        format_list = [0]
        shape_list = [(3, 3), (3, 5), (5, 3)]
        shape_format1 = [
            [np.float16, i, shape_list[0]] for i in format_list
        ]
        shape_format2 = [
            [np.float16, i, shape_list[1]] for i in format_list
        ]
        shape_format3 = [
            [np.float16, i, shape_list[2]] for i in format_list
        ]
        shape_format = [[i, j, k, "float16"]
                        for i in shape_format1 for j in shape_format2 for k in shape_format3]
        for item in shape_format:
            cpu_input1, dipu_input1 = create_common_tensor(item[0], 0, 2)
            cpu_input2, dipu_input2 = create_common_tensor(item[1], 0, 2)
            cpu_input3, dipu_input3 = create_common_tensor(item[2], 0, 2)
            cpu_input4, dipu_input4 = create_common_tensor(item[0], 0, 2)

            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)
            cpu_input3 = cpu_input3.to(torch.float32)

            scalar1 = self.generate_scalar(item[3], 0, 10)
            scalar2 = self.generate_scalar(item[3], 0, 10)

            cpu_output = self.cpu_op_exec(
                cpu_input1, cpu_input2, cpu_input3, scalar1, scalar2)
            dipu_output = self.dipu_op_exec(
                dipu_input1, dipu_input2, dipu_input3, scalar1, scalar2)
            cpu_output = cpu_output.astype(dipu_output.dtype)

            dipu_output1 = self.dipu_op_exec_out(
                dipu_input1, dipu_input2, dipu_input3, scalar1, scalar2, dipu_input4)
            dipu_output2 = self.dipu_op_exec_inplace(
                dipu_input1, dipu_input2, dipu_input3, scalar1, scalar2)

            self.assertRtolEqual(cpu_output, dipu_output)
            self.assertRtolEqual(cpu_output, dipu_output1)
            self.assertRtolEqual(cpu_output, dipu_output2)

    def test_addmm_transpose_shape_format_int(self):
        format_list = [0]
        shape_list = [(4, 5), (4, 7), (5, 7)]
        shape_format1 = [
            [np.int32, i, shape_list[0]] for i in format_list
        ]
        shape_format2 = [
            [np.int32, i, shape_list[1]] for i in format_list
        ]
        shape_format3 = [
            [np.int32, i, shape_list[2]] for i in format_list
        ]
        shape_format = [[i, j, k, "int32"]
                        for i in shape_format1 for j in shape_format2 for k in shape_format3]
        for item in shape_format:
            cpu_input1, dipu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, dipu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_input3, dipu_input3 = create_common_tensor(item[2], 0, 100)

            scalar1 = self.generate_scalar(item[3], 0, 10)
            scalar2 = self.generate_scalar(item[3], 0, 10)

            cpu_transpose_output = self.cpu_op_transpose_exec(
                cpu_input1, cpu_input2, cpu_input3, scalar1, scalar2)
            dipu_transpose_output = self.dipu_op_transpose_exec(
                dipu_input1, dipu_input2, dipu_input3, scalar1, scalar2)

            self.assertRtolEqual(cpu_transpose_output, dipu_transpose_output)

    def test_addmm_transpose_shape_format_fp32(self):
        format_list = [0]
        shape_list = [(4, 5), (4, 7), (5, 7)]
        shape_format1 = [
            [np.float32, i, shape_list[0]] for i in format_list
        ]
        shape_format2 = [
            [np.float32, i, shape_list[1]] for i in format_list
        ]
        shape_format3 = [
            [np.float32, i, shape_list[2]] for i in format_list
        ]
        shape_format = [[i, j, k, "float32"]
                        for i in shape_format1 for j in shape_format2 for k in shape_format3]
        for item in shape_format:
            cpu_input1, dipu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, dipu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_input3, dipu_input3 = create_common_tensor(item[2], 0, 100)

            scalar1 = self.generate_scalar(item[3], 0, 10)
            scalar2 = self.generate_scalar(item[3], 0, 10)

            cpu_transpose_output = self.cpu_op_transpose_exec(
                cpu_input1, cpu_input2, cpu_input3, scalar1, scalar2)
            dipu_transpose_output = self.dipu_op_transpose_exec(
                dipu_input1, dipu_input2, dipu_input3, scalar1, scalar2)

            self.assertRtolEqual(cpu_transpose_output, dipu_transpose_output)

    def test_addmm_transpose_shape_format_fp16(self):
        format_list = [0]
        shape_list = [(4, 5), (4, 7), (5, 7)]
        shape_format1 = [
            [np.float16, i, shape_list[0]] for i in format_list
        ]
        shape_format2 = [
            [np.float16, i, shape_list[1]] for i in format_list
        ]
        shape_format3 = [
            [np.float16, i, shape_list[2]] for i in format_list
        ]
        shape_format = [[i, j, k, "float16"]
                        for i in shape_format1 for j in shape_format2 for k in shape_format3]
        for item in shape_format:
            cpu_input1, dipu_input1 = create_common_tensor(item[0], 0, 2)
            cpu_input2, dipu_input2 = create_common_tensor(item[1], 0, 2)
            cpu_input3, dipu_input3 = create_common_tensor(item[2], 0, 2)

            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)
            cpu_input3 = cpu_input3.to(torch.float32)

            scalar1 = self.generate_scalar(item[3], 0, 10)
            scalar2 = self.generate_scalar(item[3], 0, 10)

            cpu_transpose_output = self.cpu_op_transpose_exec(
                cpu_input1, cpu_input2, cpu_input3, scalar1, scalar2)
            dipu_transpose_output = self.dipu_op_transpose_exec(
                dipu_input1, dipu_input2, dipu_input3, scalar1, scalar2)
            cpu_transpose_output = cpu_transpose_output.astype(
                dipu_transpose_output.dtype)

            self.assertRtolEqual(cpu_transpose_output, dipu_transpose_output)


if __name__ == "__main__":
    run_tests()
