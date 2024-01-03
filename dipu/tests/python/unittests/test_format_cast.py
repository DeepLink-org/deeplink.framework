# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
import numpy as np

from torch_dipu.testing._internal.common_utils import TestCase, run_tests, onlyOn

dipu = torch_dipu.dipu.diputype
device_dipu = torch.device(dipu)

class TestFormatCast(TestCase):
    def create_single_tensor(self, item, minvalue, maxvalue):
        dtype = item[0]
        format = item[1]
        shape = item[2]
        input = np.random.uniform(minvalue, maxvalue, shape).astype(dtype)
        dipu_input = torch.from_numpy(input).to(device_dipu)
        if format != torch_dipu.DIOPICustomFormat.Undefined:
            dipu_input = torch_dipu.format_cast(dipu_input, format)
        return dipu_input

    def check_and_get_format_tensor(self, a, memory_format):
        b = torch_dipu.format_cast(a, memory_format)
        self.assertEqual(b.diopi_format, memory_format)
        return b

    @onlyOn("NPU")
    def test_format_cast(self):
        shape_format1 = [np.float16, torch_dipu.DIOPICustomFormat.Undefined, (2, 2, 4, 4)]
        shape_format2 = [np.float16, torch_dipu.DIOPICustomFormat.Undefined, (2,2,2,2,4)]
        dipu_tensor1 = self.create_single_tensor(shape_format1, 1, 5)
        dipu_tensor2 = self.create_single_tensor(shape_format2, 1, 5)
       
        dipu_format_list1 = [
            torch_dipu.DIOPICustomFormat.ND,
            torch_dipu.DIOPICustomFormat.NC1HWC0,
            torch_dipu.DIOPICustomFormat.NCHW,
            torch_dipu.DIOPICustomFormat.FRACTAL_Z,
            torch_dipu.DIOPICustomFormat.NCHW,
            torch_dipu.DIOPICustomFormat.FRACTAL_NZ,
            torch_dipu.DIOPICustomFormat.NCHW,
            torch_dipu.DIOPICustomFormat.ND,
            torch_dipu.DIOPICustomFormat.NCHW,
            torch_dipu.DIOPICustomFormat.FRACTAL_Z,
            torch_dipu.DIOPICustomFormat.NCHW,     
            torch_dipu.DIOPICustomFormat.FRACTAL_NZ,
            torch_dipu.DIOPICustomFormat.NCHW,
            torch_dipu.DIOPICustomFormat.ND,
            torch_dipu.DIOPICustomFormat.FRACTAL_Z,
            torch_dipu.DIOPICustomFormat.NCHW,
            torch_dipu.DIOPICustomFormat.ND,
            torch_dipu.DIOPICustomFormat.FRACTAL_NZ,
        ]
        dipu_format_list2 = [
            torch_dipu.DIOPICustomFormat.NCDHW,
            torch_dipu.DIOPICustomFormat.FRACTAL_Z_3D,
            torch_dipu.DIOPICustomFormat.NCDHW,
            torch_dipu.DIOPICustomFormat.ND,
            torch_dipu.DIOPICustomFormat.FRACTAL_Z_3D,
            torch_dipu.DIOPICustomFormat.ND,
            torch_dipu.DIOPICustomFormat.NCDHW,
            torch_dipu.DIOPICustomFormat.NDC1HWC0,
            torch_dipu.DIOPICustomFormat.NCDHW,
            torch_dipu.DIOPICustomFormat.ND,
            torch_dipu.DIOPICustomFormat.NDC1HWC0,
            torch_dipu.DIOPICustomFormat.ND
        ]
        for dipu_format in dipu_format_list1:
            dipu_tensor1 = self.check_and_get_format_tensor(dipu_tensor1, dipu_format)
    
        for dipu_format in dipu_format_list2:
            dipu_tensor2 = self.check_and_get_format_tensor(dipu_tensor2, dipu_format)


if __name__ == "__main__":
    run_tests()
