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
        if format != torch_dipu.DIOPIMemoryFormat.Undefined:
            dipu_input = torch_dipu.format_cast(input, format)
        return dipu_input
    

    def check_and_get_format_tensor(self, a, memory_format):
        b = torch_dipu.format_cast(a, memory_format)
        self.assertEqual(b.diopi_format, memory_format)
        c = torch_dipu.format_cast(b, a.diopi_format)
        self.assertEqual(c.diopi_format, a.diopi_format)
        self.assertEqual(a, c)
        return b


    @onlyOn("NPU")
    def test_format_cast(self):
        shape_format = [np.float16, torch_dipu.DIOPIMemoryFormat.Undefined, (2, 2, 4, 4)]
        dipu_tensor = self.create_single_tensor(shape_format, 1, 5)
        dipu_format_list = [torch_dipu.DIOPIMemoryFormat.NCHW,
                            torch_dipu.DIOPIMemoryFormat.ND,
                            torch_dipu.DIOPIMemoryFormat.NCHW,
                            torch_dipu.DIOPIMemoryFormat.FRACTAL_Z,
                            torch_dipu.DIOPIMemoryFormat.NCHW,
                            torch_dipu.DIOPIMemoryFormat.FRACTAL_NZ,
                            torch_dipu.DIOPIMemoryFormat.NCHW,
                            torch_dipu.DIOPIMemoryFormat.ND,
                            torch_dipu.DIOPIMemoryFormat.FRACTAL_Z,
                            torch_dipu.DIOPIMemoryFormat.NCHW,     
                            torch_dipu.DIOPIMemoryFormat.ND,
                            torch_dipu.DIOPIMemoryFormat.FRACTAL_NZ,
                            torch_dipu.DIOPIMemoryFormat.NCHW,
                            torch_dipu.DIOPIMemoryFormat.NC1HWC0]
        for dipu_format in dipu_format_list:
            dipu_tensor = self.check_and_get_format_tensor(dipu_tensor, dipu_format)

        dipu_tensor = dipu_tensor.view(2,2,2,2,4).clone()
        dipu_format_list = [torch_dipu.DIOPIMemoryFormat.NCDHW,
                            torch_dipu.DIOPIMemoryFormat.FRACTAL_Z_3D,
                            torch_dipu.DIOPIMemoryFormat.NCDHW,
                            torch_dipu.DIOPIMemoryFormat.ND,
                            torch_dipu.DIOPIMemoryFormat.FRACTAL_Z_3D,
                            torch_dipu.DIOPIMemoryFormat.ND,
                            torch_dipu.DIOPIMemoryFormat.NCDHW,
                            torch_dipu.DIOPIMemoryFormat.NDC1HWC0,
                            torch_dipu.DIOPIMemoryFormat.NCDHW,
                            torch_dipu.DIOPIMemoryFormat.ND,
                            torch_dipu.DIOPIMemoryFormat.NDC1HWC0,
                            torch_dipu.DIOPIMemoryFormat.ND]
        for dipu_format in dipu_format_list:
            dipu_tensor = self.check_and_get_format_tensor(dipu_tensor, dipu_format)


if __name__ == "__main__":
    run_tests()