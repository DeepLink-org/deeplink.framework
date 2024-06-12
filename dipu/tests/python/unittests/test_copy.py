# Copyright (c) 2023, DeepLink.
import itertools
import torch
import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestCopy(TestCase):
    @staticmethod
    def _create_tensor(cfg):
        src_shape = cfg[0]
        dst_shape = cfg[1]
        src_need_expand = cfg[2]
        src_device = cfg[3]
        dst_device = cfg[4]
        src_dtype = cfg[5]
        dst_dtype = cfg[6]

        src_cpu = torch.randn(src_shape, dtype=src_dtype, pin_memory=cfg[7])
        dst_cpu = torch.randn(dst_shape, dtype=dst_dtype, pin_memory=cfg[7])
        if src_device.type == "cpu":
            src_dipu = src_cpu
        else:
            src_dipu = src_cpu.to(src_device)
        if dst_device.type == "cpu":
            dst_dipu = dst_cpu
        else:
            dst_dipu = dst_cpu.to(dst_device)
        if src_need_expand:
            src_cpu = src_cpu.expand_as(dst_cpu)
            src_dipu = src_dipu.expand_as(dst_dipu)

        return src_cpu, dst_cpu, src_dipu, dst_dipu

    def test_copy_(self):
        src_shapes = [(3, 2), (4, 3, 2)]
        dst_shapes = [(4, 3, 2)]
        src_need_expands = [True, False]
        devices = [torch.device("cpu"), torch.device("cuda:0")]
        dtypes = [torch.float32, torch.float16]
        pin_memory_settings = [False, True]
        non_blocking_settings = [False, True]

        configs = []
        for cfg in itertools.product(
            src_shapes,
            dst_shapes,
            src_need_expands,
            devices,
            devices,
            dtypes,
            dtypes,
            pin_memory_settings,
            non_blocking_settings,
        ):
            if cfg[3].type != "cpu" or cfg[4].type != "cpu":
                configs.append(cfg)

        for cfg in configs:
            # print(f"cfg = {cfg}")
            src_cpu, dst_cpu, src_dipu, dst_dipu = self._create_tensor(cfg)
            dst_cpu.copy_(src_cpu, cfg[8])
            dst_dipu.copy_(src_dipu, cfg[8])
            # if torch.allclose(dst_cpu, dst_dipu.cpu()):
            #     print(f"cfg = {cfg} passed")
            # else:
            #     print(f"src_cpu = {src_cpu}")
            #     print(f"dst_cpu = {dst_cpu}")
            #     print(f"src_dipu = {src_dipu}")
            #     print(f"dst_dipu = {dst_dipu}")
            #     assert False, "copy_ test fail"
            self.assertEqual(dst_cpu, dst_dipu.cpu())

    def test_hollow_device_copy_(self):
        device = "cuda"
        t1 = torch.rand((6, 4), device=device)
        dst1 = t1.as_strided((2, 2), (4, 1))
        src = torch.rand((2, 2), device=device)
        dst1.copy_(src)
        self.assertEqual(dst1.cpu(), src.cpu())

    def test_d2d_peer_copy_(self):
        if torch.cuda.device_count() < 2:
            return
        dst = torch.rand((6, 4), device="cuda:0")
        src = torch.rand((6, 4), device="cuda:1")
        dst.copy_(src)
        self.assertEqual(dst.cpu(), src.cpu())
        self.assertEqual(dst.device.index, 0)
        self.assertEqual(src.device.index, 1)

        dst = torch.rand((6, 4), device="cuda:1")
        src = torch.rand((6, 4), device="cuda:0")
        dst.copy_(src)
        self.assertEqual(dst.cpu(), src.cpu())
        self.assertEqual(dst.device.index, 1)
        self.assertEqual(src.device.index, 0)

    def test_d2d_copy_(self):
        index = torch.cuda.device_count() - 1
        dst = torch.rand((6, 4), device="cuda:" + str(index))
        src = torch.rand((6, 4), device="cuda:" + str(index))
        dst.copy_(src)
        self.assertEqual(dst.cpu(), src.cpu())


if __name__ == "__main__":
    run_tests()
