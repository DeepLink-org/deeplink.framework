# Copyright (c) 2023, DeepLink.
from random import choice
from threading import Thread

import torch

import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestInitAMPDtypeMultiThread(TestCase):
    NUM_THREADS = 10
    TIMEOUT = 5
    DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.float64]

    def _run_multithread_test(self, f, args=(), kwargs={}):
        class PropagatingThread(Thread):
            """Helper class to propagate exception from child
            thread to main thread on join.

            Reference: https://stackoverflow.com/a/31614591/5602957
            Reference: https://github.com/pytorch/pytorch/blob/c263bd43e8e8502d4726643bc6fd046f0130ac0e/test/test_autograd.py#L10221-L10239
            """

            def run(self):
                self.exception = None
                try:
                    self.ret = super().run()
                except Exception as e:
                    self.exception = e

            def join(self, timeout=None):
                super().join(timeout)
                if self.exception:
                    raise self.exception from self.exception
                return self.ret

        threads = [
            PropagatingThread(target=f, args=args, kwargs=kwargs)
            for _ in range(self.NUM_THREADS)
        ]
        [t.start() for t in threads]
        [t.join(self.TIMEOUT) for t in threads]
        self.assertTrue(all(not t.is_alive() for t in threads))

    def test_get_in_multithread(self):
        def f():
            self.assertEqual(torch.get_autocast_gpu_dtype(), torch.float16)

        self._run_multithread_test(f)

    def test_set_in_multithread(self):
        def f():
            dtype = choice(self.DTYPES)
            torch.set_autocast_gpu_dtype(dtype)
            self.assertEqual(torch.get_autocast_gpu_dtype(), dtype)

        self._run_multithread_test(f)


if __name__ == "__main__":
    run_tests()
