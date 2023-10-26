# Copyright (c) 2023, DeepLink.
from random import choice
from threading import Thread

import torch

import torch_dipu
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestInitAMPDtypeMultiThread(TestCase):
    NUM_THREADS = 10
    TIMEOUT = 5
    DTYPES = [torch.int32, torch.int64, torch.float16, torch.float32]

    def _run_multithread_test(self, f, args=(), kwargs={}):
        threads = [Thread(target=f, args=args, kwargs=kwargs) for _ in range(self.NUM_THREADS)]
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
