# Copyright (c) 2023, DeepLink.
import torch
import torch_dipu
from torch_dipu import diputype
from torch_dipu.testing._internal.common_utils import TestCase, run_tests


class TestGenerator(TestCase):
    def test_python_api(self):
        torch.seed()
        torch.cuda.seed_all()
        torch.cuda.random.seed_all()
        torch.cuda.manual_seed_all(1)
        rngs = torch.cuda.get_rng_state_all()
        torch.cuda.set_rng_state_all(rngs)
        torch.manual_seed(1)
        self.assertEqual(torch.cuda.initial_seed(), 1)
        self.assertEqual(torch.initial_seed(), 1)
        for i in range(torch.cuda.device_count()):
            torch.cuda.manual_seed(i)

        state = torch.cuda.get_rng_state(0)
        new_state = torch.ones_like(state)
        torch.cuda.set_rng_state(new_state, 0)
        current_state = torch.cuda.get_rng_state(0)
        self.assertTrue(
            torch.allclose(
                current_state,
                torch.tensor(1, device=current_state.device, dtype=current_state.dtype),
            )
        )

    def test_torch_generator(self):
        gen = torch.Generator()
        self.assertEqual(gen.device.type, "cpu")
        gen.manual_seed(1)
        self.assertEqual(gen.initial_seed(), 1)

        gen = torch.Generator("cpu")
        self.assertEqual(gen.device.type, "cpu")

        gen = torch.Generator("cuda")
        self.assertEqual(gen.device.type, diputype)

        gen = torch.Generator("cuda:0")
        self.assertEqual(gen.device, torch.device(diputype + ":0"))

        gen = torch.Generator("dipu")
        self.assertEqual(gen.device.type, diputype)
        gen.manual_seed(1)
        self.assertEqual(gen.initial_seed(), 1)

    def test_randn_with_generator(self):
        gen = torch.Generator()
        gen.manual_seed(1)
        data1 = torch.randn(2, 3, generator=gen)
        gen.manual_seed(1)
        data2 = torch.randn(2, 3, generator=gen)
        self.assertEqual(data1, data2)
        data2 = torch.randn(2, 3, generator=gen)
        self.assertNotEqual(data1, data2)

        gen = torch.Generator("cuda")
        gen.manual_seed(1)
        data1 = torch.randn(2, 3, generator=gen, device="cuda")
        gen.manual_seed(1)
        data2 = torch.randn(2, 3, generator=gen, device="cuda")
        self.assertEqual(data1, data2)
        data2 = torch.randn(2, 3, generator=gen, device="cuda")
        self.assertNotEqual(data1, data2)

    def test_uniform_(self):
        t1 = torch.arange(0, 100, dtype=torch.float32).cuda()
        t2 = t1.clone()
        torch.manual_seed(1)
        t1.uniform_()
        torch.manual_seed(1)
        t2.uniform_()
        self.assertEqual(t1, t2)
        t2.uniform_()
        self.assertNotEqual(t1, t2)
        # print("uniform_ allclose success")

    def test_normal_(self):
        t1 = torch.arange(0, 100, dtype=torch.float32).cuda()
        t2 = t1.clone()
        torch.manual_seed(1)
        t1.normal_()
        torch.manual_seed(1)
        t2.normal_()
        self.assertEqual(t1, t2)
        t2.normal_()
        self.assertNotEqual(t1, t2)
        # print("normal_ allclose success")

    def test_random_(self):
        t1 = torch.arange(0, 100, dtype=torch.float32).cuda()
        t2 = t1.clone()
        torch.manual_seed(1)
        t1.random_(0, 100)
        torch.manual_seed(1)
        t2.random_(0, 100)
        self.assertEqual(t1, t2)
        t2.random_(0, 100)
        self.assertNotEqual(t1, t2)

        torch.manual_seed(1)
        t1.random_()
        torch.manual_seed(1)
        t2.random_()
        self.assertEqual(t1, t2)
        t2.random_()
        self.assertNotEqual(t1, t2)
        # print("random_ allclose success")

    def test_multinomial(self):
        data = torch.arange(0, 100, dtype=torch.float).cuda()
        torch.manual_seed(1)
        data1 = torch.multinomial(data, 10)
        torch.manual_seed(1)
        data2 = torch.multinomial(data, 10)
        self.assertEqual(data1, data2)
        data2 = torch.multinomial(data, 10)
        self.assertNotEqual(data1, data2)
        # print("multinomial allclose success")

    def test_randn(self):
        torch.manual_seed(1)
        t1 = torch.randn(100, device="cuda")
        torch.manual_seed(1)
        t2 = torch.randn(100, device="cuda")
        self.assertEqual(t1, t2)
        t2 = torch.randn(100, device="cuda")
        self.assertNotEqual(t1, t2)
        # print("randn allclose success")

    def test_bernoulli(self):
        x = 2 * torch.ones(100, device="cuda")
        torch.manual_seed(1)
        t1 = x.bernoulli_(0.5)
        self.assertTrue(torch.all((t1 == 0) | (t1 == 1)))
        x = torch.zeros(100, device="cuda")
        torch.manual_seed(1)
        t2 = x.bernoulli_(0.5)
        self.assertTrue(torch.all((t2 == 0) | (t2 == 1)))
        self.assertEqual(t1, t2)
        t2 = x.bernoulli_(0.5)
        self.assertTrue(torch.all((t2 == 0) | (t2 == 1)))
        self.assertNotEqual(t1, t2)
        # print("bernoulli allclose success")

    def test_randperm(self):
        if torch_dipu.dipu.vendor_type == "MLU":
            return

        torch.manual_seed(1)
        t1 = torch.randperm(100, device="cuda")
        torch.manual_seed(1)
        t2 = torch.randperm(100, device="cuda")
        self.assertEqual(t1, t2)
        t2 = torch.randperm(100, device="cuda")
        self.assertNotEqual(t1, t2)
        # print("randperm allclose success")

    def test_dropout(self):
        m = torch.nn.Dropout(p=0.2).cuda()
        input = torch.randn(20, 16).cuda()
        torch.manual_seed(1)
        t1 = m(input)
        torch.manual_seed(1)
        t2 = m(input)
        self.assertEqual(t1, t2)
        t2 = m(input)
        self.assertNotEqual(t1, t2)
        # print("dropout allclose success")

    def test_dropout_(self):
        m = torch.nn.Dropout(p=0.2, inplace=True).cuda()
        input = torch.randn(20, 16).cuda()
        torch.manual_seed(1)
        t1 = input.clone()
        m(t1)
        torch.manual_seed(1)
        t2 = input.clone()
        m(t2)
        self.assertEqual(t1, t2)
        t2 = input.clone()
        m(t2)
        self.assertNotEqual(t1, t2)
        # print("dropout_ allclose success")

    def test_default_generators(self):
        self.assertGreater(len(torch.cuda.default_generators), 0)
        torch.cuda.default_generators[0].manual_seed(1)
        self.assertEqual(torch.cuda.default_generators[0].initial_seed(), 1)


if __name__ == "__main__":
    run_tests()
