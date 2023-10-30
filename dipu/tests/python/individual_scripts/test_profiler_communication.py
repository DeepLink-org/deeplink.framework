import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity

def setup(rank, world_size, port, backend="nccl"):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net = nn.Linear(10, 5)

    def forward(self, x):
        return self.net(x)


def demo_basic_ddp(rank, world_size, port):
    import torch_dipu
    torch.cuda.set_device(rank)
    setup(rank, world_size, port)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.MSELoss()

    input = torch.randn(20, 10).to(rank)
    input.requires_grad = True
    labels = torch.randn(20, 5).to(rank)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_modules=True,
        with_stack=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:
        output = ddp_model(input)
        output.backward(torch.ones_like(output), retain_graph=True)
        loss = loss_fn(output, labels)
        loss.backward()

    profile_output = prof.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total", row_limit=1000
    )
    assert("c10d::allreduce_" in profile_output)
    assert("LaunchKernel_DiclAllreduce" in profile_output)
    prof.export_chrome_trace(f"./dipu_resnet18_profiler_{rank}.json")
    cleanup()

def test_profiler_communication():
    port = random.randint(10000, 60000)
    world_size = 1
    mp.spawn(demo_basic_ddp, args=(world_size, port), nprocs=world_size, join=True)


if __name__ == "__main__":
    test_profiler_communication()
