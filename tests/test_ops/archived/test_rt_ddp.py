import os
import torch
import random
from torch import nn
import os
import sys
import tempfile
from torch._C import dtype
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def debugat(rank = 0):
    import os
    import ptvsd
    import socket
    # rank = int(os.environ['SLURM_PROCID'])
    # ntasks = int(os.environ['SLURM_NTASKS'])

    # rank = int(os.environ['RANK'])
    # ntasks =int(os.environ['WORLD_SIZE'])

    if rank == 0:
        pid1 = os.getpid()

        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        print(hostname, ip, flush=True)
        host = ip # or "localhost"
        # host = "127.0.0.1"
        port = 12346
        print("cwd is:",  os.getcwd(), flush=True)
        ptvsd.enable_attach(address=(host, port), redirect_output=False)
        print("-------------------------print rank,:", rank, "pid1:", pid1, flush=True)
        ptvsd.wait_for_attach()



def setup(rank, world_size, port, backend="nccl"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    print("comm using port:", str(port))
    # initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        # self.net1 = nn.Linear(10, 10)
        # self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        # o1 = self.net1(x)
        # return self.net2(self.relu(o1))
        return self.net2(x)

def demo_basic_ddp(rank, world_size, port):
    import torch_dipu
    print(f"Running basic DDP example on rank {rank} {torch.cuda.current_device()}")
    torch.cuda.set_device(rank)
    backend  ="nccl"
    dev1 = rank

    setup(rank, world_size, port, backend)

    for i in range(1, 4):
        # create model and move it to GPU with id rank
        model = ToyModel().to(dev1)
        # ddp_model = DDP(model, device_ids=[dev1])
        ddp_model = DDP(model)

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001, foreach=False)
        optimizer.zero_grad()

        in1 = torch.randn(20, 10).to(dev1)
        in1.requires_grad = True
        outputs = ddp_model(in1)
        outputs.backward(torch.ones_like(outputs), retain_graph = True)

        labels = torch.randn(20, 5).to(dev1)
        o1 = loss_fn(outputs, labels)
        o1.backward()
        optimizer.step()
        torch.cuda.synchronize()
        print("--------after bwd sync")
        assert model.net2.weight.grad is not None
    cleanup()

def demo_allreduce(rank, world_size, port):
    import torch_dipu
    print(f"Running basic DDP example on rank {rank} {torch.cuda.current_device()}")
    torch.cuda.set_device(rank)
    dev1 = rank

    setup(rank, world_size, port)

    world_size = dist.get_world_size()

    for op in [dist.reduce_op.SUM, dist.reduce_op.MAX, dist.reduce_op.MIN]:
        te_result = torch.zeros((3, 4)).to(dev1) + rank + 2
        dist.all_reduce(te_result, op=op)
    cleanup()

# need at least 2 card
def demo_p2p(rank, world_size, port):
  import torch_dipu
  setup(rank, world_size, port)

  sended_tensor = torch.arange(2).to(device=rank, dtype=torch.float16)
  received_tensor = torch.zeros(2).to(rank, dtype=torch.float16)
  for i in range(1, 3):
    if rank == 0:
      send_op = dist.P2POp(dist.isend, sended_tensor, 1)
      recv_op = dist.P2POp(dist.irecv, received_tensor, 1)
      reqs = dist.batch_isend_irecv([send_op, recv_op])
      for req in reqs:
        req.wait()
      print(received_tensor)

    if rank == 1:
      send_op = dist.P2POp(dist.isend, sended_tensor, 0)
      recv_op = dist.P2POp(dist.irecv, received_tensor, 0)

      reqs = dist.batch_isend_irecv([recv_op, send_op])

      # dicl not really support group p2p (underlying device also not support it?)
      # so, such test will block
      # reqs = dist.batch_isend_irecv([send_op, recv_op])

      for req in reqs:
        req.wait()
      print(received_tensor)
  cleanup()


def demo_allgather(rank, world_size, port):
  import torch_dipu
  setup(rank, world_size, port)

  src1 = torch.ones((2, 4)).to(rank)
  dests = torch.zeros((world_size * 2, 4)).to(rank)
  dests = [*dests.chunk(world_size, 0),]
  for i in range(1, 3):
    dist.all_gather(dests, src1)
  assert torch.allclose(src1, dests[0])
  print(dests[0])
  cleanup()

def demo_bcast(rank, world_size, port):
  import torch_dipu
  setup(rank, world_size, port)

  src1 = torch.ones((2, 4)).to(rank)
  dst = torch.empty((2, 4)).to(rank)
  # print(dst)
  for i in range(1, 3):
    if rank == 0:
      dist.broadcast(src1, 0)
    else:
      dist.broadcast(dst, 0)
  assert torch.allclose(src1, dst)
  print(dst)
  cleanup()

def demo_reduce(rank, world_size, port):
  import torch_dipu
  setup(rank, world_size, port)

  src_dst0 = torch.ones((2, 4)).to(rank)
  for i in range(1, 2):
    dist.reduce(src_dst0, 0, op=dist.reduce_op.SUM)
  if rank == 0:
    assert torch.allclose(torch.ones((2, 4)) * world_size, src_dst0.cpu())
  print(src_dst0)
  cleanup()


def demo_reducescatter(rank, world_size, port):
  import torch_dipu
  setup(rank, world_size, port)

  src1 = torch.ones((2 * world_size, 4)).to(rank)
  srcs = [*src1.chunk(world_size, 0),]

  dst = torch.zeros((2, 4)).to(rank)
  # print(dst)
  for i in range(1, 3):
    dist.reduce_scatter(dst, srcs, op=dist.reduce_op.SUM)
  
  assert torch.allclose(srcs[0], dst)
  print(dst)
  cleanup()

def demo_reducescatter_base(rank, world_size, port):
  import torch_dipu
  setup(rank, world_size, port)

  src1 = torch.ones((world_size * 2, 4)).to(rank)
  dst = torch.zeros((2, 4)).to(rank)
  for i in range(1, 3):
    dist.reduce_scatter_tensor(dst, src1, op=dist.reduce_op.SUM)
  assert torch.allclose(torch.ones((2, 4)), dst.cpu())
  print(dst)
  cleanup()

def demo_model_parallel(rank, world_size, port):
    print(f"Running DDP with model parallel example on rank {rank}.")
    backend  ="nccl"
    dev1 = rank

    # debugat(rank)
    setup(backend, rank, world_size)

    # setup mp_model and devices for this process
    dev0 = (rank * 2) % world_size
    dev1 = (rank * 2 + 1) % world_size
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

def run_demo(demo_fn, world_size, port):
    mp.spawn(demo_fn,
             args=(world_size, port,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    # world_size = 1
    # demo_allreduce(0, world_size)
    # demo_basic_ddp(0, world_size)
    port = random.randint(10000, 60000)

    world_size = 1
    run_demo(demo_basic_ddp, world_size, port)
    run_demo(demo_allreduce, world_size, port)
    run_demo(demo_allgather, world_size, port)
    run_demo(demo_reduce, world_size, port)
    run_demo(demo_reducescatter, world_size, port)
    run_demo(demo_reducescatter_base, world_size, port)


    # need 2 card to run
    # run_demo(demo_p2p, world_size, port)
    # run_demo(demo_bcast, world_size, port)

    # run_demo(demo_model_parallel, world_size)