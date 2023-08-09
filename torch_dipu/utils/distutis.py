import os

def _auto_mpi_env(addr=None):
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    # use env only when param 'rank' not set
    if 'RANK' not in os.environ:
        os.environ['RANK'] = str(rank)
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(world_size)

    if addr is not None:
        os.environ['MASTER_ADDR'] = str(addr)
    if 'MASTER_ADDR' not in os.environ:
        import re
        # eg: 1810366464.0;usock;tcp://10.148.7.30,172.17.0.1:34499
        omp_uri = os.environ["OMPI_MCA_orte_hnp_uri"]
        target_ip = re.search(r'.*tcp://((\d{1,3}\.){3}\d{1,3})[:,].*',
                              omp_uri)
        auto_addr = target_ip.group(1)
        os.environ['MASTER_ADDR'] = auto_addr


def _auto_slurm_env(addr=None):
    import subprocess
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    if 'RANK' not in os.environ:
        os.environ['RANK'] = str(proc_id)
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(ntasks)

    if addr is not None:
        os.environ['MASTER_ADDR'] = str(addr)
    if 'MASTER_ADDR' not in os.environ:
        node_list = os.environ['SLURM_NODELIST']
        auto_addr = subprocess.getoutput(
                    f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = auto_addr


def auto_env(addr=None, port=None):
    if 'SLURM_PROCID' in os.environ:
        _auto_slurm_env(addr)
    elif 'OMPI_COMM_WORLD_RANK' in os.environ:
        _auto_mpi_env(addr)
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    # specify master port
    if 'MASTER_PORT' not in os.environ:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
