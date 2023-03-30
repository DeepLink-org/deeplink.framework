
from itertools import product
import torch
import numpy as np

cpu = "cpu"
dipu = torch.device("dipu")

def get_dipu_device():
    return torch.device(dipu)

def iter_indices(tensor):
    if tensor.dim() == 0:
        return range(0)
    if tensor.dim() == 1:
        return range(tensor.size(0))
    return product(*(range(s) for s in tensor.size()))


def create_common_tensor(item, minValue, maxValue, device=None):
    if device is None:
        device = get_dipu_device()
        
    dtype = item[0]
    shape = item[2]
    input1 = np.random.uniform(minValue, maxValue, shape).astype(dtype)
    cpu_input = torch.from_numpy(input1)
    dipu_input = torch.from_numpy(input1).to(device)
    return cpu_input, dipu_input

if __name__ == "__main__":
    format_list = [0]
    shape_list = [(3, 3), (3, 5), (5, 3)]
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
