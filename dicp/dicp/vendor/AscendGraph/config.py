import torch

from torch._decomp import get_decompositions


aten = torch.ops.aten
decomp_keys = []


def get_decomp():
    return get_decompositions(decomp_keys)


decomp = get_decomp()

enable_aot_operations = False
aot_operations_prefix = "dicp.vendor.AscendGraph.ascend_op."
aot_operations = {
    "DIOPIRMSNorm": None,
}
