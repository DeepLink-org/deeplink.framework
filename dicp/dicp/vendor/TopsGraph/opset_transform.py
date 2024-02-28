import os
import torch
import torch.fx

from typing import List, Optional, Tuple
from dicp.vendor.TopsGraph.conversion import AtenToTopsTransformer
from dicp.dynamo_bridge.compile_fx import is_torch_210
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch._subclasses import FakeTensor, FakeTensorMode

if is_torch_210:
    from dicp.dynamo_bridge.op_transformer import BackendPatternMatcherTransformer
    from dicp.vendor.TopsGraph.conversion import tops_patterns, tops_patterns_cls_list


class HandleInplaceCopyPass():
    def transform(self, gm: torch.fx.GraphModule):
        nodes = list(gm.graph.nodes)
        last_node = nodes[-1]
        assert last_node.op == "output"

        origin_outputs = list(last_node.args[0])
        inplace_outputs = []
        inplace_dict = {}
        for node in reversed(nodes):
            if node.op not in ["placeholder", "output"] and not isinstance(node.target, str):
                if hasattr(node.target, "name") and node.target.name() == "Copy_":
                    if node.args[0].op == "placeholder" and node.args[0].name not in inplace_dict.values():
                        inplace_outputs.append(node.args[1])
                        inplace_dict[node.args[1].name] = node.args[0].name

        assert len(last_node.kwargs) == 0
        last_node._Node__update_args_kwargs((origin_outputs + inplace_outputs, ), inplace_dict)

        return gm


def topsgraph_opset_transform(
    gm: torch.fx.GraphModule,
):

    # 1aten to Ntops
    gm = AtenToTopsTransformer(gm).transform()

    # Ntops to Mtops
    if is_torch_210:
        gm = BackendPatternMatcherTransformer(
            tops_patterns, tops_patterns_cls_list).transform(gm)

    # handle inplace copy operation: get inplace copy args to update outputs.
    gm = HandleInplaceCopyPass().transform(gm)

    return gm

def topsgraph_infer_shape(
    gm: torch.fx.GraphModule,
):
    def make_tensor_meta(x) -> Optional[TensorMetadata]:
        if isinstance(x, FakeTensor):
            return _extract_tensor_metadata(x)
        else:
            return None
    test_infer = bool(os.environ.get("TEST_DICP_INFER", False))
    for n in gm.graph.nodes:
        fake_value = None
        if n.op == 'call_function':
            fake_value = (n.target(*n.args, **n.kwargs))
        elif n.op == 'get_attr':
            target_atoms = n.target.split('.')
            attr_itr = gm
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(
                        f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
                attr_size, attr_dtye = attr_itr.shape, attr_itr.dtype
            with FakeTensorMode():
                fake_value = torch.empty(attr_size, dtype=attr_dtye)
        else:
            continue
        if 'val' in n.meta and test_infer:
            (n_meta_val, fake_val) = ((n.meta['val'],),(fake_value,)) if not isinstance(n.meta['val'],(Tuple,List)) else (n.meta['val'], fake_value)
            for i,(meta_i,fv_i) in enumerate(zip(n_meta_val, fake_val)):
                if not isinstance(fv_i, FakeTensor):
                    continue
                log_info = f"target: {n.target}, meta_i: {meta_i}, fv_i: {fv_i}"
                assert meta_i.size() == fv_i.size(), f"check infer size failed, {log_info}"
                assert meta_i.dtype == fv_i.dtype, f"check infer dtype failed, {log_info}"
                assert meta_i.stride() == fv_i.stride(), f"check infer stride failed, {log_info}"
                assert meta_i.storage_offset() == fv_i.storage_offset(), f"check infer storage offset failed, {log_info}"
        if 'val' not in n.meta:
            n.meta['val'] = fake_value
            n.meta["tensor_meta"] = make_tensor_meta(n.meta['val'])
    return gm
