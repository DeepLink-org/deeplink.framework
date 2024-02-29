import os
import torch
import torch_dipu
from typing import List, Optional, Tuple
from dicp.dynamo_bridge.compile_fx import is_torch_210
from dicp.vendor.AscendGraph.ascend_op import MatMul, CastToCpu, IdentityInp
from dicp.vendor.AscendGraph.conversion import AtenToAscendTransformer
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch._subclasses import FakeTensor, FakeTensorMode

if is_torch_210:
    from dicp.dynamo_bridge.op_transformer import BackendPatternMatcherTransformer
    from dicp.vendor.AscendGraph.pattern_replacement import (
        ascend_pattern_matcher,
        aten_patterns_cls_list,
        ascend_patterns_cls_list
    )


class ArgsTransDataPass:
    def transform(self, gm: torch.fx.graph_module):
        for n in gm.graph.nodes:
            if hasattr(n, 'op') and n.op == 'placeholder':
                fake_tensor = n.meta['val']
                memo = fake_tensor.fake_mode.fake_tensor_converter.tensor_memo
                for key in memo:
                    if id(memo[key].fake_device) == id(fake_tensor.fake_device):
                        memory_format = torch_dipu.get_native_memory_format(key())
                        n.meta['native_memory_format'] = str(memory_format.name)
                        break
        return gm


class OutputMarkPass:
    def __init__(self):
        self.assign_args = []
        self.cpu_tensor = []

    def transform(self, gm: torch.fx.graph_module):
        # dynamic shape feature
        input_names = []
        for n in gm.graph.nodes:
            if n.op == 'placeholder':
                input_names.append(n.name)

        for n in gm.graph.nodes:
            if n.op != 'call_function':
                continue
            if type(n.target) == CastToCpu:
                self.cpu_tensor.append(n.name)
            elif type(n.target) == IdentityInp:
                if len(n.args) == 2 and n.args[1] is not None and str(n.args[1]) in input_names:
                    self.assign_args.append((n.name, input_names.index(str(n.args[1]))))
                else:
                    raise RuntimeError("Op inner copy_ error!")

        for n in gm.graph.nodes:
            if n.op == 'call_function':
                prop = {}
                if n.name in self.cpu_tensor:
                    prop.update({'cpu_tensor' : n.name})
                if len(self.assign_args) > 0 and n.name in list(zip(*self.assign_args))[0]:
                    idx = list(zip(*self.assign_args))[0].index(n.name)
                    prop.update({'assign_args' : (self.assign_args[idx][0], self.assign_args[idx][1])})
                n.meta['prop'] = prop
        return gm


def symint_in_inputs(nodes):
    # dynamic shape feature
    for node in nodes:
        if node.op == 'placeholder':
            if hasattr(node, 'meta'):
                node = node.meta['val']
            if isinstance(node, torch.SymInt):
                return True
            if hasattr(node, 'shape'):
                for dim in node.shape:
                    if isinstance(dim, torch.SymInt):
                        return True
    return False

def ascendgraph_opset_convert(
    gm: torch.fx.GraphModule,
):
    if is_torch_210:
        gm = BackendPatternMatcherTransformer(
            ascend_pattern_matcher, aten_patterns_cls_list).transform(gm)
    gm = AtenToAscendTransformer(gm).transform()
    return gm

def ascendgraph_infer_shape(
    gm: torch.fx.GraphModule,
):
    def make_tensor_meta(x) -> Optional[TensorMetadata]:
        if isinstance(x, FakeTensor):
            return _extract_tensor_metadata(x)
        else:
            return None

    def _infer_shape(gm):
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

    gm = _infer_shape(gm)
    # For bug in pytorch
    # Avoid for dynamic shape
    if is_torch_210 and not symint_in_inputs(list(gm.graph.nodes)):
        gm = BackendPatternMatcherTransformer(
            ascend_pattern_matcher, ascend_patterns_cls_list).transform(gm)
    gm = OutputMarkPass().transform(gm)
    # uncomment this after DIOPI support pytorch2.1.1
    # gm = ArgsTransDataPass().transform(gm)
    return gm
